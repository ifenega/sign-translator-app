"""Shared pieces for the training scripts: loading data, augmentation, accuracy and the training loop.

Every model takes the same input: a batch of clips as points x of shape
(clips, 64, 85, 2) and a mask of shape (clips, 64, 85) that is True where a point
was missing (see preprocess.py).

Models are compared on the val split. The test split is kept untouched until the
final model is chosen, so its score stays an honest estimate.
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PREPARED_DIR = DATA_DIR / "prepared"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

NUM_SIGNS = 250
NUM_FRAMES = 64
NUM_LANDMARKS = 85

# Augmentation: how far each training clip may be randomly changed.
MAX_ROTATION = math.radians(15)
MAX_SCALE_CHANGE = 0.2  # scaled by 0.8 to 1.2
MAX_SHIFT = 0.1  # moved by up to a tenth of a shoulder width
MAX_SPEED_CHANGE = 0.2  # played 0.8 to 1.2 times as fast
DROP_FRAME_CHANCE = 0.1
# Changes that imitate different people rather than a different camera:
MAX_HAND_ROTATION = math.radians(10)  # each hand turned around its own wrist
MAX_HAND_SCALE_CHANGE = 0.15  # each hand 0.85 to 1.15 times as big, separately from the body
HIDE_LIPS_CHANCE = 0.2  # sometimes hide the lips, so the model can't lean on them too much

# Landmark positions (see preprocess.py).
LIPS = slice(0, 20)
HANDS = [slice(20, 41), slice(53, 74)]  # other hand, dominant hand; the first point is the wrist


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_split(split, device):
    """Load one split (train, val or test) as tensors on the device.

    The points are kept as float16, which halves the memory they take (the train split
    would otherwise fill about 3 GB of a 16 GB Mac). batches() turns each batch back into
    float32. The precision lost is far below MediaPipe's own jitter.
    """
    x = torch.from_numpy(np.load(PREPARED_DIR / f"{split}_x.npy")).half().to(device)
    mask = torch.from_numpy(np.load(PREPARED_DIR / f"{split}_mask.npy")).to(device)
    y = torch.from_numpy(np.load(PREPARED_DIR / f"{split}_y.npy")).long().to(device)
    return x, mask, y


def batches(x, mask, y, batch_size, shuffle=False):
    """Yield (x, mask, y) in batches, in random order if shuffle is True."""
    if shuffle:
        order = torch.randperm(len(y), device=y.device)
    else:
        order = torch.arange(len(y), device=y.device)
    for start in range(0, len(y), batch_size):
        chosen = order[start:start + batch_size]
        yield x[chosen].float(), mask[chosen], y[chosen]


def change_speed(x, mask):
    """Play each clip a little faster or slower, by copying its frames at a different rate.

    A clip is never slowed down so much that its end no longer fits in the 64 frames.
    """
    num_clips = len(x)
    frame_number = torch.arange(NUM_FRAMES, device=x.device)
    real_frame = ~mask.all(dim=2)  # (clips, frames); padding frames are all missing
    length = (real_frame * (frame_number + 1)).amax(dim=1)  # up to and including the last real frame
    speed = 1 + (torch.rand(num_clips, device=x.device) * 2 - 1) * MAX_SPEED_CHANGE
    speed = torch.maximum(speed, length / NUM_FRAMES)

    # Frame t of the new clip is a copy of frame t * speed of the original.
    source = (frame_number * speed.unsqueeze(1)).long()  # (clips, frames)
    past_end = source >= NUM_FRAMES
    source = source.clamp(max=NUM_FRAMES - 1)
    clip_number = torch.arange(num_clips, device=x.device).unsqueeze(1)
    return x[clip_number, source], mask[clip_number, source] | past_end.unsqueeze(-1)


def random_amount(num_clips, max_change, device):
    """A random number between -max_change and +max_change for each clip, shaped (clips, 1, 1)."""
    return (torch.rand(num_clips, 1, 1, device=device) * 2 - 1) * max_change


def rotate_and_scale(points, angle, scale):
    """Rotate points (..., 2) around (0, 0) by angle, then multiply them by scale."""
    px, py = points[..., 0], points[..., 1]
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.stack([px * cos - py * sin, px * sin + py * cos], dim=-1) * scale.unsqueeze(-1)


def augment(x, mask):
    """Randomly change each clip a little, by a different amount for each clip: its speed,
    angle, size and position, the angle and size of each hand, and which points are hidden."""
    x, mask = change_speed(x, mask)
    num_clips, device = len(x), x.device

    # The whole body: rotate and scale around the centre between the shoulders, which
    # preprocessing put at (0, 0), then shift.
    angle = random_amount(num_clips, MAX_ROTATION, device)
    scale = 1 + random_amount(num_clips, MAX_SCALE_CHANGE, device)
    shift = (torch.rand(num_clips, 1, 1, 2, device=device) * 2 - 1) * MAX_SHIFT  # x and y separately
    x = rotate_and_scale(x, angle, scale) + shift

    # Each hand on its own: rotate and scale it around its wrist.
    for hand in HANDS:
        wrist = x[:, :, hand.start:hand.start + 1]
        angle = random_amount(num_clips, MAX_HAND_ROTATION, device)
        scale = 1 + random_amount(num_clips, MAX_HAND_SCALE_CHANGE, device)
        x[:, :, hand] = wrist + rotate_and_scale(x[:, :, hand] - wrist, angle, scale)

    hide_lips = torch.rand(num_clips, 1, 1, device=device) < HIDE_LIPS_CHANCE
    mask = mask.clone()
    mask[:, :, LIPS] |= hide_lips
    dropped = torch.rand(num_clips, NUM_FRAMES, 1, device=device) < DROP_FRAME_CHANCE
    mask = mask | dropped
    x = x * ~mask.unsqueeze(-1)  # missing points stay at 0
    return x, mask


@torch.no_grad()
def evaluate(model, x, mask, y, batch_size=1024):
    """Return top-1 and top-5 accuracy: how often the right sign is the model's first guess, or in its top 5."""
    model.eval()
    top1 = top5 = 0
    for x_batch, mask_batch, y_batch in batches(x, mask, y, batch_size):
        guesses = model(x_batch, mask_batch).topk(5, dim=1).indices
        top1 += (guesses[:, 0] == y_batch).sum().item()
        top5 += (guesses == y_batch.unsqueeze(1)).any(dim=1).sum().item()
    return top1 / len(y), top5 / len(y)


def train(model, name, epochs, learning_rate, use_augmentation, label_smoothing=0.0, seed=0, final=False, batch_size=256):
    """Train on the train split, keep the epoch with the best val top-1, and save it to data/checkpoints/.

    label_smoothing: the share of the target spread over the wrong signs instead of
    all of it going to the right one (0.1 = "90% this sign"). It stops the model from
    becoming overconfident on the training clips.
    seed: fixes the random order of clips and the random augmentation, so a run can be
    repeated exactly. Different seeds give models that make slightly different mistakes.
    final: for the final model, once the settings have been chosen on val. It trains on
    train and val together (more people to learn from) and keeps the last epoch, since
    there is no val left to pick the best one with. Score it on test.
    """
    torch.manual_seed(seed)
    device = get_device()
    model = model.to(device)
    train_data = load_split("train", device)
    val_data = load_split("val", device)
    if final:
        train_data = tuple(torch.cat(pair) for pair in zip(train_data, val_data))
        del val_data
    num_parameters = sum(p.numel() for p in model.parameters())
    people = "train + val" if final else "train"
    print(f"{name}: {num_parameters:,} parameters, {len(train_data[2]):,} clips ({people}), training on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps_per_epoch = math.ceil(len(train_data[2]) / batch_size)
    # Lower the learning rate smoothly to 0 over the run.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / f"{name}.pt"
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        total_loss = 0.0
        for x, mask, y in batches(*train_data, batch_size, shuffle=True):
            if use_augmentation:
                x, mask = augment(x, mask)
            loss = F.cross_entropy(model(x, mask), y, label_smoothing=label_smoothing)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(train_data[2])
        if final:
            history.append({"epoch": epoch, "train_loss": train_loss})
            torch.save(model.state_dict(), checkpoint)
            print(f"epoch {epoch:2d}  loss {train_loss:.3f}  ({time.time() - started:.0f}s)  saved")
            pd.DataFrame(history).to_csv(CHECKPOINT_DIR / f"{name}_history.csv", index=False)
            continue

        val_top1, val_top5 = evaluate(model, *val_data)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_top1": val_top1, "val_top5": val_top5})
        is_best = val_top1 == max(row["val_top1"] for row in history)
        if is_best:
            torch.save(model.state_dict(), checkpoint)
        print(
            f"epoch {epoch:2d}  loss {train_loss:.3f}  val top-1 {val_top1:.1%}  top-5 {val_top5:.1%}"
            f"  ({time.time() - started:.0f}s){'  saved' if is_best else ''}"
        )
        # Written every epoch, so a long run that is stopped early keeps its history.
        pd.DataFrame(history).to_csv(CHECKPOINT_DIR / f"{name}_history.csv", index=False)

    if final:
        print(f"Saved the last epoch -> {checkpoint}")
        return
    best = max(history, key=lambda row: row["val_top1"])
    print(f"Best: epoch {best['epoch']}, val top-1 {best['val_top1']:.1%}, top-5 {best['val_top5']:.1%} -> {checkpoint}")
