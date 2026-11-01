"""Shared pieces for the training scripts: loading data, augmentation, accuracy and the training loop.

Every model takes the same input: a batch of clips as points x of shape
(clips, 64, 74, 2) and a mask of shape (clips, 64, 74) that is True where a point
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
NUM_LANDMARKS = 74

# Augmentation: how far each training clip may be randomly changed.
MAX_ROTATION = math.radians(15)
MAX_SCALE_CHANGE = 0.2  # scaled by 0.8 to 1.2
DROP_FRAME_CHANCE = 0.1


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_split(split, device):
    """Load one split (train, val or test) as tensors on the device."""
    x = torch.from_numpy(np.load(PREPARED_DIR / f"{split}_x.npy")).to(device)
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
        yield x[chosen], mask[chosen], y[chosen]


def augment(x, mask):
    """Randomly rotate, scale and drop frames, by a different amount for each clip."""
    num_clips = len(x)
    angle = (torch.rand(num_clips, 1, 1, device=x.device) * 2 - 1) * MAX_ROTATION
    scale = 1 + (torch.rand(num_clips, 1, 1, device=x.device) * 2 - 1) * MAX_SCALE_CHANGE

    # Rotate around the centre between the shoulders, which preprocessing put at (0, 0).
    px, py = x[..., 0], x[..., 1]
    cos, sin = torch.cos(angle), torch.sin(angle)
    x = torch.stack([px * cos - py * sin, px * sin + py * cos], dim=-1) * scale.unsqueeze(-1)

    dropped = torch.rand(num_clips, NUM_FRAMES, 1, device=x.device) < DROP_FRAME_CHANCE
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


def train(model, name, epochs, learning_rate, use_augmentation, batch_size=256):
    """Train on the train split, keep the epoch with the best val top-1, and save it to data/checkpoints/."""
    torch.manual_seed(0)
    device = get_device()
    model = model.to(device)
    train_data = load_split("train", device)
    val_data = load_split("val", device)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"{name}: {num_parameters:,} parameters, training on {device}")

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
            loss = F.cross_entropy(model(x, mask), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(train_data[2])
        val_top1, val_top5 = evaluate(model, *val_data)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_top1": val_top1, "val_top5": val_top5})
        is_best = val_top1 == max(row["val_top1"] for row in history)
        if is_best:
            torch.save(model.state_dict(), checkpoint)
        print(
            f"epoch {epoch:2d}  loss {train_loss:.3f}  val top-1 {val_top1:.1%}  top-5 {val_top5:.1%}"
            f"  ({time.time() - started:.0f}s){'  saved' if is_best else ''}"
        )

    pd.DataFrame(history).to_csv(CHECKPOINT_DIR / f"{name}_history.csv", index=False)
    best = max(history, key=lambda row: row["val_top1"])
    print(f"Best: epoch {best['epoch']}, val top-1 {best['val_top1']:.1%}, top-5 {best['val_top5']:.1%} -> {checkpoint}")
