"""Convolution + Transformer model: 1D convolutions over time pick up short movements
(a few frames at a time), then a Transformer lets every frame look at every other
frame, so the start of a sign can be linked directly to its end. This is the kind of
model the top competition solutions used. Trained with augmentation (see training.py).

Besides where each point is, the model also gets how far it moved since the previous
frame (its motion), so it doesn't have to work out the movement itself.

Usage, from ml/:
    .venv/bin/python scripts/train_transformer.py --name transformer_face --epochs 40 --label-smoothing 0.1
    .venv/bin/python scripts/train_transformer.py --name transformer_big --epochs 40 --label-smoothing 0.1 --dim 256 --layers 5

The final models, once the settings are chosen on val (one per seed, then combined):
    .venv/bin/python scripts/train_transformer.py --name final_seed0 --epochs 100 --label-smoothing 0.1 --seed 0 --final

--dim and --layers set the model's size. A checkpoint only loads into a model of the
same size, so pass the same values to evaluate.py.
"""

import argparse

import torch
import torch.nn.functional as F
from torch import nn

from training import NUM_FRAMES, NUM_LANDMARKS, NUM_SIGNS, train


def motion(x, mask):
    """How far each point moved since the previous frame: 0 if it is missing in either frame."""
    moved = x[:, 1:] - x[:, :-1]
    seen_in_both = ~(mask[:, 1:] | mask[:, :-1])
    moved = moved * seen_in_both.unsqueeze(-1)
    return F.pad(moved, (0, 0, 0, 0, 1, 0))  # the first frame has no previous frame: 0


class ConvBlock(nn.Module):
    """Looks at a few neighbouring frames at a time, and adds what it finds to its input."""

    def __init__(self, dim, kernel_size=5, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Conv1d wants (clips, features, frames); the rest of the model uses (clips, frames, features).
        found = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x + self.dropout(F.gelu(self.norm(found)))


class ConvTransformer(nn.Module):
    def __init__(self, dim=192, heads=4, layers=3, dropout=0.3):
        super().__init__()
        frame_size = NUM_LANDMARKS * 5  # x, y, a "missing" flag and the x, y motion of each landmark
        # Turn each frame's raw points into a feature vector.
        self.embed = nn.Sequential(
            nn.Linear(frame_size, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.convs = nn.Sequential(ConvBlock(dim, dropout=dropout), ConvBlock(dim, dropout=dropout))
        # Attention alone doesn't know frame order, so each frame position gets a learned vector added.
        self.position = nn.Parameter(torch.zeros(1, NUM_FRAMES, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=2 * dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.classify = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, NUM_SIGNS))

    def forward(self, x, mask):
        frames = torch.cat([x.flatten(2), mask.float(), motion(x, mask).flatten(2)], dim=2)  # (clips, frames, 85 * 5)

        # Frames where every point is missing are padding or dropped frames.
        empty_frame = mask.all(dim=2)  # (clips, frames)
        real_frame = (~empty_frame).unsqueeze(-1).float()
        # Always let attention see the first frame: if augmentation drops every frame of a
        # very short clip, attending to nothing would give NaN.
        empty_frame[:, 0] = False

        features = self.convs(self.embed(frames) * real_frame) + self.position
        features = self.transformer(features, src_key_padding_mask=empty_frame)

        # Average over the real frames.
        pooled = (features * real_frame).sum(dim=1) / real_frame.sum(dim=1).clamp(min=1)
        return self.classify(pooled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the convolution + Transformer model.")
    parser.add_argument("--name", default="transformer", help="checkpoint name in data/checkpoints/")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="e.g. 0.1; see train() in training.py")
    parser.add_argument("--dim", type=int, default=192, help="size of each frame's feature vector")
    parser.add_argument("--layers", type=int, default=3, help="number of Transformer layers")
    parser.add_argument("--seed", type=int, default=0, help="change it to train another copy that makes different mistakes")
    parser.add_argument("--final", action="store_true", help="train on train + val and keep the last epoch; see train()")
    args = parser.parse_args()
    torch.manual_seed(args.seed)  # the model's starting weights are random too
    train(
        ConvTransformer(dim=args.dim, layers=args.layers), name=args.name, epochs=args.epochs, learning_rate=1e-3,
        use_augmentation=True, label_smoothing=args.label_smoothing, seed=args.seed, final=args.final,
    )
