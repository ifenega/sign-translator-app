"""Baseline model: summarise each clip by the mean and standard deviation of every
landmark over time, and classify that summary with a small MLP.

It ignores the order of the frames, so it can't tell a movement from its reverse.
It gives a first number for the sequence models to beat.

Usage, from ml/:
    .venv/bin/python scripts/train_baseline.py
"""

import torch
from torch import nn

from training import NUM_LANDMARKS, NUM_SIGNS, train


class MeanStdMLP(nn.Module):
    def __init__(self, hidden=512, dropout=0.3):
        super().__init__()
        num_features = NUM_LANDMARKS * 2 * 2  # mean and std of x and y for each landmark
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_SIGNS),
        )

    def forward(self, x, mask):
        # Average only over the frames where each landmark was seen.
        seen = (~mask).unsqueeze(-1).float()  # (clips, frames, landmarks, 1)
        count = seen.sum(dim=1).clamp(min=1)
        mean = (x * seen).sum(dim=1) / count
        std = ((((x - mean.unsqueeze(1)) * seen) ** 2).sum(dim=1) / count).sqrt()
        return self.mlp(torch.cat([mean.flatten(1), std.flatten(1)], dim=1))


if __name__ == "__main__":
    train(MeanStdMLP(), name="baseline", epochs=30, learning_rate=1e-3, use_augmentation=False)
