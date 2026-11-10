"""Sequence model: a GRU reads the clip frame by frame, so unlike the baseline it sees
the order of the movement. Trained with augmentation (see training.py).

Usage, from ml/:
    .venv/bin/python scripts/train_gru.py
"""

import torch
from torch import nn

from training import NUM_LANDMARKS, NUM_SIGNS, train


class GRUClassifier(nn.Module):
    def __init__(self, hidden=256, dropout=0.3):
        super().__init__()
        frame_size = NUM_LANDMARKS * 3  # x, y and a "missing" flag for each landmark
        # Turn each frame's raw points into a feature vector.
        self.embed = nn.Sequential(
            nn.Linear(frame_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Read the frames in order, forwards and backwards.
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.classify = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, NUM_SIGNS))

    def forward(self, x, mask):
        frames = torch.cat([x.flatten(2), mask.float()], dim=2)  # (clips, frames, 85 * 3)
        outputs, _ = self.gru(self.embed(frames))

        # Average the GRU's output over the frames that aren't padding.
        real_frame = (~mask.all(dim=2)).unsqueeze(-1).float()  # (clips, frames, 1)
        pooled = (outputs * real_frame).sum(dim=1) / real_frame.sum(dim=1).clamp(min=1)
        return self.classify(pooled)


if __name__ == "__main__":
    train(GRUClassifier(), name="gru", epochs=40, learning_rate=1e-3, use_augmentation=True)
