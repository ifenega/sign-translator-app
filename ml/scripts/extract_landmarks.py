"""Extract the landmarks we train on from the Kaggle asl-signs zip.

The zip is never unzipped (it would not fit on disk): each clip is read straight
out of it. From the 543 landmarks in every frame we keep 94:

    0-39   lips
    40-60  left hand
    61-72  upper body (shoulders, elbows, wrists, hands)
    73-93  right hand

and save everything to two files in data/processed/:

    landmarks.npy  every frame of every clip, shape (total_frames, 94, 3) for x, y, z.
                   A hand that wasn't detected in a frame is NaN.
    clips.csv      one row per clip: its sign, label, participant, and which rows
                   of landmarks.npy belong to it (start, length).

Usage, from ml/:
    .venv/bin/python scripts/extract_landmarks.py
    .venv/bin/python scripts/extract_landmarks.py --limit 500   # quick test run
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_PATH = DATA_DIR / "asl-signs" / "asl-signs.zip"
OUT_DIR = DATA_DIR / "processed"

# Every frame has 543 landmarks, always in this order.
LANDMARKS_PER_FRAME = 543
FACE_START = 0  # 468 face points
LEFT_HAND_START = 468  # 21 hand points
POSE_START = 489  # 33 body points
RIGHT_HAND_START = 522  # 21 hand points

# Outline of the outer and inner lips: 40 of the 468 face points.
LIPS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
HAND = range(21)
UPPER_BODY = range(11, 23)

KEEP = (
    [FACE_START + i for i in LIPS]
    + [LEFT_HAND_START + i for i in HAND]
    + [POSE_START + i for i in UPPER_BODY]
    + [RIGHT_HAND_START + i for i in HAND]
)


def read_clip(archive, path):
    """Read one clip from the zip as an array of shape (frames, 94, 3)."""
    data = archive.read(path)
    df = pd.read_parquet(io.BytesIO(data), columns=["x", "y", "z"])
    xyz = df.to_numpy(dtype=np.float32).reshape(-1, LANDMARKS_PER_FRAME, 3)
    # float16 halves the file size; the precision lost is far below MediaPipe's own jitter.
    return xyz[:, KEEP].astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description="Extract training landmarks from the asl-signs zip.")
    parser.add_argument("--zip", type=Path, default=ZIP_PATH, help="path to asl-signs.zip")
    parser.add_argument("--limit", type=int, help="only process the first N clips")
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip) as archive:
        clips = pd.read_csv(archive.open("train.csv"))
        sign_to_label = json.load(archive.open("sign_to_prediction_index_map.json"))
        if args.limit:
            clips = clips.head(args.limit)

        frames = []
        for i, path in enumerate(clips["path"], start=1):
            frames.append(read_clip(archive, path))
            if i % 5000 == 0:
                print(f"{i:,} / {len(clips):,} clips")

    lengths = [len(clip) for clip in frames]
    clips["label"] = clips["sign"].map(sign_to_label)
    clips["start"] = np.cumsum([0] + lengths[:-1])
    clips["length"] = lengths

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "landmarks.npy", np.concatenate(frames))
    columns = ["sequence_id", "participant_id", "sign", "label", "start", "length"]
    clips[columns].to_csv(OUT_DIR / "clips.csv", index=False)

    print(f"Saved {len(clips):,} clips ({sum(lengths):,} frames) to {OUT_DIR}")


if __name__ == "__main__":
    main()
