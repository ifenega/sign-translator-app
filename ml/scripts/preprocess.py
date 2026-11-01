"""Turn the extracted landmarks into fixed-size arrays for training.

Reads data/processed/landmarks.npy and clips.csv (made by extract_landmarks.py) and,
for each clip:

1. keeps 74 landmarks: the 20 outer-lip points, both hands and the upper body, with
   x and y only (MediaPipe's z is a rough depth guess and mostly noise),
2. trims frames with no hand from the start and end of the clip,
3. centres each frame between the shoulders and divides by shoulder width, so it
   doesn't matter where the signer stands or how close they are to the camera,
4. mirrors the clip if the left hand did most of the signing, so the dominant hand is
   always in the right-hand slot,
5. fills missing points with 0 and keeps a mask of which ones were missing,
6. resamples or pads the clip to 64 frames.

Clips are split by participant, so val and test only contain people the model never
saw in training. Saved to data/prepared/, for each split (train, val, test):

    {split}_x.npy      (clips, 64, 74, 2)  float32 x, y of each landmark
    {split}_mask.npy   (clips, 64, 74)     True where a landmark was missing or padded
    {split}_y.npy      (clips,)            sign label, 0-249
    {split}_clips.csv                      the clip behind each row

Landmark order in the output:
    0-19   lips
    20-40  other hand
    41-52  upper body (pose 11-22: shoulders, elbows, wrists, hands)
    53-73  dominant hand

Usage, from ml/:
    .venv/bin/python scripts/preprocess.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from extract_landmarks import LIPS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR / "prepared"

NUM_FRAMES = 64
TEST_PARTICIPANTS = 3
VAL_PARTICIPANTS = 3
SEED = 0  # fixes which participants land in val and test

# Positions in landmarks.npy (see extract_landmarks.py). The first 20 lip points
# are the outer lip outline; the other 20 (inner lips) are dropped.
KEEP = list(range(0, 20)) + list(range(40, 94))

# Positions after KEEP.
OUTER_LIPS = LIPS[:20]
LEFT_HAND = slice(20, 41)
UPPER_BODY_START = 41  # pose point 11
RIGHT_HAND = slice(53, 74)
LEFT_SHOULDER, RIGHT_SHOULDER = 41, 42  # pose points 11 and 12

# Face-mesh lip points and their mirror image on the other side of the mouth.
LIP_MIRROR = {61: 291, 185: 409, 40: 270, 39: 269, 37: 267, 0: 0,
              146: 375, 91: 321, 181: 405, 84: 314, 17: 17}
LIP_MIRROR.update({b: a for a, b in LIP_MIRROR.items()})


def mirror_order():
    """For each of the 74 landmarks, the position of its mirror image (left <-> right)."""
    lips = [OUTER_LIPS.index(LIP_MIRROR[point]) for point in OUTER_LIPS]
    left_hand = list(range(53, 74))  # left hand swaps with right hand
    # Pose points come in left/right pairs: 11-12, 13-14, ... 21-22.
    upper_body = [UPPER_BODY_START + (i + 1 if i % 2 == 0 else i - 1) for i in range(12)]
    right_hand = list(range(20, 41))
    return lips + left_hand + upper_body + right_hand


MIRROR = mirror_order()


def prepare_clip(frames):
    """Turn one clip of shape (frames, 94, 3) into points (64, 74, 2) and a mask (64, 74)."""
    xy = frames[:, KEEP, :2].astype(np.float32)

    # Trim frames with no hand from the start and end. Gaps in the middle stay: the
    # hand was there but not detected, and the upper-body points still show where it was.
    left_seen = ~np.isnan(xy[:, LEFT_HAND, 0]).all(axis=1)
    right_seen = ~np.isnan(xy[:, RIGHT_HAND, 0]).all(axis=1)
    hand_frames = np.flatnonzero(left_seen | right_seen)
    xy = xy[hand_frames[0]:hand_frames[-1] + 1]

    # Centre each frame between the shoulders. Shoulder width is the clip's median,
    # which is steadier than measuring it again in every frame.
    left_shoulder, right_shoulder = xy[:, LEFT_SHOULDER], xy[:, RIGHT_SHOULDER]
    centre = (left_shoulder + right_shoulder) / 2
    shoulder_width = np.nanmedian(np.linalg.norm(left_shoulder - right_shoulder, axis=1))
    xy = (xy - centre[:, np.newaxis]) / shoulder_width

    # Mirror left-handed signing: swap left and right landmarks, then flip x.
    if left_seen.sum() > right_seen.sum():
        xy = xy[:, MIRROR]
        xy[..., 0] *= -1

    missing = np.isnan(xy).any(axis=2)
    xy = np.nan_to_num(xy, nan=0.0)
    return to_fixed_length(xy, missing)


def to_fixed_length(xy, missing):
    """Pick 64 evenly spaced frames from a long clip, or pad a short one with missing frames."""
    if len(xy) > NUM_FRAMES:
        chosen = np.linspace(0, len(xy) - 1, NUM_FRAMES).round().astype(int)
        return xy[chosen], missing[chosen]

    padding = NUM_FRAMES - len(xy)
    xy = np.pad(xy, ((0, padding), (0, 0), (0, 0)))
    missing = np.pad(missing, ((0, padding), (0, 0)), constant_values=True)
    return xy, missing


def assign_splits(clips):
    """Put each participant's clips in train, val or test."""
    participants = np.random.default_rng(SEED).permutation(sorted(clips["participant_id"].unique()))
    test = participants[:TEST_PARTICIPANTS]
    val = participants[TEST_PARTICIPANTS:TEST_PARTICIPANTS + VAL_PARTICIPANTS]

    split = pd.Series("train", index=clips.index)
    split[clips["participant_id"].isin(val)] = "val"
    split[clips["participant_id"].isin(test)] = "test"
    return split


def main():
    clips = pd.read_csv(IN_DIR / "clips.csv")
    landmarks = np.load(IN_DIR / "landmarks.npy", mmap_mode="r")
    clips["split"] = assign_splits(clips)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        split_clips = clips[clips["split"] == split]
        x = np.zeros((len(split_clips), NUM_FRAMES, len(KEEP), 2), dtype=np.float32)
        mask = np.zeros((len(split_clips), NUM_FRAMES, len(KEEP)), dtype=bool)

        for i, clip in enumerate(split_clips.itertuples()):
            x[i], mask[i] = prepare_clip(landmarks[clip.start:clip.start + clip.length])

        np.save(OUT_DIR / f"{split}_x.npy", x)
        np.save(OUT_DIR / f"{split}_mask.npy", mask)
        np.save(OUT_DIR / f"{split}_y.npy", split_clips["label"].to_numpy())
        split_clips[["sequence_id", "participant_id", "sign", "label"]].to_csv(
            OUT_DIR / f"{split}_clips.csv", index=False
        )
        participants = sorted(split_clips["participant_id"].unique().tolist())
        print(f"{split}: {len(split_clips):,} clips from participants {participants}")


if __name__ == "__main__":
    main()
