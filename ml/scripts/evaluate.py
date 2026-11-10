"""Break down val (or test) accuracy for one trained Transformer, or for several combined:
overall, without the very short clips, and per person.

About 12% of clips are 6 frames or fewer. They are fragments of a sign (the hand covers
about a quarter of the distance it does in a full clip), so they are much harder than
anything the app's camera will see. The score without them is closer to real use.

Given several checkpoints, the models are combined: for each clip, their probabilities
for each sign are averaged and the highest average wins. Models that make different
mistakes cancel some of each other's mistakes out.

Usage, from ml/:
    .venv/bin/python scripts/evaluate.py transformer_face
    .venv/bin/python scripts/evaluate.py transformer_face transformer_face_seed1 transformer_face_seed2
    .venv/bin/python scripts/evaluate.py transformer_big --dim 256 --layers 5   # same size as it was trained with

Test is scored only once, for the final models (trained with --final):
    .venv/bin/python scripts/evaluate.py final_seed0 final_seed1 final_seed2 --split test
"""

import argparse

import pandas as pd
import torch

from train_transformer import ConvTransformer
from training import CHECKPOINT_DIR, PREPARED_DIR, get_device, load_split

SHORT_CLIP = 6  # frames


@torch.no_grad()
def probabilities(model, x, mask, batch_size=1024):
    """The model's probability for each sign, for every clip: shape (clips, 250)."""
    model.eval()
    scores = [model(x[i:i + batch_size].float(), mask[i:i + batch_size]) for i in range(0, len(x), batch_size)]
    return torch.cat(scores).softmax(dim=1)


def main():
    parser = argparse.ArgumentParser(description="Break down val or test accuracy for one or more checkpoints.")
    parser.add_argument("names", nargs="+", help="checkpoint names in data/checkpoints/, e.g. transformer_face")
    parser.add_argument("--dim", type=int, default=192, help="model size, the same for every checkpoint")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--split", default="val", choices=["val", "test"], help="test only for the final models")
    args = parser.parse_args()

    device = get_device()
    x, mask, y = load_split(args.split, device)
    clips = pd.read_csv(PREPARED_DIR / f"{args.split}_clips.csv")
    clips["frames"] = (~mask.all(dim=2)).sum(dim=1).cpu().numpy()
    long_enough = clips["frames"] > SHORT_CLIP

    total = 0
    for name in args.names:
        model = ConvTransformer(dim=args.dim, layers=args.layers).to(device)
        model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{name}.pt", map_location=device))
        probs = probabilities(model, x, mask)
        print(f"{name}: {args.split} top-1 {(probs.argmax(dim=1) == y).float().mean().item():.1%}")
        total = total + probs

    clips["right"] = (total.argmax(dim=1) == y).cpu().numpy()
    print(f"\n{'combined' if len(args.names) > 1 else args.names[0]}, {args.split} top-1:")
    print(f"  all clips:                       {clips['right'].mean():.1%}  ({len(clips):,} clips)")
    print(f"  without clips of <= {SHORT_CLIP} frames:   {clips[long_enough]['right'].mean():.1%}  ({long_enough.sum():,} clips)")
    print("  per person:")
    for participant, right in clips.groupby("participant_id")["right"]:
        print(f"    {participant}: {right.mean():.1%}")


if __name__ == "__main__":
    main()
