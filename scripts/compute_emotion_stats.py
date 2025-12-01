"""Summarize emotion labels and geometric features from predictions."""

import argparse
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from data import FacialKeypointsDataset
from src.data.transforms import Compose, Rescale
from src.utils.emotion import EmotionClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize emotion predictions and feature stats")
    parser.add_argument("--predictions", default="artifacts/task1_hpc/test_predictions.csv")
    parser.add_argument("--test-csv", default="data/test_frames_keypoints.csv")
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    df = pd.read_csv(pred_path)
    counts = Counter(df["emotion"])
    total = sum(counts.values()) or 1
    print("Emotion distribution:")
    for label in sorted(counts):
        print(f"  {label:>8}: {counts[label]} ({counts[label] / total:.2%})")

    dataset = FacialKeypointsDataset(
        args.test_csv,
        args.test_root,
        transform=Compose([Rescale(args.image_size)])
    )
    lookup = {sample["image_name"]: sample for sample in dataset}
    classifier = EmotionClassifier()

    feature_rows = []
    for _, row in df.iterrows():
        sample = lookup.get(row["image_name"])
        if not sample:
            continue
        features = classifier._compute_features(sample["keypoints"])
        features["emotion"] = row["emotion"]
        feature_rows.append(features)

    if feature_rows:
        features_df = pd.DataFrame(feature_rows)
        print("\nFeature means per emotion:")
        print(features_df.groupby("emotion").mean().round(3))


if __name__ == "__main__":
    main()
