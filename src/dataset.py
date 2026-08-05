"""
PyTorch Dataset for decade-classified photographs.

Reads processed images from data/processed/ and labels from data/processed_metadata.csv.
Applies weighted sampling to correct for decade imbalance (pre-1900 underrepresented).
"""

import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

PROCESSED_META_FILE = Path("data/processed_metadata.csv")

# Fixed on purpose, and deliberately NOT the same knob as train.py's --seed.
# Every run has to see the same test set: if the split moved with the training
# seed, a multi-seed comparison would score each run on photos that other runs
# had trained on.
SPLIT_SEED = 42

DECADES = list(range(1850, 2000, 10))
DECADE_TO_IDX = {d: i for i, d in enumerate(DECADES)}  # 1850->0, 1860->1, ... 1990->14
IDX_TO_DECADE = {i: d for d, i in DECADE_TO_IDX.items()}

# Normalization stats for single-channel (grayscale) ImageNet fine-tuning.
# Using the luminance-weighted mean/std derived from ImageNet RGB stats.
NORMALIZE = transforms.Normalize(mean=[0.449], std=[0.226])

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    NORMALIZE,
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    NORMALIZE,
])


def load_metadata(meta_file: Path = PROCESSED_META_FILE) -> list[dict]:
    if not meta_file.exists():
        raise FileNotFoundError(f"{meta_file} not found — run src/preprocess.py first")
    with open(meta_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # The scraper can write the same item twice on retry. Dedupe by filename before
    # splitting: one duplicate landing on either side of a boundary would put the
    # same photograph in both train and test.
    seen, unique = set(), []
    for row in rows:
        if row["filename"] not in seen:
            seen.add(row["filename"])
            unique.append(row)

    if len(unique) != len(rows):
        print(f"load_metadata: dropped {len(rows) - len(unique)} duplicate row(s)")
    return unique


class PhotoDateDataset(Dataset):
    def __init__(self, rows: list[dict], transform=None):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["filename"])  # already grayscale L from preprocess
        if self.transform:
            img = self.transform(img)
        label = DECADE_TO_IDX[int(row["decade"])]
        return img, label


def make_weighted_sampler(rows: list[dict]) -> WeightedRandomSampler:
    """
    Up-sample underrepresented decades so each epoch sees a balanced mix.
    If all decades hit the 500-image target this is effectively a uniform shuffle —
    it only matters when early decades fall short in the LOC collection.
    """
    counts: dict[int, int] = {}
    for row in rows:
        d = int(row["decade"])
        counts[d] = counts.get(d, 0) + 1

    weights = [1.0 / counts[int(row["decade"])] for row in rows]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_val_test_split(
    rows: list[dict],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = SPLIT_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratified 70/15/15 split, holding out both fractions from each decade.

        train — the only rows that produce gradient updates
        val   — used to pick the best epoch during training
        test  — scored once, at the end, for the numbers we report

    Keeping val and test separate matters: `best.pt` is chosen as the argmax of
    ~20 noisy per-epoch scores, so the winning score is inflated by whichever
    checkpoint got luckiest on that particular set. Reporting on test — which
    played no part in choosing anything — removes that bias.
    """
    rng = random.Random(seed)

    by_decade: dict[int, list[dict]] = {}
    for row in rows:
        d = int(row["decade"])
        by_decade.setdefault(d, []).append(row)

    train_rows, val_rows, test_rows = [], [], []
    # Sorted by decade so the split is identical regardless of CSV row order.
    for decade, decade_rows in sorted(by_decade.items()):
        n = len(decade_rows)
        if n < 3:
            raise ValueError(f"decade {decade}s has only {n} image(s) — cannot split three ways")

        shuffled = decade_rows[:]
        rng.shuffle(shuffled)

        n_val = max(1, int(n * val_fraction))
        n_test = max(1, int(n * test_fraction))

        val_rows.extend(shuffled[:n_val])
        test_rows.extend(shuffled[n_val:n_val + n_test])
        train_rows.extend(shuffled[n_val + n_test:])

    return train_rows, val_rows, test_rows
