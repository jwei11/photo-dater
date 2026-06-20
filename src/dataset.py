"""
PyTorch Dataset for decade-classified photographs.

Reads processed images from data/processed/ and labels from data/processed_metadata.csv.
Applies weighted sampling to correct for decade imbalance (pre-1900 underrepresented).
"""

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

PROCESSED_META_FILE = Path("data/processed_metadata.csv")

DECADES = list(range(1850, 2000, 10))
DECADE_TO_IDX = {d: i for i, d in enumerate(DECADES)}  # 1850→0, 1860→1, … 1990→14

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
        return list(csv.DictReader(f))


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


def train_val_split(
    rows: list[dict],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified split: hold out val_fraction from each decade."""
    import random
    rng = random.Random(seed)

    by_decade: dict[int, list[dict]] = {}
    for row in rows:
        d = int(row["decade"])
        by_decade.setdefault(d, []).append(row)

    train_rows, val_rows = [], []
    for decade_rows in by_decade.values():
        shuffled = decade_rows[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_rows.extend(shuffled[:n_val])
        train_rows.extend(shuffled[n_val:])

    return train_rows, val_rows
