"""
Audit the test split for contamination.

A split that is disjoint by filename can still be contaminated: archives often
hold the same photograph under several item ids — alternate scans, crops, or
plates from one sitting — and those pass every filename-level check while
leaking training content into the test set.

This checks three things:

    1. Exact duplicates   identical bytes, and identical decoded pixels
    2. Near-duplicates    perceptual hash (dhash) within a Hamming radius
    3. Impact             with --impact, accuracy on the flagged test images
                          versus the rest

The third is the one that matters. Near-duplicates are only a problem if the
model is memorising them, and memorisation shows up as near-perfect accuracy on
the flagged subset. Elevated-but-ordinary accuracy there means those images are
merely similar, not the same photograph.

Usage:
    python src/leakage_check.py              # audit only; no checkpoints needed
    python src/leakage_check.py --impact     # also score flagged vs clean
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import load_metadata, train_val_test_split

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_RADII = (0, 2, 5, 8)


def dhash(img: Image.Image) -> int:
    """
    64-bit difference hash: compare each pixel with its right-hand neighbour on a
    9x8 thumbnail. Robust to rescaling and mild tonal shifts, which is what
    separates a re-scan of one photograph from a genuinely different photograph.
    """
    small = np.asarray(img.resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def bit_matrix(hashes: list[int]) -> np.ndarray:
    """Unpack 64-bit hashes into an (n, 64) uint8 matrix for vectorised Hamming."""
    arr = np.array(hashes, dtype=np.uint64)
    return np.unpackbits(arr.view(np.uint8).reshape(-1, 8), axis=1)


def min_cross_distance(query: np.ndarray, reference: np.ndarray, chunk: int = 200) -> np.ndarray:
    """Smallest Hamming distance from each query row to any reference row."""
    out = np.empty(len(query), dtype=np.int32)
    for start in range(0, len(query), chunk):
        block = query[start:start + chunk]
        dist = (block[:, None, :] != reference[None, :, :]).sum(axis=2)
        out[start:start + chunk] = dist.min(axis=1)
    return out


def fingerprint(path: str) -> tuple[str, str, int]:
    """(file md5, pixel md5, dhash). The pixel md5 catches re-encodes of one image."""
    raw = Path(path).read_bytes()
    img = Image.open(path).convert("L")
    return (
        hashlib.md5(raw).hexdigest(),
        hashlib.md5(np.asarray(img).tobytes()).hexdigest(),
        dhash(img),
    )


def cross_split_groups(digests: dict[str, str], split_of: dict[str, str]) -> list[list[str]]:
    """Digest collisions whose members do not all sit in the same split."""
    groups = defaultdict(list)
    for name, digest in digests.items():
        groups[digest].append(name)
    return [
        files for files in groups.values()
        if len(files) > 1 and len({split_of[f] for f in files}) > 1
    ]


def score_subsets(flagged: np.ndarray, test_rows: list[dict], seeds: list[int]) -> dict:
    """Per-seed accuracy on the flagged test images versus the rest."""
    import torch
    from torch.utils.data import DataLoader

    from dataset import EVAL_TRANSFORMS, PhotoDateDataset
    from train import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        PhotoDateDataset(test_rows, transform=EVAL_TRANSFORMS),
        batch_size=64, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )

    hits = []
    for seed in seeds:
        ckpt = CHECKPOINT_DIR / f"best_seed{seed}.pt"
        if not ckpt.exists():
            raise SystemExit(f"{ckpt} not found — train that seed, or drop --impact")
        model = load_checkpoint(ckpt, device)
        preds, labels = [], []
        with torch.inference_mode():
            for imgs, y in loader:
                preds.append(model(imgs.to(device)).argmax(1).cpu().numpy())
                labels.append(y.numpy())
        hits.append(np.concatenate(preds) == np.concatenate(labels))

    correct = np.stack(hits)
    return {
        "n_flagged": int(flagged.sum()),
        "flagged_accuracy": round(float(correct[:, flagged].mean()), 4) if flagged.any() else None,
        "clean_accuracy": round(float(correct[:, ~flagged].mean()), 4),
        "full_accuracy": round(float(correct.mean()), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impact", action="store_true",
                        help="Also score flagged vs clean test images (needs checkpoints)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--radius", type=int, default=5,
                        help="Hamming radius treated as a near-duplicate for --impact")
    args = parser.parse_args()

    train, val, test = train_val_test_split(load_metadata())
    split_of = {r["filename"]: name
                for name, rows in (("train", train), ("val", val), ("test", test))
                for r in rows}
    print(f"{len(train)} train / {len(val)} val / {len(test)} test")

    print("Fingerprinting...")
    file_md5, pixel_md5, hashes = {}, {}, {}
    for i, name in enumerate(split_of, 1):
        file_md5[name], pixel_md5[name], hashes[name] = fingerprint(name)
        if i % 2000 == 0:
            print(f"  {i}/{len(split_of)}")

    report = {"n_train": len(train), "n_val": len(val), "n_test": len(test)}

    for label, digests in (("file", file_md5), ("pixel", pixel_md5)):
        groups = cross_split_groups(digests, split_of)
        report[f"exact_{label}_duplicate_groups"] = len(groups)
        print(f"\nExact {label} duplicates spanning >1 split: {len(groups)}")
        for files in groups[:5]:
            print("   " + " | ".join(f"{split_of[f]}:{Path(f).name}" for f in files))

    # Near-duplicates are measured test -> train only. Val never updates weights,
    # so a test image resembling a val image is not leakage.
    train_bits = bit_matrix([hashes[r["filename"]] for r in train])
    test_bits = bit_matrix([hashes[r["filename"]] for r in test])
    min_dist = min_cross_distance(test_bits, train_bits)

    print("\nTest images by closest training-image match:")
    report["near_duplicates"] = {}
    for radius in DEFAULT_RADII:
        n = int((min_dist <= radius).sum())
        report["near_duplicates"][f"hamming_<={radius}"] = {
            "count": n, "share": round(n / len(test), 4)
        }
        print(f"  Hamming <= {radius}: {n:4d} ({n / len(test):5.1%})")

    if args.impact:
        print(f"\nScoring flagged (Hamming <= {args.radius}) vs clean...")
        impact = score_subsets(min_dist <= args.radius, test, args.seeds)
        impact["radius"] = args.radius
        impact["seeds"] = args.seeds
        report["impact"] = impact
        print(f"  flagged ({impact['n_flagged']:3d} images): {impact['flagged_accuracy']}")
        print(f"  clean                     : {impact['clean_accuracy']}")
        print(f"  full test set             : {impact['full_accuracy']}")
        print("\nMemorised duplicates would score near 1.0 on the flagged subset.")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "leakage_audit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
