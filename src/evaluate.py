"""
Evaluate a trained checkpoint on one split.

Metrics:
    - Overall accuracy
    - Per-decade precision / recall / F1
    - Confusion matrix (saved as PNG)
    - Mean decade error (years)

Usage:
    python src/evaluate.py --split test
    python src/evaluate.py --split val --checkpoint checkpoints/best.pt

Report published numbers from --split test, and score it once. Checking test,
adjusting something, then checking again turns it into a second val split and
reintroduces exactly the selection bias it exists to avoid.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import (
    DECADES,
    IDX_TO_DECADE,
    EVAL_TRANSFORMS,
    PhotoDateDataset,
    load_metadata,
    train_val_test_split,
)
from train import load_checkpoint

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")


@torch.inference_mode()
def collect_predictions(model, loader, device):
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return np.array(all_preds), np.array(all_labels)


def mean_decade_error(preds: np.ndarray, labels: np.ndarray) -> float:
    pred_years = np.array([IDX_TO_DECADE[i] for i in preds])
    true_years = np.array([IDX_TO_DECADE[i] for i in labels])
    return float(np.mean(np.abs(pred_years - true_years)))


def hardest_decade_pair(cm: np.ndarray) -> tuple[str, str, int]:
    np.fill_diagonal(cm, 0)
    i, j = divmod(cm.argmax(), cm.shape[1])
    return f"{DECADES[i]}s", f"{DECADES[j]}s", int(cm[i, j])


def plot_confusion_matrix(cm: np.ndarray, path: Path):
    labels = [f"{d}s" for d in DECADES]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Decade Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test",
                        help="Which split to score; publish from test")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for output filenames, e.g. seed1 — keeps "
                             "multi-seed runs from overwriting each other")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    splits = dict(zip(("train", "val", "test"), train_val_test_split(load_metadata())))
    rows = splits[args.split]
    print(f"Scoring {len(rows)} images from the {args.split} split")

    ds = PhotoDateDataset(rows, transform=EVAL_TRANSFORMS)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )

    model = load_checkpoint(args.checkpoint, device)
    preds, labels = collect_predictions(model, loader, device)

    accuracy = (preds == labels).mean()
    mde = mean_decade_error(preds, labels)
    decade_labels = [f"{d}s" for d in DECADES]
    report = classification_report(
        labels, preds, labels=list(range(len(DECADES))),
        target_names=decade_labels, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(DECADES))))
    pair_a, pair_b, pair_n = hardest_decade_pair(cm.copy())

    # Standard error on the accuracy estimate — 563 test images is not 63.3% exactly.
    stderr = float(np.sqrt(accuracy * (1 - accuracy) / len(labels)))

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = f"{args.split}_{args.tag}" if args.tag else args.split
    cm_path = RESULTS_DIR / f"confusion_matrix_{suffix}.png"
    metrics_path = RESULTS_DIR / f"metrics_{suffix}.json"
    report_path = RESULTS_DIR / f"classification_report_{suffix}.txt"
    plot_confusion_matrix(cm, cm_path)

    metrics = {
        "split": args.split,
        "tag": args.tag,
        "accuracy": round(float(accuracy), 4),
        "accuracy_stderr": round(stderr, 4),
        "mean_decade_error_years": round(mde, 2),
        "hardest_decade_pair": f"{pair_a} -> {pair_b}",
        "confusion_count": pair_n,
        "n_samples": len(labels),
        "checkpoint": str(args.checkpoint),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{args.split.capitalize()} accuracy  : {accuracy:.3f} +/- {stderr:.3f}")
    print(f"Mean decade error : {mde:.1f} years")
    print(f"Hardest confusion : {pair_a} predicted as {pair_b} ({pair_n} times)")
    print(f"\nPer-decade report:\n{report}")
    print(f"Confusion matrix  : {cm_path}")
    print(f"Metrics JSON      : {metrics_path}")


if __name__ == "__main__":
    main()
