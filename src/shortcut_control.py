"""
Shortcut controls: how much accuracy survives when the content is destroyed?

The premise of this project is that the model reads era from photographic
content — clothing, subjects, composition, film characteristics. The competing
explanation is that it reads digitization artifacts: scan tone, JPEG texture,
border and mount styling that correlate with when the Library of Congress
happened to digitize a given collection.

Grad-CAM cannot settle this. A heatmap over a person's face shows where gradient
flows, not that the model would fail without the face. These probes test that
directly, by destroying one kind of information at a time and re-scoring:

    downsample_N    Resize to NxN and back. Destroys texture, grain, and fine
                    detail; preserves coarse layout and overall tone. High
                    accuracy at N=8 means the signal is global brightness and
                    contrast, not photographic content.

    patch_shuffle_N Scramble an NxN grid of tiles. Destroys global composition
                    and subject matter; preserves local texture and grain. The
                    complement of downsampling — high accuracy here means the
                    model reads film/scan texture.

    center_crop_F   Keep the central fraction F, discarding borders. A large
                    drop means borders, mounts, and vignetting carried signal.

    border_only_F   The inverse: gray out the central fraction, keep the frame.
                    Accuracy well above chance with all content removed is the
                    most direct evidence of border-artifact shortcutting.

Read the retained column, not raw accuracy: it is the share of above-chance
signal that survives, (probe - chance) / (baseline - chance), where chance is
1/15 = 6.7%.

Usage:
    python src/shortcut_control.py
    python src/shortcut_control.py --seeds 1 2 --save-samples
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import (
    DECADES,
    EVAL_TRANSFORMS,
    PhotoDateDataset,
    load_metadata,
    train_val_test_split,
)
from train import load_checkpoint

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")
IMG_SIZE = 300
CHANCE = 1.0 / len(DECADES)


class Identity:
    name = "baseline"

    def __call__(self, img: Image.Image) -> Image.Image:
        return img


class Downsample:
    """Resize down to n x n and back — keeps coarse tone, destroys texture."""

    def __init__(self, n: int):
        self.n = n
        self.name = f"downsample_{n}"

    def __call__(self, img: Image.Image) -> Image.Image:
        small = img.resize((self.n, self.n), Image.BILINEAR)
        return small.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


class PatchShuffle:
    """Scramble a grid of tiles — keeps local texture, destroys composition."""

    def __init__(self, grid: int, seed: int = 0):
        self.grid = grid
        self.name = f"patch_shuffle_{grid}"
        # One fixed permutation for every image, so the probe is reproducible.
        self.perm = np.random.default_rng(seed).permutation(grid * grid)

    def __call__(self, img: Image.Image) -> Image.Image:
        step = IMG_SIZE // self.grid
        tiles = [
            img.crop((c * step, r * step, (c + 1) * step, (r + 1) * step))
            for r in range(self.grid)
            for c in range(self.grid)
        ]
        out = Image.new("L", (step * self.grid, step * self.grid))
        for dest, src in enumerate(self.perm):
            out.paste(tiles[src], ((dest % self.grid) * step, (dest // self.grid) * step))
        return out.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


class CenterCrop:
    """Keep the central fraction, discarding borders and mounts."""

    def __init__(self, frac: float):
        self.frac = frac
        self.name = f"center_crop_{int(frac * 100)}"

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        cw, ch = int(w * self.frac), int(h * self.frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        return img.crop((left, top, left + cw, top + ch)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )


def _fill_value(img: Image.Image, fill: int | None) -> int:
    """
    Occlusion colour. Defaults to the image's own mean rather than a fixed grey:
    a hard 128 rectangle is far outside anything in training, and the model reacts
    to it unpredictably. Using the per-image mean leaks that image's average
    brightness into the masked region, which is acceptable here because the
    downsample probes establish that global tone carries almost no signal.
    """
    return fill if fill is not None else int(np.asarray(img).mean())


class BorderOnly:
    """
    Mask the central `frac` (linear), keeping only the outer ring.

    Note that `frac` is linear, not areal: frac=0.70 masks 0.70^2 = 49% of the
    area, so the surviving ring is over half the image and still full of subject
    matter. Isolating actual borders needs frac >= 0.90, which leaves 19% or less.
    """

    def __init__(self, frac: float, fill: int | None = None):
        self.frac = frac
        self.fill = fill
        self.name = f"border_only_{int(frac * 100)}"

    def __call__(self, img: Image.Image) -> Image.Image:
        out = img.copy()
        w, h = img.size
        cw, ch = int(w * self.frac), int(h * self.frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        ImageDraw.Draw(out).rectangle(
            [left, top, left + cw, top + ch], fill=_fill_value(img, self.fill)
        )
        return out


class CenterOnly:
    """
    Gray out everything outside the central fraction — the exact complement of
    BorderOnly, and the clean test for border shortcutting. CenterCrop also
    rescales what it keeps, so a drop there confounds 'borders removed' with
    'magnification changed'; this keeps every surviving pixel at its original
    scale and position.
    """

    def __init__(self, frac: float, fill: int | None = None):
        self.frac = frac
        self.fill = fill
        self.name = f"center_only_{int(frac * 100)}"

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        cw, ch = int(w * self.frac), int(h * self.frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        out = Image.new("L", (w, h), color=_fill_value(img, self.fill))
        out.paste(img.crop((left, top, left + cw, top + ch)), (left, top))
        return out


PROBES = [
    Identity(),
    Downsample(32),
    Downsample(16),
    Downsample(8),
    PatchShuffle(5),
    PatchShuffle(10),
    CenterCrop(0.70),
    # Ring thickness sweep. As the surviving ring thins, border_only isolates
    # actual borders and center_only approaches the untouched photograph.
    CenterOnly(0.70), BorderOnly(0.70),   # ring = 51% of area — mostly content
    CenterOnly(0.90), BorderOnly(0.90),   # ring = 19% of area
    CenterOnly(0.95), BorderOnly(0.95),   # ring = 10% of area — borders proper
]


@torch.inference_mode()
def accuracy_under(model, rows, probe, device, batch_size, workers) -> float:
    transform = transforms.Compose([probe, *EVAL_TRANSFORMS.transforms])
    loader = DataLoader(
        PhotoDateDataset(rows, transform=transform),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
    )
    correct = total = 0
    for imgs, labels in loader:
        preds = model(imgs.to(device)).argmax(1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / total


def save_samples(rows, path: Path):
    """Grid showing one photograph under every probe, for the writeup."""
    import matplotlib.pyplot as plt

    img = Image.open(rows[7]["filename"])
    cols = 3
    nrows = -(-len(PROBES) // cols)  # ceil, so no probe is silently dropped
    fig, axes = plt.subplots(nrows, cols, figsize=(4.6 * cols, 4.9 * nrows))
    for ax, probe in zip(axes.ravel(), PROBES):
        ax.imshow(probe(img), cmap="gray", vmin=0, vmax=255)
        ax.set_title(probe.name, fontsize=11)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Shortcut-control probes applied to one test photograph", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-samples", action="store_true",
                        help="Also write a figure showing each probe")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    splits = dict(zip(("train", "val", "test"), train_val_test_split(load_metadata())))
    rows = splits[args.split]
    print(f"{len(rows)} images from the {args.split} split, {len(args.seeds)} seed(s)\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    if args.save_samples:
        sample_path = RESULTS_DIR / "shortcut_control_examples.png"
        save_samples(rows, sample_path)
        print(f"Wrote {sample_path}\n")

    # Seeds outer, probes inner: the checkpoint load dominates, so pay it once.
    scores: dict[str, list[float]] = {probe.name: [] for probe in PROBES}
    for seed in args.seeds:
        ckpt = CHECKPOINT_DIR / f"best_seed{seed}.pt"
        if not ckpt.exists():
            raise SystemExit(f"{ckpt} not found — train that seed first")
        model = load_checkpoint(ckpt, device)
        line = [f"seed {seed}:"]
        for probe in PROBES:
            acc = accuracy_under(model, rows, probe, device, args.batch_size, args.workers)
            scores[probe.name].append(acc)
            line.append(f"{probe.name}={acc:.3f}")
        print("  ".join(line))

    baseline = statistics.mean(scores["baseline"])
    headroom = baseline - CHANCE

    print(f"\n{'probe':<18}{'accuracy':>18}{'retained':>12}")
    print("-" * 48)
    summary = {}
    for probe in PROBES:
        values = scores[probe.name]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        retained = (mean - CHANCE) / headroom if headroom > 0 else float("nan")
        summary[probe.name] = {
            "accuracy_mean": round(mean, 4),
            "accuracy_std": round(std, 4),
            "retained_fraction": round(retained, 3),
        }
        print(f"{probe.name:<18}{mean:>10.3f} +/- {std:.3f}{retained:>11.0%}")
    print("-" * 48)
    print(f"chance = {CHANCE:.3f} (1 of {len(DECADES)} decades)")

    out = RESULTS_DIR / "shortcut_control.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "seeds": args.seeds,
            "n_samples": len(rows),
            "chance": round(CHANCE, 4),
            "probes": summary,
        }, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
