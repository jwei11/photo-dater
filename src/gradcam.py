"""
Grad-CAM heatmaps for decade classification.

Usage:
    python src/gradcam.py                  # save 8 val examples to results/gradcam/
    python src/gradcam.py --n 16
"""

import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from dataset import (
    IDX_TO_DECADE,
    EVAL_TRANSFORMS,
    PhotoDateDataset,
    load_metadata,
    train_val_test_split,
)
from train import load_checkpoint

RESULTS_DIR = Path("results/gradcam")
CHECKPOINT_DIR = Path("checkpoints")


class GradCAM:
    """Grad-CAM for the last EfficientNet backbone block."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        confidence = float(logits.softmax(1)[0, class_idx].item())
        return cam, class_idx, confidence


def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a CAM heatmap over a grayscale PIL image; returns RGB uint8 array."""
    base = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    colored = cm.jet(cam)[:, :, :3]
    blended = np.clip((1 - alpha) * base + alpha * colored, 0, 1)
    return (blended * 255).astype(np.uint8)


def save_example(
    row: dict,
    model: torch.nn.Module,
    gradcam: GradCAM,
    device: torch.device,
    out_dir: Path,
):
    pil_img = Image.open(row["filename"]).convert("L")
    tensor = EVAL_TRANSFORMS(pil_img).unsqueeze(0).to(device)

    cam, pred_idx, confidence = gradcam(tensor)
    true_decade = int(row["decade"])
    pred_decade = IDX_TO_DECADE[pred_idx]
    overlay = overlay_heatmap(pil_img, cam)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(pil_img, cmap="gray")
    axes[0].set_title(f"Actual: {true_decade}s")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    correct = "OK" if pred_decade == true_decade else "MISS"
    axes[2].set_title(f"Pred: {pred_decade}s ({confidence:.0%}) [{correct}]")
    axes[2].axis("off")

    title = (row.get("title") or row.get("item_id", ""))[:80]
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()

    fname = f"{true_decade}_{pred_decade}_{Path(row['filename']).stem}.png"
    fig.savefig(out_dir / fname, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--n", type=int, default=8, help="Number of examples to visualize")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val",
                        help="Which split to sample from; val by default to leave test clean")
    parser.add_argument("--seed", type=int, default=42,
                        help="Controls which examples are sampled, not the data split")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    splits = dict(zip(("train", "val", "test"), train_val_test_split(load_metadata())))
    rows = splits[args.split]
    rng = np.random.default_rng(args.seed)
    sample_rows = [rows[i] for i in rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)]

    model = load_checkpoint(args.checkpoint, device)
    gradcam = GradCAM(model, model.blocks[-1])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for row in sample_rows:
        save_example(row, model, gradcam, device, RESULTS_DIR)

    print(f"Saved {len(sample_rows)} heatmaps to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
