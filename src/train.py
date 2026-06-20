"""
Fine-tune EfficientNet-B3 for decade classification.

Usage:
    python src/train.py

Checkpoints are written to checkpoints/ after each epoch.
Best model (by val accuracy) is saved as checkpoints/best.pt.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

from dataset import (
    load_metadata,
    train_val_split,
    make_weighted_sampler,
    PhotoDateDataset,
    TRAIN_TRANSFORMS,
    EVAL_TRANSFORMS,
    DECADES,
)

NUM_CLASSES = len(DECADES)  # 15
CHECKPOINT_DIR = Path("checkpoints")


def build_model(freeze_backbone: bool = True) -> nn.Module:
    """
    Load pretrained EfficientNet-B3, replace the classifier head for 15 classes,
    and adapt the first conv layer to accept 1-channel (grayscale) input.
    """
    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=NUM_CLASSES)

    # Adapt stem conv from 3-channel RGB to 1-channel grayscale by averaging weights.
    # This preserves the pretrained feature detector magnitudes.
    orig_conv = model.conv_stem
    new_conv = nn.Conv2d(
        1,
        orig_conv.out_channels,
        kernel_size=orig_conv.kernel_size,
        stride=orig_conv.stride,
        padding=orig_conv.padding,
        bias=orig_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
    model.conv_stem = new_conv

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    return model


def unfreeze_blocks(model: nn.Module, n_blocks: int):
    """Unfreeze the last n_blocks of the EfficientNet backbone for stage-2 fine-tuning."""
    blocks = list(model.blocks)
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    # Always keep the head unfrozen
    for param in model.classifier.parameters():
        param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def save_checkpoint(model, optimizer, epoch, val_acc, path: Path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_acc": val_acc,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-head", type=int, default=5,
                        help="Epochs to train with backbone frozen (head only)")
    parser.add_argument("--epochs-finetune", type=int, default=15,
                        help="Epochs to train with last 3 blocks unfrozen")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = load_metadata()
    train_rows, val_rows = train_val_split(rows)
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}")

    train_ds = PhotoDateDataset(train_rows, transform=TRAIN_TRANSFORMS)
    val_ds = PhotoDateDataset(val_rows, transform=EVAL_TRANSFORMS)

    sampler = make_weighted_sampler(train_rows)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    model = build_model(freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0

    # ── Stage 1: head only ────────────────────────────────────────────────────
    print(f"\n── Stage 1: head only ({args.epochs_head} epochs) ─────────────")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_head
    )

    for epoch in range(1, args.epochs_head + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:>2}/{args.epochs_head} "
            f"| train loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc,
                            CHECKPOINT_DIR / "best.pt")

        save_checkpoint(model, optimizer, epoch, val_acc,
                        CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt")

    # ── Stage 2: unfreeze last 3 blocks ──────────────────────────────────────
    print(f"\n── Stage 2: backbone fine-tune ({args.epochs_finetune} epochs) ──")
    unfreeze_blocks(model, n_blocks=3)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_finetune, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_finetune
    )

    for epoch in range(1, args.epochs_finetune + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_total = args.epochs_head + epoch
        print(
            f"Epoch {epoch_total:>2}/{args.epochs_head + args.epochs_finetune} "
            f"| train loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch_total, val_acc,
                            CHECKPOINT_DIR / "best.pt")

        save_checkpoint(model, optimizer, epoch_total, val_acc,
                        CHECKPOINT_DIR / f"epoch_{epoch_total:03d}.pt")

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    print(f"Best checkpoint: {CHECKPOINT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
