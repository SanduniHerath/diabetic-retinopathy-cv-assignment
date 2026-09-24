"""
train.py
========
Phase 5 - Training Strategy & Experimental Design
Diabetic Retinopathy Stage Detection (Computer Vision Assignment)

Implements a two-stage transfer learning curriculum:
  Stage 1 (Feature Extraction):
    - Backbone frozen. Only the randomly initialised classification head is trained.
    - High initial learning rate (1e-3) is safe because backbone is not affected.
    - Runs for `stage1_epochs` epochs (default 10) or until EarlyStopping fires.

  Stage 2 (Selective Fine-Tuning):
    - Top `unfreeze_blocks` backbone stages unlocked and trained jointly with the head.
    - Lower learning rate (1e-4) to avoid catastrophic forgetting of ImageNet features.
    - Runs for `stage2_epochs` epochs (default 20) or until EarlyStopping fires.

Overfitting Prevention Techniques:
  - Dropout (p=0.4 / p=0.2) in classification head (built into DRTransferModel).
  - Staged freezing: backbone weights preserved during head warm-up.
  - Early stopping monitoring val_loss with configurable patience.
  - ReduceLROnPlateau: halves LR when val_loss stagnates for `lr_patience` epochs.
  - Class-weighted CrossEntropyLoss: compensates for residual class imbalance
    that remains after augmentation (severe / proliferative DR are still rarer).
  - Weight decay (L2 regularisation, default 1e-4) applied via AdamW optimiser.

Outputs (saved to reports/training/):
  - best_model.pth                  – checkpoint with best val_loss
  - training_curves.png             – train/val loss and accuracy curves
  - learning_rate_curve.png         – per-epoch LR trace
  - hyperparameter_results.csv      – summary table for all HP configurations
  - run_<timestamp>/                – per-run artefacts (curves, checkpoint)

Usage (Google Colab / local):
  python src/train.py --data-root data/split --output-dir reports/training

Author: Sanduni Herath
Repository: https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Portable sys.path setup: works from repo root, src/, or Google Colab
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent          # .../src
_REPO_ROOT = _FILE_DIR.parent                        # .../diabetic-retinopathy-cv-assignment
for _p in (_FILE_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model import build_model, CLASS_NAMES, NUM_CLASSES  # noqa: E402


# ===========================================================================
# 1. Constants & defaults
# ===========================================================================
IMAGE_SIZE: int = 224          # EfficientNet-B0 / ResNet-50 native input
MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)   # ImageNet statistics
STD:  Tuple[float, ...] = (0.229, 0.224, 0.225)


# ===========================================================================
# 2. Data Transforms
# ===========================================================================
def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Training-time augmentation: mild colour jitter + geometric transforms.
    NOTE: Heavy augmentation is already applied offline by augmentation.py.
          These online transforms add lightweight stochastic diversity per
          mini-batch without risk of introducing augmented leakage into val/test.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),   # Slightly oversized for crop
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Validation / test transforms: deterministic centre crop only.
    No stochastic augmentation is applied to validation or test sets.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ===========================================================================
# 3. DataLoader Builders
# ===========================================================================
def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 2,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Constructs train and validation DataLoaders from ImageFolder layout:
      data_root/
        train_augmented/   (or train/)   <- training split
        val/                             <- validation split

    Args:
        data_root:            Path to data/split directory.
        batch_size:           Mini-batch size.
        num_workers:          Parallel data workers (set 0 on Windows if issues occur).
        use_weighted_sampler: Enables inverse-frequency weighted random sampling
                              on top of augmentation to further address imbalance.

    Returns:
        (train_loader, val_loader, class_names)
    """
    root = Path(data_root)

    # Support both augmented and plain train directories
    train_dir = root / "train_augmented"
    if not train_dir.exists():
        train_dir = root / "train"

    val_dir = root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found. Expected one of:\n"
            f"  {root / 'train_augmented'}\n  {root / 'train'}"
        )
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    train_dataset = datasets.ImageFolder(str(train_dir), transform=get_train_transforms())
    val_dataset   = datasets.ImageFolder(str(val_dir),   transform=get_val_transforms())

    class_names: List[str] = train_dataset.classes
    print(f"[Data] Train: {len(train_dataset):,} images across {len(class_names)} classes")
    print(f"[Data] Val  : {len(val_dataset):,}  images")
    print(f"[Data] Classes (train): {class_names}")

    # -------------------------------------------------------------------
    # Class-weighted sampler: draws rarer classes more frequently so each
    # mini-batch has approximately balanced class representation without
    # duplicating images artificially like oversampling.
    # -------------------------------------------------------------------
    sampler = None
    if use_weighted_sampler:
        targets = train_dataset.targets                    # List[int]
        class_counts = np.bincount(targets, minlength=len(class_names))
        class_weights = 1.0 / (class_counts + 1e-6)       # Inverse frequency
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(train_dataset),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),          # Mutually exclusive with sampler
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,                     # Prevents BatchNorm errors on tail batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, class_names


# ===========================================================================
# 4. Class-Weighted Loss Function
# ===========================================================================
def compute_class_weights(
    train_loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes inverse-frequency class weights from the dataset targets.
    These are passed to CrossEntropyLoss to penalise errors on rare classes
    (Severe, Proliferative_DR) more heavily than common classes (No_DR).

    Formula:  w_c = N_total / (C * N_c)
    where N_c = number of samples in class c, C = number of classes.
    """
    dataset = train_loader.dataset
    targets = dataset.targets  # type: ignore[attr-defined]
    counts  = np.bincount(targets, minlength=num_classes).astype(np.float32)
    total   = counts.sum()
    weights = total / (num_classes * (counts + 1e-6))
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[Loss] Class weights: { {CLASS_NAMES[i]: f'{w:.3f}' for i, w in enumerate(weights_t.cpu().tolist())} }")
    return weights_t


# ===========================================================================
# 5. Training Engine (One Epoch)
# ===========================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: optim.Optimizer,
    device: torch.device,
    scaler: Optional[Any],
) -> Tuple[float, float]:
    """
    Runs one complete training epoch with optional AMP (mixed precision).

    Returns:
        (avg_loss, accuracy_pct)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += batch_size

    avg_loss = running_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


# ===========================================================================
# 6. Validation Engine (One Epoch)
# ===========================================================================
@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Runs inference on the full validation set.

    Returns:
        (avg_loss, accuracy_pct)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += batch_size

    avg_loss = running_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


# ===========================================================================
# 7. Training History & Plotting
# ===========================================================================
def plot_training_curves(history: Dict[str, List[float]], output_dir: str, run_tag: str = "") -> None:
    """
    Saves training/validation loss and accuracy curves as PNG files.
    Requires matplotlib; gracefully skips if not available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")        # Non-interactive backend for Colab/server
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not available - skipping curve plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    suffix = f"_{run_tag}" if run_tag else ""

    # --- Loss Curve ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss",      linewidth=2, color="#2196F3")
    ax.plot(epochs, history["val_loss"],   label="Validation Loss", linewidth=2, color="#FF5722", linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(output_dir, f"loss_curve{suffix}.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Loss curve saved to: {loss_path}")

    # --- Accuracy Curve ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_acc"], label="Train Accuracy",      linewidth=2, color="#4CAF50")
    ax.plot(epochs, history["val_acc"],   label="Validation Accuracy", linewidth=2, color="#FF9800", linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_path = os.path.join(output_dir, f"accuracy_curve{suffix}.png")
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Accuracy curve saved to: {acc_path}")

    # --- Learning Rate Trace ---
    if "lr" in history and history["lr"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(epochs, history["lr"], linewidth=2, color="#9C27B0")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        lr_path = os.path.join(output_dir, f"lr_curve{suffix}.png")
        fig.savefig(lr_path, dpi=150)
        plt.close(fig)
        print(f"[Plot] LR curve saved to: {lr_path}")


# ===========================================================================
# 8. Staged Training Orchestrator
# ===========================================================================
def run_staged_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    output_dir: str,
    # Stage 1 hyperparameters
    stage1_lr: float = 1e-3,
    stage1_epochs: int = 10,
    stage1_patience: int = 5,
    # Stage 2 hyperparameters
    stage2_lr: float = 1e-4,
    stage2_epochs: int = 20,
    stage2_patience: int = 8,
    stage2_unfreeze_blocks: int = 3,
    # Shared hyperparameters
    weight_decay: float = 1e-4,
    lr_patience: int = 3,
    lr_factor: float = 0.5,
    use_amp: bool = True,
    run_tag: str = "",
) -> Dict[str, Any]:
    """
    Orchestrates the full two-stage transfer learning training curriculum.

    Stage 1 - Feature Extraction (Head Only):
      Backbone is frozen. Only the custom classification head is optimised.
      This warms up the head weights without corrupting the pretrained features.

    Stage 2 - Selective Fine-Tuning:
      Top `stage2_unfreeze_blocks` backbone stages are unfrozen alongside the head.
      A lower learning rate prevents catastrophic forgetting.

    Both stages use:
      - AdamW optimiser (L2 regularisation via weight_decay)
      - ReduceLROnPlateau scheduler (halves LR when val_loss plateaus)
      - EarlyStopping on val_loss (restores best-epoch weights on trigger)
      - Optional AMP (Automatic Mixed Precision) for Colab T4 GPU speed-up

    Returns:
        Dictionary containing per-epoch metrics and best checkpoint path.
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"best_model{('_' + run_tag) if run_tag else ''}.pth")

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [],
    }

    best_val_loss: float = float("inf")
    best_epoch: int = 0

    # AMP GradScaler (only for CUDA)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None  # type: ignore[attr-defined]

    # -----------------------------------------------------------------------
    # STAGE 1: Backbone Frozen - Train Head Only
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" STAGE 1: Feature Extraction (Backbone Frozen)")
    print(f"   LR={stage1_lr} | Epochs={stage1_epochs} | EarlyStop patience={stage1_patience}")
    print("=" * 60)

    model.freeze_backbone()  # type: ignore[attr-defined]
    optimiser_s1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage1_lr,
        weight_decay=weight_decay,
    )
    scheduler_s1 = ReduceLROnPlateau(
        optimiser_s1, mode="min", factor=lr_factor, patience=lr_patience
    )

    no_improve_s1 = 0
    stage1_start = time.time()

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimiser_s1, device, scaler)
        val_loss,   val_acc   = validate_one_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        current_lr = optimiser_s1.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"  [S1 Epoch {epoch:02d}/{stage1_epochs}] "
            f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} "
            f"TrainAcc={train_acc:.1f}% ValAcc={val_acc:.1f}% "
            f"LR={current_lr:.2e}  ({elapsed:.1f}s)"
        )

        scheduler_s1.step(val_loss)

        # Checkpoint: save model whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    [OK] Checkpoint saved (val_loss={val_loss:.4f})")
            no_improve_s1 = 0
        else:
            no_improve_s1 += 1
            if no_improve_s1 >= stage1_patience:
                print(f"  [EarlyStopping] Stage 1 stopped at epoch {epoch} (patience={stage1_patience})")
                break

    stage1_time = time.time() - stage1_start
    print(f"\n  Stage 1 complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}  ({stage1_time:.0f}s)\n")

    # -----------------------------------------------------------------------
    # STAGE 2: Selective Fine-Tuning (Top Blocks + Head)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f" STAGE 2: Selective Fine-Tuning (Top {stage2_unfreeze_blocks} Blocks + Head)")
    print(f"   LR={stage2_lr} | Epochs={stage2_epochs} | EarlyStop patience={stage2_patience}")
    print("=" * 60)

    # Restore best Stage 1 weights before beginning fine-tuning
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.unfreeze_top_blocks(num_blocks=stage2_unfreeze_blocks)  # type: ignore[attr-defined]

    optimiser_s2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage2_lr,
        weight_decay=weight_decay,
    )
    scheduler_s2 = ReduceLROnPlateau(
        optimiser_s2, mode="min", factor=lr_factor, patience=lr_patience
    )

    no_improve_s2 = 0
    stage2_start = time.time()

    for epoch in range(1, stage2_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimiser_s2, device, scaler)
        val_loss,   val_acc   = validate_one_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        global_epoch = len(history["train_loss"]) + 1
        current_lr = optimiser_s2.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"  [S2 Epoch {epoch:02d}/{stage2_epochs}] "
            f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} "
            f"TrainAcc={train_acc:.1f}% ValAcc={val_acc:.1f}% "
            f"LR={current_lr:.2e}  ({elapsed:.1f}s)"
        )

        scheduler_s2.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = global_epoch
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    [OK] Checkpoint updated (val_loss={val_loss:.4f})")
            no_improve_s2 = 0
        else:
            no_improve_s2 += 1
            if no_improve_s2 >= stage2_patience:
                print(f"  [EarlyStopping] Stage 2 stopped at epoch {epoch} (patience={stage2_patience})")
                break

    stage2_time = time.time() - stage2_start
    print(f"\n  Stage 2 complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}  ({stage2_time:.0f}s)\n")

    # Restore the single best checkpoint across both stages
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"[OK] Best model weights restored from: {checkpoint_path}")

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path,
        "stage1_time_s": stage1_time,
        "stage2_time_s": stage2_time,
    }


# ===========================================================================
# 9. Hyperparameter Search
# ===========================================================================
def run_hyperparameter_search(
    data_root: str,
    output_dir: str,
    device: torch.device,
    backbone: str = "efficientnet_b0",
    batch_size: int = 32,
    num_workers: int = 2,
) -> None:
    """
    Runs a lightweight grid search over key hyperparameters and logs results to CSV.

    Search Grid:
      - stage1_lr       : [1e-3, 5e-4]
      - stage2_lr       : [1e-4, 5e-5]
      - unfreeze_blocks : [2, 3]
      - weight_decay    : [1e-4, 1e-5]

    Results written to: output_dir/hyperparameter_results.csv
    Each run saves its own checkpoint and training curves under output_dir/run_<tag>/.
    """
    configs = [
        {"s1_lr": 1e-3,  "s2_lr": 1e-4,  "unfreeze": 3, "wd": 1e-4},   # Default / Primary
        {"s1_lr": 5e-4,  "s2_lr": 5e-5,  "unfreeze": 3, "wd": 1e-4},   # Lower LR variant
        {"s1_lr": 1e-3,  "s2_lr": 1e-4,  "unfreeze": 2, "wd": 1e-4},   # Fewer unfreeze blocks
        {"s1_lr": 1e-3,  "s2_lr": 1e-4,  "unfreeze": 3, "wd": 1e-5},   # Weaker regularisation
    ]

    results_path = os.path.join(output_dir, "hyperparameter_results.csv")
    fieldnames = [
        "run_tag", "stage1_lr", "stage2_lr", "unfreeze_blocks", "weight_decay",
        "best_val_loss", "best_epoch", "total_time_s",
    ]

    with open(results_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for i, cfg in enumerate(configs):
            run_tag = f"run{i+1:02d}_s1lr{cfg['s1_lr']:.0e}_s2lr{cfg['s2_lr']:.0e}_ub{cfg['unfreeze']}"
            run_dir = os.path.join(output_dir, run_tag)
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n{'#'*60}")
            print(f"  HP Search Run {i+1}/{len(configs)}: {run_tag}")
            print(f"{'#'*60}")

            # Fresh data loaders per run (stateless)
            train_loader, val_loader, _ = build_dataloaders(
                data_root, batch_size=batch_size, num_workers=num_workers
            )

            # Fresh model instance per run
            model = build_model(backbone_name=backbone).to(device)
            class_weights = compute_class_weights(train_loader, NUM_CLASSES, device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            t_start = time.time()
            result = run_staged_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                criterion=criterion,
                output_dir=run_dir,
                stage1_lr=cfg["s1_lr"],
                stage1_epochs=8,             # Shorter for HP search
                stage1_patience=4,
                stage2_lr=cfg["s2_lr"],
                stage2_epochs=12,
                stage2_patience=5,
                stage2_unfreeze_blocks=int(cfg["unfreeze"]),
                weight_decay=cfg["wd"],
                run_tag=run_tag,
            )
            total_time = time.time() - t_start

            plot_training_curves(result["history"], run_dir, run_tag=run_tag)

            row = {
                "run_tag": run_tag,
                "stage1_lr": cfg["s1_lr"],
                "stage2_lr": cfg["s2_lr"],
                "unfreeze_blocks": cfg["unfreeze"],
                "weight_decay": cfg["wd"],
                "best_val_loss": f"{result['best_val_loss']:.6f}",
                "best_epoch": result["best_epoch"],
                "total_time_s": f"{total_time:.0f}",
            }
            writer.writerow(row)
            csvf.flush()

            print(f"\n  [HP Run {i+1}] best_val_loss={result['best_val_loss']:.4f} | time={total_time:.0f}s")

    print(f"\n[OK] Hyperparameter results saved to: {results_path}")
    _print_hp_results_table(results_path)


def _print_hp_results_table(csv_path: str) -> None:
    """Prints a formatted summary table of hyperparameter search results."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("\n" + "=" * 90)
    print(" HYPERPARAMETER SEARCH RESULTS SUMMARY")
    print("=" * 90)
    header = f"{'Run':<45} {'S1 LR':>8} {'S2 LR':>8} {'Unfr':>5} {'WD':>8} {'BestLoss':>10} {'Epoch':>6}"
    print(header)
    print("-" * 90)
    for r in sorted(rows, key=lambda x: float(x["best_val_loss"])):
        print(
            f"{r['run_tag']:<45} {float(r['stage1_lr']):>8.0e} {float(r['stage2_lr']):>8.0e} "
            f"{r['unfreeze_blocks']:>5} {float(r['weight_decay']):>8.0e} "
            f"{float(r['best_val_loss']):>10.6f} {r['best_epoch']:>6}"
        )
    print("=" * 90)


# ===========================================================================
# 10. Argument Parser
# ===========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DR Transfer Learning model with staged fine-tuning."
    )
    parser.add_argument(
        "--data-root", type=str,
        default=os.path.join("data", "split"),
        help="Path to data/split directory (default: data/split).",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join("reports", "training"),
        help="Directory to save checkpoints and plots (default: reports/training).",
    )
    parser.add_argument(
        "--backbone", type=str, default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet50"],
        help="CNN backbone to use (default: efficientnet_b0).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size (default: 32).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader worker processes (default: 2; use 0 on Windows if errors occur).",
    )
    parser.add_argument(
        "--stage1-lr", type=float, default=1e-3,
        help="Stage 1 (head warmup) learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--stage1-epochs", type=int, default=10,
        help="Maximum Stage 1 epochs (default: 10).",
    )
    parser.add_argument(
        "--stage2-lr", type=float, default=1e-4,
        help="Stage 2 (fine-tuning) learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--stage2-epochs", type=int, default=20,
        help="Maximum Stage 2 epochs (default: 20).",
    )
    parser.add_argument(
        "--unfreeze-blocks", type=int, default=3,
        help="Number of top backbone blocks to unfreeze in Stage 2 (default: 3).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="L2 regularisation coefficient for AdamW (default: 1e-4).",
    )
    parser.add_argument(
        "--hp-search", action="store_true",
        help="Run full hyperparameter grid search instead of single training run.",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable Automatic Mixed Precision (use if CUDA AMP causes issues).",
    )
    return parser.parse_args()


# ===========================================================================
# 11. Main Entry Point
# ===========================================================================
def main() -> None:
    args = parse_args()

    # Device selection: GPU if available (Colab T4/A100), else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    use_amp = not args.no_amp

    # -------------------------------------------------------------------
    # Hyperparameter Grid Search Mode
    # -------------------------------------------------------------------
    if args.hp_search:
        print("\n[Mode] Running hyperparameter grid search ...")
        run_hyperparameter_search(
            data_root=args.data_root,
            output_dir=args.output_dir,
            device=device,
            backbone=args.backbone,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        return

    # -------------------------------------------------------------------
    # Single Training Run Mode
    # -------------------------------------------------------------------
    print("\n[Mode] Single training run ...")
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_loader, val_loader, class_names = build_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(backbone_name=args.backbone).to(device)

    class_weights = compute_class_weights(train_loader, NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    result = run_staged_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        output_dir=args.output_dir,
        stage1_lr=args.stage1_lr,
        stage1_epochs=args.stage1_epochs,
        stage2_lr=args.stage2_lr,
        stage2_epochs=args.stage2_epochs,
        stage2_unfreeze_blocks=args.unfreeze_blocks,
        weight_decay=args.weight_decay,
        use_amp=use_amp,
        run_tag=run_tag,
    )

    plot_training_curves(result["history"], args.output_dir, run_tag=run_tag)

    print("\n" + "=" * 60)
    print(" TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Val Loss    : {result['best_val_loss']:.4f}")
    print(f"  Best Epoch       : {result['best_epoch']}")
    print(f"  Checkpoint       : {result['checkpoint_path']}")
    print(f"  Stage 1 Duration : {result['stage1_time_s']:.0f}s")
    print(f"  Stage 2 Duration : {result['stage2_time_s']:.0f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
