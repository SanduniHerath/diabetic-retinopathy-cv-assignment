"""
evaluate.py
===========
Phase 5 - Model Evaluation & Performance Analysis
Diabetic Retinopathy Stage Detection (Computer Vision Assignment)

Generates a comprehensive suite of evaluation artefacts from the best checkpoint.

Preprocessing Note:
  Test images are processed through the full Phase 2 preprocessing pipeline
  (circular crop, CLAHE, Gaussian denoise, unsharp mask) via get_val_transforms()
  imported from train.py.  NO augmentation is applied at test time -- only the
  deterministic quality-enhancement steps.  This ensures test-time distribution
  matches training-time distribution exactly.

Generates a comprehensive suite of evaluation artefacts:

  1. Per-class metrics table: Accuracy, Precision, Recall, F1-Score, Support
  2. Weighted & macro-average metrics
  3. Confusion matrix (normalised + raw counts) saved as PNG
  4. Quadratic Weighted Kappa (QWK) – the standard DR competition metric
  5. Loss / accuracy curves replot (if training history JSON is available)
  6. Grad-CAM visualisations for representative test samples per class
  7. Error analysis: identifies dominant misclassification pairs and discusses
     clinical risk (e.g., Severe mis-labelled as Moderate = under-treatment risk)

Outputs (saved to reports/evaluation/):
  - classification_report.txt        – full per-class metric table
  - confusion_matrix_norm.png        – normalised confusion matrix heatmap
  - confusion_matrix_raw.png         – raw-count confusion matrix heatmap
  - metrics_summary.json             – machine-readable metric dictionary
  - gradcam_samples/                 – Grad-CAM overlays for test samples
  - error_analysis.txt               – top misclassification patterns + discussion

Usage (Google Colab / local):
  python src/evaluate.py \\
    --checkpoint reports/training/best_model.pth \\
    --data-root   data/split \\
    --output-dir  reports/evaluation

Author: Sanduni Herath
Repository: https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

# ---------------------------------------------------------------------------
# Portable sys.path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parent
for _p in (_FILE_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model import build_model, CLASS_NAMES, NUM_CLASSES  # noqa: E402
from gradcam import GradCAM, generate_gradcam_heatmap     # noqa: E402
from train import get_val_transforms                       # noqa: E402


# ===========================================================================
# 1. Metric helpers
# ===========================================================================

def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """Returns a (num_classes x num_classes) confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_per_class_metrics(
    cm: np.ndarray,
) -> Dict[str, Any]:
    """
    Derives precision, recall, F1 and accuracy per class from a confusion matrix.
    Also computes macro and weighted averages.
    """
    n = cm.shape[0]
    total_samples = cm.sum()
    per_class: List[Dict[str, Any]] = []

    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total_samples - tp - fp - fn

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        support   = int(cm[i, :].sum())
        accuracy  = (tp + tn) / (total_samples + 1e-9)

        per_class.append({
            "class": CLASS_NAMES[i],
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "support":   support,
            "accuracy":  float(accuracy),
        })

    # Weighted averages (by support)
    supports = np.array([c["support"] for c in per_class], dtype=np.float64)
    weighted_precision = float(np.average([c["precision"] for c in per_class], weights=supports))
    weighted_recall    = float(np.average([c["recall"]    for c in per_class], weights=supports))
    weighted_f1        = float(np.average([c["f1"]        for c in per_class], weights=supports))
    macro_f1           = float(np.mean([c["f1"] for c in per_class]))
    overall_accuracy   = float(np.diag(cm).sum() / total_samples)

    return {
        "per_class":            per_class,
        "weighted_precision":   weighted_precision,
        "weighted_recall":      weighted_recall,
        "weighted_f1":          weighted_f1,
        "macro_f1":             macro_f1,
        "overall_accuracy":     overall_accuracy,
    }


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> float:
    """
    Quadratic Weighted Kappa (QWK) as used in APTOS / EyePACS competitions.
    QWK rewards near-miss predictions (e.g., Grade 1 vs Grade 2) more than
    categorical accuracy, making it the gold-standard for ordinal DR grading.

    Score interpretation:
      < 0.20: Slight agreement
      0.21 – 0.40: Fair
      0.41 – 0.60: Moderate
      0.61 – 0.80: Substantial
      0.81 – 1.00: Almost perfect
    """
    # Weight matrix: w[i,j] = ((i-j)/(n-1))^2
    w = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            w[i, j] = ((i - j) / (n - 1)) ** 2

    hist_true = np.bincount(y_true, minlength=n).astype(np.float64)
    hist_pred = np.bincount(y_pred, minlength=n).astype(np.float64)
    E = np.outer(hist_true, hist_pred) / len(y_true)

    O = compute_confusion_matrix(y_true, y_pred, n).astype(np.float64)
    O /= O.sum()

    num = (w * O).sum()
    den = (w * E).sum()
    kappa = 1.0 - num / (den + 1e-9)
    return float(kappa)


# ===========================================================================
# 2. Confusion Matrix Plotting
# ===========================================================================
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: str,
    normalised: bool = True,
) -> None:
    """Saves a heatmap of the confusion matrix as a PNG file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("[Plot] matplotlib not available - skipping confusion matrix plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    title  = "Normalised Confusion Matrix" if normalised else "Confusion Matrix (Raw Counts)"
    fname  = "confusion_matrix_norm.png"   if normalised else "confusion_matrix_raw.png"

    if normalised:
        cm_plot = cm.astype(np.float32)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm_plot / row_sums, 0.0)
        fmt = ".2f"
        vmax = 1.0
        cmap = "Blues"
    else:
        cm_plot = cm
        fmt = "d"
        vmax = int(cm.max())
        cmap = "Oranges"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate cells
    thresh = cm_plot.max() / 2.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j]
            text = f"{val:{fmt}}" if fmt == ".2f" else str(int(val))
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=10)

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_path = os.path.join(output_dir, fname)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Confusion matrix saved to: {save_path}")


# ===========================================================================
# 3. Error Analysis
# ===========================================================================
def generate_error_analysis(
    cm: np.ndarray,
    metrics: Dict[str, Any],
    output_dir: str,
    qwk: float,
) -> str:
    """
    Identifies dominant misclassification pairs and produces a structured
    clinical risk discussion saved as error_analysis.txt.
    """
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(" ERROR ANALYSIS – MISCLASSIFICATION PATTERNS")
    lines.append(" Diabetic Retinopathy Stage Detection")
    lines.append("=" * 70)

    lines.append("\n[1] OVERALL PERFORMANCE SUMMARY:")
    lines.append(f"  Overall Accuracy   : {metrics['overall_accuracy']*100:.2f}%")
    lines.append(f"  Weighted F1-Score  : {metrics['weighted_f1']:.4f}")
    lines.append(f"  Macro F1-Score     : {metrics['macro_f1']:.4f}")
    lines.append(f"  Quadratic Weighted Kappa (QWK): {qwk:.4f}")

    lines.append("\n[2] PER-CLASS METRICS:")
    header = f"  {'Class':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}"
    lines.append(header)
    lines.append("  " + "-" * 57)
    for c in metrics["per_class"]:
        lines.append(
            f"  {c['class']:<20} {c['precision']:>10.4f} {c['recall']:>8.4f} "
            f"{c['f1']:>8.4f} {c['support']:>9d}"
        )

    lines.append("\n[3] TOP MISCLASSIFICATION PAIRS (True -> Predicted):")
    n = cm.shape[0]
    errors = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                errors.append((cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
    errors.sort(reverse=True)

    for count, true_c, pred_c in errors[:10]:
        row_total = cm[CLASS_NAMES.index(true_c), :].sum()
        pct = 100.0 * count / max(row_total, 1)
        lines.append(f"  {true_c:<22} -> {pred_c:<22}  {count:>5} samples ({pct:>5.1f}%)")

    lines.append("\n[4] CLINICAL RISK DISCUSSION:")
    lines.append("""
  The confusion matrix reveals the model's safety profile for clinical deployment:

  HIGH-RISK misclassifications (severe under-referral risk):
    - Severe (Grade 3) predicted as Moderate (Grade 2):
        Severe DR requires urgent ophthalmology referral. Downgrading to
        Moderate delays treatment and risks progression to blindness. Any
        non-trivial confusion rate here is clinically significant.
    - Proliferative_DR (Grade 4) predicted as Severe (Grade 3):
        Proliferative DR demands immediate intervention (laser/anti-VEGF).
        Grade 4 missed as Grade 3 defers life-saving treatment.

  LOWER-RISK misclassifications (adjacent grade confusion):
    - No_DR (Grade 0) predicted as Mild (Grade 1):
        Mild over-diagnosis increases referral burden but carries low harm;
        patients undergo unnecessary monitoring, not withheld treatment.
    - Mild (Grade 1) predicted as Moderate (Grade 2):
        Leads to slightly earlier referral; generally safe in terms of
        patient outcomes since both grades require follow-up.

  ORDINAL STRUCTURE & QWK:
    The Quadratic Weighted Kappa penalises distant-grade errors more than
    adjacent-grade errors. A QWK >= 0.80 is considered clinically acceptable
    for automated DR screening tools (APTOS 2019 benchmark). The model's QWK
    above provides context on whether predictions remain clinically ordered
    even when not perfectly correct.

  MODEL LIMITATION NOTE:
    This model was trained on a public Kaggle dataset (APTOS-style fundus
    images). Real-world deployment would require prospective validation on
    local population data and regulatory approval before clinical use.
    The Grad-CAM visualisations (see reports/evaluation/gradcam_samples/)
    support interpretability by highlighting retinal regions driving each
    prediction, enabling ophthalmologist review of model attention.
    """)

    lines.append("=" * 70)
    report = "\n".join(lines)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "error_analysis.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Error analysis saved to: {out_path}")
    return report


# ===========================================================================
# 4. Grad-CAM Visualisation for Test Samples
# ===========================================================================
def generate_gradcam_samples(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    samples_per_class: int = 3,
) -> None:
    """
    Generates Grad-CAM overlays for representative test samples.
    Saves one panel image per class showing input image + heatmap overlay.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cv2
    except ImportError:
        print("[GradCAM] matplotlib or cv2 not available - skipping Grad-CAM visualisation.")
        return

    gradcam_dir = os.path.join(output_dir, "gradcam_samples")
    os.makedirs(gradcam_dir, exist_ok=True)

    model.eval()
    # Collect samples per class
    class_samples: Dict[int, List[Tuple[torch.Tensor, int]]] = {i: [] for i in range(NUM_CLASSES)}

    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            cls = int(lbl.item())
            if len(class_samples[cls]) < samples_per_class:
                class_samples[cls].append((img, cls))
        if all(len(v) >= samples_per_class for v in class_samples.values()):
            break

    MEAN_arr = np.array([0.485, 0.456, 0.406])
    STD_arr  = np.array([0.229, 0.224, 0.225])

    for cls_idx, samples in class_samples.items():
        if not samples:
            continue

        n_samples = len(samples)
        fig, axes = plt.subplots(n_samples, 2, figsize=(8, 3.5 * n_samples))
        if n_samples == 1:
            axes = [axes]

        for row_idx, (img_tensor, true_label) in enumerate(samples):
            inp = img_tensor.unsqueeze(0).to(device)

            # Denormalise for display
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = img_np * STD_arr + MEAN_arr
            img_np = np.clip(img_np, 0, 1)

            # Grad-CAM heatmap
            try:
                heatmap = generate_gradcam_heatmap(model, inp, target_class=cls_idx)
                heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
                heatmap_colour  = cv2.applyColorMap(
                    (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB) / 255.0
                overlay = 0.55 * img_np + 0.45 * heatmap_colour
                overlay = np.clip(overlay, 0, 1)
            except Exception as e:
                print(f"    [GradCAM] Warning for class {CLASS_NAMES[cls_idx]}: {e}")
                overlay = img_np

            ax_img  = axes[row_idx][0]
            ax_cam  = axes[row_idx][1]
            ax_img.imshow(img_np)
            ax_img.set_title(f"Input – True: {CLASS_NAMES[true_label]}", fontsize=9)
            ax_img.axis("off")
            ax_cam.imshow(overlay)
            ax_cam.set_title("Grad-CAM Overlay", fontsize=9)
            ax_cam.axis("off")

        fig.suptitle(
            f"Grad-CAM: {CLASS_NAMES[cls_idx]} (Grade {cls_idx})",
            fontsize=12, fontweight="bold"
        )
        fig.tight_layout()
        save_path = os.path.join(gradcam_dir, f"gradcam_{CLASS_NAMES[cls_idx]}.png")
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [GradCAM] Saved: {save_path}")


# ===========================================================================
# 5. Full Evaluation Pipeline
# ===========================================================================
def run_evaluation(
    checkpoint_path: str,
    data_root: str,
    output_dir: str,
    backbone: str = "efficientnet_b0",
    batch_size: int = 32,
    num_workers: int = 2,
    generate_gradcam: bool = True,
    gradcam_samples: int = 3,
) -> Dict[str, Any]:
    """
    Loads the best checkpoint, runs inference on the test split, and generates
    all evaluation artefacts required by Criteria 5 and 6 of the rubric.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] Using: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load test dataset
    # -----------------------------------------------------------------------
    test_dir = Path(data_root) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # get_val_transforms() includes the full Phase 2 preprocessing pipeline
    # (PreprocessingTransform: crop->CLAHE->denoise->sharpen) before
    # centre-crop and ImageNet normalisation.  No augmentation at test time.
    test_dataset = datasets.ImageFolder(str(test_dir), transform=get_val_transforms())
    test_loader  = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[Data] Test: {len(test_dataset):,} images across {len(test_dataset.classes)} classes")

    # -----------------------------------------------------------------------
    # Load model checkpoint
    # -----------------------------------------------------------------------
    model = build_model(backbone_name=backbone)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[OK] Checkpoint loaded: {checkpoint_path}")

    # -----------------------------------------------------------------------
    # Inference pass: collect all predictions and ground-truth labels
    # -----------------------------------------------------------------------
    all_preds:  List[int] = []
    all_labels: List[int] = []
    all_probs:  List[np.ndarray] = []
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            lbls  = labels.cpu().numpy()

            total_loss    += loss.item() * len(lbls)
            total_correct += int((preds == lbls).sum())
            total_samples += len(lbls)

            all_preds.extend(preds.tolist())
            all_labels.extend(lbls.tolist())
            all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    test_loss = total_loss / max(total_samples, 1)
    test_acc  = 100.0 * total_correct / max(total_samples, 1)

    print(f"\n[Eval] Test Loss     : {test_loss:.4f}")
    print(f"[Eval] Test Accuracy : {test_acc:.2f}%")

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    cm      = compute_confusion_matrix(y_true, y_pred, NUM_CLASSES)
    metrics = compute_per_class_metrics(cm)
    qwk     = quadratic_weighted_kappa(y_true, y_pred, NUM_CLASSES)

    metrics["test_loss"]     = float(test_loss)
    metrics["test_accuracy"] = float(test_acc)
    metrics["qwk"]           = float(qwk)

    # -----------------------------------------------------------------------
    # Print classification report to console
    # -----------------------------------------------------------------------
    report_lines: List[str] = []
    report_lines.append("\n" + "=" * 65)
    report_lines.append(" CLASSIFICATION REPORT")
    report_lines.append("=" * 65)
    report_lines.append(f"  Test Loss           : {test_loss:.4f}")
    report_lines.append(f"  Test Accuracy       : {test_acc:.2f}%")
    report_lines.append(f"  Weighted F1         : {metrics['weighted_f1']:.4f}")
    report_lines.append(f"  Macro F1            : {metrics['macro_f1']:.4f}")
    report_lines.append(f"  Quadratic Wtd Kappa : {qwk:.4f}")
    report_lines.append("\n  Per-class breakdown:")
    report_lines.append(f"  {'Class':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    report_lines.append("  " + "-" * 59)
    for c in metrics["per_class"]:
        report_lines.append(
            f"  {c['class']:<22} {c['precision']:>10.4f} {c['recall']:>8.4f} "
            f"{c['f1']:>8.4f} {c['support']:>9d}"
        )
    report_lines.append("=" * 65)
    report_str = "\n".join(report_lines)
    print(report_str)

    # Save report text
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[OK] Classification report saved to: {report_path}")

    # Save metrics JSON
    json_path = os.path.join(output_dir, "metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[OK] Metrics JSON saved to: {json_path}")

    # -----------------------------------------------------------------------
    # Confusion matrices
    # -----------------------------------------------------------------------
    plot_confusion_matrix(cm, CLASS_NAMES, output_dir, normalised=True)
    plot_confusion_matrix(cm, CLASS_NAMES, output_dir, normalised=False)

    # -----------------------------------------------------------------------
    # Error analysis
    # -----------------------------------------------------------------------
    error_report = generate_error_analysis(cm, metrics, output_dir, qwk)
    print(error_report)

    # -----------------------------------------------------------------------
    # Grad-CAM visualisations
    # -----------------------------------------------------------------------
    if generate_gradcam:
        print("\n[GradCAM] Generating Grad-CAM visualisations for test samples ...")
        try:
            generate_gradcam_samples(model, test_loader, device, output_dir, gradcam_samples)
        except Exception as e:
            print(f"[GradCAM] Visualisation failed: {e} (continuing without Grad-CAM)")

    return metrics


# ===========================================================================
# 6. Argument Parser
# ===========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DR Transfer Learning model on the test split."
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join("reports", "training", "best_model.pth"),
        help="Path to the saved model checkpoint .pth file.",
    )
    parser.add_argument(
        "--data-root", type=str,
        default=os.path.join("data", "split"),
        help="Path to data/split directory containing test/ subdirectory.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join("reports", "evaluation"),
        help="Directory to write evaluation outputs (default: reports/evaluation).",
    )
    parser.add_argument(
        "--backbone", type=str, default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet50"],
        help="Backbone architecture matching the checkpoint (default: efficientnet_b0).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size (default: 32).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader worker count (default: 2).",
    )
    parser.add_argument(
        "--no-gradcam", action="store_true",
        help="Skip Grad-CAM visualisation generation.",
    )
    parser.add_argument(
        "--gradcam-samples", type=int, default=3,
        help="Number of Grad-CAM sample images per class (default: 3).",
    )
    return parser.parse_args()


# ===========================================================================
# 7. Main Entry Point
# ===========================================================================
def main() -> None:
    args = parse_args()
    metrics = run_evaluation(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generate_gradcam=not args.no_gradcam,
        gradcam_samples=args.gradcam_samples,
    )
    print(f"\n[Done] All evaluation outputs saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
