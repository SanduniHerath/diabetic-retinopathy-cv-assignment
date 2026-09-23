"""
split_dataset.py
================
Phase 1 – Dataset Organization & Train/Val/Test Stratified Split
Diabetic Retinopathy Stage Detection (Computer Vision Assignment)

Purpose
-------
Performs a stratified train/validation/test split (70% / 15% / 15%) on the
organized diabetic retinopathy dataset (data/organized/) preserving each
clinical class's original proportions.

Outputs images into:
    data/split/train/<class>/
    data/split/val/<class>/
    data/split/test/<class>/

Clinical & Methodological Justification
---------------------------------------
1. Stratified Partitioning:
   Diabetic retinopathy datasets exhibit severe class imbalance (e.g. No_DR
   accounts for ~49.3% while Severe accounts for only ~5.3%). A purely random
   split risks sampling bias where minority stages are underrepresented or
   entirely missing in validation or test subsets. Stratified splitting enforces
   identical class distributions across all partitions.

2. Partition Ratios (70 / 15 / 15):
   - Train (70%): Maximizes sample diversity for CNN feature extraction and
     subsequent targeted data augmentation.
   - Validation (15%): Provides an unbiased sample for monitoring loss, early
     stopping, learning rate scheduling, and hyperparameter tuning.
   - Test (15%): Strictly held-out, pristine evaluation set simulating clinical
     deployment on unseen patient records.

3. Zero Data Leakage:
   Exact set-intersection assertions guarantee that no image or patient id
   appears in more than one partition (train ∩ val == ∅, train ∩ test == ∅,
   val ∩ test == ∅).

4. Reproducibility:
   Alphabetical sorting prior to seeding ensures 100% deterministic results
   across different filesystems, operating systems, and Python runs.

Usage
-----
    python src/split_dataset.py
    python src/split_dataset.py --input-dir data/organized --output-dir data/split --seed 42
    python src/split_dataset.py --dry-run
"""

import argparse
import csv
import os
import random
import shutil
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless/server use
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = os.path.join("data", "organized")
DEFAULT_OUTPUT_DIR = os.path.join("data", "split")
DEFAULT_REPORT_DIR = os.path.join("reports", "dataset_overview")

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42

CLASSES: Dict[int, str] = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative_DR",
}

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# CORE SPLITTING LOGIC
# ---------------------------------------------------------------------------

def compute_stratified_split(
    files: List[str],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Deterministically partition a list of file names into train, val, and test.

    Parameters
    ----------
    files : List[str]
        List of filenames to split.
    train_ratio : float
        Fraction for training set (e.g. 0.70).
    val_ratio : float
        Fraction for validation set (e.g. 0.15).
    test_ratio : float
        Fraction for test set (e.g. 0.15).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        (train_files, val_files, test_files)
    """
    total = len(files)
    if total == 0:
        return [], [], []

    # Sort files first so directory order across different OS/filesystems does not alter shuffle
    sorted_files = sorted(files)

    # Deterministic shuffle using local isolated Random instance
    rng = random.Random(seed)
    shuffled_files = sorted_files.copy()
    rng.shuffle(shuffled_files)

    # Compute exact count targets with rounding
    n_train = int(round(total * train_ratio))
    n_val = int(round(total * val_ratio))
    n_test = total - n_train - n_val

    # Adjust if slight rounding discrepancy occurs
    if n_test < 0:
        n_test = 0
        n_val = total - n_train

    train_files = shuffled_files[:n_train]
    val_files = shuffled_files[n_train : n_train + n_val]
    test_files = shuffled_files[n_train + n_val :]

    # Sanity checks
    assert len(train_files) + len(val_files) + len(test_files) == total, (
        f"Partition count mismatch: {len(train_files)} + {len(val_files)} + {len(test_files)} != {total}"
    )
    assert len(set(train_files) & set(val_files)) == 0, "Data leakage detected: train ∩ val != ∅"
    assert len(set(train_files) & set(test_files)) == 0, "Data leakage detected: train ∩ test != ∅"
    assert len(set(val_files) & set(test_files)) == 0, "Data leakage detected: val ∩ test != ∅"

    return train_files, val_files, test_files


def scan_organized_dataset(input_dir: str) -> Dict[str, List[str]]:
    """
    Scan the organized directory and return a dictionary of class_name -> list of image filenames.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Input directory '{input_dir}' not found. Please run 'python src/organize_dataset.py' first."
        )

    class_files: Dict[str, List[str]] = {}
    for class_id, class_name in CLASSES.items():
        class_folder = os.path.join(input_dir, class_name)
        if not os.path.exists(class_folder):
            print(f"[WARNING] Class folder '{class_folder}' does not exist. Assuming 0 files.")
            class_files[class_name] = []
            continue

        images = [
            f for f in os.listdir(class_folder)
            if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTENSIONS
        ]
        class_files[class_name] = images

    return class_files


def copy_split_files(
    input_dir: str,
    output_dir: str,
    split_records: Dict[str, Dict[str, List[str]]],
    dry_run: bool = False,
) -> None:
    """
    Copy images from organized class folders into train/val/test split directories.
    """
    splits = ["train", "val", "test"]

    for split in splits:
        for class_name in CLASSES.values():
            target_folder = os.path.join(output_dir, split, class_name)
            if not dry_run:
                os.makedirs(target_folder, exist_ok=True)

    total_copied = 0
    total_to_copy = sum(
        len(files)
        for class_data in split_records.values()
        for files in class_data.values()
    )

    print(f"\n[INFO] {'[DRY RUN] Would copy' if dry_run else 'Copying'} {total_to_copy} images into '{output_dir}'...")

    for class_name, splits_dict in split_records.items():
        src_class_dir = os.path.join(input_dir, class_name)
        for split_name, filenames in splits_dict.items():
            dst_class_dir = os.path.join(output_dir, split_name, class_name)
            for fname in filenames:
                src_path = os.path.join(src_class_dir, fname)
                dst_path = os.path.join(dst_class_dir, fname)

                if not dry_run:
                    # shutil.copy2 preserves file timestamps and metadata
                    shutil.copy2(src_path, dst_path)

                total_copied += 1
                if total_copied % 500 == 0 or total_copied == total_to_copy:
                    print(f"  Processed {total_copied}/{total_to_copy} images ({total_copied / total_to_copy * 100:.1f}%) ...")

    print(f"[SUCCESS] All {total_copied} images successfully partitioned into train/val/test.")


def save_split_manifest(
    output_csv_path: str,
    split_records: Dict[str, Dict[str, List[str]]],
) -> None:
    """
    Save an exhaustive CSV manifest mapping every image to its class, label id, and split.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    name_to_id = {name: cid for cid, name in CLASSES.items()}

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_code", "filename", "class_name", "diagnosis", "split"])

        for class_name, splits_dict in split_records.items():
            cid = name_to_id[class_name]
            for split_name, filenames in splits_dict.items():
                for fname in sorted(filenames):
                    id_code = os.path.splitext(fname)[0]
                    writer.writerow([id_code, fname, class_name, cid, split_name])

    print(f"[INFO] Split manifest saved to: {output_csv_path}")


def generate_split_plot(
    output_plot_path: str,
    split_stats: List[dict],
) -> None:
    """
    Generate and save a publication-quality stacked/grouped bar chart illustrating
    the stratified proportions across Train, Val, and Test splits.
    """
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

    labels = [s["class_name"] for s in split_stats]
    train_counts = [s["train"] for s in split_stats]
    val_counts = [s["val"] for s in split_stats]
    test_counts = [s["test"] for s in split_stats]

    x = np.arange(len(labels))
    width = 0.25

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Color palette tailored for clinical reporting
    color_train = "#2b5c8f"  # Deep clinical blue
    color_val = "#e67e22"    # Amber / warm orange
    color_test = "#27ae60"   # Forest / clinical green

    rects1 = ax.bar(x - width, train_counts, width, label="Train (70%)", color=color_train, edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x, val_counts, width, label="Validation (15%)", color=color_val, edgecolor="black", alpha=0.9)
    rects3 = ax.bar(x + width, test_counts, width, label="Test (15%)", color=color_test, edgecolor="black", alpha=0.9)

    ax.set_ylabel("Image Count", fontsize=12, fontweight="bold")
    ax.set_title("APTOS 2019 Stratified Train/Val/Test Split Distribution\n(70% Train, 15% Val, 15% Test per Class)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")

    # Add count labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Stratified distribution chart saved to: {output_plot_path}")


# ---------------------------------------------------------------------------
# CLI & WORKFLOW ORCHESTRATION
# ---------------------------------------------------------------------------

def print_summary_table(split_stats: List[dict]) -> None:
    """
    Format and print a markdown-compatible and console-friendly ASCII summary table.
    """
    sep = "=" * 85
    sub_sep = "-" * 85
    print("\n" + sep)
    print("           APTOS 2019 DATASET STRATIFIED TRAIN / VAL / TEST SPLIT SUMMARY")
    print(sep)
    header = (
        f"{'Label':<6} | {'Class Name':<17} | {'Total':<7} | "
        f"{'Train (70%)':<14} | {'Val (15%)':<14} | {'Test (15%)':<14}"
    )
    print(header)
    print(sub_sep)

    tot_all = sum(s["total"] for s in split_stats)
    tot_train = sum(s["train"] for s in split_stats)
    tot_val = sum(s["val"] for s in split_stats)
    tot_test = sum(s["test"] for s in split_stats)

    for s in split_stats:
        t_pct = (s["train"] / s["total"] * 100) if s["total"] > 0 else 0
        v_pct = (s["val"] / s["total"] * 100) if s["total"] > 0 else 0
        te_pct = (s["test"] / s["total"] * 100) if s["total"] > 0 else 0

        train_str = f"{s['train']:>5} ({t_pct:>5.1f}%)"
        val_str = f"{s['val']:>5} ({v_pct:>5.1f}%)"
        test_str = f"{s['test']:>5} ({te_pct:>5.1f}%)"

        print(
            f"{s['label']:<6} | {s['class_name']:<17} | {s['total']:<7} | "
            f"{train_str:<14} | {val_str:<14} | {test_str:<14}"
        )

    print(sub_sep)
    all_t_pct = (tot_train / tot_all * 100) if tot_all > 0 else 0
    all_v_pct = (tot_val / tot_all * 100) if tot_all > 0 else 0
    all_te_pct = (tot_test / tot_all * 100) if tot_all > 0 else 0

    tot_train_str = f"{tot_train:>5} ({all_t_pct:>5.1f}%)"
    tot_val_str = f"{tot_val:>5} ({all_v_pct:>5.1f}%)"
    tot_test_str = f"{tot_test:>5} ({all_te_pct:>5.1f}%)"

    print(
        f"{'--':<6} | {'Total':<17} | {tot_all:<7} | "
        f"{tot_train_str:<14} | {tot_val_str:<14} | {tot_test_str:<14}"
    )
    print(sep + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stratified Train/Val/Test Split for Diabetic Retinopathy Dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Path to organized class directory (default: data/organized)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Path to output split directory (default: data/split)",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=DEFAULT_REPORT_DIR,
        help="Path to output report assets (default: reports/dataset_overview)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Proportion of images for training set (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Proportion of images for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Proportion of images for test set (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic splitting (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the split and display statistics without copying files",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the matplotlib distribution plot",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate ratios
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.999 <= ratio_sum <= 1.001):
        print(f"[ERROR] Split ratios must sum to 1.0 (got {ratio_sum})", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("PHASE 1: STRATIFIED TRAIN / VAL / TEST DATASET SPLIT")
    print("=" * 80)
    print(f"  Input Directory   : {args.input_dir}")
    print(f"  Output Directory  : {args.output_dir}")
    print(f"  Ratios (T/V/Te)   : {args.train_ratio:.2f} / {args.val_ratio:.2f} / {args.test_ratio:.2f}")
    print(f"  Random Seed       : {args.seed}")
    print(f"  Dry Run           : {args.dry_run}")
    print("-" * 80)

    # 1. Scan organized images
    class_files = scan_organized_dataset(args.input_dir)
    total_images = sum(len(f) for f in class_files.values())
    if total_images == 0:
        print(f"[ERROR] No valid images found in '{args.input_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Discovered {total_images} total images across {len(class_files)} classes.")

    # 2. Partition per class
    split_records: Dict[str, Dict[str, List[str]]] = {}
    split_stats = []

    for class_id, class_name in CLASSES.items():
        files = class_files.get(class_name, [])
        train_f, val_f, test_f = compute_stratified_split(
            files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        split_records[class_name] = {
            "train": train_f,
            "val": val_f,
            "test": test_f,
        }

        split_stats.append({
            "label": class_id,
            "class_name": class_name,
            "total": len(files),
            "train": len(train_f),
            "val": len(val_f),
            "test": len(test_f),
        })

    # 3. Print report table
    print_summary_table(split_stats)

    # 4. Leakage Verification Check
    print("[VERIFICATION] Running zero-leakage assertions across all classes...")
    all_train = set()
    all_val = set()
    all_test = set()

    for class_name, splits in split_records.items():
        all_train.update(splits["train"])
        all_val.update(splits["val"])
        all_test.update(splits["test"])

    overlap_tv = all_train & all_val
    overlap_tt = all_train & all_test
    overlap_vt = all_val & all_test

    assert len(overlap_tv) == 0, f"LEAKAGE DETECTED: {len(overlap_tv)} files in Train and Val!"
    assert len(overlap_tt) == 0, f"LEAKAGE DETECTED: {len(overlap_tt)} files in Train and Test!"
    assert len(overlap_vt) == 0, f"LEAKAGE DETECTED: {len(overlap_vt)} files in Val and Test!"
    assert len(all_train) + len(all_val) + len(all_test) == total_images, "Total image count mismatch!"

    print("  [OK] Train & Val intersection  = empty (0 leaking samples)")
    print("  [OK] Train & Test intersection = empty (0 leaking samples)")
    print("  [OK] Val & Test intersection   = empty (0 leaking samples)")
    print(f"  [OK] Total partition sum = {len(all_train) + len(all_val) + len(all_test)} / {total_images} (100% accounted for)")
    print("[VERIFICATION] All integrity checks passed successfully.")

    # 5. Copy files if not dry-run
    if not args.dry_run:
        copy_split_files(args.input_dir, args.output_dir, split_records, dry_run=False)

        # Save manifest CSV
        manifest_path = os.path.join(args.report_dir, "split_manifest.csv")
        save_split_manifest(manifest_path, split_records)

        # Generate plot
        if not args.no_plot:
            plot_path = os.path.join(args.report_dir, "split_distribution.png")
            generate_split_plot(plot_path, split_stats)

    print("\n[COMPLETE] Phase 1 stratified dataset split finished successfully.\n")


if __name__ == "__main__":
    main()
