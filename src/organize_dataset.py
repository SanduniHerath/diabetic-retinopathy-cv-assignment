"""
organize_dataset.py
====================
Phase 1 – Dataset Organization & Exploration
Diabetic Retinopathy Stage Detection (Computer Vision Assignment)

Purpose
-------
Reads train.csv (which maps image id_code -> diagnosis label 0–4) and
organises the raw .png retinal fundus images into a clean, named folder
structure under data/organized/ using **symbolic links** on Unix or hard
**copies** on Windows.  This makes every subsequent step (DataLoaders,
PyTorch ImageFolder, etc.) trivially easy because the directory layout
itself encodes the class label.

Organising logic
----------------
The Kaggle APTOS 2019 dataset uses integer labels:
    0 -> No_DR            (no diabetic retinopathy)
    1 -> Mild             (mild non-proliferative DR)
    2 -> Moderate         (moderate non-proliferative DR)
    3 -> Severe           (severe non-proliferative DR)
    4 -> Proliferative_DR (proliferative DR – most advanced stage)

Each image file is named  <id_code>.png  and lives in
data/raw/train_images/.  The CSV column 'id_code' provides the stem
(without extension), so we construct the source path as:
    <RAW_DIR>/<id_code>.png

We copy (not move, so the raw data stays intact) each image into:
    data/organized/<CLASS_LABEL>/<id_code>.png

Why copy rather than symlink?
    Symlinks work on Mac/Linux but are awkward on Windows without
    Developer Mode.  Copying keeps the raw data untouched and makes the
    organised folder self-contained for cloud/Colab use.

Usage
-----
    python src/organize_dataset.py
    python src/organize_dataset.py --csv data/raw/train_images/train.csv
                                   --images data/raw/train_images
                                   --out data/organized
                                   --samples 3
                                   --sample-out reports/dataset_overview

"""

import argparse
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")           # non-interactive backend – safe on servers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Mapping from integer label -> human-readable folder name.
# These names align with the clinical grading scale used in the APTOS dataset.
CLASS_MAP = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative_DR",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organise APTOS retinopathy images into class folders."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(PROJECT_ROOT, "data", "raw", "train_images", "train.csv"),
        help="Path to train.csv",
    )
    parser.add_argument(
        "--images",
        default=os.path.join(PROJECT_ROOT, "data", "raw", "train_images"),
        help="Directory containing raw .png images",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(PROJECT_ROOT, "data", "organized"),
        help="Output root for organised class folders",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample images to save per class for the report",
    )
    parser.add_argument(
        "--sample-out",
        default=os.path.join(PROJECT_ROOT, "reports", "dataset_overview"),
        help="Directory to save sample images for the report",
    )
    return parser.parse_args()


def create_class_dirs(out_root: str) -> dict:
    """Create one sub-directory per class.  Returns {label: path} mapping."""
    dirs = {}
    for label, name in CLASS_MAP.items():
        path = os.path.join(out_root, name)
        os.makedirs(path, exist_ok=True)
        dirs[label] = path
    return dirs


def organize_images(df: pd.DataFrame, images_dir: str, class_dirs: dict) -> dict:
    """
    Copy each image from images_dir into the appropriate class folder.

    Returns a dict with per-class copy counts and a list of any missing files.
    """
    counts = {label: 0 for label in CLASS_MAP}
    missing = []

    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        id_code = row["id_code"]
        label = int(row["diagnosis"])
        src = os.path.join(images_dir, f"{id_code}.png")

        if not os.path.exists(src):
            missing.append(src)
            continue

        dst = os.path.join(class_dirs[label], f"{id_code}.png")
        if not os.path.exists(dst):          # skip if already organised
            shutil.copy2(src, dst)           # copy2 preserves metadata
        counts[label] += 1

        # Progress indicator every 500 images
        if idx % 500 == 0:
            print(f"  Organised {idx}/{total} images ...")

    return {"counts": counts, "missing": missing}


def print_distribution(counts: dict, total: int) -> None:
    """Print a nicely formatted class distribution table."""
    print("\n" + "=" * 55)
    print("  CLASS DISTRIBUTION – APTOS 2019 Retinopathy Dataset")
    print("=" * 55)
    print(f"  {'Label':<4}  {'Class Name':<20}  {'Count':>6}  {'Pct':>7}")
    print("-" * 55)
    for label in sorted(counts):
        name = CLASS_MAP[label]
        count = counts[label]
        pct = 100 * count / total
        bar = "#" * int(pct / 2)
        print(f"  {label:<4}  {name:<20}  {count:>6}  {pct:>6.1f}%  {bar}")
    print("=" * 55)
    print(f"  {'TOTAL':<4}  {'':20}  {total:>6}  {'100.0%':>7}")
    print("=" * 55)

    # Imbalance warning
    dominant = max(counts.values())
    minority = min(counts.values())
    ratio = dominant / minority
    print(f"\n  Imbalance ratio (dominant / minority): {ratio:.1f}×")
    if ratio > 5:
        print("  [WARNING]  Severe imbalance detected – augmentation and class weighting")
        print("     will be critical in Phase 3.")


def save_sample_images(df: pd.DataFrame, images_dir: str, sample_out: str,
                       n_samples: int = 3) -> None:
    """
    Save n_samples example images per class to sample_out/ as individual PNGs
    AND as a single montage figure (one row per class).

    The saved originals are for reference; the montage goes in the report.
    """
    os.makedirs(sample_out, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=len(CLASS_MAP),
        ncols=n_samples,
        figsize=(4 * n_samples, 4 * len(CLASS_MAP)),
    )
    fig.suptitle("APTOS 2019 – Sample Retinal Fundus Images per DR Stage",
                 fontsize=14, fontweight="bold", y=1.01)

    for label in sorted(CLASS_MAP):
        class_name = CLASS_MAP[label]
        class_df = df[df["diagnosis"] == label].sample(
            n=min(n_samples, len(df[df["diagnosis"] == label])),
            random_state=42,
        ).reset_index(drop=True)

        for col, (_, row) in enumerate(class_df.iterrows()):
            id_code = row["id_code"]
            src = os.path.join(images_dir, f"{id_code}.png")

            if not os.path.exists(src):
                continue

            # Copy the individual sample into sample_out for quick reference
            sample_copy = os.path.join(sample_out, f"{class_name}_{col+1}_{id_code}.png")
            if not os.path.exists(sample_copy):
                shutil.copy2(src, sample_copy)

            # Add to the montage figure
            ax = axes[label][col]
            img = mpimg.imread(src)
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"Label {label}\n{class_name}", fontsize=10,
                              fontweight="bold", rotation=0, ha="right",
                              va="center", labelpad=60)

    plt.tight_layout()
    montage_path = os.path.join(sample_out, "sample_montage.png")
    plt.savefig(montage_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\n  Sample montage saved -> {montage_path}")


def save_distribution_chart(counts: dict, sample_out: str) -> None:
    """Save a bar chart of class distribution to sample_out/."""
    os.makedirs(sample_out, exist_ok=True)

    labels = [f"{l}: {CLASS_MAP[l]}" for l in sorted(counts)]
    values = [counts[l] for l in sorted(counts)]
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_title("APTOS 2019 – Class Distribution\n(Diabetic Retinopathy Severity)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("DR Stage", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.set_ylim(0, max(values) * 1.15)

    # Annotate bars with counts and percentages
    total = sum(values)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{val}\n({100*val/total:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    chart_path = os.path.join(sample_out, "class_distribution.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Distribution chart saved -> {chart_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Load the CSV --------------------------------------------------------
    print(f"\n[1] Loading CSV from: {args.csv}")
    if not os.path.exists(args.csv):
        sys.exit(f"ERROR: train.csv not found at {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"     Loaded {len(df)} records | Columns: {df.columns.tolist()}")

    total = len(df)

    # 2. Create class directories --------------------------------------------
    print(f"\n[2] Creating class directories under: {args.out}")
    class_dirs = create_class_dirs(args.out)
    for label, path in class_dirs.items():
        print(f"     {CLASS_MAP[label]:20s} -> {path}")

    # 3. Organise (copy) images into class folders ---------------------------
    print(f"\n[3] Organising {total} images into class folders ...")
    result = organize_images(df, args.images, class_dirs)
    counts = result["counts"]
    missing = result["missing"]

    if missing:
        print(f"\n  [WARNING]  {len(missing)} image(s) not found in {args.images}:")
        for m in missing[:10]:
            print(f"       {m}")

    # 4. Print distribution --------------------------------------------------
    print_distribution(counts, total)

    # 5. Save sample images & distribution chart -----------------------------
    print(f"\n[4] Saving {args.samples} sample images per class -> {args.sample_out}")
    save_sample_images(df, args.images, args.sample_out, n_samples=args.samples)
    save_distribution_chart(counts, args.sample_out)

    print("\n[DONE]  Dataset organisation complete.")
    print(f"     Organised images  : {args.out}")
    print(f"     Report samples    : {args.sample_out}")


if __name__ == "__main__":
    main()
