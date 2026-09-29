"""
add_external_data.py
====================
Phase 5 – Supplementary Training Data Ingestion (EyePACS Dataset)
Diabetic Retinopathy Stage Detection (Computer Vision Assignment)

Purpose
-------
Downloads and filters supplementary retinal fundus images from the EyePACS
dataset hosted on Hugging Face (Lelihu/eyepacs) to address class imbalance
and improve model sensitivity for minority severity stages:
    - Label 1: Mild             (mild non-proliferative DR)
    - Label 3: Severe           (severe non-proliferative DR)
    - Label 4: Proliferative_DR (proliferative DR – sight-threatening)

Clinical Rationale
------------------
Clinical DR evaluation on the APTOS test set revealed lower per-class recall
and F1 scores on under-represented clinical stages:
    - Mild: ~55 test samples (early capillary microaneurysms)
    - Severe: ~29 test samples (widespread hemorrhages, venous beading, IRMA)
    - Proliferative_DR: ~45 test samples (neovascularisation, high-risk vitreous bleed)
Adding external EyePACS training samples specifically for these three classes
enriches feature representation without polluting the validation or test splits.

Strict Split Integrity Rule
---------------------------
CRITICAL: Images are ONLY saved into the training split directory:
    data/split/train/[class]/
The validation (data/split/val/) and test (data/split/test/) directories are
NEVER modified under any circumstances, preserving strict zero-leakage evaluation.

Dataset Schema (Hugging Face: Lelihu/eyepacs)
--------------------------------------------
    - image: PIL.Image (RGB retinal fundus photograph)
    - label_code: int64 (0: No_DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative_DR)
    - label: string ("No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR")

Usage
-----
    # Ingest up to 500 supplementary images per target class (recommended for balanced speed & accuracy):
    python src/add_external_data.py --max-per-class 500

    # Ingest up to 250 images per target class:
    python src/add_external_data.py --max-per-class 250

    # Ingest all available EyePACS samples for classes 1, 3, and 4 (unlimited):
    python src/add_external_data.py --max-per-class 0

    # Custom split directory path:
    python src/add_external_data.py --split-dir data/split --max-per-class 500

Author: Sanduni Herath
Repository: https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Canonical clinical class mapping (aligned with model.py and ICDR standards)
CLASS_MAP: Dict[int, str] = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative_DR",
}

# Target classes requiring supplementary training augmentation
DEFAULT_TARGET_LABELS: List[int] = [1, 3, 4]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for EyePACS data ingestion."""
    parser = argparse.ArgumentParser(
        description="Download and filter supplementary EyePACS training data from Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Lelihu/eyepacs",
        help="Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "split"),
        help="Root split directory containing train/, val/, and test/ subdirectories.",
    )
    parser.add_argument(
        "--target-labels",
        type=int,
        nargs="+",
        default=DEFAULT_TARGET_LABELS,
        help="List of integer class labels to ingest (e.g. 1 3 4).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="Maximum number of external images to add per target class (0 = ingest all available).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode and download full Hugging Face dataset cache to disk.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="eyepacs",
        help="Filename prefix for saved external images to prevent namespace collisions.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="File format to save images.",
    )
    return parser.parse_args()


def verify_train_directory(split_dir: str) -> str:
    """
    Verifies and validates the training directory layout.
    Strictly verifies that val and test directories exist and will remain untouched.
    """
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    test_dir = os.path.join(split_dir, "test")

    if not os.path.isdir(train_dir):
        sys.exit(f"ERROR: Training directory not found at: {train_dir}\nPlease ensure Phase 3 split_dataset.py has been run.")

    # Guard checks to ensure validation and test exist and are isolated
    if not os.path.isdir(val_dir) or not os.path.isdir(test_dir):
        print(f"  [WARNING] Validation ({val_dir}) or Test ({test_dir}) directory was not detected.")
    else:
        print(f"  [GUARD] Validation ({val_dir}) and Test ({test_dir}) splits are strictly locked.")

    # Ensure class folders exist inside train/
    for label_idx, class_name in CLASS_MAP.items():
        class_folder = os.path.join(train_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)

    return train_dir


def get_class_counts(target_dir: str) -> Dict[str, int]:
    """Scans and counts existing image files per class folder."""
    valid_exts = {".png", ".jpg", ".jpeg"}
    counts: Dict[str, int] = {}
    for class_name in CLASS_MAP.values():
        c_path = os.path.join(target_dir, class_name)
        if os.path.isdir(c_path):
            n_imgs = len([f for f in os.listdir(c_path) if os.path.splitext(f)[1].lower() in valid_exts])
            counts[class_name] = n_imgs
        else:
            counts[class_name] = 0
    return counts


def print_distribution_table(
    initial_counts: Dict[str, int],
    added_counts: Dict[str, int],
    target_labels: List[int],
) -> None:
    """Prints a structured clinical distribution table showing counts before and after."""
    target_names = {CLASS_MAP[lbl] for lbl in target_labels if lbl in CLASS_MAP}

    print("\n" + "=" * 70)
    print("  TRAINING SET CLASS DISTRIBUTION - BEFORE & AFTER EYEPACS INGESTION")
    print("=" * 70)
    print(f"  {'Label':<5} {'Class Name':<18} {'Before':>8} {'Added':>8} {'Final':>8} {'Status':>16}")
    print("-" * 70)

    total_before = sum(initial_counts.values())
    total_added = sum(added_counts.values())
    total_final = total_before + total_added

    for label_idx in sorted(CLASS_MAP.keys()):
        c_name = CLASS_MAP[label_idx]
        c_before = initial_counts.get(c_name, 0)
        c_added = added_counts.get(c_name, 0)
        c_final = c_before + c_added

        if c_name in target_names:
            status = f"+{c_added} added"
        else:
            status = "untouched"

        print(f"  {label_idx:<5} {c_name:<18} {c_before:>8} {c_added:>8} {c_final:>8} {status:>16}")

    print("=" * 70)
    print(f"  {'TOTAL':<24} {total_before:>8} {total_added:>8} {total_final:>8}")
    print("=" * 70)

    # Imbalance metric
    if min(initial_counts.values()) > 0 and min([initial_counts[c] + added_counts.get(c, 0) for c in CLASS_MAP.values()]) > 0:
        old_ratio = max(initial_counts.values()) / min(initial_counts.values())
        new_counts = {c: initial_counts[c] + added_counts.get(c, 0) for c in CLASS_MAP.values()}
        new_ratio = max(new_counts.values()) / min(new_counts.values())
        print(f"\n  [Impact] Imbalance ratio: {old_ratio:.2f}x -> {new_ratio:.2f}x (dominant / minority)")


# ---------------------------------------------------------------------------
# Core Ingestion Engine
# ---------------------------------------------------------------------------

def add_eyepacs_samples(
    dataset_name: str,
    train_dir: str,
    target_labels: List[int],
    max_per_class: int = 500,
    streaming: bool = True,
    prefix: str = "eyepacs",
    image_format: str = "png",
) -> Dict[str, int]:
    """
    Connects to Hugging Face, streams/downloads EyePACS data, filters for target labels,
    and writes RGB fundus images into data/split/train/[class]/ folders.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "\nERROR: The 'datasets' library is required to stream/download EyePACS from Hugging Face.\n"
            "Please install it using:\n"
            "    pip install datasets\n"
        )

    target_label_set = set(target_labels)
    target_classes = {lbl: CLASS_MAP[lbl] for lbl in target_labels if lbl in CLASS_MAP}
    unlimited = (max_per_class <= 0)

    limit_str = "unlimited" if unlimited else f"{max_per_class} per class"
    print(f"\n[Engine] Initializing Hugging Face dataset: {dataset_name}")
    print(f"         Target Labels : {target_labels} -> {list(target_classes.values())}")
    print(f"         Quota Limit   : {limit_str}")
    print(f"         Streaming Mode: {streaming}")

    added_counts: Dict[str, int] = {c_name: 0 for c_name in CLASS_MAP.values()}

    # Determine highest existing file index to prevent file name collisions
    file_indices: Dict[str, int] = {}
    for c_name in target_classes.values():
        c_folder = os.path.join(train_dir, c_name)
        existing_eyepacs = [
            f for f in os.listdir(c_folder)
            if f.startswith(f"{prefix}_") and f.endswith(f".{image_format}")
        ]
        file_indices[c_name] = len(existing_eyepacs)

    start_time = time.time()
    try:
        # Load dataset with optional streaming
        # Streaming avoids downloading the entire 6.5 GB dataset when only a subset is required
        ds = load_dataset(dataset_name, split="train", streaming=streaming)
    except Exception as e:
        sys.exit(f"\nERROR: Failed to connect to Hugging Face dataset '{dataset_name}': {e}\n")

    print("\n[Engine] Ingesting and filtering EyePACS records ...")
    total_inspected = 0

    for record in ds:
        total_inspected += 1

        # Extract label_code
        label_code = record.get("label_code")
        if label_code is None:
            # Fallback check for 'label' integer field
            label_code = record.get("label")

        if label_code not in target_label_set:
            continue

        class_name = CLASS_MAP[label_code]

        # Check if class quota has been satisfied
        if not unlimited and added_counts[class_name] >= max_per_class:
            # Check if all targeted classes have satisfied their quota
            if all(added_counts[c] >= max_per_class for c in target_classes.values()):
                print("\n[Engine] Reached target quota for all requested classes.")
                break
            continue

        # Extract and save image
        raw_image = record.get("image")
        if raw_image is None:
            continue

        # Ensure image is in RGB mode
        if isinstance(raw_image, Image.Image):
            pil_img = raw_image.convert("RGB")
        else:
            try:
                pil_img = Image.open(raw_image).convert("RGB")
            except Exception:
                continue

        # Construct non-colliding target path
        file_indices[class_name] += 1
        seq_num = file_indices[class_name]
        filename = f"{prefix}_{class_name}_{seq_num:05d}.{image_format}"
        dst_path = os.path.join(train_dir, class_name, filename)

        # Save to disk
        pil_img.save(dst_path, format=image_format.upper(), quality=95)
        added_counts[class_name] += 1

        # Periodic progress logging
        total_added_so_far = sum(added_counts.values())
        if total_added_so_far % 50 == 0:
            status_parts = [f"{c}: {added_counts[c]}" for c in target_classes.values()]
            elapsed = time.time() - start_time
            print(f"  Processed {total_inspected:,} records -> Added: {', '.join(status_parts)} ({elapsed:.1f}s)")

    elapsed_total = time.time() - start_time
    print(f"\n[Engine] Ingestion loop completed in {elapsed_total:.1f}s (inspected {total_inspected:,} candidate images).")
    return added_counts


# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("\n" + "=" * 70)
    print("  EYEPACS SUPPLEMENTARY DATA INGESTION (PHASE 5 EXTENSION)")
    print("=" * 70)

    # 1. Verify train directory layout & lock validation/test splits
    print(f"\n[1] Verifying dataset split integrity under: {args.split_dir}")
    train_dir = verify_train_directory(args.split_dir)
    print(f"     Target Directory : {train_dir}")
    print(f"     Isolation Policy : Val and Test directories are NEVER modified.")

    # 2. Compute initial training baseline counts
    initial_counts = get_class_counts(train_dir)
    print(f"\n[2] Initial training counts baseline:")
    for label_idx in sorted(CLASS_MAP.keys()):
        c_name = CLASS_MAP[label_idx]
        print(f"     Class {label_idx} ({c_name:<16}): {initial_counts.get(c_name, 0):>5} images")

    # 3. Ingest and save EyePACS images
    print(f"\n[3] Ingesting EyePACS supplementary samples ...")
    added_counts = add_eyepacs_samples(
        dataset_name=args.dataset_name,
        train_dir=train_dir,
        target_labels=args.target_labels,
        max_per_class=args.max_per_class,
        streaming=not args.no_streaming,
        prefix=args.prefix,
        image_format=args.image_format,
    )

    # 4. Display before/after summary distribution table
    print_distribution_table(initial_counts, added_counts, args.target_labels)

    # 5. Write structured ingestion log to reports/
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "training")
    os.makedirs(reports_dir, exist_ok=True)
    log_file = os.path.join(reports_dir, "external_data_ingestion.json")

    summary_data = {
        "dataset_source": args.dataset_name,
        "target_labels": args.target_labels,
        "target_classes": [CLASS_MAP[l] for l in args.target_labels if l in CLASS_MAP],
        "max_per_class": args.max_per_class,
        "initial_counts": initial_counts,
        "added_counts": added_counts,
        "final_counts": {
            c: initial_counts.get(c, 0) + added_counts.get(c, 0)
            for c in CLASS_MAP.values()
        },
        "total_added": sum(added_counts.values()),
        "train_directory": train_dir,
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[DONE] Ingestion complete. Summary report saved -> {log_file}\n")


if __name__ == "__main__":
    main()
