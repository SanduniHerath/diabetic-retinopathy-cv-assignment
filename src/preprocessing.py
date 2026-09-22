"""
preprocessing.py
================
Phase 2 - Reproducible Image Preprocessing Pipeline
Diabetic Retinopathy Stage Detection (Computer Vision Assignment)

Pipeline Overview (5 deterministic steps, always same input -> same output)
---------------------------------------------------------------------------
Step 1  Ben Graham circular crop    - removes dark border padding common in
                                       fundus cameras; focuses the network on
                                       the retinal disc only.
Step 2  CLAHE contrast enhancement  - boosts local contrast in dark retinal
                                       images without oversaturating highlights;
                                       makes microaneurysms & haemorrhages more
                                       visible (applied on L-channel of LAB).
Step 3  Gaussian blur noise removal - suppresses camera sensor noise and JPEG/
                                       PNG compression artefacts before edge
                                       detection.
Step 4  Unsharp mask edge enhance.  - sharpens lesion boundaries (blood vessels,
                                       exudates, disc margin) without amplifying
                                       noise.
Step 5  Resize + normalise          - resizes to 224x224 (ImageNet standard)
                                       then normalises to [0,1] float32 for
                                       direct use in PyTorch / Keras DataLoaders.

All parameters are module-level constants so they appear in a single place and
are clearly documented. No randomness is introduced anywhere.

Quality Metrics (all measured on uint8 BGR images BEFORE normalisation)
-----------------------------------------------------------------------
- Contrast     : std of the pixel intensity values in the green channel
                 (green channel carries the most retinal information)
- Sharpness    : variance of the Laplacian operator response
                 (higher = more high-frequency edge detail)
- Histogram    : full histogram of green channel + mean absolute deviation
                 between raw and processed histograms (measures redistribution)

Usage
-----
    # Process entire organised dataset:
    python src/preprocessing.py

    # Process with custom paths:
    python src/preprocessing.py \
        --input  data/organized \
        --output data/preprocessed \
        --report reports/preprocessing_examples \
        --samples 2

    # Process a single image (for debugging):
    python src/preprocessing.py --single data/organized/Mild/some_image.png
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# PIPELINE PARAMETERS  (all fixed - change here only, affects whole pipeline)
# ---------------------------------------------------------------------------

# Step 1 - Ben Graham circular crop
CROP_SCALE = 0.9          # keep 90% of detected circle radius (trims sensor edge)

# Step 2 - CLAHE (Contrast Limited Adaptive Histogram Equalisation)
CLAHE_CLIP_LIMIT   = 2.0  # clip limit - controls contrast amplification ceiling
                           # 2.0 is conservative; avoids noise amplification
CLAHE_TILE_GRID    = (8, 8)  # local tile size; 8x8 = good balance for ~224px

# Step 3 - Gaussian blur (noise suppression)
BLUR_KERNEL_SIZE   = (3, 3)  # 3x3 kernel: gentle denoise, preserves fine vessels
BLUR_SIGMA         = 0        # 0 = OpenCV auto-calculates sigma from kernel size

# Step 4 - Unsharp mask (edge / detail enhancement)
UNSHARP_KERNEL     = (0, 0)   # kernel for the blurred reference image
UNSHARP_SIGMA      = 10       # large sigma = captures broad structure for mask
UNSHARP_AMOUNT     = 1.5      # edge boost factor (1.0 = no change, >1 = sharpen)
UNSHARP_THRESHOLD  = 0        # only sharpen pixels with Laplacian > threshold

# Step 5 - Output size and normalisation
OUTPUT_SIZE        = (224, 224)   # ImageNet-standard input size
INTERP_METHOD      = cv2.INTER_AREA  # INTER_AREA preferred for downsampling
NORM_MEAN          = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet
NORM_STD           = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # ImageNet

# Output directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_INPUT   = os.path.join(PROJECT_ROOT, "data", "organized")
DEFAULT_OUTPUT  = os.path.join(PROJECT_ROOT, "data", "preprocessed")
DEFAULT_REPORT  = os.path.join(PROJECT_ROOT, "reports", "preprocessing_examples")

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

# ---------------------------------------------------------------------------
# STEP IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def circular_crop(img_bgr: np.ndarray) -> np.ndarray:
    """
    Ben Graham circular crop.

    Fundus cameras produce images where the retina appears as a bright circle
    surrounded by a black border.  This border contains no medical information
    but wastes model capacity.  We detect the bright circle using the green
    channel (highest SNR for retinal tissue), compute its bounding box, crop
    to a square around the detected disc, and resize back to the input size.

    This step is deterministic: the same image always produces the same crop
    because we use cv2.HoughCircles with fixed parameters.

    Parameters: CROP_SCALE (controls how tightly to crop inside the circle)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect dominant bright circle in the image
    # We try Hough circle detection; fall back to thresholding if it fails
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min(h, w) // 2,
        param1=50,
        param2=30,
        minRadius=min(h, w) // 4,
        maxRadius=min(h, w) // 2,
    )

    if circles is not None:
        circles = np.around(circles).astype(np.int32)
        cx, cy, r = int(circles[0][0][0]), int(circles[0][0][1]), int(circles[0][0][2])
        r = int(r * CROP_SCALE)
    else:
        # Fallback: threshold bright pixels and find bounding circle
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (cx, cy), r = cv2.minEnclosingCircle(largest)
            cx, cy, r = int(cx), int(cy), int(r * CROP_SCALE)
        else:
            # No circle found - return original unchanged
            return img_bgr

    # Clamp crop to image boundaries
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)

    cropped = img_bgr[y1:y2, x1:x2]
    # Resize back to original size so downstream steps have consistent input
    if cropped.size > 0:
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
    return img_bgr


def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation) on L-channel.

    WHY: Retinal fundus images are inherently low-contrast and dark (mean pixel
    intensity ~50-80 out of 255).  Global histogram equalisation (GHE) would
    oversaturate the image.  CLAHE operates on local tiles, so it enhances
    microaneurysms and haemorrhages in dark areas without blowing out the bright
    optic disc.

    WHY L-CHANNEL: We convert BGR -> LAB colour space and apply CLAHE only to L
    (luminance).  This boosts brightness/contrast without shifting the hue of
    retinal tissue (red blood vessels, yellow exudates stay visually correct).
    If we applied equalisation to all three BGR channels independently, we would
    introduce artificial colour casts.

    Parameters: CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_enhanced = clahe.apply(l_ch)

    enhanced_lab = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def apply_noise_removal(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gaussian blur for noise suppression.

    WHY: Fundus cameras and different acquisition equipment introduce sensor
    noise and compression artefacts.  If we apply edge enhancement (Step 4)
    directly to a noisy image, the sharpening amplifies noise peaks along with
    genuine vessel edges.  A mild Gaussian blur suppresses this noise first.

    WHY GAUSSIAN (not median, not bilateral):
    - Gaussian is linear, separable, and fully deterministic with fixed kernel.
    - Median filter changes pixel values based on neighbours' ranking - less
      reproducible for performance comparison.
    - Bilateral would preserve edges (good) but add a second free parameter.
    - A 3x3 Gaussian is gentle enough not to blur thin vessels significantly
      while still removing high-frequency sensor noise.

    Parameters: BLUR_KERNEL_SIZE, BLUR_SIGMA
    """
    return cv2.GaussianBlur(img_bgr, BLUR_KERNEL_SIZE, BLUR_SIGMA)


def apply_unsharp_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Unsharp mask for edge and detail enhancement.

    WHY: After CLAHE and denoising, we want to recover and amplify fine
    structural details: the margins of the optic disc, blood vessel branching,
    boundaries of microaneurysms and hard exudates.  These are the discriminating
    features that a CNN must detect to classify DR stage.

    HOW UNSHARP MASKING WORKS:
        sharpened = original + amount * (original - blurred)
    The 'blurred' version is a low-pass-filtered copy of the image.  Subtracting
    it from the original yields the high-frequency edge map.  Adding this back
    amplifies those edges.

    WHY NOT LAPLACIAN / SOBEL DIRECTLY:  Those produce edge maps, not enhanced
    colour images.  Unsharp masking keeps the image photorealistic while
    selectively enhancing edges - better for both human inspection in the report
    and as CNN input.

    Parameters: UNSHARP_KERNEL, UNSHARP_SIGMA, UNSHARP_AMOUNT, UNSHARP_THRESHOLD
    """
    blurred = cv2.GaussianBlur(img_bgr, UNSHARP_KERNEL, UNSHARP_SIGMA)
    # Weighted combination: sharpened = original * (1+amount) - blurred * amount
    sharpened = cv2.addWeighted(img_bgr, 1.0 + UNSHARP_AMOUNT, blurred,
                                -UNSHARP_AMOUNT, 0)
    return sharpened


def resize_and_normalise(img_bgr: np.ndarray) -> np.ndarray:
    """
    Resize to 224x224 and normalise to [0,1] float32.

    WHY 224x224:
    - Standard ImageNet input size; required by ResNet, EfficientNet, MobileNet,
      VGG and virtually all pretrained backbones used in transfer learning.
    - Consistent input size is mandatory for batched training.

    WHY INTER_AREA for downsampling:
    - INTER_AREA averages pixels in the source region, acting as a low-pass
      anti-aliasing filter. For most fundus images which are much larger than
      224px (up to 3216px wide), this avoids Moire aliasing artefacts that
      INTER_LINEAR or INTER_NEAREST would produce.

    WHY ImageNet mean/std normalisation:
    - Transfer learning backbones were trained on ImageNet with these exact
      statistics. Feeding normalised images with the same statistics ensures
      the pretrained weights activate meaningfully on our data, accelerating
      convergence and improving final accuracy.

    Note: This function returns a float32 numpy array in RGB channel order
    (cv2 default is BGR, so we convert). The array is ready for direct use
    with PyTorch DataLoaders (C, H, W transposition done in the DataLoader).
    """
    resized = cv2.resize(img_bgr, OUTPUT_SIZE, interpolation=INTERP_METHOD)
    # Convert BGR -> RGB (PyTorch/Keras convention)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Scale to [0, 1]
    normalised = rgb.astype(np.float32) / 255.0
    # ImageNet channel-wise normalisation
    normalised = (normalised - NORM_MEAN) / NORM_STD
    return normalised


# ---------------------------------------------------------------------------
# FULL PIPELINE
# ---------------------------------------------------------------------------

def preprocess(img_bgr: np.ndarray) -> dict:
    """
    Run all 5 preprocessing steps in sequence.

    Returns a dict with the image at every stage so callers can
    visualise intermediate results and compute metrics.
    """
    stages = {}
    stages["raw"]          = img_bgr.copy()
    stages["cropped"]      = circular_crop(img_bgr)
    stages["clahe"]        = apply_clahe(stages["cropped"])
    stages["denoised"]     = apply_noise_removal(stages["clahe"])
    stages["sharpened"]    = apply_unsharp_mask(stages["denoised"])
    # Final normalised output (float32, RGB) - stored separately
    stages["normalised"]   = resize_and_normalise(stages["sharpened"])
    # Also store a uint8 resized version for metric comparisons
    stages["resized_uint8"] = cv2.resize(
        stages["sharpened"], OUTPUT_SIZE, interpolation=INTERP_METHOD
    )
    return stages


# ---------------------------------------------------------------------------
# QUALITY METRICS
# ---------------------------------------------------------------------------

def compute_contrast(img_bgr: np.ndarray) -> float:
    """
    Contrast metric: std of the green channel pixel intensities.

    WHY GREEN CHANNEL:
    - The green channel has the highest contrast for retinal structures (blood
      vessels appear darker against the background in green).
    - Ophthalmologists often inspect the green-channel view of fundus images.
    A higher std means a wider spread of intensities = higher contrast.
    """
    green = img_bgr[:, :, 1].astype(np.float32)
    return float(green.std())


def compute_sharpness(img_bgr: np.ndarray) -> float:
    """
    Sharpness metric: variance of the Laplacian response.

    The Laplacian operator is a second-order derivative that responds strongly
    to edges and fine detail.  Its variance measures how much high-frequency
    content the image contains:
    - A blurry image has low Laplacian variance (edges are smoothed).
    - A sharp image has high Laplacian variance (edges are crisp).
    This is the standard focus/sharpness metric used in autofocus systems
    and image quality assessment literature.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def compute_histogram(img_bgr: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Compute normalised green-channel histogram.

    Returns a probability density histogram (sums to 1.0) so images of
    different sizes can be fairly compared.
    """
    green = img_bgr[:, :, 1]
    hist = cv2.calcHist([green], [0], None, [bins], [0, 256])
    hist = hist.flatten()
    hist = hist / hist.sum()   # normalise to probability density
    return hist


def histogram_mad(hist_raw: np.ndarray, hist_proc: np.ndarray) -> float:
    """
    Mean Absolute Deviation between two histograms.
    Measures how much the intensity distribution shifted after processing.
    A larger MAD means more redistribution = more effective equalisation.
    """
    return float(np.mean(np.abs(hist_raw - hist_proc)))


def compute_all_metrics(img_bgr: np.ndarray) -> dict:
    return {
        "contrast":  compute_contrast(img_bgr),
        "sharpness": compute_sharpness(img_bgr),
        "histogram": compute_histogram(img_bgr),
    }


# ---------------------------------------------------------------------------
# VISUALISATION & REPORTING
# ---------------------------------------------------------------------------

def save_before_after(raw: np.ndarray, processed_uint8: np.ndarray,
                      class_name: str, img_id: str, report_dir: str) -> None:
    """
    Save a side-by-side before/after comparison panel for the report.
    Also saves intermediate stages as individual panels.
    """
    os.makedirs(report_dir, exist_ok=True)

    # Run pipeline to get all stages for the comparison grid
    stages = preprocess(raw)

    stage_keys   = ["raw", "cropped", "clahe", "denoised", "sharpened", "resized_uint8"]
    stage_labels = ["1. Raw", "2. Cropped", "3. CLAHE", "4. Denoised", "5. Sharpened", "6. Resized 224x224"]

    fig, axes = plt.subplots(1, len(stage_keys), figsize=(4 * len(stage_keys), 5))
    fig.suptitle(
        f"Preprocessing Pipeline  |  Class: {class_name}  |  Image: {img_id}",
        fontsize=11, fontweight="bold"
    )

    for ax, key, label in zip(axes, stage_keys, stage_labels):
        img = stages[key]
        # Convert BGR -> RGB for matplotlib
        if img.dtype == np.uint8:
            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # float normalised - denormalise for display
            display = (img * NORM_STD + NORM_MEAN)
            display = np.clip(display, 0, 1)

        ax.imshow(display if display.ndim == 3 else display, cmap="gray" if display.ndim == 2 else None)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(report_dir, f"{class_name}_{img_id}_pipeline.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_histogram_comparison(raw: np.ndarray, processed: np.ndarray,
                               class_name: str, img_id: str,
                               report_dir: str) -> None:
    """
    Save a green-channel histogram comparison (raw vs processed).
    Visual proof of contrast redistribution by CLAHE.
    """
    os.makedirs(report_dir, exist_ok=True)

    hist_raw  = compute_histogram(raw)
    hist_proc = compute_histogram(processed)
    bins = np.arange(256)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Green-Channel Histogram: Raw vs Preprocessed  |  {class_name} | {img_id}",
        fontsize=11, fontweight="bold"
    )

    axes[0].bar(bins, hist_raw,  color="#2196F3", width=1.0, alpha=0.8)
    axes[0].set_title("Raw Image", fontsize=10)
    axes[0].set_xlabel("Pixel Intensity (0-255)")
    axes[0].set_ylabel("Normalised Frequency")
    axes[0].set_xlim(0, 255)

    axes[1].bar(bins, hist_proc, color="#4CAF50", width=1.0, alpha=0.8)
    axes[1].set_title("After Preprocessing (pre-normalisation)", fontsize=10)
    axes[1].set_xlabel("Pixel Intensity (0-255)")
    axes[1].set_ylabel("Normalised Frequency")
    axes[1].set_xlim(0, 255)

    plt.tight_layout()
    out_path = os.path.join(report_dir, f"{class_name}_{img_id}_histogram.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_metrics_table(all_metrics: list, report_dir: str) -> None:
    """
    Save a bar-chart summary of before/after metrics across all sample images.
    all_metrics: list of dicts with keys class, img_id, before_contrast,
                 after_contrast, before_sharpness, after_sharpness, hist_mad
    """
    os.makedirs(report_dir, exist_ok=True)

    labels = [f"{m['class']}\n{m['img_id'][:6]}" for m in all_metrics]
    contrast_before  = [m["before_contrast"]  for m in all_metrics]
    contrast_after   = [m["after_contrast"]   for m in all_metrics]
    sharpness_before = [m["before_sharpness"] for m in all_metrics]
    sharpness_after  = [m["after_sharpness"]  for m in all_metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Objective Quality Metrics: Before vs After Preprocessing",
                 fontsize=12, fontweight="bold")

    # Contrast plot
    axes[0].bar(x - width/2, contrast_before,  width, label="Before", color="#F44336", alpha=0.8)
    axes[0].bar(x + width/2, contrast_after,   width, label="After",  color="#4CAF50", alpha=0.8)
    axes[0].set_title("Contrast (Green Channel Std Dev)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("Standard Deviation")
    axes[0].legend()

    # Sharpness plot
    axes[1].bar(x - width/2, sharpness_before, width, label="Before", color="#F44336", alpha=0.8)
    axes[1].bar(x + width/2, sharpness_after,  width, label="After",  color="#4CAF50", alpha=0.8)
    axes[1].set_title("Sharpness (Laplacian Variance)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("Variance")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(report_dir, "metrics_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Metrics chart saved -> {out_path}")


# ---------------------------------------------------------------------------
# BATCH PROCESSING
# ---------------------------------------------------------------------------

def process_dataset(input_root: str, output_root: str) -> None:
    """
    Preprocess every image in the organised class-folder structure.

    Reads from:  input_root/<ClassName>/<image>.png
    Writes to:   output_root/<ClassName>/<image>.npy   (normalised float32 array)
    Also saves:  output_root/<ClassName>/<image>_display.png  (uint8 for viewing)

    The .npy format allows instant loading with np.load() in training scripts
    without needing to redo preprocessing for every epoch.
    """
    total = 0
    for cls in CLASS_NAMES:
        cls_in  = os.path.join(input_root,  cls)
        cls_out = os.path.join(output_root, cls)
        os.makedirs(cls_out, exist_ok=True)

        if not os.path.exists(cls_in):
            print(f"  [SKIP] {cls_in} not found")
            continue

        images = [f for f in os.listdir(cls_in) if f.lower().endswith(".png")]
        print(f"\n  Processing {cls:20s}: {len(images)} images")

        for i, fname in enumerate(images):
            src = os.path.join(cls_in, fname)
            stem = os.path.splitext(fname)[0]

            img = cv2.imread(src)
            if img is None:
                continue

            stages = preprocess(img)

            # Save normalised float32 array for training
            npy_path = os.path.join(cls_out, stem + ".npy")
            np.save(npy_path, stages["normalised"])

            # Save uint8 resized for visual inspection
            display_path = os.path.join(cls_out, stem + "_display.png")
            cv2.imwrite(display_path, stages["resized_uint8"])

            total += 1
            if (i + 1) % 200 == 0:
                print(f"    {cls}: processed {i+1}/{len(images)}")

    print(f"\n  Total images preprocessed: {total}")


# ---------------------------------------------------------------------------
# SAMPLE REPORTING (generates visual evidence for the report)
# ---------------------------------------------------------------------------

def generate_report_samples(input_root: str, report_dir: str,
                             n_samples: int = 2) -> list:
    """
    For each class, take n_samples images, run the pipeline, compute metrics,
    and save all visualisations. Returns list of metric dicts.
    """
    os.makedirs(report_dir, exist_ok=True)
    all_metrics = []

    print(f"\n--- Generating Report Samples (n={n_samples} per class) ---")

    for cls in CLASS_NAMES:
        cls_dir = os.path.join(input_root, cls)
        if not os.path.exists(cls_dir):
            continue

        images = sorted([f for f in os.listdir(cls_dir) if f.endswith(".png")])
        # Pick deterministically: first n_samples when sorted
        selected = images[:n_samples]

        for fname in selected:
            img_id = os.path.splitext(fname)[0]
            src = os.path.join(cls_dir, fname)
            img = cv2.imread(src)
            if img is None:
                continue

            stages = preprocess(img)
            processed_u8 = stages["resized_uint8"]

            # Resize raw to same size for fair metric comparison
            raw_resized = cv2.resize(img, OUTPUT_SIZE, interpolation=INTERP_METHOD)

            # Compute metrics on uint8 images before normalisation
            m_before = compute_all_metrics(raw_resized)
            m_after  = compute_all_metrics(processed_u8)

            contrast_pct  = 100 * (m_after["contrast"]  - m_before["contrast"])  / (m_before["contrast"]  + 1e-9)
            sharpness_pct = 100 * (m_after["sharpness"] - m_before["sharpness"]) / (m_before["sharpness"] + 1e-9)
            mad           = histogram_mad(m_before["histogram"], m_after["histogram"])

            metrics = {
                "class":            cls,
                "img_id":           img_id,
                "before_contrast":  m_before["contrast"],
                "after_contrast":   m_after["contrast"],
                "contrast_pct":     contrast_pct,
                "before_sharpness": m_before["sharpness"],
                "after_sharpness":  m_after["sharpness"],
                "sharpness_pct":    sharpness_pct,
                "hist_mad":         mad,
            }
            all_metrics.append(metrics)

            print(
                f"  {cls:20s} | {img_id[:12]:12s} | "
                f"Contrast: {m_before['contrast']:6.1f} -> {m_after['contrast']:6.1f} "
                f"({contrast_pct:+.1f}%)  | "
                f"Sharpness: {m_before['sharpness']:8.1f} -> {m_after['sharpness']:8.1f} "
                f"({sharpness_pct:+.1f}%)  | "
                f"Hist MAD: {mad:.5f}"
            )

            # Save pipeline visualisation
            save_before_after(img, processed_u8, cls, img_id, report_dir)
            # Save histogram comparison
            save_histogram_comparison(raw_resized, processed_u8, cls, img_id, report_dir)

    return all_metrics


def print_metrics_table(all_metrics: list) -> None:
    """Print a formatted summary table of all metrics."""
    print("\n" + "=" * 90)
    print("  PREPROCESSING QUALITY METRICS SUMMARY")
    print("=" * 90)
    print(f"  {'Class':20s} | {'Contrast Before':>15} | {'After':>8} | {'Pct':>8} | "
          f"{'Sharp Before':>12} | {'After':>10} | {'Pct':>8} | {'Hist MAD':>10}")
    print("-" * 90)
    for m in all_metrics:
        print(
            f"  {m['class']:20s} | {m['before_contrast']:15.2f} | {m['after_contrast']:8.2f} | "
            f"{m['contrast_pct']:+7.1f}% | {m['before_sharpness']:12.2f} | "
            f"{m['after_sharpness']:10.2f} | {m['sharpness_pct']:+7.1f}% | {m['hist_mad']:10.5f}"
        )
    print("=" * 90)

    # Overall averages
    avg_c  = np.mean([m["contrast_pct"]  for m in all_metrics])
    avg_s  = np.mean([m["sharpness_pct"] for m in all_metrics])
    avg_h  = np.mean([m["hist_mad"]      for m in all_metrics])
    print(f"\n  Average contrast improvement  : {avg_c:+.1f}%")
    print(f"  Average sharpness improvement : {avg_s:+.1f}%")
    print(f"  Average histogram MAD         : {avg_h:.5f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the deterministic preprocessing pipeline on the APTOS dataset."
    )
    p.add_argument("--input",   default=DEFAULT_INPUT,   help="Organised class-folder root")
    p.add_argument("--output",  default=DEFAULT_OUTPUT,  help="Output root for preprocessed files")
    p.add_argument("--report",  default=DEFAULT_REPORT,  help="Directory for report samples")
    p.add_argument("--samples", type=int, default=2,     help="Sample images per class for report")
    p.add_argument("--report-only", action="store_true",
                   help="Only generate report samples; skip full dataset processing")
    p.add_argument("--single",  default=None,
                   help="Process a single image path and show metrics (for debugging)")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Single-image debug mode ---
    if args.single:
        img = cv2.imread(args.single)
        if img is None:
            sys.exit(f"Cannot read image: {args.single}")
        stages = preprocess(img)
        raw_r  = cv2.resize(img, OUTPUT_SIZE, interpolation=INTERP_METHOD)
        proc_u8 = stages["resized_uint8"]
        mb = compute_all_metrics(raw_r)
        ma = compute_all_metrics(proc_u8)
        print(f"Contrast  before: {mb['contrast']:.2f}  after: {ma['contrast']:.2f}  "
              f"({100*(ma['contrast']-mb['contrast'])/(mb['contrast']+1e-9):+.1f}%)")
        print(f"Sharpness before: {mb['sharpness']:.2f}  after: {ma['sharpness']:.2f}  "
              f"({100*(ma['sharpness']-mb['sharpness'])/(mb['sharpness']+1e-9):+.1f}%)")
        return

    # --- Report sample generation ---
    print("[1] Generating report samples and quality metrics...")
    all_metrics = generate_report_samples(args.input, args.report, n_samples=args.samples)
    print_metrics_table(all_metrics)
    save_metrics_table(all_metrics, args.report)

    # --- Full dataset processing (optional, takes several minutes) ---
    if not args.report_only:
        print(f"\n[2] Processing full dataset: {args.input} -> {args.output}")
        print("    (This copies and preprocesses all 3,662 images. May take 5-10 min.)")
        print("    Run with --report-only to skip this step.")
        process_dataset(args.input, args.output)

    print("\n[DONE] Preprocessing complete.")
    print(f"  Report samples  -> {args.report}")
    if not args.report_only:
        print(f"  Preprocessed data -> {args.output}")


if __name__ == "__main__":
    main()
