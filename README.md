# Diabetic Retinopathy Stage Detection
**Computer Vision Assignment — BSc (Hons) Computer Science, National Institute of Business Management (In Collaboration with Coventry University)**

| Student Details | Value |
|:---|:---|
| **Student Name** | H.M.S.S Herath |
| **Student Index Number** | COBSCCOMP242P-016 |
| **Coventry ID** | 16114092 |
| **Live Hosted Web Application** | [https://cobsccomp242p016-retinascan-ai.onrender.com/](https://cobsccomp242p016-retinascan-ai.onrender.com/) |

A deep-learning pipeline that classifies **5 stages of diabetic retinopathy** from retinal fundus photographs using CNN architectures and transfer learning.

---

## Project Structure

```
diabetic-retinopathy-cv-assignment/
├── app/                          # Clinical Decision Support Web Application
│   ├── main.py                   # NiceGUI interface entry point
│   ├── predictor.py              # PyTorch model inference + Grad-CAM
│   ├── pdf_report.py             # Structured clinical PDF report generator
│   └── samples/                  # Bundled demo images (1 per class, no dataset needed)
├── data/
│   └── split/                    # Stratified splits (NOT tracked by git)
│       ├── train/                # 4,063 training images (incl. EyePACS supplementary)
│       ├── val/                  # 550 validation images
│       └── test/                 # 549 test images (held-out, never modified)
├── models/
│   └── best_model.pth            # Deployed EfficientNet-B0 checkpoint (17.7 MB)
├── notebooks/                    # Colab and Kaggle training notebooks
├── reports/
│   ├── hp_results/               # 4-run HP search curves, CSVs, classification reports
│   ├── training/                 # Final model training summary and curves
│   └── evaluation/               # Confusion matrix, Grad-CAM, classification report
├── src/                          # Modular Python pipeline scripts
│   ├── organize_dataset.py       # Phase 1: CSV → class folders
│   ├── split_dataset.py          # Phase 1: 70/15/15 stratified split
│   ├── preprocessing.py          # Phase 2: CLAHE, crop, normalize
│   ├── augmentation.py           # Phase 3: Offline augmentation + class balancing
│   ├── model.py                  # Phase 4: EfficientNet-B0 architecture
│   ├── gradcam.py                # Phase 4: Gradient-weighted class activation maps
│   ├── add_external_data.py      # Phase 5: EyePACS minority-class supplementation
│   ├── train.py                  # Phase 5: Two-stage fine-tuning + HP grid search
│   └── evaluate.py               # Phase 5: Test-set evaluation + metrics
├── requirements.txt
└── README.md
```

---

## Dataset

### Source
**APTOS 2019 Blindness Detection** — available on Kaggle:  
https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

### How to Download
1. Install the Kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` API key in `~/.kaggle/`
3. Run:
   ```bash
   kaggle competitions download -c aptos2019-blindness-detection -p data/raw/
   unzip data/raw/aptos2019-blindness-detection.zip -d data/raw/
   ```

> **Note:** Raw data, organized images, and split directories are excluded from version control via `.gitignore`.  
> You must download the dataset manually before running any scripts.

### Dataset Summary
| Stat | Value |
|------|-------|
| Total images | 3,662 retinal fundus photos |
| Format | PNG (variable resolution) |
| Labels | 5 classes (DR stages 0–4) |
| Source | APTOS 2019 (Kaggle), graded by clinical specialists |

### Class Distribution (Severe Imbalance)
| Label | Class Name | Count | % |
|-------|-----------|-------|---|
| 0 | No_DR | 1805 | 49.3% |
| 1 | Mild | 370 | 10.1% |
| 2 | Moderate | 999 | 27.3% |
| 3 | Severe | 193 | 5.3% |
| 4 | Proliferative_DR | 295 | 8.1% |

The imbalance ratio (dominant/minority) is ~9.4×. Addressed via augmentation, class-weighted loss, and per-class evaluation metrics.

---

## Phase 1 – Dataset Organisation & Stratified Split

### 1. Dataset Organisation (`src/organize_dataset.py`)
Reads `data/raw/train_images/train.csv` and copies each image into a named class folder:

```
data/organized/
    No_DR/              # 1805 images
    Mild/               # 370 images
    Moderate/           # 999 images
    Severe/             # 193 images
    Proliferative_DR/   # 295 images
```

Run it with:
```bash
python src/organize_dataset.py
```

### 2. Stratified Train / Validation / Test Split (`src/split_dataset.py`)
To prevent data leakage and evaluate generalization on imbalanced clinical stages, `src/split_dataset.py` creates a **70% / 15% / 15%** stratified split (`RANDOM_SEED = 42`) preserving class distributions across all sets:

```
data/split/
    train/              # 2,563 images (70.0%)
        No_DR/ (1,264) | Mild/ (259) | Moderate/ (699) | Severe/ (135) | Proliferative_DR/ (206)
    val/                # 550 images (15.0%)
        No_DR/ (271)   | Mild/ (56)  | Moderate/ (150) | Severe/ (29)  | Proliferative_DR/ (44)
    test/               # 549 images (15.0%)
        No_DR/ (270)   | Mild/ (55)  | Moderate/ (150) | Severe/ (29)  | Proliferative_DR/ (45)
```

Run the split script:
```bash
python src/split_dataset.py
```

- **Zero Data Leakage**: Enforces $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- **Manifest**: Full mapping stored at `reports/dataset_overview/split_manifest.csv`.
- **ImageFolder Compatible**: Both `data/organized/` and `data/split/{train,val,test}/` are directly compatible with PyTorch `ImageFolder` and TensorFlow `image_dataset_from_directory`.

See [`reports/dataset_overview/summary.md`](reports/dataset_overview/summary.md) for the full dataset analysis, split visualizations, and ethical discussion.

---

## Phase 2 – Image Preprocessing Pipeline

The preprocessing pipeline in `src/preprocessing.py` standardizes and enhances raw fundus images through 5 deterministic, reproducible steps:

1. **Circular Crop (Ben Graham method)**: Removes black camera borders and isolates the retinal disc.
2. **Contrast Enhancement (CLAHE on LAB L-channel)**: Enhances microaneurysms and deep lesions locally without hue shift or highlight blowout (`clipLimit=2.0`, `tileGridSize=(8,8)`).
3. **Noise Reduction (Gaussian Blur)**: Suppresses sensor and compression artifacts (`kernel=(3,3)`) prior to sharpening.
4. **Edge Enhancement (Unsharp Masking)**: Sharpens vessel branching and lesion boundaries (`sigma=10`, `amount=1.5`) while preserving photorealistic RGB color profiles.
5. **Resizing & Normalization**: Resizes to 224×224 (`INTER_AREA`) and normalizes with ImageNet channel statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### How to Run
Generate quality metrics, pipeline stage panels, and histogram comparisons on sample images:
```bash
python src/preprocessing.py --report-only --samples 2
```

Preprocess the entire organized dataset to `.npy` tensors and display `.png` files:
```bash
python src/preprocessing.py
```

### Quantitative Improvements
- **Sharpness (Laplacian Variance)**: **+707.3%** average improvement across all stages.
- **Contrast Redistribution (Histogram MAD)**: Mean absolute deviation of **0.00343** confirms significant intensity equalization across low-contrast regions.

See [`reports/preprocessing_examples/summary.md`](reports/preprocessing_examples/summary.md) for full clinical rationales, mathematical formulas, and sample visualizations.

---

## Phase 3 – Data Augmentation & Class Balancing (`src/augmentation.py`)

`src/augmentation.py` applies a reproducible, clinically justified augmentation pipeline **exclusively to the training split** (`data/split/train/`) and writes the balanced output to `data/split/train_augmented/`. The validation and test sets are **never touched**.

### Why Training-Only Augmentation?

Augmenting val/test would constitute **data leakage**: augmented images are synthetic near-duplicates of their source. If near-duplicates of training images appear in the evaluation sets, reported metrics inflate and do not reflect real-world clinical performance. The test set simulates a clinician's camera — always one natural, unaugmented photograph per patient eye.

### Augmentation Techniques

| Technique | Parameters | Clinical Rationale |
|:----------|:-----------|:------------------|
| Horizontal Flip | p = 0.5 | Left/right eyes are mirror images; doubles anatomical diversity |
| Vertical Flip | p = 0.5 | Handheld cameras can be tilted to any orientation; anatomically plausible |
| Random Rotation | ±30°, mirror padding | Simulates patient head-tilt and camera misalignment in field screening |
| Random Zoom / Crop | Scale 0.80–1.00 | Simulates focal-length variation across camera models |
| Brightness & Contrast Jitter | HSV-V ±30; ×0.85–1.15 | Hue preserved (clinically important colours); simulates illumination variation |
| Gaussian Blur | p = 0.30, 3×3 kernel | Simulates minor camera defocus at low probability |

### Flip Decision: Both H + V Flips (Literature Evidence)

Both flips are applied following the majority consensus in DR classification literature:
- Haque et al. 2021 [arXiv:2108.04358](https://arxiv.org/abs/2108.04358) — H+V flips on APTOS fundus images
- Al-Antary et al. 2025 [arXiv:2604.23079](https://arxiv.org/abs/2604.23079) — H+V flips, QWK=0.934 on APTOS 2019
- Huang et al. 2021 [arXiv:2110.14160](https://arxiv.org/abs/2110.14160) — Ablation confirms H+V flips beneficial for DR grading
- Hannan et al. 2025 [arXiv:2507.19199](https://arxiv.org/abs/2507.19199) — H+V flips on APTOS 2019 with rotation

Iqbal et al. 2024 [arXiv:2408.06784](https://arxiv.org/abs/2408.06784) use horizontal-only flips, but for *exudate segmentation* — a pixel-level task where vertical orientation relative to the disc matters. For our *classification* objective, both flips are appropriate.

### Class Balancing (Training Set Only)

The training split has a **9.36× imbalance ratio** (No_DR: 1,264 vs Severe: 135 images). Differential multipliers reduce this to **1.15×**:

| Class | Train Before | Multiplier | Train After |
|:------|:-----------:|:----------:|:-----------:|
| No_DR | 1,264 | ×1 | 1,264 |
| Mild | 259 | ×5 | 1,295 |
| Moderate | 699 | ×2 | 1,398 |
| Severe | 135 | **×9** | 1,215 |
| Proliferative_DR | 206 | ×6 | 1,236 |
| **Total** | **2,563** | | **6,408** |

Severe receives ×9 (highest) because it is both the rarest class and the most clinically critical — misclassifying Severe as a lower grade risks preventable blindness.

### How to Run

```bash
# Dry-run (prints before/after tables, no files written):
python src/augmentation.py --dry-run

# Full pipeline (generates 6,408 training images):
python src/augmentation.py

# Report-only (demo strips + chart only):
python src/augmentation.py --report-only --samples 3
```

- **Seed**: `RANDOM_SEED = 42` — fully reproducible output
- **Output**: `data/split/train_augmented/[class]/`
- **Never reads from**: `data/split/val/` or `data/split/test/`

See [`reports/augmentation_examples/summary.md`](reports/augmentation_examples/summary.md) for the full clinical justification, flip decision citation table, and before/after distribution charts.

---

## Phase 4 – CNN Architecture with Transfer Learning & Grad-CAM (`src/model.py`, `src/gradcam.py`)

Phase 4 defines a transfer learning architecture in **PyTorch** designed to run seamlessly in **Google Colab** (free-tier T4 GPU) and local environments using portable, relative-path logic.

### Unified 5-Class Model: Dual Assignment Deliverables
A single model satisfies both core requirements simultaneously from one forward pass:
- **Binary DR Presence (Screening)**: $P(\text{DR Positive}) = 1.0 - p_0 = \sum_{k=1}^4 p_k$. If $P(\text{DR}) \ge 0.50$, flags DR present.
- **Disease Stage Grading (Severity)**: $\hat{y} = \arg\max_{i \in \{0..4\}} p_i$ (0: No_DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative_DR).

### Architecture Highlights
- **Primary Backbone**: **EfficientNet-B0** (pretrained on ImageNet-1K).
  - *Parameters*: **4,337,281 (~4.3M)** — 5.5× lighter than ResNet-50.
  - *Squeeze-and-Excitation (SE)*: Dynamically recalibrates feature maps to highlight tiny retinal lesions (microaneurysms, hemorrhages) against homogeneous reddish retinal backgrounds.
  - *Colab T4 Efficiency*: ~2.5–3.5 min/epoch with batch size 32; avoids OOM errors on free-tier 15GB VRAM limits.
- **Alternative Baseline**: **ResNet-50** (24,034,373 parameters) supported via `--backbone resnet50`.
- **Classification Head**:
  - `AdaptiveAvgPool2d((1, 1))` $\to$ `Flatten()`
  - `Dropout(p=0.4)` $\to$ `Linear(1280, 256)` $\to$ `BatchNorm1d(256)` $\to$ `SiLU()`
  - `Dropout(p=0.2)` $\to$ `Linear(256, 5)` $\to$ Raw unnormalized logits.

### Staged Freezing Scheme
1. **Stage 1 (Feature Extraction / Head Warmup)**: Backbone frozen (4,007,548 parameters frozen; 329,733 trainable, 7.6%). Prevents destroying pretrained ImageNet weights while training random head.
2. **Stage 2 (Differential Fine-Tuning)**: Top 2 MBConv blocks unfrozen (1,459,125 trainable parameters, 33.6%). Adapts high-level semantics to retinal pathology at reduced learning rate (`5e-5`).

### Grad-CAM Visual Explainability (`src/gradcam.py`)
- Computes gradient-weighted class activation maps targeting the last convolutional layer (`features[-1]`).
- Overlays a heatmap on RGB fundus images to clinically verify that the network attends to pathological lesions rather than camera artifacts or borders.

### How to Run Locally or in Colab
```bash
# EfficientNet-B0 architecture summary and verification:
python src/model.py --backbone efficientnet_b0

# ResNet-50 baseline comparison:
python src/model.py --backbone resnet50 --output-summary reports/model_architecture/model_summary_resnet50.txt
```

See [`reports/model_architecture/summary.md`](reports/model_architecture/summary.md) for the complete design justification, parameter audits, and the Phase 5 hyperparameter tuning plan.

---

## Phase 5 – Transfer Learning, EyePACS Data & Hyperparameter Search (`src/train.py`, `src/add_external_data.py`)

### EfficientNet-B0 Two-Stage Staged Fine-Tuning

Training uses a two-stage protocol to avoid catastrophic forgetting of ImageNet representations:

| Stage | Backbone | Trainable Layers | LR | Max Epochs |
|:------|:---------|:-----------------|:---|:----------:|
| Stage 1: Head Warmup | Frozen | Classification head only | `1e-3` (AdamW) | 10 |
| Stage 2: Selective Fine-Tuning | Top 3 blocks unfrozen | Top blocks + head | `1e-4` (AdamW) | 20 |

### EyePACS Supplementary Data Ingestion

The APTOS training set has extreme class imbalance (9.4× ratio). To improve sensitivity on the three clinically under-represented minority grades, 500 supplementary images per class were streamed from the EyePACS dataset hosted on HuggingFace (`Lelihu/eyepacs`):

| Class | APTOS Train | EyePACS Added | Final Train |
|:------|:-----------:|:-------------:|:-----------:|
| No_DR | 1,264 | 0 | 1,264 |
| **Mild** | 259 | +500 | 759 |
| Moderate | 699 | 0 | 699 |
| **Severe** | 135 | +500 | 635 |
| **Proliferative_DR** | 206 | +500 | 706 |
| **Total** | **2,563** | **+1,500** | **4,063** |

```bash
python src/add_external_data.py --max-per-class 500
```

### Hyperparameter Grid Search

A systematic 4-configuration search was run using `--hp-search` (8 Stage 1 + 12 Stage 2 epochs per run). Results are fully reproducible from the executed Kaggle notebook at `notebooks/kaggle_hyperparameter_search.ipynb`.

| Run | Stage1 LR | Stage2 LR | Unfreeze | Weight Decay | Val Loss | Test Acc | QWK |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **run01 ✓** | `1e-3` | `1e-4` | 3 | `1e-4` | **0.4911** | **79.96%** | **0.8762** |
| run02 | `5e-4` | `5e-5` | 3 | `1e-4` | 0.5032 | 77.60% | 0.8673 |
| run03 | `1e-3` | `1e-4` | 2 | `1e-4` | 0.5538 | 76.14% | 0.8420 |
| run04 | `1e-3` | `1e-4` | 3 | `1e-5` | 0.5089 | 77.23% | 0.8649 |

Run01 won on every metric — validation loss was a reliable selection proxy.

### Final Deployed Model Performance

| Metric | Value |
|:-------|:-----:|
| **Test Accuracy** | **80.33%** |
| **Quadratic Weighted Kappa (QWK)** | **0.8799** |
| **Weighted F1** | **0.78** |
| **Macro F1** | **0.59** |
| No_DR F1 | 0.966 |
| Mild F1 | 0.348 |
| Moderate F1 | 0.758 |
| Severe F1 | 0.346 |
| Proliferative_DR F1 | 0.548 |

> The QWK of 0.88 sits in the "Almost Perfect Agreement" band — comparable to gold-medal solutions in the APTOS 2019 Kaggle competition (0.88–0.92).

### How to Train

```bash
# HP grid search (produces reports/hp_results/hyperparameter_results.csv):
python src/train.py \
  --data-root data/split \
  --output-dir reports/hp_results \
  --backbone efficientnet_b0 \
  --batch-size 32 --num-workers 2 --fast-loader \
  --hp-search

# Full final training with best config:
python src/train.py \
  --data-root data/split \
  --output-dir reports/training \
  --backbone efficientnet_b0 \
  --stage1-epochs 10 --stage2-epochs 20 \
  --batch-size 32 --num-workers 2 --fast-loader

# Evaluate on held-out test set:
python src/evaluate.py \
  --checkpoint reports/training/best_model_*.pth \
  --data-root data/split \
  --output-dir reports/evaluation
```

See [`reports/training/summary.md`](reports/training/summary.md) and [`reports/hp_results/test_evaluation_comparison.md`](reports/hp_results/test_evaluation_comparison.md) for full training details and cross-run comparison tables.

---

## Phase 6 – Clinical Decision Support Application (`app/`)

### Overview

A production-quality clinical web interface built with **NiceGUI** + **PyTorch**, modeled after deployed DR screening platforms (EyeArt, IDx-DR). The app performs real-time EfficientNet-B0 inference and provides structured triage output for clinical decision support.

### Features

| Feature | Description |
|:--------|:------------|
| **Drag-and-Drop Upload** | Upload any retinal fundus photograph (PNG/JPG) |
| **1-Click Clinical Samples** | Instantly load a verified exemplar from each of the 5 DR stages — no dataset required |
| **Retinal Workstation** | Side-by-side: original fundus image + real-time Grad-CAM saliency heatmap |
| **Clinical Severity Staging** | Grade 0–4 with confidence distribution bar |
| **Referral Triage** | Color-coded: No Referral Needed / Routine Follow-up / Refer Urgently |
| **Clinical AI Chatbot** | Severity-aware assistant for lesion questions, follow-up schedules, referral guidance |
| **PDF Screening Report** | One-click structured audit-ready clinical report with imaging, metrics & recommendations |

### Installation

```bash
# Clone the repository
git clone https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment.git
cd diabetic-retinopathy-cv-assignment

# Install dependencies
pip install -r requirements.txt
```

### Getting the Model Checkpoint (`best_model.pth`)

The trained model checkpoint is **17.7 MB** and is tracked in the `models/` directory of this repository. If it is missing locally (e.g. after a fresh clone without Git LFS), download it directly:

```bash
# Option 1 — Direct download from the GitHub release (if published):
# https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment/releases

# Option 2 — From Kaggle working output (if you ran the Kaggle notebook):
# Download from notebooks/kaggle_hyperparameter_search.ipynb Output tab → best_model.pth
# Place at: models/best_model.pth

# Option 3 — Re-train from scratch:
python src/train.py --data-root data/split --output-dir reports/training \
  --backbone efficientnet_b0 --stage1-epochs 10 --stage2-epochs 20 \
  --batch-size 32 --num-workers 2 --fast-loader
# Then copy the produced checkpoint: cp reports/training/best_model_*.pth models/best_model.pth
```

### Running the App

#### Option A — Live Cloud Deployment (No Installation Needed)
The application is hosted 24/7 on Render and can be accessed directly from any web browser:
👉 **[https://cobsccomp242p016-retinascan-ai.onrender.com/](https://cobsccomp242p016-retinascan-ai.onrender.com/)**

#### Option B — Running Locally
```bash
# Launch the clinical interface locally (no dataset needed — bundled samples in app/samples/):
python app/main.py
```
Open your browser at **http://localhost:8080** (or `http://127.0.0.1:8080`).

> **Regulatory Notice:** This application is an investigational research prototype. It is NOT an FDA, CE-mark, or MHRA cleared diagnostic device. All automated findings must be verified by a licensed ophthalmologist before clinical use.

---

## GitHub Repository
https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment
*(Repository must be public for submission)*
