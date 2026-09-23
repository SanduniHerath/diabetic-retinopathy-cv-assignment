# Diabetic Retinopathy Stage Detection
**Computer Vision Assignment — BSc (Hons) Computer Science, National Institute of Business Management**

A deep-learning pipeline that classifies **5 stages of diabetic retinopathy** from retinal fundus photographs using CNN architectures and transfer learning.

---

## Project Structure

```
diabetic-retinopathy-cv-assignment/
├── app/                  # Interactive Streamlit/Gradio UI prototype
├── data/
│   ├── raw/              # Original downloaded dataset (NOT tracked by git)
│   ├── organized/        # Class-separated images (NOT tracked by git)
│   └── split/            # Stratified Train/Val/Test split (NOT tracked by git)
├── docs/                 # Assignment brief and lecturer checklist
├── notebooks/            # Exploratory and experimental Jupyter notebooks
├── reports/              # Figures, charts, and markdown summaries
├── src/                  # Modular Python source scripts
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

## GitHub Repository
https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment  
*(Repository must be public for submission)*

