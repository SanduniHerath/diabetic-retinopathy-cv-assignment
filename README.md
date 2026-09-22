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
│   └── organized/        # Class-separated images (NOT tracked by git)
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

> **Note:** Raw data and organised images are excluded from version control via `.gitignore`.  
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

## Phase 1 – Dataset Organisation

The script `src/organize_dataset.py` reads `data/raw/train_images/train.csv` and copies each image into a named class folder:

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

This structure is compatible with `torchvision.datasets.ImageFolder` and `tf.keras.preprocessing.image_dataset_from_directory` for seamless data loading in later phases.

See [`reports/dataset_overview/summary.md`](reports/dataset_overview/summary.md) for the full dataset analysis and ethical discussion.

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

