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

## GitHub Repository
https://github.com/SanduniHerath/diabetic-retinopathy-cv-assignment  
*(Repository must be public for submission)*
