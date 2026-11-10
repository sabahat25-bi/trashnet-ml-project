# TrashNet – waste image classification

A machine learning group project that classifies images of waste into six categories: **cardboard, glass, metal, paper, plastic and trash**. The goal is to help people sort their waste correctly, using a Streamlit app where you upload a photo and get a prediction.

## Project structure

```
trashnet-ml-project/
├── data/                              # not on GitHub (git-ignored)
│   ├── raw/Trashnet/dataset-resized/  # cardboard/ glass/ metal/ paper/ plastic/ trash/
│   └── processed/                     # created by the notebooks
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb     # exploratory data analysis + colour baseline
│   ├── 02_baseline_cnn.ipynb          # CNN trained from scratch
│   └── 03_transfer_learning.ipynb     # pretrained model
├── src/
│   └── data_utils.py                  # shared code: paths, split, image features
├── reports/figures/                   # saved plots
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repo and install packages**
   ```bash
   git clone https://github.com/sabahat25-bi/trashnet-ml-project.git
   cd trashnet-ml-project
   python -m pip install -r requirements.txt
   ```

2. **Download the data.** Get **TrashNet (dataset-resized)** from Kaggle and unzip it so the class folders sit exactly here:
   ```
   data/raw/Trashnet/dataset-resized/cardboard/...
   ```
   Kaggle zips sometimes add an extra folder level; remove it so the six class folders are directly inside `dataset-resized`.

3. **Check the setup** from the project folder:
   ```bash
   python -m src.data_utils
   ```
   Expected output: `Dataset found: True`, the six categories, and 1,764 / 378 / 382 train / validation / test images (2,524 in total).

> Never hard-code your own path (e.g. `C:\Users\...`). All paths are built from the project folder, so the code works on every computer.

## Shared code

Import from `src/data_utils.py` instead of writing your own versions:

```python
from src.data_utils import DATASET_PATH, CLASSES, split_dataset, load_image, color_histogram

train, val, test = split_dataset(seed=42)   # lists of (path, label)
```

- `split_dataset(seed=42)`: the team's **one** split, stratified 70/15/15 per class, with exact duplicates removed. Same seed = same split for everyone.
- `load_image()`: opens an image (file path or Streamlit upload) as RGB.
- `color_histogram()`: HSV colour histogram, a 1 × 128 feature vector.

## Key EDA findings

- 2,527 images, all 512 × 384 RGB; **imbalanced** classes (paper 594 vs trash 137).
- 3 photos are duplicated under different labels; the split keeps only one copy.
- 37 % of cardboard images are close-ups without background (possible shortcut).
- **Benchmark to beat: macro-F1 = 0.77** (random forest on HSV colour histograms).

## Team rules

- Work on your own branch and merge into `main` through a pull request.
- Always use `split_dataset(seed=42)`, so all models are compared on the same test set.
- Main metric: **macro-F1**, plus per-class recall and confusion matrices.
- Never commit `data/` or `__pycache__/`.
- Notebooks must run with **Restart → Run All**.
