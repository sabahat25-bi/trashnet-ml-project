"""
Shared code for the TrashNet project:
1. Paths to the dataset
2. Train/validation/test split (without duplicates)
3. Image features (colour histograms) used by the EDA, the models and the app
"""
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------
# Main project folder (this file is in src/, so go up one level)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# TrashNet dataset location
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Trashnet" / "dataset-resized"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


# ---------------------------------------------------------------
# 2. Train/Validation/Test split
# ---------------------------------------------------------------
def split_dataset(dataset_path=DATASET_PATH, seed=42):
    """
    Split the images 70/15/15 inside every class (stratified).
    Exact duplicates are removed, so no copy ends up in both train and test.
    Returns three lists of (path, label).
    """
    random.seed(seed)

    train_files = []
    val_files = []
    test_files = []
    seen = set()  # fingerprints of images already used

    for category in sorted(dataset_path.iterdir()):
        if not category.is_dir():
            continue

        images = [
            (str(image), category.name)
            for image in sorted(category.iterdir())
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        # Remove exact duplicates (same file content)
        unique_images = []
        for path, label in images:
            fingerprint = hashlib.md5(Path(path).read_bytes()).hexdigest()
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique_images.append((path, label))
        images = unique_images

        random.shuffle(images)

        n = len(images)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_files.extend(images[:train_end])
        val_files.extend(images[train_end:val_end])
        test_files.extend(images[val_end:])

    return train_files, val_files, test_files


# ---------------------------------------------------------------
# 3. Image features (colour histogram)
# ---------------------------------------------------------------
IMG_SIZE = (128, 128)   # all images are resized to this before features
HSV_BINS = (8, 4, 4)    # hue, saturation, value bins -> 8*4*4 = 128 features


def load_image(path_or_file, size=IMG_SIZE):
    """Open an image (file path or Streamlit upload) as an RGB array."""
    img = Image.open(path_or_file).convert("RGB").resize(size)
    return np.asarray(img)


def color_histogram(img_rgb, bins=HSV_BINS):
    """HSV colour histogram -> 128 numbers that sum to 1."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    hist = hist.flatten()
    return hist / hist.sum()


def channel_histograms(img_rgb, bins=32):
    """One histogram per R, G, B channel (for plots)."""
    return np.stack([
        np.histogram(img_rgb[..., c], bins=bins, range=(0, 256), density=True)[0]
        for c in range(3)
    ])


def basic_stats(img_rgb):
    """Brightness, contrast and saturation of one image."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return {
        "brightness": gray.mean(),
        "contrast": gray.std(),
        "saturation": hsv[..., 1].mean(),
    }


def average_hash(path, hash_size=16):
    """Fingerprint of how an image looks; near-identical photos get the same hash."""
    img = Image.open(path).convert("L").resize((hash_size, hash_size))
    px = np.asarray(img, dtype=np.float32)
    return "".join("1" if b else "0" for b in (px > px.mean()).flatten())


def extract_features(paths, fn=color_histogram):
    """Run a feature function on many images -> one row per image."""
    return np.stack([fn(load_image(p)) for p in paths])


# ---------------------------------------------------------------
# Quick check: run with  python -m src.data_utils
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Dataset found:", DATASET_PATH.exists())
    if DATASET_PATH.exists():
        categories = sorted(f.name for f in DATASET_PATH.iterdir() if f.is_dir())
        print("Categories:", categories)

        train, val, test = split_dataset()
        print("Training images:", len(train))
        print("Validation images:", len(val))
        print("Testing images:", len(test))
        print("Total images:", len(train) + len(val) + len(test))
