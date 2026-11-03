from pathlib import Path

# Main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# TrashNet dataset location
DATASET_PATH = PROJECT_ROOT / "data" / "Trashnet" / "dataset-resized"

# Check dataset location
print("Dataset found:", DATASET_PATH.exists())

# Check category folders
if DATASET_PATH.exists():
    categories = sorted(
        folder.name
        for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    )
    print("Categories:", categories)




# Train/Validation/Test split 
    import random

def split_dataset(dataset_path=DATASET_PATH, seed=42):
    random.seed(seed)

    train_files = []
    val_files = []
    test_files = []

    for category in sorted(dataset_path.iterdir()):
        if not category.is_dir():
            continue

        images = [
            (str(image), category.name)
            for image in sorted(category.iterdir())
            if image.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        random.shuffle(images)

        n = len(images)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_files.extend(images[:train_end])
        val_files.extend(images[train_end:val_end])
        test_files.extend(images[val_end:])

    return train_files, val_files, test_files


if __name__ == "__main__":
    train, val, test = split_dataset()

    print("Training images:", len(train))
    print("Validation images:", len(val))
    print("Testing images:", len(test))
    print("Total images:", len(train) + len(val) + len(test))