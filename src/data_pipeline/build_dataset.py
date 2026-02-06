import cv2
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# -------------------------------
# CONFIG
# -------------------------------
RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"
RESIZED_DIR = "data/resized"
SPLITS_DIR = "data/splits"

IMAGE_SIZE = (224, 224)
MIN_WIDTH = 100
MIN_HEIGHT = 100
BLUR_THRESHOLD = 50  # Lower = more tolerant

CLASSES = ["good", "defective"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
random.seed(SEED)

# -------------------------------
# UTILS
# -------------------------------
def is_blurry(image, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def create_dirs():
    # cleaned + resized
    for base in [CLEANED_DIR, RESIZED_DIR]:
        for cls in CLASSES:
            Path(f"{base}/{cls}").mkdir(parents=True, exist_ok=True)

    # splits
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            Path(f"{SPLITS_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)


# -------------------------------
# STEP 1: CLEAN DATA
# -------------------------------
def clean_images():
    print("Cleaning raw images...")

    for cls in CLASSES:
        src_dir = f"{RAW_DIR}/{cls}"
        dst_dir = f"{CLEANED_DIR}/{cls}"

        for img_name in tqdm(os.listdir(src_dir), desc=f"Cleaning {cls}"):
            src_path = os.path.join(src_dir, img_name)
            dst_path = os.path.join(dst_dir, img_name)

            try:
                img = cv2.imread(src_path)

                if img is None:
                    continue

                h, w = img.shape[:2]

                if h < MIN_HEIGHT or w < MIN_WIDTH:
                    continue

                if is_blurry(img):
                    continue

                shutil.copy(src_path, dst_path)

            except Exception:
                continue


# -------------------------------
# STEP 2: RESIZE (NO NORMALIZATION HERE)
# -------------------------------
def resize_images():
    print("Resizing images...")

    for cls in CLASSES:
        src_dir = f"{CLEANED_DIR}/{cls}"
        dst_dir = f"{RESIZED_DIR}/{cls}"

        for img_name in tqdm(os.listdir(src_dir), desc=f"Resizing {cls}"):
            src_path = os.path.join(src_dir, img_name)
            dst_path = os.path.join(dst_dir, img_name)

            img = cv2.imread(src_path)
            if img is None:
                continue

            img = cv2.resize(img, IMAGE_SIZE)

            # Save as image (uint8, 0–255)
            cv2.imwrite(dst_path, img)


# -------------------------------
# STEP 3: SPLIT INTO TRAIN / VAL / TEST
# -------------------------------
def split_dataset():
    print("Splitting dataset into train / val / test...")

    # Clear existing splits (optional but safer)
    if os.path.exists(SPLITS_DIR):
        shutil.rmtree(SPLITS_DIR)

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            Path(f"{SPLITS_DIR}/{split}/{cls}").mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        src_dir = f"{RESIZED_DIR}/{cls}"
        images = os.listdir(src_dir)

        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        for split, files in splits.items():
            dst_dir = f"{SPLITS_DIR}/{split}/{cls}"
            for fname in files:
                shutil.copy(
                    os.path.join(src_dir, fname),
                    os.path.join(dst_dir, fname)
                )

        print(f"Class '{cls}': {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    create_dirs()
    clean_images()
    resize_images()
    split_dataset()

    print("Dataset build complete!")
