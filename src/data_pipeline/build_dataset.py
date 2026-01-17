import cv2
import os
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
# -------------------------------
# CONFIG
# -------------------------------
RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"
RESIZED_DIR = "data/resized"

IMAGE_SIZE = (224, 224)
MIN_WIDTH = 100
MIN_HEIGHT = 100
BLUR_THRESHOLD = 50  # Lower = more tolerant

CLASSES = ["good", "defective"]

# -------------------------------
# UTILS
# -------------------------------
def is_blurry(image, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def create_dirs():
    for base in [CLEANED_DIR, RESIZED_DIR]:
        for cls in CLASSES:
            Path(f"{base}/{cls}").mkdir(parents=True, exist_ok=True)


# -------------------------------
# STEP 1: CLEAN DATA
# -------------------------------
def clean_images():
    print(" Cleaning raw images...")

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

                h, w, _ = img.shape

                if h < MIN_HEIGHT or w < MIN_WIDTH:
                    continue

                if is_blurry(img):
                    continue

                shutil.copy(src_path, dst_path)

            except Exception:
                continue


# -------------------------------
# STEP 2: RESIZE + NORMALIZE
# -------------------------------
def resize_and_normalize():
    print(" Resizing and normalizing images...")

    for cls in CLASSES:
        src_dir = f"{CLEANED_DIR}/{cls}"
        dst_dir = f"{RESIZED_DIR}/{cls}"

        for img_name in tqdm(os.listdir(src_dir), desc=f"Resizing {cls}"):
            src_path = os.path.join(src_dir, img_name)
            dst_path = os.path.join(dst_dir, img_name)

            img = cv2.imread(src_path)
            img = cv2.resize(img, IMAGE_SIZE)

            # Normalize to 0–1
            img = img.astype("float32") / 255.0

            np.save(dst_path.replace(".jpg", ""), img)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    create_dirs()
    clean_images()
    resize_and_normalize()

    print(" Dataset build complete.")
