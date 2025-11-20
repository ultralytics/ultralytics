import shutil
from pathlib import Path

import cv2

RAW = Path("data/raw/images")
CLEANED = Path("data/processed/cleaned_images")
CLEANED.mkdir(parents=True, exist_ok=True)


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def is_blurry(image_path, threshold=60.0):
    img = cv2.imread(str(image_path))
    if img is None:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm < threshold


for img_path in RAW.glob("*.*"):
    if is_blurry(img_path):
        print(f"Removing blurry image: {img_path.name}")
        continue
    shutil.copy(img_path, CLEANED / img_path.name)

print("Cleaning complete!")
