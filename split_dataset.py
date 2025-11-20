import random
import shutil
from pathlib import Path

# paths
RAW_IMG = Path("data/raw/images")
RAW_LBL = Path("data/raw/labels")

DEST = Path("data")
TRAIN = DEST / "train"
VAL = DEST / "validation"
TEST = DEST / "test"


def create_folders():
    for folder in [TRAIN, VAL, TEST]:
        (folder / "images").mkdir(parents=True, exist_ok=True)
        (folder / "labels").mkdir(parents=True, exist_ok=True)


def split_dataset(train_ratio=0.7, val_ratio=0.15):
    imgs = list(RAW_IMG.glob("*.jpg")) + list(RAW_IMG.glob("*.png"))
    random.shuffle(imgs)

    n = len(imgs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    sets = [("train", imgs[:train_end]), ("validation", imgs[train_end:val_end]), ("test", imgs[val_end:])]

    for name, files in sets:
        for img_path in files:
            lbl_path = RAW_LBL / (img_path.stem + ".txt")
            dest_img = DEST / name / "images" / img_path.name
            dest_lbl = DEST / name / "labels" / lbl_path.name

            shutil.copy(img_path, dest_img)
            if lbl_path.exists():
                shutil.copy(lbl_path, dest_lbl)


if __name__ == "__main__":
    print("Creating folder structure...")
    create_folders()
    print("Splitting dataset...")
    split_dataset()
    print("Done! Train/Val/Test sets generated.")
