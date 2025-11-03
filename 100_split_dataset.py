# Split dataset in datasets/100%/raw into train/val (80/20) with images/labels folders
import random
import shutil
from pathlib import Path

# Configuration
base_dir = Path("datasets") / "real_100"  # root for this dataset variant
raw_dir = base_dir / "raw"
train_dir = base_dir / "train"
val_dir = base_dir / "val"
train_ratio = 0.80
force_overwrite = False  # set True to re-create splits
copy_files = True  # set False to move instead of copy
random_seed = 42
random.seed(random_seed)

# Helper: create directory tree
for split_root in [train_dir, val_dir]:
    for sub in ["images", "labels"]:
        (split_root / sub).mkdir(parents=True, exist_ok=True)

# Collect image files (common extensions)
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
images = [p for p in raw_dir.glob("*") if p.suffix.lower() in image_exts]
images.sort()

if not images:
    raise FileNotFoundError(f"No images found in {raw_dir}.")

# If not forcing overwrite, detect existing split
if not force_overwrite:
    existing_train = list((train_dir / "images").glob("*"))
    existing_val = list((val_dir / "images").glob("*"))
    if existing_train or existing_val:
        print("⚠️ Split appears to already exist. Set force_overwrite=True to regenerate.")
        print(f"Train images: {len(existing_train)}, Val images: {len(existing_val)}")
        # Exit early to avoid duplicating
        raise SystemExit

# Clear existing if force_overwrite
if force_overwrite:
    for split_root in [train_dir, val_dir]:
        for sub in ["images", "labels"]:
            target = split_root / sub
            if target.exists():
                for f in target.glob("*"):
                    f.unlink()

# Shuffle and split
random.shuffle(images)
split_index = int(len(images) * train_ratio)
train_images = images[:split_index]
val_images = images[split_index:]

print(f"Total images: {len(images)}")
print(f"Train: {len(train_images)}  Val: {len(val_images)}  (ratio {train_ratio:.2f})")

missing_labels = 0


# Function to process one list
def distribute(img_paths, split_name, dest_root):
    global missing_labels
    for img_path in img_paths:
        label_path = img_path.with_suffix(".txt")
        # Destination paths
        dest_img = dest_root / "images" / img_path.name
        dest_lbl = dest_root / "labels" / label_path.name
        # Copy/move image
        if copy_files:
            shutil.copy2(img_path, dest_img)
        else:
            shutil.move(str(img_path), dest_img)
        # Copy/move label if exists
        if label_path.exists():
            if copy_files:
                shutil.copy2(label_path, dest_lbl)
            else:
                shutil.move(str(label_path), dest_lbl)
        else:
            missing_labels += 1
            print(f"⚠️ Missing label for {img_path.name}")


# Distribute
distribute(train_images, "train", train_dir)
distribute(val_images, "val", val_dir)

print("✅ Split complete.")
print(f"Missing labels: {missing_labels}")
print("Train images path:", train_dir / "images")
print("Val images path:", val_dir / "images")
