import os
import shutil

from sklearn.model_selection import train_test_split


def split_dataset(source_path="datasets/raw", base_path="datasets/mydata"):
    # Create YOLO folder structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_path, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_path, "labels", split), exist_ok=True)

    # Collect all images
    images = [f for f in os.listdir(source_path) if f.endswith((".jpg", ".png"))]

    # Train/Val/Test split (80/10/10)
    train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

    # Move images + labels
    for split, files in splits.items():
        for img in files:
            label = os.path.splitext(img)[0] + ".txt"

            shutil.copy(os.path.join(source_path, img), os.path.join(base_path, "images", split, img))

            if os.path.exists(os.path.join(source_path, label)):
                shutil.copy(os.path.join(source_path, label), os.path.join(base_path, "labels", split, label))

    print(f"✅ Dataset split completed. Output at {base_path}")


if __name__ == "__main__":
    split_dataset()
