# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate COCO12 dataset for testing all supported image formats.

This script creates a test dataset with images in all 12 supported format extensions:
    avif, bmp, dng, heic, jpeg, jpg, mpo, pfm, png, tif, tiff, webp

Usage:
    python generate_coco12.py

Requirements:
    pip install pillow pillow-heif pillow-avif-plugin numpy
"""

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# Supported formats from ultralytics/data/utils.py (alphabetically sorted)
# IMG_FORMATS = {"avif", "bmp", "dng", "heic", "jpeg", "jpg", "mpo", "pfm", "png", "tif", "tiff", "webp"}

# Format assignments: 6 train + 6 val = 12 total
TRAIN_FORMATS = ["avif", "bmp", "dng", "heic", "jpeg", "jpg"]
VAL_FORMATS = ["mpo", "pfm", "png", "tif", "tiff", "webp"]


def write_pfm(path: Path, image: np.ndarray, scale: float = 1.0):
    """Write a PFM (Portable FloatMap) file."""
    if image.ndim == 2:
        color = False
    elif image.ndim == 3 and image.shape[2] == 3:
        color = True
    else:
        raise ValueError(f"Image must be 2D grayscale or 3D RGB, got shape {image.shape}")

    image = image.astype(np.float32)
    height, width = image.shape[:2]

    with open(path, "wb") as f:
        f.write(b"PF\n" if color else b"Pf\n")
        f.write(f"{width} {height}\n".encode())
        endian = -1.0 if np.little_endian else 1.0
        f.write(f"{endian * scale}\n".encode())
        # PFM stores rows bottom-to-top
        image = np.flipud(image)
        image.tofile(f)


def write_mpo(path: Path, img: Image.Image):
    """Write an MPO (Multi-Picture Object) file - saves as JPEG with MPO extension."""
    # MPO is essentially JPEG with additional metadata for stereo pairs
    # For testing purposes, we save as JPEG with .mpo extension
    img.save(path, "JPEG", quality=95)


def write_dng(path: Path, img: Image.Image):
    """Write a minimal DNG file for testing.

    Note: Creating a proper DNG requires complex TIFF/DNG specification compliance.
    For testing purposes, we create a simple TIFF-based file with .dng extension
    that can be read by most image libraries.
    """
    # Save as TIFF with DNG extension (DNG is based on TIFF format)
    # This creates a readable file for testing the format extension handling
    img.save(path, "TIFF")


def convert_image(src_path: Path, dst_path: Path, fmt: str):
    """Convert an image to the specified format."""
    img = Image.open(src_path)

    # Convert to RGB if necessary (some formats don't support RGBA)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")

    fmt_lower = fmt.lower()

    if fmt_lower in ("jpg", "jpeg"):
        img.save(dst_path, "JPEG", quality=95)
    elif fmt_lower == "png":
        img.save(dst_path, "PNG")
    elif fmt_lower == "bmp":
        img.save(dst_path, "BMP")
    elif fmt_lower in ("tif", "tiff"):
        img.save(dst_path, "TIFF")
    elif fmt_lower == "webp":
        img.save(dst_path, "WEBP", quality=95)
    elif fmt_lower == "pfm":
        arr = np.array(img).astype(np.float32) / 255.0
        write_pfm(dst_path, arr)
    elif fmt_lower == "mpo":
        write_mpo(dst_path, img)
    elif fmt_lower == "dng":
        write_dng(dst_path, img)
    elif fmt_lower == "heic":
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
            img.save(dst_path, "HEIF", quality=95)
        except ImportError:
            print(f"Warning: pillow-heif not installed, saving {dst_path} as JPEG-based HEIC")
            # Fallback: save as JPEG with .heic extension for basic testing
            img.save(dst_path.with_suffix(".heic.jpg"), "JPEG", quality=95)
            shutil.move(dst_path.with_suffix(".heic.jpg"), dst_path)
    elif fmt_lower == "avif":
        try:
            import pillow_avif  # noqa: F401 - registers AVIF plugin

            img.save(dst_path, "AVIF", quality=95)
        except ImportError:
            print(f"Warning: pillow-avif-plugin not installed, saving {dst_path} as JPEG-based AVIF")
            # Fallback: save as JPEG with .avif extension for basic testing
            img.save(dst_path.with_suffix(".avif.jpg"), "JPEG", quality=95)
            shutil.move(dst_path.with_suffix(".avif.jpg"), dst_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def generate_coco12(
    output_dir: str | Path = None,
    coco8_dir: str | Path = None,
    coco128_dir: str | Path = None,
):
    """
    Generate the COCO12 dataset with all 12 supported image formats.

    Args:
        output_dir: Output directory for coco12 dataset
        coco8_dir: Path to coco8 dataset (will download if not exists)
        coco128_dir: Path to coco128 dataset (will download if not exists)
    """
    from ultralytics.utils import DATASETS_DIR
    from ultralytics.utils.downloads import safe_download, unzip_file

    # Set default paths
    if output_dir is None:
        output_dir = DATASETS_DIR / "coco12"
    output_dir = Path(output_dir)

    if coco8_dir is None:
        coco8_dir = DATASETS_DIR / "coco8"
    coco8_dir = Path(coco8_dir)

    if coco128_dir is None:
        coco128_dir = DATASETS_DIR / "coco128"
    coco128_dir = Path(coco128_dir)

    # Download datasets if needed
    if not coco8_dir.exists():
        print("Downloading coco8 dataset...")
        zip_path = safe_download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip")
        unzip_file(zip_path, DATASETS_DIR)

    if not coco128_dir.exists():
        print("Downloading coco128 dataset...")
        zip_path = safe_download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip")
        unzip_file(zip_path, DATASETS_DIR)

    # Create output directories
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get source images from coco8
    coco8_train_imgs = sorted((coco8_dir / "images" / "train").glob("*.jpg"))
    coco8_val_imgs = sorted((coco8_dir / "images" / "val").glob("*.jpg"))
    coco8_imgs = coco8_train_imgs + coco8_val_imgs

    # Get additional images from coco128
    coco128_imgs = sorted((coco128_dir / "images" / "train2017").glob("*.jpg"))
    # Filter out images already in coco8
    coco8_stems = {p.stem for p in coco8_imgs}
    coco128_extra = [p for p in coco128_imgs if p.stem not in coco8_stems][:4]

    # Combine: 8 from coco8 + 4 from coco128 = 12 images
    all_source_imgs = coco8_imgs + coco128_extra

    if len(all_source_imgs) < 12:
        raise RuntimeError(f"Need 12 source images, found only {len(all_source_imgs)}")

    print(f"Using {len(all_source_imgs)} source images")

    # Process train images (6 formats)
    print("\nGenerating train images:")
    for i, fmt in enumerate(TRAIN_FORMATS):
        src_img = all_source_imgs[i]
        dst_img = train_img_dir / f"{src_img.stem}.{fmt}"

        print(f"  {src_img.name} -> {dst_img.name}")
        convert_image(src_img, dst_img, fmt)

        # Copy corresponding label
        src_lbl = find_label(src_img, coco8_dir, coco128_dir)
        if src_lbl and src_lbl.exists():
            dst_lbl = train_lbl_dir / f"{src_img.stem}.txt"
            shutil.copy(src_lbl, dst_lbl)

    # Process val images (6 formats)
    print("\nGenerating val images:")
    for i, fmt in enumerate(VAL_FORMATS):
        src_img = all_source_imgs[len(TRAIN_FORMATS) + i]
        dst_img = val_img_dir / f"{src_img.stem}.{fmt}"

        print(f"  {src_img.name} -> {dst_img.name}")
        convert_image(src_img, dst_img, fmt)

        # Copy corresponding label
        src_lbl = find_label(src_img, coco8_dir, coco128_dir)
        if src_lbl and src_lbl.exists():
            dst_lbl = val_lbl_dir / f"{src_img.stem}.txt"
            shutil.copy(src_lbl, dst_lbl)

    print(f"\nDataset generated at: {output_dir}")
    print(f"  Train images: {len(list(train_img_dir.iterdir()))}")
    print(f"  Val images: {len(list(val_img_dir.iterdir()))}")

    return output_dir


def find_label(img_path: Path, coco8_dir: Path, coco128_dir: Path) -> Path | None:
    """Find the label file for a given image."""
    stem = img_path.stem

    # Check coco8 labels
    for subdir in ["train", "val"]:
        lbl = coco8_dir / "labels" / subdir / f"{stem}.txt"
        if lbl.exists():
            return lbl

    # Check coco128 labels
    lbl = coco128_dir / "labels" / "train2017" / f"{stem}.txt"
    if lbl.exists():
        return lbl

    return None


if __name__ == "__main__":
    generate_coco12()
