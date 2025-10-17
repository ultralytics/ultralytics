# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Utilities to download the KITTI Object dataset and convert it to YOLO MDE format."""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image

from ultralytics.utils import DATASETS_DIR, LOGGER, TQDM
from ultralytics.utils.downloads import download

KITTI_OBJECT_URLS = (
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
)
KITTI_CLASS_MAP = {"Car": 0, "Pedestrian": 1, "Cyclist": 2, "Van": 3, "Truck": 4}
KITTI_RAW_DIR_NAME = "kitti_object_raw"
KITTI_DEPTH_MAX_DEFAULT = 100.0


def prepare_kitti_mde_dataset(
    dataset_dir: str | Path,
    *,
    depth_max: float = KITTI_DEPTH_MAX_DEFAULT,
    train_ratio: float = 0.8,
    download_assets: bool = True,
    overwrite: bool = False,
    shuffle: bool = False,
    seed: int | None = 0,
) -> Dict[str, Dict[str, int]]:
    """Download the KITTI Object dataset and convert labels to YOLO MDE format.

    Args:
        dataset_dir: Target directory for the prepared dataset.
        depth_max: Maximum depth value used to normalize depth annotations.
        train_ratio: Proportion of images assigned to the training split (0-1).
        download_assets: Whether to download the KITTI Object dataset assets.
        overwrite: Re-create the dataset even if it already exists.
        shuffle: Shuffle image list before splitting into train/val.
        seed: Random seed used when shuffling images.
    """
    dataset_dir = _resolve_dataset_dir(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not overwrite and _dataset_exists(dataset_dir):
        LOGGER.info(f"KITTI MDE dataset already exists at {dataset_dir}. Skipping conversion.")
        return _summarize_existing_dataset(dataset_dir)

    raw_dir = dataset_dir.parent / KITTI_RAW_DIR_NAME
    image_dir, label_dir = _ensure_kitti_raw(raw_dir, download_assets=download_assets)

    image_files = sorted(image_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No KITTI images found in {image_dir}.")

    if shuffle and seed is not None:
        rng = random.Random(seed)
        rng.shuffle(image_files)

    train_count = int(len(image_files) * train_ratio)
    if train_count <= 0 or train_count >= len(image_files):
        train_count = len(image_files) - 1
    train_ids = image_files[:train_count]
    val_ids = image_files[train_count:]

    summary = {
        "train": _prepare_subset(dataset_dir, "train", train_ids, label_dir, depth_max, overwrite),
        "val": _prepare_subset(dataset_dir, "val", val_ids, label_dir, depth_max, overwrite),
    }
    summary["depth_max"] = depth_max
    summary["dataset_dir"] = str(dataset_dir)

    LOGGER.info(
        "KITTI MDE dataset prepared at %s (train: %d images, val: %d images).",
        dataset_dir,
        summary["train"]["images"],
        summary["val"]["images"],
    )
    return summary


def _resolve_dataset_dir(dataset_dir: str | Path) -> Path:
    path = Path(dataset_dir)
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()
    return path


def _dataset_exists(dataset_dir: Path) -> bool:
    train_images = dataset_dir / "train" / "images"
    val_images = dataset_dir / "val" / "images"
    return train_images.exists() and any(train_images.glob("*")) and val_images.exists() and any(val_images.glob("*"))


def _summarize_existing_dataset(dataset_dir: Path) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val"):
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        image_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
        object_count = 0
        class_counts: Counter = Counter()
        if labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                with label_file.open("r", encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            class_id = int(parts[0])
                            object_count += 1
                            class_counts[class_id] += 1
        summary[split] = {
            "images": image_count,
            "objects": object_count,
            "class_counts": dict(sorted(class_counts.items())),
        }
    summary["depth_max"] = KITTI_DEPTH_MAX_DEFAULT
    summary["dataset_dir"] = str(dataset_dir)
    return summary


def _ensure_kitti_raw(raw_dir: Path, download_assets: bool) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    image_dir = _find_training_subdir(raw_dir, "image_2")
    label_dir = _find_training_subdir(raw_dir, "label_2")

    if download_assets or image_dir is None or label_dir is None:
        LOGGER.info("Downloading KITTI Object dataset assets...")
        download(KITTI_OBJECT_URLS, dir=raw_dir, unzip=True, exist_ok=True)
        image_dir = _find_training_subdir(raw_dir, "image_2")
        label_dir = _find_training_subdir(raw_dir, "label_2")

    if image_dir is None or label_dir is None:
        raise FileNotFoundError(
            "Failed to locate KITTI object dataset assets. Please ensure the archives are available in "
            f"{raw_dir} or enable download_assets."
        )

    return image_dir, label_dir


def _find_training_subdir(root: Path, name: str) -> Path | None:
    candidates = [p for p in root.glob(f"**/training/{name}") if p.is_dir()]
    if not candidates:
        return None
    return min(candidates, key=lambda p: len(p.parts))


def _prepare_subset(
    dataset_dir: Path,
    split: str,
    image_paths: Iterable[Path],
    label_dir: Path,
    depth_max: float,
    overwrite: bool,
) -> Dict[str, int]:
    split_dir = dataset_dir / split
    images_out = split_dir / "images"
    labels_out = split_dir / "labels"

    if overwrite and split_dir.exists():
        shutil.rmtree(split_dir)

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    object_count = 0
    class_counts: Counter = Counter()

    image_paths_list = list(image_paths)
    for image_path in TQDM(image_paths_list, desc=f"Preparing KITTI {split}", unit="image"):
        dest_image = images_out / image_path.name
        if overwrite or not dest_image.exists():
            shutil.copy2(image_path, dest_image)

        with Image.open(image_path) as im:
            width, height = im.size

        label_path = label_dir / f"{image_path.stem}.txt"
        label_lines, counts = _convert_kitti_label(label_path, width, height, depth_max)
        dest_label = labels_out / f"{image_path.stem}.txt"
        dest_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        object_count += sum(counts.values())
        class_counts.update(counts)

    return {
        "images": len(image_paths_list),
        "objects": object_count,
        "class_counts": dict(sorted(class_counts.items())),
    }


def _convert_kitti_label(
    label_path: Path,
    width: int,
    height: int,
    depth_max: float,
) -> tuple[List[str], Counter]:
    lines: List[str] = []
    counts: Counter = Counter()

    if not label_path.exists():
        return lines, counts

    with label_path.open("r", encoding="utf-8") as lf:
        for raw_line in lf:
            parts = raw_line.strip().split()
            if len(parts) < 15:
                continue

            class_name = parts[0]
            if class_name not in KITTI_CLASS_MAP:
                continue

            bbox_left, bbox_top, bbox_right, bbox_bottom = map(float, parts[4:8])
            bbox_left = max(0.0, min(bbox_left, width))
            bbox_top = max(0.0, min(bbox_top, height))
            bbox_right = max(0.0, min(bbox_right, width))
            bbox_bottom = max(0.0, min(bbox_bottom, height))

            bbox_width = bbox_right - bbox_left
            bbox_height = bbox_bottom - bbox_top
            if bbox_width <= 0 or bbox_height <= 0:
                continue

            x_center = ((bbox_left + bbox_right) / 2) / width
            y_center = ((bbox_top + bbox_bottom) / 2) / height
            w_norm = bbox_width / width
            h_norm = bbox_height / height

            depth = float(parts[13])
            if depth <= 0:
                continue
            depth_norm = min(depth / depth_max, 1.0)

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            class_id = KITTI_CLASS_MAP[class_name]
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {depth_norm:.6f}")
            counts[class_id] += 1

    return lines, counts


__all__ = ("prepare_kitti_mde_dataset",)
