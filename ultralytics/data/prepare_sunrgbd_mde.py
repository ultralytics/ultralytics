# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Utilities to download the SUN RGB-D dataset and convert it to YOLO MDE format."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from ultralytics.utils import DATASETS_DIR, LOGGER, TQDM
from ultralytics.utils.downloads import download

# SUN RGB-D dataset URLs
SUNRGBD_URLS = (
    "http://rgbd.cs.princeton.edu/data/SUNRGBD.zip",
    "http://rgbd.cs.princeton.edu/data/SUNRGBDMeta2DBB_v2.mat",
    "http://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat",
    "http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip",
)

# SUN RGB-D 37 object classes - most common classes for object-level depth estimation
SUNRGBD_CLASS_MAP = {
    "bed": 0,
    "table": 1,
    "sofa": 2,
    "chair": 3,
    "toilet": 4,
    "desk": 5,
    "dresser": 6,
    "night_stand": 7,
    "bookshelf": 8,
    "bathtub": 9,
    "person": 10,
    "cabinet": 11,
    "box": 12,
    "pillow": 13,
    "door": 14,
    "tv": 15,
    "lamp": 16,
    "bag": 17,
    "computer": 18,
    "monitor": 19,
    "bin": 20,
    "sink": 21,
    "books": 22,
    "curtain": 23,
    "mirror": 24,
    "floor_mat": 25,
    "clothes": 26,
    "ceiling": 27,
    "book": 28,
    "fridge": 29,
    "window": 30,
    "blinds": 31,
    "shelves": 32,
    "picture": 33,
    "counter": 34,
    "floor": 35,
    "wall": 36,
}

SUNRGBD_RAW_DIR_NAME = "sunrgbd_raw"
SUNRGBD_DEPTH_MAX_DEFAULT = 10.0  # SUN RGB-D depth is typically in meters (0-10m range)

# Baidu AI Studio dataset info
BAIDU_DATASET_ID = "71231/cpEKg8sF"


def _is_china_network() -> bool:
    """Detect if the system is in China based on network connectivity."""
    import socket
    import urllib.request
    
    # Test 1: Try to access a China-specific site (fast check)
    try:
        socket.create_connection(("www.baidu.com", 80), timeout=2)
        LOGGER.info("Detected China network environment (Baidu accessible).")
        return True
    except (socket.timeout, OSError):
        pass
    
    # Test 2: Try to access Princeton site (if slow/fails, likely in China)
    try:
        req = urllib.request.Request("http://rgbd.cs.princeton.edu/", method="HEAD")
        with urllib.request.urlopen(req, timeout=5) as response:
            # If Princeton site is accessible and fast, likely not in China
            LOGGER.info("Detected non-China network environment (Princeton accessible).")
            return False
    except (urllib.error.URLError, socket.timeout):
        # Princeton site not accessible/slow, likely in China
        LOGGER.info("Detected China network environment (Princeton not accessible).")
        return True
    
    # Default to False if uncertain
    return False


def _install_aistudio_sdk() -> bool:
    """Install aistudio-sdk for Baidu downloads."""
    try:
        import importlib.util
        # Check if aistudio_sdk module is available
        if importlib.util.find_spec("aistudio_sdk") is not None:
            LOGGER.info("aistudio-sdk is already installed.")
            return True
        
        LOGGER.info("Installing aistudio-sdk for Baidu downloads (this may take a moment)...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "aistudio-sdk"],
            capture_output=True,
            text=True,
            check=True,
        )
        LOGGER.info("aistudio-sdk installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.warning(f"Failed to install aistudio-sdk: {e}")
        if e.stderr:
            LOGGER.warning(f"Error details: {e.stderr}")
        return False
    except Exception as e:
        LOGGER.warning(f"Unexpected error installing aistudio-sdk: {e}")
        return False


def _download_from_baidu(raw_dir: Path) -> bool:
    """Download SUN RGB-D dataset from Baidu AI Studio.
    
    Note: aistudio-sdk must be installed before calling this function.
    """
    try:
        LOGGER.info(f"Downloading SUN RGB-D dataset from Baidu AI Studio (dataset: {BAIDU_DATASET_ID})...")
        LOGGER.info("This may take a while, please be patient...")
        
        # Use aistudio CLI to download dataset
        # Note: Do NOT capture_output=True as it will buffer and appear to hang
        # Let the output stream directly to console so user can see progress
        cmd = [
            "aistudio",
            "download",
            "--dataset",
            BAIDU_DATASET_ID,
            "--local_dir",
            str(raw_dir),
        ]
        
        # Run without capturing output to show progress
        result = subprocess.run(cmd, check=True)
        LOGGER.info("Successfully downloaded SUN RGB-D dataset from Baidu AI Studio.")
        
        # Check if download was successful by looking for the dataset
        if _find_sunrgbd_dir(raw_dir) is not None:
            return True
        else:
            LOGGER.warning("Download completed but could not locate SUN RGB-D directory.")
            return False
            
    except subprocess.CalledProcessError as e:
        LOGGER.warning(f"Failed to download from Baidu AI Studio: {e}")
        return False
    except Exception as e:
        LOGGER.warning(f"Unexpected error during Baidu download: {e}")
        return False


def prepare_sunrgbd_mde_dataset(
    dataset_dir: str | Path,
    *,
    depth_max: float = SUNRGBD_DEPTH_MAX_DEFAULT,
    train_ratio: float = 0.8,
    download_assets: bool = True,
    overwrite: bool = False,
    shuffle: bool = True,
    seed: int | None = 0,
    use_existing_splits: bool = True,
    min_box_size: int = 10,
) -> Dict[str, Dict[str, int]]:
    """Download the SUN RGB-D dataset and convert labels to YOLO MDE format.

    Args:
        dataset_dir: Target directory for the prepared dataset.
        depth_max: Maximum depth value used to normalize depth annotations (in meters).
        train_ratio: Proportion of images assigned to the training split (0-1). Ignored if use_existing_splits=True.
        download_assets: Whether to download the SUN RGB-D dataset assets.
        overwrite: Re-create the dataset even if it already exists.
        shuffle: Shuffle image list before splitting into train/val (only if use_existing_splits=False).
        seed: Random seed used when shuffling images.
        use_existing_splits: Use the official SUN RGB-D train/test splits if available.
        min_box_size: Minimum bounding box size (width or height in pixels) to include.

    Returns:
        Dictionary with dataset statistics including image counts and object counts per class.
    """
    dataset_dir = _resolve_dataset_dir(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not overwrite and _dataset_exists(dataset_dir):
        LOGGER.info(f"SUN RGB-D MDE dataset already exists at {dataset_dir}. Skipping conversion.")
        return _summarize_existing_dataset(dataset_dir)

    raw_dir = dataset_dir.parent / SUNRGBD_RAW_DIR_NAME
    sunrgbd_dir = _ensure_sunrgbd_raw(raw_dir, download_assets=download_assets)

    # Get train/val splits
    train_samples, val_samples = _get_dataset_splits(
        sunrgbd_dir, use_existing_splits, train_ratio, shuffle, seed
    )

    if not train_samples and not val_samples:
        raise FileNotFoundError(f"No valid SUN RGB-D samples found in {sunrgbd_dir}.")

    summary = {
        "train": _prepare_subset(
            dataset_dir, "train", train_samples, depth_max, min_box_size, overwrite
        ),
        "val": _prepare_subset(dataset_dir, "val", val_samples, depth_max, min_box_size, overwrite),
    }
    summary["depth_max"] = depth_max
    summary["dataset_dir"] = str(dataset_dir)

    LOGGER.info(
        "SUN RGB-D MDE dataset prepared at %s (train: %d images, val: %d images).",
        dataset_dir,
        summary["train"]["images"],
        summary["val"]["images"],
    )
    return summary


def _resolve_dataset_dir(dataset_dir: str | Path) -> Path:
    """Resolve dataset directory to absolute path."""
    path = Path(dataset_dir)
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()
    return path


def _dataset_exists(dataset_dir: Path) -> bool:
    """Check if the dataset already exists."""
    train_images = dataset_dir / "train" / "images"
    val_images = dataset_dir / "val" / "images"
    return (
        train_images.exists()
        and any(train_images.glob("*"))
        and val_images.exists()
        and any(val_images.glob("*"))
    )


def _summarize_existing_dataset(dataset_dir: Path) -> Dict[str, Dict[str, int]]:
    """Summarize statistics of an existing dataset."""
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
    summary["depth_max"] = SUNRGBD_DEPTH_MAX_DEFAULT
    summary["dataset_dir"] = str(dataset_dir)
    return summary


def _ensure_sunrgbd_raw(raw_dir: Path, download_assets: bool) -> Path:
    """Ensure raw SUN RGB-D data is available."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    sunrgbd_dir = _find_sunrgbd_dir(raw_dir)

    if download_assets or sunrgbd_dir is None:
        LOGGER.info("Downloading SUN RGB-D dataset assets (this may take a while)...")
        
        # Detect network location and choose download method
        use_baidu = _is_china_network()
        download_success = False
        
        if use_baidu:
            # Install aistudio-sdk first if in China
            if not _install_aistudio_sdk():
                LOGGER.warning("Failed to install aistudio-sdk, falling back to Princeton URLs...")
                use_baidu = False
            else:
                LOGGER.info("Using Baidu AI Studio for download (China network detected)...")
                download_success = _download_from_baidu(raw_dir)
                
                if not download_success:
                    LOGGER.warning("Baidu download failed, falling back to Princeton URLs...")
        
        # If not in China or Baidu download failed, use Princeton URLs
        if not use_baidu or not download_success:
            try:
                download(SUNRGBD_URLS, dir=raw_dir, unzip=True, exist_ok=True)
            except Exception as e:
                LOGGER.error(f"Failed to download from Princeton URLs: {e}")
                if use_baidu:
                    raise FileNotFoundError(
                        f"Both Baidu and Princeton downloads failed. "
                        f"Please manually download the dataset to {raw_dir}"
                    ) from e
                raise
        
        sunrgbd_dir = _find_sunrgbd_dir(raw_dir)

    if sunrgbd_dir is None:
        raise FileNotFoundError(
            f"Failed to locate SUN RGB-D dataset. Please ensure the dataset is available in {raw_dir} "
            "or enable download_assets."
        )

    return sunrgbd_dir


def _find_sunrgbd_dir(root: Path) -> Path | None:
    """Find the main SUN RGB-D directory."""
    # Look for SUNRGBD directory
    candidates = [p for p in root.glob("**/SUNRGBD") if p.is_dir()]
    if not candidates:
        # Alternative: look for directories with 'kv1', 'kv2', 'realsense', or 'xtion' subdirs
        for subdir in root.rglob("*"):
            if subdir.is_dir():
                has_sensor = any((subdir / sensor).exists() for sensor in ["kv1", "kv2", "realsense", "xtion"])
                if has_sensor:
                    return subdir
        return None
    return min(candidates, key=lambda p: len(p.parts))


def _get_dataset_splits(
    sunrgbd_dir: Path,
    use_existing_splits: bool,
    train_ratio: float,
    shuffle: bool,
    seed: Optional[int],
) -> Tuple[List[Dict], List[Dict]]:
    """Get train/val splits for the dataset."""
    # Try to load existing split files
    train_split_file = sunrgbd_dir.parent / "train_data_idx.txt"
    val_split_file = sunrgbd_dir.parent / "val_data_idx.txt"

    all_samples = _scan_sunrgbd_samples(sunrgbd_dir)

    if use_existing_splits and train_split_file.exists() and val_split_file.exists():
        LOGGER.info("Using existing train/val split files...")
        train_indices = set(train_split_file.read_text().strip().split("\n"))
        val_indices = set(val_split_file.read_text().strip().split("\n"))

        train_samples = [s for s in all_samples if str(s["index"]) in train_indices]
        val_samples = [s for s in all_samples if str(s["index"]) in val_indices]
    else:
        LOGGER.info("Creating new train/val split...")
        if shuffle and seed is not None:
            rng = random.Random(seed)
            rng.shuffle(all_samples)

        train_count = int(len(all_samples) * train_ratio)
        if train_count <= 0 or train_count >= len(all_samples):
            train_count = len(all_samples) - 1
        train_samples = all_samples[:train_count]
        val_samples = all_samples[train_count:]

    return train_samples, val_samples


def _scan_sunrgbd_samples(sunrgbd_dir: Path) -> List[Dict]:
    """Scan SUN RGB-D directory for all valid samples."""
    samples = []
    sample_idx = 0

    # SUN RGB-D has subdirectories organized by sensor type
    sensor_dirs = ["kv1", "kv2", "realsense", "xtion"]

    for sensor_dir in sensor_dirs:
        sensor_path = sunrgbd_dir / sensor_dir
        if not sensor_path.exists():
            continue

        # Recursively search for scene directories that contain image/ subdirectories
        # The structure can be: sensor/dataset_name/scene/ or sensor/scene/
        for potential_scene_dir in sensor_path.rglob("*"):
            if not potential_scene_dir.is_dir():
                continue

            # Check if this directory has image/ subdirectory (indicates it's a scene dir)
            if not (potential_scene_dir / "image").exists():
                continue

            # Look for RGB image, depth image, and annotation
            rgb_candidates = list(potential_scene_dir.glob("image/*.jpg")) + list(
                potential_scene_dir.glob("image/*.png")
            )
            depth_candidates = list(potential_scene_dir.glob("depth/*.png")) + list(
                potential_scene_dir.glob("depth_bfx/*.png")
            )
            annotation_candidates = list(potential_scene_dir.glob("annotation2Dfinal/*.json")) + list(
                potential_scene_dir.glob("annotation/*.txt")
            )

            if rgb_candidates and depth_candidates:
                rgb_path = rgb_candidates[0]
                depth_path = depth_candidates[0]
                annotation_path = annotation_candidates[0] if annotation_candidates else None

                samples.append(
                    {
                        "index": sample_idx,
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "annotation_path": annotation_path,
                        "sensor": sensor_dir,
                        "scene": potential_scene_dir.name,
                    }
                )
                sample_idx += 1

    LOGGER.info(f"Found {len(samples)} valid SUN RGB-D samples.")
    return samples


def _prepare_subset(
    dataset_dir: Path,
    split: str,
    samples: Iterable[Dict],
    depth_max: float,
    min_box_size: int,
    overwrite: bool,
) -> Dict[str, int]:
    """Prepare a data subset (train or val)."""
    split_dir = dataset_dir / split
    images_out = split_dir / "images"
    labels_out = split_dir / "labels"

    if overwrite and split_dir.exists():
        shutil.rmtree(split_dir)

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    object_count = 0
    class_counts: Counter = Counter()
    samples_list = list(samples)

    for sample in TQDM(samples_list, desc=f"Preparing SUN RGB-D {split}", unit="image"):
        rgb_path = sample["rgb_path"]
        depth_path = sample["depth_path"]
        annotation_path = sample["annotation_path"]

        # Generate unique filename
        sample_name = f"{sample['sensor']}_{sample['scene']}"
        dest_image = images_out / f"{sample_name}.jpg"
        dest_label = labels_out / f"{sample_name}.txt"

        # Copy RGB image (convert to JPG if PNG)
        if overwrite or not dest_image.exists():
            with Image.open(rgb_path) as img:
                img = img.convert("RGB")
                img.save(dest_image, "JPEG", quality=95)
                width, height = img.size

        else:
            with Image.open(dest_image) as img:
                width, height = img.size

        # Load depth map
        depth_map = None
        if depth_path.exists():
            depth_map = np.array(Image.open(depth_path))

        # Convert annotations
        label_lines, counts = _convert_sunrgbd_annotation(
            annotation_path, depth_map, width, height, depth_max, min_box_size
        )

        dest_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        object_count += sum(counts.values())
        class_counts.update(counts)

    return {
        "images": len(samples_list),
        "objects": object_count,
        "class_counts": dict(sorted(class_counts.items())),
    }


def _convert_sunrgbd_annotation(
    annotation_path: Optional[Path],
    depth_map: Optional[np.ndarray],
    width: int,
    height: int,
    depth_max: float,
    min_box_size: int,
) -> Tuple[List[str], Counter]:
    """Convert SUN RGB-D annotation to YOLO MDE format."""
    lines: List[str] = []
    counts: Counter = Counter()

    if annotation_path is None or not annotation_path.exists():
        return lines, counts

    # Parse annotation based on format
    if annotation_path.suffix == ".json":
        # JSON format annotations (SUN RGB-D format)
        try:
            with annotation_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                
                # SUN RGB-D format: {"frames": [...], "objects": [...]}
                if "frames" in data and "objects" in data:
                    objects_list = data["objects"]
                    frames = data["frames"]
                    
                    for frame in frames:
                        if "polygon" in frame:
                            for poly in frame["polygon"]:
                                if "object" in poly and "x" in poly and "y" in poly:
                                    obj_idx = poly["object"]
                                    if 0 <= obj_idx < len(objects_list):
                                        obj_name = objects_list[obj_idx]["name"].lower()
                                        
                                        # Convert polygon to bounding box
                                        x_coords = poly["x"]
                                        y_coords = poly["y"]
                                        
                                        # Ensure x_coords and y_coords are lists
                                        if not isinstance(x_coords, list):
                                            x_coords = [x_coords]
                                        if not isinstance(y_coords, list):
                                            y_coords = [y_coords]
                                        
                                        if x_coords and y_coords:
                                            x1, x2 = min(x_coords), max(x_coords)
                                            y1, y2 = min(y_coords), max(y_coords)
                                            
                                            label_line = _create_yolo_line(
                                                obj_name, x1, y1, x2, y2, depth_map, width, height, depth_max, min_box_size
                                            )
                                            if label_line:
                                                lines.append(label_line)
                                                if obj_name in SUNRGBD_CLASS_MAP:
                                                    counts[SUNRGBD_CLASS_MAP[obj_name]] += 1
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
            # Skip malformed JSON files
            LOGGER.warning(f"Skipping malformed annotation file {annotation_path}: {e}")
            return lines, counts

    else:
        # Text format annotations (groundtruth_2d format)
        with annotation_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_name = parts[0].lower()
                if class_name not in SUNRGBD_CLASS_MAP:
                    continue

                try:
                    # Format: class_name x1 y1 x2 y2
                    x1, y1, x2, y2 = map(float, parts[1:5])

                    label_line = _create_yolo_line(
                        class_name, x1, y1, x2, y2, depth_map, width, height, depth_max, min_box_size
                    )
                    if label_line:
                        lines.append(label_line)
                        counts[SUNRGBD_CLASS_MAP[class_name]] += 1

                except (ValueError, IndexError):
                    continue

    return lines, counts


def _create_yolo_line(
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    depth_map: Optional[np.ndarray],
    width: int,
    height: int,
    depth_max: float,
    min_box_size: int,
) -> Optional[str]:
    """Create a YOLO MDE format line from bounding box and depth."""
    # Clip to image bounds
    x1 = max(0.0, min(x1, width))
    y1 = max(0.0, min(y1, height))
    x2 = max(0.0, min(x2, width))
    y2 = max(0.0, min(y2, height))

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Filter small boxes
    if bbox_width < min_box_size or bbox_height < min_box_size:
        return None

    # Normalize coordinates
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    w_norm = bbox_width / width
    h_norm = bbox_height / height

    # Compute depth from depth map
    depth_norm = 0.5  # Default middle depth if no depth map
    if depth_map is not None:
        try:
            # Sample depth from bounding box region
            x1_int, x2_int = int(x1), int(x2)
            y1_int, y2_int = int(y1), int(y2)

            x1_int = max(0, min(x1_int, depth_map.shape[1] - 1))
            x2_int = max(0, min(x2_int, depth_map.shape[1]))
            y1_int = max(0, min(y1_int, depth_map.shape[0] - 1))
            y2_int = max(0, min(y2_int, depth_map.shape[0]))

            if x2_int > x1_int and y2_int > y1_int:
                bbox_depths = depth_map[y1_int:y2_int, x1_int:x2_int]
                valid_depths = bbox_depths[bbox_depths > 0]

                if len(valid_depths) > 0:
                    # Use minimum depth (nearest point in the bounding box)
                    depth_meters = np.min(valid_depths) / 1000.0  # Convert mm to meters
                    depth_norm = min(depth_meters / depth_max, 1.0)
        except (IndexError, ValueError):
            pass

    # Clamp all values
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    depth_norm = max(0.0, min(1.0, depth_norm))

    class_id = SUNRGBD_CLASS_MAP[class_name]
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {depth_norm:.6f}"


__all__ = ("prepare_sunrgbd_mde_dataset", "SUNRGBD_CLASS_MAP")

