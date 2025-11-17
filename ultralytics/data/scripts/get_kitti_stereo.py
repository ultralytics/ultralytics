#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
KITTI Stereo Dataset Download and Conversion Script

This script downloads and converts the KITTI dataset to YOLO format with stereo support.
KITTI dataset requires manual registration at: https://www.cvlibs.net/datasets/kitti/user_login.php

Usage:
    python get_kitti_stereo.py --kitti-root /path/to/kitti --output /path/to/output
    python get_kitti_stereo.py --kitti-root /path/to/kitti  # outputs to ../datasets/kitti-stereo
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2

try:
    from ultralytics.utils import LOGGER, TQDM
except ImportError:
    # Fallback if not in ultralytics environment
    import sys
    from tqdm import tqdm as TQDM
    
    class LOGGER:
        @staticmethod
        def info(msg):
            print(f"[INFO] {msg}")
        
        @staticmethod
        def warning(msg):
            print(f"[WARNING] {msg}")
        
        @staticmethod
        def error(msg):
            print(f"[ERROR] {msg}")


def convert_kitti_label_to_yolo(kitti_label: Path, yolo_label: Path, img_path: Path):
    """Convert KITTI label format to YOLO format.
    
    KITTI format: class truncated occluded alpha bbox_x1 bbox_y1 bbox_x2 bbox_y2 height width length x y z rotation_y
    YOLO format: class x_center y_center width height (normalized)
    
    Args:
        kitti_label: Path to KITTI format label file
        yolo_label: Path to output YOLO format label file
        img_path: Path to corresponding image (for dimensions)
    """
    # Read image dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        LOGGER.warning(f"Could not read image: {img_path}")
        return
    img_h, img_w = img.shape[:2]
    
    # Class mapping (KITTI to YOLO)
    class_map = {
        "Car": 0, "Van": 1, "Truck": 2,
        "Pedestrian": 3, "Person_sitting": 4,
        "Cyclist": 5, "Tram": 6, "Misc": 7,
        "DontCare": -1  # Skip DontCare
    }
    
    yolo_lines = []
    with open(kitti_label, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            class_name = parts[0]
            if class_name == "DontCare":
                continue
            
            if class_name not in class_map:
                continue
            
            cls_id = class_map[class_name]
            
            # Extract 2D bounding box (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(float, parts[4:8])
            
            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = ((x1 + x2) / 2.0) / img_w
            y_center = ((y1 + y2) / 2.0) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            
            # Ensure values are in [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Write YOLO format label
    with open(yolo_label, "w") as f:
        f.writelines(yolo_lines)


def convert_kitti_to_yolo_stereo(kitti_root: Path, output_dir: Path, split: str = "training"):
    """Convert KITTI dataset to YOLO format with stereo support.
    
    Args:
        kitti_root: Path to raw KITTI dataset root (contains 'training' and 'testing' folders)
        output_dir: Path to output directory for converted dataset
        split: 'training' or 'testing'
    """
    kitti_dir = kitti_root / split
    if not kitti_dir.exists():
        LOGGER.warning(f"KITTI {split} directory not found: {kitti_dir}")
        return
    
    # Create output directories
    for subset in ["train", "val"]:
        for side in ["left", "right"]:
            (output_dir / "images" / subset / side).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / "calib" / subset).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    left_images = sorted((kitti_dir / "image_2").glob("*.png"))
    right_images = sorted((kitti_dir / "image_3").glob("*.png"))
    
    if not left_images:
        LOGGER.warning(f"No images found in {kitti_dir / 'image_2'}")
        return
    
    # Split into train/val (80/20 split)
    n_total = len(left_images)
    n_train = int(n_total * 0.8)
    
    train_left = left_images[:n_train]
    train_right = right_images[:n_train] if len(right_images) >= n_train else []
    val_left = left_images[n_train:]
    val_right = right_images[n_train:] if len(right_images) > n_train else []
    
    # Process training set
    LOGGER.info(f"Processing {len(train_left)} training images...")
    for i, left_img in enumerate(TQDM(train_left, desc="Converting training set")):
        img_id = left_img.stem
        
        # Copy left image
        shutil.copy2(left_img, output_dir / "images" / "train" / "left" / f"{img_id}.png")
        
        # Copy right image if available
        if i < len(train_right) and train_right[i].exists():
            shutil.copy2(train_right[i], output_dir / "images" / "train" / "right" / f"{img_id}.png")
        
        # Convert label
        label_file = kitti_dir / "label_2" / f"{img_id}.txt"
        if label_file.exists():
            yolo_label = output_dir / "labels" / "train" / f"{img_id}.txt"
            convert_kitti_label_to_yolo(label_file, yolo_label, left_img)
        
        # Copy calibration file
        calib_file = kitti_dir / "calib" / f"{img_id}.txt"
        if calib_file.exists():
            shutil.copy2(calib_file, output_dir / "calib" / "train" / f"{img_id}.txt")
    
    # Process validation set
    LOGGER.info(f"Processing {len(val_left)} validation images...")
    for i, left_img in enumerate(TQDM(val_left, desc="Converting validation set")):
        img_id = left_img.stem
        
        # Copy left image
        shutil.copy2(left_img, output_dir / "images" / "val" / "left" / f"{img_id}.png")
        
        # Copy right image if available
        if i < len(val_right) and val_right[i].exists():
            shutil.copy2(val_right[i], output_dir / "images" / "val" / "right" / f"{img_id}.png")
        
        # Convert label
        label_file = kitti_dir / "label_2" / f"{img_id}.txt"
        if label_file.exists():
            yolo_label = output_dir / "labels" / "val" / f"{img_id}.txt"
            convert_kitti_label_to_yolo(label_file, yolo_label, left_img)
        
        # Copy calibration file
        calib_file = kitti_dir / "calib" / f"{img_id}.txt"
        if calib_file.exists():
            shutil.copy2(calib_file, output_dir / "calib" / "val" / f"{img_id}.txt")
    
    LOGGER.info(f"Conversion complete! Train: {len(train_left)}, Val: {len(val_left)}")


def main():
    parser = argparse.ArgumentParser(description="Convert KITTI dataset to YOLO format with stereo support")
    parser.add_argument(
        "--kitti-root",
        type=str,
        required=True,
        help="Path to KITTI dataset root (contains 'training' and 'testing' folders)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ../datasets/kitti-stereo)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "testing"],
        help="Which split to convert (default: training)"
    )
    
    args = parser.parse_args()
    
    kitti_root = Path(args.kitti_root)
    if not kitti_root.exists():
        LOGGER.error(f"KITTI root directory not found: {kitti_root}")
        return 1
    
    if not (kitti_root / "training" / "image_2").exists():
        LOGGER.error(f"KITTI training images not found at {kitti_root / 'training' / 'image_2'}")
        LOGGER.error("Please ensure KITTI dataset is properly extracted.")
        return 1
    
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default to ../datasets/kitti-stereo relative to script location
        script_dir = Path(__file__).parent.parent.parent.parent
        output_dir = script_dir / "datasets" / "kitti-stereo"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("=" * 60)
    LOGGER.info("KITTI Stereo Dataset Conversion")
    LOGGER.info("=" * 60)
    LOGGER.info(f"KITTI Root: {kitti_root}")
    LOGGER.info(f"Output Dir: {output_dir}")
    LOGGER.info(f"Split: {args.split}")
    LOGGER.info("=" * 60)
    LOGGER.info("")
    
    convert_kitti_to_yolo_stereo(kitti_root, output_dir, split=args.split)
    
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("Conversion Complete!")
    LOGGER.info(f"Dataset saved to: {output_dir}")
    LOGGER.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

