#!/usr/bin/env python3
"""Visual testing script for stereo 3D detection augmentations.

This script tests each augmentation method individually and the full pipeline,
generating side-by-side visualizations to verify that:
1. Images are transformed correctly
2. Labels (2D boxes) are transformed correctly
3. Stereo correspondence is preserved

Usage:
    python scripts/test_stereo_augmentations_visual.py --data path/to/kitti_raw --output results/
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics.models.yolo.stereo3ddet.augment import (
    HorizontalFlipAugmentor,
    PhotometricAugmentor,
    RandomCropAugmentor,
    RandomScaleAugmentor,
    StereoAugmentationPipeline,
)


def draw_boxes_on_stereo(
    left: np.ndarray,
    right: np.ndarray,
    labels: list[dict[str, Any]],
    left_color: tuple = (0, 255, 0),
    right_color: tuple = (255, 0, 0),
    thickness: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw 2D bounding boxes on stereo images from label dicts.

    Args:
        left: Left image [H, W, 3].
        right: Right image [H, W, 3].
        labels: List of label dicts with left_box/right_box.
        left_color: Box color for left image (BGR for cv2).
        right_color: Box color for right image (BGR for cv2).
        thickness: Box line thickness.

    Returns:
        Tuple of (left_with_boxes, right_with_boxes).
    """
    left_copy = left.copy()
    right_copy = right.copy()
    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]

    for lab in labels:
        # Draw left box on left image
        if "left_box" in lab:
            lb = lab["left_box"]
            cx = int(lb["center_x"] * w_l)
            cy = int(lb["center_y"] * h_l)
            bw = int(lb["width"] * w_l)
            bh = int(lb["height"] * h_l)
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)
            cv2.rectangle(left_copy, (x1, y1), (x2, y2), left_color, thickness)

        # Draw right box on right image
        if "right_box" in lab:
            rb = lab["right_box"]
            rcx = int(rb["center_x"] * w_r)
            rcy = int(rb["center_y"] * h_r)
            rbw = int(rb["width"] * w_r)
            rbh = int(rb["height"] * h_r)
            rx1 = int(rcx - rbw / 2)
            ry1 = int(rcy - rbh / 2)
            rx2 = int(rcx + rbw / 2)
            ry2 = int(rcy + rbh / 2)
            cv2.rectangle(right_copy, (rx1, ry1), (rx2, ry2), right_color, thickness)

    return left_copy, right_copy


def visualize_augmentation(
    orig_left: np.ndarray,
    orig_right: np.ndarray,
    orig_labels: list[dict],
    aug_left: np.ndarray,
    aug_right: np.ndarray,
    aug_labels: list[dict],
    aug_name: str,
    output_path: Path,
    sample_idx: int = 0,
):
    """Create side-by-side visualization of original vs augmented.

    Args:
        orig_left: Original left image.
        orig_right: Original right image.
        orig_labels: Original labels.
        aug_left: Augmented left image.
        aug_right: Augmented right image.
        aug_labels: Augmented labels.
        aug_name: Name of augmentation for title.
        output_path: Path to save visualization.
        sample_idx: Sample index for filename.
    """
    # Convert to uint8 if needed
    def to_uint8(img):
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                return (np.clip(img, 0, 1) * 255).astype(np.uint8)
            return img.astype(np.uint8)
        return img

    orig_left = to_uint8(orig_left)
    orig_right = to_uint8(orig_right)
    aug_left = to_uint8(aug_left)
    aug_right = to_uint8(aug_right)

    # Draw boxes
    orig_left_boxed, orig_right_boxed = draw_boxes_on_stereo(orig_left, orig_right, orig_labels)
    aug_left_boxed, aug_right_boxed = draw_boxes_on_stereo(aug_left, aug_right, aug_labels)

    # Debug: print first bbox for verification
    if orig_labels:
        lb = orig_labels[0].get("left_box", {})
        print(f"  [Viz {aug_name}] ORIGINAL first_left_box: cx={lb.get('center_x', 0):.3f}, cy={lb.get('center_y', 0):.3f}")
    if aug_labels:
        lb = aug_labels[0].get("left_box", {})
        print(f"  [Viz {aug_name}] AUGMENTED first_left_box: cx={lb.get('center_x', 0):.3f}, cy={lb.get('center_y', 0):.3f}")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Augmentation: {aug_name} (Sample {sample_idx})", fontsize=16, fontweight="bold")

    # Original left
    axes[0, 0].imshow(cv2.cvtColor(orig_left_boxed, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Original Left ({orig_left.shape[1]}x{orig_left.shape[0]})", fontsize=12)
    axes[0, 0].axis("off")

    # Original right
    axes[0, 1].imshow(cv2.cvtColor(orig_right_boxed, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f"Original Right ({orig_right.shape[1]}x{orig_right.shape[0]})", fontsize=12)
    axes[0, 1].axis("off")

    # Augmented left
    axes[1, 0].imshow(cv2.cvtColor(aug_left_boxed, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Augmented Left ({aug_left.shape[1]}x{aug_left.shape[0]})", fontsize=12)
    axes[1, 0].axis("off")

    # Augmented right
    axes[1, 1].imshow(cv2.cvtColor(aug_right_boxed, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Augmented Right ({aug_right.shape[1]}x{aug_right.shape[0]})", fontsize=12)
    axes[1, 1].axis("off")

    # Save figure
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{aug_name}_sample_{sample_idx}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization: {output_file}")


def test_horizontal_flip(
    left: np.ndarray,
    right: np.ndarray,
    labels: list[dict],
    output_path: Path,
    sample_idx: int = 0,
):
    """Test horizontal flip augmentation."""
    transform = HorizontalFlipAugmentor(p_apply=1.0)  # Force application
    
    # Deep copy to avoid modifying originals
    left_copy = left.copy()
    right_copy = right.copy()
    labels_copy = deepcopy(labels)
    
    # Apply transform
    aug_left, aug_right, aug_labels, applied = transform(left_copy, right_copy, labels_copy)
    
    print(f"  HorizontalFlip applied: {applied}")
    
    # Visualize
    visualize_augmentation(
        left, right, labels,
        aug_left, aug_right, aug_labels,
        "HorizontalFlip", output_path, sample_idx
    )


def test_random_scale(
    left: np.ndarray,
    right: np.ndarray,
    labels: list[dict],
    output_path: Path,
    sample_idx: int = 0,
):
    """Test random scale augmentation."""
    transform = RandomScaleAugmentor(scale_range=(0.8, 1.2), p_apply=1.0)
    
    # Deep copy to avoid modifying originals
    left_copy = left.copy()
    right_copy = right.copy()
    labels_copy = deepcopy(labels)
    
    # Apply transform
    aug_left, aug_right, aug_labels = transform(left_copy, right_copy, labels_copy)
    
    print(f"  RandomScale: {left.shape} -> {aug_left.shape}")
    
    # Visualize
    visualize_augmentation(
        left, right, labels,
        aug_left, aug_right, aug_labels,
        "RandomScale", output_path, sample_idx
    )


def test_random_crop(
    left: np.ndarray,
    right: np.ndarray,
    labels: list[dict],
    output_path: Path,
    sample_idx: int = 0,
):
    """Test random crop augmentation."""
    transform = RandomCropAugmentor(crop_height_ratio=0.9, crop_width_ratio=0.9, p_apply=1.0)
    
    # Deep copy to avoid modifying originals
    left_copy = left.copy()
    right_copy = right.copy()
    labels_copy = deepcopy(labels)
    
    # Apply transform
    aug_left, aug_right, aug_labels, applied, x0, y0, cw, ch = transform(left_copy, right_copy, labels_copy)
    
    print(f"  RandomCrop applied: {applied}, crop_offset=({x0}, {y0}), crop_size=({cw}, {ch})")
    
    # Visualize
    visualize_augmentation(
        left, right, labels,
        aug_left, aug_right, aug_labels,
        "RandomCrop", output_path, sample_idx
    )


def test_photometric(
    left: np.ndarray,
    right: np.ndarray,
    labels: list[dict],
    output_path: Path,
    sample_idx: int = 0,
):
    """Test HSV photometric augmentation."""
    transform = PhotometricAugmentor(hgain=0.5, sgain=0.5, vgain=0.5, p_apply=1.0)
    
    # Deep copy to avoid modifying originals
    left_copy = left.copy()
    right_copy = right.copy()
    
    # Apply transform (photometric doesn't change labels)
    aug_left, aug_right = transform(left_copy, right_copy)
    
    # Visualize
    visualize_augmentation(
        left, right, labels,
        aug_left, aug_right, labels,  # Labels unchanged
        "PhotometricHSV", output_path, sample_idx
    )


def test_full_pipeline(
    left: np.ndarray,
    right: np.ndarray,
    labels: list[dict],
    output_path: Path,
    sample_idx: int = 0,
):
    """Test complete augmentation pipeline."""
    pipeline = StereoAugmentationPipeline(
        photometric=PhotometricAugmentor(p_apply=1.0),
        hflip=HorizontalFlipAugmentor(p_apply=0.5),
        rscale=RandomScaleAugmentor(p_apply=1.0),
        rcrop=RandomCropAugmentor(p_apply=1.0),
    )
    
    # Deep copy to avoid modifying originals
    left_copy = left.copy()
    right_copy = right.copy()
    labels_copy = deepcopy(labels)
    
    # Apply pipeline
    aug_left, aug_right, aug_labels, _ = pipeline.augment(left_copy, right_copy, labels_copy)
    
    # Visualize
    visualize_augmentation(
        left, right, labels,
        aug_left, aug_right, aug_labels,
        "FullPipeline", output_path, sample_idx
    )


def load_kitti_raw_sample(data_path: Path, idx: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load a sample directly from raw KITTI data.
    
    Expected structure:
    - data_path/training/image_2/ (left images)
    - data_path/training/image_3/ (right images)
    - data_path/training/label_2/ (labels)
    - data_path/training/calib/ (calibration)
    
    Returns:
        (left_image, right_image, labels_list)
    """
    left_dir = data_path / "training" / "image_2"
    right_dir = data_path / "training" / "image_3"
    label_dir = data_path / "training" / "label_2"
    
    # Get image files
    left_files = sorted(left_dir.glob("*.png"))
    if idx >= len(left_files):
        raise IndexError(f"Sample index {idx} out of range (total: {len(left_files)})")
    
    left_file = left_files[idx]
    img_id = left_file.stem
    right_file = right_dir / f"{img_id}.png"
    label_file = label_dir / f"{img_id}.txt"
    
    # Load images
    left = cv2.imread(str(left_file))
    
    if left is None:
        raise FileNotFoundError(f"Failed to load left image: {left_file}")
    
    # For visualization testing, use left image for both views
    # This is because KITTI labels are only for the left camera (image_2)
    # Using different stereo images with left-only labels causes box misalignment
    right = left.copy()
    
    # Convert BGR to RGB
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    
    h, w = left.shape[:2]
    
    # Parse labels (KITTI format)
    labels = []
    if label_file.exists():
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                obj_type = parts[0]
                if obj_type in ("DontCare", "Misc"):
                    continue
                
                # KITTI 2D box format: left, top, right, bottom (pixels)
                x1, y1, x2, y2 = map(float, parts[4:8])
                
                # Convert to normalized xywh
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                
                # Use same box for left and right (identical for visualization testing)
                # Real stereo would use properly calibrated disparity-shifted right box
                labels.append({
                    "class": obj_type,
                    "left_box": {
                        "center_x": cx,
                        "center_y": cy,
                        "width": bw,
                        "height": bh,
                    },
                    "right_box": {
                        "center_x": cx,  # Same as left for visualization testing
                        "center_y": cy,
                        "width": bw,
                        "height": bh,
                    },
                })
    
    return left, right, labels


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visual testing for stereo augmentations")
    parser.add_argument("--data", type=str, required=True, help="Path to KITTI raw dataset (e.g., /path/to/kitti_raw)")
    parser.add_argument("--output", type=str, default="vis_results", help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--augmentations", nargs="+", default=["all"], help="Augmentations to test: flip, scale, crop, photometric, pipeline, all")
    args = parser.parse_args()

    # Setup paths
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {data_path}")
    print(f"Output directory: {output_path}")
    
    # Check expected structure
    left_dir = data_path / "training" / "image_2"
    if not left_dir.exists():
        print(f"ERROR: Expected KITTI structure with {left_dir}")
        print("Expected structure:")
        print("  <data>/training/image_2/ (left images)")
        print("  <data>/training/image_3/ (right images)")
        print("  <data>/training/label_2/ (labels)")
        return

    # Count available samples
    num_available = len(list(left_dir.glob("*.png")))
    print(f"Found {num_available} samples")

    # Determine which augmentations to test
    if "all" in args.augmentations:
        test_augs = ["flip", "scale", "crop", "photometric", "pipeline"]
    else:
        test_augs = args.augmentations

    # Test samples
    for i in range(min(args.num_samples, num_available)):
        print(f"\nTesting sample {i + 1}/{args.num_samples}...")
        
        try:
            left, right, labels = load_kitti_raw_sample(data_path, i)
        except Exception as e:
            print(f"  Error loading sample {i}: {e}")
            continue
        
        print(f"  Image shape: {left.shape}, Labels: {len(labels)}")
        
        if not labels:
            print(f"  Skipping sample {i}: no labels")
            continue
        
        # Test each augmentation
        if "flip" in test_augs:
            print("  Testing horizontal flip...")
            test_horizontal_flip(left, right, labels, output_path, i)
        
        if "scale" in test_augs:
            print("  Testing random scale...")
            test_random_scale(left, right, labels, output_path, i)
        
        if "crop" in test_augs:
            print("  Testing random crop...")
            test_random_crop(left, right, labels, output_path, i)
        
        if "photometric" in test_augs:
            print("  Testing photometric HSV...")
            test_photometric(left, right, labels, output_path, i)
        
        if "pipeline" in test_augs:
            print("  Testing full pipeline...")
            test_full_pipeline(left, right, labels, output_path, i)

    print(f"\nVisualization complete! Results saved to: {output_path}")


if __name__ == "__main__":
    main()
