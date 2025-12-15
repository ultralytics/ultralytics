#!/usr/bin/env python3
"""Test script to validate 3D bounding box visualization with real stereo image pairs.

This script:
1. Loads three pairs of real left/right images and their corresponding labels from the dataset
2. Uses the visualization code from val.py to generate 3D bounding boxes
3. Saves visualization outputs for manual inspection
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ultralytics.data.stereo.box3d import Box3D
from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetAdapterDataset
from ultralytics.models.yolo.stereo3ddet.val import Stereo3DDetValidator, _labels_to_box3d_list
from ultralytics.utils.plotting import plot_stereo3d_boxes


def load_sample_data(dataset_root: str | Path, split: str = "val", num_samples: int = 3):
    """Load real stereo image pairs and labels from the dataset.
    
    Args:
        dataset_root: Root directory of the KITTI stereo dataset
        split: Dataset split ('train' or 'val')
        num_samples: Number of sample pairs to load
        
    Returns:
        List of tuples: (left_img, right_img, labels, calib, im_file)
    """
    dataset_root = Path(dataset_root)
    
    # Create dataset instance
    dataset = Stereo3DDetAdapterDataset(
        root=dataset_root,
        split=split,
        imgsz=640,  # Standard image size
        names={0: "Car", 1: "Pedestrian", 2: "Cyclist"},
        max_samples=num_samples
    )
    
    samples = []
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            img_tensor = sample["img"]  # [6, H, W] tensor
            
            # Extract left and right images from 6-channel tensor
            left_tensor = img_tensor[0:3, :, :]  # [3, H, W]
            right_tensor = img_tensor[3:6, :, :]  # [3, H, W]
            
            # Convert to numpy and transpose from CHW to HWC
            left_img = left_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            right_img = right_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            
            # Denormalize from [0, 1] to [0, 255] and convert to uint8
            if left_img.dtype in (np.float32, np.float16):
                left_img = np.clip(left_img * 255.0, 0, 255).astype(np.uint8)
            if right_img.dtype in (np.float32, np.float16):
                right_img = np.clip(right_img * 255.0, 0, 255).astype(np.uint8)
            
            # Convert from RGB to BGR for OpenCV
            left_img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR)
            
            # Get labels, calibration, and original shape
            labels = sample.get("labels", [])
            calib = sample.get("calib", {})
            im_file = sample.get("im_file", f"sample_{i}")
            ori_shape = sample.get("ori_shape", None)  # (h, w) of original image before letterboxing
            
            samples.append((left_img, right_img, labels, calib, im_file, ori_shape))
            print(f"Loaded sample {i+1}/{num_samples}: {im_file}")
            
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
    
    return samples


def labels_to_box3d_list(labels: list, calib: dict):
    """Convert labels to Box3D list for visualization.
    
    Args:
        labels: List of label dictionaries with keys like 'cls', 'bboxes', etc.
        calib: Calibration parameters dictionary
        
    Returns:
        List of Box3D objects
    """
    return _labels_to_box3d_list(labels, calib)


def test_visualization(dataset_root: str | Path, output_dir: str | Path = "test_visualization_output"):
    """Test visualization function with real data.
    
    Args:
        dataset_root: Root directory of the KITTI stereo dataset
        output_dir: Directory to save visualization outputs
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading samples from: {dataset_root}")
    print(f"Output directory: {output_dir}")
    
    # Load three sample pairs
    samples = load_sample_data(dataset_root, split="val", num_samples=3)
    
    if not samples:
        print("ERROR: No samples loaded. Please check dataset path.")
        return
    
    print(f"\nLoaded {len(samples)} samples. Generating visualizations...\n")
    
    # Generate visualizations for each sample
    for idx, (left_img, right_img, labels, calib, im_file, ori_shape) in enumerate(samples):
        print(f"\nProcessing sample {idx + 1}: {im_file}")
        
        # Skip if no calibration
        if not calib or not isinstance(calib, dict):
            print(f"  WARNING: Missing calibration for sample {idx + 1}, skipping...")
            continue
        
        # Calculate letterbox parameters to adjust 3D box coordinates
        # Images are letterboxed: resized and padded to square
        letterbox_scale = None
        letterbox_pad_left = None
        letterbox_pad_top = None
        
        if ori_shape:
            ori_h, ori_w = ori_shape  # Original image dimensions
            curr_h, curr_w = left_img.shape[:2]  # Current (letterboxed) image dimensions
            
            # Calculate letterbox parameters (matching dataset._letterbox logic)
            # scale = min(new_shape / h, new_shape / w)
            # new_unpad = (int(round(w * scale)), int(round(h * scale)))
            # dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
            # pad_left = dw // 2, pad_top = dh // 2
            imgsz = max(curr_h, curr_w)  # Letterboxed size (should be square)
            scale = min(imgsz / ori_h, imgsz / ori_w)
            new_unpad_w = int(round(ori_w * scale))
            new_unpad_h = int(round(ori_h * scale))
            dw = imgsz - new_unpad_w
            dh = imgsz - new_unpad_h
            letterbox_pad_left = dw // 2
            letterbox_pad_top = dh // 2
            letterbox_scale = scale
            print(f"  Letterbox params: scale={scale:.3f}, pad_left={letterbox_pad_left}, pad_top={letterbox_pad_top}")
        
        # Convert labels to Box3D for ground truth
        gt_boxes = []
        if labels:
            try:
                gt_boxes = labels_to_box3d_list(labels, calib)
                print(f"  Found {len(gt_boxes)} ground truth boxes")
            except Exception as e:
                print(f"  WARNING: Error converting labels to Box3D: {e}")
        
        # For testing, we'll use ground truth boxes as "predictions" too
        # In real usage, predictions would come from the model
        pred_boxes = gt_boxes.copy() if gt_boxes else []
        
        # Generate visualization
        try:
            left_canvas, right_canvas, combined = plot_stereo3d_boxes(
                left_img=left_img,
                right_img=right_img,
                pred_boxes3d=pred_boxes,
                gt_boxes3d=gt_boxes,
                left_calib=calib,
                letterbox_scale=letterbox_scale,
                letterbox_pad_left=letterbox_pad_left,
                letterbox_pad_top=letterbox_pad_top,
            )
            
            # Save outputs
            sample_name = Path(im_file).stem if im_file else f"sample_{idx}"
            left_path = output_dir / f"{sample_name}_left.jpg"
            right_path = output_dir / f"{sample_name}_right.jpg"
            combined_path = output_dir / f"{sample_name}_combined.jpg"
            
            cv2.imwrite(str(left_path), left_canvas)
            cv2.imwrite(str(right_path), right_canvas)
            cv2.imwrite(str(combined_path), combined)
            
            print(f"  ✓ Saved visualizations:")
            print(f"    - Left: {left_path}")
            print(f"    - Right: {right_path}")
            print(f"    - Combined: {combined_path}")
            
        except Exception as e:
            print(f"  ERROR: Failed to generate visualization: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Visualization test complete! Check outputs in: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test 3D bounding box visualization with real data")
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory of the KITTI stereo dataset (e.g., /path/to/kitti_stereo)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_visualization_output",
        help="Directory to save visualization outputs (default: test_visualization_output)"
    )
    
    args = parser.parse_args()
    
    test_visualization(args.dataset_root, args.output_dir)
