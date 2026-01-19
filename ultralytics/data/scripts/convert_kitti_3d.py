#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
KITTI to YOLO 3D Stereo Format Converter

This script converts KITTI dataset to YOLO 3D stereo format with 26 values per object:
class x_l y_l w_l h_l x_r y_r w_r h_r dim_l dim_w dim_h loc_x loc_y loc_z rot_y kp1_x kp1_y kp2_x kp2_y kp3_x kp3_y kp4_x kp4_y truncated occluded

All coordinates normalized to [0, 1], dimensions in meters, rot_y in radians.
Uses 3DOP split strategy: training set split into train (0-3711) and val (3712+).

Usage:
    python convert_kitti_3d.py --kitti-root /path/to/kitti
    python convert_kitti_3d.py --kitti-root /path/to/kitti --filter-classes Car Pedestrian Cyclist
"""

import argparse
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    from ultralytics.utils import LOGGER, TQDM
except ImportError:
    # Fallback if not in ultralytics environment
    from tqdm import tqdm as TQDM

    class SimpleLogger:
        @staticmethod
        def info(msg):
            print(f"[INFO] {msg}")

        @staticmethod
        def warning(msg):
            print(f"[WARNING] {msg}")

        @staticmethod
        def error(msg):
            print(f"[ERROR] {msg}")

    # Match expected interface name
    LOGGER = SimpleLogger


SPLIT_MAP = {"training": "train", "testing": "val"}


class KITTIToYOLO3D:
    """Convert KITTI dataset to YOLO 3D Stereo format with directory layout:

    root/
        images/{train|val}/left/*.png
        images/{train|val}/right/*.png
        labels/{train|val}/*.txt
        calib/{train|val}/*.txt
        dataset.yaml
        train.txt (image index list for training split if created)
        val.txt   (image index list for validation/testing split if created)
    """

    def __init__(self, kitti_root, output_root, filter_classes=None, split_strategy="single", filter_occluded=False, max_occlusion_level=1):
        """
        Args:
            kitti_root: Path to KITTI dataset root
            output_root: Path to output directory
            filter_classes: List of class names to include (None = include all)
            filter_occluded: Whether to filter out heavily occluded objects (default: False)
            max_occlusion_level: Maximum occlusion level to include (default: 1, excludes heavy/unknown)
        """
        self.kitti_root = Path(kitti_root)
        self.output_root = Path(output_root)

        # KITTI image dimensions (standard)
        self.img_width = 1242
        self.img_height = 375

        # Class mapping
        self.class_map = {
            "Car": 0,
            "Van": 1,
            "Truck": 2,
            "Pedestrian": 3,
            "Person_sitting": 4,
            "Cyclist": 5,
            "Tram": 6,
            "Misc": 7,
        }

        # Split strategy: 'single' (original behavior) or '3dop' (fixed train/val split on training set indices)
        self.split_strategy = split_strategy

    # Filter classes (if specified)
        self.filter_classes = filter_classes
        # Occlusion filtering settings
        self.filter_occluded = filter_occluded
        self.max_occlusion_level = max_occlusion_level
        if filter_occluded:
            LOGGER.info(f"Occlusion filtering enabled: excluding objects with occlusion level > {max_occlusion_level} "
                       f"(0=visible, 1=partial, 2=heavy, 3=unknown)")
        self.class_id_remap = None  # Will store remapping from original to new IDs
        if self.filter_classes is not None:
            # Convert to set for faster lookup
            self.filter_classes = set([c.strip() for c in self.filter_classes])
            # Validate that all filter classes exist
            invalid_classes = self.filter_classes - set(self.class_map.keys())
            if invalid_classes:
                LOGGER.warning(f"Invalid classes in filter: {invalid_classes}. Ignoring them.")
                self.filter_classes = self.filter_classes - invalid_classes
            if not self.filter_classes:
                LOGGER.error("No valid classes in filter. Using all classes.")
                self.filter_classes = None
            else:
                # Create remapping: original class_id -> new consecutive class_id
                self.class_id_remap = {}
                new_id = 0
                for class_name, old_id in sorted(self.class_map.items(), key=lambda x: x[1]):
                    if class_name in self.filter_classes:
                        self.class_id_remap[old_id] = new_id
                        new_id += 1
                LOGGER.info(f"Filtering classes: {sorted(self.filter_classes)}")
                LOGGER.info(f"Class ID remapping: {self.class_id_remap}")

        # Create base output directories (split-specific subfolders created during conversion)
        self._setup_output_dirs()
        
        # Mean dimensions will be computed during conversion and stored here
        self.mean_dims = None

    def _setup_output_dirs(self):
        """Create base directory containers (no split yet)."""
        for parent in ["images", "labels", "calib"]:
            (self.output_root / parent).mkdir(parents=True, exist_ok=True)

    def parse_calibration(self, calib_file):
        """
        Parse KITTI calibration file.

        Returns:
            dict with fx, fy, cx, cy, baseline, P2, P3
        """
        with open(calib_file, "r") as f:
            lines = f.readlines()

        # Parse projection matrices
        P2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
        P3 = np.array([float(x) for x in lines[3].split()[1:]]).reshape(3, 4)

        # Extract intrinsics (left camera)
        fx = P2[0, 0]
        fy = P2[1, 1]
        cx = P2[0, 2]
        cy = P2[1, 2]
        # Extract intrinsics (right camera)
        right_cx = P3[0, 2]
        right_cy = P3[1, 2]

        # Calculate baseline
        baseline = (P2[0, 3] - P3[0, 3]) / fx

        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "right_cx": right_cx,
            "right_cy": right_cy,
            "baseline": abs(baseline),
            "image_width": self.img_width,
            "image_height": self.img_height,
            "P2": P2,
            "P3": P3,
        }

    def compute_bottom_vertices(self, X, Y, Z, h, w, l, ry, calib):
        """
        Compute 4 bottom vertices of 3D box projected to left image.

        Args:
            X, Y, Z: 3D location (bottom center) in camera coords
            h, w, l: 3D dimensions
            ry: rotation around Y axis
            calib: calibration dict

        Returns:
            4x2 array of [u, v] coordinates
        """
        # Define 4 bottom corners in object coordinate system
        # Y=0 because KITTI location is at bottom center
        corners_3d_obj = np.array(
            [
                [-l / 2, 0, -w / 2],  # rear-left
                [l / 2, 0, -w / 2],  # front-left
                [l / 2, 0, w / 2],  # front-right
                [-l / 2, 0, w / 2],  # rear-right
            ]
        )

        # Rotation matrix around Y axis
        cos_ry = np.cos(ry)
        sin_ry = np.sin(ry)
        R = np.array([[cos_ry, 0, sin_ry], [0, 1, 0], [-sin_ry, 0, cos_ry]])

        # Rotate corners
        corners_3d_rotated = corners_3d_obj @ R.T

        # Translate to camera coordinates
        corners_3d_cam = corners_3d_rotated + np.array([X, Y, Z])

        # Project to image plane
        corners_2d = []
        for corner in corners_3d_cam:
            if corner[2] <= 0:  # Behind camera
                corners_2d.append([0, 0])
                continue

            u = calib["fx"] * (corner[0] / corner[2]) + calib["cx"]
            v = calib["fy"] * (corner[1] / corner[2]) + calib["cy"]
            corners_2d.append([u, v])

        return np.array(corners_2d)

    def compute_right_box(self, X, Y, Z, h, w, l, ry, calib, left_box_2d):
        """
        Compute right image 2D box center and width.

        Uses disparity-based formula for center (geometrically correct) and
        corner projection for width estimation.

        Returns:
            center_x_r, width_r (in pixels)
        """
        x1_l, y1_l, x2_l, y2_l = left_box_2d
        center_x_l = (x1_l + x2_l) / 2
        width_l = x2_l - x1_l

        # Use simple disparity formula for center (always correct for stereo geometry)
        # disparity = (fx * baseline) / depth
        # In stereo: left camera sees objects more to the right than right camera
        # So: center_x_r = center_x_l - disparity (disparity > 0 for objects in front)
        if Z > 0:
            disparity = (calib["fx"] * calib["baseline"]) / Z
            center_x_r = center_x_l - disparity
        else:
            # Object behind camera, shouldn't happen for valid labels
            center_x_r = center_x_l
            return center_x_r, width_l

        # For width, project corners to right image to get accurate width
        # Get all 8 corners of 3D box
        corners_3d_obj = np.array(
            [
                # Bottom 4
                [-l / 2, 0, -w / 2],
                [l / 2, 0, -w / 2],
                [l / 2, 0, w / 2],
                [-l / 2, 0, w / 2],
                # Top 4
                [-l / 2, -h, -w / 2],
                [l / 2, -h, -w / 2],
                [l / 2, -h, w / 2],
                [-l / 2, -h, w / 2],
            ]
        )

        # Rotation
        cos_ry = np.cos(ry)
        sin_ry = np.sin(ry)
        R = np.array([[cos_ry, 0, sin_ry], [0, 1, 0], [-sin_ry, 0, cos_ry]])

        corners_3d_rotated = corners_3d_obj @ R.T
        corners_3d_cam = corners_3d_rotated + np.array([X, Y, Z])

        # Project to right image (account for baseline)
        corners_2d_right = []
        for corner in corners_3d_cam:
            if corner[2] <= 0:
                continue

            # Right camera sees object shifted by baseline
            X_right = corner[0] - calib["baseline"]
            u_r = calib["fx"] * (X_right / corner[2]) + calib["cx"]
            corners_2d_right.append(u_r)

        if len(corners_2d_right) == 0:
            # Fallback: use same width as left box
            return center_x_r, width_l

        # Compute width from projected corners
        x_min = min(corners_2d_right)
        x_max = max(corners_2d_right)
        width_r = x_max - x_min

        # Sanity check: width should be positive and reasonable
        if width_r <= 0 or width_r > self.img_width * 2:
            width_r = width_l

        return center_x_r, width_r

    def convert_label(self, label_file, calib_file):
        """
        Convert single KITTI label file to YOLO 3D format.

        Returns:
            List of label strings
        """
        if not os.path.exists(label_file):
            return []

        calib = self.parse_calibration(calib_file)

        with open(label_file, "r") as f:
            lines = f.readlines()

        yolo_labels = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            obj_type = parts[0]

            # Skip DontCare and unknown classes
            if obj_type == "DontCare" or obj_type not in self.class_map:
                continue

            # Filter classes if specified
            if self.filter_classes is not None and obj_type not in self.filter_classes:
                continue

            # Get class ID and remap if filtering is active
            class_id = self.class_map[obj_type]
            if self.class_id_remap is not None:
                class_id = self.class_id_remap[class_id]

            # Parse KITTI label fields
            truncated = float(parts[1])
            occluded = int(parts[2])
            alpha = float(parts[3])

            # Filter heavily occluded objects if enabled
            if self.filter_occluded and occluded > self.max_occlusion_level:
                # Skip this object (it's too occluded to be useful for training)
                continue

            # 2D bounding box (left image)
            x1, y1, x2, y2 = [float(x) for x in parts[4:8]]

            # Skip if box is too small or invalid
            if x2 <= x1 or y2 <= y1:
                continue

            # 3D dimensions
            h, w, l = [float(x) for x in parts[8:11]]

            # 3D location (bottom center)
            X, Y, Z = [float(x) for x in parts[11:14]]

            # Rotation
            rotation_y = float(parts[14])

            # Skip if behind camera or too far
            if Z <= 0 or Z > 100:
                continue

            # ===== Left 2D Box (normalized) =====
            center_x_l = (x1 + x2) / 2
            center_y_l = (y1 + y2) / 2
            width_l = x2 - x1
            height_l = y2 - y1

            center_x_l_norm = center_x_l / self.img_width
            center_y_l_norm = center_y_l / self.img_height
            width_l_norm = width_l / self.img_width
            height_l_norm = height_l / self.img_height

            # ===== Right 2D Box (normalized) =====
            center_x_r, width_r = self.compute_right_box(X, Y, Z, h, w, l, rotation_y, calib, left_box_2d=[x1, y1, x2, y2])

            center_x_r_norm = center_x_r / self.img_width
            center_y_r_norm = center_y_l_norm  # Same y due to epipolar constraint
            width_r_norm = width_r / self.img_width
            height_r_norm = height_l_norm  # Same height due to epipolar constraint

            # Clamp to valid range
            center_x_r_norm = np.clip(center_x_r_norm, 0, 1)
            center_y_r_norm = np.clip(center_y_r_norm, 0, 1)
            width_r_norm = np.clip(width_r_norm, 0, 1)
            height_r_norm = np.clip(height_r_norm, 0, 1)

            # ===== Bottom 4 Vertices (normalized) =====
            vertices_2d = self.compute_bottom_vertices(X, Y, Z, h, w, l, rotation_y, calib)

            # Normalize vertices
            vertices_norm = []
            for v in vertices_2d:
                v_x = np.clip(v[0] / self.img_width, 0, 1)
                v_y = np.clip(v[1] / self.img_height, 0, 1)
                vertices_norm.extend([v_x, v_y])

            # ===== Build YOLO label line =====
            # Format: 26 values total
            # class x_l y_l w_l h_l x_r y_r w_r h_r dim_l dim_w dim_h loc_x loc_y loc_z rot_y kp1_x kp1_y kp2_x kp2_y kp3_x kp3_y kp4_x kp4_y truncated occluded
            label = f"{class_id} "
            # Left box (4 values)
            label += f"{center_x_l_norm:.6f} {center_y_l_norm:.6f} "
            label += f"{width_l_norm:.6f} {height_l_norm:.6f} "
            # Right box (4 values)
            label += f"{center_x_r_norm:.6f} {center_y_r_norm:.6f} "
            label += f"{width_r_norm:.6f} {height_r_norm:.6f} "
            # Dimensions: length, width, height (3 values)
            label += f"{l:.2f} {w:.2f} {h:.2f} "
            # 3D location: X, Y, Z (3 values)
            label += f"{X:.4f} {Y:.4f} {Z:.4f} "
            # Rotation around Y-axis (1 value)
            label += f"{rotation_y:.4f} "
            # Bottom 4 vertices (8 values)
            label += " ".join([f"{v:.6f}" for v in vertices_norm])
            # Truncation and occlusion (2 values)
            label += f" {truncated:.6f} {occluded}"

            yolo_labels.append(label)

        return yolo_labels

    def copy_calibration(self, calib_file, output_name, split_name):
        """Copy calibration file to output (simplified format) under split directory."""
        calib = self.parse_calibration(calib_file)

        output_file = self.output_root / "calib" / split_name / f"{output_name}.txt"

        with open(output_file, "w") as f:
            f.write(f"fx: {calib['fx']:.6f}\n")
            f.write(f"fy: {calib['fy']:.6f}\n")
            f.write(f"cx: {calib['cx']:.6f}\n")
            f.write(f"cy: {calib['cy']:.6f}\n")
            f.write(f"right_cx: {calib['right_cx']:.6f}\n")
            f.write(f"right_cy: {calib['right_cy']:.6f}\n")
            f.write(f"baseline: {calib['baseline']:.6f}\n")
            f.write(f"image_width: {calib['image_width']}\n")
            f.write(f"image_height: {calib['image_height']}\n")

    def _compute_mean_dimensions(self, split_name="train"):
        """Compute mean dimensions from converted label files.
        
        Args:
            split_name: Split name to compute means from (typically "train")
        
        Returns:
            dict: Mapping from class name to mean dimensions [L, W, H] in meters
        """
        label_dir = self.output_root / "labels" / split_name
        if not label_dir.exists():
            LOGGER.warning(f"Label directory {label_dir} does not exist. Cannot compute mean dimensions.")
            return None
        
        # Collect dimensions per class
        # Format: {class_name: [[l, w, h], ...]}
        class_dimensions = {}
        
        # Get reverse class mapping (class_id -> class_name)
        # Handle filtered classes and remapping
        if self.filter_classes is not None:
            # Build reverse mapping for filtered classes
            id_to_name = {}
            for class_name, old_id in sorted(self.class_map.items(), key=lambda x: x[1]):
                if class_name in self.filter_classes:
                    new_id = self.class_id_remap[old_id]
                    id_to_name[new_id] = class_name
        else:
            id_to_name = {v: k for k, v in self.class_map.items()}
        
        # Iterate through all label files
        label_files = sorted(label_dir.glob("*.txt"))
        total_labels = 0
        
        for label_file in TQDM(label_files, desc="Computing mean dimensions"):
            with open(label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    
                    try:
                        class_id = int(float(parts[0]))
                        # Extract 3D dimensions from converted label format:
                        # Format: class x_l y_l w_l h_l x_r y_r w_r h_r dim_l dim_w dim_h ...
                        # Indices:   0    1   2   3   4   5   6   7   8    9    10    11
                        dim_l = float(parts[9])   # Length in meters
                        dim_w = float(parts[10])  # Width in meters
                        dim_h = float(parts[11])  # Height in meters

                        # Map class_id to class name
                        class_name = id_to_name.get(class_id)
                        if class_name is None:
                            continue

                        # Store dimensions as [L, W, H] (length, width, height)
                        if class_name not in class_dimensions:
                            class_dimensions[class_name] = []
                        class_dimensions[class_name].append([dim_l, dim_w, dim_h])
                        total_labels += 1
                    except (ValueError, IndexError) as e:
                        LOGGER.debug(f"Error parsing label line in {label_file}: {e}")
                        continue
        
        # Compute mean dimensions for each class
        mean_dims = {}
        for class_name, dims_list in class_dimensions.items():
            if len(dims_list) == 0:
                continue
            
            # Compute mean for each dimension
            dims_array = np.array(dims_list)
            mean_l = float(np.mean(dims_array[:, 0]))
            mean_w = float(np.mean(dims_array[:, 1]))
            mean_h = float(np.mean(dims_array[:, 2]))
            
            # Also compute standard deviation for each dimension
            std_l = float(np.std(dims_array[:, 0]))
            std_w = float(np.std(dims_array[:, 1]))
            std_h = float(np.std(dims_array[:, 2]))
            
            mean_dims[class_name] = [mean_l, mean_w, mean_h]
            LOGGER.info(f"  {class_name}: mean_dims = [L={mean_l:.2f}, W={mean_w:.2f}, H={mean_h:.2f}], std_dims = [L={std_l:.2f}, W={std_w:.2f}, H={std_h:.2f}] (from {len(dims_list)} samples)")
        
        LOGGER.info(f"Computed mean dimensions from {total_labels} labels across {len(mean_dims)} classes")
        return mean_dims if mean_dims else None

    def _compute_std_dimensions(self, split_name="train"):
        """Compute standard deviation of dimensions from converted label files.
        
        Args:
            split_name: Split name to compute std from (typically "train")
        
        Returns:
            dict: Mapping from class name to std dimensions [L, W, H] in meters
        """
        label_dir = self.output_root / "labels" / split_name
        if not label_dir.exists():
            LOGGER.warning(f"Label directory {label_dir} does not exist. Cannot compute std dimensions.")
            return None
        
        # Collect dimensions per class
        # Format: {class_name: [[l, w, h], ...]}
        class_dimensions = {}
        
        # Get reverse class mapping (class_id -> class_name)
        # Handle filtered classes and remapping
        if self.filter_classes is not None:
            # Build reverse mapping for filtered classes
            id_to_name = {}
            for class_name, old_id in sorted(self.class_map.items(), key=lambda x: x[1]):
                if class_name in self.filter_classes:
                    new_id = self.class_id_remap[old_id]
                    id_to_name[new_id] = class_name
        else:
            id_to_name = {v: k for k, v in self.class_map.items()}
        
        # Iterate through all label files
        label_files = sorted(label_dir.glob("*.txt"))
        total_labels = 0
        
        for label_file in TQDM(label_files, desc="Computing std dimensions"):
            with open(label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    
                    try:
                        class_id = int(float(parts[0]))
                        # Extract 3D dimensions from converted label format:
                        # Format: class x_l y_l w_l h_l x_r y_r w_r h_r dim_l dim_w dim_h ...
                        # Indices:   0    1   2   3   4   5   6   7   8    9    10    11
                        dim_l = float(parts[9])   # Length in meters
                        dim_w = float(parts[10])  # Width in meters
                        dim_h = float(parts[11])  # Height in meters

                        # Map class_id to class name
                        class_name = id_to_name.get(class_id)
                        if class_name is None:
                            continue

                        # Store dimensions as [L, W, H] (length, width, height)
                        if class_name not in class_dimensions:
                            class_dimensions[class_name] = []
                        class_dimensions[class_name].append([dim_l, dim_w, dim_h])
                        total_labels += 1
                    except (ValueError, IndexError) as e:
                        LOGGER.debug(f"Error parsing label line in {label_file}: {e}")
                        continue
        
        # Compute std for each class
        std_dims = {}
        for class_name, dims_list in class_dimensions.items():
            if len(dims_list) == 0:
                continue
            
            # Compute std for each dimension
            dims_array = np.array(dims_list)
            std_l = float(np.std(dims_array[:, 0]))
            std_w = float(np.std(dims_array[:, 1]))
            std_h = float(np.std(dims_array[:, 2]))
            
            std_dims[class_name] = [std_l, std_w, std_h]
            LOGGER.info(f" {class_name}: std_dims = [L={std_l:.2f}, W={std_w:.2f}, H={std_h:.2f}] (from {len(dims_list)} samples)")
        
        LOGGER.info(f"Computed std dimensions from {total_labels} labels across {len(std_dims)} classes")
        return std_dims if std_dims else None

    def _compute_std_dimensions(self, split_name="train"):
        """Compute standard deviation of dimensions from converted label files.
        
        Args:
            split_name: Split name to compute std from (typically "train")
        
        Returns:
            dict: Mapping from class name to std dimensions [L, W, H] in meters
        """
        label_dir = self.output_root / "labels" / split_name
        if not label_dir.exists():
            LOGGER.warning(f"Label directory {label_dir} does not exist. Cannot compute std dimensions.")
            return None
        
        # Collect dimensions per class
        # Format: {class_name: [[l, w, h], ...]}
        class_dimensions = {}
        
        # Get reverse class mapping (class_id -> class_name)
        # Handle filtered classes and remapping
        if self.filter_classes is not None:
            # Build reverse mapping for filtered classes
            id_to_name = {}
            for class_name, old_id in sorted(self.class_map.items(), key=lambda x: x[1]):
                if class_name in self.filter_classes:
                    new_id = self.class_id_remap[old_id]
                    id_to_name[new_id] = class_name
        else:
            id_to_name = {v: k for k, v in self.class_map.items()}
        
        # Iterate through all label files
        label_files = sorted(label_dir.glob("*.txt"))
        total_labels = 0
        
        for label_file in TQDM(label_files, desc="Computing std dimensions"):
            with open(label_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    
                    try:
                        class_id = int(float(parts[0]))
                        # Extract 3D dimensions from converted label format:
                        # Format: class x_l y_l w_l h_l x_r y_r w_r h_r dim_l dim_w dim_h ...
                        # Indices:   0    1   2   3   4   5   6   7   8    9    10    11
                        dim_l = float(parts[9])   # Length in meters
                        dim_w = float(parts[10])  # Width in meters
                        dim_h = float(parts[11])  # Height in meters

                        # Map class_id to class name
                        class_name = id_to_name.get(class_id)
                        if class_name is None:
                            continue

                        # Store dimensions as [L, W, H] (length, width, height)
                        if class_name not in class_dimensions:
                            class_dimensions[class_name] = []
                        class_dimensions[class_name].append([dim_l, dim_w, dim_h])
                        total_labels += 1
                    except (ValueError, IndexError) as e:
                        LOGGER.debug(f"Error parsing label line in {label_file}: {e}")
                        continue
        
        # Compute std for each class
        std_dims = {}
        for class_name, dims_list in class_dimensions.items():
            if len(dims_list) == 0:
                continue
            
            # Compute std for each dimension
            dims_array = np.array(dims_list)
            std_l = float(np.std(dims_array[:, 0]))
            std_w = float(np.std(dims_array[:, 1]))
            std_h = float(np.std(dims_array[:, 2]))
            
            std_dims[class_name] = [std_l, std_w, std_h]
            LOGGER.info(f"  {class_name}: std_dims = [L={std_l:.2f}, W={std_w:.2f}, H={std_h:.2f}] (from {len(dims_list)} samples)")
        
        LOGGER.info(f"Computed std dimensions from {total_labels} labels across {len(std_dims)} classes")
        return std_dims if std_dims else None

    def convert_split(self, split="training"):
        """Convert entire KITTI split.

        Args:
            split: 'training' or 'testing'
        Behavior:
            - 'single' strategy: mirrors KITTI splits to train/val.
            - '3dop' strategy: when split=='training', internally slices indices 0-3711 -> train, 3712-end -> val.
        """
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Converting KITTI {split} split (strategy={self.split_strategy})")
        LOGGER.info(f"{'='*60}\n")

        # Paths
        image_2_dir = self.kitti_root / split / "image_2"
        image_3_dir = self.kitti_root / split / "image_3"
        label_dir = self.kitti_root / split / "label_2"
        calib_dir = self.kitti_root / split / "calib"

        image_files = sorted(image_2_dir.glob("*.png"))
        total = len(image_files)
        LOGGER.info(f"Found {total} images in raw '{split}' split")

        # 3DOP branch
        if self.split_strategy == "3dop" and split == "training":
            TRAIN_CUTOFF = 3712  # indices 0..3711 inclusive
            if total < TRAIN_CUTOFF:
                LOGGER.error(f"Not enough images ({total}) for 3DOP split (need >= {TRAIN_CUTOFF}).")
                return

            # Prepare directories for both subsets
            for subset in ["train", "val"]:
                for d in [f"images/{subset}/left", f"images/{subset}/right", f"labels/{subset}", f"calib/{subset}"]:
                    (self.output_root / d).mkdir(parents=True, exist_ok=True)

            train_index, val_index = [], []
            for idx, img_file in enumerate(TQDM(image_files, desc="Converting (3DOP)")):
                img_id = img_file.stem
                subset = "train" if idx < TRAIN_CUTOFF else "val"

                left_img = image_2_dir / f"{img_id}.png"
                right_img = image_3_dir / f"{img_id}.png"
                label_file = label_dir / f"{img_id}.txt"
                calib_file = calib_dir / f"{img_id}.txt"

                if not left_img.exists() or not right_img.exists():
                    LOGGER.warning(f"Missing images for {img_id}, skipping")
                    continue
                if not calib_file.exists():
                    LOGGER.warning(f"Missing calibration for {img_id}, skipping")
                    continue

                if label_file.exists():
                    yolo_labels = self.convert_label(str(label_file), str(calib_file))
                    with open(self.output_root / "labels" / subset / f"{img_id}.txt", "w") as f:
                        f.write("\n".join(yolo_labels))

                self.copy_calibration(str(calib_file), img_id, subset)

                try:
                    out_left = self.output_root / "images" / subset / "left" / f"{img_id}.png"
                    out_right = self.output_root / "images" / subset / "right" / f"{img_id}.png"
                    if not out_left.exists():
                        shutil.copy2(left_img, out_left)
                    if not out_right.exists():
                        shutil.copy2(right_img, out_right)
                except Exception as e:
                    LOGGER.warning(f"Failed to copy images for {img_id}: {e}")

                rel_left = f"images/{subset}/left/{img_id}.png"
                (train_index if subset == "train" else val_index).append(rel_left)

            # Write index files
            with open(self.output_root / "train.txt", "w") as f:
                f.write("\n".join(train_index))
            with open(self.output_root / "val.txt", "w") as f:
                f.write("\n".join(val_index))

            LOGGER.info(f"3DOP conversion complete: train={len(train_index)} val={len(val_index)}")
            
            # Compute mean and std dimensions from training split
            LOGGER.info("\nComputing mean dimensions from training split...")
            self.mean_dims = self._compute_mean_dimensions("train")
            LOGGER.info("\nComputing std dimensions from training split...")
            self.std_dims = self._compute_std_dimensions("train")
            return

        # ----- Single split behavior -----
        split_name = SPLIT_MAP.get(split, split)

        # Ensure directories exist
        for d in [f"images/{split_name}/left", f"images/{split_name}/right", f"labels/{split_name}", f"calib/{split_name}"]:
            (self.output_root / d).mkdir(parents=True, exist_ok=True)

        index_list = []
        for img_file in TQDM(image_files, desc="Converting"):
            img_id = img_file.stem
            left_img = image_2_dir / f"{img_id}.png"
            right_img = image_3_dir / f"{img_id}.png"
            label_file = label_dir / f"{img_id}.txt"
            calib_file = calib_dir / f"{img_id}.txt"

            if not left_img.exists() or not right_img.exists():
                LOGGER.warning(f"Missing images for {img_id}, skipping")
                continue
            if not calib_file.exists():
                LOGGER.warning(f"Missing calibration for {img_id}, skipping")
                continue

            if label_file.exists():
                yolo_labels = self.convert_label(str(label_file), str(calib_file))
                with open(self.output_root / "labels" / split_name / f"{img_id}.txt", "w") as f:
                    f.write("\n".join(yolo_labels))

            self.copy_calibration(str(calib_file), img_id, split_name)

            try:
                out_left = self.output_root / "images" / split_name / "left" / f"{img_id}.png"
                out_right = self.output_root / "images" / split_name / "right" / f"{img_id}.png"
                if not out_left.exists():
                    shutil.copy2(left_img, out_left)
                if not out_right.exists():
                    shutil.copy2(right_img, out_right)
            except Exception as e:
                LOGGER.warning(f"Failed to copy images for {img_id}: {e}")

            index_list.append(f"images/{split_name}/left/{img_id}.png")

        with open(self.output_root / f"{split_name}.txt", "w") as f:
            f.write("\n".join(index_list))

        LOGGER.info(f"Conversion complete for split '{split_name}'")
        LOGGER.info(f"Output directory: {self.output_root}")
        
        # Compute mean dimensions from training split if this is the training split
        if split_name == "train" and self.mean_dims is None:
            LOGGER.info("\nComputing mean dimensions from training split...")
            self.mean_dims = self._compute_mean_dimensions("train")

    def create_dataset_yaml(self, splits=["train", "val"]):
        """Create dataset.yaml for YOLO training."""
        # Build class names based on filter - use OrderedDict to preserve insertion order
        if self.filter_classes is not None:
            # Remap class IDs to be consecutive starting from 0
            filtered_class_map = {}
            class_names = OrderedDict()
            new_id = 0
            for class_name, old_id in sorted(self.class_map.items(), key=lambda x: x[1]):
                if class_name in self.filter_classes:
                    filtered_class_map[class_name] = new_id
                    class_names[new_id] = class_name
                    new_id += 1
            num_classes = len(class_names)
        else:
            # Use OrderedDict to preserve order based on class IDs (0, 1, 2, ...)
            # class_names should map: {class_id: class_name}
            class_names = OrderedDict()
            for old_id in sorted(self.class_map.values()):
                # Find class name for this ID
                class_name = next(name for name, id_ in self.class_map.items() if id_ == old_id)
                class_names[old_id] = class_name
            num_classes = len(self.class_map)

        # Build names section - OrderedDict preserves order
        names_section = "\n".join([f"  {k}: {v}" for k, v in class_names.items()])

        # Determine available splits dynamically
        available_splits = []
        for s in splits:
            left_dir = self.output_root / "images" / s / "left"
            if left_dir.exists() and any(left_dir.glob("*.png")):
                available_splits.append(s)

        # Fallback: if val missing, duplicate train for val reference (YOLO requirement)
        train_ref = "train.txt" if "train" in available_splits else None
        val_ref = "val.txt" if "val" in available_splits else ("train.txt" if train_ref else None)

        yaml_lines = ["# KITTI 3D Object Detection Dataset (Stereo)", f"path: {self.output_root}"]
        if train_ref:
            yaml_lines.append(f"train: {train_ref}")
        if val_ref:
            yaml_lines.append(f"val: {val_ref}")
        yaml_lines.extend([
            "# Classes",
            "names:",
            names_section,
            "# Dataset info",
            f"nc: {num_classes}  # number of classes",
            "stereo: true",
            "image_size: [375, 1242]  # height, width",
            "# Calibration",
            "baseline: 0.54  # meters (approximate)",
            "focal_length: 721.5  # pixels (approximate)",
            "channels: 6",
        ])
        
        # Add mean dimensions if computed - use integer keys to match class IDs
        if self.mean_dims is not None and len(self.mean_dims) > 0:
            yaml_lines.append("# Mean dimensions per class [L, W, H] in meters")
            yaml_lines.append("mean_dims:")
            # Use the same order as class_names to ensure 1-to-1 correspondence
            for class_id, class_name in class_names.items():
                if class_name in self.mean_dims:
                    dims = self.mean_dims[class_name]
                    yaml_lines.append(f"  {class_id}: [{dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f}]  # {class_name}")
                    LOGGER.info(f"  Class {class_id} ({class_name}): mean_dims = [L={dims[0]:.2f}, W={dims[1]:.2f}, H={dims[2]:.2f}]")
                else:
                    LOGGER.warning(f"  Missing mean_dims for class {class_id} ({class_name})")

        # Add std dimensions if computed - use integer keys to match class IDs
        if hasattr(self, 'std_dims') and self.std_dims is not None and len(self.std_dims) > 0:
            yaml_lines.append("# Standard deviation dimensions per class [L, W, H] in meters")
            yaml_lines.append("std_dims:")
            # Use the same order as class_names to ensure 1-to-1 correspondence
            for class_id, class_name in class_names.items():
                if class_name in self.std_dims:
                    dims = self.std_dims[class_name]
                    yaml_lines.append(f"  {class_id}: [{dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f}]  # {class_name}")
                    LOGGER.info(f"  Class {class_id} ({class_name}): std_dims = [L={dims[0]:.2f}, W={dims[1]:.2f}, H={dims[2]:.2f}]")
                else:
                    LOGGER.warning(f"  Missing std_dims for class {class_id} ({class_name})")

        yaml_content = "\n".join(yaml_lines) + "\n"

        yaml_file = self.output_root / "dataset.yaml"
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        LOGGER.info(f"\nCreated dataset.yaml at {yaml_file}")


def main():
    """Main conversion script. Converts KITTI training split using 3DOP strategy to generate train and val splits."""
    parser = argparse.ArgumentParser(
        description="Convert KITTI to YOLO 3D format using 3DOP split strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all classes
  python convert_kitti_3d.py --kitti-root /path/to/kitti

  # Convert only specific classes (e.g., Car, Pedestrian, Cyclist)
  python convert_kitti_3d.py --kitti-root /path/to/kitti --filter-classes Car Pedestrian Cyclist

The script will:
  - Process the KITTI training split
  - Use 3DOP strategy: indices 0-3711 -> train, 3712+ -> val
  - Output converted dataset to the same directory as --kitti-root
  - Include all classes by default, or only specified classes if --filter-classes is used
  - Available classes: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc
        """,
    )
    parser.add_argument("--kitti-root", type=str, required=True, help="Path to KITTI dataset root directory")
    parser.add_argument(
        "--filter-classes",
        type=str,
        nargs="+",
        default=None,
        help="List of class names to include (e.g., Car Pedestrian Cyclist). If not specified, all classes are included. Available: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc",
    )
    parser.add_argument(
        "--filter-occluded",
        action="store_true",
        help="Filter out heavily occluded objects during conversion. Excludes objects with occlusion level > max-occlusion-level",
    )
    parser.add_argument(
        "--max-occlusion-level",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Maximum occlusion level to include when using --filter-occluded. KITTI levels: 0=fully visible, 1=partially occluded, 2=heavily occluded, 3=unknown. Default is 1 (excludes heavily occluded and unknown objects).",
    )

    args = parser.parse_args()

    # Output to same directory as input
    output_root = args.kitti_root

    # Initialize converter with 3DOP strategy and optional class/occlusion filtering
    converter = KITTIToYOLO3D(
        args.kitti_root,
        output_root,
        filter_classes=args.filter_classes,  # None = all classes, or list of class names
        split_strategy="3dop",  # Always use 3DOP split strategy
        filter_occluded=args.filter_occluded,  # Filter heavily occluded objects
        max_occlusion_level=args.max_occlusion_level,  # Max occlusion level to include
    )

    # Always convert training split (which creates both train and val with 3DOP strategy)
    converter.convert_split("training")

    # Create dataset.yaml
    converter.create_dataset_yaml()

    LOGGER.info("\n✓ Conversion complete!")
    LOGGER.info(f"\nYour dataset is ready at: {output_root}")
    LOGGER.info("\nDirectory structure:")
    LOGGER.info("  images/train/left/  - Train left images")
    LOGGER.info("  images/train/right/ - Train right images")
    LOGGER.info("  images/val/left/    - Val left images")
    LOGGER.info("  images/val/right/   - Val right images")
    LOGGER.info("  labels/train/       - Train labels")
    LOGGER.info("  labels/val/         - Val labels")
    LOGGER.info("  calib/train/        - Train calibration")
    LOGGER.info("  calib/val/          - Val calibration")
    LOGGER.info("  train.txt / val.txt - Image index lists")
    LOGGER.info("  dataset.yaml        - Dataset configuration")


if __name__ == "__main__":
    main()