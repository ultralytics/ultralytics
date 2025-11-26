#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
KITTI to YOLO 3D Stereo Format Converter

This script converts KITTI dataset to YOLO 3D stereo format with 19 values per object:
class x_l y_l w_l h_l x_r w_r dim_h dim_w dim_l alpha v1_x v1_y v2_x v2_y v3_x v3_y v4_x v4_y

All coordinates normalized to [0, 1], dimensions in meters, alpha in radians.

Usage:
    python convert_kitti_3d.py --kitti-root /path/to/kitti --output-root /path/to/output
    python convert_kitti_3d.py --kitti-root /path/to/kitti --output-root /path/to/output --split training
"""

import argparse
import os
from pathlib import Path

import numpy as np

try:
    from ultralytics.utils import LOGGER, TQDM
except ImportError:
    # Fallback if not in ultralytics environment
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


class KITTIToYOLO3D:
    """Convert KITTI dataset to YOLO 3D Stereo format."""

    def __init__(self, kitti_root, output_root, filter_classes=None):
        """
        Args:
            kitti_root: Path to KITTI dataset root
            output_root: Path to output directory
            filter_classes: List of class names to include (None = include all)
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

        # Filter classes (if specified)
        self.filter_classes = filter_classes
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

        # Create output directories
        self._setup_output_dirs()

    def _setup_output_dirs(self):
        """Create output directory structure."""
        dirs = ["images/left", "images/right", "labels", "calib"]
        for d in dirs:
            (self.output_root / d).mkdir(parents=True, exist_ok=True)

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

        # Extract intrinsics
        fx = P2[0, 0]
        fy = P2[1, 1]
        cx = P2[0, 2]
        cy = P2[1, 2]

        # Calculate baseline
        baseline = (P2[0, 3] - P3[0, 3]) / fx

        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": abs(baseline), "P2": P2, "P3": P3}

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
        Compute right image 2D box by projecting 3D box.

        Returns:
            center_x_r, width_r (in pixels)
        """
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
            v_r = calib["fy"] * (corner[1] / corner[2]) + calib["cy"]
            corners_2d_right.append([u_r, v_r])

        if len(corners_2d_right) == 0:
            # Fallback: simple disparity-based calculation
            x1_l, y1_l, x2_l, y2_l = left_box_2d
            center_x_l = (x1_l + x2_l) / 2
            disparity = (calib["fx"] * calib["baseline"]) / Z
            center_x_r = center_x_l - disparity
            width_r = x2_l - x1_l
            return center_x_r, width_r

        corners_2d_right = np.array(corners_2d_right)

        # Get bounding box
        x_min = corners_2d_right[:, 0].min()
        x_max = corners_2d_right[:, 0].max()

        center_x_r = (x_min + x_max) / 2
        width_r = x_max - x_min

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
            width_r_norm = width_r / self.img_width

            # Clamp to valid range
            center_x_r_norm = np.clip(center_x_r_norm, 0, 1)
            width_r_norm = np.clip(width_r_norm, 0, 1)

            # ===== Bottom 4 Vertices (normalized) =====
            vertices_2d = self.compute_bottom_vertices(X, Y, Z, h, w, l, rotation_y, calib)

            # Normalize vertices
            vertices_norm = []
            for v in vertices_2d:
                v_x = np.clip(v[0] / self.img_width, 0, 1)
                v_y = np.clip(v[1] / self.img_height, 0, 1)
                vertices_norm.extend([v_x, v_y])

            # ===== Build YOLO label line =====
            label = f"{class_id} "
            label += f"{center_x_l_norm:.6f} {center_y_l_norm:.6f} "
            label += f"{width_l_norm:.6f} {height_l_norm:.6f} "
            label += f"{center_x_r_norm:.6f} {width_r_norm:.6f} "
            label += f"{h:.2f} {w:.2f} {l:.2f} "
            label += f"{alpha:.4f} "
            label += " ".join([f"{v:.6f}" for v in vertices_norm])

            yolo_labels.append(label)

        return yolo_labels

    def copy_calibration(self, calib_file, output_name):
        """Copy calibration file to output (simplified format)."""
        calib = self.parse_calibration(calib_file)

        output_file = self.output_root / "calib" / f"{output_name}.txt"

        with open(output_file, "w") as f:
            f.write(f"fx: {calib['fx']:.6f}\n")
            f.write(f"fy: {calib['fy']:.6f}\n")
            f.write(f"cx: {calib['cx']:.6f}\n")
            f.write(f"cy: {calib['cy']:.6f}\n")
            f.write(f"baseline: {calib['baseline']:.6f}\n")

    def convert_split(self, split="training"):
        """
        Convert entire KITTI split (training or testing).

        Args:
            split: 'training' or 'testing'
        """
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Converting KITTI {split} split")
        LOGGER.info(f"{'='*60}\n")

        # Paths
        image_2_dir = self.kitti_root / split / "image_2"
        image_3_dir = self.kitti_root / split / "image_3"
        label_dir = self.kitti_root / split / "label_2"
        calib_dir = self.kitti_root / split / "calib"

        # Get all image files
        image_files = sorted(image_2_dir.glob("*.png"))

        LOGGER.info(f"Found {len(image_files)} images")

        # Process each image
        for img_file in TQDM(image_files, desc="Converting"):
            img_id = img_file.stem

            # File paths
            left_img = image_2_dir / f"{img_id}.png"
            right_img = image_3_dir / f"{img_id}.png"
            label_file = label_dir / f"{img_id}.txt"
            calib_file = calib_dir / f"{img_id}.txt"

            # Check if files exist
            if not left_img.exists() or not right_img.exists():
                LOGGER.warning(f"Missing images for {img_id}, skipping")
                continue

            if not calib_file.exists():
                LOGGER.warning(f"Missing calibration for {img_id}, skipping")
                continue

            # Convert labels (if training split)
            if split == "training" and label_file.exists():
                yolo_labels = self.convert_label(str(label_file), str(calib_file))

                # Write labels
                output_label = self.output_root / "labels" / f"{img_id}.txt"
                with open(output_label, "w") as f:
                    f.write("\n".join(yolo_labels))

            # Copy calibration
            self.copy_calibration(str(calib_file), img_id)

            # Note: Images can be symlinked or copied
            # For now, we just note their location
            # You can add shutil.copy2 if you want to copy them

        LOGGER.info(f"\nConversion complete!")
        LOGGER.info(f"Output directory: {self.output_root}")

    def create_dataset_yaml(self, splits=["train", "val"]):
        """Create dataset.yaml for YOLO training."""
        # Build class names based on filter
        if self.filter_classes is not None:
            # Remap class IDs to be consecutive starting from 0
            filtered_class_map = {}
            class_names = {}
            new_id = 0
            for class_name, old_id in sorted(self.class_map.items(), key=lambda x: x[1]):
                if class_name in self.filter_classes:
                    filtered_class_map[class_name] = new_id
                    class_names[new_id] = class_name
                    new_id += 1
            num_classes = len(class_names)
        else:
            class_names = {v: k for k, v in self.class_map.items()}
            num_classes = len(self.class_map)

        # Build names section
        names_section = "\n".join([f"  {k}: {v}" for k, v in sorted(class_names.items())])

        yaml_content = f"""# KITTI 3D Object Detection Dataset (Stereo)
path: {self.output_root}
train: train.txt
val: val.txt
# Classes
names:
{names_section}
# Dataset info
nc: {num_classes}  # number of classes
stereo: true
image_size: [375, 1242]  # height, width
# Calibration
baseline: 0.54  # meters (approximate)
focal_length: 721.5  # pixels (approximate)
"""

        yaml_file = self.output_root / "dataset.yaml"
        with open(yaml_file, "w") as f:
            f.write(yaml_content)

        LOGGER.info(f"\nCreated dataset.yaml at {yaml_file}")


def main():
    """Main conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert KITTI to YOLO 3D format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all classes
  python convert_kitti_3d.py --kitti-root /path/to/kitti --output-root /path/to/output

  # Convert only Car and Pedestrian classes
  python convert_kitti_3d.py --kitti-root /path/to/kitti --output-root /path/to/output --filter-classes Car Pedestrian

  # Convert only vehicle classes
  python convert_kitti_3d.py --kitti-root /path/to/kitti --output-root /path/to/output --filter-classes Car Van Truck
        """,
    )
    parser.add_argument("--kitti-root", type=str, required=True, help="Path to KITTI dataset root directory")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Path to output directory (default: same as --kitti-root)",
    )
    parser.add_argument(
        "--split", type=str, default="training", choices=["training", "testing"], help="Which split to convert"
    )
    parser.add_argument(
        "--filter-classes",
        type=str,
        nargs="+",
        default=None,
        help="List of class names to include (e.g., Car Pedestrian). Available: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc",
    )

    args = parser.parse_args()

    # Determine output root fallback
    output_root = args.output_root or args.kitti_root
    if args.output_root is None:
        LOGGER.info("--output-root not provided. Using --kitti-root as output directory.")

    # Initialize converter
    converter = KITTIToYOLO3D(args.kitti_root, output_root, filter_classes=args.filter_classes)

    # Convert dataset
    converter.convert_split(args.split)

    # Create dataset.yaml
    converter.create_dataset_yaml()

    LOGGER.info("\n✓ Conversion complete!")
    LOGGER.info(f"\nYour dataset is ready at: {output_root}")
    LOGGER.info("\nDirectory structure:")
    LOGGER.info("  images/left/    - Left camera images (reference)")
    LOGGER.info("  images/right/   - Right camera images (reference)")
    LOGGER.info("  labels/         - YOLO 3D format labels")
    LOGGER.info("  calib/          - Calibration files")
    LOGGER.info("  dataset.yaml    - Dataset configuration")


if __name__ == "__main__":
    main()