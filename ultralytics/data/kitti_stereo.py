# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ultralytics.utils import LOGGER


class KITTIStereoDataset:
    """Dataset class for loading KITTI stereo images, calibration, and labels.

    This class loads stereo image pairs (left/right), calibration files, and YOLO 3D format labels
    from a KITTI-style dataset structure.

    Attributes:
        root (Path): Root directory of the dataset.
        split (str): Dataset split ('train' or 'val').
        image_ids (list[str]): List of image identifiers.
        left_img_dir (Path): Directory containing left camera images.
        right_img_dir (Path): Directory containing right camera images.
        label_dir (Path): Directory containing label files.
        calib_dir (Path): Directory containing calibration files.

    Examples:
        >>> dataset = KITTIStereoDataset(root="kitti-stereo-debug", split="train")
        >>> sample = dataset[0]
        >>> left_img = sample["left_img"]
        >>> right_img = sample["right_img"]
        >>> labels = sample["labels"]
        >>> calib = sample["calib"]
    """

    def __init__(self, root: str | Path, split: str = "train", max_samples: int | None = None):
        """Initialize KITTI Stereo Dataset.

        Args:
            root (str | Path): Root directory of the dataset. Should contain:
                - images/{split}/left/ and images/{split}/right/
                - labels/{split}/
                - calib/{split}/
            split (str): Dataset split, either 'train' or 'val'. Defaults to 'train'.
            max_samples (int | None): Maximum number of samples to load. If None, loads all available samples.
                If specified, only the first max_samples samples will be loaded. Defaults to None.
        """
        self.root = Path(root)
        self.split = split

        # Set up directory paths
        self.left_img_dir = self.root / "images" / split / "left"
        self.right_img_dir = self.root / "images" / split / "right"
        self.label_dir = self.root / "labels" / split
        self.calib_dir = self.root / "calib" / split

        # Validate directories exist
        if not self.left_img_dir.exists():
            raise FileNotFoundError(f"Left image directory not found: {self.left_img_dir}")
        if not self.right_img_dir.exists():
            raise FileNotFoundError(f"Right image directory not found: {self.right_img_dir}")
        if not self.calib_dir.exists():
            raise FileNotFoundError(f"Calibration directory not found: {self.calib_dir}")

        # Discover image files
        self.image_ids = self._get_image_files()

        if len(self.image_ids) == 0:
            raise ValueError(f"No images found in {self.left_img_dir}")

        # Apply max_samples limit if specified (T188, T189, T190, T191)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError(f"max_samples must be > 0, got {max_samples}")
            total_samples = len(self.image_ids)
            self.image_ids = self.image_ids[:max_samples]
            if len(self.image_ids) < total_samples:
                LOGGER.info(f"Limited dataset to {len(self.image_ids)} samples (from {total_samples} total)")

        LOGGER.info(f"KITTI Stereo Dataset initialized: {len(self.image_ids)} samples in '{split}' split")

    def _get_image_files(self) -> list[str]:
        """Discover and pair left/right images.

        Returns:
            list[str]: List of image identifiers (without extension) that have both left and right images.
        """
        # Get all left image files
        left_files = sorted(self.left_img_dir.glob("*.png"))
        if not left_files:
            left_files = sorted(self.left_img_dir.glob("*.jpg"))

        image_ids = []
        for left_file in left_files:
            image_id = left_file.stem
            right_file = self.right_img_dir / left_file.name

            # Check if corresponding right image exists
            if right_file.exists():
                image_ids.append(image_id)
            else:
                LOGGER.warning(f"Missing right image for {image_id}, skipping")

        return image_ids

    def _parse_calibration(self, calib_file: Path) -> dict[str, Any]:
        """Parse calibration file into a structured dictionary.

        Supports both original KITTI format (P0..P3, R0_rect, Tr_*) and the simplified
        converted format (fx, fy, cx, cy, right_cx, right_cy, baseline, image_width, image_height).

        Args:
            calib_file (Path): Path to calibration file.

        Returns:
            dict[str, Any]: Dictionary containing intrinsics and (when available) matrices.
        """
        with open(calib_file, "r") as f:
            lines = f.readlines()

        calib_dict = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(":")
            if len(parts) != 2:
                continue

            key = parts[0].strip()
            value_str = parts[1].strip()

            # Simplified format keys (single numeric value)
            if key in {"fx", "fy", "cx", "cy", "right_cx", "right_cy", "baseline"}:
                try:
                    calib_dict[key] = float(value_str)
                except ValueError:
                    continue
                continue
            if key in {"image_width", "image_height"}:
                try:
                    calib_dict[key] = int(float(value_str))
                except ValueError:
                    continue
                continue

            # Original KITTI format keys (lists of numbers)
            try:
                values = [float(x) for x in value_str.split()]
            except ValueError:
                continue

            if key == "P0":
                calib_dict["P0"] = np.array(values).reshape(3, 4)
            elif key == "P1":
                calib_dict["P1"] = np.array(values).reshape(3, 4)
            elif key == "P2":
                calib_dict["P2"] = np.array(values).reshape(3, 4)
            elif key == "P3":
                calib_dict["P3"] = np.array(values).reshape(3, 4)
            elif key == "R0_rect":
                calib_dict["R0_rect"] = np.array(values).reshape(3, 3)
            elif key == "Tr_velo_to_cam":
                calib_dict["Tr_velo_to_cam"] = np.array(values).reshape(3, 4)
            elif key == "Tr_imu_to_velo":
                calib_dict["Tr_imu_to_velo"] = np.array(values).reshape(3, 4)

        # Extract intrinsics from P2 (left camera) if not provided directly
        if "P2" in calib_dict and not all(k in calib_dict for k in ("fx", "fy", "cx", "cy")):
            P2 = calib_dict["P2"]
            calib_dict["fx"] = P2[0, 0]
            calib_dict["fy"] = P2[1, 1]
            calib_dict["cx"] = P2[0, 2]
            calib_dict["cy"] = P2[1, 2]

        # Extract right camera principal point if available in P3 and not provided directly
        if "P3" in calib_dict and not all(k in calib_dict for k in ("right_cx", "right_cy")):
            P3 = calib_dict["P3"]
            calib_dict["right_cx"] = P3[0, 2]
            calib_dict["right_cy"] = P3[1, 2]

        # Calculate baseline from P2 and P3 if not provided directly
        if "P2" in calib_dict and "P3" in calib_dict and "fx" in calib_dict and "baseline" not in calib_dict:
            P2 = calib_dict["P2"]
            P3 = calib_dict["P3"]
            fx = calib_dict["fx"]
            baseline = (P2[0, 3] - P3[0, 3]) / fx
            calib_dict["baseline"] = abs(baseline)

        return calib_dict

    def _parse_labels(self, label_file: Path) -> list[dict[str, Any]]:
        """Parse YOLO 3D format label file.

        Format (26 values): class x_l y_l w_l h_l x_r y_r w_r h_r dim_l dim_w dim_h loc_x loc_y loc_z rot_y kp1_x kp1_y kp2_x kp2_y kp3_x kp3_y kp4_x kp4_y truncated occluded

        Args:
            label_file (Path): Path to label file.

        Returns:
            list[dict[str, Any]]: List of label dictionaries, each containing:
                - class_id: Class ID (int) - remapped if filter_classes=True
                - original_class_id: Original class ID (int) - only if filter_classes=True
                - left_box: Left image 2D box [center_x, center_y, width, height] (normalized)
                - right_box: Right image 2D box [center_x, center_y, width, height] (normalized)
                - dimensions: 3D dimensions [height, width, length] in meters
                - rotation_y: Global rotation around Y-axis in radians
                - vertices: Bottom 4 vertices of 3D box [v1_x, v1_y, v2_x, v2_y, v3_x, v3_y, v4_x, v4_y] (normalized)
                - location_3d: 3D location [X, Y, Z] in camera coordinates (Y is bottom center)
                - truncated: Truncation level [0.0, 1.0] (float)
                - occluded: Occlusion level [0, 1, 2, 3] (int)
        """
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")

        labels = []
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                # 26-value format: class x_l y_l w_l h_l x_r y_r w_r h_r dim_l dim_w dim_h loc_x loc_y loc_z rot_y kp1_x kp1_y kp2_x kp2_y kp3_x kp3_y kp4_x kp4_y truncated occluded
                if len(parts) != 26:
                    LOGGER.warning(f"Invalid label format in {label_file}: expected 26 values, got {len(parts)}")
                    continue
                
                
                values = [float(x) for x in parts]
                original_class_id = int(values[0])
                class_id = original_class_id

                # 26-value format structure:
                # 0: class
                # 1-4: x_l y_l w_l h_l (left box)
                # 5-8: x_r y_r w_r h_r (right box)
                # 9-11: dim_l dim_w dim_h (dimensions: length, width, height)
                # 12-14: loc_x loc_y loc_z (3D location)
                # 15: rot_y (rotation around Y-axis)
                # 16-23: kp1_x kp1_y kp2_x kp2_y kp3_x kp3_y kp4_x kp4_y (vertices)
                # 24: truncated (float, [0.0, 1.0])
                # 25: occluded (int, [0, 1, 2, 3])
                
                label_dict = {
                    "class_id": class_id,
                    "left_box": {
                        "center_x": values[1],
                        "center_y": values[2],
                        "width": values[3],
                        "height": values[4],
                    },
                    "right_box": {
                        "center_x": values[5],
                        "center_y": values[6],
                        "width": values[7],
                        "height": values[8],
                    },
                    "dimensions": {
                        "length": values[9],   # dim_l
                        "width": values[10],  # dim_w
                        "height": values[11], # dim_h
                    },
                    "rotation_y": values[15],  # rot_y (global yaw)
                    "vertices": {
                        "v1": [values[16], values[17]],  # kp1
                        "v2": [values[18], values[19]],  # kp2
                        "v3": [values[20], values[21]],  # kp3
                        "v4": [values[22], values[23]],  # kp4
                    },
                    "location_3d": {
                        "x": values[12],  # loc_x
                        "y": values[13],  # loc_y (bottom center Y in KITTI coords)
                        "z": values[14],  # loc_z (depth)
                    },
                    "truncated": float(values[24]),  # truncated (float, [0.0, 1.0])
                    "occluded": int(values[25]),  # occluded (int, [0, 1, 2, 3])
                }
                
                # Assertions
                assert -np.pi <= label_dict['rotation_y'] <= np.pi, f"rotation_y is out of range: {label_dict['rotation_y']}"
                assert 0.1 < label_dict['dimensions']['height'] < 5 and 0.1 < label_dict['dimensions']['width'] < 3 and 0.1 < label_dict['dimensions']['length'] < 20, f"dimensions are out of range: {label_dict['dimensions']}"

                labels.append(label_dict)

        return labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict[str, Any]: Dictionary containing:
                - left_img: Left camera image as numpy array (BGR format)
                - right_img: Right camera image as numpy array (BGR format)
                - labels: List of parsed label dictionaries
                - calib: Dictionary with parsed calibration data
                - calib_file: Path to raw calibration file
                - image_id: String identifier (e.g., "000000")
        """
        image_id = self.image_ids[idx]

        # Load images
        left_img_path = self.left_img_dir / f"{image_id}.png"
        if not left_img_path.exists():
            left_img_path = self.left_img_dir / f"{image_id}.jpg"

        right_img_path = self.right_img_dir / left_img_path.name

        left_img = cv2.imread(str(left_img_path))
        right_img = cv2.imread(str(right_img_path))

        if left_img is None:
            raise FileNotFoundError(f"Could not load left image: {left_img_path}")
        if right_img is None:
            raise FileNotFoundError(f"Could not load right image: {right_img_path}")

        # Load calibration
        calib_file = self.calib_dir / f"{image_id}.txt"
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")

        calib = self._parse_calibration(calib_file)

        # Load labels
        label_file = self.label_dir / f"{image_id}.txt"
        labels = self._parse_labels(label_file)

        return {
            "left_img": left_img,
            "right_img": right_img,
            "labels": labels,
            "calib": calib,
            "calib_file": str(calib_file),
            "image_id": image_id,
        }
