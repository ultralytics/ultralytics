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

    def __init__(self, root: str | Path, split: str = "train", filter_classes: bool = False):
        """Initialize KITTI Stereo Dataset.

        Args:
            root (str | Path): Root directory of the dataset. Should contain:
                - images/{split}/left/ and images/{split}/right/
                - labels/{split}/
                - calib/{split}/
            split (str): Dataset split, either 'train' or 'val'. Defaults to 'train'.
            filter_classes (bool): If True, filter to only paper classes (Car, Pedestrian, Cyclist)
                and remap class IDs from 0,3,5 to 0,1,2. Defaults to False.
        """
        self.root = Path(root)
        self.split = split
        self.filter_classes = filter_classes

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

        Format: class x_l y_l w_l h_l x_r w_r dim_h dim_w dim_l alpha v1_x v1_y v2_x v2_y v3_x v3_y v4_x v4_y

        Args:
            label_file (Path): Path to label file.

        Returns:
            list[dict[str, Any]]: List of label dictionaries, each containing:
                - class_id: Class ID (int) - remapped if filter_classes=True
                - original_class_id: Original class ID (int) - only if filter_classes=True
                - left_box: Left image 2D box [center_x, center_y, width, height] (normalized)
                - right_box: Right image 2D box [center_x, width] (normalized)
                - dimensions: 3D dimensions [height, width, length] in meters
                - alpha: Observation angle in radians
                - vertices: Bottom 4 vertices of 3D box [v1_x, v1_y, v2_x, v2_y, v3_x, v3_y, v4_x, v4_y] (normalized)
        """
        if not label_file.exists():
            return []

        # Import class mapping utilities
        if self.filter_classes:
            from ultralytics.models.yolo.stereo3ddet.utils import filter_and_remap_class_id

        labels = []
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 19:
                    LOGGER.warning(f"Invalid label format in {label_file}: expected 19 values, got {len(parts)}")
                    continue

                try:
                    values = [float(x) for x in parts]
                    original_class_id = int(values[0])

                    # Filter and remap class ID if filtering is enabled
                    if self.filter_classes:
                        remapped_class_id = filter_and_remap_class_id(original_class_id)
                        if remapped_class_id is None:
                            # Class is not in paper set, skip it
                            # LOGGER.warning(f"Filtering out class {original_class_id} (not in paper set: Car, Pedestrian, Cyclist)")
                            continue
                        class_id = remapped_class_id
                    else:
                        class_id = original_class_id

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
                            "width": values[6],
                        },
                        "dimensions": {
                            "height": values[7],
                            "width": values[8],
                            "length": values[9],
                        },
                        "alpha": values[10],
                        "vertices": {
                            "v1": [values[11], values[12]],
                            "v2": [values[13], values[14]],
                            "v3": [values[15], values[16]],
                            "v4": [values[17], values[18]],
                        },
                    }

                    # Store original class ID if filtering is enabled
                    if self.filter_classes:
                        label_dict["original_class_id"] = original_class_id

                    labels.append(label_dict)
                except (ValueError, IndexError) as e:
                    LOGGER.warning(f"Error parsing label line in {label_file}: {e}")
                    continue

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


if __name__ == "__main__":
    """Test the KITTI Stereo Dataset dataloader."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Test KITTI Stereo Dataset Dataloader")
    parser.add_argument("--root", type=str, default="kitti-stereo-debug", help="Root directory of the dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use ('train' or 'val')")


    # Test dataset initialization
    print("=" * 60)
    print("Testing KITTI Stereo Dataset")
    print("=" * 60)

    args = parser.parse_args()
    print(f"root = {args.root}")

    try:
        # Initialize dataset
        dataset = KITTIStereoDataset(root=args.root, split=args.split)
        print(f"✓ Dataset initialized successfully")
        print(f"  - Dataset length: {len(dataset)}")
        print(f"  - Split: {dataset.split}")
        print(f"  - Root: {dataset.root}")

        # Test loading a sample
        if len(dataset) > 0:
            print("\n" + "-" * 60)
            print("Testing sample loading...")
            print("-" * 60)

            sample = dataset[0]
            print(f"✓ Sample loaded successfully")
            print(f"  - Image ID: {sample['image_id']}")
            print(f"  - Calibration file: {sample['calib_file']}")

            # Check images
            print(f"\n  Images:")
            print(f"    - Left image shape: {sample['left_img'].shape}")
            print(f"    - Right image shape: {sample['right_img'].shape}")
            print(f"    - Left image dtype: {sample['left_img'].dtype}")
            print(f"    - Right image dtype: {sample['right_img'].dtype}")

            # Check calibration
            print(f"\n  Calibration:")
            calib_keys = list(sample["calib"].keys())
            print(f"    - Calibration keys: {calib_keys}")
            if "fx" in sample["calib"]:
                print(f"    - fx: {sample['calib']['fx']:.2f}")
                print(f"    - fy: {sample['calib']['fy']:.2f}")
                print(f"    - cx: {sample['calib']['cx']:.2f}")
                print(f"    - cy: {sample['calib']['cy']:.2f}")
                print(f"    - baseline: {sample['calib']['baseline']:.4f} m")

            # Check labels
            print(f"\n  Labels:")
            print(f"    - Number of labels: {len(sample['labels'])}")
            if sample["labels"]:
                first_label = sample["labels"][0]
                print(f"    - First label keys: {list(first_label.keys())}")
                print(f"    - First label class_id: {first_label['class_id']}")
                print(f"    - First label left_box: {first_label['left_box']}")
                print(f"    - First label dimensions: {first_label['dimensions']}")
                print(f"    - First label alpha: {first_label['alpha']:.4f}")

            # Test multiple samples
            print("\n" + "-" * 60)
            print("Testing multiple samples...")
            print("-" * 60)

            num_test_samples = min(5, len(dataset))
            print(f"Testing {num_test_samples} samples...")
            for i in range(num_test_samples):
                sample = dataset[i]
                assert "left_img" in sample, f"Sample {i}: missing 'left_img'"
                assert "right_img" in sample, f"Sample {i}: missing 'right_img'"
                assert "labels" in sample, f"Sample {i}: missing 'labels'"
                assert "calib" in sample, f"Sample {i}: missing 'calib'"
                assert "calib_file" in sample, f"Sample {i}: missing 'calib_file'"
                assert "image_id" in sample, f"Sample {i}: missing 'image_id'"
                assert sample["left_img"].shape == sample["right_img"].shape, (
                    f"Sample {i}: left and right images have different shapes"
                )
            print(f"✓ All {num_test_samples} samples loaded successfully")

            # Test val split if it exists
            print("\n" + "-" * 60)
            print("Testing validation split...")
            print("-" * 60)

            try:
                val_dataset = KITTIStereoDataset(root=args.root, split="val")
                print(f"✓ Validation dataset initialized successfully")
                print(f"  - Validation dataset length: {len(val_dataset)}")
                if len(val_dataset) > 0:
                    val_sample = val_dataset[0]
                    print(f"  - First validation sample ID: {val_sample['image_id']}")
            except FileNotFoundError as e:
                print(f"⚠ Validation split not found: {e}")

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Make sure the dataset directory 'kitti-stereo-debug' exists")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

