#!/usr/bin/env python3
"""
MDE Dataset Builder for YOLO11 Training

This script creates a proper dataset builder that extracts depth information
from label files and includes it in the training batches.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import torch
import yaml

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER


class MDEDataset:
    """
    MDE Dataset class that extends YOLO dataset to include depth information.

    This class handles loading images, labels, and depth information for
    Monocular Depth Estimation training.
    """

    def __init__(self, data_config: Dict[str, Any], mode: str = "train", batch: int = None):
        """
        Initialize MDE Dataset.

        Args:
            data_config: Dataset configuration dictionary
            mode: 'train' or 'val'
            batch: Batch size (for compatibility)
        """
        self.data_config = data_config
        self.mode = mode
        self.batch = batch

        # Extract paths
        self.path = Path(data_config["path"])
        self.train_path = self.path / data_config.get("train", "images")
        self.val_path = self.path / data_config.get("val", "images")

        # Use appropriate split path
        self.split_path = self.train_path if mode == "train" else self.val_path
        self.labels_path = self.path / "labels"

        # Class information
        self.nc = data_config["nc"]
        self.names = data_config["names"]

        # MDE specific settings
        self.depth_loss_weight = data_config.get("depth_loss_weight", 1.0)
        self.depth_loss_type = data_config.get("depth_loss_type", "l1")

        # Load dataset
        self.images = []
        self.labels = []
        self.depths = []

        self._load_dataset()

        LOGGER.info(f"MDE Dataset loaded: {len(self.images)} images, {len(self.labels)} labels")

    def _load_dataset(self):
        """Load images, labels, and depth information from the dataset."""

        # Get image files
        image_files = []
        for ext in IMG_FORMATS:
            image_files.extend(self.split_path.glob(f"*.{ext}"))

        self.images = sorted(image_files)

        # Load labels and extract depth information
        for img_path in self.images:
            label_path = self.labels_path / f"{img_path.stem}.txt"

            if label_path.exists():
                # Parse label file
                labels, depths = self._parse_label_file(label_path)
                self.labels.append(labels)
                self.depths.append(depths)
            else:
                # Empty labels for images without annotations
                self.labels.append([])
                self.depths.append([])

    def _parse_label_file(self, label_path: Path) -> Tuple[List, List]:
        """
        Parse label file and extract depth information.

        Expected format: class_id x_center y_center width height depth

        Args:
            label_path: Path to label file

        Returns:
            Tuple of (labels, depths)
        """
        labels = []
        depths = []

        try:
            with open(label_path, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 6:  # Has depth information
                            cls_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            depth = float(parts[5])

                            labels.append([cls_id, x_center, y_center, width, height])
                            depths.append(depth)
                        else:
                            # Fallback for labels without depth
                            cls_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            depth = 10.0  # Default depth value

                            labels.append([cls_id, x_center, y_center, width, height])
                            depths.append(depth)
        except Exception as e:
            LOGGER.warning(f"Error parsing label file {label_path}: {e}")
            return [], []

        return labels, depths

    def __len__(self):
        """Return dataset length."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item with depth information.

        Args:
            idx: Dataset index

        Returns:
            Dictionary containing image, labels, and depth information
        """
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image to standard size (640x640)
        target_size = 640
        img_resized = cv2.resize(img, (target_size, target_size))
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0

        # Get labels and depths
        labels = self.labels[idx]
        depths = self.depths[idx]

        # Create depth map (simplified approach)
        depth_map = self._create_depth_map(depths, target_size, target_size, labels)

        # Prepare batch item
        item = {
            "img": img_tensor,
            "img_path": str(img_path),
            "labels": labels,
            "depths": depths,
            "depth": depth_map,
            "nc": self.nc,
            "names": self.names,
        }

        return item

    def _create_depth_map(self, depths: List[float], height: int, width: int, labels: List) -> torch.Tensor:
        """
        Create a depth map from depth values and labels.

        Args:
            depths: List of depth values for each object
            height: Image height
            width: Image width
            labels: List of labels [cls_id, x_center, y_center, width, height]

        Returns:
            Depth map tensor
        """
        # Create empty depth map
        depth_map = torch.zeros(1, height, width)

        if not depths or not labels:
            # Return default depth map
            return torch.full((1, height, width), 10.0)

        # Fill depth map with object depths
        for depth, label in zip(depths, labels):
            cls_id, x_center, y_center, w, h = label

            # Convert normalized coordinates to pixel coordinates
            x_center_px = int(x_center * width)
            y_center_px = int(y_center * height)
            w_px = int(w * width)
            h_px = int(h * height)

            # Calculate bounding box coordinates
            x1 = max(0, x_center_px - w_px // 2)
            y1 = max(0, y_center_px - h_px // 2)
            x2 = min(width, x_center_px + w_px // 2)
            y2 = min(height, y_center_px + h_px // 2)

            # Fill depth map
            depth_map[0, y1:y2, x1:x2] = depth

        # Fill background with average depth or default
        if depth_map.sum() == 0:
            depth_map.fill_(10.0)  # Default depth
        else:
            # Fill background with average object depth
            avg_depth = depth_map[depth_map > 0].mean()
            depth_map[depth_map == 0] = avg_depth

        return depth_map


def build_mde_dataset(data_config: Dict[str, Any], mode: str = "train", batch: int = None) -> MDEDataset:
    """
    Build MDE dataset with depth information.

    Args:
        data_config: Dataset configuration dictionary
        mode: 'train' or 'val'
        batch: Batch size

    Returns:
        MDE Dataset instance
    """
    return MDEDataset(data_config, mode, batch)


class MDEDataLoader:
    """Wrapper class for MDE DataLoader to provide compatibility with YOLO trainer."""

    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader
        self._iterator = None

    def __iter__(self):
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self):
        return next(self._iterator)

    def __len__(self):
        return len(self.dataloader)

    def reset(self):
        """Reset the dataloader iterator (compatibility method)."""
        self._iterator = None

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying dataloader."""
        return getattr(self.dataloader, name)


def create_mde_dataloader(
    data_config: Dict[str, Any], batch_size: int = 16, mode: str = "train", workers: int = 8
) -> MDEDataLoader:
    """
    Create MDE dataloader with proper collate function.

    Args:
        data_config: Dataset configuration
        batch_size: Batch size
        mode: 'train' or 'val'
        workers: Number of workers

    Returns:
        MDEDataLoader with MDE batches
    """
    dataset = build_mde_dataset(data_config, mode)

    def collate_fn(batch):
        """Custom collate function for MDE batches."""
        # Separate components
        imgs = [item["img"] for item in batch]
        img_paths = [item["img_path"] for item in batch]
        all_labels = [item["labels"] for item in batch]
        all_depths = [item["depths"] for item in batch]
        depth_maps = [item["depth"] for item in batch]

        # Stack images
        imgs = torch.stack(imgs, 0)

        # Stack depth maps
        depth_maps = torch.stack(depth_maps, 0)

        # Create labels tensor
        labels = []
        batch_idx = 0
        for img_labels, img_depths in zip(all_labels, all_depths):
            for label, depth in zip(img_labels, img_depths):
                cls_id, x_center, y_center, width, height = label
                labels.append([batch_idx, cls_id, x_center, y_center, width, height])
            batch_idx += 1

        if labels:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 6), dtype=torch.float32)

        # Create batch
        batch_dict = {
            "img": imgs,
            "img_paths": img_paths,
            "labels": labels,
            "depth": depth_maps,
            "nc": dataset.nc,
            "names": dataset.names,
        }

        return batch_dict

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return MDEDataLoader(dataloader)


def test_mde_dataset(data_yaml_path: str = "ultralytics/cfg/datasets/kitti_mde_debug.yaml"):
    """Test the MDE dataset implementation."""

    print("🧪 Testing MDE Dataset Implementation")
    print("=" * 50)

    # Load dataset config
    try:
        with open(data_yaml_path, "r") as f:
            data_config = yaml.safe_load(f)
        print(f"✅ Loaded dataset config: {data_yaml_path}")
    except Exception as e:
        print(f"❌ Failed to load dataset config: {e}")
        return

    # Test dataset creation
    try:
        dataset = build_mde_dataset(data_config, mode="train")
        print(f"✅ Dataset created: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test dataloader creation
    try:
        dataloader = create_mde_dataloader(data_config, batch_size=2, mode="train")
        print(f"✅ DataLoader created: {len(dataloader)} batches")
    except Exception as e:
        print(f"❌ Failed to create dataloader: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test batch loading
    try:
        batch = next(iter(dataloader))
        print("✅ Batch loaded successfully!")
        print(f"   Images shape: {batch['img'].shape}")
        print(f"   Depth maps shape: {batch['depth'].shape}")
        print(f"   Labels shape: {batch['labels'].shape}")
        print(f"   Batch keys: {list(batch.keys())}")

        # Check if depth information is present
        if "depth" in batch:
            print("✅ Depth information found in batch!")
            print(f"   Depth range: [{batch['depth'].min():.3f}, {batch['depth'].max():.3f}]")
        else:
            print("❌ No depth information in batch!")

    except Exception as e:
        print(f"❌ Failed to load batch: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_mde_dataset()
