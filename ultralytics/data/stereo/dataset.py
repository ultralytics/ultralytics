# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""KITTI stereo dataset loader for 3D object detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.kitti_stereo import KITTIStereoDataset
from ultralytics.data.stereo.calib import CalibrationParameters, load_kitti_calibration
from ultralytics.utils import LOGGER


class KITTIStereo3DDataset(Dataset):
    """KITTI stereo dataset for 3D object detection training.

    Extends KITTIStereoDataset to provide format compatible with Stereo CenterNet training.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        imgsz: int = 1280,
        augment: bool = True,
    ):
        """Initialize KITTI Stereo 3D Dataset.

        Args:
            root: Root directory of KITTI dataset.
            split: Dataset split ('train' or 'val').
            imgsz: Input image size (width).
            augment: Whether to apply data augmentation.
        """
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz
        self.augment = augment

        # Use existing KITTIStereoDataset for data loading
        self.base_dataset = KITTIStereoDataset(root=root, split=split)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
                - img: Stereo image pair [6, H, W] (concatenated left+right, normalized [0, 1])
                - labels: Ground truth labels for target generation
                - calib: Calibration parameters
                - image_id: Image identifier
        """
        # Get sample from base dataset
        sample = self.base_dataset[idx]

        # Extract components
        left_img = sample["left_img"]  # [H, W, 3] BGR
        right_img = sample["right_img"]  # [H, W, 3] BGR
        labels = sample["labels"]
        calib_dict = sample["calib"]
        image_id = sample["image_id"]

        # Convert BGR to RGB
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        # Resize to target size (maintain aspect ratio, pad if needed)
        left_img, right_img = self._resize_stereo_pair(left_img, right_img, self.imgsz)

        # Convert to CHW format and normalize
        left_img = left_img.transpose(2, 0, 1).astype(np.float32) / 255.0  # [3, H, W]
        right_img = right_img.transpose(2, 0, 1).astype(np.float32) / 255.0  # [3, H, W]

        # Concatenate to stereo pair
        stereo_img = np.concatenate([left_img, right_img], axis=0)  # [6, H, W]

        # Convert to tensor
        stereo_img = torch.from_numpy(stereo_img).float()

        # Parse calibration
        calib = self._parse_calibration(calib_dict)

        return {
            "img": stereo_img,  # [6, H, W]
            "labels": labels,
            "calib": calib,
            "image_id": image_id,
        }

    def _resize_stereo_pair(
        self, left_img: np.ndarray, right_img: np.ndarray, target_width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize stereo pair maintaining aspect ratio.

        Args:
            left_img: Left image [H, W, 3].
            right_img: Right image [H, W, 3].
            target_width: Target width in pixels.

        Returns:
            Resized left and right images.
        """
        h, w = left_img.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        new_w = target_width

        # Resize
        left_img = cv2.resize(left_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        right_img = cv2.resize(right_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target height if needed (target height = 384 for 1280×384)
        target_height = 384
        if new_h < target_height:
            pad_h = target_height - new_h
            left_img = np.pad(left_img, ((0, pad_h), (0, 0), (0, 0)), mode="constant", constant_values=0)
            right_img = np.pad(right_img, ((0, pad_h), (0, 0), (0, 0)), mode="constant", constant_values=0)

        return left_img, right_img

    def _parse_calibration(self, calib_dict: dict[str, Any]) -> CalibrationParameters:
        """Parse calibration dictionary to CalibrationParameters.

        Args:
            calib_dict: Calibration dictionary from base dataset.

        Returns:
            CalibrationParameters object.
        """
        # Extract parameters
        fx = calib_dict.get("fx", 721.5377)  # Default KITTI value
        fy = calib_dict.get("fy", 721.5377)
        cx = calib_dict.get("cx", 609.5593)
        cy = calib_dict.get("cy", 172.8540)
        baseline = calib_dict.get("baseline", 0.54)  # Default KITTI baseline
        image_width = calib_dict.get("image_width", 1242)
        image_height = calib_dict.get("image_height", 375)

        return CalibrationParameters(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            baseline=baseline,
            image_width=image_width,
            image_height=image_height,
        )

