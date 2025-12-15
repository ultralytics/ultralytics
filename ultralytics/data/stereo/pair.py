# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Stereo image pair data structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ultralytics.data.stereo.calib import CalibrationParameters


@dataclass
class StereoImagePair:
    """Synchronized pair of left and right camera images with metadata.

    Attributes:
        left_image (numpy.ndarray | torch.Tensor): Left camera RGB image.
            Shape: [H, W, 3] (RGB) or [3, H, W] (CHW format).
            Type: uint8 (0-255) or float32 (0.0-1.0 normalized).
        right_image (numpy.ndarray | torch.Tensor): Right camera RGB image.
            Shape: [H, W, 3] (RGB) or [3, H, W] (CHW format).
            Type: uint8 (0-255) or float32 (0.0-1.0 normalized).
        image_id (str): Unique identifier for the image pair (e.g., "000001").
        calibration (CalibrationParameters): Camera intrinsic and extrinsic parameters.
        timestamp (float | None): Timestamp for synchronization (if available).
    """

    left_image: np.ndarray | torch.Tensor
    right_image: np.ndarray | torch.Tensor
    image_id: str
    calibration: CalibrationParameters
    timestamp: float | None = None

    def __post_init__(self):
        """Validate stereo image pair."""
        left_shape = self.left_image.shape
        right_shape = self.right_image.shape

        # Check if shapes match (accounting for different formats)
        if len(left_shape) == 3 and len(right_shape) == 3:
            # HWC or CHW format
            if left_shape[:2] != right_shape[:2] and left_shape[1:] != right_shape[1:]:
                raise ValueError(
                    f"Left and right images must have same dimensions, "
                    f"got left={left_shape}, right={right_shape}"
                )
        else:
            raise ValueError(f"Images must be 3D arrays, got left={left_shape}, right={right_shape}")

        # Validate calibration
        if self.calibration.fx <= 0 or self.calibration.fy <= 0:
            raise ValueError("Calibration focal lengths must be positive")
        if self.calibration.baseline <= 0:
            raise ValueError("Calibration baseline must be positive")

    def to_tensor(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Convert stereo pair to concatenated tensor.

        Args:
            device: Target device for tensor.

        Returns:
            torch.Tensor: Concatenated stereo pair [6, H, W] in CHW format, float32, normalized [0, 1].
        """
        # Convert to torch if needed
        if isinstance(self.left_image, np.ndarray):
            left = torch.from_numpy(self.left_image).float()
        else:
            left = self.left_image.float()

        if isinstance(self.right_image, np.ndarray):
            right = torch.from_numpy(self.right_image).float()
        else:
            right = self.right_image.float()

        # Handle different formats
        if len(left.shape) == 3:
            if left.shape[0] == 3:  # CHW format
                pass  # Already in CHW
            elif left.shape[2] == 3:  # HWC format
                left = left.permute(2, 0, 1)  # HWC -> CHW
                right = right.permute(2, 0, 1)  # HWC -> CHW

        # Normalize if needed (assuming uint8 input)
        if left.max() > 1.0:
            left = left / 255.0
        if right.max() > 1.0:
            right = right / 255.0

        # Concatenate along channel dimension
        stereo = torch.cat([left, right], dim=0)  # [6, H, W]

        if device is not None:
            stereo = stereo.to(device)

        return stereo

    def to_dict(self) -> dict[str, Any]:
        """Convert stereo pair to dictionary.

        Returns:
            dict: Dictionary containing all pair attributes.
        """
        return {
            "left_image": self.left_image,
            "right_image": self.right_image,
            "image_id": self.image_id,
            "calibration": self.calibration.to_dict(),
            "timestamp": self.timestamp,
        }

