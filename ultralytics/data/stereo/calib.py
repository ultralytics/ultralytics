# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Calibration parameter handling for stereo vision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ultralytics.utils import LOGGER


@dataclass
class CalibrationParameters:
    """Camera calibration parameters for stereo vision.

    Attributes:
        fx (float): Focal length in x direction (pixels).
        fy (float): Focal length in y direction (pixels).
        cx (float): Principal point x coordinate (pixels).
        cy (float): Principal point y coordinate (pixels).
        baseline (float): Distance between left and right camera centers (meters).
        image_width (int): Original image width in pixels.
        image_height (int): Original image height in pixels.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float
    image_width: int
    image_height: int

    def __post_init__(self):
        """Validate calibration parameters."""
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"Focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        if self.baseline <= 0:
            raise ValueError(f"Baseline must be positive, got {self.baseline}")

    def to_dict(self) -> dict[str, Any]:
        """Convert calibration parameters to dictionary.

        Returns:
            dict: Dictionary containing all calibration parameters.
        """
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "baseline": self.baseline,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationParameters:
        """Create CalibrationParameters from dictionary.

        Args:
            data: Dictionary containing calibration parameters.

        Returns:
            CalibrationParameters: Instance with parameters from dictionary.
        """
        return cls(
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            baseline=data["baseline"],
            image_width=data["image_width"],
            image_height=data["image_height"],
        )


def load_kitti_calibration(calib_file: str | Path) -> CalibrationParameters:
    """Load KITTI calibration parameters from file.

    Extracts fx, fy, cx, cy, baseline from P2 (left camera projection matrix) and
    Tr (transformation matrix from velodyne to left camera).

    Args:
        calib_file: Path to KITTI calibration file.

    Returns:
        CalibrationParameters: Parsed calibration parameters.

    Raises:
        FileNotFoundError: If calibration file does not exist.
        ValueError: If calibration file format is invalid.
    """
    calib_file = Path(calib_file)
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    # Parse calibration file
    calib_data = {}
    with open(calib_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(":", 1)
            if len(parts) != 2:
                continue

            key = parts[0].strip()
            values_str = parts[1].strip()
            values = [float(x) for x in values_str.split()]

            if key.startswith("P"):
                # Projection matrix: P0, P1, P2, P3
                # P2 is left camera: [fx, 0, cx, tx, 0, fy, cy, ty, 0, 0, 1, tz]
                matrix = np.array(values).reshape(3, 4)
                calib_data[key] = matrix
            elif key.startswith("Tr"):
                # Transformation matrix: 4x4
                matrix = np.array(values).reshape(3, 4)
                calib_data[key] = matrix

    # Extract parameters from P2 (left camera projection matrix)
    if "P2" not in calib_data:
        raise ValueError(f"P2 (left camera projection) not found in {calib_file}")

    P2 = calib_data["P2"]
    fx = P2[0, 0]
    fy = P2[1, 1]
    cx = P2[0, 2]
    cy = P2[1, 2]

    # Extract baseline from P2[0, 3] = -fx * baseline
    # baseline = -P2[0, 3] / fx
    baseline_pixel = -P2[0, 3] / fx if fx > 0 else 0.0

    # If Tr matrix exists, use it to get baseline in meters
    # Otherwise, estimate from pixel baseline (approximate)
    if "Tr" in calib_data:
        # Tr[0, 3] gives translation in x direction (baseline in meters)
        baseline = abs(calib_data["Tr"][0, 3])
    else:
        # Approximate: assume pixel baseline corresponds to ~0.54m (KITTI standard)
        # This is a fallback - should use Tr matrix when available
        baseline = baseline_pixel * 0.54 / 721.5377  # KITTI approximate conversion
        LOGGER.warning(f"Tr matrix not found, using approximate baseline: {baseline:.3f}m")

    # Default image dimensions (KITTI standard)
    image_width = 1242
    image_height = 375

    return CalibrationParameters(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        baseline=baseline,
        image_width=image_width,
        image_height=image_height,
    )

