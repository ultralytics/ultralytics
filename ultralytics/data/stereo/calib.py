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
    """Load stereo calibration parameters from a KITTI-format file.

    Supports both the raw KITTI projection matrices (``P2``/``P3``) and the simplified ``fx/fy/cx/cy/baseline``
    key-value format produced by ``convert_kitti_3d.py`` and shipped with the kitti-stereo datasets. Intrinsics are
    taken from the explicit keys when present, otherwise derived from ``P2``; the stereo baseline is derived from the
    ``P2``/``P3`` horizontal offset when needed.

    Args:
        calib_file: Path to the calibration file.

    Returns:
        CalibrationParameters: Parsed calibration parameters.

    Raises:
        FileNotFoundError: If the calibration file does not exist.
        ValueError: If neither explicit intrinsics nor a ``P2`` matrix are present.
    """
    calib_file = Path(calib_file)
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    scalars: dict[str, float] = {}  # fx, fy, cx, cy, baseline, image_width, image_height
    mats: dict[str, np.ndarray] = {}  # P0..P3
    with open(calib_file) as f:
        for line in f:
            key, sep, value_str = line.strip().partition(":")
            key, value_str = key.strip(), value_str.strip()
            if not sep or not value_str:
                continue
            if key in {"fx", "fy", "cx", "cy", "baseline"}:
                scalars[key] = float(value_str)
            elif key in {"image_width", "image_height"}:
                scalars[key] = int(float(value_str))
            elif key in {"P0", "P1", "P2", "P3"}:
                values = [float(x) for x in value_str.split()]
                if len(values) == 12:
                    mats[key] = np.array(values).reshape(3, 4)

    # Derive intrinsics from P2 (left camera) when explicit keys are absent.
    if "P2" in mats:
        P2 = mats["P2"]
        scalars.setdefault("fx", float(P2[0, 0]))
        scalars.setdefault("fy", float(P2[1, 1]))
        scalars.setdefault("cx", float(P2[0, 2]))
        scalars.setdefault("cy", float(P2[1, 2]))
        # Baseline from left/right projection offset: b = (P2[0,3] - P3[0,3]) / fx (meters).
        if "baseline" not in scalars and "P3" in mats and scalars["fx"]:
            scalars["baseline"] = float(abs((P2[0, 3] - mats["P3"][0, 3]) / scalars["fx"]))

    if not all(k in scalars for k in ("fx", "fy", "cx", "cy")):
        raise ValueError(f"Calibration missing intrinsics (need fx/fy/cx/cy or a P2 matrix) in {calib_file}")
    if "baseline" not in scalars:
        scalars["baseline"] = 0.54  # KITTI-standard fallback
        LOGGER.warning(f"Baseline not found in {calib_file}, using KITTI default {scalars['baseline']}m")

    return CalibrationParameters(
        fx=scalars["fx"],
        fy=scalars["fy"],
        cx=scalars["cx"],
        cy=scalars["cy"],
        baseline=scalars["baseline"],
        image_width=int(scalars.get("image_width", 1242)),
        image_height=int(scalars.get("image_height", 375)),
    )
