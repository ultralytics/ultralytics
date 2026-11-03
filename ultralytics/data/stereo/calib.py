# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Calibration parameter handling for stereo vision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
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
        R_cam_tilt (np.ndarray | None): Optional (3, 3) rotation from the gravity-aligned (level) frame into the
            camera frame, for a camera mounted with pitch/roll. Labels then place a box as
            ``R_cam_tilt @ Ry(rotation_y) @ corners + location``. None means a level camera (KITTI).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float
    image_width: int
    image_height: int
    R_cam_tilt: np.ndarray | None = None

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
        d = {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "baseline": self.baseline,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }
        if self.R_cam_tilt is not None:  # only tilted cameras carry the key, so level-camera dicts are unchanged
            d["R_cam_tilt"] = self.R_cam_tilt
        return d

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
            R_cam_tilt=data.get("R_cam_tilt"),
        )


def cam_tilt(calib: dict[str, Any] | CalibrationParameters | None) -> np.ndarray | None:
    """Return the (3, 3) level->camera rotation of a calibration, or None for a level camera.

    Every tilt-aware code path branches on this returning None and then runs the original level-camera arithmetic
    untouched, so KITTI (and any calibration without ``R_cam_tilt``) is bit-identical to the pre-tilt code.
    """
    if calib is None:
        return None
    R = calib.get("R_cam_tilt") if isinstance(calib, dict) else getattr(calib, "R_cam_tilt", None)
    return None if R is None else np.asarray(R, dtype=np.float64).reshape(3, 3)


def tilt_center_offset(R: np.ndarray | None, height: float) -> np.ndarray:
    """Camera-frame vector from a box's bottom-center to its geometric center (``-h/2`` along the level up axis)."""
    if R is None:
        return np.array([0.0, -height / 2.0, 0.0])
    return R @ np.array([0.0, -height / 2.0, 0.0])


def level_ray_angle(R: np.ndarray | None, x: float, z: float, y: float = 0.0) -> float:
    """Horizontal viewing-ray angle of a camera-frame point, measured in the level frame (``atan2`` of ``R^T p``).

    With no tilt this is the KITTI ``atan2(x, z)``. ``rotation_y`` is a yaw about the level vertical, so alpha
    (= rotation_y - ray angle) must use the level-frame ray. If R embeds a yaw (e.g. a 180 deg flip) the ray angle
    shifts by that yaw and so does rotation_y; callers wrap alpha, so encode/decode stay consistent.
    """
    if R is None:
        return math.atan2(x, z)
    p = R.T @ np.array([x, y, z])
    return math.atan2(float(p[0]), float(p[2]))


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
    mats: dict[str, np.ndarray] = {}  # P0..P3, R_cam_tilt
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
            elif key == "R_cam_tilt":
                values = [float(x) for x in value_str.split()]
                if len(values) != 9:
                    raise ValueError(f"R_cam_tilt needs 9 values (row-major 3x3), got {len(values)} in {calib_file}")
                mats[key] = np.array(values).reshape(3, 3)

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
        R_cam_tilt=mats.get("R_cam_tilt"),
    )
