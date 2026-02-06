# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""3D bounding box data structure for stereo vision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Box3D:
    """3D bounding box representation.

    Attributes:
        center_3d (tuple[float, float, float]): 3D center position (x, y, z) in meters.
            Coordinate system: Camera coordinate system (x: right, y: down, z: forward).
        dimensions (tuple[float, float, float]): Object dimensions (length, width, height) in meters.
            length: forward, width: right, height: up.
        orientation (float): Rotation angle around vertical (Y) axis in radians, range [-π, π].
        class_label (str): Object class name ("Car", "Pedestrian", "Cyclist").
        class_id (int): Numeric class identifier (0=Car, 1=Pedestrian, 2=Cyclist).
        confidence (float): Detection confidence score [0.0, 1.0].
        truncated (float | None): Truncation level [0.0, 1.0] (0=fully visible, 1=fully truncated).
        occluded (int | None): Occlusion level (0=fully visible, 1=partially occluded, 2=largely occluded, 3=unknown).
    """

    center_3d: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    orientation: float
    class_label: str
    class_id: int
    confidence: float
    truncated: float | None = None
    occluded: int | None = None

    def __post_init__(self):
        """Validate 3D bounding box parameters."""
        length, width, height = self.dimensions
        if length <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Dimensions must be positive, got {self.dimensions}")

        x, y, z = self.center_3d
        if z <= 0:
            raise ValueError(f"Depth (z) must be positive, got z={z}")

        # Normalize orientation to [-π, π] to be robust to float32 π (≈ 3.141592741...)
        # torch.atan2(sin, cos) can yield a float32 representation of π that is slightly > np.pi (float64),
        # so we normalize and use a small tolerance to avoid false positives.
        self.orientation = float(np.arctan2(np.sin(self.orientation), np.cos(self.orientation)))
        eps = 1e-6
        if not (-np.pi - eps <= self.orientation <= np.pi + eps):
            raise ValueError(f"Orientation must be in [-π, π], got {self.orientation}")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Convert 3D box to dictionary.

        Returns:
            dict: Dictionary containing all box attributes.
        """
        return {
            "center_3d": self.center_3d,
            "dimensions": self.dimensions,
            "orientation": self.orientation,
            "class_label": self.class_label,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "truncated": self.truncated,
            "occluded": self.occluded,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Box3D:
        """Create Box3D from dictionary.

        Args:
            data: Dictionary containing box attributes.

        Returns:
            Box3D: Instance with attributes from dictionary.
        """
        return cls(
            center_3d=tuple(data["center_3d"]),
            dimensions=tuple(data["dimensions"]),
            orientation=data["orientation"],
            class_label=data["class_label"],
            class_id=data["class_id"],
            confidence=data["confidence"],
            truncated=data.get("truncated"),
            occluded=data.get("occluded"),
        )

    @classmethod
    def from_label(
        cls,
        label: dict[str, Any],
        calib: dict[str, float],
        class_names: Any = None,
        image_hw: tuple[int, int] | None = None,
    ) -> Box3D | None:
        """Create Box3D from a single dataset label dict.

        Handles KITTI bottom-center to geometric-center Y conversion and
        disparity-based 3D fallback when location_3d is missing.

        Args:
            label: Label dict with keys: class_id, left_box, dimensions,
                rotation_y, and optionally location_3d, right_box, truncated, occluded.
            calib: Calibration dict with fx, fy, cx, cy, baseline.
            class_names: Class name mapping (dict {id: name}, list, or None).
            image_hw: (H, W) used for disparity-based 3D fallback when location_3d is missing.

        Returns:
            Box3D instance, or None if insufficient data for 3D reconstruction.
        """
        class_id = int(label["class_id"])
        dims = label.get("dimensions", {})
        length = float(dims.get("length", 1.0))
        width = float(dims.get("width", 1.0))
        height = float(dims.get("height", 1.0))
        rot_y = float(label.get("rotation_y", 0.0))

        fx = float(calib.get("fx", 0.0))
        fy = float(calib.get("fy", 0.0))
        cx_cal = float(calib.get("cx", 0.0))
        cy_cal = float(calib.get("cy", 0.0))
        baseline = float(calib.get("baseline", 0.0))
        if fx <= 0 or fy <= 0:
            return None

        # Prefer GT 3D location (KITTI bottom-center -> geometric center)
        loc = label.get("location_3d")
        if isinstance(loc, dict) and all(k in loc for k in ("x", "y", "z")):
            x_3d = float(loc["x"])
            y_3d = float(loc["y"]) - height / 2.0  # bottom-center -> geometric center
            z_3d = float(loc["z"])
        else:
            # Fallback: reconstruct from stereo disparity
            lb = label["left_box"]
            rb = label.get("right_box")
            if rb is None or baseline <= 0:
                return None
            if image_hw is not None:
                H, W = int(image_hw[0]), int(image_hw[1])
            else:
                W = int(calib.get("image_width", 1242))
                H = int(calib.get("image_height", 375))
            left_u = float(lb["center_x"]) * W
            right_u = float(rb["center_x"]) * W
            disparity = left_u - right_u
            eps = 1e-6
            if not np.isfinite(disparity) or disparity <= eps:
                return None
            z_3d = (fx * baseline) / max(disparity, eps)
            x_3d = (left_u - cx_cal) * z_3d / fx
            y_3d = (float(lb.get("center_y", 0.5)) * H - cy_cal) * z_3d / fy

        # Resolve class name
        if class_names is None:
            class_label = str(class_id)
        elif isinstance(class_names, Mapping):
            class_label = str(class_names.get(class_id, class_id))
        elif isinstance(class_names, Sequence) and not isinstance(class_names, (str, bytes)):
            class_label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        else:
            class_label = str(class_id)

        try:
            return cls(
                center_3d=(float(x_3d), float(y_3d), float(z_3d)),
                dimensions=(float(length), float(width), float(height)),
                orientation=float(rot_y),
                class_label=class_label,
                class_id=class_id,
                confidence=1.0,
                truncated=float(label["truncated"]) if label.get("truncated") is not None else None,
                occluded=int(label["occluded"]) if label.get("occluded") is not None else None,
            )
        except (ValueError, TypeError):
            return None

    def project_to_2d(
        self,
        calibration: dict[str, float],
        image_size: tuple[int, int] | None = None,
        camera: str = "left",
    ) -> np.ndarray:
        """Project this 3D box to 2D image coordinates.

        Converts geometric center back to KITTI bottom-center for projection.

        Args:
            calibration: Dict with keys: fx, fy, cx, cy, baseline.
            image_size: (width, height) in pixels for clipping. None = no clipping.
            camera: "left" or "right" camera view.

        Returns:
            (4,) array [x1, y1, x2, y2] in pixels, or zeros if projection fails.
        """
        from ultralytics.models.yolo.stereo3ddet.augment import project_3d_box_to_2d

        length, width, height = self.dimensions
        x, y_center, z = self.center_3d
        # Geometric center -> bottom-center (KITTI convention): y_bottom = y_center + height/2
        y_bottom = y_center + height / 2.0

        return project_3d_box_to_2d(
            location_3d=np.array([x, y_bottom, z]),
            dimensions=np.array([length, width, height]),
            rotation_y=self.orientation,
            calibration=calibration,
            camera=camera,
            clip=image_size is not None,
            image_size=image_size,
        )