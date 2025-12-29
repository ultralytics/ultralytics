# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""3D bounding box data structure for stereo vision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ultralytics.utils import LOGGER


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
        bbox_2d (tuple[float, float, float, float] | None): 2D bounding box (x_min, y_min, x_max, y_max) in pixels.
        truncated (float | None): Truncation level [0.0, 1.0] (0=fully visible, 1=fully truncated).
        occluded (int | None): Occlusion level (0=fully visible, 1=partially occluded, 2=largely occluded, 3=unknown).
    """

    center_3d: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    orientation: float
    class_label: str
    class_id: int
    confidence: float
    bbox_2d: tuple[float, float, float, float] | None = None
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

        if not (-np.pi <= self.orientation <= np.pi):
            raise ValueError(f"Orientation must be in [-π, π], got {self.orientation}")

        if self.class_label not in {"Car", "Pedestrian", "Cyclist"}:
            raise ValueError(f"Invalid class label: {self.class_label}")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

        # Validate bbox_2d (xyxy format) if provided
        if self.bbox_2d is not None:
            if not isinstance(self.bbox_2d, (tuple, list)) or len(self.bbox_2d) != 4:
                raise ValueError(
                    f"bbox_2d must be a tuple/list of 4 numbers (xyxy format: x_min, y_min, x_max, y_max), "
                    f"got {self.bbox_2d}"
                )
            x_min, y_min, x_max, y_max = self.bbox_2d
            if not all(isinstance(v, (int, float)) for v in self.bbox_2d):
                raise ValueError(
                    f"bbox_2d must contain numeric values, got {self.bbox_2d}"
                )
            if x_min >= x_max:
                LOGGER.warning(
                    f"bbox_2d xyxy format: x_min ({x_min}) must be < x_max ({x_max})"
                )
            if y_min >= y_max:
                LOGGER.warning(
                    f"bbox_2d xyxy format: y_min ({y_min}) must be < y_max ({y_max})"
                )
            if x_min < 0 or y_min < 0:
                LOGGER.warning(
                    f"bbox_2d xyxy format: x_min and y_min must be non-negative, got ({x_min}, {y_min})"
                )

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
            "bbox_2d": self.bbox_2d,
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
            bbox_2d=tuple(data["bbox_2d"]) if data.get("bbox_2d") else None,
            truncated=data.get("truncated"),
            occluded=data.get("occluded"),
        )