# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stereo 3D visualization: project 3D boxes to the image plane and draw them as wireframes.

Lives under models/yolo/s3d/ rather than utils/plotting.py because every symbol here is
stereo-3D-only — nothing outside this task imports them. Keeping them in the shared plotting module
also forced a runtime import of ultralytics.data.stereo.calib at utils scope, which is circular: it
failed on every `import ultralytics` and bound CalibrationParameters to None.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors

if TYPE_CHECKING:
    from ultralytics.data.stereo.box3d import Box3D
    from ultralytics.data.stereo.calib import CalibrationParameters


@dataclass
class VisualizationConfig:
    """Configuration container for stereo 3D visualization helpers."""

    line_width: int = 2
    font_size: float = 0.5
    show_labels: bool = True
    show_conf: bool = True
    camera_view: str = "both"
    pred_color_scheme: dict[int, tuple[int, int, int]] = field(
        default_factory=lambda: {
            0: (0, 128, 255),  # Car - orange tone in BGR
            1: (64, 64, 255),  # Pedestrian - reddish in BGR
            2: (221, 111, 255),  # Cyclist - magenta
        }
    )
    gt_color_scheme: dict[int, tuple[int, int, int]] = field(
        default_factory=lambda: {
            0: (0, 255, 0),  # Car - green
            1: (255, 180, 0),  # Pedestrian - cyan/blue tone
            2: (0, 223, 183),  # Cyclist - teal
        }
    )

    def __post_init__(self) -> None:
        """Reject invalid camera views and non-positive line widths at construction time."""
        if self.camera_view not in {"left", "right", "both"}:
            raise ValueError(f"camera_view must be 'left', 'right', or 'both', got '{self.camera_view}'")
        if self.line_width <= 0:
            raise ValueError("line_width must be positive")
        if self.font_size <= 0:
            raise ValueError("font_size must be positive")


EDGE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

# Smallest depth (metres) a 3D box centre may have and still be projected. Guards the 1/z in the
# pinhole projection only — it is deliberately not a plausibility floor, so close-range stereo rigs
# (objects at tens of centimetres) render as well as automotive ones.
EPS_PROJECT_Z = 1e-3


def project_box3d_corners(
    box3d: Box3D,
    calib: CalibrationParameters | dict[str, float],
    letterbox_scale: float | None = None,
    letterbox_pad_left: float | None = None,
    letterbox_pad_top: float | None = None,
    camera: str = "left",
) -> np.ndarray:
    """Project the eight corners of a 3D bounding box to 2D pixel coordinates.

    Args:
        box3d: 3D bounding box to project
        calib: Camera calibration parameters (for original image size)
        letterbox_scale: Scale factor from letterboxing (if images were letterboxed)
        letterbox_pad_left: Left padding from letterboxing (if images were letterboxed)
        letterbox_pad_top: Top padding from letterboxing (if images were letterboxed)
        camera: Camera view to project to ("left" or "right")

    Returns:
        Array of 2D pixel coordinates [8, 2] with shape (u, v) for each corner
    """

    def _get_calib_params(cal: CalibrationParameters | dict[str, float]) -> tuple[float, float, float, float, float]:
        if isinstance(cal, dict):
            return cal["fx"], cal["fy"], cal["cx"], cal["cy"], cal.get("baseline", 0.0)
        # Assume it's a CalibrationParameters object
        return cal.fx, cal.fy, cal.cx, cal.cy, cal.baseline

    fx, fy, cx, cy, baseline = _get_calib_params(calib)

    x, y, z = box3d.center_3d
    length, width, height = box3d.dimensions
    orientation = box3d.orientation

    # Skip only boxes that are not projectable at all: behind the camera, on it, or non-finite.
    # A metric depth floor must NOT live here — it is dataset-specific (a close-range rig images
    # objects at tens of centimetres) and callers own it via the head's configured depth range.
    # Extreme pixel coordinates from a small z are already bounded by the caller, which clips
    # every corner to +/-1e6 and then to the image rect via cv2.clipLine.
    if not np.isfinite(z) or z <= EPS_PROJECT_Z:
        return np.zeros((8, 2), dtype=np.float32)  # Return dummy corners (will be clipped out)

    # KITTI convention: rotation_y=0 means object faces camera X direction
    # So object's length (forward direction) should be along X axis
    # EDGE_CONNECTIONS expects: bottom face (0,1,2,3), top face (4,5,6,7)
    # In KITTI camera coords: Y points down, so bottom has y=+height/2, top has y=-height/2
    corners_obj = np.array(
        [
            # Bottom face corners (y = +height/2): 0, 1, 2, 3
            # Top face corners (y = -height/2): 4, 5, 6, 7
            [
                -length / 2,
                length / 2,
                length / 2,
                -length / 2,
                -length / 2,
                length / 2,
                length / 2,
                -length / 2,
            ],  # x (length)
            [
                height / 2,
                height / 2,
                height / 2,
                height / 2,
                -height / 2,
                -height / 2,
                -height / 2,
                -height / 2,
            ],  # y (height) - bottom first, then top
            [width / 2, width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2],  # z (width)
        ]
    )

    cos_rot = np.cos(orientation)
    sin_rot = np.sin(orientation)
    rotation = np.array([[cos_rot, 0, sin_rot], [0, 1, 0], [-sin_rot, 0, cos_rot]])

    corners_world = rotation @ corners_obj
    corners_world[0, :] += x
    corners_world[1, :] += y
    corners_world[2, :] += z

    # For right camera, shift X by baseline (right camera is offset to the right,
    # so objects appear shifted left in the image)
    if camera == "right":
        corners_world[0, :] -= baseline

    X, Y, Z = corners_world
    Z = np.maximum(Z, 1e-6)

    # Project to original image coordinates
    u_orig = fx * X / Z + cx
    v_orig = fy * Y / Z + cy

    # Adjust for letterboxing if provided
    if letterbox_scale is not None and letterbox_pad_left is not None and letterbox_pad_top is not None:
        u = u_orig * letterbox_scale + letterbox_pad_left
        v = v_orig * letterbox_scale + letterbox_pad_top
    else:
        u = u_orig
        v = v_orig

    corners = np.stack((u, v), axis=-1).astype(np.float32)

    return corners


def _select_color(
    class_id: int,
    scheme: dict[int, tuple[int, int, int]],
) -> tuple[int, int, int]:
    color = scheme.get(class_id)
    if color is None:
        color = colors(class_id, bgr=True)
    return tuple(int(c) for c in color)


def plot_boxes3d(
    img: np.ndarray,
    boxes3d: list[Box3D] | None,
    calib: CalibrationParameters | dict[str, float],
    config: VisualizationConfig | None = None,
    is_ground_truth: bool = False,
    letterbox_scale: float | None = None,
    letterbox_pad_left: float | None = None,
    letterbox_pad_top: float | None = None,
    camera: str = "left",
) -> np.ndarray:
    """Draw wireframe representations of Box3D objects onto an image.

    Args:
        img: Image to draw on (may be letterboxed)
        boxes3d: List of 3D bounding boxes to draw
        calib: Camera calibration parameters (for original image size)
        config: Visualization configuration
        is_ground_truth: Whether boxes are ground truth (affects color scheme)
        letterbox_scale: Scale factor from letterboxing (if images were letterboxed)
        letterbox_pad_left: Left padding from letterboxing (if images were letterboxed)
        letterbox_pad_top: Top padding from letterboxing (if images were letterboxed)
        camera: Camera view to project to ("left" or "right")
    """
    config = config or VisualizationConfig()

    # Ensure input image is uint8 and properly initialized
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Create a properly initialized copy of the image
    canvas = img.copy().astype(np.uint8)

    if not boxes3d:
        return canvas

    height, width = canvas.shape[:2]
    rect = (0, 0, width - 1, height - 1)
    line_width = max(1, config.line_width)
    scheme = config.gt_color_scheme if is_ground_truth else config.pred_color_scheme

    for box in boxes3d:
        try:
            corners = project_box3d_corners(
                box,
                calib,
                letterbox_scale=letterbox_scale,
                letterbox_pad_left=letterbox_pad_left,
                letterbox_pad_top=letterbox_pad_top,
                camera=camera,
            )
        except Exception as exc:
            LOGGER.warning("Skipping invalid Box3D during visualization: %s", exc)
            continue

        # Skip if corners are invalid (all zeros from Z < MIN_VALID_Z)
        if np.allclose(corners, 0.0, atol=1e-6):
            LOGGER.debug("Skipping Box3D with invalid Z depth (corners all zero)")
            continue

        color = _select_color(getattr(box, "class_id", 0), scheme)

        for start, end in EDGE_CONNECTIONS:
            pt1 = (int(np.clip(round(corners[start][0]), -1e6, 1e6)), int(np.clip(round(corners[start][1]), -1e6, 1e6)))
            pt2 = (int(np.clip(round(corners[end][0]), -1e6, 1e6)), int(np.clip(round(corners[end][1]), -1e6, 1e6)))
            clipped, clip_pt1, clip_pt2 = cv2.clipLine(rect, pt1, pt2)
            if clipped:
                cv2.line(canvas, clip_pt1, clip_pt2, color, line_width, lineType=cv2.LINE_AA)

        if config.show_labels:
            label = getattr(box, "class_label", "object")
            if config.show_conf and hasattr(box, "confidence"):
                label = f"{label} {box.confidence:.2f}"
            anchor = (
                int(np.clip(corners[0][0], 0, width - 1)),
                int(np.clip(corners[0][1], 0, height - 1)),
            )
            cv2.putText(
                canvas,
                label,
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.font_size,
                color,
                max(1, line_width // 2),
                cv2.LINE_AA,
            )

    return canvas


def plot_stereo3d_boxes(
    left_img: np.ndarray,
    right_img: np.ndarray,
    pred_boxes3d: list[Box3D] | None = None,
    gt_boxes3d: list[Box3D] | None = None,
    left_calib: CalibrationParameters | dict[str, float] | None = None,
    right_calib: CalibrationParameters | dict[str, float] | None = None,
    config: VisualizationConfig | None = None,
    letterbox_scale: float | None = None,
    letterbox_pad_left: float | None = None,
    letterbox_pad_top: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw predictions and ground truth on stereo image pairs.

    Args:
        left_img: Left camera image (may be letterboxed)
        right_img: Right camera image (may be letterboxed)
        pred_boxes3d: List of predicted 3D bounding boxes
        gt_boxes3d: List of ground truth 3D bounding boxes
        left_calib: Left camera calibration parameters (for original image size)
        right_calib: Right camera calibration parameters (defaults to left_calib)
        config: Visualization configuration
        letterbox_scale: Scale factor from letterboxing (if images were letterboxed)
        letterbox_pad_left: Left padding from letterboxing (if images were letterboxed)
        letterbox_pad_top: Top padding from letterboxing (if images were letterboxed)
    """
    if left_calib is None:
        raise ValueError("left_calib is required for stereo visualization")
    config = config or VisualizationConfig()
    right_calib = right_calib or left_calib

    left_canvas = plot_boxes3d(
        left_img,
        pred_boxes3d,
        left_calib,
        config,
        is_ground_truth=False,
        letterbox_scale=letterbox_scale,
        letterbox_pad_left=letterbox_pad_left,
        letterbox_pad_top=letterbox_pad_top,
    )
    left_canvas = plot_boxes3d(
        left_canvas,
        gt_boxes3d,
        left_calib,
        config,
        is_ground_truth=True,
        letterbox_scale=letterbox_scale,
        letterbox_pad_left=letterbox_pad_left,
        letterbox_pad_top=letterbox_pad_top,
    )

    right_canvas = plot_boxes3d(
        right_img,
        pred_boxes3d,
        right_calib,
        config,
        is_ground_truth=False,
        letterbox_scale=letterbox_scale,
        letterbox_pad_left=letterbox_pad_left,
        letterbox_pad_top=letterbox_pad_top,
        camera="right",
    )
    right_canvas = plot_boxes3d(
        right_canvas,
        gt_boxes3d,
        right_calib,
        config,
        is_ground_truth=True,
        letterbox_scale=letterbox_scale,
        letterbox_pad_left=letterbox_pad_left,
        letterbox_pad_top=letterbox_pad_top,
        camera="right",
    )

    combined = combine_stereo_views(left_canvas, right_canvas)
    return left_canvas, right_canvas, combined


def combine_stereo_views(
    left_img: np.ndarray,
    right_img: np.ndarray,
    pad_value: int = 0,
) -> np.ndarray:
    """Horizontally stack stereo images, padding the shorter view if necessary."""
    if left_img.ndim != 3 or right_img.ndim != 3:
        raise ValueError("Stereo images must be rank-3 tensors shaped [H, W, C].")

    # Ensure images are uint8 and properly initialized
    if left_img.dtype != np.uint8:
        left_img = np.clip(left_img, 0, 255).astype(np.uint8)
    if right_img.dtype != np.uint8:
        right_img = np.clip(right_img, 0, 255).astype(np.uint8)

    max_height = max(left_img.shape[0], right_img.shape[0])

    def _pad_to_height(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == max_height:
            return img.copy()  # Make a copy to avoid modifying original
        max_height - img.shape[0]
        # Create a new array with proper initialization
        padded = np.full((max_height, img.shape[1], img.shape[2]), pad_value, dtype=np.uint8)
        padded[: img.shape[0], :, :] = img
        return padded

    left_padded = _pad_to_height(left_img)
    right_padded = _pad_to_height(right_img)
    return np.hstack((left_padded, right_padded))
