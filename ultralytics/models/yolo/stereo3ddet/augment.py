from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional
import cv2
import numpy as np
from ultralytics.utils.instance import Bboxes, Instances


# ============================================================================
# 3D to 2D Projection Functions
# ============================================================================


def project_3d_box_to_2d(
    location_3d: np.ndarray,
    dimensions: np.ndarray,
    rotation_y: float,
    calibration: Dict[str, float],
    camera: str = "left",
    clip: bool = True,
    image_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Project a 3D bounding box to 2D image coordinates.

    Uses KITTI convention where location is at bottom-center of the object,
    Y points down, and dimensions are (length, width, height) where length
    is along Z (forward) and width is along X (right).

    Args:
        location_3d: (3,) array [X, Y, Z] in camera coordinates (meters).
        dimensions: (3,) array [length, width, height] in meters.
        rotation_y: Yaw angle in radians.
        calibration: Dict with keys: fx, fy, cx, cy, baseline.
        camera: "left" or "right" camera view.
        clip: Whether to clip the box to image boundaries.
        image_size: (width, height) in pixels for clipping. Required if clip=True.

    Returns:
        (4,) array [x1, y1, x2, y2] in pixels, or zeros if projection fails.
    """
    # Extract parameters
    X, Y, Z = location_3d
    length, width, height = dimensions
    fx = calibration.get("fx", 721.5377)
    fy = calibration.get("fy", 721.5377)
    cx = calibration.get("cx", 609.5593)
    cy = calibration.get("cy", 172.854)
    baseline = calibration.get("baseline", 0.54)

    # Build 8 corners in object frame (KITTI: bottom-center origin, Y down)
    # Bottom 4 corners (Y=0), then top 4 corners (Y=-height)
    corners_obj = np.array([
        [-length / 2, 0, -width / 2],   # rear-left bottom
        [length / 2, 0, -width / 2],    # front-left bottom
        [length / 2, 0, width / 2],     # front-right bottom
        [-length / 2, 0, width / 2],    # rear-right bottom
        [-length / 2, -height, -width / 2],  # rear-left top
        [length / 2, -height, -width / 2],   # front-left top
        [length / 2, -height, width / 2],    # front-right top
        [-length / 2, -height, width / 2],   # rear-right top
    ])

    # Rotation matrix around Y axis
    cos_ry = np.cos(rotation_y)
    sin_ry = np.sin(rotation_y)
    R = np.array([[cos_ry, 0, sin_ry], [0, 1, 0], [-sin_ry, 0, cos_ry]])

    # Rotate and translate to camera coordinates
    corners_cam = corners_obj @ R.T + np.array([X, Y, Z])

    # For right camera, shift X by baseline
    if camera == "right":
        corners_cam[:, 0] -= baseline

    # Project to 2D (only corners in front of camera)
    z_min = 0.1
    valid_mask = corners_cam[:, 2] > z_min
    if not valid_mask.any():
        return np.zeros(4, dtype=np.float32)

    corners_valid = corners_cam[valid_mask]
    u_coords = fx * (corners_valid[:, 0] / corners_valid[:, 2]) + cx
    v_coords = fy * (corners_valid[:, 1] / corners_valid[:, 2]) + cy

    # Get bounding rectangle
    x1, x2 = u_coords.min(), u_coords.max()
    y1, y2 = v_coords.min(), v_coords.max()

    # Clip to image boundaries if requested
    if clip and image_size is not None:
        img_w, img_h = image_size
        x1 = np.clip(x1, 0, img_w)
        x2 = np.clip(x2, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        y2 = np.clip(y2, 0, img_h)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def project_3d_boxes_to_2d(
    instances: Instances,
    calibration: Dict[str, float],
    camera: str = "left",
    clip: bool = True,
    image_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Project multiple 3D bboxes to 2D for a batch of instances.

    Args:
        instances: Instances object with location_3d, dimensions_3d, rotation_y.
        calibration: Dict with keys: fx, fy, cx, cy, baseline.
        camera: "left" or "right" camera view.
        clip: Whether to clip boxes to image boundaries.
        image_size: (width, height) in pixels for clipping.

    Returns:
        (N, 4) array of [x1, y1, x2, y2] in pixels.
    """
    n = len(instances)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Check for required 3D attributes
    if instances.location_3d is None or instances.dimensions_3d is None or instances.rotation_y is None:
        return np.zeros((n, 4), dtype=np.float32)

    boxes_2d = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        boxes_2d[i] = project_3d_box_to_2d(
            location_3d=instances.location_3d[i],
            dimensions=instances.dimensions_3d[i],
            rotation_y=instances.rotation_y[i],
            calibration=calibration,
            camera=camera,
            clip=clip,
            image_size=image_size,
        )
    return boxes_2d


@dataclass
class StereoCalibration:
    """Stereo camera calibration parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float
    height: int
    width: int

    def to_dict(self) -> Dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "baseline": self.baseline,
            "height": self.height,
            "width": self.width,
        }


class StereoLabels:
    """Container that synchronizes Instances and calibration during transforms.

    This class ensures that when geometric transformations (scale, flip, crop, pad)
    are applied to instances, the corresponding calibration parameters are updated
    automatically. This eliminates the fragile pattern of manually updating both
    in each transform.

    Attributes:
        instances (Instances): Object annotations (bboxes, 3D attributes, etc.).
        calibration (Dict[str, float]): Camera calibration parameters.

    Example:
        >>> labels = StereoLabels.from_labels(labels_dict)
        >>> labels.scale(0.5, 0.5)  # Scales both instances AND calibration
        >>> labels.fliplr(width)    # Flips both instances AND calibration
        >>> labels.to_labels(labels_dict)  # Write back to dict
    """

    def __init__(self, instances: Optional[Instances], calibration: Optional[Dict[str, Any]]):
        """Initialize StereoLabels.

        Args:
            instances: Instances object or None.
            calibration: Calibration dict with fx, fy, cx, cy, baseline, width, height.
        """
        self.instances = instances
        self.calibration = calibration or {}

    @classmethod
    def from_labels(cls, labels: Dict[str, Any]) -> "StereoLabels":
        """Create StereoLabels from a labels dict.

        Args:
            labels: Dict containing 'instances' and 'calibration' keys.

        Returns:
            StereoLabels instance.
        """
        return cls(
            instances=labels.get("instances"),
            calibration=labels.get("calibration"),
        )

    def to_labels(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """Write instances and calibration back to labels dict.

        Args:
            labels: Target labels dict to update.

        Returns:
            Updated labels dict.
        """
        labels["instances"] = self.instances
        labels["calibration"] = self.calibration
        return labels

    def has_instances(self) -> bool:
        """Check if there are valid instances."""
        return self.instances is not None and len(self.instances) > 0

    def has_calibration(self) -> bool:
        """Check if calibration is present."""
        return bool(self.calibration)

    # -------------------------------------------------------------------------
    # Synchronized transform methods
    # -------------------------------------------------------------------------

    def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False) -> "StereoLabels":
        """Scale coordinates and calibration by given factors.

        Args:
            scale_w: Scale factor for width.
            scale_h: Scale factor for height.
            bbox_only: If True, only scale bboxes (not segments/keypoints).

        Returns:
            Self for method chaining.
        """
        # Scale instances
        if self.has_instances():
            self.instances.scale(scale_w, scale_h, bbox_only=bbox_only)

        # Scale calibration
        if self.has_calibration():
            self.calibration["fx"] = self.calibration.get("fx", 1.0) * scale_w
            self.calibration["fy"] = self.calibration.get("fy", 1.0) * scale_h
            self.calibration["cx"] = self.calibration.get("cx", 0.0) * scale_w
            self.calibration["cy"] = self.calibration.get("cy", 0.0) * scale_h

        return self

    def fliplr(self, w: int) -> "StereoLabels":
        """Flip coordinates horizontally.

        Note: This flips coordinates but does NOT swap left/right bboxes or
        update location_3d - those are handled by the transform that also
        swaps the image channels.

        Args:
            w: Image width.

        Returns:
            Self for method chaining.
        """
        # Flip instances
        if self.has_instances():
            self.instances.fliplr(w)

        # Flip calibration principal point
        if self.has_calibration():
            self.calibration["cx"] = float((w - 1) - self.calibration.get("cx", w / 2))

        return self

    def flipud(self, h: int) -> "StereoLabels":
        """Flip coordinates vertically.

        Args:
            h: Image height.

        Returns:
            Self for method chaining.
        """
        # Flip instances
        if self.has_instances():
            self.instances.flipud(h)

        # Flip calibration principal point
        if self.has_calibration():
            self.calibration["cy"] = float((h - 1) - self.calibration.get("cy", h / 2))

        return self

    def add_padding(self, padw: int, padh: int) -> "StereoLabels":
        """Add padding offset to coordinates and calibration.

        Args:
            padw: Padding width (left offset).
            padh: Padding height (top offset).

        Returns:
            Self for method chaining.
        """
        # Add padding to instances
        if self.has_instances():
            self.instances.add_padding(padw, padh)

        # Shift calibration principal point
        if self.has_calibration():
            self.calibration["cx"] = self.calibration.get("cx", 0.0) + padw
            self.calibration["cy"] = self.calibration.get("cy", 0.0) + padh

        return self

    def clip(self, w: int, h: int) -> "StereoLabels":
        """Clip coordinates to image boundaries.

        Args:
            w: Image width.
            h: Image height.

        Returns:
            Self for method chaining.
        """
        # Clip instances
        if self.has_instances():
            self.instances.clip(w, h)

        # Clip calibration principal point (must stay within image)
        if self.has_calibration():
            self.calibration["cx"] = float(np.clip(self.calibration.get("cx", 0.0), 0.0, max(0, w - 1)))
            self.calibration["cy"] = float(np.clip(self.calibration.get("cy", 0.0), 0.0, max(0, h - 1)))

        return self

    def denormalize(self, w: int, h: int) -> "StereoLabels":
        """Convert normalized coordinates to absolute coordinates.

        Args:
            w: Image width.
            h: Image height.

        Returns:
            Self for method chaining.
        """
        if self.has_instances():
            self.instances.denormalize(w, h)
        # Calibration is always in absolute coordinates, no change needed
        return self

    def normalize(self, w: int, h: int) -> "StereoLabels":
        """Convert absolute coordinates to normalized coordinates.

        Args:
            w: Image width.
            h: Image height.

        Returns:
            Self for method chaining.
        """
        if self.has_instances():
            self.instances.normalize(w, h)
        # Calibration stays in absolute coordinates
        return self

    def update_size(self, new_w: int, new_h: int) -> "StereoLabels":
        """Update calibration image size.

        Args:
            new_w: New image width.
            new_h: New image height.

        Returns:
            Self for method chaining.
        """
        if self.has_calibration():
            self.calibration["width"] = new_w
            self.calibration["height"] = new_h
        return self

    def swap_left_right_bboxes(self) -> "StereoLabels":
        """Swap left and right bboxes (used after horizontal flip with view swap).

        Returns:
            Self for method chaining.
        """
        if self.has_instances() and self.instances.right_bboxes is not None:
            left = self.instances.bboxes.copy()
            right = self.instances.right_bboxes.copy()
            self.instances._bboxes.bboxes[:] = right
            self.instances.right_bboxes = left
        return self

    def mirror_location_3d(self, baseline: float) -> "StereoLabels":
        """Mirror 3D location x-coordinate (used after horizontal flip with view swap).

        Args:
            baseline: Stereo baseline distance.

        Returns:
            Self for method chaining.
        """
        if self.has_instances() and self.instances.location_3d is not None:
            self.instances.location_3d[:, 0] = baseline - self.instances.location_3d[:, 0]
        return self

    def regenerate_2d_bboxes_from_3d(self, image_size: Tuple[int, int]) -> "StereoLabels":
        """Regenerate 2D bboxes by projecting 3D boxes to both cameras.

        This is useful after geometric transforms that modify 3D attributes
        (location, rotation) to ensure 2D bboxes stay consistent.

        Args:
            image_size: (width, height) in pixels.

        Returns:
            Self for method chaining.
        """
        if not self.has_instances() or not self.has_calibration():
            return self

        instances = self.instances
        has_3d = (
            instances.location_3d is not None
            and instances.dimensions_3d is not None
            and instances.rotation_y is not None
        )
        if not has_3d or len(instances) == 0:
            return self

        w, h = image_size
        calib = self.calibration

        # Project to left camera (clip to image bounds)
        left_boxes_px = project_3d_boxes_to_2d(
            instances, calib, camera="left", clip=True, image_size=(w, h)
        )
        # Project to right camera (don't clip - preserve truncated info)
        right_boxes_px = project_3d_boxes_to_2d(
            instances, calib, camera="right", clip=False, image_size=(w, h)
        )

        # Convert to normalized xywh format if instances are normalized
        is_normalized = getattr(instances, "normalized", True)
        if is_normalized:
            left_boxes_px[:, [0, 2]] /= w
            left_boxes_px[:, [1, 3]] /= h
            right_boxes_px[:, [0, 2]] /= w
            right_boxes_px[:, [1, 3]] /= h

        # Convert xyxy to xywh
        left_xywh = np.zeros_like(left_boxes_px)
        left_xywh[:, 0] = (left_boxes_px[:, 0] + left_boxes_px[:, 2]) / 2  # cx
        left_xywh[:, 1] = (left_boxes_px[:, 1] + left_boxes_px[:, 3]) / 2  # cy
        left_xywh[:, 2] = left_boxes_px[:, 2] - left_boxes_px[:, 0]        # w
        left_xywh[:, 3] = left_boxes_px[:, 3] - left_boxes_px[:, 1]        # h

        right_xywh = np.zeros_like(right_boxes_px)
        right_xywh[:, 0] = (right_boxes_px[:, 0] + right_boxes_px[:, 2]) / 2
        right_xywh[:, 1] = (right_boxes_px[:, 1] + right_boxes_px[:, 3]) / 2
        right_xywh[:, 2] = right_boxes_px[:, 2] - right_boxes_px[:, 0]
        right_xywh[:, 3] = right_boxes_px[:, 3] - right_boxes_px[:, 1]

        # Update instances
        instances._bboxes = Bboxes(left_xywh.astype(np.float32), format="xywh")
        instances.right_bboxes = right_xywh.astype(np.float32)

        return self

    def __repr__(self) -> str:
        return f"StereoLabels(instances={self.instances}, calibration={self.calibration})"


# ============================================================================
# Ultralytics-style stereo transforms
# ============================================================================


class StereoHSV:
    """HSV augmentation for stereo images.

    Applies identical HSV augmentation to both left and right views to preserve stereo correspondence.
    Works with 6-channel stereo images (H, W, 6) in labels dict.
    """

    def __init__(self, hgain: float = 0.015, sgain: float = 0.7, vgain: float = 0.4, p: float = 0.5):
        """Initialize StereoHSV.

        Args:
            hgain: Maximum hue gain.
            sgain: Maximum saturation gain.
            vgain: Maximum value gain.
            p: Probability of applying augmentation.
        """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p = p

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """Apply HSV augmentation to stereo image.

        Args:
            labels: Dict with 'img' (6-channel stereo image).

        Returns:
            Updated labels dict.
        """
        if np.random.random() > self.p:
            return labels

        img = labels.get("img")
        if img is None or img.shape[-1] != 6:
            return labels

        # Split stereo image
        left = img[:, :, :3]
        right = img[:, :, 3:6]

        # Generate random gains (same for both views)
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        dtype = left.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # Apply to left
        hue_l, sat_l, val_l = cv2.split(cv2.cvtColor(left, cv2.COLOR_BGR2HSV))
        left_aug = cv2.merge((cv2.LUT(hue_l, lut_hue), cv2.LUT(sat_l, lut_sat), cv2.LUT(val_l, lut_val)))
        left_aug = cv2.cvtColor(left_aug, cv2.COLOR_HSV2BGR)

        # Apply same transform to right
        hue_r, sat_r, val_r = cv2.split(cv2.cvtColor(right, cv2.COLOR_BGR2HSV))
        right_aug = cv2.merge((cv2.LUT(hue_r, lut_hue), cv2.LUT(sat_r, lut_sat), cv2.LUT(val_r, lut_val)))
        right_aug = cv2.cvtColor(right_aug, cv2.COLOR_HSV2BGR)

        # Recombine
        labels["img"] = np.concatenate([left_aug, right_aug], axis=-1)
        return labels


class StereoHFlip:
    """Horizontal flip for stereo images.

    Flips both views horizontally and swaps left/right to preserve stereo geometry.
    Updates instances (bboxes, right_bboxes, rotation_y) and calibration via StereoLabels.
    """

    def __init__(self, p: float = 0.5):
        """Initialize StereoHFlip.

        Args:
            p: Probability of applying flip.
        """
        self.p = p

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """Apply horizontal flip to stereo image and instances.

        Flips both images horizontally and swaps left/right views. For labels:
        1. Flips rotation_y: yaw -> -yaw
        2. Mirrors location_3d x-coordinate: X -> baseline - X
        3. Regenerates BOTH 2D bboxes from 3D projection (handles truncation correctly)
        4. Flips calibration principal point

        Args:
            labels: Dict with 'img' (6-channel), 'instances', 'calibration'.

        Returns:
            Updated labels dict.
        """
        if np.random.random() > self.p:
            return labels

        img = labels.get("img")
        if img is None or img.shape[-1] != 6:
            return labels

        h, w = img.shape[:2]

        # Split, flip, and swap left/right views
        left_f = cv2.flip(img[:, :, :3], 1)
        right_f = cv2.flip(img[:, :, 3:6], 1)
        labels["img"] = np.concatenate([right_f, left_f], axis=-1)

        # Use StereoLabels to synchronize instance and calibration updates
        stereo = StereoLabels.from_labels(labels)

        if stereo.has_instances() and stereo.has_calibration():
            instances = stereo.instances
            baseline = stereo.calibration.get("baseline", 0.0)

            # 1. Flip rotation_y: yaw -> -yaw (normalized to [-pi, pi])
            if instances.rotation_y is not None:
                rot = instances.rotation_y
                instances.rotation_y = np.arctan2(np.sin(-rot), np.cos(-rot)).astype(rot.dtype)

            # 2. Mirror location_3d x coordinate: X -> baseline - X
            if instances.location_3d is not None:
                instances.location_3d[:, 0] = baseline - instances.location_3d[:, 0]

            # 3. Flip calibration principal point BEFORE projecting
            stereo.calibration["cx"] = float((w - 1) - stereo.calibration.get("cx", w / 2))

            # 4. Regenerate 2D bboxes from 3D projection
            # This is the key fix: instead of flip+swap+clip, we project fresh from 3D
            stereo.regenerate_2d_bboxes_from_3d((w, h))

        elif stereo.has_calibration():
            # No instances but still need to flip calibration
            stereo.calibration["cx"] = float((w - 1) - stereo.calibration.get("cx", w / 2))

        return stereo.to_labels(labels)


class StereoScale:
    """Random scale augmentation for stereo images.

    Scales both views by the same factor. Normalized labels remain unchanged
    since the scale operation preserves relative coordinates.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
        """Initialize StereoScale.

        Args:
            scale_range: Min and max scale factors.
            p: Probability of applying scale.
        """
        self.scale_range = scale_range
        self.p = p

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random scale to stereo image.

        Args:
            labels: Dict with 'img' (6-channel).

        Returns:
            Updated labels dict.
        """
        if np.random.random() > self.p:
            return labels

        img = labels.get("img")
        if img is None or img.shape[-1] != 6:
            return labels

        h, w = img.shape[:2]
        s = float(np.random.uniform(*self.scale_range))
        new_w = max(1, int(round(w * s)))
        new_h = max(1, int(round(h * s)))

        # Resize 6-channel image
        img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        labels["img"] = img_scaled

        # Use StereoLabels to synchronize calibration updates
        # Note: Normalized labels don't need instance scaling (relative coords preserved)
        stereo = StereoLabels.from_labels(labels)
        stereo.scale(s, s).update_size(new_w, new_h)

        return stereo.to_labels(labels)


class StereoCrop:
    """Random crop augmentation for stereo images.

    Crops both views identically and updates instances accordingly.
    """

    def __init__(self, crop_h: float = 0.9, crop_w: float = 0.9, p: float = 0.3):
        """Initialize StereoCrop.

        Args:
            crop_h: Crop height ratio (0-1).
            crop_w: Crop width ratio (0-1).
            p: Probability of applying crop.
        """
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.p = p

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random crop to stereo image and update instances.

        Args:
            labels: Dict with 'img' (6-channel), 'instances'.

        Returns:
            Updated labels dict.
        """
        if np.random.random() > self.p:
            return labels

        img = labels.get("img")
        if img is None or img.shape[-1] != 6:
            return labels

        h, w = img.shape[:2]
        ch = max(1, int(h * self.crop_h))
        cw = max(1, int(w * self.crop_w))

        # Random crop position
        y0 = int(np.random.randint(0, max(1, h - ch + 1)))
        x0 = int(np.random.randint(0, max(1, w - cw + 1)))

        # Crop image
        labels["img"] = img[y0 : y0 + ch, x0 : x0 + cw]

        # Use StereoLabels to synchronize instance and calibration updates
        stereo = StereoLabels.from_labels(labels)

        if stereo.has_instances():
            is_normalized = getattr(stereo.instances, "normalized", True)

            # Denormalize to pixel coords, apply crop offset, clip, renormalize
            stereo.denormalize(w, h)
            stereo.add_padding(-x0, -y0)  # Negative offset = crop
            stereo.clip(cw, ch)
            stereo.normalize(cw, ch)

            # Handle right_bboxes (not yet integrated into Instances.add_padding/clip)
            instances = stereo.instances
            if instances.right_bboxes is not None:
                rb = instances.right_bboxes
                if is_normalized:
                    rb = rb * np.array([w, h, w, h])
                rb[:, [0, 2]] -= x0
                rb[:, [1, 3]] -= y0
                # Do not clip right_bboxes so truncated objects keep full projected box
                rb = rb / np.array([cw, ch, cw, ch])
                instances.right_bboxes = rb.astype(np.float32)

        # Update calibration size (principal point already shifted by add_padding)
        stereo.update_size(cw, ch)

        return stereo.to_labels(labels)


class StereoLetterBox:
    """Letterbox transform for stereo images.

    Resizes and pads a 6-channel stereo image to the target size while preserving aspect ratio.
    Updates both instances and calibration via StereoLabels.
    """

    def __init__(self, new_shape: Tuple[int, int] = (640, 640), scaleup: bool = True, stride: int = 32):
        """Initialize StereoLetterBox.

        Args:
            new_shape: Target (height, width).
            scaleup: Whether to scale up smaller images.
            stride: Stride for padding alignment.
        """
        self.new_shape = new_shape
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels: Dict[str, Any]) -> Dict[str, Any]:
        """Apply letterbox transform to stereo image.

        Args:
            labels: Dict with 'img' (6-channel stereo image).

        Returns:
            Updated labels dict with letterboxed image.
        """
        img = labels.get("img")
        if img is None:
            return labels

        # Handle 6-channel stereo image
        if img.shape[-1] != 6:
            return labels

        h, w = img.shape[:2]
        new_h, new_w = self.new_shape

        # Compute scale
        r = min(new_h / h, new_w / w)
        if not self.scaleup:
            r = min(r, 1.0)

        # Compute new size
        new_unpad_h = int(round(h * r))
        new_unpad_w = int(round(w * r))

        # Compute padding
        dh = new_h - new_unpad_h
        dw = new_w - new_unpad_w

        # Divide padding evenly
        top = dh // 2
        bottom = dh - top
        left_pad = dw // 2
        right_pad = dw - left_pad

        # Resize and pad
        if (h, w) != (new_unpad_h, new_unpad_w):
            img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

        # Pad with gray (114) - use numpy for 6-channel images since cv2 only supports up to 4
        img = np.pad(img, ((top, bottom), (left_pad, right_pad), (0, 0)), mode="constant", constant_values=114)

        labels["img"] = img
        labels["resized_shape"] = (new_h, new_w)

        # Use StereoLabels to synchronize instance and calibration updates
        stereo = StereoLabels.from_labels(labels)

        # Apply letterbox transforms: denormalize -> scale -> pad -> normalize
        # This updates both instances AND calibration (fx, fy, cx, cy)
        stereo.denormalize(w, h)
        stereo.scale(r, r)
        stereo.add_padding(left_pad, top)
        stereo.normalize(new_w, new_h)
        stereo.update_size(new_w, new_h)

        return stereo.to_labels(labels)
    
    def __repr__(self) -> str:
        return f"StereoLetterBox(new_shape={self.new_shape}, scaleup={self.scaleup}, stride={self.stride})"
