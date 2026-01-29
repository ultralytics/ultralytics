from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional
import cv2
import numpy as np
from ultralytics.data.augment import Compose, Format
from ultralytics.utils.instance import Instances


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
        hue, sat, val = cv2.split(cv2.cvtColor(left, cv2.COLOR_BGR2HSV))
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

        if stereo.has_instances():
            instances = stereo.instances
            # Convert to xywh format for simpler flip logic
            instances.convert_bbox(format="xywh")
            flip_w = 1 if instances.normalized else w

            # Flip center_x coordinates for both bboxes
            instances.bboxes[:, 0] = flip_w - instances.bboxes[:, 0]
            if instances.right_bboxes is not None:
                instances.right_bboxes[:, 0] = flip_w - instances.right_bboxes[:, 0]

            # Swap left and right bboxes
            stereo.swap_left_right_bboxes()

            # Flip rotation_y: yaw -> -yaw (normalized to [-pi, pi])
            if instances.rotation_y is not None:
                rot = instances.rotation_y
                instances.rotation_y = np.arctan2(np.sin(-rot), np.cos(-rot)).astype(rot.dtype)

            # Mirror location_3d x coordinate
            if instances.location_3d is not None and stereo.has_calibration():
                baseline = stereo.calibration.get("baseline", 0.0)
                stereo.mirror_location_3d(baseline)

        # Flip calibration principal point (synchronized via StereoLabels)
        if stereo.has_calibration():
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
                rb[:, [0, 2]] = np.clip(rb[:, [0, 2]], 0, cw)
                rb[:, [1, 3]] = np.clip(rb[:, [1, 3]], 0, ch)
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
