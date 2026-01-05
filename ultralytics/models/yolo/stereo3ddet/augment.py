from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import cv2
import numpy as np


@dataclass
class StereoCalibration:
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


class PhotometricAugmentor:
    """Wrapper around Ultralytics RandomHSV to apply identical HSV augmentation to both stereo views.

    Ensures the same random gains are used for left and right images to preserve stereo correspondence.
    Falls back to returning images unchanged if probability threshold not met or images are not 3-channel.
    """

    def __init__(
        self,
        hgain: float = 0.5,
        sgain: float = 0.5,
        vgain: float = 0.5,
        p_apply: float = 0.5,
        seed: Optional[int] = None,
    ):
        from ultralytics.data.augment import RandomHSV  # lazy import

        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p_apply = float(p_apply)
        self._hsv_impl = RandomHSV(hgain=hgain, sgain=sgain, vgain=vgain)
        if seed is not None:
            np.random.seed(seed)

    def __call__(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() >= self.p_apply:
            return left, right
        if left.shape[-1] != 3 or right.shape[-1] != 3:
            return left, right
        # Save RNG state to replicate identical gains
        state = np.random.get_state()
        left_labels = {"img": left.copy()}
        left_aug = self._hsv_impl(left_labels)["img"]
        np.random.set_state(state)
        right_labels = {"img": right.copy()}
        right_aug = self._hsv_impl(right_labels)["img"]
        return left_aug, right_aug


class HorizontalFlipAugmentor:
    """Horizontal flip that preserves stereo by swapping views and updating boxes.

    Operates on normalized boxes in our parsed label format:
    - left_box: {center_x, center_y, width, height}
    - right_box: {center_x, width}
    """

    def __init__(self, p_apply: float = 0.5):
        self.p_apply = float(p_apply)

    @staticmethod
    def _flip_norm_x(x: float) -> float:
        return 1.0 - x

    def __call__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        labels: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], bool]:
        if np.random.rand() >= self.p_apply:
            return left, right, labels, False

        # Flip both and swap
        left_f = cv2.flip(left, 1)
        right_f = cv2.flip(right, 1)
        left_f, right_f = right_f, left_f

        # Update labels
        new_labels: List[Dict[str, Any]] = []
        for obj in labels:
            lb = dict(obj["left_box"])
            rb = dict(obj["right_box"])
            # flip x center and swap left/right boxes
            lb_flipped = {
                "center_x": self._flip_norm_x(float(rb["center_x"])),
                "center_y": lb["center_y"],
                "width": lb["width"],
                "height": lb["height"],
            }
            rb_flipped = {
                "center_x": self._flip_norm_x(float(lb["center_x"])),
                "center_y": rb["center_y"],
                "width": rb["width"],
                "height": rb["height"],
            }
            new_obj = dict(obj)
            new_obj["left_box"] = lb_flipped
            new_obj["right_box"] = rb_flipped
            # 3D orientation must be mirrored on horizontal flip.
            # Reflection across the camera YZ plane maps yaw -> -yaw (see derivation in issue discussion).
            if "rotation_y" in new_obj:
                rot = float(new_obj["rotation_y"])
                new_obj["rotation_y"] = float(np.arctan2(np.sin(-rot), np.cos(-rot)))
            # Vertices are view-specific 2D keypoints; after swapping cameras we can't reliably transform them here.
            # Drop them so downstream target generation recomputes vertices from 3D parameters.
            if "vertices" in new_obj:
                new_obj.pop("vertices", None)
            new_labels.append(new_obj)

        return left_f, right_f, new_labels, True


class RandomScaleAugmentor:
    """Uniform random scaling of images with corresponding label scaling (normalized).

    Since labels are normalized, scaling the image does not change normalized values,
    but subsequent letterbox/pad may. We apply scaling here to the raw images only.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), p_apply: float = 0.5):
        self.scale_range = scale_range
        self.p_apply = float(p_apply)

    def __call__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        labels: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if np.random.rand() >= self.p_apply:
            return left, right, labels
        s = float(np.random.uniform(*self.scale_range))
        new_w = max(1, int(round(left.shape[1] * s)))
        new_h = max(1, int(round(left.shape[0] * s)))
        left_s = cv2.resize(left, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        right_s = cv2.resize(right, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # normalized labels unchanged
        return left_s, right_s, labels


class RandomCropAugmentor:
    """Identical random crop for left/right; adjusts normalized label centers.

    Applies a crop window and updates centers in normalized coordinates.
    Width/height are preserved unless cropped; we clamp boxes to remain inside.
    """

    def __init__(self, crop_height_ratio: float = 0.9, crop_width_ratio: float = 0.9, p_apply: float = 0.3):
        self.crop_height_ratio = crop_height_ratio
        self.crop_width_ratio = crop_width_ratio
        self.p_apply = float(p_apply)

    def __call__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        labels: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], bool, int, int, int, int]:
        if np.random.rand() >= self.p_apply:
            return left, right, labels, False, 0, 0, left.shape[1], left.shape[0]
        H, W = left.shape[:2]
        ch = max(1, int(H * self.crop_height_ratio))
        cw = max(1, int(W * self.crop_width_ratio))
        y0 = int(np.random.randint(0, H - ch + 1))
        x0 = int(np.random.randint(0, W - cw + 1))
        left_c = left[y0 : y0 + ch, x0 : x0 + cw]
        right_c = right[y0 : y0 + ch, x0 : x0 + cw]

        # Update labels: shift centers by crop and renormalize to new size
        new_labels: List[Dict[str, Any]] = []
        for obj in labels:
            lb = dict(obj["left_box"])
            rb = dict(obj["right_box"])
            # denorm
            cx_px = float(lb["center_x"]) * W
            cy_px = float(lb["center_y"]) * H
            w_px = float(lb["width"]) * W
            h_px = float(lb["height"]) * H
            rx_px = float(rb["center_x"]) * W
            ry_px = float(rb["center_y"]) * H
            rw_px = float(rb["width"]) * W
            rh_px = float(rb["height"]) * H

            # shift
            cx_px -= x0
            cy_px -= y0
            rx_px -= x0

            # clamp box to crop bounds
            cx_px = float(min(max(cx_px, 0.0), cw))
            cy_px = float(min(max(cy_px, 0.0), ch))

            # renorm to new size
            lb_new = {
                "center_x": cx_px / cw,
                "center_y": cy_px / ch,
                "width": w_px / cw,
                "height": h_px / ch,
            }
            rb_new = {
                "center_x": rx_px / cw,
                "center_y": ry_px / ch,
                "width": rw_px / cw,
                "height": rh_px / ch,
            }
            new_obj = dict(obj)
            new_obj["left_box"] = lb_new
            new_obj["right_box"] = rb_new
            new_labels.append(new_obj)

        return left_c, right_c, new_labels, True, x0, y0, cw, ch


class StereoAugmentationPipeline:
    """Stereo augmentation pipeline combining photometric, flip, scale and crop.

    Applies geometric augs first (flip/scale/crop) then photometric.
    """

    def __init__(
        self,
        photometric: PhotometricAugmentor | None = None,
        hflip: HorizontalFlipAugmentor | None = None,
        rscale: RandomScaleAugmentor | None = None,
        rcrop: RandomCropAugmentor | None = None,
    ):
        self.photometric = photometric or PhotometricAugmentor()
        self.hflip = hflip or HorizontalFlipAugmentor()
        self.rscale = rscale or RandomScaleAugmentor()
        self.rcrop = rcrop or RandomCropAugmentor()

    def augment(
        self,
        left: np.ndarray,
        right: np.ndarray,
        labels: List[Dict[str, Any]],
        calibration: StereoCalibration | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], StereoCalibration | None]:
        # Geometric first
        left, right, labels, did_flip = self.hflip(left, right, labels)
        if did_flip and calibration is not None:
            # After cv2.flip, pixel u becomes u' = (W - 1) - u, so principal point must update too.
            calibration.cx = float((calibration.width - 1) - calibration.cx)
            # Swapping cameras changes the origin for 3D X in left-camera coordinates.
            # If original left-camera center_x is x_l, then in original right-camera coords x_r = x_l - baseline.
            # After horizontal reflection, x_new = -x_r = baseline - x_l.
            b = float(calibration.baseline)
            for lab in labels:
                loc = lab.get("location_3d")
                if isinstance(loc, dict) and "x" in loc:
                    try:
                        x_l = float(loc["x"])
                        loc["x"] = float(b - x_l)
                    except Exception:
                        pass

        # Keep track of original size to adjust calibration if scaled
        orig_h, orig_w = left.shape[:2]
        left, right, labels = self.rscale(left, right, labels)
        new_h, new_w = left.shape[:2]

        # If calibration provided and size changed, update intrinsics to preserve geometry
        if calibration is not None and (new_h != orig_h or new_w != orig_w):
            sx = new_w / float(orig_w)
            sy = new_h / float(orig_h)
            # Scale focal lengths and principal points; baseline unchanged
            calibration.fx *= sx
            calibration.fy *= sy
            calibration.cx *= sx
            calibration.cy *= sy
            calibration.width = int(new_w)
            calibration.height = int(new_h)

        left, right, labels, did_crop, x0, y0, cw, ch = self.rcrop(left, right, labels)
        if did_crop and calibration is not None:
            # Cropping shifts the pixel coordinate frame by (x0, y0).
            calibration.cx = float(calibration.cx - x0)
            calibration.cy = float(calibration.cy - y0)
            calibration.width = int(cw)
            calibration.height = int(ch)
            # Keep cx/cy within the new image bounds to avoid numerical weirdness.
            calibration.cx = float(np.clip(calibration.cx, 0.0, max(0, calibration.width - 1)))
            calibration.cy = float(np.clip(calibration.cy, 0.0, max(0, calibration.height - 1)))
        # Photometric last
        left, right = self.photometric(left, right)
        return left, right, labels, calibration
