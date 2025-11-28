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
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if np.random.rand() >= self.p_apply:
            return left, right, labels

        # Flip both and swap
        left_f = cv2.flip(left, 1)
        right_f = cv2.flip(right, 1)
        left_f, right_f = right_f, left_f

        # Update labels
        new_labels: List[Dict[str, Any]] = []
        for obj in labels:
            lb = dict(obj.get("left_box", {}))
            rb = dict(obj.get("right_box", {}))
            # flip x center and swap left/right boxes
            lb_flipped = {
                "center_x": self._flip_norm_x(rb.get("center_x", 0.0)),
                "center_y": lb.get("center_y", 0.0),
                "width": lb.get("width", 0.0),
                "height": lb.get("height", 0.0),
            }
            rb_flipped = {
                "center_x": self._flip_norm_x(lb.get("center_x", 0.0)),
                "width": rb.get("width", 0.0),
            }
            new_obj = dict(obj)
            new_obj["left_box"] = lb_flipped
            new_obj["right_box"] = rb_flipped
            new_labels.append(new_obj)

        return left_f, right_f, new_labels


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
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if np.random.rand() >= self.p_apply:
            return left, right, labels
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
            lb = dict(obj.get("left_box", {}))
            rb = dict(obj.get("right_box", {}))
            # denorm
            cx_px = float(lb.get("center_x", 0.0)) * W
            cy_px = float(lb.get("center_y", 0.0)) * H
            w_px = float(lb.get("width", 0.0)) * W
            h_px = float(lb.get("height", 0.0)) * H
            rx_px = float(rb.get("center_x", 0.0)) * W
            rw_px = float(rb.get("width", 0.0)) * W

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
                "width": rw_px / cw,
            }
            new_obj = dict(obj)
            new_obj["left_box"] = lb_new
            new_obj["right_box"] = rb_new
            new_labels.append(new_obj)

        return left_c, right_c, new_labels


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
        left, right, labels = self.hflip(left, right, labels)

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

        left, right, labels = self.rcrop(left, right, labels)
        # Photometric last
        left, right = self.photometric(left, right)
        return left, right, labels, calibration
