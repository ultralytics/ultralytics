# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Stereo-consistent data augmentation maintaining epipolar geometry."""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch

from ultralytics.data.stereo.calib import CalibrationParameters


class StereoAugment:
    """Stereo-consistent data augmentation.

    Maintains epipolar geometry by applying synchronized transforms to both images.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        hue: float = 0.1,
        flip_prob: float = 0.5,
        scale_range: tuple[float, float] = (0.8, 1.2),
    ):
        """Initialize stereo augmentation.

        Args:
            brightness: Brightness adjustment range.
            contrast: Contrast adjustment range.
            hue: Hue adjustment range.
            flip_prob: Probability of horizontal flip.
            scale_range: Scale range (min, max).
        """
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.flip_prob = flip_prob
        self.scale_range = scale_range

    def __call__(
        self,
        left_img: np.ndarray | torch.Tensor,
        right_img: np.ndarray | torch.Tensor,
        calib: CalibrationParameters | None = None,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, CalibrationParameters | None]:
        """Apply stereo-consistent augmentation.

        Args:
            left_img: Left image [H, W, 3] or [3, H, W].
            right_img: Right image [H, W, 3] or [3, H, W].
            calib: Calibration parameters (will be updated if scaling applied).

        Returns:
            Tuple of (augmented_left, augmented_right, updated_calib).
        """
        # Convert to numpy if needed
        if isinstance(left_img, torch.Tensor):
            left_img = left_img.numpy()
            right_img = right_img.numpy()
            is_tensor = True
        else:
            is_tensor = False

        # Handle CHW format
        if len(left_img.shape) == 3 and left_img.shape[0] == 3:
            left_img = left_img.transpose(1, 2, 0)  # CHW -> HWC
            right_img = right_img.transpose(1, 2, 0)  # CHW -> HWC

        # 1. Synchronized photometric augmentation
        left_img, right_img = self._photometric_augment(left_img, right_img)

        # 2. Horizontal flip with image swap
        if random.random() < self.flip_prob:
            left_img, right_img = self._horizontal_flip(left_img, right_img)

        # 3. Synchronized scaling (with calib update)
        if self.scale_range[0] < 1.0 or self.scale_range[1] > 1.0:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            left_img, right_img, calib = self._synchronized_scale(left_img, right_img, calib, scale)

        # Convert back to original format
        if is_tensor:
            left_img = torch.from_numpy(left_img)
            right_img = torch.from_numpy(right_img)
        elif len(left_img.shape) == 3 and left_img.shape[2] == 3:
            # Convert back to CHW if needed
            left_img = left_img.transpose(2, 0, 1)  # HWC -> CHW
            right_img = right_img.transpose(2, 0, 1)  # HWC -> CHW

        return left_img, right_img, calib

    def _photometric_augment(
        self, left_img: np.ndarray, right_img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply synchronized photometric augmentation.

        Args:
            left_img: Left image [H, W, 3].
            right_img: Right image [H, W, 3].

        Returns:
            Augmented images.
        """
        # Brightness
        if self.brightness > 0:
            delta = random.uniform(-self.brightness, self.brightness)
            left_img = np.clip(left_img + delta, 0, 1)
            right_img = np.clip(right_img + delta, 0, 1)

        # Contrast
        if self.contrast > 0:
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            left_img = np.clip(alpha * (left_img - 0.5) + 0.5, 0, 1)
            right_img = np.clip(alpha * (right_img - 0.5) + 0.5, 0, 1)

        # Hue (simplified - would need HSV conversion for proper hue shift)
        # For now, skip hue augmentation to maintain simplicity

        return left_img, right_img

    def _horizontal_flip(
        self, left_img: np.ndarray, right_img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply horizontal flip with image swap.

        Args:
            left_img: Left image [H, W, 3].
            right_img: Right image [H, W, 3].

        Returns:
            Flipped and swapped images.
        """
        # Flip both images horizontally
        left_flipped = np.fliplr(left_img)
        right_flipped = np.fliplr(right_img)

        # Swap left and right (maintains stereo geometry)
        return right_flipped, left_flipped

    def _synchronized_scale(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        calib: CalibrationParameters | None,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, CalibrationParameters | None]:
        """Apply synchronized scaling with calibration update.

        Args:
            left_img: Left image [H, W, 3].
            right_img: Right image [H, W, 3].
            calib: Calibration parameters.
            scale: Scale factor.

        Returns:
            Scaled images and updated calibration.
        """
        h, w = left_img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize both images
        left_scaled = cv2.resize(left_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        right_scaled = cv2.resize(right_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Update calibration parameters
        if calib is not None:
            calib = CalibrationParameters(
                fx=calib.fx * scale,
                fy=calib.fy * scale,
                cx=calib.cx * scale,
                cy=calib.cy * scale,
                baseline=calib.baseline,  # Baseline doesn't change with image scaling
                image_width=new_w,
                image_height=new_h,
            )

        return left_scaled, right_scaled, calib

