# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ultralytics.models.yolo.detect import DetectionPredictor


def load_stereo_pair(
    left_path: str | Path,
    right_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load stereo image pair (left and right images).

    Args:
        left_path: Path to left image file.
        right_path: Path to right image file.

    Returns:
        tuple: (left_image, right_image) as numpy arrays in BGR format.

    Raises:
        FileNotFoundError: If either image file does not exist.
        ValueError: If images cannot be loaded or have different sizes.
    """
    left_path = Path(left_path)
    right_path = Path(right_path)

    if not left_path.exists():
        raise FileNotFoundError(f"Left image not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Right image not found: {right_path}")

    # Load images using OpenCV (BGR format)
    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    if left_img is None:
        raise ValueError(f"Failed to load left image: {left_path}")
    if right_img is None:
        raise ValueError(f"Failed to load right image: {right_path}")

    # Verify images have the same size
    if left_img.shape != right_img.shape:
        raise ValueError(
            f"Image size mismatch: left {left_img.shape} vs right {right_img.shape}"
        )

    return left_img, right_img


class Stereo3DDetPredictor(DetectionPredictor):
    """Stereo 3D Detection predictor.

    Reuses the detection predictor for now. Custom stereo visualization can be layered on top.
    """

    pass
