# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any, List, Optional

import cv2
import numpy as np


def _clip_xyxy_to_image(
    x1: int, y1: int, x2: int, y2: int, W: int, H: int
) -> tuple[int, int, int, int]:
    """Clip xyxy box to image bounds [0, W] x [0, H]. Returns clipped coords (may be degenerate)."""
    x1c = max(0, min(x1, W))
    x2c = max(0, min(x2, W))
    y1c = max(0, min(y1, H))
    y2c = max(0, min(y2, H))
    return x1c, y1c, x2c, y2c


def labels_to_box3d(
    labels: list[dict[str, Any]],
    calib: dict[str, Any] | None,
    image_hw: tuple[int, int],
    class_names: Any = None,
) -> list["Box3D"]:
    """Convert dataset label dicts to Box3D for visualization.

    Delegates to Box3D.from_label() for each label.

    Args:
        labels: List of label dicts from Stereo3DDetDataset (letterboxed space, normalized coords).
        calib: Calibration dict (ideally already transformed to the same letterboxed space as the images).
        image_hw: (H, W) of the image we are drawing onto (letterboxed training tensor).
        class_names: Optional list of class names for Box3D.class_label.
    """
    from ultralytics.data.stereo.box3d import Box3D

    if not labels or calib is None:
        return []
    return [b for lab in labels if (b := Box3D.from_label(lab, calib, class_names, image_hw)) is not None]


def plot_stereo_predictions(
    left_img: np.ndarray,
    right_img: np.ndarray,
    preds: List[dict[str, Any]],
    class_names: Optional[List[str]] = None,
    color: tuple[int, int, int] = (255, 128, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Draw predicted 2D boxes (xyxy) on left/right images. Expects preds with keys:
    - 'bboxes': Tensor/ndarray [N, 4] in xyxy absolute pixels
    - 'cls'   : Tensor/ndarray [N]
    - (optional) 'right_bboxes': like left but for right image if available
    """
    L = left_img.copy()
    R = right_img.copy()

    if not preds:
        return L, R

    p = preds[0]
    b = p.get("bboxes")
    c = p.get("cls")
    rb = p.get("right_bboxes")

    if b is None:
        return L, R

    b = np.asarray(b)
    c = np.asarray(c) if c is not None else np.full((len(b),), -1)

    Hr, Wr = R.shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(b):
        cls = int(c[i]) if i < len(c) else -1
        label = class_names[cls] if class_names and 0 <= cls < len(class_names) else str(cls)
        cv2.rectangle(L, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(L, label, (int(x1), int(max(0, y1 - 5))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if rb is not None:
            x1r, y1r, x2r, y2r = map(int, rb[i])
            x1r, y1r, x2r, y2r = _clip_xyxy_to_image(x1r, y1r, x2r, y2r, Wr, Hr)
            if x2r > x1r and y2r > y1r:
                cv2.rectangle(R, (x1r, y1r), (x2r, y2r), color, 2)
                cv2.putText(R, label, (x1r, int(max(0, y1r - 5))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return L, R
