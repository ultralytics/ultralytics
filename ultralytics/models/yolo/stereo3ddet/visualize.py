# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any, List, Optional

import cv2
import numpy as np


def _denorm_box(cx: float, cy: float, w: float, h: float, W: int, H: int) -> tuple[int, int, int, int]:
    x = cx * W
    y = cy * H
    bw = w * W
    bh = h * H
    x1 = int(round(x - bw / 2))
    y1 = int(round(y - bh / 2))
    x2 = int(round(x + bw / 2))
    y2 = int(round(y + bh / 2))
    assert x2 - x1 > 3
    assert y2 - y1 > 3
    return x1, y1, x2, y2


def plot_stereo_sample(
    left_img: np.ndarray,
    right_img: np.ndarray,
    labels: List[dict[str, Any]],
    class_names: Optional[List[str]] = None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ground-truth stereo boxes from KITTIStereoDataset-like labels on left/right images.

    labels expects entries with keys: class_id, left_box{center_x,center_y,width,height}, right_box{center_x,width}.
    """
    L = left_img.copy()
    R = right_img.copy()
    H, W = L.shape[:2]

    for label in labels:
        cls = int(label["class_id"])
        labelbox_left = label["left_box"]
        labelbox_right = label["right_box"]
        cx, cy, w, h = labelbox_left["center_x"], labelbox_left["center_y"], labelbox_left["width"], labelbox_left["height"]
        x1, y1, x2, y2 = _denorm_box(cx, cy, w, h, W, H)
        cv2.rectangle(L, (x1, y1), (x2, y2), color, 2)
        label_name = class_names[cls]
        cv2.putText(L, label_name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Right image (center_x and width only available)
        cx_r = labelbox_right["center_x"]
        cy_r = labelbox_right["center_y"]
        w_r = labelbox_right["width"]
        h_r = labelbox_right["height"]
        x1r, y1r, x2r, y2r = _denorm_box(cx_r, cy_r, w_r, h_r, W, H)
        cv2.rectangle(R, (x1r, y1r), (x2r, y2r), color, 2)
        cv2.putText(R, label_name, (x1r, max(0, y1r - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return L, R


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

    for i, (x1, y1, x2, y2) in enumerate(b):
        cls = int(c[i]) if i < len(c) else -1
        label = class_names[cls] if class_names and 0 <= cls < len(class_names) else str(cls)
        cv2.rectangle(L, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(L, label, (int(x1), int(max(0, y1 - 5))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if rb is not None:
            x1r, y1r, x2r, y2r = map(int, rb[i])
            cv2.rectangle(R, (x1r, y1r), (x2r, y2r), color, 2)
            cv2.putText(R, label, (x1r, int(max(0, y1r - 5))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return L, R
