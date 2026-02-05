# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, List, Optional

import cv2
import numpy as np


def _class_name(class_names: Any, cls: int) -> str:
    """Resolve class id to name for both list/tuple and dict-like mappings."""
    if class_names is None:
        return str(cls)
    if isinstance(class_names, Mapping):
        return str(class_names.get(cls, cls))
    if isinstance(class_names, Sequence) and not isinstance(class_names, (str, bytes)):
        return class_names[cls] if 0 <= cls < len(class_names) else str(cls)
    return str(cls)


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


def _clip_xyxy_to_image(
    x1: int, y1: int, x2: int, y2: int, W: int, H: int
) -> tuple[int, int, int, int]:
    """Clip xyxy box to image bounds [0, W] x [0, H]. Returns clipped coords (may be degenerate)."""
    x1c = max(0, min(x1, W))
    x2c = max(0, min(x2, W))
    y1c = max(0, min(y1, H))
    y2c = max(0, min(y2, H))
    return x1c, y1c, x2c, y2c


def _left_box_from_3d(
    label: dict[str, Any],
    calib: dict[str, Any],
    W: int,
    H: int,
) -> Optional[tuple[float, float, float, float]]:
    """Compute left-camera 2D box (normalized cx, cy, w, h) from 3D box. Returns None if invalid."""
    loc = label.get("location_3d")
    dims = label.get("dimensions")
    if not isinstance(loc, dict) or not isinstance(dims, dict):
        return None
    for k in ("x", "y", "z"):
        if k not in loc:
            return None
    for k in ("length", "width", "height"):
        if k not in dims:
            return None
    fx = float(calib.get("fx", 0))
    fy = float(calib.get("fy", 0))
    cx = float(calib.get("cx", 0))
    cy = float(calib.get("cy", 0))
    if fx <= 0 or fy <= 0:
        return None
    X = float(loc["x"])
    Y = float(loc["y"])  # KITTI bottom-center
    Z = float(loc["z"])
    length = float(dims["length"])
    width = float(dims["width"])
    height = float(dims["height"])
    ry = float(label.get("rotation_y", 0))
    # 8 corners in object frame: bottom y=0, top y=-height (camera Y down)
    corners_obj = np.array([
        [-length / 2, 0, -width / 2],
        [length / 2, 0, -width / 2],
        [length / 2, 0, width / 2],
        [-length / 2, 0, width / 2],
        [-length / 2, -height, -width / 2],
        [length / 2, -height, -width / 2],
        [length / 2, -height, width / 2],
        [-length / 2, -height, width / 2],
    ])
    cos_ry = np.cos(ry)
    sin_ry = np.sin(ry)
    R = np.array([[cos_ry, 0, sin_ry], [0, 1, 0], [-sin_ry, 0, cos_ry]])
    corners_cam = corners_obj @ R.T + np.array([X, Y, Z])
    us, vs = [], []
    for c in corners_cam:
        if c[2] <= 1e-6:
            continue
        u = fx * (c[0] / c[2]) + cx
        v = fy * (c[1] / c[2]) + cy
        us.append(u)
        vs.append(v)
    if len(us) < 2:
        return None
    u1, u2 = min(us), max(us)
    v1, v2 = min(vs), max(vs)
    u1 = max(0, min(u1, W))
    u2 = max(0, min(u2, W))
    v1 = max(0, min(v1, H))
    v2 = max(0, min(v2, H))
    if u2 <= u1 or v2 <= v1:
        return None
    cx_px = (u1 + u2) / 2
    cy_px = (v1 + v2) / 2
    bw = u2 - u1
    bh = v2 - v1
    return (cx_px / W, cy_px / H, bw / W, bh / H)


def plot_stereo_sample(
    left_img: np.ndarray,
    right_img: np.ndarray,
    labels: List[dict[str, Any]],
    class_names: Any = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
    calib: Optional[dict[str, Any]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ground-truth stereo boxes from KITTIStereoDataset-like labels on left/right images.

    labels expects entries with keys: class_id, left_box{center_x,center_y,width,height}, right_box{center_x,width}.
    If calib is provided and a label has location_3d/dimensions, the left 2D box is drawn from 3D projection
    so it matches the 3D wireframe (avoids wide boxes after flip/truncation).
    """
    L = left_img.copy()
    R = right_img.copy()
    H, W = L.shape[:2]

    for label in labels:
        cls = int(label["class_id"])
        labelbox_left = label["left_box"]
        labelbox_right = label["right_box"]
        # Prefer 3D-projected left box when calib and 3D data exist (aligns 2D box with 3D wireframe)
        box_3d = _left_box_from_3d(label, calib, W, H) if calib else None
        if box_3d is not None and box_3d[2] > 0.01 and box_3d[3] > 0.01:
            cx, cy, w, h = box_3d
        else:
            cx, cy, w, h = labelbox_left["center_x"], labelbox_left["center_y"], labelbox_left["width"], labelbox_left["height"]
        # Denorm left box; clip to image
        x_c_l = cx * W
        y_c_l = cy * H
        bw_l = w * W
        bh_l = h * H
        x1 = int(round(x_c_l - bw_l / 2))
        y1 = int(round(y_c_l - bh_l / 2))
        x2 = int(round(x_c_l + bw_l / 2))
        y2 = int(round(y_c_l + bh_l / 2))
        x1, y1, x2, y2 = _clip_xyxy_to_image(x1, y1, x2, y2, W, H)
        label_name = _class_name(class_names, cls)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(L, (x1, y1), (x2, y2), color, max(1, int(thickness)))
            cv2.putText(
                L,
                label_name,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(font_scale),
                color,
                1,
                cv2.LINE_AA,
            )

        # Right image: denorm then clip to image so truncated boxes draw only visible part
        cx_r = labelbox_right["center_x"]
        cy_r = labelbox_right["center_y"]
        w_r = labelbox_right["width"]
        h_r = labelbox_right["height"]
        x_c = cx_r * W
        y_c = cy_r * H
        bw_r = w_r * W
        bh_r = h_r * H
        x1r = int(round(x_c - bw_r / 2))
        y1r = int(round(y_c - bh_r / 2))
        x2r = int(round(x_c + bw_r / 2))
        y2r = int(round(y_c + bh_r / 2))
        x1r, y1r, x2r, y2r = _clip_xyxy_to_image(x1r, y1r, x2r, y2r, W, H)
        if x2r > x1r and y2r > y1r:
            cv2.rectangle(R, (x1r, y1r), (x2r, y2r), color, max(1, int(thickness)))
            cv2.putText(
                R,
                label_name,
                (x1r, max(0, y1r - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(font_scale),
                color,
                1,
                cv2.LINE_AA,
            )

    return L, R


def labels_to_box3d(
    labels: list[dict[str, Any]],
    calib: dict[str, Any] | None,
    image_hw: tuple[int, int],
    class_names: Any = None,
) -> list["Box3D"]:
    """Convert dataset label dicts to Box3D for visualization.

    Prefer ground-truth `location_3d` when present (more accurate for large/near objects).
    Fall back to disparity-based reconstruction only if `location_3d` is missing.

    Args:
        labels: List of label dicts from Stereo3DDetDataset (letterboxed space, normalized coords).
        calib: Calibration dict (ideally already transformed to the same letterboxed space as the images).
        image_hw: (H, W) of the image we are drawing onto (letterboxed training tensor).
        class_names: Optional list of class names for Box3D.class_label.
    """
    from ultralytics.data.stereo.box3d import Box3D  # local import to avoid circulars

    if not labels or calib is None:
        return []

    H, W = int(image_hw[0]), int(image_hw[1])
    fx = float(calib.get("fx", 0.0))
    fy = float(calib.get("fy", 0.0))
    cx = float(calib.get("cx", 0.0))
    cy = float(calib.get("cy", 0.0))
    baseline = float(calib.get("baseline", 0.0))
    if fx <= 0 or fy <= 0:
        return []

    boxes3d: list[Box3D] = []
    eps = 1e-6

    for lab in labels:
        try:
            class_id = int(lab["class_id"])
            lb = lab["left_box"]
            rb = lab.get("right_box", None)

            dims = lab.get("dimensions", {})
            length = float(dims.get("length", 1.0))
            width = float(dims.get("width", 1.0))
            height = float(dims.get("height", 1.0))

            rot_y = float(lab.get("rotation_y", 0.0))

            # Prefer GT 3D location when present. KITTI `location` is bottom-center; convert to geometric center.
            loc = lab.get("location_3d", None)
            if isinstance(loc, dict) and all(k in loc for k in ("x", "y", "z")):
                x_3d = float(loc["x"])
                y_bottom = float(loc["y"])
                z_3d = float(loc["z"])
                y_3d = y_bottom - height / 2.0  # bottom-center -> geometric center (Y points down)
            else:
                # Fallback: reconstruct from stereo disparity between box centers (less accurate for large/near objects).
                if rb is None or baseline <= 0:
                    continue
                left_u = float(lb["center_x"]) * W
                right_u = float(rb["center_x"]) * W
                disparity = left_u - right_u
                if not np.isfinite(disparity) or disparity <= eps:
                    continue
                z_3d = (fx * baseline) / max(disparity, eps)
                center_x_2d = left_u
                center_y_2d = float(lb.get("center_y", 0.5)) * H
                x_3d = (center_x_2d - cx) * z_3d / fx
                y_3d = (center_y_2d - cy) * z_3d / fy

            # Pixel-space 2D bbox (left) for reference/debug (not used by 3D wireframe drawing).
            center_x_2d = float(lb.get("center_x", 0.5)) * W
            center_y_2d = float(lb.get("center_y", 0.5)) * H
            bw = float(lb.get("width", 0.0)) * W
            bh = float(lb.get("height", 0.0)) * H
            x1 = center_x_2d - bw / 2
            y1 = center_y_2d - bh / 2
            x2 = center_x_2d + bw / 2
            y2 = center_y_2d + bh / 2

            class_label = _class_name(class_names, class_id)

            boxes3d.append(
                Box3D(
                    center_3d=(float(x_3d), float(y_3d), float(z_3d)),
                    dimensions=(float(length), float(width), float(height)),
                    orientation=float(rot_y),
                    class_label=class_label,
                    class_id=class_id,
                    confidence=1.0,
                    bbox_2d=(float(x1), float(y1), float(x2), float(y2)),
                    truncated=float(lab.get("truncated", 0.0)) if lab.get("truncated") is not None else None,
                    occluded=int(lab.get("occluded")) if lab.get("occluded") is not None else None,
                )
            )
        except Exception:
            continue

    return boxes3d


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
