# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Render pose3d predictions from viewpoints the camera never had.

A skeleton drawn at the predicted pixel coordinates is just the 2D pose, however it is tinted — the depth is
in the data but not in the picture. The only render that actually shows the 3D is one taken from somewhere
else: lift every joint to metric camera space, rotate the viewpoint, and draw it again. If the depth is real
the rotated body still reads as a person; if it is flat, the side view collapses into a plane.

Each figure is the image with its 2D overlay, then per person a strip of orthographic views at increasing
azimuth, ending at 90 degrees — a pure side view, where nothing of the original projection survives.

Usage:
    python research/src/render_pose3d.py --model <best.pt> --images a.jpg b.jpg --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.pose3d import decode_z

SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
AZIMUTHS = (0, 30, 60, 90)
VIEW = 240  # pixels per orthographic view
BODY_M = 2.0  # metres spanned by a view box, so every person is drawn at the same scale


def to_camera(kpts: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    """Lift one person's (18, 4) prediction to metric camera-space joints, shape (17, 3), root-centred."""
    z_rel, z_root = decode_z(kpts[:17, 3], kpts[17, 3])
    z = z_rel + z_root
    x = (kpts[:17, 0] - cx) * z / focal
    y = (kpts[:17, 1] - cy) * z / focal
    j = np.stack([x, y, z], 1)
    return j - (j[11] + j[12]) / 2  # centre on the mid-hip root


def draw_view(canvas: np.ndarray, org: tuple[int, int], joints: np.ndarray, vis: np.ndarray, az: float) -> None:
    """Draw one orthographic view of a root-centred skeleton, rotated `az` degrees about the vertical axis."""
    x0, y0 = org
    t = np.radians(az)
    r = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    p = joints @ r.T
    s = VIEW / BODY_M
    px = (x0 + VIEW / 2 + p[:, 0] * s).astype(int)
    py = (y0 + VIEW / 2 + p[:, 1] * s).astype(int)

    cv2.rectangle(canvas, (x0, y0), (x0 + VIEW, y0 + VIEW), (52, 58, 68), 1)
    cv2.putText(canvas, f"{int(az)}deg", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 158, 170), 1)

    order = np.argsort(-p[:, 2])  # far joints first, so near limbs overdraw them
    for a, b in SKELETON:
        if vis[a] > 0.5 and vis[b] > 0.5:
            c = (250, 170, 90) if (a in LEFT and b in LEFT) else (110, 150, 250)
            cv2.line(canvas, (px[a], py[a]), (px[b], py[b]), c, 2, cv2.LINE_AA)
    for j in order:
        if vis[j] > 0.5:
            depth = float(np.clip((p[j, 2] + 0.5) / 1.0, 0, 1))  # nearer joints drawn larger
            cv2.circle(canvas, (px[j], py[j]), int(5 - 2 * depth), (245, 245, 245), -1, cv2.LINE_AA)


def gt_depths(lb: Path, boxes: np.ndarray, shape: tuple[int, int]) -> list[float | None]:
    """Return each prediction's ground-truth root depth by box IoU, or None where nothing matches."""
    if not lb.exists():
        return [None] * len(boxes)
    g = np.array([x.split() for x in lb.read_text().strip().splitlines()], dtype=np.float32)
    h, w = shape
    gb = np.stack(
        [
            (g[:, 1] - g[:, 3] / 2) * w,
            (g[:, 2] - g[:, 4] / 2) * h,
            (g[:, 1] + g[:, 3] / 2) * w,
            (g[:, 2] + g[:, 4] / 2) * h,
        ],
        1,
    )
    gz = decode_z(g[:, 5:].reshape(len(g), -1, 4)[:, :17, 3], g[:, 5:].reshape(len(g), -1, 4)[:, 17, 3])[1]
    out = []
    for b in boxes:
        x1 = np.maximum(gb[:, 0], b[0])
        y1 = np.maximum(gb[:, 1], b[1])
        x2 = np.minimum(gb[:, 2], b[2])
        y2 = np.minimum(gb[:, 3], b[3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        union = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        iou = inter / np.clip(union, 1e-9, None)
        j = int(iou.argmax())
        out.append(float(gz[j]) if iou[j] >= 0.5 else None)
    return out


def render(
    img: np.ndarray, kpts: np.ndarray, focal: float, max_people: int = 3, gt: list[float | None] | None = None
) -> np.ndarray:
    """Return the image with its 2D overlay beside a rotation strip per person."""
    h, w = img.shape[:2]
    people = list(kpts)[:max_people]
    strip_w = VIEW * len(AZIMUTHS) + 20
    row_h = VIEW + 46
    canvas = np.full((max(h, row_h * len(people) + 10), w + strip_w, 3), 26, np.uint8)
    canvas[:h, :w] = img

    for person in people:  # 2D overlay, purely as the reference view
        xy, vis = person[:, :2].astype(int), person[:, 2]
        for a, b in SKELETON:
            if vis[a] > 0.5 and vis[b] > 0.5:
                c = (250, 170, 90) if (a in LEFT and b in LEFT) else (110, 150, 250)
                cv2.line(canvas, tuple(xy[a]), tuple(xy[b]), c, 2, cv2.LINE_AA)

    for i, person in enumerate(people):
        y0 = 10 + i * row_h
        j = to_camera(person, focal, w / 2, h / 2)
        _, z_root = decode_z(person[:17, 3], person[17, 3])
        label = f"person {i + 1}   predicted {float(z_root):.2f} m"
        g = None if gt is None else gt[i]
        if g is not None:
            label += f"   ground truth {g:.2f} m   error {100 * abs(float(z_root) - g) / g:.1f}%"
        cv2.putText(
            canvas,
            label,
            (w + 12, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210, 214, 222) if g is None else (150, 235, 150),
            1,
            cv2.LINE_AA,
        )
        for k, az in enumerate(AZIMUTHS):
            draw_view(canvas, (w + 12 + k * VIEW, y0 + 26), j, person[:, 2], az)
    return canvas


def main(model: str, images: list[str], out: Path, focal_ratio: float, conf: float, gt_labels: Path | None) -> None:
    """Predict on each image and write the rotation-strip render."""
    out.mkdir(parents=True, exist_ok=True)
    m = YOLO(model)
    for path in images:
        r = m.predict(path, conf=conf, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints.data) == 0:
            print(f"no detections: {path}")
            continue
        img = r.orig_img.copy()
        g = None
        if gt_labels is not None:
            g = gt_depths(gt_labels / f"{Path(path).stem}.txt", r.boxes.xyxy.cpu().numpy(), img.shape[:2])
        canvas = render(img, r.keypoints.data.cpu().numpy(), focal_ratio * max(img.shape[:2]), gt=g)
        dst = out / f"{Path(path).stem}.jpg"
        cv2.imwrite(str(dst), canvas)
        print(f"{dst}  {len(r.keypoints.data)} people")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--focal-ratio", type=float, default=1.0239, dest="focal_ratio")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument(
        "--gt-labels",
        type=Path,
        default=None,
        dest="gt_labels",
        help="pose3d label dir in the SAME convention as the model, to print true distances",
    )
    main(**vars(parser.parse_args()))
