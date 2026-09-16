# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Distil SAM 3D Body into pose3d labels.

Runs SAM 3D Body over each person box in a YOLO-format pose/detect dataset and writes pose3d label files
(18 keypoints x (x, y, visible, z), see research/design/pose3d-task-design.md). Boxes come from the existing
ground-truth labels, so detection supervision stays real and only the 3D is pseudo.

Requires the SAM 3D Body install and its gated checkpoints; the wrapper is reused from ~/snowboard. Point at it
with `PYTHONPATH=$HOME/snowboard` and the `SAM3D_CKPT` / `SAM3D_MHR_PATH` environment variables.

Usage:
    PYTHONPATH=$HOME/snowboard python research/src/pseudo_label_sam3d.py \
        --images /path/to/images --labels /path/to/labels --out /path/to/labels3d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ultralytics.utils.pose3d import encode_z

# MHR70 index for each COCO-17 joint, in COCO order. Taken from ~/snowboard/snowpose/mhr.py, which copies them
# from the SAM 3D Body source (`sam_3d_body/metadata/mhr70.py`). Every COCO joint exists in MHR70 exactly.
MHR_TO_COCO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
MHR_HIPS = (9, 10)  # left, right — their midpoint is the root


def person_to_label(out: dict, w: int, h: int, box: np.ndarray) -> str | None:
    """Convert one SAM 3D Body output dict into a pose3d label row.

    Args:
        out (dict): One person's estimator output, with `pred_keypoints_2d`, `pred_keypoints_3d` and `pred_cam_t`.
        w (int): Image width in pixels.
        h (int): Image height in pixels.
        box (np.ndarray): The person's normalized xywh box, reused verbatim as the detection target.

    Returns:
        (str | None): The label row, or None if the estimator produced nothing usable.
    """
    if out is None:
        return None
    kp2d = np.asarray(out["pred_keypoints_2d"], dtype=np.float32)
    kp3d = np.asarray(out["pred_keypoints_3d"], dtype=np.float32)
    cam_t = np.asarray(out["pred_cam_t"], dtype=np.float32)

    root2d = kp2d[list(MHR_HIPS)].mean(0)
    root3d = kp3d[list(MHR_HIPS)].mean(0)
    z_root = float(root3d[2] + cam_t[2])  # absolute depth of the root in metres
    z_rel = kp3d[MHR_TO_COCO, 2] - root3d[2]
    xy = kp2d[MHR_TO_COCO] / (w, h)
    root_xy = root2d / (w, h)

    # SAM 3D Body reconstructs occluded joints rather than flagging them, so visibility here means "inside the
    # frame", not "unoccluded". That is a deliberate difference from COCO and it is what lets the student learn
    # through occlusion; it also means OKS against COCO GT is not directly comparable.
    vis = np.where((xy > 0).all(1) & (xy < 1).all(1), 2.0, 0.0)

    z_rel_enc, z_root_enc = encode_z(z_rel, np.float32(z_root))
    row = [0.0, *box]
    for (x, y), v, z in zip(xy, vis, z_rel_enc):
        row += [x, y, v, z]
    row += [root_xy[0], root_xy[1], 2.0, float(z_root_enc)]
    return " ".join(f"{v:.6g}" for v in row)


def main(images: Path, labels: Path, out: Path, device: str) -> None:
    """Pseudo-label every image, writing one pose3d label file per image."""
    import cv2
    from snowpose.model import BodyModel  # ~/snowboard

    model = BodyModel(device=device)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for i, img_path in enumerate(files):
        lb_path = labels / f"{img_path.stem}.txt"
        if not lb_path.exists():
            continue
        h, w = cv2.imread(str(img_path)).shape[:2]
        rows = []
        for line in lb_path.read_text().strip().splitlines():
            v = np.array(line.split(), dtype=np.float32)
            if int(v[0]) != 0:  # person class only
                continue
            cx, cy, bw, bh = v[1:5]
            xyxy = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            row = person_to_label(model.infer(img_path, xyxy), w, h, v[1:5])
            if row:
                rows.append(row)
        if rows:
            (out / f"{img_path.stem}.txt").write_text("\n".join(rows) + "\n")
        if i % 100 == 0:
            print(f"{i}/{len(files)}", flush=True)
    print(f"wrote {len(list(out.glob('*.txt')))} label files to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="YOLO-format labels providing person boxes")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    main(**vars(parser.parse_args()))
