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
import time
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


def main(images: Path, labels: Path, out: Path, device: str, shard: int = 0, shards: int = 1) -> None:
    """Pseudo-label every image, writing one pose3d label file per image.

    All of an image's people go through the estimator in one call: it accepts an (N, 4) box array and the crop
    batch amortizes both the image decode and the backbone launch, which matters when the median COCO image has
    several people in it.
    """
    import cv2
    from snowpose.model import BodyModel  # ~/snowboard, or /data/rick/sam3d on ultra11

    model = BodyModel(device=device)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in labels.glob("*.txt"))[shard::shards]
    t0, done, npersons = time.time(), 0, 0
    for lb_path in files:
        img_path = images / f"{lb_path.stem}.jpg"
        dst = out / f"{lb_path.stem}.txt"
        if dst.exists() or not img_path.exists():
            continue
        boxes = []
        for line in lb_path.read_text().strip().splitlines():
            v = np.array(line.split(), dtype=np.float32)
            if int(v[0]) == 0:  # person class only
                boxes.append(v[1:5])
        if not boxes:
            continue
        h, w = cv2.imread(str(img_path)).shape[:2]
        cxcywh = np.stack(boxes)
        xyxy = np.stack(
            [
                (cxcywh[:, 0] - cxcywh[:, 2] / 2) * w,
                (cxcywh[:, 1] - cxcywh[:, 3] / 2) * h,
                (cxcywh[:, 0] + cxcywh[:, 2] / 2) * w,
                (cxcywh[:, 1] + cxcywh[:, 3] / 2) * h,
            ],
            axis=1,
        ).astype(np.float32)
        try:
            outs = model.est.process_one_image(str(img_path), bboxes=xyxy)
        except Exception as e:  # a single bad image must not end a multi-hour run
            print(f"skip {img_path.name}: {e}", flush=True)
            continue
        rows = [r for o, b in zip(outs, cxcywh) if (r := person_to_label(o, w, h, b))]
        if rows:
            dst.write_text("\n".join(rows) + "\n")
        done, npersons = done + 1, npersons + len(rows)
        if done % 200 == 0:
            el = time.time() - t0
            print(
                f"{done}/{len(files)} images, {npersons} persons, {el / done:.2f}s/img, "
                f"eta {(len(files) - done) * el / done / 3600:.1f}h",
                flush=True,
            )
    print(f"wrote {done} label files ({npersons} persons) to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="YOLO-format labels providing person boxes")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0, help="this worker's index, for splitting a run")
    parser.add_argument("--shards", type=int, default=1)
    main(**vars(parser.parse_args()))
