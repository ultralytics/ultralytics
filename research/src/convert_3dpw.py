# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert 3DPW into pose3d evaluation labels.

3DPW is the real-ground-truth half of this project: everything measured against SAM 3D Body pseudo-labels is
teacher agreement, and only this tells us whether the student is actually right.

Two deliberate protocol choices, both forced by what 3DPW ships:

1. **12 joints, not the usual 14.** The GT here is `jointPositions`, SMPL-24 in world coordinates. SMPL-24 has
   no nose, eyes or ears, and the 14-joint LSP protocol used by the HMR literature needs a joint regressor
   applied to SMPL *vertices* — which needs the SMPL body model, a separately gated asset. So the evaluation
   uses the 12 joints SMPL-24 and COCO-17 genuinely share (shoulders, elbows, wrists, hips, knees, ankles).
   The five face joints are written with visibility 0, so `Pose3DValidator` excludes them from MPJPE on its own
   and no validator change is needed.
2. **Real intrinsics.** Each sequence ships `cam_intrinsics`, so `focal_ratio = fx / max(width, height)` is
   measured rather than assumed, removing the pseudo-focal caveat that applies to the COCO pseudo-labels.

Usage:
    python research/src/convert_3dpw.py --root /data/rick/datasets/3dpw --split test --out /home/rick/3dpw-pose3d
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from ultralytics.utils.pose3d import encode_z

# COCO-17 index -> SMPL-24 index, for the twelve joints the two skeletons share.
COCO_FROM_SMPL = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
SMPL_HIPS = (1, 2)  # their midpoint is the root, matching the mid-hip root used for the COCO pseudo-labels
BOX_MARGIN = 0.15  # box padding as a fraction of the joint-hull size; GT boxes are not shipped per frame


def camera_joints(joints_world: np.ndarray, cam_pose: np.ndarray) -> np.ndarray:
    """Transform (24, 3) world joints into camera coordinates with a (4, 4) world-to-camera matrix."""
    return joints_world @ cam_pose[:3, :3].T + cam_pose[:3, 3]


def frame_rows(j_cam: np.ndarray, k: np.ndarray, w: int, h: int, conv: float = 1.0) -> str | None:
    """Build one pose3d label row from a person's camera-space SMPL joints, or None if unusable.

    `conv` rescales the depths into the teacher's focal convention. The projection still uses the true depths,
    because where a joint lands on the sensor is a fact about the real camera and does not move.
    """
    root = j_cam[list(SMPL_HIPS)].mean(0)
    if not (0.3 < float(root[2]) < 40.0):  # behind the camera, or absurdly far: the sequence is mistracked
        return None

    uv = (j_cam / j_cam[:, 2:3]) @ k.T  # pinhole projection with the TRUE depths, (24, 3) -> pixels
    root_uv = ((root / root[2]) @ k.T)[:2]
    j_cam = j_cam * (1.0, 1.0, conv)  # only the stored depths move into the teacher's convention
    root = j_cam[list(SMPL_HIPS)].mean(0)
    z_root = float(root[2])
    kpts = np.zeros((18, 4), dtype=np.float64)
    for coco_i, smpl_i in COCO_FROM_SMPL.items():
        kpts[coco_i, :2] = uv[smpl_i, :2] / (w, h)
        kpts[coco_i, 2] = 2.0
        kpts[coco_i, 3] = j_cam[smpl_i, 2] - z_root
    kpts[17, :2] = root_uv / (w, h)
    kpts[17, 2] = 2.0
    kpts[17, 3] = z_root

    body = kpts[list(COCO_FROM_SMPL), :2]
    if not ((body > -0.5).all() and (body < 1.5).all()):  # person essentially out of frame
        return None
    lo, hi = body.min(0), body.max(0)
    pad = (hi - lo) * BOX_MARGIN
    lo, hi = np.clip(lo - pad, 0, 1), np.clip(hi + pad, 0, 1)
    cx, cy = (lo + hi) / 2
    bw, bh = hi - lo
    if bw <= 0 or bh <= 0:
        return None

    z_rel_enc, z_root_enc = encode_z(kpts[:17, 3], np.float64(z_root))
    kpts[:17, 3], kpts[17, 3] = z_rel_enc, z_root_enc
    kpts[:, :2] = kpts[:, :2].clip(0, 1)
    return " ".join(f"{v:.6g}" for v in [0.0, cx, cy, bw, bh, *kpts.reshape(-1)])


def main(root: Path, split: str, out: Path, stride: int, teacher_convention: bool, focal_ratio: float | None) -> None:
    """Write pose3d labels, an image list and the measured focal ratio for one 3DPW split."""
    seqs = sorted((root / "sequenceFiles" / split).glob("*.pkl"))
    if not seqs:
        raise SystemExit(f"no sequences under {root / 'sequenceFiles' / split}")
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    listing, ratios, n_rows, n_dropped = [], [], 0, 0
    conv_factors = []
    for pkl in seqs:
        with open(pkl, "rb") as fh:
            s = pickle.load(fh, encoding="latin1")
        name = str(s["sequence"])
        img_dir = root / "imageFiles" / name
        k = np.asarray(s["cam_intrinsics"], dtype=np.float64)
        cam_poses = np.asarray(s["cam_poses"], dtype=np.float64)
        people = [np.asarray(j, dtype=np.float64).reshape(-1, 24, 3) for j in s["jointPositions"]]
        valid = [np.asarray(v) for v in s["campose_valid"]]

        first = next(img_dir.glob("image_*.jpg"), None)
        if first is None:
            print(f"skip {name}: no images")
            continue
        import cv2

        h, w = cv2.imread(str(first)).shape[:2]
        # Normalize by the LONGER side, not the width: 3DPW mixes portrait and landscape sequences with the
        # same physical camera, and fx/width then splits into two clusters (1.03 vs 1.82) that are the same
        # camera seen twice. The validator scales its assumed focal by max(h, w) of the letterboxed image, so
        # fx/max(w, h) is the quantity that stays constant across both orientations.
        #
        # SAM 3D Body runs with no FOV estimator and defaults to focal = the image diagonal (measured
        # f/diagonal = 1.0000, sd 0.0000, across 18 image shapes), so every pseudo-label depth is expressed in
        # that convention and a model trained on them predicts depth-under-a-diagonal-focal, not metric depth.
        # By default this converts 3DPW's true metric GT into the same convention, which is the only way the
        # two are comparable. --true-metric writes the unconverted depths instead.
        diag = float(np.hypot(w, h))
        # focal_ratio None means the teacher's raw diagonal convention; a number means the declared
        # `ratio * max(w, h)` convention that refocal_labels.py re-keys the training labels onto.
        f_conv = diag if focal_ratio is None else focal_ratio * max(w, h)
        conv = f_conv / float(k[0, 0]) if teacher_convention else 1.0
        conv_factors.append(conv)
        ratios.append((f_conv if teacher_convention else float(k[0, 0])) / max(w, h))

        for f in range(0, len(cam_poses), stride):
            rows = []
            for person, ok in zip(people, valid):
                if f >= len(person) or not ok[f]:
                    continue
                row = frame_rows(camera_joints(person[f], cam_poses[f]), k, w, h, conv)
                if row:
                    rows.append(row)
                else:
                    n_dropped += 1
            img = img_dir / f"image_{f:05d}.jpg"
            if not rows or not img.exists():
                continue
            stem = f"{name}_{f:05d}"
            (out / "labels" / f"{stem}.txt").write_text("\n".join(rows) + "\n")
            link = out / "images" / f"{stem}.jpg"
            if not link.exists():
                link.symlink_to(img)
            listing.append(f"./images/{stem}.jpg")
            n_rows += len(rows)

    (out / f"{split}.txt").write_text("\n".join(listing) + "\n")
    ratios = np.array(ratios)
    print(f"{split}: {len(seqs)} sequences -> {len(listing)} frames, {n_rows} persons, {n_dropped} dropped")
    cf = np.array(conv_factors)
    print(f"focal_ratio for the yaml: mean {ratios.mean():.4f}  min {ratios.min():.4f}  max {ratios.max():.4f}")
    print(f"depth convention factor diag/f_true: mean {cf.mean():.4f}  min {cf.min():.4f}  max {cf.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/rick/datasets/3dpw"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=Path("/home/rick/3dpw-pose3d"))
    parser.add_argument(
        "--stride", type=int, default=5, help="keep every Nth frame; 3DPW is 30 fps and highly redundant"
    )
    parser.add_argument(
        "--true-metric",
        dest="teacher_convention",
        action="store_false",
        help="store true metric depth instead of the teacher's diagonal-focal convention",
    )
    parser.add_argument(
        "--focal-ratio",
        type=float,
        default=None,
        help="declared focal as a multiple of max(w, h); omit for the teacher's raw diagonal convention",
    )
    parser.set_defaults(teacher_convention=True)
    main(**vars(parser.parse_args()))
