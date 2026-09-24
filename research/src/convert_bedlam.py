# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert BEDLAM into pose3d training labels.

BEDLAM is the scale half of the real ground truth: synthetic sequences with exact per-frame camera intrinsics
and extrinsics, which is what metric root depth needs and what no pseudo-label can supply. Two things it does
not ship shape this converter.

**No 3D joints.** `gtkps` is (127, 3), but its third channel is a constant 1.0 — pixels plus a confidence
column, verified on the data rather than read from the docs. BEDLAM's ground truth *is* the SMPL-X parameter
set (`pose_cam`, `shape`); every consumer poses the body model to obtain joints, and that model is separately
gated. So root-relative depth is unavailable and the seventeen COCO joints are written with **visibility 1** —
labelled in 2D, depth unknown — against `Pose3DLoss`, which supervises relative depth only where visibility is
2. Nothing here fabricates a relative-depth target; the stored value is the encoding of exactly zero.

**No root depth field.** `trans_cam` is not it: its z is median -0.07 m over [-4.47, 3.87], which is not a
distance from a camera. Root depth is the SMPL-X translation carried into camera coordinates with the shipped
extrinsic, `cam_ext @ trans_world`, which measures a sensible 4.70-15.29 m on the sequence it was checked
against. That point is the SMPL-X translation origin rather than the pelvis joint, which sits `J0(shape)` away
from it — reprojecting it lands 45-66 px from the mid-hip keypoint, the offset expected from ~0.25 m at 10 m
with fx 1312. The depth error that offset contributes is ~1-2%, well inside the error the depth head is
currently making, so the root's *depth* is taken from the translation while the root's *pixel* is taken from
the mid-hip keypoint, keeping the joint anatomically the same one COCO and 3DPW use.

The 127-joint layout was established structurally, not assumed: joint 0 sits 26 px from the midpoint of joints
1 and 2 against a 107 px control, joint 12 sits 15 px from the shoulder midpoint, joint 55 sits 14 px from the
eye midpoint, left and right limb lengths agree within 5%, and the hand blocks at 25-39 and 40-54 cluster on
joints 20 and 21 respectively. That is the standard SMPL-X joint set, so the names follow from its spec.

Usage:
    python research/src/convert_bedlam.py --npz-dir <root>/labels/all_npz_12_training \
        --images <root>/images --out /home/rick/bedlam-pose3d --focal-ratio 1.2
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics.utils.pose3d import Z_ROOT_MAX, encode_z

# SMPL-X 127-joint index for each COCO-17 joint, in COCO order.
SMPLX_FROM_COCO = [55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
SMPLX_HIPS = (1, 2)  # left and right hip; their midpoint is the root, as for COCO and 3DPW
BOX_MARGIN = 0.15  # box padding as a fraction of the joint-hull size; BEDLAM ships no per-frame boxes
MIN_VISIBLE = 6  # COCO joints that must be inside the frame for the person to be worth a row
Z_NEAR = 0.3  # metres; closer than this the person is behind or through the camera


def frame_rows(kps: np.ndarray, z_true: float, w: int, h: int, conv: float) -> str | None:
    """Build one pose3d label row from a person's 2D keypoints and true root depth, or None if unusable.

    Args:
        kps (np.ndarray): The person's (127, 2) SMPL-X keypoints in pixels.
        z_true (float): Root depth in metres under the sequence's own focal length.
        w (int): Image width in pixels.
        h (int): Image height in pixels.
        conv (float): Factor carrying the depth into the declared focal convention.

    Returns:
        (str | None): The label row, or None if the person is out of frame or out of encoding range.
    """
    z_root = z_true * conv
    if not (Z_NEAR < z_true and z_root < Z_ROOT_MAX):  # gate the converted depth: the cap is what gets stored
        return None

    norm = kps / (w, h)
    xy = norm[SMPLX_FROM_COCO]
    root_xy = norm[list(SMPLX_HIPS)].mean(0)

    # Visibility means "inside the frame", read before any clipping, exactly as the pseudo-labeller does it.
    # 1 rather than 2: the 2D is ground truth but the relative depth behind it is not available at all.
    inside = (xy > 0).all(1) & (xy < 1).all(1)
    if inside.sum() < MIN_VISIBLE:
        return None
    vis = np.where(inside, 1.0, 0.0)

    # Box from the hull of every joint that is actually in frame — hands and feet included, since they extend
    # the real extent — so a person half outside the image gets a box over the visible part rather than one
    # stretched to the edge by a single off-screen limb.
    pts = norm[(norm > 0).all(1) & (norm < 1).all(1)]
    lo, hi = pts.min(0), pts.max(0)
    pad = (hi - lo) * BOX_MARGIN
    lo, hi = np.clip(lo - pad, 0, 1), np.clip(hi + pad, 0, 1)
    bw, bh = hi - lo
    if bw <= 0 or bh <= 0:
        return None
    cx, cy = (lo + hi) / 2

    # Clip after visibility is read: one out-of-frame keypoint otherwise rejects the whole image in
    # verify_image_label, which is how half of coco-pose3d went missing.
    xy, root_xy = xy.clip(0.0, 1.0), root_xy.clip(0.0, 1.0)

    # Relative depth is unknown, so it is stored as the encoding of zero and masked off by visibility 1.
    z_rel_enc, z_root_enc = encode_z(np.zeros(17, dtype=np.float64), np.float64(z_root))
    row = [0.0, cx, cy, bw, bh]
    for (x, y), v, z in zip(xy, vis, z_rel_enc):
        row += [x, y, v, z]
    row += [root_xy[0], root_xy[1], 2.0, float(z_root_enc)]
    return " ".join(f"{v:.6g}" for v in row)


def convert_npz(npz: Path, images: Path, out: Path, focal_ratio: float, stride: int) -> tuple[int, int, int]:
    """Convert one sequence. Returns (frames written, persons written, persons dropped)."""
    seq = npz.stem
    img_root = images / seq / "png"
    if not img_root.is_dir():
        print(f"skip {seq}: no images at {img_root}")
        return 0, 0, 0

    d = np.load(npz, allow_pickle=True)
    names, kps, K, ext, tw = d["imgname"], d["gtkps"], d["cam_int"], d["cam_ext"], d["trans_world"]

    # Root depth: the SMPL-X translation carried into camera coordinates. See the module docstring for why this
    # and not trans_cam.
    hom = np.concatenate([tw, np.ones((len(tw), 1))], 1)
    z_true = np.einsum("nij,nj->ni", ext, hom)[:, 2]

    by_image = defaultdict(list)
    for i, name in enumerate(names):
        by_image[str(name)].append(i)

    n_frames, n_rows, n_dropped = 0, 0, 0
    for name in sorted(by_image)[::stride]:
        idx = by_image[name]
        img = img_root / name
        if not img.exists():
            continue
        # Width and height come from the principal point, so no image is decoded to find them.
        w, h = int(round(2 * K[idx[0]][0, 2])), int(round(2 * K[idx[0]][1, 2]))
        conv = focal_ratio * max(w, h) / float(K[idx[0]][0, 0])

        rows = []
        for i in idx:
            row = frame_rows(kps[i, :, :2].astype(np.float64), float(z_true[i]), w, h, conv)
            if row:
                rows.append(row)
            else:
                n_dropped += 1
        if not rows:
            continue

        stem = f"{seq}_{Path(name).stem}"
        (out / "labels" / f"{stem}.txt").write_text("\n".join(rows) + "\n")
        link = out / "images" / f"{stem}.png"
        if not link.exists():
            link.symlink_to(img.resolve())
        n_frames, n_rows = n_frames + 1, n_rows + len(rows)

    print(f"{seq}: {n_frames} frames, {n_rows} persons, {n_dropped} dropped")
    return n_frames, n_rows, n_dropped


def main(npz_dir: Path, images: Path, out: Path, focal_ratio: float, stride: int, include_agora: bool) -> None:
    """Convert every sequence and write the image list."""
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in npz_dir.glob("*.npz") if include_agora or not p.stem.startswith("agora"))
    if not files:
        raise SystemExit(f"no npz files under {npz_dir}")

    totals = [0, 0, 0]
    for npz in files:
        for i, v in enumerate(convert_npz(npz, images, out, focal_ratio, stride)):
            totals[i] += v

    listing = sorted(p.stem for p in (out / "labels").glob("*.txt"))
    (out / "train.txt").write_text("\n".join(f"./images/{s}.png" for s in listing) + "\n")
    print(f"\ntotal: {totals[0]} frames, {totals[1]} persons, {totals[2]} dropped, {len(listing)} listed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz-dir", type=Path, default=Path("/data/shared-datasets/bedlam-rick/labels/all_npz_12_training")
    )
    parser.add_argument("--images", type=Path, default=Path("/data/shared-datasets/bedlam-rick/images"))
    parser.add_argument("--out", type=Path, default=Path("/home/rick/bedlam-pose3d"))
    parser.add_argument("--focal-ratio", type=float, default=1.2, help="declared focal as a multiple of max(w, h)")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth frame; the 6fps sets are already thinned")
    parser.add_argument("--include-agora", action="store_true", help="also convert the two agora-*.npz files")
    main(**vars(parser.parse_args()))
