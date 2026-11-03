# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for tilted-camera support (calibration key ``R_cam_tilt``): projection, flip, centroid, alpha and IoU."""

import math

import numpy as np
import pytest

from ultralytics.data.stereo.box3d import Box3D
from ultralytics.data.stereo.calib import cam_tilt, level_ray_angle, load_kitti_calibration, tilt_center_offset
from ultralytics.models.yolo.s3d.augment import StereoHFlip, StereoLabels, project_3d_box_to_2d
from ultralytics.models.yolo.s3d.val import compute_3d_iou_batch, compute_bev_iou_batch
from ultralytics.utils.instance import Instances

H, W = 600, 960
FX, CX, CY, BASE = 742.0, 494.8, 312.2, 0.05
LOC, DIMS, RY = np.array([0.1, 0.2, 1.4]), np.array([1.2, 0.8, 0.135]), 0.4


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


R_TILT = _rx(math.radians(-23)) @ _rz(math.radians(4))  # camera pitched down 23 deg with a deliberately large roll


def _calib(R=R_TILT):
    c = {"fx": FX, "fy": FX, "cx": CX, "cy": CY, "baseline": BASE}
    if R is not None:
        c["R_cam_tilt"] = R
    return c


def _corners_px(loc, dims, ry, R, cam="left"):
    """Reference: tilted box corners (bottom-center origin, y down) projected with a pinhole."""
    length, w, h = dims
    obj = np.array(
        [[sx * length / 2, -h * t, sz * w / 2] for t in (0, 1) for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    )
    cam_pts = obj @ (R @ _ry(ry)).T + loc
    if cam == "right":
        cam_pts[:, 0] -= BASE
    return np.stack([FX * cam_pts[:, 0] / cam_pts[:, 2] + CX, FX * cam_pts[:, 1] / cam_pts[:, 2] + CY], 1)


def test_calib_parses_tilt_and_omits_it_when_absent(tmp_path):
    """R_cam_tilt is read row-major; a file without it yields a dict with no tilt key (level camera unchanged)."""
    p2 = f"P2: {FX} 0 {CX} 0 0 {FX} {CY} 0 0 0 1 0\nP3: {FX} 0 {CX} {-FX * BASE} 0 {FX} {CY} 0 0 0 1 0\n"
    (tmp_path / "a.txt").write_text(p2 + "R_cam_tilt: " + " ".join(f"{v:.9f}" for v in R_TILT.ravel()) + "\n")
    (tmp_path / "b.txt").write_text(p2)
    a, b = load_kitti_calibration(tmp_path / "a.txt").to_dict(), load_kitti_calibration(tmp_path / "b.txt").to_dict()
    assert np.allclose(a["R_cam_tilt"], R_TILT, atol=1e-8)
    assert "R_cam_tilt" not in b and cam_tilt(b) is None
    assert a["baseline"] == pytest.approx(BASE)
    (tmp_path / "c.txt").write_text(p2 + "R_cam_tilt: 2 0 0 0 1 0 0 0 1\n")  # a scale, not a rotation
    with pytest.raises(ValueError):
        load_kitti_calibration(tmp_path / "c.txt")


def test_tilted_projection_matches_tilted_corners():
    """project_3d_box_to_2d must place the box with R_tilt @ Ry, for both cameras."""
    for cam in ("left", "right"):
        uv = _corners_px(LOC, DIMS, RY, R_TILT, cam)
        ref = [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
        got = project_3d_box_to_2d(LOC, DIMS, RY, _calib(), camera=cam, clip=False)
        assert np.allclose(got, ref, atol=1e-3), cam
    level = project_3d_box_to_2d(LOC, DIMS, RY, _calib(None), clip=False)
    assert not np.allclose(level, project_3d_box_to_2d(LOC, DIMS, RY, _calib(), clip=False), atol=5.0)


def test_hflip_with_tilt_regenerates_the_mirror_of_the_old_right_box():
    """After a flip the new left view IS the mirrored old right view, so its regenerated box must be that mirror.

    Under roll this only holds when the tilt is mirrored too (S R S): keeping R leaves the regenerated boxes
    several pixels off the flipped pixels.
    """
    inst = Instances(
        bboxes=np.zeros((1, 4), dtype=np.float32),
        bbox_format="xywh",
        normalized=True,
        right_bboxes=np.zeros((1, 4), dtype=np.float32),
        dimensions_3d=DIMS[None].astype(np.float32),
        location_3d=LOC[None].astype(np.float32),
        rotation_y=np.array([RY], dtype=np.float32),
    )
    img = np.zeros((H, W, 6), dtype=np.uint8)
    labels = {"img": img, "instances": inst, "calibration": _calib(), "cls": np.array([0])}
    StereoLabels.from_labels(labels).regenerate_2d_bboxes_from_3d((W, H)).to_labels(labels)
    rb = labels["instances"].right_bboxes[0] * [W, H, W, H]
    old_right = np.array([rb[0] - rb[2] / 2, rb[1] - rb[3] / 2, rb[0] + rb[2] / 2, rb[1] + rb[3] / 2])

    lab = StereoHFlip(p=1.0)(labels)
    lb = np.asarray(lab["instances"].bboxes[0]) * [W, H, W, H]
    new_left = np.array([lb[0] - lb[2] / 2, lb[1] - lb[3] / 2, lb[0] + lb[2] / 2, lb[1] + lb[3] / 2])
    mirrored = np.array([(W - 1) - old_right[2], old_right[1], (W - 1) - old_right[0], old_right[3]])
    assert np.allclose(new_left, mirrored, atol=0.05)
    assert np.allclose(lab["calibration"]["R_cam_tilt"], np.diag([-1, 1, 1]) @ R_TILT @ np.diag([-1, 1, 1]))


def test_box3d_center_roundtrip_with_tilt():
    """from_label lifts the bottom-center by h/2 along the tilted vertical; project_to_2d must undo exactly that."""
    label = {
        "class_id": 0,
        "left_box": {"center_x": 0.5, "center_y": 0.5, "width": 0.1, "height": 0.1},
        "dimensions": {"length": DIMS[0], "width": DIMS[1], "height": DIMS[2]},
        "location_3d": {"x": LOC[0], "y": LOC[1], "z": LOC[2]},
        "rotation_y": RY,
    }
    box = Box3D.from_label(label, _calib())
    assert np.allclose(box.center_3d, LOC + R_TILT @ [0, -DIMS[2] / 2, 0])
    assert np.allclose(box.project_to_2d(_calib()), project_3d_box_to_2d(LOC, DIMS, RY, _calib(), clip=False))
    level = Box3D.from_label(label, _calib(None))
    assert level.center_3d == (LOC[0], LOC[1] - DIMS[2] / 2, LOC[2]), "level camera must keep the KITTI arithmetic"


def test_level_ray_angle():
    """Level camera: exactly atan2(x, z). Tilted: bottom-center and centroid share one ray (they differ only in level y)."""
    assert level_ray_angle(None, 0.3, 2.0, 0.7) == math.atan2(0.3, 2.0)
    center = LOC + tilt_center_offset(R_TILT, DIMS[2])
    a = level_ray_angle(R_TILT, LOC[0], LOC[2], LOC[1])
    assert a == pytest.approx(level_ray_angle(R_TILT, center[0], center[2], center[1]), abs=1e-12)
    p = R_TILT.T @ LOC
    assert a == pytest.approx(math.atan2(p[0], p[2]))


def test_iou_is_computed_in_the_level_frame():
    """Two boxes upright in the level frame, seen by a tilted camera, must score their level-frame IoU."""
    lvl = [(np.array([0.0, 0.3, 2.0]), 0.2), (np.array([0.15, 0.32, 2.1]), 0.35)]
    dims = (1.2, 0.8, 0.135)
    ref = [
        Box3D(center_3d=tuple(c), dimensions=dims, orientation=r, class_label="p", class_id=0, confidence=1.0)
        for c, r in lvl
    ]
    cam = [
        Box3D(center_3d=tuple(R_TILT @ c), dimensions=dims, orientation=r, class_label="p", class_id=0, confidence=1.0)
        for c, r in lvl
    ]
    want, want_bev = compute_3d_iou_batch(ref[:1], ref[1:]), compute_bev_iou_batch(ref[:1], ref[1:])
    assert 0.05 < want[0, 0] < 0.95
    assert np.allclose(compute_3d_iou_batch(cam[:1], cam[1:], R_tilt=R_TILT), want)
    assert np.allclose(compute_bev_iou_batch(cam[:1], cam[1:], R_tilt=R_TILT), want_bev)
    assert not np.allclose(compute_3d_iou_batch(cam[:1], cam[1:]), want, atol=1e-3), "camera-frame IoU should differ"
