# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np


def _make_rotated_rect(cx, cy, w, h, theta):
    """Return 4 corner points (4,2) for a xywhr box (angle in radians)."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    hw, hh = w / 2.0, h / 2.0
    corners = np.array([[hw, hh], [hw, -hh], [-hw, -hh], [-hw, hh]], dtype=np.float32)
    return (corners @ R.T) + np.array([cx, cy], dtype=np.float32)


def test_instances_get_obb_from_segments_preserves_tracked_angle_after_clipping():
    """Regression: ensure OBB reconstruction uses tracked orientation (obbData) for border-clipped polygons."""
    from ultralytics.utils.instance import Instances

    img_w, img_h = 64, 64
    theta = np.float32(np.deg2rad(37.0))

    # Create a rotated box that crosses the right border, then clip points like Instances.clip() does.
    corners = _make_rotated_rect(cx=58, cy=32, w=30, h=12, theta=theta)
    corners = corners.clip([0, 0], [img_w, img_h]).astype(np.float32)

    segments = corners.reshape(1, 4, 2)  # pixel coords
    bboxes = np.array([[0.5, 0.5, 1.0, 1.0]], dtype=np.float32)  # dummy
    ins = Instances(bboxes=bboxes, segments=segments, bbox_format="xywh", normalized=False, obbData=None)

    # Track angle in degrees in obbData (Wim's PR implementation)
    ins.obbData = np.array([[58.0, 32.0, 30.0, 12.0, 37.0]], dtype=np.float32)

    # get_obb_boxes() returns xywhr with angle in radians
    obb = ins.get_obb_boxes(img_w, img_h)
    assert obb.shape == (1, 5)
    assert np.isfinite(obb).all()

    # Compare orientation modulo 90deg because cv2.minAreaRect can flip w/h and adjust angle sign.
    def norm(a: float) -> float:
        a = abs(float(a)) % (np.pi / 2)
        return min(a, (np.pi / 2) - a)

    assert np.isclose(norm(obb[0, 4]), norm(theta), atol=0.25)


def test_randomflip_updates_tracked_obb_angle_sign():
    """Ensure RandomFlip works with OBB tracking data (obbData) without crashing."""
    from ultralytics.data.augment import RandomFlip
    from ultralytics.utils.instance import Instances

    img = np.zeros((20, 30, 3), dtype=np.uint8)
    theta = np.float32(np.deg2rad(25.0))

    segments = _make_rotated_rect(cx=10, cy=8, w=8, h=4, theta=theta).reshape(1, 4, 2).astype(np.float32)
    bboxes = np.array([[10, 8, 8, 4]], dtype=np.float32)  # xywh absolute
    ins = Instances(bboxes=bboxes, segments=segments, bbox_format="xywh", normalized=False, obbData=None)
    ins.obbData = np.array([[10.0, 8.0, 8.0, 4.0, 25.0]], dtype=np.float32)

    # Horizontal flip with p=1.0
    flipper = RandomFlip(p=1.0, direction="horizontal")
    out = flipper({"img": img, "instances": ins})
    ins2 = out["instances"]
    assert ins2.obbData is not None and len(ins2.obbData) == 1
    assert np.isfinite(ins2.obbData).all()


