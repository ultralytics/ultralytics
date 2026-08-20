# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the stereo augmentation transforms: StereoZoom (scale jitter) and StereoHFlip (mirror + view swap)."""

import numpy as np
import pytest

from ultralytics.models.yolo.s3d.augment import StereoHFlip, StereoLabels, StereoZoom
from ultralytics.utils.instance import Instances

H, W = 96, 160
FX, CX, CY, BASE = 100.0, 80.0, 48.0, 0.2
Z = 4.0


def _labels():
    """Stereo label dict with one 0.6 m cube at z=4 m; 2D boxes canonicalized from 3D projection."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (H, W, 6), dtype=np.uint8)
    inst = Instances(
        bboxes=np.array([[0.5, 0.5, 0.2, 0.3]], dtype=np.float32),
        bbox_format="xywh",
        normalized=True,
        right_bboxes=np.array([[0.48, 0.5, 0.2, 0.3]], dtype=np.float32),
        dimensions_3d=np.array([[0.6, 0.6, 0.6]], dtype=np.float32),
        location_3d=np.array([[0.0, 0.3, Z]], dtype=np.float32),
        rotation_y=np.array([0.0], dtype=np.float32),
    )
    calib = {"fx": FX, "fy": FX, "cx": CX, "cy": CY, "baseline": BASE, "width": W, "height": H}
    labels = {"img": img, "instances": inst, "calibration": calib, "cls": np.array([0])}
    StereoLabels.from_labels(labels).regenerate_2d_bboxes_from_3d((W, H)).to_labels(labels)
    return labels


def test_zoom_in_scales_size_and_disparity_not_depth():
    """s=1.25 zoom-in: apparent size and normalized disparity scale by s; 3D depth target unchanged."""
    lab0 = _labels()
    h0 = float(lab0["instances"].bboxes[0][3])
    d0 = float(lab0["instances"].bboxes[0][0] - lab0["instances"].right_bboxes[0][0])

    lab = StereoZoom(scale_range=(1.25, 1.25), p=1.0)(_labels())
    assert lab["img"].shape == (H, W, 6)
    h1 = float(lab["instances"].bboxes[0][3])
    d1 = float(lab["instances"].bboxes[0][0] - lab["instances"].right_bboxes[0][0])
    assert h1 / h0 == pytest.approx(1.25, rel=0.05)
    assert d1 / d0 == pytest.approx(1.25, rel=0.05)
    assert float(lab["instances"].location_3d[0, 2]) == pytest.approx(Z)  # depth target untouched
    assert lab["calibration"]["fx"] == pytest.approx(FX * 1.25, rel=0.02)
    # Geometric invariant the loss targets rely on: box-center disparity_px == fx' * baseline / z_near
    # (for a front-facing cube the projected 2D box extremes come from the NEAR face, z_near = Z - depth/2)
    assert d1 * W == pytest.approx(lab["calibration"]["fx"] * BASE / (Z - 0.3), rel=1e-3)


def test_zoom_out_pads_and_shrinks():
    """s=0.8 zoom-out: size/disparity shrink by s; canvas padded with letterbox gray (114)."""
    lab0 = _labels()
    h0 = float(lab0["instances"].bboxes[0][3])
    lab = StereoZoom(scale_range=(0.8, 0.8), p=1.0)(_labels())
    assert lab["img"].shape == (H, W, 6)
    assert float(lab["instances"].bboxes[0][3]) / h0 == pytest.approx(0.8, rel=0.05)
    assert lab["calibration"]["fx"] == pytest.approx(FX * 0.8, rel=0.02)
    corner_pad = np.all(lab["img"][0, 0] == 114) or np.all(lab["img"][-1, -1] == 114)
    assert corner_pad  # some border region must be padding


def test_zoom_identity_when_range_is_one():
    """scale_range (1,1) or p=0 → labels pass through untouched."""
    lab0 = _labels()
    lab = StereoZoom(scale_range=(1.0, 1.0), p=1.0)(_labels())
    assert np.array_equal(lab["img"], lab0["img"])
    assert np.allclose(lab["instances"].bboxes, lab0["instances"].bboxes)
    lab = StereoZoom(scale_range=(0.5, 1.5), p=0.0)(_labels())
    assert np.array_equal(lab["img"], lab0["img"])


def test_zoom_keeps_object_in_view():
    """Strong zoom-in over many draws: the crop is biased so the object's projection stays in frame."""
    np.random.seed(3)
    zoom = StereoZoom(scale_range=(1.6, 1.6), p=1.0)
    for _ in range(20):
        lab = zoom(_labels())
        cx = float(lab["instances"].bboxes[0][0])
        bw = float(lab["instances"].bboxes[0][2])
        assert 0.0 < cx < 1.0 and bw > 0.0


def test_scale_jitter_is_live_in_the_dataset_pipeline():
    """End-to-end gate: with `scale_jitter>0` the image AND the 2D targets must change, calib must scale with the zoom,
    and the 3D truth must NOT move.

    This gate exists because the previous scale augmentation, `StereoScale`, was geometrically INERT — its canvas resize
    was fully undone by the subsequent `StereoLetterBox`, so no scale jitter ever actually ran in any s3d training. A
    unit test of the transform in isolation would not have caught that; only driving the real dataset pipeline does.

    Note which field to assert on: the sample's `labels` entry holds the raw per-object 3D truth and correctly does NOT
    change under zoom. The trainable 2D targets are `bboxes`. Checking `labels` makes a working augmentation look inert.
    """
    from pathlib import Path

    import numpy as np
    import torch

    from ultralytics.models.yolo.s3d.dataset import Stereo3DDetDataset

    root = Path("/home/rick/datasets/kitti-chen-small")
    if not root.exists():
        pytest.skip("screening dataset not present on this machine")

    names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    mean_dims = {"Car": [3.9, 1.6, 1.5], "Pedestrian": [0.8, 0.6, 1.7], "Cyclist": [1.8, 0.6, 1.7]}
    std_dims = {"Car": [0.42, 0.10, 0.15], "Pedestrian": [0.20, 0.08, 0.12], "Cyclist": [0.25, 0.10, 0.15]}

    class Hyp(dict):
        __getattr__ = dict.get

    def sample(jit, idx=3):
        torch.manual_seed(0)
        np.random.seed(0)
        hyp = Hyp({"fliplr": 0.0, "crop_fraction": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "scale_jitter": jit})
        ds = Stereo3DDetDataset(
            root=str(root),
            split="val",
            imgsz=(384, 1248),
            names=names,
            mean_dims=mean_dims,
            std_dims=std_dims,
            augment=True,
            hyp=hyp,
        )
        return ds, ds[idx]

    ds0, a = sample(0.0)
    ds1, b = sample(0.4)

    assert "StereoZoom" not in [type(t).__name__ for t in ds0.transforms.transforms]
    assert "StereoZoom" in [type(t).__name__ for t in ds1.transforms.transforms]

    assert not torch.equal(a["img"].float(), b["img"].float()), "zoom must change the image"
    assert not torch.equal(a["bboxes"].float(), b["bboxes"].float()), "zoom must move the 2D targets"
    assert torch.equal(a["cls"], b["cls"]), "zoom must not change class labels"

    s = b["calibration"]["fx"] / a["calibration"]["fx"]
    assert s > 1.01 or s < 0.99, f"calib fx must scale with the zoom, got ratio {s}"
    assert np.allclose(np.asarray(a["location_3d"]), np.asarray(b["location_3d"])), (
        "zoom must NOT move objects in 3D — only their apparent size and the calibration change"
    )


def test_hflip_swaps_the_two_views():
    """The mirrored scene puts the right camera where the left was, so the views must SWAP, not just flip.

    Flipping each half in place without swapping leaves the rig handedness reversed: the left image would see the scene
    from the right camera's viewpoint, making every disparity negative while the 3D depth targets stay positive.
    Training absorbs that silently -- all losses stay finite.
    """
    lab0 = _labels()
    left0, right0 = lab0["img"][:, :, :3].copy(), lab0["img"][:, :, 3:6].copy()

    lab = StereoHFlip(p=1.0)(_labels())

    assert np.array_equal(lab["img"][:, :, :3], right0[:, ::-1]), "new left view must be the flipped OLD RIGHT"
    assert np.array_equal(lab["img"][:, :, 3:6], left0[:, ::-1]), "new right view must be the flipped OLD LEFT"


def test_hflip_mirrors_geometry_and_keeps_disparity_physical():
    """Yaw negates, X mirrors about the baseline, depth is untouched, and disparity stays fx*B/z.

    Disparity is the invariant that ties the flipped labels back to the flipped pixels. A sign error anywhere in the
    mirror leaves it negative or inconsistent with depth, which is exactly the class of corruption that runs for 600
    epochs without a single failed assertion anywhere else.
    """
    lab = StereoHFlip(p=1.0)(_labels())
    inst = lab["instances"]

    assert float(inst.location_3d[0, 0]) == pytest.approx(BASE - 0.0)  # X -> baseline - X
    assert float(inst.location_3d[0, 2]) == pytest.approx(Z), "flipping must not touch the depth target"
    assert lab["calibration"]["cx"] == pytest.approx((W - 1) - CX)
    assert lab["calibration"]["fx"] == pytest.approx(FX), "a mirror does not change focal length"

    d_px = float(inst.bboxes[0][0] - inst.right_bboxes[0][0]) * W
    assert d_px > 0, "left-minus-right disparity must stay POSITIVE; negative means the views were not swapped"
    assert d_px == pytest.approx(FX * BASE / (Z - 0.3), rel=1e-3)  # near face, as in the zoom tests


def test_hflip_ties_the_label_mirror_to_the_pixel_flip():
    """The new left box center must land where the OLD RIGHT box center lands after mirroring the canvas.

    This is the cross-check between the two halves of the transform: labels are mirrored analytically while pixels are
    mirrored by cv2. `X -> -X` (forgetting the baseline term) still yields positive, depth-consistent disparity and
    would pass the test above -- it only breaks here.
    """
    lab0 = _labels()
    u_right_old_px = float(lab0["instances"].right_bboxes[0][0]) * W

    lab = StereoHFlip(p=1.0)(_labels())
    u_left_new_px = float(lab["instances"].bboxes[0][0]) * W

    assert u_left_new_px == pytest.approx((W - 1) - u_right_old_px, abs=0.5)


def test_hflip_twice_is_identity():
    """Two flips restore the original geometry -- an involution check no single-direction test can give."""
    lab0 = _labels()
    lab2 = StereoHFlip(p=1.0)(StereoHFlip(p=1.0)(_labels()))

    assert np.array_equal(lab2["img"], lab0["img"])
    for field in ("bboxes", "right_bboxes", "location_3d", "rotation_y"):
        assert np.allclose(getattr(lab2["instances"], field), getattr(lab0["instances"], field), atol=1e-5), field
    assert lab2["calibration"]["cx"] == pytest.approx(CX)


def test_hflip_p_zero_is_a_noop():
    """P=0 must leave pixels, labels and calibration byte-for-byte alone."""
    lab0 = _labels()
    lab = StereoHFlip(p=0.0)(_labels())
    assert np.array_equal(lab["img"], lab0["img"])
    assert np.array_equal(lab["instances"].location_3d, lab0["instances"].location_3d)
    assert lab["calibration"]["cx"] == pytest.approx(CX)


def test_hflip_negates_a_non_zero_yaw():
    """Mirroring reverses handedness, so yaw must negate.

    Kept separate and given a NON-ZERO angle deliberately. The shared fixture's yaw is 0.0, and 0.0 negates to 0.0 -- a
    transform that dropped the negation entirely passed the geometry test above until this case was added (confirmed by
    mutating the negation out). A rotated box also moves its near face, which is why this cannot simply be folded into
    the disparity assertion.
    """
    src = _labels()
    src["instances"].rotation_y = np.array([0.6], dtype=np.float32)

    lab = StereoHFlip(p=1.0)(src)

    assert float(lab["instances"].rotation_y[0]) == pytest.approx(-0.6, abs=1e-5)
