# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for Stereo 3D Detection (s3d) task."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ultralytics import YOLO
from ultralytics.data.stereo.box3d import Box3D
from ultralytics.utils.metrics import compute_3d_iou, compute_bev_iou

MODEL = "yolo26n-s3d.yaml"
DATA = "kitti-stereo8.yaml"


def test_train():
    """Test s3d training for 2 epochs on mini dataset."""
    model = YOLO(MODEL)
    model.train(data=DATA, epochs=2, imgsz=[384, 1248], batch=2, val=False)


def test_val():
    """Test s3d validation on mini dataset."""
    model = YOLO(MODEL)
    model.val(data=DATA, imgsz=[384, 1248], batch=2)


def test_predict(tmp_path):
    """Test s3d prediction on synthetic stereo pair."""
    import cv2

    left_img = tmp_path / "left.png"
    right_img = tmp_path / "right.png"
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    cv2.imwrite(str(left_img), img)
    cv2.imwrite(str(right_img), img)

    model = YOLO(MODEL)
    results = model.predict(source=[(str(left_img), str(right_img))], imgsz=[384, 1248])
    assert len(results) >= 0


def test_export_onnx():
    """Test ONNX export for s3d model with two stereo inputs and full 3D output."""
    import onnx

    model = YOLO(MODEL)
    path = model.export(format="onnx", imgsz=[384, 1248])
    assert path.endswith(".onnx")

    # Verify two-input stereo model
    m = onnx.load(path)
    inputs = {inp.name for inp in m.graph.input}
    assert inputs == {"left_img", "right_img"}, f"Expected stereo inputs, got {inputs}"

    # Both inputs should be [1, 3, 384, 1248]
    for inp in m.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        assert dims == [1, 3, 384, 1248], f"{inp.name} shape {dims} != [1, 3, 384, 1248]"

    # Output aux channels: 4(box) + 3(cls) + 1(lr) + 3(dims) + 6(orient MultiBin) + n_bins(depth).
    # Derived from the model, not hardcoded: the depth branch emits one logit per decode bin, so the export
    # width tracks depth_bins. It was 33 when 16 bins were the default and is 81 at the shipped 64.
    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS

    n_bins = model.model.model[-1].depth_dfl.n_bins
    expected = 7 + 1 + 3 + ORIENT_CHANNELS + n_bins
    out_shape = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    assert out_shape[1] == expected, (
        f"Expected {expected} output channels (7 det + 10 aux + {n_bins} depth bins), got {out_shape[1]}"
    )


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="TensorRT requires CUDA")
def test_export_engine():
    """Test TensorRT engine export for s3d model."""
    model = YOLO(MODEL)
    path = model.export(format="engine", imgsz=[384, 1248])
    assert path.endswith(".engine")


def test_head_localization_branches_unconditional():
    """The promoted head always builds the projected-center (2ch) and uncertainty (2ch lr) branches."""
    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS

    head = YOLO(MODEL).model.model[-1]
    assert head.aux_specs["proj_offset"] == 2
    assert head.aux["proj_offset"][0][-1].out_channels == 2  # (Δu, Δv)
    assert head.aux["lr_distance"][0][-1].out_channels == 2  # value + log-variance
    assert head.aux_specs["orientation"] == ORIENT_CHANNELS  # unchanged


def test_proj_center_loss_present():
    """The proj_center smooth-L1 term is always computed when proj_offset targets/preds are present."""
    import torch

    from ultralytics import YOLO
    from ultralytics.models.yolo.s3d.loss import Stereo3DDetLoss

    model = YOLO("yolo26n-s3d.yaml").model
    crit = Stereo3DDetLoss(model, loss_weights={"proj_center": 1.0})
    B, HW = 2, 5
    preds = {"proj_offset": torch.zeros(B, 2, HW)}
    gt = torch.ones(B, 3, 2)  # [B, max_n, 2]
    idx = torch.zeros(B, HW, dtype=torch.long)
    fg = torch.ones(B, HW, dtype=torch.bool)
    losses = crit._compute_aux_losses(preds, {"aux_targets": {"proj_offset": gt}}, idx, fg)
    assert "proj_center" in losses and losses["proj_center"] > 0


def test_lr_nll_attenuates_with_uncertainty():
    """Laplacian NLL: for a fixed residual, a larger predicted log-variance lowers the loss (attenuation), but the
    log-variance penalty prevents collapse — loss is convex in logvar.
    """
    import torch

    from ultralytics.models.yolo.s3d.loss import laplacian_nll

    pred = torch.tensor([1.0])
    tgt = torch.tensor([3.0])  # residual 2.0 (must exceed 1.0: exp(-logvar) starts at 1, so attenuation
    # only lowers the loss below the logvar=0 baseline when the residual is large enough to dominate it)
    low = laplacian_nll(pred, tgt, logvar=torch.tensor([0.0]))
    mid = laplacian_nll(pred, tgt, logvar=torch.tensor([1.0]))
    high = laplacian_nll(pred, tgt, logvar=torch.tensor([5.0]))
    assert mid < low, "some uncertainty should reduce loss vs zero-variance for a nonzero residual"
    assert high > mid, "excessive uncertainty is penalized by the logvar term"


def test_lr_nll_loss_wired():
    """Integration test: when lr_logvar is present, _compute_aux_losses routes lr_distance through the Laplacian-NLL
    gather-wiring in _lr_nll_loss (the promoted, always-on path), not smooth-L1.
    """
    import math

    import torch

    from ultralytics import YOLO
    from ultralytics.models.yolo.s3d.loss import Stereo3DDetLoss

    model = YOLO("yolo26n-s3d.yaml").model
    crit = Stereo3DDetLoss(model, loss_weights={"lr_distance": 1.0})

    B, HW = 2, 5
    val = torch.ones(B, 1, HW)  # lr_distance prediction
    logvar = torch.full((B, 1, HW), 5.0)  # large predicted log-variance (attenuation)
    preds = {"lr_distance": val, "lr_logvar": logvar}
    gt = torch.full((B, 3, 1), 3.0)  # [B, max_n, 1] — residual of 2.0 for every anchor
    idx = torch.zeros(B, HW, dtype=torch.long)
    fg = torch.ones(B, HW, dtype=torch.bool)

    nll_loss = crit._compute_aux_losses(preds, {"aux_targets": {"lr_distance": gt}}, idx, fg)["lr_distance"]
    assert torch.isfinite(nll_loss) and nll_loss >= 0

    # It must equal the Laplacian NLL (|2|*exp(-5)+5 ≈ 5.013), NOT the plain smooth-L1 value (1.5) —
    # proving lr_distance went through the NLL branch, not the generic _aux_loss.
    expected_nll = 2.0 * math.exp(-5.0) + 5.0
    assert torch.isclose(nll_loss, torch.tensor(expected_nll), atol=1e-3), f"{nll_loss} != NLL {expected_nll}"
    assert not torch.isclose(nll_loss, torch.tensor(1.5), atol=1e-2), "collapsed to smooth-L1 — NLL routing broken"


def test_3d_iou():
    """Test 3D IoU computation: identical, no overlap, and partial overlap."""
    box = Box3D(
        center_3d=(10.0, 2.0, 30.0),
        dimensions=(3.88, 1.63, 1.53),
        orientation=0.0,
        class_label="Car",
        class_id=0,
        confidence=0.95,
    )
    assert abs(compute_3d_iou(box, box) - 1.0) < 1e-6

    far_box = Box3D(
        center_3d=(100.0, 2.0, 30.0),
        dimensions=(3.88, 1.63, 1.53),
        orientation=0.0,
        class_label="Car",
        class_id=0,
        confidence=0.95,
    )
    assert compute_3d_iou(box, far_box) == 0.0

    near_box = Box3D(
        center_3d=(11.0, 2.0, 30.0),
        dimensions=(4.0, 2.0, 2.0),
        orientation=0.0,
        class_label="Car",
        class_id=0,
        confidence=0.95,
    )
    box2 = Box3D(
        center_3d=(10.0, 2.0, 30.0),
        dimensions=(4.0, 2.0, 2.0),
        orientation=0.0,
        class_label="Car",
        class_id=0,
        confidence=0.95,
    )
    assert 0.0 < compute_3d_iou(box2, near_box) < 1.0


def test_3d_iou_rotated_45deg():
    """Two identical square-footprint boxes offset by 45 deg of yaw.

    True 3D IoU of a unit square and the same square rotated 45 deg (shared center/dims) is exactly 1/sqrt(2) ~= 0.7071.
    The old axis-aligned-bbox approximation returns ~1.0 here because the 45 deg box's AABB fully contains the other
    box, so this case discriminates true rotated IoU from the AABB hack.
    """
    a = Box3D(
        center_3d=(0.0, 0.0, 10.0),
        dimensions=(2.0, 2.0, 2.0),
        orientation=0.0,
        class_label="Car",
        class_id=0,
        confidence=0.95,
    )
    b = Box3D(
        center_3d=(0.0, 0.0, 10.0),
        dimensions=(2.0, 2.0, 2.0),
        orientation=np.pi / 4,
        class_label="Car",
        class_id=0,
        confidence=0.95,
    )
    assert abs(compute_3d_iou(a, b) - (1.0 / np.sqrt(2))) < 1e-3


def test_3d_iou_rotated_no_overlap():
    """Two 45 deg boxes whose AABBs overlap but whose true rotated footprints do not.

    Both boxes are unit-ish square footprints rotated 45 deg (diamonds), offset diagonally by (2, 2) in the x-z plane.
    The diamonds are disjoint (L1 center distance 4 > 2*sqrt(2)), so true 3D IoU is 0. But their axis-aligned bounding
    boxes (side 2*sqrt(2)) still overlap, so the old AABB approximation reports a spurious positive IoU.
    """
    a = Box3D(
        center_3d=(0.0, 0.0, 10.0),
        dimensions=(2.0, 2.0, 2.0),
        orientation=np.pi / 4,
        class_label="Car",
        class_id=0,
        confidence=0.95,
    )
    b = Box3D(
        center_3d=(2.0, 0.0, 12.0),
        dimensions=(2.0, 2.0, 2.0),
        orientation=np.pi / 4,
        class_label="Car",
        class_id=0,
        confidence=0.95,
    )
    assert compute_3d_iou(a, b) == 0.0


def test_bev_corners_uses_kitti_axis_convention():
    """At rotation_y=0 the object's LENGTH must lie along camera x and its WIDTH along camera z.

    This is the KITTI devkit convention (`compute_box_3d` builds x_corners from l and z_corners from w), and it is what
    `s3d/augment.py`, `s3d/geometric.py` and `utils/plotting.py` already use. `_bev_corners` previously had the two axes
    swapped, which every other IoU test here passes unchanged because they are all self-consistent: both boxes get the
    same wrong footprint, so identical-box, 45-degree, disjoint and stacked cases are all unaffected. Only tying the
    footprint to a named axis catches it.
    """
    from ultralytics.utils.metrics import _bev_corners

    extent = lambda v: float(v.max() - v.min())  # noqa: E731

    corners = _bev_corners(0.0, 0.0, length=4.0, width=2.0, rot=0.0)
    assert abs(extent(corners[:, 0]) - 4.0) < 1e-6, "length must span camera x at rot=0"
    assert abs(extent(corners[:, 1]) - 2.0) < 1e-6, "width must span camera z at rot=0"

    # A car driving away from the camera (the common KITTI case) has ry ~= +/-pi/2 and must present its LENGTH to depth.
    corners = _bev_corners(0.0, 0.0, length=4.0, width=2.0, rot=np.pi / 2)
    assert abs(extent(corners[:, 0]) - 2.0) < 1e-6
    assert abs(extent(corners[:, 1]) - 4.0) < 1e-6, "length must span camera z (depth) at rot=pi/2"


def test_3d_iou_depth_tolerance_matches_kitti_devkit():
    """A car displaced purely in depth must lose IoU against its LENGTH, not its width.

    For pure longitudinal translation the exact 3D IoU is (D - dz) / (D + dz) where D is the footprint's depth extent, so
    IoU=0.7 is reached at dz = 0.176*D. With the axes swapped the tolerance came out 2.36x too tight (0.289 m instead of
    0.683 m for a 3.87 m car), making AP3D@0.7 far harsher than the published KITTI benchmark. This pins the tolerance.
    """
    length, width, height = 3.87, 1.64, 1.53
    z, ry = 20.0, np.pi / 2  # pointing along depth: the length faces the camera axis
    tol = length * 0.3 / 1.7  # dz at which IoU should be exactly 0.7

    at_tol = compute_3d_iou([0, 0, z, length, width, height, ry], [0, 0, z + tol, length, width, height, ry])
    assert abs(at_tol - 0.7) < 1e-3, f"depth tolerance for IoU 0.7 should be {tol:.3f} m, got IoU {at_tol:.3f}"

    # Sanity: the same displacement applied laterally must cost MORE, because width is the short axis.
    lateral = compute_3d_iou([0, 0, z, length, width, height, ry], [tol, 0, z, length, width, height, ry])
    assert lateral < at_tol, "lateral error should cost more IoU than the same depth error for a depth-facing car"


def test_3d_iou_rotated_90deg():
    """Two identical (L=4, W=2) boxes offset by 90 deg of yaw, shared center/dims.

    The rot-0 footprint is 2x4 and the rot-90 footprint is 4x2; their BEV intersection is the 2x2 square, giving true 3D
    IoU = 4 / (16 - 4) = 1/3. The old AABB approximation returns 1.0 here (the 90 deg box's AABB grows), so this is a
    clean analytic regression guard.
    """
    a = Box3D(
        center_3d=(0.0, 0.0, 20.0),
        dimensions=(4.0, 2.0, 2.0),
        orientation=0.0,
        class_label="Car",
        class_id=0,
        confidence=0.95,
    )
    b = Box3D(
        center_3d=(0.0, 0.0, 20.0),
        dimensions=(4.0, 2.0, 2.0),
        orientation=np.pi / 2,
        class_label="Car",
        class_id=0,
        confidence=0.95,
    )
    assert abs(compute_3d_iou(a, b) - 1.0 / 3.0) < 1e-3


def test_bev_iou_ignores_height():
    """BEV IoU uses only the ground-plane footprint; height offset must not reduce it.

    Two boxes with an identical footprint but stacked vertically (full height
    offset) share no 3D volume (3D IoU ~= 0) yet have identical bird's-eye-view
    footprints (BEV IoU == 1.0). This pins BEV down as a distinct metric.
    """
    low = Box3D(
        center_3d=(0.0, 0.0, 15.0),
        dimensions=(4.0, 1.8, 1.6),
        orientation=0.3,
        class_label="Car",
        class_id=0,
        confidence=0.9,
    )
    high = Box3D(
        center_3d=(0.0, -1.6, 15.0),
        dimensions=(4.0, 1.8, 1.6),
        orientation=0.3,
        class_label="Car",
        class_id=0,
        confidence=0.9,
    )
    assert abs(compute_bev_iou(low, high) - 1.0) < 1e-6
    assert compute_3d_iou(low, high) < 0.05  # stacked: ~no vertical overlap

    # BEV of the 45deg square case equals its 3D IoU (heights coincide): 1/sqrt(2).
    a = Box3D(
        center_3d=(0.0, 0.0, 10.0),
        dimensions=(2.0, 2.0, 2.0),
        orientation=0.0,
        class_label="Car",
        class_id=0,
        confidence=0.9,
    )
    b = Box3D(
        center_3d=(0.0, 0.0, 10.0),
        dimensions=(2.0, 2.0, 2.0),
        orientation=np.pi / 4,
        class_label="Car",
        class_id=0,
        confidence=0.9,
    )
    assert abs(compute_bev_iou(a, b) - 1.0 / np.sqrt(2)) < 1e-3


def _single_image_stat(gt_ori, pred_ori):
    """Build a one-image, one-Car stat: a perfectly localized pred at given headings."""
    gt = Box3D(
        center_3d=(0.0, 0.0, 10.0),
        dimensions=(4.0, 1.6, 1.5),
        orientation=gt_ori,
        class_label="Car",
        class_id=0,
        confidence=1.0,
        truncated=0.0,
        occluded=0,
    )
    pred = Box3D(
        center_3d=(0.0, 0.0, 10.0),
        dimensions=(4.0, 1.6, 1.5),
        orientation=pred_ori,
        class_label="Car",
        class_id=0,
        confidence=0.9,
    )
    return {
        "gt_boxes": [gt],
        "pred_boxes": [pred],
        "iou_matrix": np.array([[1.0]]),  # perfect 3D localization (given)
        "bev_iou_matrix": np.array([[1.0]]),  # perfect BEV localization (given)
        "gt_difficulties": np.array([0]),  # Easy
        "pred_heights_2d": np.array([50.0]),  # above 25px min
    }


def test_metrics_aos_independent_of_ap():
    """AOS must reward heading: perfect box, flipped heading -> AP3D=1.0 but AOS=0.0."""
    from ultralytics.models.yolo.s3d.metrics import Stereo3DDetMetrics

    # Aligned heading: AP3D, AP_BEV and AOS all perfect.
    m = Stereo3DDetMetrics(names={0: "Car"})
    m.update_stats(_single_image_stat(gt_ori=0.0, pred_ori=0.0))
    m.process()
    assert abs(m.ap3d[0.7][0][0] - 1.0) < 1e-6
    assert abs(m.apbev[0.7][0][0] - 1.0) < 1e-6
    assert abs(m.aos[0.7][0][0] - 1.0) < 1e-6

    # Flipped heading (pi): still a localization TP (AP3D=1.0) but AOS collapses to 0.
    m2 = Stereo3DDetMetrics(names={0: "Car"})
    m2.update_stats(_single_image_stat(gt_ori=0.0, pred_ori=np.pi))
    m2.process()
    assert abs(m2.ap3d[0.7][0][0] - 1.0) < 1e-6
    assert m2.aos[0.7][0][0] < 1e-6

    # 90deg heading error -> AOS = (1 + cos(pi/2)) / 2 = 0.5.
    m3 = Stereo3DDetMetrics(names={0: "Car"})
    m3.update_stats(_single_image_stat(gt_ori=0.0, pred_ori=np.pi / 2))
    m3.process()
    assert abs(m3.aos[0.7][0][0] - 0.5) < 1e-6


def test_dimension_target_decode_roundtrip():
    """Dimension encode->decode round-trip must recover GT dims (regression for the string-vs-int keyed mean/std bug
    that inflated 3D box dimensions / length).

    The dataset encodes the SmoothL1 dimension target with compute_dimension_offset(), looking up per-class mean/std
    priors by INTEGER class_id. The dataset YAML keys those priors by class NAME ("Car"). If the dataset does not rekey
    name->int, every lookup misses and falls back to a generic mean/std, so the trained target is huge and the decoder
    (which uses correct int-keyed (H,W,L) priors) mis-expands dimensions.
    """
    from ultralytics.models.yolo.s3d.dataset import Stereo3DDetDataset, compute_dimension_offset

    # YAML-style string-keyed priors, [L, W, H] order (matches kitti-stereo*.yaml).
    names = {0: "Car", 1: "Cyclist"}
    mean_dims = {"Car": [3.9, 1.6, 1.5], "Cyclist": [1.8, 0.6, 1.7]}
    std_dims = {"Car": [0.42, 0.10, 0.15], "Cyclist": [0.25, 0.10, 0.15]}

    # Rekey to int ids the way the dataset constructor does, without touching disk.
    rekey = Stereo3DDetDataset._rekey_dims_to_int.__get__(
        type("S", (), {"names": names})()  # minimal object exposing .names
    )
    md_int = rekey(mean_dims)
    sd_int = rekey(std_dims)
    assert set(md_int) == {0, 1}, f"priors not rekeyed to int ids: {list(md_int)}"

    # Decode-side priors are int-keyed (H, W, L) — same reorder used by train/val.
    def to_HWL(d):
        return {cid: (v[2], v[1], v[0]) for cid, v in d.items()}

    md_dec, sd_dec = to_HWL(md_int), to_HWL(sd_int)

    # Known objects: (class_id, L, W, H). Car NOT at the mean (length 4.2 != 3.9).
    cases = [(0, 4.2, 1.6, 1.5), (1, 2.02, 0.60, 1.86)]
    for cid, L, W, H in cases:
        off = compute_dimension_offset((L, W, H), cid, md_int, sd_int)  # [dH, dW, dL]
        mh, mw, ml = md_dec[cid]
        sh, sw, sl = sd_dec[cid]
        dH = mh + float(off[0]) * sh
        dW = mw + float(off[1]) * sw
        dL = ml + float(off[2]) * sl
        assert abs(dL - L) < 1e-4, f"length round-trip failed cls{cid}: {dL} != {L}"
        assert abs(dW - W) < 1e-4, f"width round-trip failed cls{cid}: {dW} != {W}"
        assert abs(dH - H) < 1e-4, f"height round-trip failed cls{cid}: {dH} != {H}"


def test_orientation_multibin_roundtrip():
    """MultiBin encode->decode must recover the observation angle across the full circle.

    This is the guard against the encode/decode mismatch class of bug (cf. the dimension-prior bug). The decoder
    argmaxes the (one-hot) confidence and adds the residual to the chosen bin center; it must invert the encoder
    exactly.
    """
    import numpy as np

    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, decode_orientation, encode_orientation

    for alpha in np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 73):
        enc = encode_orientation(float(alpha))
        assert len(enc) == ORIENT_CHANNELS == 6
        dec = decode_orientation(enc)
        # angular difference wrapped to [-pi, pi]
        d = (dec - float(alpha) + np.pi) % (2 * np.pi) - np.pi
        assert abs(d) < 1e-5, f"alpha={alpha:.3f} -> decoded {dec:.3f} (err {d:.2e})"


def test_orientation_multibin_resolves_180():
    """Headings ~180 deg apart land in different bins, so a confident bin disambiguates them.

    A single sin/cos regressor that smears toward 0 cannot distinguish front/back; MultiBin assigns alpha and alpha+pi
    to different argmax bins.
    """
    from ultralytics.models.yolo.s3d.orientation import NUM_ORIENT_BINS, encode_orientation

    enc_a = encode_orientation(0.2)
    enc_b = encode_orientation(0.2 + 3.14159)  # ~180 deg apart
    bin_a = max(range(NUM_ORIENT_BINS), key=lambda i: enc_a[i])
    bin_b = max(range(NUM_ORIENT_BINS), key=lambda i: enc_b[i])
    assert bin_a != bin_b, f"180-deg-apart headings must differ in bin: {bin_a} vs {bin_b}"


def test_depth_decode_imgsz_invariant():
    """Stereo disparity->depth decode must recover true metric Z independent of imgsz.

    The dataset encodes the lr_distance target as log(disparity) in letterbox-NORMALIZED coordinates (dataset.py:
    "Normalized xywh (letterboxed input space)", disparity_norm = cx - right_cx). decode_stereo3d_outputs must invert
    that back to the same metric Z whether the letterbox is aspect-preserving (384x1248 on KITTI 375x1242, scale~=1) or
    square (640x640 / 384x384, scale<1). This is the regression guard for the bug where the decode scaled focal length
    by 1/letterbox_scale, yielding z_from_disp = Z_true / scale — correct only at scale~=1 and silently inflating stereo
    depth ~1.4-3.2x under a square imgsz.

    The direct-depth head (log(z_3d) target) is imgsz-invariant by construction and serves as the control: it must
    round-trip at every imgsz.
    """
    import math

    import torch

    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation
    from ultralytics.models.yolo.s3d.preprocess import compute_letterbox_params, decode_stereo3d_outputs

    calib = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.8540, "baseline": 0.54}
    ori_hw = (375, 1242)  # KITTI (H, W)
    nc = 3

    def decode_z(z_true, imgsz, use_disp, use_depth):
        """Faithfully encode the aux targets for a car at depth z_true, then run the real decode."""
        input_h, input_w = imgsz
        scale, _, _ = compute_letterbox_params(ori_hw[0], ori_hw[1], imgsz)
        # Single-anchor Detect output; the 2D box (hence x,y) is irrelevant to depth.
        det = torch.zeros(1, 4 + nc, 1)
        det[0, :4, 0] = torch.tensor([input_w / 2.0, input_h / 2.0, 20.0, 20.0])
        det[0, 4, 0] = 0.99  # class-0 (Car) score
        outputs = {"det": det, "dimensions": torch.zeros(1, 3, 1)}
        outputs["orientation"] = torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float()
        if use_disp:
            disparity_px_orig = calib["fx"] * calib["baseline"] / z_true
            # dataset.py encoding: disparity normalized by the letterbox canvas width.
            lr_log = math.log(disparity_px_orig * scale / input_w)
            outputs["lr_distance"] = torch.tensor([[[lr_log]]], dtype=torch.float32)
        if use_depth:
            outputs["depth"] = torch.tensor([[[math.log(z_true)]]], dtype=torch.float32)
        boxes = decode_stereo3d_outputs(outputs, conf_threshold=0.25, calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw])
        per_img = boxes[0] if boxes and isinstance(boxes[0], list) else boxes
        assert len(per_img) == 1, f"expected 1 decoded box, got {len(per_img)}"
        return float(per_img[0].center_3d[2])

    for imgsz in [(384, 1248), (640, 640), (384, 384)]:
        for z_true in (8.0, 25.0, 60.0):
            z_direct = decode_z(z_true, imgsz, use_disp=False, use_depth=True)
            z_disp = decode_z(z_true, imgsz, use_disp=True, use_depth=False)
            z_fused = decode_z(z_true, imgsz, use_disp=True, use_depth=True)
            assert abs(z_direct - z_true) / z_true < 0.02, f"direct depth imgsz={imgsz} z={z_true}: got {z_direct:.2f}"
            assert abs(z_disp - z_true) / z_true < 0.02, f"stereo depth imgsz={imgsz} z={z_true}: got {z_disp:.2f}"
            assert abs(z_fused - z_true) / z_true < 0.02, f"fused depth imgsz={imgsz} z={z_true}: got {z_fused:.2f}"


def test_decode_letterbox_calib_imgsz_invariant():
    """Decode must be imgsz-invariant when fed LETTERBOX-space calib (the real val/predict path).

    Regression guard for the square-imgsz bug: the batch supplies calib in letterbox-input space (fx,cx scaled by
    letterbox_scale, principal point shifted by padding). decode_stereo3d_outputs is called with calib_letterboxed=True
    (as decode_and_refine_predictions does) and must reverse it to original coords, so BOTH the stereo depth (uses fx)
    and the 2D-center back-projection (uses cx,cy) recover the true metric X and Z at any imgsz. The pre-fix decode used
    letterbox fx/cx directly → z halved and x/y garbage under a square imgsz (scale~=0.5) while dormant at rect
    (scale~=1).
    """
    import math

    import torch

    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation
    from ultralytics.models.yolo.s3d.preprocess import compute_letterbox_params, decode_stereo3d_outputs

    fx0, fy0, cx0, cy0, baseline = 721.5377, 721.5377, 609.5593, 172.8540, 0.54
    ori_hw = (375, 1242)  # KITTI (H, W)
    nc = 3

    def decode_xz(z_true, u_orig_true, imgsz):
        """Encode a car at (u_orig_true, z_true) with LETTERBOX-space calib; return decoded (x, z)."""
        _input_h, input_w = imgsz
        scale, pad_left, pad_top = compute_letterbox_params(ori_hw[0], ori_hw[1], imgsz)
        # Calib as the batch provides it: letterbox-input space.
        calib_lb = {
            "fx": fx0 * scale,
            "fy": fy0 * scale,
            "cx": cx0 * scale + pad_left,
            "cy": cy0 * scale + pad_top,
            "baseline": baseline,
        }
        # 2D box center in letterbox-input pixels for a feature at original u_orig_true (v at horizon).
        v_orig_true = cy0
        u_lb = u_orig_true * scale + pad_left
        v_lb = v_orig_true * scale + pad_top
        det = torch.zeros(1, 4 + nc, 1)
        det[0, :4, 0] = torch.tensor([u_lb, v_lb, 20.0, 20.0])
        det[0, 4, 0] = 0.99
        disparity_px_orig = fx0 * baseline / z_true
        outputs = {
            "det": det,
            "dimensions": torch.zeros(1, 3, 1),
            "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
            "lr_distance": torch.tensor([[[math.log(disparity_px_orig * scale / input_w)]]], dtype=torch.float32),
            "depth": torch.tensor([[[math.log(z_true)]]], dtype=torch.float32),
        }
        boxes = decode_stereo3d_outputs(
            outputs,
            conf_threshold=0.25,
            calib=[calib_lb],
            imgsz=imgsz,
            ori_shapes=[ori_hw],
            calib_letterboxed=True,
        )
        per_img = boxes[0] if boxes and isinstance(boxes[0], list) else boxes
        assert len(per_img) == 1, f"expected 1 decoded box, got {len(per_img)}"
        c = per_img[0].center_3d
        return float(c[0]), float(c[2])

    for imgsz in [(384, 1248), (640, 640), (384, 384)]:
        for z_true, u_orig_true in [(25.0, 609.5593), (12.0, 300.0), (45.0, 900.0)]:
            x_true = (u_orig_true - cx0) * z_true / fx0
            x_dec, z_dec = decode_xz(z_true, u_orig_true, imgsz)
            assert abs(z_dec - z_true) / z_true < 0.02, f"z imgsz={imgsz} z={z_true}: got {z_dec:.2f}"
            assert abs(x_dec - x_true) < 0.5, f"x imgsz={imgsz} z={z_true}: got {x_dec:.2f} want {x_true:.2f}"


def test_decode_uses_proj_offset():
    """With use_proj_center, a nonzero proj_offset shifts the recovered x_3d by du*input_w/scale*z/fx."""
    import math

    import torch

    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation
    from ultralytics.models.yolo.s3d.preprocess import compute_letterbox_params, decode_stereo3d_outputs

    calib = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.8540, "baseline": 0.54}
    ori_hw = (375, 1242)
    imgsz = (384, 1248)
    nc = 3
    input_h, input_w = imgsz
    scale, _pad_left, _ = compute_letterbox_params(*ori_hw, imgsz)
    z = 20.0
    du = 0.01

    def make_outputs(with_proj):
        # non_max_suppression converts outputs["det"] xywh->xyxy in place (via a transposed
        # view aliasing the same storage), so each decode call needs its own fresh det tensor.
        det = torch.zeros(1, 4 + nc, 1)
        det[0, :4, 0] = torch.tensor([input_w / 2, input_h / 2, 20.0, 20.0])
        det[0, 4, 0] = 0.99
        out = {
            "det": det,
            "dimensions": torch.zeros(1, 3, 1),
            "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
            "depth": torch.tensor([[[math.log(z)]]]),
        }
        if with_proj:  # promoted decode applies proj_offset unconditionally when present
            out["proj_offset"] = torch.tensor([[[du], [0.0]]]).float()
        return out

    # bs=1 -> decode_stereo3d_outputs returns a flat list[Box3D] (unwrapped), so index once.
    x_off = decode_stereo3d_outputs(make_outputs(True), calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw])[0].center_3d[0]
    x_no = decode_stereo3d_outputs(make_outputs(False), calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw])[0].center_3d[0]
    expected_shift = (du * input_w / scale) * z / calib["fx"]
    assert abs((x_off - x_no) - expected_shift) < 1e-2, f"{x_off - x_no} != {expected_shift}"


def test_ivw_fusion_equal_sigma_matches_geomean():
    """With equal per-cue variance, inverse-variance fusion in log-space == geometric mean (A0 continuity)."""
    import math

    import torch

    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation
    from ultralytics.models.yolo.s3d.preprocess import decode_stereo3d_outputs

    calib = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.8540, "baseline": 0.54}
    ori_hw = (375, 1242)
    imgsz = (384, 1248)
    nc = 3

    def make_outputs(with_logvar):
        # non_max_suppression converts outputs["det"] xywh->xyxy in place (via a transposed
        # view aliasing the same storage), so each decode call needs its own fresh det tensor.
        det = torch.zeros(1, 4 + nc, 1)
        det[0, :4, 0] = torch.tensor([624.0, 192.0, 20.0, 20.0])
        det[0, 4, 0] = 0.99
        # disparity cue and direct cue encode different depths so the mean is nontrivial.
        out = {
            "det": det,
            "dimensions": torch.zeros(1, 3, 1),
            "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
            "lr_distance": torch.tensor([[[math.log(0.03)]]]),
            "depth": torch.tensor([[[math.log(25.0)]]]),
        }
        if with_logvar:  # equal per-cue variance (logvar=0 -> var_disp=1; no depth_bins -> var_direct=1.0)
            out["lr_logvar"] = torch.tensor([[[0.0]]])
        return out

    # bs=1 -> decode_stereo3d_outputs returns a flat list[Box3D] (unwrapped), so index once.
    # No lr_logvar -> geometric-mean fallback; equal-variance lr_logvar -> IVW. They must coincide.
    z_geo = decode_stereo3d_outputs(make_outputs(False), calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw])[0].center_3d[
        2
    ]
    z_ivw = decode_stereo3d_outputs(make_outputs(True), calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw])[0].center_3d[2]
    assert abs(z_ivw - z_geo) < 1e-2, f"ivw {z_ivw} != geomean {z_geo}"


def test_ivw_fusion_uses_dfl_variance():
    """With depth_bins present, ivw_fusion must weight the direct cue by its DFL spread (not the constant 1.0 fallback),
    pulling the fused z away from the plain geomean when the DFL distribution is narrow (low variance) while the
    stereo cue is highly uncertain.
    """
    import math

    import torch

    from ultralytics.models.yolo.s3d.head import DEPTH_BINS, DEPTH_MAX, DEPTH_MIN
    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation
    from ultralytics.models.yolo.s3d.preprocess import decode_stereo3d_outputs

    calib = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.8540, "baseline": 0.54}
    ori_hw = (375, 1242)
    imgsz = (384, 1248)
    nc = 3

    # A depth_bins distribution heavily peaked at one bin => near-zero DFL variance.
    log_min, log_max = math.log(DEPTH_MIN), math.log(DEPTH_MAX)
    peak_idx = 13
    bin_log = log_min + peak_idx * (log_max - log_min) / (DEPTH_BINS - 1)
    z_direct_true = math.exp(bin_log)
    logits = torch.full((DEPTH_BINS,), -20.0)
    logits[peak_idx] = 20.0

    def make_outputs(with_logvar):
        # non_max_suppression mutates outputs["det"] in place, so build a fresh tensor per call.
        det = torch.zeros(1, 4 + nc, 1)
        det[0, :4, 0] = torch.tensor([624.0, 192.0, 20.0, 20.0])
        det[0, 4, 0] = 0.99
        out = {
            "det": det,
            "dimensions": torch.zeros(1, 3, 1),
            "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
            "lr_distance": torch.tensor([[[math.log(0.03)]]]),  # z_from_disp ~ 10.5
            "depth": torch.tensor([[[bin_log]]]),  # z_from_direct == z_direct_true (matches the peak bin)
            "depth_bins": logits.view(1, DEPTH_BINS, 1),
        }
        if with_logvar:  # high stereo variance (var_disp = exp(3) ~ 20.1) -> IVW leans on the direct cue
            out["lr_logvar"] = torch.tensor([[[3.0]]])
        return out

    # No lr_logvar -> geometric-mean fallback; with lr_logvar -> IVW using the DFL spread.
    z_geo = decode_stereo3d_outputs(make_outputs(False), calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw])[0].center_3d[
        2
    ]
    z_ivw = decode_stereo3d_outputs(make_outputs(True), calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw])[0].center_3d[2]

    # The direct cue is nearly certain (narrow DFL distribution) while the stereo cue is highly
    # uncertain, so IVW fusion must sit much closer to z_direct than the plain geometric mean.
    assert abs(z_ivw - z_geo) > 1.0, f"ivw ({z_ivw}) did not diverge from geomean ({z_geo}); depth_bins not used"
    assert abs(z_ivw - z_direct_true) < abs(z_geo - z_direct_true), "ivw should be pulled toward the direct cue"


def _fusion_scale_outputs():
    """Build s3d outputs whose two depth cues disagree and whose DFL spread is moderate.

    Flat bin logits give var_direct ~ the grid's own variance (O(1)), well clear of the decode's 1e-6 floor --
    a peaked distribution would be clamped there and make any variance rescaling invisible.
    non_max_suppression rewrites outputs["det"] in place, so every decode call needs its own dict.
    """
    import math

    import torch

    from ultralytics.models.yolo.s3d.head import DEPTH_BINS, DepthDFL
    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation

    det = torch.zeros(1, 4 + 3, 1)
    det[0, :4, 0] = torch.tensor([624.0, 192.0, 20.0, 20.0])
    det[0, 4, 0] = 0.99
    return {
        "det": det,
        "dimensions": torch.zeros(1, 3, 1),
        "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
        "lr_distance": torch.tensor([[[math.log(0.03)]]]),  # z_from_disp ~ 10.5
        "depth": torch.tensor([[[math.log(25.0)]]]),  # z_from_direct = 25.0, far from the disparity cue
        "lr_logvar": torch.tensor([[[0.0]]]),  # var_disp = 1, so the two weights are comparable
        "depth_bins": torch.zeros(1, DEPTH_BINS, 1),  # flat => var_direct = the grid's variance
        "depth_bin_values": DepthDFL().bin_values,
    }


def _decode_fusion_scale(**kwargs):
    """Decode `_fusion_scale_outputs()` and return every numeric field as one float64 tensor."""
    import torch

    from ultralytics.models.yolo.s3d.preprocess import decode_stereo3d_outputs

    calib = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.8540, "baseline": 0.54}
    boxes = decode_stereo3d_outputs(
        _fusion_scale_outputs(), calib=[calib], imgsz=(384, 1248), ori_shapes=[(375, 1242)], **kwargs
    )
    assert boxes, "decode returned no boxes; the fixture no longer produces a detection"
    flat = [v for b in boxes for v in (*b.center_3d, *b.dimensions, b.orientation, b.confidence)]
    return torch.tensor(flat, dtype=torch.float64)  # float64 keeps the Python floats bit-exact


def test_depth_var_scale_default_is_a_bitwise_noop():
    """depth_var_scale=1.0 must be indistinguishable from not passing it at all.

    The knob exists only to reproduce a bin count's fusion re-weighting at a fixed bin count, so every run
    that does not ask for it -- every existing checkpoint, every other arm -- must decode bit for bit as
    before. A float multiply by 1.0 is exact in IEEE 754, and this pins that.
    """
    import torch

    assert torch.equal(_decode_fusion_scale(depth_var_scale=1.0), _decode_fusion_scale())


def test_depth_var_scale_shifts_the_fusion_toward_the_direct_cue():
    """Shrinking var_direct must raise the direct cue's inverse-variance weight, pulling z toward it."""
    z_default = float(_decode_fusion_scale()[2])
    z_scaled = float(_decode_fusion_scale(depth_var_scale=0.5)[2])
    z_direct = 25.0  # outputs["depth"] in _fusion_scale_outputs

    assert abs(z_scaled - z_direct) < abs(z_default - z_direct), (
        f"scale=0.5 ({z_scaled}) is no closer to the direct cue ({z_direct}) than scale=1.0 ({z_default})"
    )
    assert z_scaled > z_default, "the direct cue is the deeper one, so more weight on it must increase z"


def test_decode_no_overflow_with_large_lr_logvar():
    """A pathological lr_logvar must not crash math.exp (Fix A: lr_logvar is clamped at sample time)."""
    import math

    import torch

    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation
    from ultralytics.models.yolo.s3d.preprocess import decode_stereo3d_outputs

    calib = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.8540, "baseline": 0.54}
    ori_hw = (375, 1242)
    imgsz = (384, 1248)
    nc = 3

    det = torch.zeros(1, 4 + nc, 1)
    det[0, :4, 0] = torch.tensor([624.0, 192.0, 20.0, 20.0])
    det[0, 4, 0] = 0.9
    outputs = {
        "det": det,
        "dimensions": torch.zeros(1, 3, 1),
        "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
        "lr_distance": torch.tensor([[[math.log(0.03)]]]),
        "depth": torch.tensor([[[math.log(25.0)]]]),
        "lr_logvar": torch.tensor([[[1000.0]]]),  # would overflow math.exp without a clamp
    }
    boxes = decode_stereo3d_outputs(outputs, calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw])
    assert len(boxes) == 1


def test_proj_offset_roundtrip():
    """Projected-centroid offset must invert: encode (centroid->projected px->offset) then decode (box_center+offset ->
    back-project at true z) recovers the centroid X/Y.
    """
    from ultralytics.models.yolo.s3d.dataset import encode_proj_offset

    fx = fy = 721.5377
    cx = 609.5593
    cy = 172.8540
    calib = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    input_w, input_h = 1248, 384
    scale, pad_left, pad_top = 1.0048, 0, 3  # aspect-preserving letterbox of 375x1242
    # A car: bottom-center location (X,Y,Z), height h. Centroid Y = Y - h/2 (camera y-down).
    X, Y, Z, h = 3.0, 1.65, 20.0, 1.5
    # 2D box center (letterbox-normalized) — deliberately NOT the projected centroid.
    box_center_norm = (0.42, 0.55)  # (u,v) normalized in letterbox space

    du, dv = encode_proj_offset((X, Y, Z), h, calib, box_center_norm, (scale, pad_left, pad_top), (input_w, input_h))

    # Decode: recovered projected center in original px, then back-project at Z.
    u_norm = box_center_norm[0] + du
    v_norm = box_center_norm[1] + dv
    u_lb = u_norm * input_w
    v_lb = v_norm * input_h
    u_orig = (u_lb - pad_left) / scale
    v_orig = (v_lb - pad_top) / scale
    x_rec = (u_orig - cx) * Z / fx
    y_rec = (v_orig - cy) * Z / fy
    assert abs(x_rec - X) < 1e-3, f"x {x_rec} != {X}"
    assert abs(y_rec - (Y - h / 2)) < 1e-3, f"y {y_rec} != centroid {Y - h / 2}"


# Simplified fx/fy/cx/cy calib format that convert_kitti_3d.py emits and datasets (kitti-stereo8) ship.
SIMPLE_CALIB = (
    "fx: 721.537700\nfy: 721.537700\ncx: 609.559300\ncy: 172.854000\n"
    "right_cx: 609.559300\nright_cy: 172.854000\nbaseline: 0.532725\n"
    "image_width: 1242\nimage_height: 375\n"
)
# Raw KITTI P0..P3 projection-matrix format.
KITTI_CALIB = (
    "P0: 721.5377 0 609.5593 0 0 721.5377 172.854 0 0 0 1 0\n"
    "P1: 721.5377 0 609.5593 -387.5744 0 721.5377 172.854 0 0 0 1 0\n"
    "P2: 721.5377 0 609.5593 44.85728 0 721.5377 172.854 0.2163791 0 0 1 0.002745884\n"
    "P3: 721.5377 0 609.5593 -339.5242 0 721.5377 172.854 2.199936 0 0 1 0.002729905\n"
)


@pytest.mark.parametrize("content", [SIMPLE_CALIB, KITTI_CALIB])
def test_calib_dual_format(tmp_path, content):
    """load_kitti_calibration must parse BOTH the simplified fx/fy format and raw KITTI P-matrices."""
    from ultralytics.data.stereo.calib import load_kitti_calibration

    f = tmp_path / "000000.txt"
    f.write_text(content)
    calib = load_kitti_calibration(f)
    assert abs(calib.fx - 721.5377) < 1e-2
    assert abs(calib.cx - 609.5593) < 1e-2
    assert abs(calib.cy - 172.854) < 1e-2
    # Both formats encode the same stereo baseline (P2/P3 -> (44.857 - -339.524)/721.54 ~= 0.5327 m).
    assert abs(calib.baseline - 0.532725) < 1e-3, f"baseline {calib.baseline} != 0.5327"


def test_predict_resolves_dataset_calib(tmp_path):
    """Predictor must find + parse per-image calib in the dataset `calib/<split>/` layout, not silently fall back to
    default calibration.
    """
    import cv2

    root = tmp_path / "ds"
    left_dir = root / "images" / "val" / "left"
    right_dir = root / "images" / "val" / "right"
    calib_dir = root / "calib" / "val"
    for d in (left_dir, right_dir, calib_dir):
        d.mkdir(parents=True)
    img = np.zeros((375, 1242, 3), dtype=np.uint8)
    cv2.imwrite(str(left_dir / "000000.png"), img)
    cv2.imwrite(str(right_dir / "000000.png"), img)
    (calib_dir / "000000.txt").write_text(SIMPLE_CALIB)

    model = YOLO(MODEL)
    results = model.predict(source=[(str(left_dir / "000000.png"), str(right_dir / "000000.png"))], imgsz=[384, 1248])
    calib = getattr(results[0], "_calib", None)
    assert calib is not None, "predictor did not attach calibration"
    # Real baseline is 0.532725; the default fallback is 0.54 — must be the real one.
    assert abs(calib["baseline"] - 0.532725) < 1e-3, f"got baseline {calib['baseline']} (default fallback?)"


def test_configurable_depth_range():
    """Depth bins and loss normalization use the configured range."""
    import math

    from ultralytics.models.yolo.s3d.head import DepthDFL

    dfl = DepthDFL()
    dfl._set_range(0.2, 2.5)
    assert math.exp(dfl.bin_values[0].item()) == pytest.approx(0.2)
    assert math.exp(dfl.bin_values[-1].item()) == pytest.approx(2.5)

    model = YOLO(MODEL).model
    model.model[-1].depth_dfl._set_range(0.2, 2.5)
    criterion = model.init_criterion()
    assert criterion.depth_log_min == pytest.approx(math.log(0.2))
    assert criterion.depth_log_range == pytest.approx(math.log(2.5 / 0.2))


@pytest.mark.parametrize("imgsz", [(384, 1248), (640, 640)])
def test_stereo_reaches_every_scale(imgsz):
    """Depth and disparity at EVERY FPN scale must be a function of the right image.

    The cost volume is the only right-image path into the depth branches. If it is concatenated at one
    scale only, objects whose anchors land on the other scales are decoded monocularly by construction —
    no amount of training can recover stereo for them.

    eval() is mandatory: in train mode backbone layers 0-1 run on the concatenated [2B, 3, H, W] siamese
    batch, so BatchNorm mixes left/right statistics and the left path acquires a spurious right-image
    dependence at every scale, which would make this test pass for the wrong reason.
    """
    import torch

    h, w = imgsz
    model = YOLO(MODEL).model.eval()
    x = torch.rand(1, 6, h, w, requires_grad=True)
    preds = model(x)[1]

    offset, magnitudes = 0, {}
    for stride in (int(s) for s in model.stride.tolist()):
        gh, gw = h // stride, w // stride
        idx = offset + (gh // 2) * gw + gw // 2  # centre anchor of this scale
        offset += gh * gw
        for key in ("depth", "lr_distance"):
            grad = torch.autograd.grad(preds[key][0, 0, idx], x, retain_graph=True)[0]
            magnitudes[f"{key}@stride{stride}"] = grad[:, 3:].abs().sum().item()

    # Guards the flat-index -> scale mapping: the strides must tile the anchor axis exactly.
    assert offset == preds["depth"].shape[2], f"anchor mapping wrong: {offset} != {preds['depth'].shape[2]}"

    blind = {k: v for k, v in magnitudes.items() if v == 0.0}
    assert not blind, f"no right-image gradient at {sorted(blind)}; all magnitudes: {magnitudes}"


@pytest.mark.parametrize("true_disp_px", [3.0, 7.5, 12.25])
def test_cost_volume_recovers_known_disparity(true_disp_px):
    """The soft-argmin readout must recover a known disparity from a synthetic shifted pair.

    This is the learnability gate for the stereo cue: a model-free NCC block matcher recovers per-object
    disparity on real KITTI to ~1 px, so the readout is the only thing that can lose it. A conv over the
    cost channels cannot express peak position at all; soft-argmin over the disparity grid can, and being
    an expectation it is continuous, so it must also land between grid levels.
    """
    import torch

    from ultralytics.nn.modules.block import StereoCostVolume

    c1, groups, levels = 32, 4, 48
    cv = StereoCostVolume(c1, 64, levels, groups).eval()
    H, W = 64, 320
    # Grid spans 0..32 feature px at this width, so true_disp_px is interior and sub-level.
    cv.set_disparity_range(0.0, 32.0 / W)

    torch.manual_seed(0)
    # Spatially smooth features, so correlation varies with disparity the way real features do.
    base = torch.randn(1, c1, H, W * 2)
    base = torch.nn.functional.avg_pool2d(base, 5, 1, 2)
    xs = torch.arange(W, dtype=torch.float32) + W // 2

    def sample(shift):
        gx = ((xs - shift) / (2 * W - 1) * 2 - 1).view(1, 1, -1).expand(1, H, W)
        gy = (torch.arange(H, dtype=torch.float32) / (H - 1) * 2 - 1).view(1, -1, 1).expand(1, H, W)
        return torch.nn.functional.grid_sample(base, torch.stack([gx, gy], -1), align_corners=True)

    # Real stereo puts a left-image point at u_R = u_L - d, i.e. right[x] == left[x + d], so the right
    # view samples the base image at a NEGATIVE offset. Getting this sign backwards silently produces an
    # unmatchable pair, which is why the module's own convention is pinned by the real-KITTI argmax.
    left, right = sample(0.0), sample(-true_disp_px)

    with torch.no_grad():
        out = cv((left, right))
    # Channel 0 is the [0,1] readout; convert back to feature pixels via the grid.
    disp_px = float(cv.to_disparity(out[:, :1]).median()) * W

    assert abs(disp_px - true_disp_px) < 2.0, f"recovered {disp_px:.2f} px, expected {true_disp_px:.2f} px"


def test_cost_volume_disparity_grid_is_resolution_invariant():
    """The width-normalized grid must give the same disparity in *normalized* units at any input width.

    Offsets are stored as a fraction of image width (what a stereo rig physically fixes) and converted to
    feature pixels at forward time, so one checkpoint serves rectangular and square imgsz alike. Storing
    raw pixel offsets instead is what made the grid silently mis-scaled per dataset and per imgsz.
    """
    import torch

    from ultralytics.nn.modules.block import StereoCostVolume

    cv = StereoCostVolume(32, 64, 32, 4).eval()
    cv.set_disparity_range(0.01, 0.10)
    torch.manual_seed(0)
    for W in (160, 320):
        left = torch.nn.functional.avg_pool2d(torch.randn(1, 32, 48, W), 5, 1, 2)
        with torch.no_grad():
            # roll(-k) gives out[x] == left[x + k], the real stereo relation for disparity k.
            out = cv((left, left.roll(shifts=-int(0.05 * W), dims=-1)))
        # A 0.05*W shift is 0.05 in normalized units at BOTH widths.
        assert abs(float(cv.to_disparity(out[:, :1]).median()) - 0.05) < 0.02, f"width {W} mis-scaled"


def test_cost_volume_correlation_is_spatially_centred():
    """Correlation must run on spatially-centred features, or the disparity gradient starves at init.

    Cosine similarity of raw post-activation features is DC-dominated: measured on real KITTI pairs it is
    ~0.92 at EVERY disparity offset (3.8% dynamic range), versus 32% once each channel's spatial mean is
    removed. Without centering there is almost no disparity signal to descend on.
    """
    import torch

    from ultralytics.nn.modules.block import StereoCostVolume

    cv = StereoCostVolume(32, 64, 24, 4).eval()
    torch.manual_seed(0)
    # Strong positive DC offset, as post-SiLU backbone features have.
    feats = torch.nn.functional.avg_pool2d(torch.randn(1, 32, 48, 256), 5, 1, 2) + 5.0

    costs = []
    for shift in (2, 10, 24):
        with torch.no_grad():
            out = cv((feats, feats.roll(shifts=-shift, dims=-1)))
        costs.append(float(cv.to_disparity(out[:, :1]).median()))
    # Centred correlation keeps the readout responsive to disparity despite the DC offset.
    assert max(costs) - min(costs) > 1e-3, f"readout is flat across disparities: {costs}"


def test_decode_anchor_count_ignores_single_scale_maps():
    """`hw_total` must come from a per-anchor aux map, never from a single-scale map like cv_disparity.

    The aux maps are [B, C, HW_total] and detections index into them by flat anchor index. The head also
    emits cv_disparity on one scale's grid ([B, 1, H/8, W/8]); taking shape[2] off that yields the grid
    HEIGHT, which silently clamps every detection into the first few anchors. It zeroes every 3D metric
    while leaving 2D detection and all training losses untouched — so nothing else in this suite, and no
    loss curve, reveals it.
    """
    import torch

    from ultralytics.models.yolo.s3d.preprocess import PER_ANCHOR_AUX_KEYS

    hw_total, h8, w8 = 6300, 48, 156
    # cv_disparity deliberately FIRST, which is the dict order forward_head produces.
    outputs = {
        "cv_disparity": torch.zeros(1, 1, h8, w8),
        "lr_distance": torch.zeros(1, 1, hw_total),
        "depth": torch.zeros(1, 1, hw_total),
        "dimensions": torch.zeros(1, 3, hw_total),
    }
    resolved = next((outputs[k].shape[2] for k in PER_ANCHOR_AUX_KEYS if k in outputs and outputs[k].ndim == 3), 0)
    assert resolved == hw_total, f"resolved {resolved}, expected {hw_total} (cv_disparity would give {h8})"
    assert "cv_disparity" not in PER_ANCHOR_AUX_KEYS


def _sampler_drive_counts(sampler, drive_of):
    """Return the empirical per-drive frame counts of one pass over a sampler."""
    from collections import Counter

    return Counter(drive_of[i] for i in sampler)


def test_drive_balanced_sampler_flattens_drive_concentration():
    """Drive-balanced sampling must actually rebalance, and must never upweight the long tail.

    The drive-disjoint KITTI split puts 37.8% of its 3712 frames in 5 of 71 drives, so every epoch's
    gradient is a handful of scenes seen from slightly different positions. The cap only down-weights
    over-represented drives: frames in drives at or below the cap keep the uniform weight they have
    today, which is what distinguishes this from inverse-frequency balancing (which would draw a
    one-frame drive ~52 times an epoch).
    """
    from ultralytics.models.yolo.s3d.train import drive_balanced_sampler

    sizes = [320] * 5 + [10] * 66  # 71 drives, 2260 frames, median 10 — the real split's shape
    ids, drives, drive_of = [], {}, {}
    for d, n in enumerate(sizes):
        for k in range(n):
            stem = f"{d:03d}_{k:04d}"
            ids.append(stem)
            drives[stem] = f"drive_{d}"
            drive_of[len(ids) - 1] = f"drive_{d}"
    big = {f"drive_{d}" for d in range(5)}
    uniform_top5 = sum(sizes[:5]) / len(ids)
    assert uniform_top5 > 0.7, "synthetic split is not concentrated enough to be a test"

    sampler = drive_balanced_sampler(ids, drives, balance=1.0)
    assert len(sampler) == len(ids), "epoch length must be preserved so the LR schedule is unchanged"

    w = sampler.weights
    assert w.max() <= 1.0 + 1e-9, "the cap must never upweight a frame above uniform"
    tail = [float(w[i]) for i in range(len(ids)) if drive_of[i] not in big]
    assert min(tail) == max(tail) == 1.0, "frames in under-represented drives must keep uniform weight"

    # Measured rebalancing, on the drawn indices rather than on the weights alone.
    counts = _sampler_drive_counts(sampler, drive_of)
    drawn_top5 = sum(counts[d] for d in big) / sum(counts.values())
    assert drawn_top5 < 0.3, f"top-5 drive share only fell {uniform_top5:.3f} -> {drawn_top5:.3f}"

    # The strength is configurable, and its large-balance limit is exactly uniform sampling.
    flatter = drive_balanced_sampler(ids, drives, balance=0.25)
    flat_top5 = sum(_sampler_drive_counts(flatter, drive_of)[d] for d in big) / len(ids)
    assert flat_top5 < drawn_top5, f"balance=0.25 ({flat_top5:.3f}) is not flatter than 1.0 ({drawn_top5:.3f})"
    assert drive_balanced_sampler(ids, drives, balance=1e6).weights.min() == 1.0, "large balance must be a no-op"


def test_drive_sampler_seed_varies_with_the_run_seed(tmp_path):
    """Two runs with different `seed` must draw different frame orders, not just different weights.

    `_drive_sampler` previously seeded from `max(rank, 0)`, which is 0 for every single-process run. A
    multi-seed noise-floor measurement would then vary initialisation and augmentation but reuse one
    sampling order, understating the real run-to-run spread — the exact quantity such a measurement exists
    to bound. The sampler seed now mixes `args.seed` with the rank.
    """
    from ultralytics.models.yolo.s3d.train import Stereo3DDetTrainer

    trainer = Stereo3DDetTrainer(overrides={"model": MODEL, "data": DATA, "epochs": 1, "imgsz": [384, 1248]})
    dataset = trainer.build_dataset(trainer.data["train"], mode="train")
    # `drives` is a PATH to a json map, not the map itself. One drive for every frame, so the only thing
    # the seed can change is the drawn order.
    dmap = tmp_path / "drives.json"
    dmap.write_text(json.dumps({Path(f).stem: "d0" for f in dataset.im_files}))
    trainer.data["drives"] = str(dmap)

    orders = []
    for seed in (0, 1):
        trainer.args.seed = seed
        sampler = trainer._drive_sampler(dataset, -1)
        assert sampler is not None, "a drives map is present, so a sampler must be built"
        orders.append(list(sampler))
    assert orders[0] != orders[1], "seed must change the drawn frame order"

    trainer.args.seed = 0
    assert list(trainer._drive_sampler(dataset, -1)) == orders[0], "the same seed must reproduce its order"


def test_drive_sampler_supports_the_ddp_set_epoch_contract():
    """The train sampler must implement `set_epoch`, and each epoch must draw a different sample.

    `BaseTrainer` calls `self.train_loader.sampler.set_epoch(epoch)` for every epoch whenever `RANK != -1`.
    A plain `WeightedRandomSampler` has no such method, so a multi-GPU run died with AttributeError on both
    ranks before its first step while single-process runs passed — which is exactly how this reached a GPU
    box unnoticed. Reseeding from (seed, epoch) also makes a resumed run reproduce its draw order.
    """
    from ultralytics.models.yolo.s3d.train import drive_balanced_sampler

    ids = [f"{i:06d}" for i in range(200)]
    drives = {str(i): f"drive_{i // 50}" for i in range(200)}
    sampler = drive_balanced_sampler(ids, drives, balance=1.0)

    assert hasattr(sampler, "set_epoch"), "sampler must satisfy the DDP set_epoch contract"

    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second, "consecutive epochs must not draw an identical sample order"

    sampler.set_epoch(0)
    assert list(sampler) == first, "set_epoch(e) must be reproducible, so a resume repeats the same draw"

    # Ranks must not collide: rank r at epoch e cannot mirror rank r' at epoch e'.
    other = drive_balanced_sampler(ids, drives, balance=1.0, seed=1)
    other.set_epoch(0)
    sampler.set_epoch(1)
    assert list(other) != list(sampler), "per-rank seeds must stay decorrelated across epochs"


def test_drive_map_accepts_zero_padded_and_plain_integer_keys():
    """The drives map must resolve whether it is keyed "000003" or "3".

    KITTI frame ids are zero-padded on disk, so `dataset.im_files` stems are "000003", while split files
    conventionally key frames as plain integers ("3"). If only the exact stem is tried, every lookup misses,
    every frame becomes its own single-frame drive, the cap becomes non-binding, and the sampler silently
    degrades to uniform — a no-op that looks like a working feature. Verified against the real split file:
    0/5 stems resolved by exact match, 5/5 by integer normalisation.
    """
    from ultralytics.models.yolo.s3d.train import drive_balanced_sampler

    # Drive sizes must be UNEQUAL, otherwise the assertion cannot tell success from total failure: with two
    # equal drives the cap yields one weight, and so does the all-singleton fallback. Sizes 50 and 10 with
    # balance=1.0 give cap = 60/2 = 30, hence weights {min(1, 30/50), min(1, 30/10)} = {0.6, 1.0} — two
    # distinct values, where a failed lookup produces 60 singletons, cap 1.0, and a single weight of 1.0.
    stems = [f"{i:06d}" for i in range(60)]  # on-disk form
    drive_of = lambda i: "drive_big" if i < 50 else "drive_small"  # noqa: E731
    plain = {str(i): drive_of(i) for i in range(60)}  # split-file form
    padded = {f"{i:06d}": drive_of(i) for i in range(60)}  # already-padded form

    for label, mapping in (("plain-int keys", plain), ("zero-padded keys", padded)):
        w = drive_balanced_sampler(stems, mapping, balance=1.0).weights
        got = sorted(round(v, 6) for v in w.unique().tolist())
        assert got == [0.6, 1.0], f"{label}: expected per-drive weights [0.6, 1.0], got {got}"

    # Contrast: an unresolvable map must not masquerade as balanced. Every frame becomes its own drive, so
    # the weights come out flat (their absolute value is an artefact of the cap) and sampling is uniform —
    # which is why the function also warns about unmapped frames rather than relying on the weights to show it.
    unmatched = drive_balanced_sampler(stems, {f"x{i}": "d" for i in range(60)}, balance=0.5).weights
    assert len(unmatched.unique()) == 1, f"an unusable map must degrade to uniform, got {unmatched.unique()}"


def test_dataset_without_drives_key_is_unaffected():
    """A dataset YAML with no `drives:` key must keep the default uniform/DistributedSampler behaviour."""
    from ultralytics.models.yolo.s3d.train import Stereo3DDetTrainer

    trainer = Stereo3DDetTrainer(overrides={"model": MODEL, "data": DATA, "epochs": 1, "imgsz": [384, 1248]})
    assert trainer.data["drives"] is None, "kitti-stereo8.yaml declares no drives map"
    dataset = trainer.build_dataset(trainer.data["train"], mode="train")
    assert trainer._drive_sampler(dataset, -1) is None, "no drives map must mean no sampler override"


def test_drives_yaml_key_builds_balanced_sampler(tmp_path):
    """The optional `drives:` YAML key plumbs a drive map through get_dataset into the train sampler."""
    from ultralytics.models.yolo.s3d.train import Stereo3DDetTrainer
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    cfg = YAML.load(check_yaml(DATA))
    drives_file = tmp_path / "drives.json"
    yaml_file = tmp_path / "kitti-stereo8-drives.yaml"
    cfg["drives"] = str(drives_file)
    cfg["drive_balance"] = 1.0
    YAML.save(yaml_file, cfg)

    trainer = Stereo3DDetTrainer(overrides={"model": MODEL, "data": str(yaml_file), "epochs": 1, "imgsz": [384, 1248]})
    dataset = trainer.build_dataset(trainer.data["train"], mode="train")
    stems = [Path(f).stem for f in dataset.im_files]
    # All but one frame in a single drive: the cap must shrink that drive's share of the epoch.
    drive_map = dict.fromkeys(stems[:-1], "d0")
    drive_map[stems[-1]] = "d1"
    drives_file.write_text(json.dumps({"drives": drive_map}))

    sampler = trainer._drive_sampler(dataset, -1)
    assert sampler is not None, "drives: key did not reach the trainer"
    assert float(sampler.weights[:-1].max()) < 1.0, f"over-represented drive was not capped: {sampler.weights}"
    assert float(sampler.weights[-1]) == 1.0, "the single-frame drive must keep uniform weight"


def test_val_false_warns_that_best_is_not_selected(monkeypatch):
    """`val=False` silently disables model selection, so it must warn loudly.

    With val=False the trainer never sets self.fitness, so `if self.best_fitness == self.fitness:` is
    `None == None` and best.pt is rewritten every epoch — verified identical to last.pt in all 806 tensors
    on a real checkpoint. Nothing in the log said so.
    """
    from ultralytics.models.yolo.s3d.train import Stereo3DDetTrainer

    warnings = []
    monkeypatch.setattr(
        "ultralytics.engine.trainer.LOGGER.warning",
        lambda msg, *a: warnings.append(str(msg) % a if a else str(msg)),
    )
    Stereo3DDetTrainer(overrides={"model": MODEL, "data": DATA, "epochs": 1, "imgsz": [384, 1248], "val": False})
    assert any("val=False disables model selection" in w for w in warnings), f"no warning fired: {warnings}"

    warnings.clear()
    Stereo3DDetTrainer(overrides={"model": MODEL, "data": DATA, "epochs": 1, "imgsz": [384, 1248], "val": True})
    assert not any("val=False" in w for w in warnings), f"warned with val=True: {warnings}"


def test_dfl_variance_uses_the_retargeted_bin_grid():
    """The depth-bin variance must be read on the head's own grid, not a grid rebuilt from the defaults.

    DepthDFL._set_range() retargets the bins per dataset, and this variance is the inverse-variance
    weight for the depth cue against the disparity cue in decode. Evaluating the logits on the default
    2-80 m axis while the head was trained on another silently mis-weights every fused depth.
    """
    import math

    import torch

    from ultralytics.models.yolo.s3d.head import DEPTH_BINS, DEPTH_MAX, DEPTH_MIN, DepthDFL
    from ultralytics.models.yolo.s3d.preprocess import _dfl_variance

    dfl = DepthDFL()
    dfl._set_range(0.2, 2.5)  # a short-range rig (cube_s3d-like), nothing like the KITTI default

    # Equal mass on two adjacent bins => mean at their midpoint => variance is exactly (Δ/2)².
    # -40 not -20: the suppressed bins' residual softmax mass sits at the grid extremes where (b-μ)² is
    # largest, so with a fine grid (64 bins => 62 suppressed) -20 perturbs the variance past this test's
    # tolerance. -40 keeps the two-bin idealisation exact at any bin count.
    logits = torch.full((1, dfl.n_bins, 1), -40.0)
    logits[0, 3, 0] = logits[0, 4, 0] = 0.0
    got = _dfl_variance({"depth_bins": logits, "depth_bin_values": dfl.bin_values}, 0, 0)

    delta = float(dfl.bin_values[4] - dfl.bin_values[3])
    assert got == pytest.approx((delta / 2) ** 2, rel=1e-4)

    # ...and that must NOT match what the old default-grid computation would have returned.
    default_delta = (math.log(DEPTH_MAX) - math.log(DEPTH_MIN)) / (DEPTH_BINS - 1)
    assert got != pytest.approx((default_delta / 2) ** 2, rel=1e-2)

    # Without the grid the caller gets the neutral fallback rather than a confidently wrong number.
    assert _dfl_variance({"depth_bins": logits}, 0, 0) == 1.0


def test_head_publishes_its_depth_bin_grid():
    """The head must ship the grid alongside the logits, or decode silently falls back to variance 1.0."""
    import torch

    model = YOLO(MODEL).model.eval()
    with torch.no_grad():
        _, preds = model(torch.zeros(1, 6, 384, 1248))
    assert "depth_bin_values" in preds, f"head did not publish the bin grid: {sorted(preds)}"
    assert torch.equal(preds["depth_bin_values"], model.model[-1].depth_dfl.bin_values)


def test_depth_channel_width_follows_the_decode_grid():
    """The depth branch must emit exactly one logit per decode bin.

    AUX_SPECS carries a default, but the head's own DepthDFL owns the bin count. If the two are allowed to
    disagree, forward_head's `view(bs, out_c, -1)` reshapes the branch output against the wrong width.
    """
    head = YOLO(MODEL).model.model[-1]
    assert head.aux_specs["depth"] == head.depth_dfl.n_bins
    assert head.aux["depth"][0][-1].out_channels == head.depth_dfl.n_bins


def test_set_depth_mode_preserves_a_nondefault_bin_count():
    """set_depth_mode must filter the head's OWN specs, not rebuild them from module-level AUX_SPECS.

    Rebuilding from the global silently reset a customized depth width back to the default while the built
    branches kept the custom one -- a shape mismatch that no existing test would have caught.
    """
    import torch

    from ultralytics.models.yolo.s3d.head import AUX_SPECS, DepthDFL

    model = YOLO("yolo26n-s3d.yaml").model
    head = model.model[-1]
    n = 24
    assert n != AUX_SPECS["depth"], "pick a bin count that differs from the module default"

    # Rebuild the depth path at a non-default width, exactly as a bin-count experiment would.
    head.depth_dfl = DepthDFL(n, 2.0, 80.0)
    head.aux_specs["depth"] = n
    for i, branch in enumerate(head.aux["depth"]):
        head.aux["depth"][i][-1] = torch.nn.Conv2d(branch[-1].in_channels, n, 1)

    head.set_depth_mode("depth_only")  # the pruning path that used to clobber it

    assert head.aux_specs["depth"] == n, "set_depth_mode reset the depth width to the module default"
    assert "lr_distance" not in head.aux_specs and "lr_distance" not in head.aux  # pruning still works
    head.eval()
    with torch.no_grad():
        _, preds = model(torch.zeros(1, 6, 384, 1248))
    assert preds["depth_bins"].shape[1] == n
    assert preds["depth"].shape[1] == 1  # decode still collapses to a scalar log-depth


def test_set_depth_mode_rejects_an_unknown_mode():
    """Validation must survive folding get_aux_specs into set_depth_mode."""
    head = YOLO("yolo26n-s3d.yaml").model.model[-1]
    with pytest.raises(ValueError, match="Unknown depth_mode"):
        head.set_depth_mode("nonsense")


def test_depth_dfl_loss_is_sized_from_the_head_not_the_module_default():
    """DFLoss must be sized from the head's grid, or a non-default bin count clamps targets to bin 15.

    DFLoss clamps its target to reg_max-1.01 and gathers (tl, tl+1); _depth_bin_loss builds the target as a
    fractional index over the head's own n_bins. A stale reg_max therefore silently discards every target
    above the default range and trains the wrong pair of bins.
    """
    from ultralytics.models.yolo.s3d.head import AUX_SPECS, DepthDFL
    from ultralytics.models.yolo.s3d.loss import Stereo3DDetLoss

    model = YOLO("yolo26n-s3d.yaml").model
    n = 24
    assert n != AUX_SPECS["depth"]
    model.model[-1].depth_dfl = DepthDFL(n, 2.0, 80.0)

    crit = Stereo3DDetLoss(model)
    assert crit.depth_dfl_loss.reg_max == n


def test_fitness_weights_classes_by_gt_instance_count():
    """Early stopping and best.pt must not be dominated by the rarest class.

    s3d `fitness` was an UNWEIGHTED mean AP3D@0.5 Moderate across classes. On the 189-frame screening split
    (680 Car, 81 Pedestrian, 37 Cyclist GT) two thirds of that mean is noise on ~120 objects, and it stopped
    8 calibration runs while Car AP was still climbing 3-4x. Weighting by GT count is approximately
    inverse-variance weighting, since a per-class AP estimate's variance falls about as 1/n.

    Constructed so the two aggregations disagree sharply: Car is good and common, the VRU classes are ~0 and
    rare. Unweighted -> (30+0+0)/3 = 10.0; weighted -> 30*680/798 = 25.56.
    """
    from ultralytics.models.yolo.s3d.metrics import DIFFICULTY_MODERATE, Stereo3DDetMetrics

    m = Stereo3DDetMetrics(names={0: "Car", 1: "Pedestrian", 2: "Cyclist"})
    m.ap3d = {0.5: {DIFFICULTY_MODERATE: {0: 0.30, 1: 0.0, 2: 0.0}}}
    m.gt_counts = {(DIFFICULTY_MODERATE, 0): 680, (DIFFICULTY_MODERATE, 1): 81, (DIFFICULTY_MODERATE, 2): 37}

    assert abs(m.maps3d_50 - 0.10) < 1e-6, "the unweighted mean must stay unweighted for continuity"
    assert abs(m.fitness - 0.30 * 680 / 798) < 1e-6, f"fitness must be instance-weighted, got {m.fitness}"
    assert m.fitness > m.maps3d_50, "with a strong common class, weighting must raise fitness above the mean"

    # A class with no eligible GT must contribute nothing rather than dragging in a zero.
    m.gt_counts[(DIFFICULTY_MODERATE, 1)] = 0
    m.gt_counts[(DIFFICULTY_MODERATE, 2)] = 0
    assert abs(m.fitness - 0.30) < 1e-6, "classes with zero GT must not dilute fitness"

    # No counts recorded at all (nothing processed yet) must fall back, not report a spurious 0.
    m.gt_counts = {}
    assert abs(m.fitness - 0.10) < 1e-6, "with no counts, fitness must fall back to the unweighted mean"


def test_metric_keys_and_results_dict_stay_in_step():
    """Every summary in `results_dict` must also be in `keys`, or it never reaches results.csv.

    The CSV header is built from `keys`; `results_dict` supplies the values. A summary added to only one of
    them is silently dropped from the log — which is what happened to `ap3d_50_weighted`, the new
    instance-weighted fitness signal, on its first run.
    """
    from ultralytics.models.yolo.s3d.metrics import DIFFICULTY_MODERATE, Stereo3DDetMetrics

    m = Stereo3DDetMetrics(names={0: "Car", 1: "Pedestrian", 2: "Cyclist"})
    m.ap3d = {0.5: {DIFFICULTY_MODERATE: {0: 0.3, 1: 0.1, 2: 0.0}}}
    m.gt_counts = {(DIFFICULTY_MODERATE, 0): 680, (DIFFICULTY_MODERATE, 1): 81, (DIFFICULTY_MODERATE, 2): 37}

    # `fitness` is popped by the trainer before logging, so it is exempt.
    missing = {k for k in m.results_dict if k not in set(m.keys) and k != "fitness"}
    assert not missing, f"in results_dict but absent from keys, so dropped from results.csv: {sorted(missing)}"
    assert "ap3d_50_weighted" in m.keys


def test_patience_zero_disables_early_stopping():
    """`patience=0` must mean "never stop", so every A/B arm trains exactly --epochs.

    A/B comparability, not noise, is the reason: an arm that halts at a data-dependent epoch confounds the
    treatment with training length, so a measured difference cannot be attributed. Ultralytics encodes
    "off" as `patience or float("inf")`, which is easy to misread as "stop immediately" — this pins it.
    """
    from ultralytics.utils.torch_utils import EarlyStopping

    off = EarlyStopping(patience=0)
    assert off.patience == float("inf")
    # Fitness that never improves after epoch 0 must still not trigger a stop, however long the run.
    off(0, 1.0)
    assert not any(off(e, 0.0) for e in range(1, 500)), "patience=0 must never stop the run"

    on = EarlyStopping(patience=10)
    on(0, 1.0)
    assert any(on(e, 0.0) for e in range(1, 100)), "a positive patience must still stop on a plateau"


def test_cost_volume_variants_differ_from_the_parent_ONLY_in_level_count():
    """A sweep variant must differ from `yolo26-s3d.yaml` in nothing but `StereoCostVolume` num_levels.

    This is the guard, not a nicety. The cv variants were forked from the parent YAML, and a later commit
    added `depth_bins: 64` to the parent only — so the control built a 64-bin head while every treatment arm
    built a 16-bin one. Since 16 -> 64 bins is worth about +6 points of frac-within-tolerance, on its own
    larger than the effect the sweep was measuring, that would have produced a confident and completely wrong
    conclusion that a finer cost volume is harmful.

    Comparing the full key space rather than a whitelist means the next key added to the parent is caught
    automatically, whatever it is.
    """
    from ultralytics.utils import YAML

    root = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/26"
    parent = YAML.load(root / "yolo26-s3d.yaml")
    for levels in (24, 96, 144):
        child = YAML.load(root / f"yolo26-s3d-cv{levels}.yaml")
        assert set(child) == set(parent), (
            f"cv{levels} key set diverged from the parent: "
            f"only in child {set(child) - set(parent)}, only in parent {set(parent) - set(child)}"
        )
        for key in parent:
            if key in {"head", "backbone"}:
                continue  # the layer lists legitimately differ, and are asserted on the built model below
            assert child[key] == parent[key], f"cv{levels} diverged from the parent at '{key}'"


@pytest.mark.parametrize("tag,levels", [("", 48), ("-cv24", 24), ("-cv96", 96), ("-cv144", 144)])
def test_cost_volume_level_variants_build_with_their_declared_resolution(tag, levels):
    """Each cost-volume resolution variant must build with exactly the level count its YAML declares.

    The sweep exists to test whether a finer disparity grid is LEARNABLE, not merely representable: the
    adjacent depth-bin curve improved to 64 bins and then turned over, being worse than baseline at 256. So
    the variants deliberately bracket the default (24 / 48 / 96 / 144) rather than only going finer.

    A silent fallback to the default level count would make every arm identical and the sweep would report a
    flat, meaningless curve — hence asserting the built module, not just that the file parses.
    """
    from ultralytics.models.yolo.s3d.model import Stereo3DDetModel
    from ultralytics.nn.modules.block import StereoCostVolume

    model = Stereo3DDetModel(f"yolo26n-s3d{tag}.yaml", ch=6, nc=3, verbose=False)
    volumes = [m for m in model.modules() if isinstance(m, StereoCostVolume)]
    assert len(volumes) == 1, f"expected exactly one cost volume, found {len(volumes)}"
    assert int(volumes[0].d_norm.numel()) == levels


def _s3d_yaml_with(tmp_path, **training):
    """Write a yolo26n-s3d YAML whose `training:` block carries the given overrides."""
    import yaml

    from ultralytics.utils import ROOT

    cfg = yaml.safe_load(open(ROOT / "cfg/models/26/yolo26-s3d.yaml"))
    cfg["training"] = {**cfg.get("training", {}), **training}
    cfg["scale"] = "n"
    p = tmp_path / "arm-s3d.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_configure_depth_resizes_the_branch_and_the_loss(tmp_path):
    """A YAML-declared bin count must reach the branch width, the decode grid AND the DFL loss together."""
    from ultralytics.models.yolo.s3d.loss import Stereo3DDetLoss

    model = YOLO(_s3d_yaml_with(tmp_path, depth_bins=64)).model
    head = model.model[-1]
    assert head.depth_dfl.n_bins == 64
    assert head.aux_specs["depth"] == 64
    assert head.aux["depth"][0][-1].out_channels == 64
    assert Stereo3DDetLoss(model).depth_dfl_loss.reg_max == 64
    # The grid must still span the same depth range it was built with.
    assert float(head.depth_dfl.bin_values[0]) == pytest.approx(np.log(2.0), abs=1e-6)
    assert float(head.depth_dfl.bin_values[-1]) == pytest.approx(np.log(80.0), abs=1e-6)


def test_shipped_yamls_default_to_64_depth_bins():
    """The shipped s3d configs must build a 64-bin depth head.

    64 is the measured optimum of an inverted-U bin curve (findings C9/C11/C12): better than 16 at every GT
    range band on the full Chen split, while 192+ is worse than 16. This guards against a silent revert --
    the model summary printed during training reports the pre-`configure_depth` channel count, so a
    regression here would not be visible in any training log.
    """
    from ultralytics.utils import ROOT

    for name in ("yolo26-s3d.yaml", "yolo26-s3d-kitti.yaml"):
        head = YOLO(str(ROOT / "cfg/models/26" / name)).model.model[-1]
        assert head.depth_dfl.n_bins == 64, f"{name} built {head.depth_dfl.n_bins} bins"
        assert head.aux_specs["depth"] == 64
        assert head.aux["depth"][0][-1].out_channels == 64


def test_bin_count_default_is_64_and_overridable(tmp_path):
    """A config that declares nothing gets 64 bins; a config that declares a count gets that count.

    64 is the module-level default (`DEPTH_BINS`), so every s3d model — including variant YAMLs that never
    mention depth bins — inherits the measured optimum. The override path still has to work, because it is
    what the bin sweep itself used and what any future per-dataset tuning needs.
    """
    import yaml

    from ultralytics.models.yolo.s3d.head import AUX_SPECS, DEPTH_BINS
    from ultralytics.utils import ROOT

    assert DEPTH_BINS == 64, "the shipped default is the measured optimum of the bin curve"
    assert AUX_SPECS["depth"] == 64, "AUX_SPECS must track DEPTH_BINS or the branch is sized wrong"

    cfg = yaml.safe_load(open(ROOT / "cfg/models/26/yolo26-s3d.yaml"))
    cfg["training"] = {k: v for k, v in cfg.get("training", {}).items() if k != "depth_bins"}
    cfg["scale"] = "n"
    silent = tmp_path / "silent-s3d.yaml"
    silent.write_text(yaml.safe_dump(cfg))
    head = YOLO(str(silent)).model.model[-1]
    assert head.depth_dfl.n_bins == 64
    assert head.aux["depth"][0][-1].out_channels == 64

    cfg["training"]["depth_bins"] = 16  # the override must still win
    override = tmp_path / "override-s3d.yaml"
    override.write_text(yaml.safe_dump(cfg))
    head = YOLO(str(override)).model.model[-1]
    assert head.depth_dfl.n_bins == 16
    assert head.aux["depth"][0][-1].out_channels == 16


def test_validator_metric_names_survive_a_ddp_wrapped_model(monkeypatch):
    """`get_validator` must read `names` through the parallel wrapper, not off `self.model` directly.

    `BaseTrainer._setup_train` wraps `self.model` in DistributedDataParallel BEFORE calling
    `get_validator()`, and DDP does not forward attribute lookups to the module it wraps. Reading
    `self.model.names` therefore returned None on every multi-GPU run, with two silent consequences:
    results.csv lost all 18 per-class AP3D columns (28 columns instead of 82), and the surviving
    `ap3d_50` summary changed meaning, because `Stereo3DDetMetrics._mean_metric` averages over whichever
    classes appear when `names` is empty rather than over all of them. A full-split DDP run reported
    `ap3d_50` near 29 where the true 3-class mean was 6.96.

    The validator is stubbed: building a real one needs a live dataloader, and the defect under test is
    purely how `get_validator` reaches the model's `names`.
    """
    import torch

    from ultralytics.models.yolo import s3d as s3d_mod
    from ultralytics.models.yolo.s3d.metrics import Stereo3DDetMetrics
    from ultralytics.models.yolo.s3d.train import Stereo3DDetTrainer

    class _StubValidator:
        def __init__(self, *args, **kwargs):
            self.metrics = Stereo3DDetMetrics()

    monkeypatch.setattr(s3d_mod, "Stereo3DDetValidator", _StubValidator)

    names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    trainer = object.__new__(Stereo3DDetTrainer)
    trainer.test_loader, trainer.save_dir, trainer.callbacks = None, Path("."), {}
    trainer.args = SimpleNamespace()

    plain = torch.nn.Linear(2, 2)
    plain.names = names
    trainer.model = plain
    bare_keys = len(trainer.get_validator().metrics.keys)
    assert bare_keys > 20, "single-process baseline should include the per-class AP3D keys"

    class _FakeDDP(torch.nn.Module):
        """Mimics the one property that matters: attributes are not forwarded to `.module`."""

        def __init__(self, module):
            super().__init__()
            self.module = module

    trainer.model = _FakeDDP(plain)
    assert getattr(trainer.model, "names", None) is None, "the wrapper must hide .names, as DDP does"

    wrapped = trainer.get_validator()
    assert wrapped.metrics.names == names, "a DDP-wrapped model must still yield names"
    assert len(wrapped.metrics.keys) == bare_keys, (
        f"DDP run lost metric columns: {len(wrapped.metrics.keys)} vs {bare_keys} single-process"
    )


def test_close_range_boxes_are_rendered():
    """A close-range 3D box must draw: the projection guard is 1/z-safety, not a depth plausibility floor.

    Regression for a hardcoded 2.0 m visualization floor that silently dropped every box from a
    short-baseline rig (cube_s3d objects sit at 0.45-1.66 m), so Results.plot() drew nothing at all.
    """
    from ultralytics.utils.plotting import plot_boxes3d, project_box3d_corners

    calib = {"fx": 645.66, "fy": 645.66, "cx": 480.0, "cy": 300.0, "baseline": 0.063}
    img = np.zeros((600, 960, 3), dtype=np.uint8)
    for z in (0.45, 0.87, 1.66):  # the cube_s3d depth range, all below the old 2.0 m floor
        box = Box3D(
            center_3d=(0.0, 0.0, z),
            dimensions=(0.05, 0.05, 0.10),
            orientation=0.0,
            class_id=0,
            class_label="cubetto",
            confidence=0.9,
        )
        corners = project_box3d_corners(box, calib)
        assert not np.allclose(corners, 0.0), f"z={z} m was dropped by the projection guard"
        assert (plot_boxes3d(img, [box], calib) != img).any(), f"z={z} m drew no pixels"

    # Non-projectable centres must still be rejected rather than producing garbage coordinates.
    # Box3D itself rejects z <= 0, so only the non-finite and sub-epsilon cases reach this guard.
    for bad_z in (float("nan"), float("inf"), 1e-9):
        bad = Box3D(
            center_3d=(0.0, 0.0, bad_z),
            dimensions=(0.05, 0.05, 0.10),
            orientation=0.0,
            class_id=0,
            class_label="cubetto",
            confidence=0.9,
        )
        assert np.allclose(project_box3d_corners(bad, calib), 0.0), f"z={bad_z} should be rejected"


def test_shipped_label_format_actually_loads_objects(tmp_path):
    """The dataset must yield non-zero objects for BOTH shipped label layouts, 18- and 26-value.

    This is the guard that was missing. Every shipped asset — kitti-stereo.zip, the kitti-stereo8 test
    fixture, and the kitti-stereo-chen.zip published from this branch — uses the 26-value layout with 8
    projected-corner values at indices 16-23. When the 26-value branch was deleted from `_parse_labels`,
    every label was rejected with a warning and the loader returned zero objects, yet the whole suite still
    passed: the existing train/val tests only assert that training RUNS, and a run with no supervision runs
    perfectly happily. An 8-arm dose-response then trained 600 epochs per arm on nothing and reported
    AP3D 0.00 across the board.

    So this asserts on the object count, which is the property that was silently violated, rather than on
    the absence of an exception.
    """
    from ultralytics.models.yolo.s3d.dataset import Stereo3DDetDataset

    common = "0 0.54 0.62 0.09 0.27 0.51 0.62 0.09 0.27 4.15 1.73 1.57 1.0 1.75 13.22 1.62"
    corners = "0.49 0.68 0.49 0.76 0.58 0.76 0.56 0.68"
    layouts = {"18-value": f"{common} 0.0 0", "26-value": f"{common} {corners} 0.0 0"}

    for name, line in layouts.items():
        assert len(line.split()) == int(name.split("-")[0]), f"{name} fixture is malformed"
        root = tmp_path / name
        for sub in ("images/train/left", "images/train/right", "labels/train", "calib/train"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        (root / "labels/train/000000.txt").write_text(line + "\n")
        (root / "calib/train/000000.txt").write_text(
            "fx: 721.5377\nfy: 721.5377\ncx: 609.5593\ncy: 172.854\nbaseline: 0.54\n"
            "right_cx: 609.5593\nright_cy: 172.854\nimage_width: 1242\nimage_height: 375\n"
        )
        img = np.zeros((375, 1242, 3), dtype=np.uint8)
        for side in ("left", "right"):
            cv2.imwrite(str(root / f"images/train/{side}/000000.jpg"), img)

        ds = Stereo3DDetDataset(
            root=str(root),
            split="train",
            imgsz=(384, 1248),
            names={0: "Car", 1: "Pedestrian", 2: "Cyclist"},
            mean_dims={"Car": [3.9, 1.6, 1.5], "Pedestrian": [0.8, 0.6, 1.7], "Cyclist": [1.8, 0.6, 1.7]},
            std_dims={"Car": [0.42, 0.10, 0.15], "Pedestrian": [0.20, 0.08, 0.12], "Cyclist": [0.25, 0.10, 0.15]},
            augment=False,
        )
        sample = ds[0]
        assert len(sample["bboxes"]) == 1, f"{name}: loader returned {len(sample['bboxes'])} objects, expected 1"
        assert int(sample["cls"].flatten()[0]) == 0, f"{name}: wrong class id"
        assert abs(float(sample["location_3d"][0][2]) - 13.22) < 1e-3, f"{name}: depth field misread"
