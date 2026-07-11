# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tests for Stereo 3D Detection (s3d) task."""

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

    # Output aux channels: 4(box) + 3(cls) + 1(lr) + 3(dims) + 6(orient MultiBin) + 16(depth) = 33
    out_shape = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    assert out_shape[1] == 33, f"Expected 33 output channels (7 det + 26 aux), got {out_shape[1]}"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="TensorRT requires CUDA")
def test_export_engine():
    """Test TensorRT engine export for s3d model."""
    model = YOLO(MODEL)
    path = model.export(format="engine", imgsz=[384, 1248])
    assert path.endswith(".engine")


def test_feature_flags_gate_head_branches():
    """use_proj_center / use_depth_uncertainty in YAML training block add head branches; default off is unchanged."""
    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS

    base = YOLO(MODEL).model
    head = base.model[-1]
    assert "proj_offset" not in head.aux_specs
    assert head.aux["lr_distance"][0][-1].out_channels == 1  # scalar disparity
    assert getattr(head, "use_uncertainty", False) is False

    # Enable both features directly (model.py wires these from the YAML training block).
    head.enable_proj_center()
    head.enable_depth_uncertainty()
    assert head.aux_specs["proj_offset"] == 2
    assert head.aux["proj_offset"][0][-1].out_channels == 2
    assert head.aux["lr_distance"][0][-1].out_channels == 2  # value + log-variance
    assert head.use_uncertainty is True
    # orientation/depth untouched
    assert head.aux_specs["orientation"] == ORIENT_CHANNELS


def test_proj_center_loss_present():
    """When use_proj_center is set, the aux-loss dict gains a smooth-L1 proj_center term."""
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.s3d.loss import Stereo3DDetLoss

    model = YOLO("yolo26n-s3d.yaml").model
    model.model[-1].enable_proj_center()
    crit = Stereo3DDetLoss(model, loss_weights={"proj_center": 1.0}, use_proj_center=True)
    B, HW = 2, 5
    preds = {"proj_offset": torch.zeros(B, 2, HW)}
    gt = torch.ones(B, 3, 2)  # [B, max_n, 2]
    idx = torch.zeros(B, HW, dtype=torch.long)
    fg = torch.ones(B, HW, dtype=torch.bool)
    losses = crit._compute_aux_losses(preds, {"aux_targets": {"proj_offset": gt}}, idx, fg)
    assert "proj_center" in losses and losses["proj_center"] > 0


def test_lr_nll_attenuates_with_uncertainty():
    """Laplacian NLL: for a fixed residual, a larger predicted log-variance lowers the loss
    (attenuation), but the log-variance penalty prevents collapse — loss is convex in logvar."""
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
    """Integration test: with use_uncertainty=True and lr_logvar present, _compute_aux_losses
    routes lr_distance through the Laplacian-NLL gather-wiring in _lr_nll_loss, not smooth-L1."""
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.s3d.loss import Stereo3DDetLoss

    model = YOLO("yolo26n-s3d.yaml").model
    model.model[-1].enable_depth_uncertainty()
    crit = Stereo3DDetLoss(model, loss_weights={"lr_distance": 1.0}, use_uncertainty=True)

    B, HW = 2, 5
    val = torch.ones(B, 1, HW)  # lr_distance prediction
    logvar = torch.full((B, 1, HW), 5.0)  # large predicted log-variance (attenuation)
    preds = {"lr_distance": val, "lr_logvar": logvar}
    gt = torch.full((B, 3, 1), 3.0)  # [B, max_n, 1] — residual of 2.0 for every anchor
    idx = torch.zeros(B, HW, dtype=torch.long)
    fg = torch.ones(B, HW, dtype=torch.bool)

    losses = crit._compute_aux_losses(preds, {"aux_targets": {"lr_distance": gt}}, idx, fg)
    assert "lr_distance" in losses
    nll_loss = losses["lr_distance"]
    assert torch.isfinite(nll_loss)
    assert nll_loss >= 0

    # With use_uncertainty=False the same inputs fall back to plain smooth-L1 (lr_logvar
    # ignored). If the NLL routing in _compute_aux_losses were reverted/broken, nll_loss above
    # would collapse to this same value instead of the attenuated NLL value.
    crit_no_unc = Stereo3DDetLoss(model, loss_weights={"lr_distance": 1.0}, use_uncertainty=False)
    smooth_l1_loss = crit_no_unc._compute_aux_losses(
        preds, {"aux_targets": {"lr_distance": gt}}, idx, fg
    )["lr_distance"]

    assert not torch.isclose(nll_loss, smooth_l1_loss), (
        "lr_distance loss should differ from plain smooth-L1, proving it went through the NLL branch"
    )


def test_3d_iou():
    """Test 3D IoU computation: identical, no overlap, and partial overlap."""
    box = Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0,
                class_label="Car", class_id=0, confidence=0.95)
    assert abs(compute_3d_iou(box, box) - 1.0) < 1e-6

    far_box = Box3D(center_3d=(100.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0,
                    class_label="Car", class_id=0, confidence=0.95)
    assert compute_3d_iou(box, far_box) == 0.0

    near_box = Box3D(center_3d=(11.0, 2.0, 30.0), dimensions=(4.0, 2.0, 2.0), orientation=0.0,
                     class_label="Car", class_id=0, confidence=0.95)
    box2 = Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(4.0, 2.0, 2.0), orientation=0.0,
                 class_label="Car", class_id=0, confidence=0.95)
    assert 0.0 < compute_3d_iou(box2, near_box) < 1.0


def test_3d_iou_rotated_45deg():
    """Two identical square-footprint boxes offset by 45 deg of yaw.

    True 3D IoU of a unit square and the same square rotated 45 deg (shared
    center/dims) is exactly 1/sqrt(2) ~= 0.7071. The old axis-aligned-bbox
    approximation returns ~1.0 here because the 45 deg box's AABB fully contains
    the other box, so this case discriminates true rotated IoU from the AABB hack.
    """
    a = Box3D(center_3d=(0.0, 0.0, 10.0), dimensions=(2.0, 2.0, 2.0), orientation=0.0,
              class_label="Car", class_id=0, confidence=0.95)
    b = Box3D(center_3d=(0.0, 0.0, 10.0), dimensions=(2.0, 2.0, 2.0), orientation=np.pi / 4,
              class_label="Car", class_id=0, confidence=0.95)
    assert abs(compute_3d_iou(a, b) - (1.0 / np.sqrt(2))) < 1e-3


def test_3d_iou_rotated_no_overlap():
    """Two 45 deg boxes whose AABBs overlap but whose true rotated footprints do not.

    Both boxes are unit-ish square footprints rotated 45 deg (diamonds), offset
    diagonally by (2, 2) in the x-z plane. The diamonds are disjoint (L1 center
    distance 4 > 2*sqrt(2)), so true 3D IoU is 0. But their axis-aligned bounding
    boxes (side 2*sqrt(2)) still overlap, so the old AABB approximation reports a
    spurious positive IoU.
    """
    a = Box3D(center_3d=(0.0, 0.0, 10.0), dimensions=(2.0, 2.0, 2.0), orientation=np.pi / 4,
              class_label="Car", class_id=0, confidence=0.95)
    b = Box3D(center_3d=(2.0, 0.0, 12.0), dimensions=(2.0, 2.0, 2.0), orientation=np.pi / 4,
              class_label="Car", class_id=0, confidence=0.95)
    assert compute_3d_iou(a, b) == 0.0


def test_3d_iou_rotated_90deg():
    """Two identical (L=4, W=2) boxes offset by 90 deg of yaw, shared center/dims.

    The rot-0 footprint is 2x4 and the rot-90 footprint is 4x2; their BEV
    intersection is the 2x2 square, giving true 3D IoU = 4 / (16 - 4) = 1/3.
    The old AABB approximation returns 1.0 here (the 90 deg box's AABB grows),
    so this is a clean analytic regression guard.
    """
    a = Box3D(center_3d=(0.0, 0.0, 20.0), dimensions=(4.0, 2.0, 2.0), orientation=0.0,
              class_label="Car", class_id=0, confidence=0.95)
    b = Box3D(center_3d=(0.0, 0.0, 20.0), dimensions=(4.0, 2.0, 2.0), orientation=np.pi / 2,
              class_label="Car", class_id=0, confidence=0.95)
    assert abs(compute_3d_iou(a, b) - 1.0 / 3.0) < 1e-3


def test_bev_iou_ignores_height():
    """BEV IoU uses only the ground-plane footprint; height offset must not reduce it.

    Two boxes with an identical footprint but stacked vertically (full height
    offset) share no 3D volume (3D IoU ~= 0) yet have identical bird's-eye-view
    footprints (BEV IoU == 1.0). This pins BEV down as a distinct metric.
    """
    low = Box3D(center_3d=(0.0, 0.0, 15.0), dimensions=(4.0, 1.8, 1.6), orientation=0.3,
                class_label="Car", class_id=0, confidence=0.9)
    high = Box3D(center_3d=(0.0, -1.6, 15.0), dimensions=(4.0, 1.8, 1.6), orientation=0.3,
                 class_label="Car", class_id=0, confidence=0.9)
    assert abs(compute_bev_iou(low, high) - 1.0) < 1e-6
    assert compute_3d_iou(low, high) < 0.05  # stacked: ~no vertical overlap

    # BEV of the 45deg square case equals its 3D IoU (heights coincide): 1/sqrt(2).
    a = Box3D(center_3d=(0.0, 0.0, 10.0), dimensions=(2.0, 2.0, 2.0), orientation=0.0,
              class_label="Car", class_id=0, confidence=0.9)
    b = Box3D(center_3d=(0.0, 0.0, 10.0), dimensions=(2.0, 2.0, 2.0), orientation=np.pi / 4,
              class_label="Car", class_id=0, confidence=0.9)
    assert abs(compute_bev_iou(a, b) - 1.0 / np.sqrt(2)) < 1e-3


def _single_image_stat(gt_ori, pred_ori):
    """Build a one-image, one-Car stat: a perfectly localised pred at given headings."""
    gt = Box3D(center_3d=(0.0, 0.0, 10.0), dimensions=(4.0, 1.6, 1.5), orientation=gt_ori,
               class_label="Car", class_id=0, confidence=1.0, truncated=0.0, occluded=0)
    pred = Box3D(center_3d=(0.0, 0.0, 10.0), dimensions=(4.0, 1.6, 1.5), orientation=pred_ori,
                 class_label="Car", class_id=0, confidence=0.9)
    return {
        "gt_boxes": [gt],
        "pred_boxes": [pred],
        "iou_matrix": np.array([[1.0]]),       # perfect 3D localisation (given)
        "bev_iou_matrix": np.array([[1.0]]),   # perfect BEV localisation (given)
        "gt_difficulties": np.array([0]),      # Easy
        "pred_heights_2d": np.array([50.0]),   # above 25px min
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

    # Flipped heading (pi): still a localisation TP (AP3D=1.0) but AOS collapses to 0.
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
    """Dimension encode->decode round-trip must recover GT dims (regression for the
    string-vs-int keyed mean/std bug that inflated 3D box dimensions / length).

    The dataset encodes the SmoothL1 dimension target with compute_dimension_offset(),
    looking up per-class mean/std priors by INTEGER class_id. The dataset YAML keys those
    priors by class NAME ("Car"). If the dataset does not rekey name->int, every lookup
    misses and falls back to a generic mean/std, so the trained target is huge and the
    decoder (which uses correct int-keyed (H,W,L) priors) mis-expands dimensions.
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

    This is the guard against the encode/decode mismatch class of bug (cf. the
    dimension-prior bug). The decoder argmaxes the (one-hot) confidence and adds the
    residual to the chosen bin center; it must invert the encoder exactly.
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

    A single sin/cos regressor that smears toward 0 cannot distinguish front/back;
    MultiBin assigns alpha and alpha+pi to different argmax bins.
    """
    from ultralytics.models.yolo.s3d.orientation import NUM_ORIENT_BINS, encode_orientation

    enc_a = encode_orientation(0.2)
    enc_b = encode_orientation(0.2 + 3.14159)  # ~180 deg apart
    bin_a = max(range(NUM_ORIENT_BINS), key=lambda i: enc_a[i])
    bin_b = max(range(NUM_ORIENT_BINS), key=lambda i: enc_b[i])
    assert bin_a != bin_b, f"180-deg-apart headings must differ in bin: {bin_a} vs {bin_b}"


def test_depth_decode_imgsz_invariant():
    """Stereo disparity->depth decode must recover true metric Z independent of imgsz.

    The dataset encodes the lr_distance target as log(disparity) in letterbox-NORMALIZED
    coordinates (dataset.py: "Normalized xywh (letterboxed input space)", disparity_norm =
    cx - right_cx). decode_stereo3d_outputs must invert that back to the same metric Z whether
    the letterbox is aspect-preserving (384x1248 on KITTI 375x1242, scale~=1) or square
    (640x640 / 384x384, scale<1). This is the regression guard for the bug where the decode
    scaled focal length by 1/letterbox_scale, yielding z_from_disp = Z_true / scale — correct
    only at scale~=1 and silently inflating stereo depth ~1.4-3.2x under a square imgsz.

    The direct-depth head (log(z_3d) target) is imgsz-invariant by construction and serves as
    the control: it must round-trip at every imgsz.
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
    scale, pad_left, _ = compute_letterbox_params(*ori_hw, imgsz)
    z = 20.0
    du = 0.01

    def make_outputs():
        # non_max_suppression converts outputs["det"] xywh->xyxy in place (via a transposed
        # view aliasing the same storage), so each decode call needs its own fresh det tensor.
        det = torch.zeros(1, 4 + nc, 1)
        det[0, :4, 0] = torch.tensor([input_w / 2, input_h / 2, 20.0, 20.0])
        det[0, 4, 0] = 0.99
        return {
            "det": det,
            "dimensions": torch.zeros(1, 3, 1),
            "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
            "depth": torch.tensor([[[math.log(z)]]]),
            "proj_offset": torch.tensor([[[du], [0.0]]]).float(),
        }

    # bs=1 -> decode_stereo3d_outputs returns a flat list[Box3D] (unwrapped), so index once.
    x_off = decode_stereo3d_outputs(
        make_outputs(), calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw], use_proj_center=True
    )[0].center_3d[0]
    x_no = decode_stereo3d_outputs(
        make_outputs(), calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw], use_proj_center=False
    )[0].center_3d[0]
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

    def make_outputs():
        # non_max_suppression converts outputs["det"] xywh->xyxy in place (via a transposed
        # view aliasing the same storage), so each decode call needs its own fresh det tensor.
        det = torch.zeros(1, 4 + nc, 1)
        det[0, :4, 0] = torch.tensor([624.0, 192.0, 20.0, 20.0])
        det[0, 4, 0] = 0.99
        # disparity cue and direct cue encode different depths so the mean is nontrivial.
        return {
            "det": det,
            "dimensions": torch.zeros(1, 3, 1),
            "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
            "lr_distance": torch.tensor([[[math.log(0.03)]]]),
            "depth": torch.tensor([[[math.log(25.0)]]]),
            "lr_logvar": torch.tensor([[[0.0]]]),
        }

    # bs=1 -> decode_stereo3d_outputs returns a flat list[Box3D] (unwrapped), so index once.
    z_geo = decode_stereo3d_outputs(
        make_outputs(), calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw], ivw_fusion=False
    )[0].center_3d[2]
    z_ivw = decode_stereo3d_outputs(
        make_outputs(), calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw], ivw_fusion=True
    )[0].center_3d[2]
    # equal-variance IVW reduces to the geometric mean
    assert abs(z_ivw - z_geo) < 1e-2, f"ivw {z_ivw} != geomean {z_geo}"


def test_score_weight_demotes_uncertain():
    """score_weight multiplies confidence by exp(-k*sigma): higher lr_logvar => lower final score."""
    import math

    import torch

    from ultralytics.models.yolo.s3d.orientation import ORIENT_CHANNELS, encode_orientation
    from ultralytics.models.yolo.s3d.preprocess import decode_stereo3d_outputs

    calib = {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.8540, "baseline": 0.54}
    ori_hw = (375, 1242)
    imgsz = (384, 1248)
    nc = 3

    def conf(logvar):
        det = torch.zeros(1, 4 + nc, 1)
        det[0, :4, 0] = torch.tensor([624.0, 192.0, 20.0, 20.0])
        det[0, 4, 0] = 0.9
        outputs = {
            "det": det,
            "dimensions": torch.zeros(1, 3, 1),
            "orientation": torch.tensor(encode_orientation(0.0)).view(1, ORIENT_CHANNELS, 1).float(),
            "lr_distance": torch.tensor([[[math.log(0.03)]]]),
            "depth": torch.tensor([[[math.log(25.0)]]]),
            "lr_logvar": torch.tensor([[[logvar]]]),
        }
        # bs=1 -> decode_stereo3d_outputs returns a flat list[Box3D] (unwrapped), so index once.
        return decode_stereo3d_outputs(
            outputs, calib=[calib], imgsz=imgsz, ori_shapes=[ori_hw], score_weight=True, score_k=0.5
        )[0].confidence

    assert conf(4.0) < conf(0.0), "higher uncertainty must lower the score"


def test_proj_offset_roundtrip():
    """Projected-centroid offset must invert: encode (centroid->projected px->offset) then
    decode (box_center+offset -> back-project at true z) recovers the centroid X/Y."""
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
