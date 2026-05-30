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

    # Output should include aux channels: 4(box) + 3(cls) + 1(lr) + 1(cost_disp) + 3(dims) + 2(orient) + 16(depth) = 30
    out_shape = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    assert out_shape[1] == 30, f"Expected 30 output channels (7 det + 23 aux), got {out_shape[1]}"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="TensorRT requires CUDA")
def test_export_engine():
    """Test TensorRT engine export for s3d model."""
    model = YOLO(MODEL)
    path = model.export(format="engine", imgsz=[384, 1248])
    assert path.endswith(".engine")


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


# =============================================================================
# Phase 1: Cost-volume soft-argmax disparity tests
# =============================================================================


def test_soft_argmax_disparity_correctness():
    """SoftArgmaxDisparity returns the bin center at the correlation peak and is monotone."""
    import torch

    from ultralytics.models.yolo.s3d.head import COST_MAX_DISP, COST_NUM_BINS, SoftArgmaxDisparity

    disparities = torch.linspace(0, COST_MAX_DISP, COST_NUM_BINS).round().int().tolist()
    decoder = SoftArgmaxDisparity(disparities)

    prev = -1.0
    for k in range(COST_NUM_BINS):
        bins = torch.full((1, COST_NUM_BINS, 4), -10.0)
        bins[:, k, :] = 50.0  # sharp peak at bin k → softmax ~ one-hot
        out = decoder(bins)  # [1, 1, 4]
        assert out.shape == (1, 1, 4)
        expected = float(disparities[k])
        assert abs(float(out[0, 0, 0].item()) - expected) < 1e-3, f"bin {k}: {out[0, 0, 0].item()} != {expected}"
        # Strictly increasing as the peak bin index increases (skip duplicate rounded centers).
        if expected > prev:
            prev = expected

    # Softmax weights sum to 1 per location.
    bins = torch.randn(2, COST_NUM_BINS, 5)
    w = bins.softmax(dim=1)
    assert torch.allclose(w.sum(dim=1), torch.ones(2, 5), atol=1e-5)


def test_costvolume_return_bins_backcompat():
    """StereoCostVolume default returns one tensor; return_bins=True returns (refined, raw_bins)."""
    import torch

    from ultralytics.nn.modules.block import StereoCostVolume

    left = torch.randn(2, 32, 16, 24)
    right = torch.randn(2, 32, 16, 24)

    cv = StereoCostVolume(64, 64, 48, 24)
    out = cv((left, right))
    assert isinstance(out, torch.Tensor), "default path must return a single tensor"
    assert out.shape[0] == 2 and out.shape[1] == 64
    assert out.shape[2] == 8 and out.shape[3] == 12  # downsampled by 2

    cv2 = StereoCostVolume(64, 64, 48, 24, refine_layers=2, return_bins=True)
    cv2.load_state_dict(cv.state_dict())
    refined, raw_bins = cv2((left, right))
    assert raw_bins.shape == (2, 24, 16, 24), f"raw bins shape {raw_bins.shape}"
    assert torch.allclose(refined, out, atol=1e-5), "refined map must match default path"


def test_cost_disp_shape_and_export():
    """forward_head produces preds['cost_disp'] [B,1,HW_total]; export grows 29->30, slot after lr_distance."""
    import torch

    from ultralytics.models.yolo.s3d.head import Stereo3DDetHead

    head = Stereo3DDetHead(nc=3, ch=(64, 128, 256, 64))
    head.eval()
    bs = 2
    p3 = torch.randn(bs, 64, 12, 40)
    p4 = torch.randn(bs, 128, 6, 20)
    p5 = torch.randn(bs, 256, 3, 10)
    refined = torch.randn(bs, 64, 12, 40)
    raw_bins = torch.randn(bs, 24, 24, 80)  # cost-volume pre-refine stride

    preds = head.forward_head([p3, p4, p5, (refined, raw_bins)], **head.one2many)
    hw_total = preds["lr_distance"].shape[2]
    assert "cost_disp" in preds
    assert preds["cost_disp"].shape == (bs, 1, hw_total), preds["cost_disp"].shape

    # Export: channel count 30, cost_disp directly after lr_distance.
    head.export = True
    y = head([p3, p4, p5, (refined, raw_bins)])
    head.export = False
    assert y.shape[1] == 30, f"expected 30 channels, got {y.shape[1]}"
    nc, na = 3, hw_total
    # det = 4 box + 3 cls = 7 ; then lr_distance(1), cost_disp(1)
    lr_chan = y[:, 7:8, :]
    cost_chan = y[:, 8:9, :]
    assert torch.allclose(lr_chan, preds["lr_distance"], atol=1e-4)
    assert torch.allclose(cost_chan, preds["cost_disp"], atol=1e-4)


def test_cost_disp_loss_uses_lr_target():
    """cost_disp loss uses the lr_distance GT and populates index 6 of a length-7 loss vector."""
    import torch

    from ultralytics.models.yolo.s3d.loss import Stereo3DDetLoss

    model = YOLO(MODEL).model
    crit = Stereo3DDetLoss(model, loss_weights={"cost_disp": 2.0, "lr_distance": 2.0})
    crit.device = torch.device("cpu")

    bs, hw = 1, 30
    fg_mask = torch.zeros(bs, hw, dtype=torch.bool)
    fg_mask[0, :3] = True
    target_gt_idx = torch.zeros(bs, hw, dtype=torch.int64)
    aux_preds = {
        "lr_distance": torch.randn(bs, 1, hw),
        "cost_disp": torch.randn(bs, 1, hw),
    }
    batch = {"aux_targets": {"lr_distance": torch.randn(bs, 2, 1)}}
    aux_losses = crit._compute_aux_losses(aux_preds, batch, target_gt_idx, fg_mask)
    assert "cost_disp" in aux_losses
    assert torch.isfinite(aux_losses["cost_disp"]) and aux_losses["cost_disp"] > 0

    # cost_disp loss equals _aux_loss against lr_distance GT.
    ref = crit._aux_loss(
        aux_preds["cost_disp"], batch["aux_targets"]["lr_distance"], target_gt_idx, fg_mask, None
    )
    assert torch.allclose(aux_losses["cost_disp"], ref, atol=1e-6)

    # Loss vector has length 7.
    assert crit._loss_vec_len() == 7


def test_decode_cost_disp_primary():
    """cost_disp is the primary z-source; absent -> lr_distance; both absent -> depth fallback."""
    import torch

    from ultralytics.models.yolo.s3d import preprocess as pp

    fx, baseline = 700.0, 0.54
    input_w = 1248
    calib = [{"fx": fx, "fy": fx, "cx": 600.0, "cy": 180.0, "baseline": baseline}]
    imgsz = (384, input_w)
    ori_shapes = [(384, input_w)]  # letterbox_scale == 1

    # Build a single synthetic detection at a known flat index.
    hw = (input_w // 8) * (384 // 8)
    det = torch.zeros(1, 7, hw)
    flat = 5
    det[0, 0, flat] = 600.0  # x1
    det[0, 1, flat] = 180.0
    det[0, 2, flat] = 640.0
    det[0, 3, flat] = 220.0
    det[0, 4, flat] = 0.9  # class score (high)

    # cost_disp encodes disparity 100 px (z = fx*b/disp).
    disp_cost = 100.0
    cost_log = float(np.log(disp_cost / input_w))
    # lr_distance encodes a conflicting disparity (50 px).
    lr_log = float(np.log(50.0 / input_w))

    outputs = {
        "det": det,
        "lr_distance": torch.full((1, 1, hw), lr_log),
        "cost_disp": torch.full((1, 1, hw), cost_log),
        "depth": torch.full((1, 1, hw), float(np.log(7.0))),  # conflicting monocular z
        "dimensions": torch.zeros(1, 3, hw),
        "orientation": torch.zeros(1, 2, hw),
    }
    assert pp.MONO_BLEND == 0.0

    boxes = pp.decode_stereo3d_outputs(
        outputs, conf_threshold=0.1, calib=calib, imgsz=imgsz, ori_shapes=ori_shapes
    )
    assert len(boxes) == 1
    z = boxes[0].center_3d[2]
    z_expected = (fx * baseline) / disp_cost
    assert abs(z - z_expected) < 0.5, f"cost_disp primary: z={z} expected {z_expected}"

    # Without cost_disp -> falls back to lr_distance.
    outputs.pop("cost_disp")
    boxes = pp.decode_stereo3d_outputs(
        outputs, conf_threshold=0.1, calib=calib, imgsz=imgsz, ori_shapes=ori_shapes
    )
    z = boxes[0].center_3d[2]
    z_lr = (fx * baseline) / 50.0
    assert abs(z - z_lr) < 0.5, f"lr fallback: z={z} expected {z_lr}"

    # Without cost_disp and lr_distance -> monocular depth fallback.
    outputs.pop("lr_distance")
    boxes = pp.decode_stereo3d_outputs(
        outputs, conf_threshold=0.1, calib=calib, imgsz=imgsz, ori_shapes=ori_shapes
    )
    z = boxes[0].center_3d[2]
    assert abs(z - 7.0) < 0.5, f"mono fallback: z={z} expected 7.0"


def test_smoke_train_one_epoch_cost_disp():
    """1-epoch CPU train+val on the mini dataset; cost_disp loss logged, no NaN/Inf."""
    import math as _m

    model = YOLO(MODEL)
    results = model.train(data=DATA, epochs=1, imgsz=[384, 1248], batch=2, val=True, plots=False)
    # The cost_disp loss term must be tracked by the trainer (logged alongside the others).
    assert "cost_disp" in model.trainer.loss_names, model.trainer.loss_names
    # Training must complete without NaN/Inf in the recorded per-epoch losses.
    csv = model.trainer.save_dir / "results.csv"
    assert csv.exists()
    import csv as _csv

    with open(csv) as f:
        rows = list(_csv.DictReader(f))
    assert rows, "no results logged"
    cost_cols = [k for k in rows[0] if "cost_disp" in k]
    assert cost_cols, f"cost_disp not in results columns: {list(rows[0])}"
    for row in rows:
        for k, v in row.items():
            if v not in ("", None):
                try:
                    fv = float(v)
                except ValueError:
                    continue
                assert _m.isfinite(fv), f"{k}={v}"
