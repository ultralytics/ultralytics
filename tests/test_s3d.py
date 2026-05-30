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

    # Output should include aux channels: 4(box) + 3(cls) + 1(lr) + 3(dims) + 2(orient) + 16(depth) = 29
    out_shape = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    assert out_shape[1] == 29, f"Expected 29 output channels (7 det + 22 aux), got {out_shape[1]}"


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
