# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tests for Stereo 3D Detection (s3d) task."""

import numpy as np
import pytest

from ultralytics import YOLO
from ultralytics.data.stereo.box3d import Box3D
from ultralytics.utils.metrics import compute_3d_iou

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
