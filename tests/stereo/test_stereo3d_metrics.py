# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unit tests for stereo 3D detection metrics."""

import numpy as np
import pytest
import torch

from ultralytics.data.stereo.box3d import Box3D
from ultralytics.models.yolo.stereo3ddet.metrics import Stereo3DDetMetrics
from ultralytics.utils.metrics import compute_3d_iou


class TestCompute3DIoU:
    """Test suite for compute_3d_iou function."""

    def test_compute_3d_iou_identical_boxes(self):
        """Test that identical boxes have IoU of 1.0."""
        box = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        iou = compute_3d_iou(box, box)
        assert abs(iou - 1.0) < 1e-6, f"Expected IoU=1.0 for identical boxes, got {iou}"

    def test_compute_3d_iou_non_overlapping_boxes(self):
        """Test that non-overlapping boxes have IoU of 0.0."""
        box1 = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        box2 = Box3D(
            center_3d=(100.0, 2.0, 30.0),  # Far away in x
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        iou = compute_3d_iou(box1, box2)
        assert iou == 0.0, f"Expected IoU=0.0 for non-overlapping boxes, got {iou}"

    def test_compute_3d_iou_partially_overlapping_boxes(self):
        """Test IoU calculation for partially overlapping boxes."""
        box1 = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(4.0, 2.0, 2.0),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        box2 = Box3D(
            center_3d=(11.0, 2.0, 30.0),  # Slightly offset
            dimensions=(4.0, 2.0, 2.0),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        iou = compute_3d_iou(box1, box2)
        assert 0.0 < iou < 1.0, f"Expected IoU between 0 and 1, got {iou}"

    def test_compute_3d_iou_with_numpy_array(self):
        """Test compute_3d_iou with numpy array input."""
        box1 = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        # Array format: [x, y, z, l, w, h, orientation]
        box2_array = np.array([10.0, 2.0, 30.0, 3.88, 1.63, 1.53, 0.0])
        iou = compute_3d_iou(box1, box2_array)
        assert abs(iou - 1.0) < 1e-6, f"Expected IoU=1.0 for identical boxes, got {iou}"

    def test_compute_3d_iou_with_rotation(self):
        """Test IoU calculation with rotated boxes."""
        box1 = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(4.0, 2.0, 2.0),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        box2 = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(4.0, 2.0, 2.0),
            orientation=np.pi / 4,  # 45 degree rotation
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        iou = compute_3d_iou(box1, box2)
        assert 0.0 <= iou <= 1.0, f"IoU must be in [0, 1], got {iou}"


class TestStereo3DDetMetrics:
    """Test suite for Stereo3DDetMetrics class."""

    def test_metrics_initialization(self):
        """Test that Stereo3DDetMetrics initializes correctly."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        assert metrics.names == names
        assert metrics.nc == 3
        assert metrics.ap3d_50 == {}
        assert metrics.ap3d_70 == {}
        assert metrics.stats == []

    def test_metrics_update_stats(self):
        """Test that update_stats correctly stores statistics."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)

        stat = {
            "tp": np.array([1, 0, 1]),
            "fp": np.array([0, 1, 0]),
            "conf": np.array([0.9, 0.8, 0.7]),
            "pred_cls": np.array([0, 1, 0]),
            "target_cls": np.array([0, 1, 2]),
            "boxes3d_pred": [None, None, None],  # Placeholder
            "boxes3d_target": [None, None, None],  # Placeholder
        }
        metrics.update_stats(stat)
        assert len(metrics.stats) == 1
        assert metrics.stats[0] == stat

    def test_metrics_maps3d_properties(self):
        """Test that maps3d_50 and maps3d_70 return 0.0 when no metrics computed."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        assert metrics.maps3d_50 == 0.0
        assert metrics.maps3d_70 == 0.0

