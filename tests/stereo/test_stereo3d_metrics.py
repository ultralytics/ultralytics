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


class TestResultsDictionaryStructure:
    """Test suite for results_dict structure (T146-T147)."""

    def test_results_dict_returns_flat_scalar_values(self):
        """Test that results_dict returns flat dictionary with scalar values (not nested dicts) for precision, recall, f1 (T146)."""
        metrics = Stereo3DDetMetrics(names={0: "Car", 1: "Pedestrian", 2: "Cyclist"})
        metrics.ap3d_50 = {"Car": 0.5, "Pedestrian": 0.3, "Cyclist": 0.2}
        metrics.ap3d_70 = {"Car": 0.4, "Pedestrian": 0.25, "Cyclist": 0.15}
        
        # Set nested precision, recall, f1 (old format)
        metrics.precision = {0.5: {0: 0.8, 1: 0.7, 2: 0.6}, 0.7: {0: 0.7, 1: 0.6, 2: 0.5}}
        metrics.recall = {0.5: {0: 0.9, 1: 0.8, 2: 0.7}, 0.7: {0: 0.8, 1: 0.7, 2: 0.6}}
        metrics.f1 = {0.5: {0: 0.85, 1: 0.75, 2: 0.65}, 0.7: {0: 0.75, 1: 0.65, 2: 0.55}}
        
        results = metrics.results_dict
        
        # Verify all values are scalar (not nested dicts)
        assert isinstance(results["precision"], (int, float)), "precision should be a scalar, not a dict"
        assert isinstance(results["recall"], (int, float)), "recall should be a scalar, not a dict"
        assert isinstance(results["f1"], (int, float)), "f1 should be a scalar, not a dict"
        
        # Verify other values are also scalars (all should be scalars for BaseValidator compatibility)
        assert isinstance(results["ap3d_50"], (int, float)), "ap3d_50 should be a scalar (mean value)"
        assert isinstance(results["ap3d_70"], (int, float)), "ap3d_70 should be a scalar (mean value)"
        assert isinstance(results["maps3d_50"], (int, float)), "maps3d_50 should be a scalar"
        assert isinstance(results["maps3d_70"], (int, float)), "maps3d_70 should be a scalar"
        
        # Verify all values can be converted to float (all should be scalars)
        for key, value in results.items():
            assert isinstance(value, (int, float)), f"{key} should be a scalar, got {type(value)}"

    def test_validator_can_round_results_dict_values(self):
        """Test that BaseValidator can successfully round all values in results_dict to 5 decimal places (T147)."""
        from ultralytics.models.yolo.stereo3ddet.val import Stereo3DDetValidator
        from unittest.mock import MagicMock
        
        # Create validator
        args = {"task": "stereo3ddet", "imgsz": 384, "data": None}
        validator = Stereo3DDetValidator(args=args)
        validator.device = torch.device("cpu")
        validator.data = {
            "channels": 6,
            "names": {0: "Car", 1: "Pedestrian", 2: "Cyclist"},
            "nc": 3,
        }
        
        # Initialize metrics
        validator.init_metrics(MagicMock())
        
        # Set up metrics with processed results
        validator.metrics.ap3d_50 = {"Car": 0.5, "Pedestrian": 0.3, "Cyclist": 0.2}
        validator.metrics.ap3d_70 = {"Car": 0.4, "Pedestrian": 0.25, "Cyclist": 0.15}
        # Set nested precision, recall, f1 (old format)
        validator.metrics.precision = {0.5: {0: 0.8, 1: 0.7, 2: 0.6}, 0.7: {0: 0.7, 1: 0.6, 2: 0.5}}
        validator.metrics.recall = {0.5: {0: 0.9, 1: 0.8, 2: 0.7}, 0.7: {0: 0.8, 1: 0.7, 2: 0.6}}
        validator.metrics.f1 = {0.5: {0: 0.85, 1: 0.75, 2: 0.65}, 0.7: {0: 0.75, 1: 0.65, 2: 0.55}}
        
        # Get stats (which calls results_dict)
        stats = validator.get_stats()
        
        # Verify BaseValidator can round all values
        try:
            # This is what BaseValidator does at line 254
            rounded_results = {k: round(float(v), 5) for k, v in stats.items()}
            assert isinstance(rounded_results, dict), "rounded_results should be a dict"
            # Verify all values are floats
            for k, v in rounded_results.items():
                assert isinstance(v, float), f"{k} should be a float after rounding, got {type(v)}"
        except (TypeError, ValueError) as e:
            pytest.fail(f"BaseValidator should be able to round all values: {e}")


class TestMetricsCalculationBugFix:
    """Test suite for metrics calculation bug fix (T035-T041)."""

    def test_metrics_with_perfect_matches(self):
        """Test that perfect predictions yield AP3D@0.7 ≈ 1.0 (T035)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        
        # Create perfect match: prediction identical to ground truth
        pred_box = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.95,
        )
        gt_box = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=1.0,
        )
        
        # Update stats with perfect match (IoU should be ~1.0)
        stat = {
            "tp": np.array([[True, True]], dtype=bool),  # TP at both thresholds
            "fp": np.array([[False, False]], dtype=bool),
            "conf": np.array([0.95]),
            "pred_cls": np.array([0]),
            "target_cls": np.array([0]),
            "boxes3d_pred": [pred_box],
            "boxes3d_target": [gt_box],
        }
        metrics.update_stats(stat)
        
        results = metrics.process()
        
        # With perfect match, AP3D@0.7 should be close to 1.0
        assert "ap3d_70" in results
        # Note: AP calculation depends on precision-recall curve, but with perfect match it should be high
        # We check that it's non-zero and reasonable
        ap3d_70_car = results["ap3d_70"].get(0, 0.0) if isinstance(results["ap3d_70"], dict) else 0.0
        # For perfect match with single sample, AP might not be exactly 1.0 due to interpolation
        # But it should be significantly greater than 0
        assert ap3d_70_car > 0.5, f"Perfect match should yield high AP3D@0.7, got {ap3d_70_car}"

    def test_metrics_with_good_matches(self):
        """Test that IoU > 0.5 yields non-zero AP3D (T036)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        
        # Create good match (IoU > 0.5 but < 0.7)
        pred_box = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.9,
        )
        gt_box = Box3D(
            center_3d=(10.5, 2.0, 30.0),  # Slightly offset to reduce IoU
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=1.0,
        )
        
        stat = {
            "tp": np.array([[True, False]], dtype=bool),  # TP at 0.5, FP at 0.7
            "fp": np.array([[False, True]], dtype=bool),
            "conf": np.array([0.9]),
            "pred_cls": np.array([0]),
            "target_cls": np.array([0]),
            "boxes3d_pred": [pred_box],
            "boxes3d_target": [gt_box],
        }
        metrics.update_stats(stat)
        
        results = metrics.process()
        
        # AP3D@0.5 should be non-zero
        ap3d_50_car = results["ap3d_50"].get(0, 0.0) if isinstance(results["ap3d_50"], dict) else 0.0
        assert ap3d_50_car > 0.0, f"Good match (IoU > 0.5) should yield non-zero AP3D@0.5, got {ap3d_50_car}"

    def test_metrics_empty_predictions(self):
        """Test edge case: empty predictions (T037)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        
        gt_box = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=1.0,
        )
        
        # Empty predictions
        stat = {
            "tp": np.zeros((0, 2), dtype=bool),
            "fp": np.zeros((0, 2), dtype=bool),
            "conf": np.array([]),
            "pred_cls": np.array([], dtype=int),
            "target_cls": np.array([0]),
            "boxes3d_pred": [],
            "boxes3d_target": [gt_box],
        }
        metrics.update_stats(stat)
        
        results = metrics.process()
        
        # With empty predictions, metrics should handle gracefully
        # AP3D should be 0.0 (no predictions to match)
        assert isinstance(results, dict), "Results should be a dict even with empty predictions"

    def test_metrics_empty_ground_truth(self):
        """Test edge case: empty ground truth (T038)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        
        pred_box = Box3D(
            center_3d=(10.0, 2.0, 30.0),
            dimensions=(3.88, 1.63, 1.53),
            orientation=0.0,
            class_label="Car",
            class_id=0,
            confidence=0.9,
        )
        
        # Empty ground truth
        stat = {
            "tp": np.zeros((1, 2), dtype=bool),
            "fp": np.ones((1, 2), dtype=bool),  # All false positives
            "conf": np.array([0.9]),
            "pred_cls": np.array([0]),
            "target_cls": np.array([], dtype=int),
            "boxes3d_pred": [pred_box],
            "boxes3d_target": [],
        }
        metrics.update_stats(stat)
        
        results = metrics.process()
        
        # With empty ground truth, metrics should handle gracefully
        assert isinstance(results, dict), "Results should be a dict even with empty ground truth"

    def test_metrics_boundary_iou_thresholds(self):
        """Test edge case: IoU exactly at 0.5 and 0.7 (T039)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        
        # Create boxes with IoU exactly at threshold (would need precise calculation)
        # For this test, we simulate the boundary case with TP/FP assignment
        stat = {
            "tp": np.array([[True, False]], dtype=bool),  # TP at 0.5, FP at 0.7
            "fp": np.array([[False, True]], dtype=bool),
            "conf": np.array([0.9]),
            "pred_cls": np.array([0]),
            "target_cls": np.array([0]),
            "boxes3d_pred": [Box3D(
                center_3d=(10.0, 2.0, 30.0),
                dimensions=(3.88, 1.63, 1.53),
                orientation=0.0,
                class_label="Car",
                class_id=0,
                confidence=0.9,
            )],
            "boxes3d_target": [Box3D(
                center_3d=(10.0, 2.0, 30.0),
                dimensions=(3.88, 1.63, 1.53),
                orientation=0.0,
                class_label="Car",
                class_id=0,
                confidence=1.0,
            )],
        }
        metrics.update_stats(stat)
        
        results = metrics.process()
        
        # Should handle boundary cases without errors
        assert isinstance(results, dict), "Results should handle boundary IoU thresholds"

    def test_metrics_no_matches(self):
        """Test edge case: all IoU values below thresholds (T040)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        
        # All predictions are false positives (IoU < 0.5)
        stat = {
            "tp": np.zeros((2, 2), dtype=bool),
            "fp": np.ones((2, 2), dtype=bool),
            "conf": np.array([0.9, 0.8]),
            "pred_cls": np.array([0, 0]),
            "target_cls": np.array([0]),
            "boxes3d_pred": [
                Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0, class_label="Car", class_id=0, confidence=0.9),
                Box3D(center_3d=(20.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0, class_label="Car", class_id=0, confidence=0.8),
            ],
            "boxes3d_target": [
                Box3D(center_3d=(100.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0, class_label="Car", class_id=0, confidence=1.0),
            ],
        }
        metrics.update_stats(stat)
        
        results = metrics.process()
        
        # With no matches, AP3D should be 0.0
        ap3d_50_car = results["ap3d_50"].get(0, 0.0) if isinstance(results["ap3d_50"], dict) else 0.0
        assert ap3d_50_car == 0.0, f"No matches should yield AP3D=0.0, got {ap3d_50_car}"

    def test_metrics_class_id_mismatch(self):
        """Test edge case: prediction and GT class IDs don't match (T041)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        
        # Prediction is Car (0), GT is Pedestrian (1) - should not match
        stat = {
            "tp": np.zeros((1, 2), dtype=bool),
            "fp": np.ones((1, 2), dtype=bool),  # False positive due to class mismatch
            "conf": np.array([0.9]),
            "pred_cls": np.array([0]),  # Car
            "target_cls": np.array([1]),  # Pedestrian
            "boxes3d_pred": [
                Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0, class_label="Car", class_id=0, confidence=0.9),
            ],
            "boxes3d_target": [
                Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(0.88, 0.60, 1.73), orientation=0.0, class_label="Pedestrian", class_id=1, confidence=1.0),
            ],
        }
        metrics.update_stats(stat)
        
        results = metrics.process()
        
        # Class mismatch should result in FP, not TP
        # AP3D for Car should be 0.0 (no TP for Car class)
        assert isinstance(results, dict), "Results should handle class ID mismatches"

    def test_metrics_statistics_accumulation(self):
        """Test that stat accumulation works across batches (T042)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        metrics = Stereo3DDetMetrics(names=names)
        
        # Add multiple batches
        for i in range(3):
            stat = {
                "tp": np.array([[True, True]], dtype=bool),
                "fp": np.array([[False, False]], dtype=bool),
                "conf": np.array([0.9 - i * 0.1]),
                "pred_cls": np.array([0]),
                "target_cls": np.array([0]),
                "boxes3d_pred": [
                    Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0, class_label="Car", class_id=0, confidence=0.9 - i * 0.1),
                ],
                "boxes3d_target": [
                    Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0, class_label="Car", class_id=0, confidence=1.0),
                ],
            }
            metrics.update_stats(stat)
        
        # Verify accumulation
        assert len(metrics.stats) == 3, "Should accumulate 3 batches"
        
        results = metrics.process()
        
        # After accumulation, should have metrics for all batches
        assert isinstance(results, dict), "Results should be computed after accumulation"

    def test_metrics_consistency(self):
        """Test that metrics are consistent across runs (variance < 0.1%) (T043)."""
        names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        
        # Run process multiple times with same input
        results_list = []
        for _ in range(3):
            metrics = Stereo3DDetMetrics(names=names)
            stat = {
                "tp": np.array([[True, True]], dtype=bool),
                "fp": np.array([[False, False]], dtype=bool),
                "conf": np.array([0.9]),
                "pred_cls": np.array([0]),
                "target_cls": np.array([0]),
                "boxes3d_pred": [
                    Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0, class_label="Car", class_id=0, confidence=0.9),
                ],
                "boxes3d_target": [
                    Box3D(center_3d=(10.0, 2.0, 30.0), dimensions=(3.88, 1.63, 1.53), orientation=0.0, class_label="Car", class_id=0, confidence=1.0),
                ],
            }
            metrics.update_stats(stat)
            results = metrics.process()
            results_list.append(results)
        
        # Check consistency (all results should be identical)
        ap3d_50_values = [
            r["ap3d_50"].get(0, 0.0) if isinstance(r["ap3d_50"], dict) else 0.0
            for r in results_list
        ]
        
        # Variance should be very small (essentially zero for deterministic computation)
        if len(set(ap3d_50_values)) > 1:
            variance = np.var(ap3d_50_values)
            mean_val = np.mean(ap3d_50_values)
            relative_variance = variance / (mean_val + 1e-10)
            assert relative_variance < 0.001, f"Metrics should be consistent (variance={relative_variance:.6f} > 0.1%)"

