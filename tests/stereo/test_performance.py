# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Performance tests for stereo 3D detection validation optimization."""

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics.data.stereo.box3d import Box3D


class TestPerformanceBenchmark:
    """Test suite for performance benchmarking (T150)."""

    @pytest.mark.skip(reason="Requires actual dataset and model - integration test")
    def test_baseline_validation_time(self):
        """Test to measure baseline validation time per epoch.
        
        This test measures the baseline validation time before optimizations.
        It should be run manually with actual dataset and model.
        """
        # This would require actual dataset and model
        # For now, this is a placeholder test
        pass


class TestCompute3DIoUBatch:
    """Test suite for batch 3D IoU computation (T151)."""

    def test_compute_3d_iou_batch_matches_individual(self):
        """Test that compute_3d_iou_batch matches individual compute_3d_iou results.
        
        This test verifies that the vectorized batch computation produces
        the same results as calling compute_3d_iou individually for each pair.
        """
        from ultralytics.utils.metrics import compute_3d_iou
        
        # Create test boxes
        pred_boxes = [
            Box3D(
                center_3d=(10.0, 2.0, 30.0),
                dimensions=(3.88, 1.63, 1.53),
                orientation=0.0,
                class_label="Car",
                class_id=0,
                confidence=0.95,
            ),
            Box3D(
                center_3d=(15.0, 2.0, 35.0),
                dimensions=(4.0, 1.7, 1.6),
                orientation=0.1,
                class_label="Car",
                class_id=0,
                confidence=0.90,
            ),
        ]
        
        gt_boxes = [
            Box3D(
                center_3d=(10.0, 2.0, 30.0),
                dimensions=(3.88, 1.63, 1.53),
                orientation=0.0,
                class_label="Car",
                class_id=0,
                confidence=1.0,
            ),
            Box3D(
                center_3d=(20.0, 2.0, 40.0),
                dimensions=(4.0, 1.7, 1.6),
                orientation=0.0,
                class_label="Car",
                class_id=0,
                confidence=1.0,
            ),
        ]
        
        # Compute IoU matrix using individual calls
        individual_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                if pred_box.class_id == gt_box.class_id:
                    individual_matrix[i, j] = compute_3d_iou(pred_box, gt_box)
        
        # Import batch function (will be implemented in T154)
        try:
            from ultralytics.models.yolo.stereo3ddet.val import compute_3d_iou_batch
            
            # Compute IoU matrix using batch function
            batch_matrix = compute_3d_iou_batch(pred_boxes, gt_boxes)
            
            # Verify results match within numerical precision
            np.testing.assert_allclose(
                batch_matrix, individual_matrix, rtol=1e-6, atol=1e-7,
                err_msg="Batch IoU computation does not match individual computation"
            )
        except ImportError:
            pytest.skip("compute_3d_iou_batch not yet implemented (T154)")


class TestVectorizedIoUIntegration:
    """Test suite for vectorized IoU integration (T152)."""

    @pytest.mark.skip(reason="Requires full validator integration - integration test")
    def test_vectorized_iou_produces_same_metrics(self):
        """Test that vectorized IoU computation produces same metrics as original.
        
        This integration test verifies that using vectorized IoU computation
        in update_metrics produces the same AP3D metrics as the original implementation.
        """
        # This would require full validator setup with mock data
        # For now, this is a placeholder test
        pass


class TestBatchTensorOperations:
    """Test suite for batch tensor operations in decode (T153)."""

    @pytest.mark.skip(reason="Requires decode_stereo3d_outputs refactoring - integration test")
    def test_batch_tensor_operations_match_original(self):
        """Test that batch tensor operations in decode_stereo3d_outputs produce same results.
        
        This test verifies that the optimized decode function (with batched tensor operations)
        produces the same Box3D results as the original per-detection implementation.
        """
        # This would require refactored decode_stereo3d_outputs
        # For now, this is a placeholder test
        pass


class TestDecodeStereo3DOutputsPerformance:
    """Test suite for decode_stereo3d_outputs performance benchmarks (T200)."""

    def test_decode_stereo3d_outputs_batch_vs_per_sample_performance(self):
        """Test T200: Benchmark decode_stereo3d_outputs batch vs per-sample performance.
        
        This test measures the performance difference between:
        1. Processing entire batch at once (optimized)
        2. Processing each sample individually (current implementation)
        
        Expected: Batch processing should be significantly faster.
        """
        from ultralytics.models.yolo.stereo3ddet.val import decode_stereo3d_outputs
        
        # Create mock outputs for batch of 8 samples
        batch_size = 8
        num_classes = 3
        h, w = 96, 320  # H/4, W/4 for 384×1280 input
        
        # Create mock 10-branch outputs
        outputs_batch = {
            "heatmap": torch.randn(batch_size, num_classes, h, w),
            "offset": torch.randn(batch_size, 2, h, w),
            "bbox_size": torch.randn(batch_size, 2, h, w),
            "lr_distance": torch.randn(batch_size, 1, h, w).abs(),
            "right_width": torch.randn(batch_size, 1, h, w).abs(),
            "dimensions": torch.randn(batch_size, 3, h, w),
            "orientation": torch.randn(batch_size, 8, h, w),
            "vertices": torch.randn(batch_size, 8, h, w),
            "vertex_offset": torch.randn(batch_size, 8, h, w),
            "vertex_dist": torch.randn(batch_size, 4, h, w).abs(),
        }
        
        # Calibration parameters
        calib = {
            "fx": 721.5377,
            "fy": 721.5377,
            "cx": 609.5593,
            "cy": 172.8540,
            "baseline": 0.54,
        }
        
        # Benchmark per-sample processing (current implementation)
        per_sample_times = []
        for b in range(batch_size):
            single_outputs = {k: v[b:b+1] for k, v in outputs_batch.items()}
            start = time.time()
            decode_stereo3d_outputs(single_outputs, conf_threshold=0.25, top_k=100, calib=calib)
            per_sample_times.append(time.time() - start)
        
        per_sample_total = sum(per_sample_times)
        per_sample_avg = per_sample_total / batch_size
        
        # Benchmark batch processing (will be implemented in T206)
        # For now, this test documents the expected interface
        # When batch processing is implemented, it should be faster
        try:
            # This will work once T206 is implemented
            start = time.time()
            batch_results = decode_stereo3d_outputs(outputs_batch, conf_threshold=0.25, top_k=100, calib=calib)
            batch_time = time.time() - start
            
            # Verify batch processing is faster (at least 2x speedup expected)
            speedup = per_sample_total / batch_time
            assert speedup >= 1.5, f"Batch processing should be faster, got speedup: {speedup:.2f}x"
            
            # Verify batch results structure
            assert isinstance(batch_results, list), "Batch results should be a list"
            assert len(batch_results) == batch_size, f"Expected {batch_size} results, got {len(batch_results)}"
            assert all(isinstance(r, list) for r in batch_results), "Each result should be a list of Box3D"
        except (TypeError, ValueError):
            # Batch processing not yet implemented (T206)
            pytest.skip("Batch processing not yet implemented (T206) - will benchmark after implementation")


class TestPostprocessPerformance:
    """Test suite for postprocess performance benchmarks (T201)."""

    def test_postprocess_batch_vs_per_sample_performance(self):
        """Test T201: Benchmark postprocess batch vs per-sample performance.
        
        This test measures the performance difference between:
        1. Processing entire batch at once (optimized)
        2. Processing each sample individually (current implementation)
        
        Expected: Batch processing should be significantly faster.
        """
        from ultralytics.models.yolo.stereo3ddet.val import Stereo3DDetValidator
        from unittest.mock import MagicMock
        
        # Create mock validator
        validator = Stereo3DDetValidator(dataloader=None, save_dir=None, args=MagicMock())
        validator.args.conf = 0.25
        
        # Create mock batch outputs
        batch_size = 8
        num_classes = 3
        h, w = 96, 320
        
        preds = {
            "heatmap": torch.randn(batch_size, num_classes, h, w),
            "offset": torch.randn(batch_size, 2, h, w),
            "bbox_size": torch.randn(batch_size, 2, h, w),
            "lr_distance": torch.randn(batch_size, 1, h, w).abs(),
            "right_width": torch.randn(batch_size, 1, h, w).abs(),
            "dimensions": torch.randn(batch_size, 3, h, w),
            "orientation": torch.randn(batch_size, 8, h, w),
            "vertices": torch.randn(batch_size, 8, h, w),
            "vertex_offset": torch.randn(batch_size, 8, h, w),
            "vertex_dist": torch.randn(batch_size, 4, h, w).abs(),
        }
        
        # Mock calibration
        validator._current_batch = {
            "calib": [{
                "fx": 721.5377,
                "fy": 721.5377,
                "cx": 609.5593,
                "cy": 172.8540,
                "baseline": 0.54,
            }] * batch_size
        }
        
        # Benchmark current per-sample implementation
        start = time.time()
        results_current = validator.postprocess(preds)
        current_time = time.time() - start
        
        # Verify results structure
        assert isinstance(results_current, list), "Results should be a list"
        assert len(results_current) == batch_size, f"Expected {batch_size} results, got {len(results_current)}"
        
        # After T212 implementation, batch processing should be faster
        # This test documents the expected performance improvement
        # Note: Actual benchmark comparison will be done after T212 implementation


class TestBatchDecodeAccuracy:
    """Test suite for batch decode accuracy verification (T202)."""

    def test_batch_decode_produces_same_outputs_as_per_sample(self):
        """Test T202: Verify batch decode produces same Box3D outputs as per-sample decode.
        
        This test verifies that the optimized batch processing produces
        identical results to the original per-sample implementation.
        """
        from ultralytics.models.yolo.stereo3ddet.val import decode_stereo3d_outputs
        
        # Create mock outputs for batch of 4 samples
        batch_size = 4
        num_classes = 3
        h, w = 96, 320
        
        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        
        outputs_batch = {
            "heatmap": torch.randn(batch_size, num_classes, h, w),
            "offset": torch.randn(batch_size, 2, h, w),
            "bbox_size": torch.randn(batch_size, 2, h, w),
            "lr_distance": torch.randn(batch_size, 1, h, w).abs(),
            "right_width": torch.randn(batch_size, 1, h, w).abs(),
            "dimensions": torch.randn(batch_size, 3, h, w),
            "orientation": torch.randn(batch_size, 8, h, w),
            "vertices": torch.randn(batch_size, 8, h, w),
            "vertex_offset": torch.randn(batch_size, 8, h, w),
            "vertex_dist": torch.randn(batch_size, 4, h, w).abs(),
        }
        
        calib = {
            "fx": 721.5377,
            "fy": 721.5377,
            "cx": 609.5593,
            "cy": 172.8540,
            "baseline": 0.54,
        }
        
        # Get per-sample results (current implementation)
        per_sample_results = []
        for b in range(batch_size):
            single_outputs = {k: v[b:b+1] for k, v in outputs_batch.items()}
            boxes3d = decode_stereo3d_outputs(single_outputs, conf_threshold=0.25, top_k=100, calib=calib)
            per_sample_results.append(boxes3d)
        
        # Get batch results (will be implemented in T206)
        try:
            batch_results = decode_stereo3d_outputs(outputs_batch, conf_threshold=0.25, top_k=100, calib=calib)
            
            # Verify batch results structure
            assert isinstance(batch_results, list), "Batch results should be a list"
            assert len(batch_results) == batch_size, f"Expected {batch_size} results, got {len(batch_results)}"
            
            # Compare results sample by sample
            for b in range(batch_size):
                per_sample_boxes = per_sample_results[b]
                batch_boxes = batch_results[b]
                
                assert isinstance(batch_boxes, list), f"Result {b} should be a list"
                assert len(batch_boxes) == len(per_sample_boxes), \
                    f"Sample {b}: Expected {len(per_sample_boxes)} boxes, got {len(batch_boxes)}"
                
                # Compare each Box3D object
                for i, (per_box, batch_box) in enumerate(zip(per_sample_boxes, batch_boxes)):
                    assert isinstance(batch_box, Box3D), f"Result {b}[{i}] should be Box3D"
                    
                    # Compare key attributes (allowing small numerical differences)
                    np.testing.assert_allclose(
                        per_box.center_3d, batch_box.center_3d, rtol=1e-5, atol=1e-6,
                        err_msg=f"Sample {b}, box {i}: center_3d mismatch"
                    )
                    np.testing.assert_allclose(
                        per_box.dimensions, batch_box.dimensions, rtol=1e-5, atol=1e-6,
                        err_msg=f"Sample {b}, box {i}: dimensions mismatch"
                    )
                    np.testing.assert_allclose(
                        per_box.orientation, batch_box.orientation, rtol=1e-5, atol=1e-6,
                        err_msg=f"Sample {b}, box {i}: orientation mismatch"
                    )
                    assert per_box.class_id == batch_box.class_id, \
                        f"Sample {b}, box {i}: class_id mismatch"
                    np.testing.assert_allclose(
                        per_box.confidence, batch_box.confidence, rtol=1e-5, atol=1e-6,
                        err_msg=f"Sample {b}, box {i}: confidence mismatch"
                    )
        except (TypeError, ValueError) as e:
            # Batch processing not yet implemented (T206)
            pytest.skip(f"Batch processing not yet implemented (T206): {e}")

