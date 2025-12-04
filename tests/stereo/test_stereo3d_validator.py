# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unit tests for stereo 3D detection validator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ultralytics.data.stereo.box3d import Box3D
from ultralytics.models.yolo.stereo3ddet.val import Stereo3DDetValidator, decode_stereo3d_outputs


class TestDecodeStereo3dOutputs:
    """Test suite for decode_stereo3d_outputs function."""

    def test_decode_stereo3d_outputs_basic(self):
        """Test basic decoding of 10-branch outputs."""
        batch_size = 1
        num_classes = 3
        h, w = 96, 320  # Feature map size (H/4, W/4)

        # Create mock 10-branch outputs
        outputs = {
            "heatmap": torch.randn(batch_size, num_classes, h, w),
            "offset": torch.randn(batch_size, 2, h, w),
            "bbox_size": torch.randn(batch_size, 2, h, w) * 10 + 20,  # Positive sizes
            "lr_distance": torch.randn(batch_size, 1, h, w) * 5 + 10,  # Positive distances
            "right_width": torch.randn(batch_size, 1, h, w) * 5 + 10,
            "dimensions": torch.randn(batch_size, 3, h, w) * 0.5,  # Small offsets
            "orientation": torch.randn(batch_size, 8, h, w),
            "vertices": torch.randn(batch_size, 8, h, w),
            "vertex_offset": torch.randn(batch_size, 8, h, w),
            "vertex_dist": torch.randn(batch_size, 4, h, w),
        }

        # Set high confidence in heatmap for one detection
        outputs["heatmap"][0, 0, 10, 20] = 0.9  # Class 0, position (10, 20)

        calib = {
            "fx": 721.5377,
            "fy": 721.5377,
            "cx": 609.5593,
            "cy": 172.8540,
            "baseline": 0.54,
        }

        boxes3d = decode_stereo3d_outputs(outputs, conf_threshold=0.5, top_k=100, calib=calib)
        assert isinstance(boxes3d, list)
        assert len(boxes3d) > 0, "Should decode at least one box"
        assert all(isinstance(box, Box3D) for box in boxes3d), "All items should be Box3D objects"

    def test_decode_stereo3d_outputs_empty(self):
        """Test decoding with low confidence threshold (no detections)."""
        batch_size = 1
        num_classes = 3
        h, w = 96, 320

        outputs = {
            "heatmap": torch.zeros(batch_size, num_classes, h, w),  # All zeros
            "offset": torch.randn(batch_size, 2, h, w),
            "bbox_size": torch.randn(batch_size, 2, h, w),
            "lr_distance": torch.randn(batch_size, 1, h, w),
            "right_width": torch.randn(batch_size, 1, h, w),
            "dimensions": torch.randn(batch_size, 3, h, w),
            "orientation": torch.randn(batch_size, 8, h, w),
            "vertices": torch.randn(batch_size, 8, h, w),
            "vertex_offset": torch.randn(batch_size, 8, h, w),
            "vertex_dist": torch.randn(batch_size, 4, h, w),
        }

        boxes3d = decode_stereo3d_outputs(outputs, conf_threshold=0.9, top_k=100)
        assert isinstance(boxes3d, list)
        assert len(boxes3d) == 0, "Should return empty list when no detections above threshold"


class TestStereo3DDetValidator:
    """Test suite for Stereo3DDetValidator class."""

    def test_validator_initialization(self):
        """Test that Stereo3DDetValidator initializes correctly."""
        args = {"task": "stereo3ddet", "imgsz": 640}
        validator = Stereo3DDetValidator(args=args)
        assert validator.args.task == "stereo3ddet"
        assert hasattr(validator, "metrics")

    def test_validator_preprocess(self):
        """Test that preprocess handles stereo batch correctly."""
        args = {"task": "stereo3ddet", "imgsz": 640, "half": False}
        validator = Stereo3DDetValidator(args=args)
        validator.device = torch.device("cpu")

        # Create mock batch with stereo images [B, 6, H, W]
        batch = {
            "img": torch.randn(2, 6, 384, 1280),
            "labels": [None, None],
            "calib": [None, None],
        }
        processed = validator.preprocess(batch)
        assert "img" in processed
        assert processed["img"].shape == (2, 6, 384, 1280)

    def test_validator_postprocess(self):
        """Test that postprocess decodes model outputs."""
        args = {"task": "stereo3ddet", "imgsz": 640}
        validator = Stereo3DDetValidator(args=args)

        # Create mock 10-branch outputs
        batch_size = 1
        num_classes = 3
        h, w = 96, 320
        outputs = {
            "heatmap": torch.randn(batch_size, num_classes, h, w),
            "offset": torch.randn(batch_size, 2, h, w),
            "bbox_size": torch.randn(batch_size, 2, h, w) * 10 + 20,
            "lr_distance": torch.randn(batch_size, 1, h, w) * 5 + 10,
            "right_width": torch.randn(batch_size, 1, h, w) * 5 + 10,
            "dimensions": torch.randn(batch_size, 3, h, w) * 0.5,
            "orientation": torch.randn(batch_size, 8, h, w),
            "vertices": torch.randn(batch_size, 8, h, w),
            "vertex_offset": torch.randn(batch_size, 8, h, w),
            "vertex_dist": torch.randn(batch_size, 4, h, w),
        }

        # Set high confidence
        outputs["heatmap"][0, 0, 10, 20] = 0.9

        preds = validator.postprocess(outputs)
        assert isinstance(preds, list)
        assert len(preds) == batch_size

    def test_validator_init_metrics(self):
        """Test that init_metrics initializes Stereo3DDetMetrics."""
        args = {"task": "stereo3ddet", "imgsz": 640}
        validator = Stereo3DDetValidator(args=args)

        # Create mock model
        model = MagicMock()
        model.names = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

        validator.init_metrics(model)
        assert hasattr(validator.metrics, "names")
        assert validator.metrics.names == model.names

    @pytest.mark.skip(reason="Requires full implementation of update_metrics")
    def test_validator_update_metrics(self):
        """Test that update_metrics processes predictions and ground truth."""
        # This test will be implemented after update_metrics is complete
        pass

    @pytest.mark.skip(reason="Requires full implementation of __call__")
    def test_validator_full_workflow(self):
        """Test full validation workflow with mock dataloader."""
        # This test will be implemented after __call__ is complete
        pass

