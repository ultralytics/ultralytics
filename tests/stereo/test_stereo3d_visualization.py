# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Integration tests for stereo 3D visualization in validation workflow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ultralytics.data.stereo.box3d import Box3D
from ultralytics.models.yolo.stereo3ddet.val import Stereo3DDetValidator


class TestStereo3DVisualization:
    """Integration tests for 3D visualization in validation."""

    def test_validator_saves_visualizations(self, tmp_path):
        """Verify that validator generates and saves visualization images when plots=True."""
        save_dir = tmp_path / "val_results"
        save_dir.mkdir()

        args = {
            "task": "stereo3ddet",
            "imgsz": 384,
            "plots": True,
            "half": False,
        }
        validator = Stereo3DDetValidator(args=args, save_dir=save_dir)
        validator.device = torch.device("cpu")
        validator.batch_i = 0

        # Create mock batch with stereo images
        batch = {
            "img": torch.randint(0, 255, (1, 6, 96, 320), dtype=torch.uint8),  # [B, 6, H, W]
            "labels": [[]],
            "calib": [
                {
                    "fx": 721.5377,
                    "fy": 721.5377,
                    "cx": 609.5593,
                    "cy": 172.8540,
                    "baseline": 0.54,
                    "image_width": 1242,
                    "image_height": 375,
                }
            ],
            "ori_shape": [(375, 1242)],
        }

        # Create mock predictions
        pred_boxes3d = [
            [
                Box3D(
                    center_3d=(10.0, 2.0, 30.0),
                    dimensions=(3.88, 1.63, 1.53),
                    orientation=0.0,
                    class_label="Car",
                    class_id=0,
                    confidence=0.95,
                )
            ]
        ]

        # Call visualization method
        validator.plot_validation_samples(batch, pred_boxes3d, batch_idx=0)

        # Verify visualization file was created
        vis_file = save_dir / "val_batch0_sample0_pred.jpg"
        assert vis_file.exists(), f"Visualization file should be created at {vis_file}"

        # Verify file is not empty
        assert vis_file.stat().st_size > 0, "Visualization file should not be empty"
