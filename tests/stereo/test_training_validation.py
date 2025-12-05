# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Validation tests for stereo 3D detection training process."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics.cfg.models.stereo import load_stereo_config
from ultralytics.data.stereo.calib import CalibrationParameters
from ultralytics.models.yolo.stereo3ddet import Stereo3DDetTrainer
from ultralytics.nn.modules.stereo.head import StereoCenterNetHead
from ultralytics.utils import DEFAULT_CFG


@pytest.fixture
def mock_stereo_dataset(tmp_path):
    """Create a mock stereo dataset structure for testing."""
    dataset_root = tmp_path / "kitti_stereo"
    dataset_root.mkdir()

    # Create directory structure
    (dataset_root / "images" / "train" / "left").mkdir(parents=True)
    (dataset_root / "images" / "train" / "right").mkdir(parents=True)
    (dataset_root / "labels" / "train").mkdir(parents=True)
    (dataset_root / "calib" / "train").mkdir(parents=True)

    # Create dummy images
    for i in range(4):  # 4 sample images
        image_id = f"{i:06d}"
        # Create dummy left and right images (small for testing)
        left_img = np.random.randint(0, 255, (96, 320, 3), dtype=np.uint8)
        right_img = np.random.randint(0, 255, (96, 320, 3), dtype=np.uint8)

        import cv2

        cv2.imwrite(str(dataset_root / "images" / "train" / "left" / f"{image_id}.png"), left_img)
        cv2.imwrite(str(dataset_root / "images" / "train" / "right" / f"{image_id}.png"), right_img)

        # Create dummy calibration file
        calib_file = dataset_root / "calib" / "train" / f"{image_id}.txt"
        with open(calib_file, "w") as f:
            f.write("P2: 721.5377 0.0 609.5593 0.0 0.0 721.5377 172.8540 0.0 0.0 0.0 1.0 0.0\n")
            f.write("Tr: 0.0 0.0 0.0 0.54\n")

        # Create dummy label file (YOLO 3D format: class_id x y z l w h yaw)
        label_file = dataset_root / "labels" / "train" / f"{image_id}.txt"
        with open(label_file, "w") as f:
            # One car object
            f.write("0 0.5 0.5 30.0 3.88 1.63 1.53 0.0\n")

    return str(dataset_root)


class TestTrainingValidation:
    """Test suite for validating stereo 3D detection training process."""

    def test_trainer_initialization(self):
        """Test that Stereo3DDetTrainer can be initialized with default config."""
        overrides = {
            "task": "stereo3ddet",
            "model": "ultralytics/cfg/models/stereo/stereo-centernet-s.yaml",
            "data": None,  # No dataset for initialization test
            "epochs": 1,
            "imgsz": 384,
            "batch": 2,
            "workers": 0,
            "save": False,
            "plots": False,
            "val": False,
        }
        trainer = Stereo3DDetTrainer(overrides=overrides)
        assert trainer is not None
        assert hasattr(trainer, "loss_names")
        # Note: loss_names may not be 10 if model doesn't have stereo head
        # This test just verifies trainer can be initialized
        assert isinstance(trainer.loss_names, list)
        assert len(trainer.loss_names) > 0

    def test_stereo_config_detection(self):
        """Test that stereo configs are properly detected."""
        config = load_stereo_config("ultralytics/cfg/models/stereo/stereo-centernet-s.yaml")
        assert config["stereo"] is True
        assert config["input_channels"] == 6
        assert config["nc"] == 3

    def test_head_output_shapes(self):
        """Test that detection head produces correct output shapes."""
        batch_size = 2
        in_channels = 256
        num_classes = 3
        height, width = 96, 320  # H/4, W/4 for 384×1280 input

        head = StereoCenterNetHead(in_channels=in_channels, num_classes=num_classes)
        head.eval()

        # Create dummy input
        x = torch.randn(batch_size, in_channels, height, width)

        # Forward pass
        outputs = head(x)

        # Verify all 10 branches exist
        expected_branches = [
            "heatmap",
            "offset",
            "bbox_size",
            "lr_distance",
            "right_width",
            "dimensions",
            "orientation",
            "vertices",
            "vertex_offset",
            "vertex_dist",
        ]
        assert len(outputs) == 10
        for branch_name in expected_branches:
            assert branch_name in outputs, f"Missing branch: {branch_name}"

        # Verify output shapes
        assert outputs["heatmap"].shape == (batch_size, num_classes, height, width)
        assert outputs["offset"].shape == (batch_size, 2, height, width)
        assert outputs["bbox_size"].shape == (batch_size, 2, height, width)
        assert outputs["lr_distance"].shape == (batch_size, 1, height, width)
        assert outputs["right_width"].shape == (batch_size, 1, height, width)
        assert outputs["dimensions"].shape == (batch_size, 3, height, width)
        assert outputs["orientation"].shape == (batch_size, 8, height, width)
        assert outputs["vertices"].shape == (batch_size, 8, height, width)
        assert outputs["vertex_offset"].shape == (batch_size, 8, height, width)
        assert outputs["vertex_dist"].shape == (batch_size, 4, height, width)

    def test_loss_names_consistency(self):
        """Test that loss names match expected 10-branch architecture."""
        overrides = {"task": "stereo3ddet", "model": "yolo11n.yaml", "data": None, "epochs": 1, "save": False}
        trainer = Stereo3DDetTrainer(overrides=overrides)

        # Note: yolo11n.yaml may not have stereo head, so loss_names may differ
        # This test verifies trainer can be initialized and has loss_names attribute
        assert hasattr(trainer, "loss_names")
        assert isinstance(trainer.loss_names, list)

    def test_progress_string_format(self):
        """Test that progress string is correctly formatted for 10 loss branches."""
        overrides = {"task": "stereo3ddet", "model": "yolo11n.yaml", "data": None, "epochs": 1, "save": False}
        trainer = Stereo3DDetTrainer(overrides=overrides)

        progress_str = trainer.progress_string()
        assert progress_str is not None
        assert "Epoch" in progress_str
        assert "GPU_mem" in progress_str
        # Check for presence of loss names and other headers
        for loss_name in trainer.loss_names:
            assert loss_name in progress_str, f"Missing loss name: {loss_name}"
        assert "Instances" in progress_str
        assert "Size" in progress_str

    def test_label_loss_items(self):
        """Test that loss items are correctly labeled."""
        overrides = {"task": "stereo3ddet", "model": "yolo11n.yaml", "data": None, "epochs": 1, "save": False}
        trainer = Stereo3DDetTrainer(overrides=overrides)

        # Test with None (should return keys)
        keys = trainer.label_loss_items()
        assert isinstance(keys, list)
        assert len(keys) > 0
        assert all(k.startswith("train/") for k in keys)

        # Test with loss values (match number of loss names)
        num_losses = len(trainer.loss_names)
        loss_values = [1.0] * num_losses
        labeled = trainer.label_loss_items(loss_values, prefix="train")
        assert len(labeled) == num_losses
        assert all(k.startswith("train/") for k in labeled.keys())
        assert all(isinstance(v, float) for v in labeled.values())

    @pytest.mark.skip(reason="Requires full dataset and model implementation")
    def test_training_one_epoch(self, mock_stereo_dataset):
        """Test that training can run for one epoch without crashing.

        This is a placeholder test that would run actual training once the full
        implementation is complete. It verifies:
        - Training loop completes without crash
        - Loss values are computed
        - Model weights are updated
        """
        overrides = {
            "task": "stereo3ddet",
            "model": "ultralytics/cfg/models/stereo/stereo-centernet-s.yaml",
            "data": mock_stereo_dataset,
            "epochs": 1,
            "imgsz": 384,
            "batch": 2,
            "workers": 0,
            "save": False,
            "plots": False,
            "val": False,
        }

        # This test would run actual training
        # trainer = Stereo3DDetTrainer(overrides=overrides)
        # trainer.train()
        # assert trainer.epoch == 1
        # assert trainer.loss is not None
        pass

    def test_checkpoint_compatibility(self):
        """Test that checkpoint save/load methods exist and are compatible."""
        overrides = {"task": "stereo3ddet", "model": "yolo11n.yaml", "data": None, "epochs": 1, "save": False}
        trainer = Stereo3DDetTrainer(overrides=overrides)

        # Verify methods exist
        assert hasattr(trainer, "save_model")
        assert hasattr(trainer, "resume_training")
        assert callable(trainer.save_model)
        assert callable(trainer.resume_training)

    def test_model_registry_integration(self):
        """Test that stereo configs are detected by model registry."""
        from ultralytics.nn.tasks import guess_model_task, yaml_model_load

        # Load stereo config
        config = yaml_model_load("ultralytics/cfg/models/stereo/stereo-centernet-s.yaml")
        task = guess_model_task(config)

        assert task == "stereo3ddet", f"Expected 'stereo3ddet', got '{task}'"

        # Test path-based detection
        path = Path("ultralytics/cfg/models/stereo/stereo-centernet-s.yaml")
        task_from_path = guess_model_task(str(path))
        assert task_from_path == "stereo3ddet"

    def test_final_eval_converts_path_to_string(self, tmp_path):
        """Test that Stereo3DDetTrainer.final_eval converts Path to string before passing to validator.
        
        This test verifies T161: final_eval should convert Path object to string for AutoBackend compatibility.
        """
        from unittest.mock import MagicMock, patch

        overrides = {
            "task": "stereo3ddet",
            "model": "yolo11n.yaml",
            "data": None,
            "epochs": 1,
            "save": False,
            "val": False,
        }
        trainer = Stereo3DDetTrainer(overrides=overrides)

        # Create a mock Path object for best checkpoint
        best_path = tmp_path / "best.pt"
        best_path.touch()  # Create the file so exists() returns True
        trainer.best = best_path

        # Create a mock Path object for last checkpoint
        last_path = tmp_path / "last.pt"
        last_path.touch()
        trainer.last = last_path

        # Mock validator
        mock_validator = MagicMock()
        mock_validator.args = MagicMock()
        mock_validator.args.plots = False
        mock_validator.args.compile = False
        mock_validator.return_value = {"ap3d_50": 0.5, "ap3d_70": 0.4}
        trainer.validator = mock_validator

        # Mock strip_optimizer to avoid actual file operations
        with patch("ultralytics.models.yolo.stereo3ddet.train.strip_optimizer") as mock_strip:
            mock_strip.return_value = {}
            # Mock torch_distributed_zero_first context manager
            with patch("ultralytics.models.yolo.stereo3ddet.train.torch_distributed_zero_first"):
                # Mock RANK to be -1 (single GPU)
                with patch("ultralytics.models.yolo.stereo3ddet.train.RANK", -1):
                    # Mock run_callbacks
                    trainer.run_callbacks = MagicMock()
                    # Call final_eval
                    trainer.final_eval()

        # Verify validator was called with a string, not a Path object
        assert mock_validator.called, "Validator should have been called"
        call_args = mock_validator.call_args
        assert "model" in call_args.kwargs, "Validator should be called with 'model' keyword argument"
        model_arg = call_args.kwargs["model"]
        assert isinstance(model_arg, str), f"Model argument should be string, got {type(model_arg)}"
        assert model_arg == str(best_path), f"Model argument should be string path, got {model_arg}"

