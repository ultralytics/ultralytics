import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.data.utils import calculate_class_weights
from ultralytics.data.build import build_dataloader
from ultralytics.cfg import get_cfg


class TestClassBalancing:
    """Test class balancing features including pos_weight and WeightedRandomSampler."""

    def test_pos_weight_logic(self):
        """Test that pos_weight correctly weights positive classes."""
        mock_model = Mock()
        mock_model.model = [Mock()]
        mock_model.model[-1].nc = 3
        mock_model.model[-1].reg_max = 16
        mock_model.model[-1].stride = torch.tensor([8, 16, 32])
        
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        mock_args = Mock()
        mock_args.cls_weights = [1.0, 2.0, 0.5]
        mock_model.args = mock_args
        
        loss_fn = v8DetectionLoss(mock_model)
        
        assert loss_fn.bce.pos_weight is not None
        assert torch.allclose(loss_fn.bce.pos_weight, torch.tensor([1.0, 2.0, 0.5]))

    def test_pos_weight_auto_calculation(self):
        """Test automatic pos_weight calculation from dataset."""
        mock_model = Mock()
        mock_model.model = [Mock()]
        mock_model.model[-1].nc = 2
        mock_model.model[-1].reg_max = 16
        mock_model.model[-1].stride = torch.tensor([8, 16, 32])
        
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=4)
        mock_dataset.__getitem__ = Mock(side_effect=[
            {'cls': torch.tensor([0])},
            {'cls': torch.tensor([0])},
            {'cls': torch.tensor([1])},
            {'cls': torch.tensor([0])}
        ])
        
        mock_trainer = Mock()
        mock_trainer.train_loader = Mock()
        mock_trainer.train_loader.dataset = mock_dataset
        mock_model.trainer = mock_trainer
        
        mock_args = Mock()
        mock_args.cls_weights = True
        mock_model.args = mock_args
        
        loss_fn = v8DetectionLoss(mock_model)
        
        assert loss_fn.bce.pos_weight is not None

    def test_calculate_class_weights(self):
        """Test class weight calculation function."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=6)
        mock_dataset.__getitem__ = Mock(side_effect=[
            {'cls': torch.tensor([0])},
            {'cls': torch.tensor([0])},
            {'cls': torch.tensor([0])},
            {'cls': torch.tensor([1])},
            {'cls': torch.tensor([2])},
            {'cls': torch.tensor([2])}
        ])
        
        weights = calculate_class_weights(mock_dataset, nc=3)
        
        assert len(weights) == 3
        assert weights[0] < weights[1]
        assert weights[1] == weights[2]

    def test_weighted_random_sampler(self):
        """Test WeightedRandomSampler balances class distribution."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=4)
        mock_dataset.__getitem__ = Mock(side_effect=[
            {'cls': torch.tensor([0])},
            {'cls': torch.tensor([0])},
            {'cls': torch.tensor([1])},
            {'cls': torch.tensor([0])}
        ])
        
        cls_weights = [0.5, 2.0]
        
        dataloader = build_dataloader(
            mock_dataset, 
            batch=2, 
            workers=0, 
            shuffle=False,
            use_weighted_sampler=True,
            cls_weights=cls_weights
        )
        
        assert dataloader.sampler is not None
        assert hasattr(dataloader.sampler, 'weights')

    def test_cls_weights_yaml_loading(self):
        """Test model.yaml loads cls_weights without error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
task: detect
mode: train
cls_weights: [1.0, 2.0, 1.5]
epochs: 1
""")
            temp_yaml = f.name
        
        try:
            cfg = get_cfg(temp_yaml)
            assert hasattr(cfg, 'cls_weights')
            assert cfg.cls_weights == [1.0, 2.0, 1.5]
        finally:
            os.unlink(temp_yaml)

    def test_cls_weights_auto_mode(self):
        """Test cls_weights=True for auto calculation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
task: detect
mode: train
cls_weights: true
epochs: 1
""")
            temp_yaml = f.name
        
        try:
            cfg = get_cfg(temp_yaml)
            assert hasattr(cfg, 'cls_weights')
            assert cfg.cls_weights is True
        finally:
            os.unlink(temp_yaml)

    def test_backward_compatibility(self):
        """Test that existing configs work unchanged."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
task: detect
mode: train
epochs: 1
""")
            temp_yaml = f.name
        
        try:
            cfg = get_cfg(temp_yaml)
            assert not hasattr(cfg, 'cls_weights') or cfg.cls_weights is None
        finally:
            os.unlink(temp_yaml)

    def test_empty_dataset_handling(self):
        """Test handling of empty or malformed datasets."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=0)
        
        weights = calculate_class_weights(mock_dataset, nc=3)
        assert torch.allclose(weights, torch.ones(3))

    def test_malformed_sample_handling(self):
        """Test handling of samples without cls labels."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=3)
        mock_dataset.__getitem__ = Mock(side_effect=[
            {'cls': torch.tensor([0])},
            {'no_cls': 'invalid'},
            {'cls': torch.tensor([1])}
        ])
        
        weights = calculate_class_weights(mock_dataset, nc=2)
        assert len(weights) == 2


if __name__ == "__main__":
    pytest.main([__file__])
