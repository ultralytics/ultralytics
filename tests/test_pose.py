# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.utils.loss import v8PoseLoss


def test_v8_pose_loss_custom_sigmas():
    """Test custom OKS sigmas initialization and validation in v8PoseLoss."""

    class DummyHead:
        kpt_shape = [4, 3]
        stride = torch.tensor([8.0, 16.0, 32.0])
        nc = 1
        reg_max = 16
        no = 64

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)  # Parameter needed to infer device
            self.model = [DummyHead()]
            self.args = get_cfg(DEFAULT_CFG)  # Default hyperparameters

    model = DummyModel()

    # Valid custom sigmas
    model.kpt_oks_sigmas = [0.1, 0.2, 0.3, 0.4]
    loss = v8PoseLoss(model)
    expected = torch.tensor([0.1, 0.2, 0.3, 0.4], device=loss.device, dtype=torch.float32)
    assert torch.allclose(loss.keypoint_loss.sigmas, expected)

    # Count mismatch raises ValueError
    model.kpt_oks_sigmas = [0.1, 0.2]
    with pytest.raises(ValueError, match="does not match keypoint count"):
        v8PoseLoss(model)

    # Non-positive or non-finite values raise ValueError
    model.kpt_oks_sigmas = [0.1, 0.0, -0.2, 0.4]
    with pytest.raises(ValueError, match="strictly positive"):
        v8PoseLoss(model)
