# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.nn.modules.head import Depth


def test_depth_head_export_upsamples_to_input():
    """Test depth head export upsamples to input."""
    head = Depth(c_mid=32, ch=(32, 64, 128)).eval()
    feats = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)]
    head.export, head.format = True, "onnx"
    out = head(feats)
    assert out.shape[-2:] == (256, 256)  # x4 upsample from P2 back to the input resolution
    head.export = False
    out2 = head(feats)
    assert out2.shape[-2:] != (256, 256)  # inference returns native head resolution


def test_depth_head_training_returns_dict():
    """Test depth head training returns dict."""
    head = Depth(c_mid=32, ch=(32, 64, 128)).train()
    feats = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)]
    out = head(feats)
    assert isinstance(out, dict) and "depth" in out


def test_depth_head_export_coreml_no_upsample():
    """Test depth head export coreml no upsample."""
    head = Depth(c_mid=32, ch=(32, 64, 128)).eval()
    feats = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)]
    head.export, head.format = True, "coreml"
    out = head(feats)
    assert out.shape[-2:] != (256, 256)  # coreml export must NOT interpolate


def test_depth_head_no_dead_parameters():
    """Every head parameter receives gradient — DDP then needs no find_unused_parameters."""
    head = Depth(c_mid=32, ch=(32, 64, 128)).train()
    feats = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)]
    head(feats)["depth"].sum().backward()
    unused = [n for n, p in head.named_parameters() if p.grad is None]
    assert not unused, f"parameters with no gradient: {unused}"
