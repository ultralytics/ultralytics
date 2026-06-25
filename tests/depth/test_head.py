import torch

from ultralytics.nn.modules.head import Depth


def test_depth_head_export_upsamples_to_input():
    head = Depth(c_mid=32, ch=(32, 64, 128)).eval()
    feats = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)]
    head.export, head.format, head.input_hw = True, "onnx", (256, 256)
    out = head(feats)
    assert out.shape[-2:] == (256, 256)          # upsampled to input resolution in the head
    head.export = False
    out2 = head(feats)
    assert out2.shape[-2:] != (256, 256)          # inference returns native head resolution


def test_depth_head_training_returns_dict():
    head = Depth(c_mid=32, ch=(32, 64, 128)).train()
    feats = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)]
    out = head(feats)
    assert isinstance(out, dict) and "depth" in out


def test_depth_head_export_coreml_no_upsample():
    head = Depth(c_mid=32, ch=(32, 64, 128)).eval()
    feats = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)]
    head.export, head.format, head.input_hw = True, "coreml", (256, 256)
    out = head(feats)
    assert out.shape[-2:] != (256, 256)   # coreml export must NOT interpolate


def test_depth_head_export_no_input_hw_no_upsample():
    head = Depth(c_mid=32, ch=(32, 64, 128)).eval()
    feats = [torch.randn(1, 32, 32, 32), torch.randn(1, 64, 16, 16), torch.randn(1, 128, 8, 8)]
    head.export, head.format, head.input_hw = True, "onnx", None
    out = head(feats)
    assert out.shape[-2:] != (256, 256)   # no input_hw → native resolution
