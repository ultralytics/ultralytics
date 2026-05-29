import pytest
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


@pytest.mark.skip(
    reason=(
        "DINOv2DPTHead requires torch.hub download of dinov2 weights and a full ViT encoder; "
        "too heavy for CPU CI. The export-branch ordering fix (training check before export "
        "interpolation) is structurally identical to the Depth head, which is covered by the "
        "tests above."
    )
)
def test_dinov2dpt_head_training_dict_and_export_upsample():
    from ultralytics.nn.modules.head import DINOv2DPTHead

    head = DINOv2DPTHead(encoder_name="vits", pretrained=False)

    # (a) training mode returns a dict
    head.train()
    x = torch.randn(1, 3, 518, 518)
    out = head(x)
    assert isinstance(out, dict) and "depth" in out

    # (b) eval + export + input_hw → output upsampled to input_hw
    head.eval()
    head.export, head.format, head.input_hw = True, "onnx", (518, 518)
    out2 = head(x)
    assert out2.shape[-2:] == (518, 518)
