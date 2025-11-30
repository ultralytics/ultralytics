import pytest
import torch

from ultralytics import YOLO

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="XPU not available",
)


def test_yolo_xpu_forward():
    """Test that YOLO forward works on Intel XPU."""
    model = YOLO("yolo11n.pt")
    model.to("xpu")
    x = torch.rand(1, 3, 64, 64, device="xpu")
    y = model.model(x)
    assert y is not None
    print("\n[XPU Test] YOLO XPU forward passed successfully ✔")
