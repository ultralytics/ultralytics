# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import pytest
import torch

from tests import MODEL, ROCM_IS_AVAILABLE, SOURCE
from ultralytics import YOLO

# ROCm has no NVML equivalent for idle-GPU selection, so take the first visible devices
DEVICES = list(range(torch.cuda.device_count()))[:2] if ROCM_IS_AVAILABLE else []


@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_migraphx_inference():
    """Test ONNX export and inference route to the MIGraphX execution provider on AMD GPU."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32, device=DEVICES[0])
    model = YOLO(file)
    assert model(SOURCE, imgsz=32, device=DEVICES[0])
    assert "MIGraphXExecutionProvider" in model.predictor.model.session.get_providers()
    Path(file).unlink()


@pytest.mark.skipif(not ROCM_IS_AVAILABLE, reason="ROCm/HIP not available")
def test_rocm_cpu_fallback():
    """Test ONNX inference falls back to CPU when device='cpu' on a ROCm system."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32)
    model = YOLO(file)
    assert model(SOURCE, imgsz=32, device="cpu")
    assert "CPUExecutionProvider" in model.predictor.model.session.get_providers()
    Path(file).unlink()
