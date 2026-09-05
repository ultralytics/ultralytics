# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

import pytest

from tests import MODEL, ROCM_IS_AVAILABLE, SOURCE
from ultralytics import YOLO


@pytest.mark.skipif(not ROCM_IS_AVAILABLE, reason="ROCm/HIP not available")
def test_rocm_migraphx_inference():
    """Test ONNX export and inference route to the MIGraphX execution provider on AMD GPU."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32, device=0)
    model = YOLO(file)
    assert model(SOURCE, imgsz=32, device=0)
    assert "MIGraphXExecutionProvider" in model.predictor.model.session.get_providers()
    Path(file).unlink()


@pytest.mark.skipif(not ROCM_IS_AVAILABLE, reason="ROCm/HIP not available")
def test_rocm_cpu_fallback():
    """Test ONNX inference falls back to CPU when device='cpu' on a ROCm system."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32)
    model = YOLO(file)
    assert model(SOURCE, imgsz=32, device="cpu")
    assert model.predictor.model.session.get_providers() == ["CPUExecutionProvider"]
    Path(file).unlink()
