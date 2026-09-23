# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest

from tests import ROCM_IS_AVAILABLE, SOURCE
from tests.conftest import isolated_model_path
from ultralytics import YOLO
from ultralytics.cfg import TASK2MODEL, TASKS
from ultralytics.utils import WEIGHTS_DIR


@pytest.mark.skipif(not ROCM_IS_AVAILABLE, reason="ROCm/HIP not available")
def test_rocm_migraphx_inference(isolated_model):
    """Test ONNX export and inference route to the MIGraphX execution provider on AMD GPU."""
    file = YOLO(isolated_model).export(format="onnx", imgsz=32, device=0)
    model = YOLO(file)
    assert model(SOURCE, imgsz=32, device=0)
    assert "MIGraphXExecutionProvider" in model.predictor.model.session.get_providers()


@pytest.mark.slow
@pytest.mark.skipif(not ROCM_IS_AVAILABLE, reason="ROCm/HIP not available")
@pytest.mark.parametrize("task", sorted(TASKS))
def test_rocm_migraphx_matrix(task, tmp_path):
    """Test every YOLO26 task exports to ONNX and runs on the MIGraphX execution provider."""
    file = YOLO(isolated_model_path(tmp_path, WEIGHTS_DIR / TASK2MODEL[task])).export(format="onnx", imgsz=32, device=0)
    model = YOLO(file)
    assert model(SOURCE, imgsz=32, device=0)
    assert "MIGraphXExecutionProvider" in model.predictor.model.session.get_providers()


@pytest.mark.skipif(not ROCM_IS_AVAILABLE, reason="ROCm/HIP not available")
def test_rocm_cpu_fallback(isolated_model):
    """Test ONNX inference falls back to CPU when device='cpu' on a ROCm system."""
    file = YOLO(isolated_model).export(format="onnx", imgsz=32)
    model = YOLO(file)
    assert model(SOURCE, imgsz=32, device="cpu")
    assert model.predictor.model.session.get_providers() == ["CPUExecutionProvider"]
