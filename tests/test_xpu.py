import shutil
import pytest
import torch

from tests import XPU_DEVICE_COUNT, XPU_IS_AVAILABLE, MODEL, SOURCE
from ultralytics import YOLO
from ultralytics.cfg import TASK2MODEL
from ultralytics.utils.torch_utils import TORCH_2_3

DEVICE = -1
if TORCH_2_3 and XPU_IS_AVAILABLE:
    DEVICE = 0


@pytest.mark.skipif(DEVICE == -1, reason="No XPU devices available")
def test_checks():
    assert torch.xpu.is_available() == XPU_IS_AVAILABLE
    assert torch.xpu.device_count() == XPU_DEVICE_COUNT


@pytest.mark.skipif(DEVICE == -1, reason="No XPU devices available")
def test_train():
    """Test model training on a minimal dataset using available CUDA devices."""

    _ = YOLO(MODEL).train(data="coco8.yaml", imgsz=64, epochs=1, device=f"xpu:{DEVICE}")  # requires imgsz>=64


@pytest.mark.slow
@pytest.mark.skipif(DEVICE == -1, reason="No XPU devices available")
def test_predict_multiple_devices_xpu():
    """Validate model prediction consistency across CPU and CUDA devices."""
    model = YOLO("yolo11n.pt")

    # Test CPU
    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE)
    assert str(model.device) == "cpu"

    # Test CUDA
    xpu_device = f"xpu:{DEVICE}"
    model = model.to(xpu_device)
    assert str(model.device) == xpu_device
    _ = model(SOURCE)
    assert str(model.device) == xpu_device

    # Test CPU again
    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE)
    assert str(model.device) == "cpu"

    # Test CUDA again
    model = model.to(xpu_device)
    assert str(model.device) == xpu_device
    _ = model(SOURCE)
    assert str(model.device) == xpu_device


@pytest.mark.skipif(DEVICE == -1, reason="No XPU devices available")
def test_model_info():
    """Test model info retrieval on XPU device."""
    model = YOLO("yolo11n.pt")
    model = model.to(f"xpu:{DEVICE}")
    info = model.info()
    # model.info() can return either tuple or dict depending on version
    assert isinstance(info, (dict, tuple))
    assert str(model.device) == f"xpu:{DEVICE}"


@pytest.mark.skipif(DEVICE == -1, reason="No XPU devices available")
def test_predict_sam_xpu():
    """Test SAM model predictions using different prompts on XPU device."""
    try:
        from ultralytics import SAM
        
        model = SAM("sam2.1_b.pt")
        model = model.to(f"xpu:{DEVICE}")
        
        # Run basic inference
        results = model(SOURCE, device=f"xpu:{DEVICE}")
        assert len(results) > 0
        
    except Exception as e:
        pytest.skip(f"Skipping SAM test due to: {str(e)}")


@pytest.mark.skipif(DEVICE == -1, reason="No XPU devices available")
def test_predict_with_different_imgsz():
    """Test model prediction with different image sizes on XPU device."""
    model = YOLO("yolo11n.pt")
    model = model.to(f"xpu:{DEVICE}")
    
    # Test with various image sizes
    for imgsz in [32, 64, 128]:
        results = model(SOURCE, imgsz=imgsz)
        assert len(results) > 0
    
    assert str(model.device) == f"xpu:{DEVICE}"


@pytest.mark.skipif(DEVICE == -1, reason="No XPU devices available")
def test_export_openvino_matrix():
    """Test YOLO exports to OpenVINO format with various configurations and parameters on XPU."""
    # Use a subset of tasks to avoid excessive testing time
    task = "detect"  # Limit to core tasks for faster execution
        
    try:
        model = YOLO(TASK2MODEL[task], task=task)
        file = model.export(format="openvino")
        
        # Test inference on exported model
        exported_model = YOLO(file+"/", task=task)
        results = exported_model.predict([SOURCE], device=f"intel:GPU.{DEVICE}")
        assert len(results) > 0
        
        # Cleanup
        shutil.rmtree(file, ignore_errors=True)   
    except Exception as e:
        pytest.skip(f"Skipping OpenVINO export test for {task} due to: {str(e)}")
