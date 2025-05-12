# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from itertools import product
from pathlib import Path

import pytest
import torch

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE, MODEL, SOURCE
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, IS_JETSON, WEIGHTS_DIR
from ultralytics.utils.autodevice import GPUInfo
from ultralytics.utils.checks import check_amp
from ultralytics.utils.torch_utils import TORCH_1_13

# Try to find idle devices if CUDA is available
DEVICES = []
if CUDA_IS_AVAILABLE:
    if IS_JETSON:
        DEVICES = [0]  # NVIDIA Jetson only has one GPU and does not fully support pynvml library
    else:
        gpu_info = GPUInfo()
        gpu_info.print_status()
        idle_gpus = gpu_info.select_idle_gpu(count=2, min_memory_mb=2048)
        if idle_gpus:
            DEVICES = idle_gpus


def test_checks():
    """Validate CUDA settings against torch CUDA functions."""
    assert torch.cuda.is_available() == CUDA_IS_AVAILABLE
    assert torch.cuda.device_count() == CUDA_DEVICE_COUNT


@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_amp():
    """Test AMP training checks."""
    model = YOLO("yolo11n.pt").model.to(f"cuda:{DEVICES[0]}")
    assert check_amp(model)


@pytest.mark.slow
# @pytest.mark.skipif(IS_JETSON, reason="Temporary disable ONNX for Jetson")
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, simplify, nms",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, int8, half, batch, simplify, nms)
        for task, dynamic, int8, half, batch, simplify, nms in product(
            TASKS, [True, False], [False], [False], [1, 2], [True, False], [True, False]
        )
        if not (
            (int8 and half)
            or (task == "classify" and nms)
            or (task == "obb" and nms and (not TORCH_1_13 or IS_JETSON))  # obb nms fails on NVIDIA Jetson
            or (simplify and dynamic)  # onnxslim is slow when dynamic=True
        )
    ],
)
def test_export_onnx_matrix(task, dynamic, int8, half, batch, simplify, nms):
    """Test YOLO exports to ONNX format with various configurations and parameters."""
    file = YOLO(TASK2MODEL[task]).export(
        format="onnx",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        simplify=simplify,
        nms=nms,
        device=DEVICES[0],
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32, device=DEVICES[0])  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(True, reason="CUDA export tests disabled pending additional Ultralytics GPU server availability")
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [  # generate all combinations but exclude those where both int8 and half are True
        (task, dynamic, int8, half, batch)
        # Note: tests reduced below pending compute availability expansion as GPU CI runner utilization is high
        # for task, dynamic, int8, half, batch in product(TASKS, [True, False], [True, False], [True, False], [1, 2])
        for task, dynamic, int8, half, batch in product(TASKS, [True], [True], [False], [2])
        if not (int8 and half)  # exclude cases where both int8 and half are True
    ],
)
def test_export_engine_matrix(task, dynamic, int8, half, batch):
    """Test YOLO model export to TensorRT format for various configurations and run inference."""
    file = YOLO(TASK2MODEL[task]).export(
        format="engine",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        data=TASK2DATA[task],
        workspace=1,  # reduce workspace GB for less resource utilization during testing
        simplify=True,
        device=DEVICES[0],
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32, device=DEVICES[0])  # exported model inference
    Path(file).unlink()  # cleanup
    Path(file).with_suffix(".cache").unlink() if int8 else None  # cleanup INT8 cache


@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_train():
    """Test model training on a minimal dataset using available CUDA devices."""
    import os

    device = tuple(DEVICES) if len(DEVICES) > 1 else DEVICES[0]
    results = YOLO(MODEL).train(data="coco8.yaml", imgsz=64, epochs=1, device=device)  # requires imgsz>=64
    # NVIDIA Jetson only has one GPU and therefore skipping checks
    if not IS_JETSON:
        visible = eval(os.environ["CUDA_VISIBLE_DEVICES"])
        assert visible == device, f"Passed GPUs '{device}', but used GPUs '{visible}'"
        assert results is (None if len(DEVICES) > 1 else not None)  # DDP returns None, single-GPU returns metrics


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_predict_multiple_devices():
    """Validate model prediction consistency across CPU and CUDA devices."""
    model = YOLO("yolo11n.pt")

    # Test CPU
    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE)
    assert str(model.device) == "cpu"

    # Test CUDA
    cuda_device = f"cuda:{DEVICES[0]}"
    model = model.to(cuda_device)
    assert str(model.device) == cuda_device
    _ = model(SOURCE)
    assert str(model.device) == cuda_device

    # Test CPU again
    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE)
    assert str(model.device) == "cpu"

    # Test CUDA again
    model = model.to(cuda_device)
    assert str(model.device) == cuda_device
    _ = model(SOURCE)
    assert str(model.device) == cuda_device


@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_autobatch():
    """Check optimal batch size for YOLO model training using autobatch utility."""
    from ultralytics.utils.autobatch import check_train_batch_size

    check_train_batch_size(YOLO(MODEL).model.to(f"cuda:{DEVICES[0]}"), imgsz=128, amp=True)


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_utils_benchmarks():
    """Profile YOLO models for performance benchmarks."""
    from ultralytics.utils.benchmarks import ProfileModels

    # Pre-export a dynamic engine model to use dynamic inference
    YOLO(MODEL).export(format="engine", imgsz=32, dynamic=True, batch=1, device=DEVICES[0])
    ProfileModels(
        [MODEL],
        imgsz=32,
        half=False,
        min_time=1,
        num_timed_runs=3,
        num_warmup_runs=1,
        device=DEVICES[0],
    ).run()


@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_predict_sam():
    """Test SAM model predictions using different prompts."""
    from ultralytics import SAM
    from ultralytics.models.sam import Predictor as SAMPredictor

    model = SAM(WEIGHTS_DIR / "sam2.1_b.pt")
    model.info()

    # Run inference with various prompts
    model(SOURCE, device=DEVICES[0])
    model(SOURCE, bboxes=[439, 437, 524, 709], device=DEVICES[0])
    model(ASSETS / "zidane.jpg", points=[900, 370], device=DEVICES[0])
    model(ASSETS / "zidane.jpg", points=[900, 370], labels=[1], device=DEVICES[0])
    model(ASSETS / "zidane.jpg", points=[[900, 370]], labels=[1], device=DEVICES[0])
    model(ASSETS / "zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1], device=DEVICES[0])
    model(ASSETS / "zidane.jpg", points=[[[900, 370], [1000, 100]]], labels=[[1, 1]], device=DEVICES[0])

    # Test predictor
    predictor = SAMPredictor(
        overrides=dict(
            conf=0.25,
            task="segment",
            mode="predict",
            imgsz=1024,
            model=WEIGHTS_DIR / "mobile_sam.pt",
            device=DEVICES[0],
        )
    )
    predictor.set_image(ASSETS / "zidane.jpg")
    # predictor(bboxes=[439, 437, 524, 709])
    # predictor(points=[900, 370], labels=[1])
    predictor.reset_image()
