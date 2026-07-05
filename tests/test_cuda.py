# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from itertools import product
from pathlib import Path

import pytest
import torch

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE, MODEL, SOURCE
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, IS_JETSON, WEIGHTS_DIR
from ultralytics.utils.autodevice import GPUInfo
from ultralytics.utils.checks import check_amp, check_tensorrt
from ultralytics.utils.torch_utils import TORCH_1_13, parse_device

# Try to find idle devices if CUDA is available
DEVICES = []
if CUDA_IS_AVAILABLE:
    if IS_JETSON:
        DEVICES = [0]  # NVIDIA Jetson only has one GPU and does not fully support pynvml library
    else:
        gpu_info = GPUInfo()
        gpu_info.print_status()
        autodevice_fraction = __import__("os").environ.get("YOLO_AUTODEVICE_FRACTION_FREE", 0.3)
        if idle_gpus := gpu_info.select_idle_gpu(
            count=2,
            min_memory_fraction=autodevice_fraction,
            min_util_fraction=autodevice_fraction,
        ):
            DEVICES = idle_gpus


def test_checks():
    """Validate CUDA settings against torch CUDA functions."""
    assert torch.cuda.is_available() == CUDA_IS_AVAILABLE
    assert torch.cuda.device_count() == CUDA_DEVICE_COUNT


@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_amp():
    """Test AMP training checks."""
    model = YOLO("yolo26n.pt").model.to(f"cuda:{DEVICES[0]}")
    assert check_amp(model)


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
@pytest.mark.parametrize(
    "task, dynamic, batch, simplify, nms",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, batch, simplify, nms)
        for task, dynamic, batch, simplify, nms in product(
            sorted(TASKS), [True, False], [1, 2], [True, False], [True, False]
        )
        if not ((task == "classify" and nms) or (task == "obb" and nms and (not TORCH_1_13 or IS_JETSON)))
    ],
)
def test_export_onnx_matrix(task, dynamic, batch, simplify, nms):
    """Test YOLO exports to ONNX format with various configurations and parameters."""
    file = YOLO(TASK2MODEL[task]).export(
        format="onnx",
        imgsz=32,
        dynamic=dynamic,
        batch=batch,
        simplify=simplify,
        nms=nms,
        device=DEVICES[0],
        # opset=20 if nms else None,  # fix ONNX Runtime errors with NMS
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32, device=DEVICES[0])  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
@pytest.mark.parametrize(
    "task, dynamic, quantize, batch",
    [
        (task, dynamic, quantize, batch)
        # Limit Jetson task coverage for slow CI speed; full task coverage remains on GPU CI.
        # for task, dynamic, quantize, batch in product(TASKS, [True, False], [8, 16], [1, 2])
        for task, dynamic, quantize, batch in product(["detect"] if IS_JETSON else sorted(TASKS), [True], [8, 16], [2])
    ],
)
def test_export_engine_matrix(task, dynamic, quantize, batch):
    """Test YOLO model export to TensorRT format for various configurations and run inference."""
    check_tensorrt()
    import tensorrt as trt

    is_trt11 = int(trt.__version__.split(".", 1)[0]) >= 11
    if not is_trt11 and quantize == 8 and dynamic:
        # TensorRT 7-10 calibrator path cannot quantize dynamic-shape models; TensorRT 11 uses ModelOpt explicit Q/DQ
        pytest.skip("INT8 + dynamic export requires explicit quantization, available on TensorRT 11+")

    file = YOLO(TASK2MODEL[task]).export(
        format="engine",
        imgsz=32,
        dynamic=dynamic,
        quantize=quantize,
        batch=batch,
        data=TASK2DATA[task],  # use the smallest task datasets for fast INT8 calibration
        workspace=1,  # reduce workspace GB for less resource utilization during testing
        simplify=True,
        device=DEVICES[0],
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32, device=DEVICES[0])  # exported model inference
    Path(file).unlink()  # cleanup
    if quantize == 8:
        Path(file).with_suffix(".cache").unlink(missing_ok=True)  # cleanup TensorRT 7-10 INT8 calibration cache
        Path(file).with_suffix(".int8.onnx").unlink(missing_ok=True)  # cleanup TensorRT 11 ModelOpt INT8 ONNX
    if quantize == 16:
        Path(file).with_suffix(".fp16.onnx").unlink(missing_ok=True)  # cleanup TensorRT 11 ModelOpt FP16 ONNX


@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
@pytest.mark.skipif(IS_JETSON, reason="Edge devices not intended for training")
def test_train():
    """Test model training on a minimal dataset using available CUDA devices."""
    device = tuple(DEVICES) if len(DEVICES) > 1 else DEVICES[0]
    expected = parse_device(device)  # canonical torch indices, e.g. physical ids translate under external CVD
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    results = YOLO(MODEL).train(data="coco8-grayscale.yaml", imgsz=64, epochs=1, device=DEVICES[0], batch=-1)
    model = YOLO(MODEL)
    results = model.train(data="coco8.yaml", imgsz=64, epochs=1, device=device, batch=15, compile=True)
    assert model.trainer.args.device == expected, "trained on wrong GPUs"
    assert model.trainer.device.index == int(expected.split(",")[0]), "trained on wrong GPU"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible, "CUDA_VISIBLE_DEVICES must never be mutated"
    results = YOLO(MODEL).train(data="coco128.yaml", imgsz=64, epochs=1, device=device, batch=15, val=False)
    # Both single-GPU and DDP return metrics (recovered from the saved checkpoint under DDP)
    assert results is not None


@pytest.mark.skipif(not DEVICES or max(DEVICES) == 0, reason="requires an idle CUDA device with nonzero index")
@pytest.mark.skipif(IS_JETSON, reason="Edge devices not intended for training")
def test_train_cold_process_nonzero_device():
    """Train on a nonzero GPU index in a fresh process with cold CUDA state, reproducing real CLI usage.

    A warm pytest process has CUDA initialized, so a subprocess without CUDA_VISIBLE_DEVICES is the only way to
    reproduce cold-start device selection as on production pods (e.g. Ultralytics Platform).
    """
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    cmd = ["yolo", "train", f"model={MODEL}", "data=coco8.yaml", "imgsz=32", "epochs=1", f"device={max(DEVICES)}"]
    subprocess.run(cmd, check=True, env=env)


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_predict_multiple_devices():
    """Validate model prediction consistency across CPU and CUDA devices."""
    model = YOLO("yolo26n.pt")

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

    check_train_batch_size(YOLO(MODEL).model.to(f"cuda:{DEVICES[0]}"), imgsz=64, amp=True)


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No CUDA devices available")
def test_utils_benchmarks(isolated_model):
    """Profile YOLO models for performance benchmarks."""
    from ultralytics.utils.benchmarks import ProfileModels

    # Pre-export a dynamic engine model to use dynamic inference
    YOLO(isolated_model).export(format="engine", imgsz=32, dynamic=True, batch=1, device=DEVICES[0])
    ProfileModels(
        [isolated_model],
        imgsz=32,
        quantize=32,
        min_time=1,
        num_timed_runs=3,
        num_warmup_runs=1,
        device=DEVICES[0],
    ).run()


@pytest.mark.slow
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
            quantize=16,
        )
    )
    predictor.set_image(ASSETS / "zidane.jpg")
    # predictor(bboxes=[439, 437, 524, 709])
    # predictor(points=[900, 370], labels=[1])
    predictor.reset_image()
