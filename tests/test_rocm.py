# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from itertools import product
from pathlib import Path

import pytest
import torch

from tests import MODEL, ROCM_DEVICE_COUNT, ROCM_IS_AVAILABLE, SOURCE
from ultralytics import YOLO
from ultralytics.cfg import TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR
from ultralytics.utils.checks import check_amp
from ultralytics.utils.torch_utils import TORCH_1_13, attempt_compile, parse_device

# ROCm has no NVML equivalent for idle-GPU selection, so take the first two visible devices to mirror test_cuda.py
DEVICES = list(range(ROCM_DEVICE_COUNT))[:2] if ROCM_IS_AVAILABLE else []

MIGRAPHX_AVAILABLE = False
if ROCM_IS_AVAILABLE:
    try:
        import onnxruntime

        MIGRAPHX_AVAILABLE = "MIGraphXExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        pass


def test_checks():
    """Validate ROCm HIP settings against torch CUDA functions."""
    assert torch.cuda.is_available() == ROCM_IS_AVAILABLE
    assert torch.cuda.device_count() == ROCM_DEVICE_COUNT
    if ROCM_IS_AVAILABLE:
        assert torch.version.hip


@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_amp():
    """Test AMP training checks on ROCm GPU; xfail on hardware known to report AMP anomalies."""
    if "R9700" in torch.cuda.get_device_name(DEVICES[0]):
        pytest.xfail("AMD Radeon AI PRO R9700 reports AMP anomalies; AMP disabled on this GPU")
    model = YOLO(MODEL).model.to(f"cuda:{DEVICES[0]}")
    assert check_amp(model)


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.parametrize(
    "task, dynamic, batch, simplify, nms",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, batch, simplify, nms)
        for task, dynamic, batch, simplify, nms in product(
            sorted(TASKS), [True, False], [1, 2], [True, False], [True, False]
        )
        if not ((task == "classify" and nms) or (task == "obb" and nms and not TORCH_1_13))
    ],
)
def test_rocm_export_onnx_matrix(task, dynamic, batch, simplify, nms):
    """Test YOLO ONNX export and inference across all tasks and parameter combinations on ROCm."""
    if MIGRAPHX_AVAILABLE and task == "segment":
        pytest.xfail("MIGraphX v7.2 does not support Resize with keep_aspect_ratio_policy used in segment models")
    if MIGRAPHX_AVAILABLE and task == "pose":
        pytest.xfail("MIGraphX v7.2 fails to compile GPU kernels for pose models on gfx950 (std::bad_alloc)")
    file = YOLO(TASK2MODEL[task]).export(
        format="onnx",
        imgsz=32,
        dynamic=dynamic,
        batch=batch,
        simplify=simplify,
        nms=nms,
        device=DEVICES[0],
    )
    model = YOLO(file)
    results = model([SOURCE] * batch, imgsz=64 if dynamic else 32, device=DEVICES[0])
    assert len(results) == batch
    if MIGRAPHX_AVAILABLE:
        assert "MIGraphXExecutionProvider" in model.predictor.model.session.get_providers()
    Path(file).unlink()  # cleanup


# NOTE: test_cuda.py::test_export_engine_matrix has no ROCm counterpart, TensorRT is NVIDIA only.


@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.parametrize("nc", [1, 3])
def test_rocm_semantic_loss_all_ignore_amp(nc):
    """All-ignore guard must stay finite with large fp16 logits, where sum() overflows to inf (AMP is a GPU path)."""
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SemanticSegmentationModel
    from ultralytics.utils.loss import SemanticSegmentationLoss

    model = SemanticSegmentationModel(cfg="yolo26-sem.yaml", nc=nc, verbose=False)
    model.args = get_cfg()
    loss_fn = SemanticSegmentationLoss(model)
    preds = (torch.randn(1, nc, 64, 64, device=f"cuda:{DEVICES[0]}") + 50).half().requires_grad_()
    loss, items = loss_fn(preds, {"semantic_mask": torch.full((1, 64, 64), 255, dtype=torch.long)})
    assert torch.isfinite(loss).all() and all(torch.isfinite(x).all() for x in items.values())
    loss.backward()
    assert preds.grad is not None


@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_train():
    """Test model training on a minimal dataset using available ROCm devices."""
    device = tuple(DEVICES) if len(DEVICES) > 1 else DEVICES[0]
    expected = parse_device(device)  # canonical torch indices, e.g. physical ids translate under external CVD
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    YOLO(MODEL).train(data="coco8-grayscale.yaml", imgsz=64, epochs=1, device=DEVICES[0], batch=-1)
    model = YOLO(MODEL)
    model.train(data="coco8.yaml", imgsz=64, epochs=1, device=device, batch=15, compile=True)
    assert model.trainer.args.device == expected, "trained on wrong GPUs"
    assert model.trainer.device.index == int(expected.split(",")[0]), "trained on wrong GPU"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible, "CUDA_VISIBLE_DEVICES must never be mutated"
    results = YOLO(MODEL).train(data="coco128.yaml", imgsz=64, epochs=1, device=device, batch=15, val=False)
    # Both single-GPU and DDP return metrics (recovered from the saved checkpoint under DDP)
    assert results is not None


@pytest.mark.skipif(not DEVICES or max(DEVICES) == 0, reason="requires a ROCm device with nonzero index")
def test_rocm_train_cold_process_nonzero_device():
    """Train on a nonzero GPU index in a fresh process with cold HIP state, reproducing real CLI usage.

    A warm pytest process has HIP initialized, so a subprocess without CUDA_VISIBLE_DEVICES is the only way to reproduce
    cold-start device selection as on production pods (e.g. Ultralytics Platform).
    """
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    cmd = ["yolo", "train", f"model={MODEL}", "data=coco8.yaml", "imgsz=32", "epochs=1", f"device={max(DEVICES)}"]
    subprocess.run(cmd, check=True, env=env)


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_predict_multiple_devices():
    """Validate prediction consistency when switching between CPU and ROCm devices."""
    model = YOLO(MODEL)
    rocm_device = f"cuda:{DEVICES[0]}"

    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE, imgsz=32)
    assert str(model.device) == "cpu"

    model = model.to(rocm_device)
    assert str(model.device) == rocm_device
    _ = model(SOURCE, imgsz=32)
    assert str(model.device) == rocm_device

    model = model.cpu()
    assert str(model.device) == "cpu"
    _ = model(SOURCE, imgsz=32)
    assert str(model.device) == "cpu"

    model = model.to(rocm_device)
    assert str(model.device) == rocm_device
    _ = model(SOURCE, imgsz=32)
    assert str(model.device) == rocm_device


@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_track_exported_model():
    """Track with an exported model on GPU; exported backends return raw preds as a single Tensor."""
    file = YOLO(MODEL).export(format="torchscript", imgsz=160, device=DEVICES[0])
    results = YOLO(file).track(SOURCE, imgsz=160, device=DEVICES[0])
    assert len(results[0].boxes)
    Path(file).unlink()  # cleanup


@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_autobatch():
    """Check optimal batch size for YOLO model training on ROCm using autobatch."""
    from ultralytics.utils.autobatch import check_train_batch_size

    check_train_batch_size(YOLO(MODEL).model.to(f"cuda:{DEVICES[0]}"), imgsz=128, amp=True)


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_utils_benchmarks(isolated_model):
    """Profile YOLO models for performance benchmarks on ROCm (skips TRT, not available on AMD)."""
    from ultralytics.utils.benchmarks import ProfileModels

    ProfileModels(
        [isolated_model],
        imgsz=32,
        quantize=32,
        min_time=1,
        num_timed_runs=3,
        num_warmup_runs=1,
        device=DEVICES[0],
        trt=False,
    ).run()


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_predict_sam():
    """Test SAM model predictions on ROCm GPU using various prompts and SAMPredictor."""
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
        overrides={
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "imgsz": 1024,
            "model": WEIGHTS_DIR / "mobile_sam.pt",
            "device": DEVICES[0],
            "quantize": 16,
        }
    )
    predictor.set_image(ASSETS / "zidane.jpg")
    predictor.reset_image()


# ROCm-specific coverage below: MIGraphX Execution Provider paths that have no test_cuda.py counterpart.


@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_detection():
    """Test ONNX export followed by inference using the MIGraphX Execution Provider on AMD GPU."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32)
    model = YOLO(file)
    results = model(SOURCE, imgsz=32, device=DEVICES[0])
    assert results
    assert "MIGraphXExecutionProvider" in model.predictor.model.session.get_providers()
    Path(file).unlink()


@pytest.mark.skipif(not ROCM_IS_AVAILABLE, reason="ROCm/HIP not available")
def test_rocm_cpu_fallback():
    """Test ONNX export and inference on CPU when running on a ROCm system."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32)
    model = YOLO(file)
    results = model(SOURCE, imgsz=32, device="cpu")
    assert results
    assert "CPUExecutionProvider" in model.predictor.model.session.get_providers()
    Path(file).unlink()


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
def test_rocm_torch_compile():
    """Test torch.compile with inductor backend on ROCm using a lightweight inference pass."""
    device = torch.device(f"cuda:{DEVICES[0]}")
    model = YOLO(MODEL).model.to(device).eval()
    model = attempt_compile(model, device=device, imgsz=32, warmup=True, mode="default")
    x = torch.zeros(1, 3, 32, 32, device=device)
    with torch.inference_mode():
        y = model(x)
    assert y is not None


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
def test_rocm_pytorch_val():
    """Test validation pipeline with PyTorch model on ROCm GPU device."""
    metrics = YOLO(MODEL).val(data="coco8.yaml", device=DEVICES[0], imgsz=32)
    assert metrics is not None


@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_export_simplify():
    """Test ONNX export with simplify=True on ROCm (exercises exporter.py ROCm branch)."""
    file = YOLO(MODEL).export(format="onnx", simplify=True, imgsz=32)
    model = YOLO(file)
    results = model(SOURCE, imgsz=32, device=DEVICES[0])
    assert results
    assert "MIGraphXExecutionProvider" in model.predictor.model.session.get_providers()
    Path(file).unlink()


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_dynamic_batch_inference():
    """Test dynamic ONNX export with varying batch sizes on MIGraphX EP."""
    file = YOLO(MODEL).export(format="onnx", dynamic=True, imgsz=32)
    model = YOLO(file)
    for batch_size in [1, 2, 4]:
        results = model([SOURCE] * batch_size, imgsz=32, device=DEVICES[0])
        assert len(results) == batch_size
    Path(file).unlink()


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_dynamic_imgsz_inference():
    """Test dynamic ONNX inference with different image sizes on MIGraphX EP."""
    file = YOLO(MODEL).export(format="onnx", dynamic=True, imgsz=32)
    model = YOLO(file)
    for sz in [32, 64]:
        results = model(SOURCE, imgsz=sz, device=DEVICES[0])
        assert results
    Path(file).unlink()


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_half_precision():
    """Test FP16 ONNX export and inference on ROCm with MIGraphX EP."""
    file = YOLO(MODEL).export(format="onnx", quantize=16, imgsz=32)
    model = YOLO(file)
    results = model(SOURCE, imgsz=32, device=DEVICES[0])
    assert results
    Path(file).unlink()


@pytest.mark.slow
@pytest.mark.skipif(not DEVICES, reason="No ROCm devices available")
@pytest.mark.skipif(not MIGRAPHX_AVAILABLE, reason="MIGraphX Execution Provider not available")
def test_rocm_onnx_io_binding():
    """Test that static ONNX inference on ROCm uses IO binding for zero-copy GPU transfers."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32)
    model = YOLO(file)
    results = model(SOURCE, imgsz=32, device=DEVICES[0])
    assert results

    backend = model.predictor.model
    assert not backend.dynamic, "Static ONNX model should not be marked dynamic"
    assert backend.use_io_binding, "IO binding should be enabled for static ONNX on GPU"
    assert "MIGraphXExecutionProvider" in backend.session.get_providers()

    assert hasattr(backend, "bindings") and backend.bindings, "Output bindings should be pre-allocated"
    for tensor in backend.bindings:
        assert tensor.is_cuda, f"Output tensor should be on GPU, got {tensor.device}"
        assert tensor.device.index == DEVICES[0], f"Output tensor on wrong device: {tensor.device}"

    Path(file).unlink()
