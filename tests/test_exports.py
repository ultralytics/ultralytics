# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import shutil
import sys
import threading
import time
import uuid
from contextlib import redirect_stderr, redirect_stdout
from itertools import product
from pathlib import Path

import pytest
import torch

from tests import SOURCE
from tests.conftest import isolated_model_path
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.engine.exporter import EXPORT_ENVS, export_formats
from ultralytics.utils import (
    ARM64,
    IS_DOCKER,
    IS_RASPBERRYPI,
    LINUX,
    MACOS,
    MACOS_VERSION,
    WEIGHTS_DIR,
    WINDOWS,
    checks,
)
from ultralytics.utils.export.engine import modelopt_quantize_onnx, torch2onnx
from ultralytics.utils.torch_utils import (
    TORCH_1_10,
    TORCH_1_11,
    TORCH_1_13,
    TORCH_2_0,
    TORCH_2_1,
    TORCH_2_8,
    TORCH_2_9,
    TORCH_2_12,
)


def skip_rpi_semantic(task):
    """Skip semantic segmentation export tests on Raspberry Pi due to memory constraints."""
    if IS_RASPBERRYPI and task == "semantic":
        pytest.skip("Semantic segmentation export tests are skipped on Raspberry Pi due to memory constraints.")


@pytest.mark.parametrize("end2end", [False, True])
def test_export_torchscript(end2end, isolated_model):
    """Test YOLO model export to TorchScript format for compatibility and correctness."""
    file = YOLO(isolated_model).export(format="torchscript", optimize=False, imgsz=32, end2end=end2end)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.parametrize("end2end", [False, True])
def test_export_onnx(end2end, isolated_model):
    """Test YOLO model export to ONNX format with dynamic axes."""
    file = YOLO(isolated_model).export(format="onnx", dynamic=True, imgsz=32, end2end=end2end)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.slow
def test_export_onnx_int8(isolated_model):
    """Test YOLO model export to INT8 ONNX format with calibration data."""
    file = YOLO(isolated_model).export(format="onnx", int8=True, data="coco8.yaml", fraction=0.25, imgsz=32)
    assert Path(file).name.endswith("_int8.onnx")
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference
    Path(file).unlink()  # cleanup


def test_modelopt_quantize_onnx_requires_int8_dataset():
    """Check INT8 ModelOpt quantization fails early without calibration data."""
    with pytest.raises(ValueError, match="requires a calibration dataset"):
        modelopt_quantize_onnx("model.onnx", int8=True)


def test_torch2onnx_serializes_concurrent_exports(monkeypatch, tmp_path):
    """Ensure ONNX exports do not overlap across worker threads."""
    active = 0
    max_active = 0
    errors = []
    state_lock = threading.Lock()

    def fake_export(*args, **kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    def export_model(index: int):
        try:
            torch2onnx(torch.nn.Identity(), torch.zeros(1, 3, 8, 8), str(tmp_path / f"export-{index}.onnx"))
        except Exception as error:  # pragma: no cover - assertion handled below
            errors.append(error)

    threads = [threading.Thread(target=export_model, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"Concurrent export errors: {errors}"
    assert max_active == 1, f"Expected max 1 concurrent export, got {max_active}"


@pytest.mark.skipif(not TORCH_2_1, reason="OpenVINO requires torch>=2.1")
@pytest.mark.parametrize("end2end", [False, True])
def test_export_openvino(end2end, isolated_model):
    """Test YOLO export to OpenVINO format for model inference compatibility."""
    file = YOLO(isolated_model).export(format="openvino", imgsz=32, end2end=end2end)
    if WINDOWS:
        # Ensure a unique export path per test to prevent OpenVINO file writes
        file = Path(file)
        file = file.rename(file.with_stem(f"{file.stem}-{uuid.uuid4()}"))
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_2_1, reason="OpenVINO requires torch>=2.1")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, nms, end2end",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, int8, half, batch, nms, end2end)
        for task, dynamic, int8, half, batch, nms, end2end in product(
            sorted(TASKS), [True, False], [True, False], [True, False], [1, 2], [True, False], [True]
        )
        if not ((int8 and half) or (task == "classify" and nms) or (end2end and nms))
    ],
)
# disable end2end=False test for now due to github runner OOM during openvino tests
def test_export_openvino_matrix(task, dynamic, int8, half, batch, nms, end2end):
    """Test YOLO model export to OpenVINO under various configuration matrix conditions."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(
        format="openvino",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        data=TASK2DATA[task],
        nms=nms,
        end2end=end2end,
    )
    if WINDOWS:
        # Use unique filenames due to Windows file permissions bug possibly due to latent threaded use
        file = Path(file)
        file = file.rename(file.with_stem(f"{file.stem}-{uuid.uuid4()}"))
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32, batch=batch)  # exported model inference
    shutil.rmtree(file, ignore_errors=True)  # retry in case of potential lingering multi-threaded file usage errors


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, simplify, nms, end2end",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, int8, half, batch, simplify, nms, end2end)
        for task, dynamic, int8, half, batch, simplify, nms, end2end in product(
            sorted(TASKS), [True, False], [False], [False], [1, 2], [True, False], [True, False], [True, False]
        )
        if not ((int8 and half) or (task == "classify" and nms) or (nms and not TORCH_1_13) or (end2end and nms))
    ],
)
def test_export_onnx_matrix(task, dynamic, int8, half, batch, simplify, nms, end2end):
    """Test YOLO export to ONNX format with various configurations and parameters."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(
        format="onnx",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        simplify=simplify,
        nms=nms,
        end2end=end2end,
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, nms, end2end",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, int8, half, batch, nms, end2end)
        for task, dynamic, int8, half, batch, nms, end2end in product(
            sorted(TASKS), [False, True], [False], [False, True], [1, 2], [True, False], [True, False]
        )
        if not ((task == "classify" and nms) or (end2end and nms))
    ],
)
def test_export_torchscript_matrix(task, dynamic, int8, half, batch, nms, end2end, tmp_path):
    """Test YOLO model export to TorchScript format under varied configurations."""
    skip_rpi_semantic(task)
    file = YOLO(isolated_model_path(tmp_path, WEIGHTS_DIR / TASK2MODEL[task])).export(
        format="torchscript", imgsz=32, dynamic=dynamic, int8=int8, half=half, batch=batch, nms=nms, end2end=end2end
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not MACOS, reason="CoreML inference only supported on macOS")
@pytest.mark.skipif(not TORCH_1_11, reason="CoreML export requires torch>=1.11")
@pytest.mark.skipif(
    MACOS and MACOS_VERSION and MACOS_VERSION >= "15", reason="CoreML YOLO26 matrix test crashes on macOS 15+"
)
@pytest.mark.parametrize(
    "task, dynamic, int8, half, nms, batch, end2end",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, int8, half, nms, batch, end2end)
        for task, dynamic, int8, half, nms, batch, end2end in product(
            sorted(TASKS), [True, False], [True, False], [True, False], [True, False], [1], [True, False]
        )
        if not (int8 and half)
        and not (task != "detect" and nms)
        and not (dynamic and nms)
        and not (task == "classify" and dynamic)
        and not (end2end and nms)
    ],
)
def test_export_coreml_matrix(task, dynamic, int8, half, nms, batch, end2end):
    """Test YOLO export to CoreML format with various parameter configurations."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(
        format="coreml",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        nms=nms,
        end2end=end2end,
    )
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference
    shutil.rmtree(file)  # cleanup


@pytest.mark.skipif(not TORCH_1_11, reason="CoreML export requires torch>=1.11")
@pytest.mark.skipif(WINDOWS, reason="CoreML not supported on Windows")  # RuntimeError: BlobWriter not loaded
@pytest.mark.skipif(LINUX and ARM64, reason="CoreML not supported on aarch64 Linux")
@pytest.mark.skipif(
    MACOS and checks.IS_PYTHON_MINIMUM_3_13,
    reason="coremltools deadlocks after OpenVINO on macOS Python 3.13 (conflicting OpenMP runtimes)",
)
def test_export_coreml(isolated_model):
    """Test YOLO export to CoreML format and check for errors."""
    # Capture stdout and stderr
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        YOLO(isolated_model).export(format="coreml", nms=True, imgsz=32)
        if MACOS:
            file = YOLO(isolated_model).export(format="coreml", imgsz=32)
            YOLO(file)(SOURCE, imgsz=32)  # model prediction only supported on macOS for nms=False models

    # Check captured output for errors
    output = stdout.getvalue() + stderr.getvalue()
    assert "Error" not in output, f"CoreML export produced errors: {output}"
    assert "You will not be able to run predict()" not in output, "CoreML export has predict() error"


@pytest.mark.skipif(not TORCH_1_11, reason="RTDETR CoreML export requires torch>=1.11")
@pytest.mark.skipif(WINDOWS, reason="CoreML not supported on Windows")
@pytest.mark.skipif(LINUX and ARM64, reason="CoreML not supported on aarch64 Linux")
@pytest.mark.skipif(
    MACOS and checks.IS_PYTHON_MINIMUM_3_13,
    reason="coremltools deadlocks after OpenVINO on macOS Python 3.13 (conflicting OpenMP runtimes)",
)
def test_export_coreml_rtdetr():
    """Test RT-DETR export to CoreML format and check for errors."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        file = YOLO(WEIGHTS_DIR / "rtdetr-l.pt").export(format="coreml", imgsz=160)
        if MACOS:
            YOLO(file)(SOURCE, imgsz=160)

    output = stdout.getvalue() + stderr.getvalue()
    assert "Error" not in output, f"RTDETR CoreML export produced errors: {output}"
    assert "You will not be able to run predict()" not in output, "RTDETR CoreML export has predict() error"


@pytest.mark.skipif(True, reason="Test disabled")
@pytest.mark.skipif(not LINUX, reason="TF suffers from install conflicts on Windows and macOS")
def test_export_pb(isolated_model):
    """Test YOLO export to TensorFlow's Protobuf (*.pb) format."""
    model = YOLO(isolated_model)
    file = model.export(format="pb", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)


@pytest.mark.skipif(True, reason="Test disabled as Paddle protobuf and ONNX protobuf requirements conflict.")
def test_export_paddle(isolated_model):
    """Test YOLO export to Paddle format, noting protobuf conflicts with ONNX."""
    YOLO(isolated_model).export(format="paddle", imgsz=32)


@pytest.mark.skipif(not TORCH_1_10, reason="MNN export requires torch>=1.10")
@pytest.mark.skipif(
    LINUX and checks.IS_PYTHON_MINIMUM_3_13,
    reason="MNN ONNX-parser protobuf conflicts with TensorFlow protobuf>=6.31.1 loaded earlier in the shared Python 3.13 test process",
)
def test_export_mnn(isolated_model):
    """Test YOLO export to MNN format (WARNING: MNN test must precede NCNN test or CI error on Windows)."""
    file = YOLO(isolated_model).export(format="mnn", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_1_10, reason="MNN export requires torch>=1.10")
@pytest.mark.parametrize(
    "task, int8, half, batch, end2end",
    [  # generate all combinations except for exclusion cases
        (task, int8, half, batch, end2end)
        for task, int8, half, batch, end2end in product(
            sorted(TASKS), [True, False], [True, False], [1, 2], [True, False]
        )
        if not (int8 and half)
    ],
)
def test_export_mnn_matrix(task, int8, half, batch, end2end):
    """Test YOLO export to MNN format considering various export configurations."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(format="mnn", imgsz=32, int8=int8, half=half, batch=batch, end2end=end2end)
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.skipif(not TORCH_2_0, reason="NCNN inference causes segfault on PyTorch<2.0")
def test_export_ncnn(isolated_model):
    """Test YOLO export to NCNN format."""
    file = YOLO(isolated_model).export(format="ncnn", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_2_0, reason="NCNN inference causes segfault on PyTorch<2.0")
@pytest.mark.parametrize("task, half, batch", list(product(sorted(TASKS), [True, False], [1])))
def test_export_ncnn_matrix(task, half, batch):
    """Test YOLO export to NCNN format considering various export configurations."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(format="ncnn", imgsz=32, half=half, batch=batch)
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference
    shutil.rmtree(file, ignore_errors=True)  # retry in case of potential lingering multi-threaded file usage errors


@pytest.mark.skipif(not TORCH_2_9, reason="IMX export requires torch>=2.9.0")
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_9, reason="IMX export requires Python>=3.9")
@pytest.mark.skipif(not LINUX, reason="IMX export only supported on Linux")
@pytest.mark.skipif(
    IS_RASPBERRYPI, reason="Test disabled as IMX export suffers from OOM (Out of Memory) on Raspberry Pi 5 16GB"
)
def test_export_imx():
    """Test YOLO export to IMX format."""
    model = YOLO("yolo11n.pt")  # IMX export only supports YOLO11
    file = model.export(format="imx", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)


@pytest.mark.slow
@pytest.mark.skipif(not LINUX or ARM64, reason="RKNN export only supported on non-aarch64 Linux")
@pytest.mark.parametrize("int8", [True, False])
def test_export_rknn(isolated_model, int8):
    """Test YOLO export to RKNN format."""
    file = YOLO(isolated_model).export(format="rknn", imgsz=32, int8=int8)
    assert next(Path(file).rglob("*.rknn"), None), f"RKNN export failed, no RKNN model found in: {file}"
    shutil.rmtree(file, ignore_errors=True)


# @pytest.mark.skipif(True, reason="Disabled for debugging ruamel.yaml installation required by executorch")
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10 or not TORCH_2_9, reason="Requires Python>=3.10 and Torch>=2.9.0")
@pytest.mark.skipif(WINDOWS, reason="Skipping test on Windows")
def test_export_executorch(isolated_model):
    """Test YOLO model export to ExecuTorch format."""
    file = YOLO(isolated_model).export(format="executorch", imgsz=32)
    assert Path(file).exists(), f"ExecuTorch export failed, directory not found: {file}"
    # Check that .pte file exists in the exported directory
    pte_file = Path(file) / "model.pte"
    assert pte_file.exists(), f"ExecuTorch .pte file not found: {pte_file}"
    # Check that metadata.yaml exists
    metadata_file = Path(file) / "metadata.yaml"
    assert metadata_file.exists(), f"ExecuTorch metadata.yaml not found: {metadata_file}"
    # Note: Inference testing skipped as ExecuTorch requires special runtime setup
    shutil.rmtree(file, ignore_errors=True)  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not LINUX or ARM64, reason="LiteRT export only supported on Linux x86")
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="litert-torch requires Python>=3.10")
@pytest.mark.parametrize("task, int8", [(task, int8) for task in sorted(TASKS) for int8 in (False, True)])
def test_export_litert_matrix(task, int8):
    """Test YOLO export to LiteRT format (FP32 and INT8) for various task types."""
    file = YOLO(TASK2MODEL[task]).export(format="litert", imgsz=32, int8=int8)
    assert Path(file).exists(), f"LiteRT export failed for task '{task}', directory not found: {file}"
    assert next(Path(file).glob("*.tflite"), None), f"LiteRT .tflite file not found for task '{task}': {file}"
    assert (Path(file) / "metadata.yaml").exists(), f"LiteRT metadata.yaml not found for task '{task}': {file}"
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference
    shutil.rmtree(file, ignore_errors=True)  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10 or not TORCH_2_9, reason="Requires Python>=3.10 and Torch>=2.9.0")
@pytest.mark.skipif(WINDOWS, reason="Skipping test on Windows")
@pytest.mark.parametrize("task", sorted(TASKS))
def test_export_executorch_matrix(task):
    """Test YOLO export to ExecuTorch format for various task types."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(format="executorch", imgsz=32)
    assert Path(file).exists(), f"ExecuTorch export failed for task '{task}', directory not found: {file}"
    # Check that .pte file exists in the exported directory
    pte_file = Path(file) / "model.pte"
    assert pte_file.exists(), f"ExecuTorch .pte file not found for task '{task}': {pte_file}"
    # Check that metadata.yaml exists
    metadata_file = Path(file) / "metadata.yaml"
    assert metadata_file.exists(), f"ExecuTorch metadata.yaml not found for task '{task}': {metadata_file}"
    # Note: Inference testing skipped as ExecuTorch requires special runtime setup
    shutil.rmtree(file, ignore_errors=True)  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_2_8 or TORCH_2_12, reason="Axelera export requires 2.8.0<=torch<2.12.0")
@pytest.mark.skipif(checks.IS_PYTHON_MINIMUM_3_13, reason="Axelera devkit 1.6.0 does not support Python 3.13")
@pytest.mark.skipif(
    not LINUX or (ARM64 and IS_DOCKER),
    reason="Axelera export is only supported on Linux and is not supported on ARM64 Docker",
)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Test disabled due to OOM (Out of Memory) issues on Raspberry Pi 5 16GB")
def test_export_axelera(isolated_model):
    """Test YOLO export to Axelera format."""
    # For faster testing, use a smaller calibration dataset (32 image size crashes axelera export, so 64 is used)
    file = YOLO(isolated_model).export(format="axelera", imgsz=64, data="coco8.yaml")
    assert Path(file).exists(), f"Axelera export failed, directory not found: {file}"
    # Note: Inference testing skipped as it requires Axelera hardware
    shutil.rmtree(file, ignore_errors=True)  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not LINUX or ARM64, reason="DEEPX export only supported on non-aarch64 Linux")
@pytest.mark.skipif(
    not checks.IS_PYTHON_3_12, reason="Requires Python 3.12; dx-com 2.3.0 does not provide Python 3.13 wheels"
)
def test_export_deepx(isolated_model):
    """Test YOLO export to DEEPX format."""
    file = YOLO(isolated_model).export(format="deepx", imgsz=32)
    assert Path(file).exists(), f"DEEPX export failed, directory not found: {file}"
    # Note: Inference testing skipped as it requires DEEPX hardware
    shutil.rmtree(file, ignore_errors=True)  # cleanup


@pytest.mark.skipif(
    not (WINDOWS or (LINUX and ARM64)) or sys.version_info < (3, 11),
    reason="onnxruntime-qnn ships prebuilt wheels only for Windows (x64/ARM64) and Linux ARM64 on Python>=3.11",
)
def test_export_qnn(isolated_model):
    """Test YOLO export to Qualcomm QNN format via the ONNX Runtime QNN Execution Provider."""
    import importlib.util

    # QNN EP ships either as the 'onnxruntime_qnn' plugin module (Windows/Linux-aarch64) or as a provider library
    # bundled in onnxruntime/capi (Linux x86-64). Skip cleanly only when neither is present.
    has_qnn = importlib.util.find_spec("onnxruntime_qnn") is not None
    if not has_qnn and importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime

        capi = Path(onnxruntime.__file__).parent / "capi"
        has_qnn = (capi / "libonnxruntime_providers_qnn.so").exists() or (
            capi / "onnxruntime_providers_qnn.dll"
        ).exists()
    if not has_qnn:
        pytest.skip("onnxruntime-qnn / QNN Execution Provider not available")
    file = YOLO(isolated_model).export(format="qnn", imgsz=32)
    assert Path(file).is_file() and file.endswith("_qnn.onnx"), f"QNN export failed, no context binary found: {file}"
    # Note: on-device inference is not exercised here as it requires Qualcomm Snapdragon hardware
    Path(file).unlink(missing_ok=True)  # cleanup


@pytest.mark.parametrize("env", [k for k, v in EXPORT_ENVS.items() if k != "base" or v["smoke"]])
def test_export_env_has_smoke(env):
    """Ensure every non-base export environment declares a build-time smoke export."""
    assert EXPORT_ENVS[env]["smoke"], f"export env '{env}' has no smoke command"


def test_every_format_env_is_registered():
    """Ensure every export format points at a registered export environment."""
    for fmt, env in zip(export_formats()["Argument"], export_formats()["Env"]):
        assert env in EXPORT_ENVS, f"format '{fmt}' references unknown env '{env}'"
