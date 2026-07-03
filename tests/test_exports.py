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
from types import SimpleNamespace

import pytest
import torch

from tests import SOURCE
from tests.conftest import isolated_model_path
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS, _handle_deprecation, get_cfg
from ultralytics.engine.exporter import EXPORT_ENVS, export_formats, validate_args
from ultralytics.utils import (
    ARM64,
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
    TORCH_2_9,
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
@pytest.mark.parametrize("precision", [{"int8": True}, {"quantize": 8}])
def test_export_onnx_int8(isolated_model, precision):
    """Test INT8 ONNX export via both the legacy int8 alias and the unified quantize arg."""
    file = YOLO(isolated_model).export(format="onnx", data="coco8.yaml", fraction=0.25, imgsz=32, **precision)
    assert Path(file).name.endswith("_int8.onnx")
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference
    Path(file).unlink()  # cleanup


def test_quantize_canonicalization():
    """Quantize accepts 8/16/32 (int or str) and w-notation, canonicalizing to the int form (unset stays None)."""
    for value, expected in [
        (8, 8),
        (16, 16),
        (32, 32),
        ("8", 8),
        ("int8", 8),
        ("INT8", 8),
        ("w8a8", 8),
        ("W8A8", 8),
        ("fp16", 16),
        ("Fp16", 16),
        ("w16a16", 16),
        ("fp32", 32),
        ("fP32", 32),
        ("w8a16", "w8a16"),
        ("W8a16", "w8a16"),
        ("w8a32", "w8a32"),
        ("W8A32", "w8a32"),
    ]:
        assert get_cfg(overrides={"quantize": value}).quantize == expected
    assert get_cfg().quantize is None  # unset default is FP32
    with pytest.raises(ValueError, match="quantize"):
        get_cfg(overrides={"quantize": "x4"})
    with pytest.raises(ValueError, match="quantize"):
        get_cfg(overrides={"quantize": "a8w8"})


def test_quantize_deprecation():
    """Legacy half/int8 forward to the unified quantize arg in all modes; int8 wins on conflict."""
    assert _handle_deprecation({"int8": True})["quantize"] == 8
    assert _handle_deprecation({"half": True})["quantize"] == 16
    assert _handle_deprecation({"half": True, "int8": True})["quantize"] == 8  # int8 wins
    assert "half" not in _handle_deprecation({"half": True})  # legacy flag is removed after forwarding


def test_qnn_quantize_requires_w8a16():
    """QNN exports are W8A16; explicit INT8 activation quantization is not supported."""
    valid_args = ["batch", "data", "dynamic", "fraction", "keras", "nms"]
    validate_args("qnn", SimpleNamespace(quantize="w8a16"), valid_args)
    with pytest.raises(AssertionError, match=r"quantize=8 \(INT8\) is not supported"):
        validate_args("qnn", SimpleNamespace(quantize=8), valid_args)


def test_modelopt_quantize_onnx_requires_int8_dataset():
    """Check INT8 ModelOpt quantization fails early without calibration data."""
    with pytest.raises(ValueError, match="requires a calibration dataset"):
        modelopt_quantize_onnx("model.onnx", quantize=8)


def test_modelopt_quantize_onnx_excludes_sigmoid(monkeypatch):
    """Verify that INT8 ModelOpt quantization excludes precision-sensitive ops from quantization.

    Monkeypatches modelopt.quantize to capture the actual kwargs passed, confirming
    op_types_to_exclude includes Sigmoid and Softmax.
    See https://github.com/ultralytics/ultralytics/issues/24668
    """
    from ultralytics.utils.export.engine import _FP32_PRESERVED_OPS

    # The constant should include both Sigmoid and Softmax
    assert "Sigmoid" in _FP32_PRESERVED_OPS, "op_types_to_exclude should include 'Sigmoid'"
    assert "Softmax" in _FP32_PRESERVED_OPS, "op_types_to_exclude should include 'Softmax'"

    # Stub check_requirements so it doesn't trigger a network install attempt in CI/local
    monkeypatch.setattr("ultralytics.utils.export.engine.check_requirements", lambda *a, **k: None)

    # Monkeypatch modelopt.quantize to capture kwargs without actually running it
    captured = {}

    def fake_quantize(*args, **kwargs):
        captured.update(kwargs)
        # Write a dummy file so the caller doesn't fail
        out = kwargs.get("output_path", "dummy.int8.onnx")
        Path(out).touch()
        return out

    # modelopt_quantize_onnx imports modelopt.quantize at call time, so patch sys.modules
    fake_module = SimpleNamespace(quantize=fake_quantize)
    monkeypatch.setitem(sys.modules, "modelopt.onnx.quantization", fake_module)
    # Also need onnx since the function imports it at the top
    import onnx

    monkeypatch.setattr(
        onnx, "load", lambda *a, **k: SimpleNamespace(graph=SimpleNamespace(input=[SimpleNamespace(name="images")]))
    )
    monkeypatch.setattr(onnx, "save", lambda *a, **k: None)

    # Create a dummy ONNX file to satisfy the path check
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        modelopt_quantize_onnx(onnx_path, quantize=8, dataset=iter([{"img": torch.zeros(1, 3, 8, 8)}]))
        # Verify op_types_to_exclude was actually passed with the right values
        assert "op_types_to_exclude" in captured, "modelopt.quantize should receive op_types_to_exclude"
        assert set(captured["op_types_to_exclude"]) == set(_FP32_PRESERVED_OPS), (
            f"op_types_to_exclude should match _FP32_PRESERVED_OPS, got {captured['op_types_to_exclude']}"
        )
    finally:
        Path(onnx_path).unlink(missing_ok=True)
        Path(onnx_path.replace(".onnx", ".int8.onnx")).unlink(missing_ok=True)


def test_trt10_constrain_sensitive_activations_call_site(monkeypatch):
    """Verify that onnx2engine calls the constraint helper only for TRT 10 INT8.

    Monkeypatches _constrain_sensitive_activations_fp32 and tensorrt to verify
    the helper is invoked when use_int8=True and is_trt11=False, and NOT invoked
    for FP16 or TRT 11+.
    See https://github.com/ultralytics/ultralytics/issues/24668
    """
    import onnx
    from onnx import TensorProto, helper
    from unittest.mock import MagicMock

    from ultralytics.utils.export.engine import onnx2engine

    # Create a minimal valid ONNX model
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 1, 8, 8])
    graph = helper.make_graph([], "test", [inp], [out])
    model = helper.make_model(graph)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(model, f.name)
        onnx_path = f.name

    # Track calls to the constraint helper
    call_count = {"n": 0}

    def fake_constrain(network, config, trt, prefix=""):
        call_count["n"] += 1
        return 0

    monkeypatch.setattr("ultralytics.utils.export.engine._constrain_sensitive_activations_fp32", fake_constrain)

    # Build a comprehensive fake tensorrt module
    def make_fake_trt(version="10.7.0"):
        fake_config = MagicMock()
        fake_config.set_flag = MagicMock()
        fake_config.profiling_verbosity = MagicMock()
        fake_config.int8_calibrator = None
        fake_config.add_optimization_profile = MagicMock()
        fake_config.set_calibration_profile = MagicMock()
        fake_config.default_device_type = None
        fake_config.DLA_core = None

        fake_network = MagicMock()
        fake_network.num_inputs = 1
        fake_network.num_outputs = 1
        fake_network.num_layers = 0
        fake_input = MagicMock(name="images", shape=[1, 3, 8, 8], dtype="float32")
        fake_output = MagicMock(name="output0", shape=[1, 1, 8, 8], dtype="float32")
        fake_network.get_input = MagicMock(return_value=fake_input)
        fake_network.get_output = MagicMock(return_value=fake_output)
        fake_network.get_layer = MagicMock()

        fake_builder = MagicMock()
        fake_builder.create_builder_config = MagicMock(return_value=fake_config)
        fake_builder.create_network = MagicMock(return_value=fake_network)
        fake_builder.build_serialized_network = MagicMock(return_value=b"\x00" * 8)
        fake_builder.platform_has_fast_fp16 = True
        fake_builder.platform_has_fast_int8 = True

        fake_parser = MagicMock()
        fake_parser.parse_from_file = MagicMock(return_value=True)

        class FakeLogger:
            INFO = 1

            class Severity:
                VERBOSE = 0

            def __init__(self, *a, **k):
                pass

        class FakeIInt8Calibrator:
            def __init__(self):
                pass

        trt = SimpleNamespace(
            __version__=version,
            Logger=FakeLogger,
            Builder=MagicMock(return_value=fake_builder),
            OnnxParser=MagicMock(return_value=fake_parser),
            IInt8Calibrator=FakeIInt8Calibrator,
            BuilderFlag=SimpleNamespace(
                INT8=1,
                FP16=2,
                OBEY_PRECISION_CONSTRAINTS=3,
                GPU_FALLBACK=4,
                EXPLICIT_BATCH=5,
            ),
            LayerType=SimpleNamespace(ACTIVATION=1, SOFTMAX=2, CONVOLUTION=3),
            ActivationType=SimpleNamespace(SIGMOID=10, RELU=12),
            MemoryPoolType=SimpleNamespace(WORKSPACE=1),
            NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=1),
            ProfilingVerbosity=SimpleNamespace(DETAILED=1),
            CalibrationAlgoType=SimpleNamespace(ENTROPY_CALIBRATION_2=1, MINMAX_CALIBRATION=2),
            DeviceType=SimpleNamespace(DLA=1),
            float32="float32",
        )
        return trt

    try:
        # Fake dataset with batch_size attribute (needed by EngineCalibrator)
        class FakeDataset:
            batch_size = 1

            def __iter__(self):
                yield {"img": torch.zeros(1, 3, 8, 8)}

        fake_dataset = FakeDataset()

        # Case 1: INT8 on TRT 10 — helper SHOULD be called
        call_count["n"] = 0
        monkeypatch.setitem(sys.modules, "tensorrt", make_fake_trt("10.7.0"))
        monkeypatch.setattr("ultralytics.utils.export.engine.check_version", lambda *a, **k: None)
        monkeypatch.setattr("ultralytics.utils.export.engine.is_jetson", lambda **k: False)
        monkeypatch.setattr("ultralytics.utils.export.engine.is_dgx", lambda: False)
        monkeypatch.setattr("ultralytics.utils.export.engine.check_tensorrt", lambda *a, **k: None)
        onnx2engine(onnx_path, quantize=8, dataset=fake_dataset, prefix="[TEST]")
        assert call_count["n"] == 1, f"Helper should be called once for TRT 10 INT8, got {call_count['n']}"

        # Case 2: FP16 on TRT 10 — helper should NOT be called
        call_count["n"] = 0
        onnx2engine(onnx_path, quantize=16, prefix="[TEST]")
        assert call_count["n"] == 0, f"Helper should NOT be called for FP16, got {call_count['n']}"

        # Case 3: INT8 on TRT 11 — helper should NOT be called (ModelOpt handles it)
        call_count["n"] = 0
        monkeypatch.setitem(sys.modules, "tensorrt", make_fake_trt("11.0.0"))
        # TRT 11 path calls modelopt_quantize_onnx before parsing; stub it to just copy the file
        monkeypatch.setattr("ultralytics.utils.export.engine.modelopt_quantize_onnx", lambda *a, **k: onnx_path)
        onnx2engine(onnx_path, quantize=8, dataset=fake_dataset, prefix="[TEST]")
        assert call_count["n"] == 0, f"Helper should NOT be called for TRT 11 INT8, got {call_count['n']}"
    finally:
        Path(onnx_path).unlink(missing_ok=True)


def test_trt10_constrain_sensitive_activations_execution():
    """Execute the TRT 10 Sigmoid/Softmax FP32 constraint logic with a stubbed tensorrt module.

    Verifies that OBEY_PRECISION_CONSTRAINTS is set, Sigmoid (ACTIVATION) and Softmax (SOFTMAX)
    layers are constrained to FP32, non-sensitive layers are left untouched, and the correct
    count is returned — all without requiring a GPU or TensorRT installation.
    See https://github.com/ultralytics/ultralytics/issues/24668
    """
    from unittest.mock import MagicMock

    from ultralytics.utils.export.engine import _constrain_sensitive_activations_fp32

    # Build a fake tensorrt module — Softmax is LayerType.SOFTMAX, not an ACTIVATION
    fake_trt = SimpleNamespace(
        BuilderFlag=SimpleNamespace(OBEY_PRECISION_CONSTRAINTS=999),
        LayerType=SimpleNamespace(ACTIVATION=1, SOFTMAX=2, CONVOLUTION=3),
        ActivationType=SimpleNamespace(SIGMOID=10, RELU=12),  # no SOFTMAX in ActivationType
        float32="float32",
        IActivationLayer=None,  # simulate TRT 10.7+ where cast constructor is unavailable
    )

    # Fake layers: Sigmoid (ACTIVATION), Softmax (SOFTMAX), ReLU (ACTIVATION), Conv (CONVOLUTION)
    sigmoid_layer = MagicMock(name="Sigmoid_0")
    sigmoid_layer.type = fake_trt.LayerType.ACTIVATION
    sigmoid_layer.num_outputs = 1
    sigmoid_layer.name = "Sigmoid_0"

    softmax_layer = MagicMock(name="Softmax_0")
    softmax_layer.type = fake_trt.LayerType.SOFTMAX  # Softmax is a separate layer type in TRT
    softmax_layer.num_outputs = 1
    softmax_layer.name = "Softmax_0"

    relu_layer = MagicMock(name="Relu_0")
    relu_layer.type = fake_trt.LayerType.ACTIVATION
    relu_layer.num_outputs = 1
    relu_layer.name = "Relu_0"

    conv_layer = MagicMock(name="Conv_0")
    conv_layer.type = fake_trt.LayerType.CONVOLUTION
    conv_layer.num_outputs = 1
    conv_layer.name = "Conv_0"

    fake_network = SimpleNamespace(
        num_layers=4, get_layer=MagicMock(side_effect=[sigmoid_layer, softmax_layer, relu_layer, conv_layer])
    )
    fake_config = MagicMock()

    count = _constrain_sensitive_activations_fp32(fake_network, fake_config, fake_trt, prefix="[TEST]")

    # OBEY_PRECISION_CONSTRAINTS flag should be set
    fake_config.set_flag.assert_called_once_with(fake_trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # Sigmoid layer should be constrained to FP32
    assert sigmoid_layer.precision == fake_trt.float32, "Sigmoid layer precision should be set to float32"
    sigmoid_layer.set_output_type.assert_called_once_with(0, fake_trt.float32)
    # Softmax layer should also be constrained to FP32 (it's LayerType.SOFTMAX, not ACTIVATION)
    assert softmax_layer.precision == fake_trt.float32, "Softmax layer precision should be set to float32"
    softmax_layer.set_output_type.assert_called_once_with(0, fake_trt.float32)
    # Non-sensitive layers should NOT be constrained
    relu_layer.set_output_type.assert_not_called()
    conv_layer.set_output_type.assert_not_called()
    # Count should be 2 (Sigmoid + Softmax)
    assert count == 2, f"Expected 2 constrained layers (Sigmoid + Softmax), got {count}"
    # Log which layers were constrained for reviewer visibility
    constrained = [sigmoid_layer.name, softmax_layer.name]
    print(f"\nConstrained layers ({count}): {constrained}")


def test_trt10_binding_version_detection_branches():
    """Verify that _is_precision_sensitive_activation takes the correct branch for each TRT binding shape.

    Simulates three TRT binding variants:
    - TRT 10.3-: IActivationLayer(layer) cast works, exposes .activation_type enum
    - TRT 10.7+: IActivationLayer exists but cast constructor raises TypeError; fall back to name matching
    - TRT 10.7+ (no IActivationLayer at all): fall back to name matching
    See https://github.com/ultralytics/ultralytics/issues/24668
    """
    from unittest.mock import MagicMock

    from ultralytics.utils.export.engine import _is_precision_sensitive_activation

    sigmoid_name = "Sigmoid_42"
    relu_name = "Relu_7"

    # --- Case 1: TRT 10.3- — typed cast works, enum-based detection ---
    class FakeActivationLayer103:
        """Simulates the IActivationLayer cast result in TRT 10.3-."""

        def __init__(self, layer):
            self.activation_type = layer._fake_act_type

    fake_trt_103 = SimpleNamespace(
        IActivationLayer=FakeActivationLayer103,
        ActivationType=SimpleNamespace(SIGMOID=10, RELU=12),
    )
    sigmoid_layer_103 = MagicMock(name=sigmoid_name)
    sigmoid_layer_103._fake_act_type = fake_trt_103.ActivationType.SIGMOID
    relu_layer_103 = MagicMock(name=relu_name)
    relu_layer_103._fake_act_type = fake_trt_103.ActivationType.RELU

    assert _is_precision_sensitive_activation(sigmoid_layer_103, fake_trt_103) is True, (
        "TRT 10.3: Sigmoid should be detected via typed cast + enum"
    )
    assert _is_precision_sensitive_activation(relu_layer_103, fake_trt_103) is False, (
        "TRT 10.3: ReLU should not be detected via typed cast + enum"
    )

    # --- Case 2: TRT 10.7+ — IActivationLayer exists but cast raises TypeError ---
    class BrokenActivationLayer:
        """Simulates TRT 10.7+ where the cast constructor was removed."""

        def __init__(self, layer):
            raise TypeError("IActivationLayer() constructor not available in TRT 10.7+")

    fake_trt_107 = SimpleNamespace(
        IActivationLayer=BrokenActivationLayer,
        ActivationType=SimpleNamespace(SIGMOID=10, RELU=12),
    )
    sigmoid_layer_107 = MagicMock(name=sigmoid_name)
    sigmoid_layer_107.name = sigmoid_name
    relu_layer_107 = MagicMock(name=relu_name)
    relu_layer_107.name = relu_name

    assert _is_precision_sensitive_activation(sigmoid_layer_107, fake_trt_107) is True, (
        "TRT 10.7+: Sigmoid should be detected via name fallback when cast fails"
    )
    assert _is_precision_sensitive_activation(relu_layer_107, fake_trt_107) is False, (
        "TRT 10.7+: ReLU should not be detected via name fallback"
    )

    # --- Case 3: TRT 10.7+ — IActivationLayer attribute missing entirely ---
    fake_trt_nocast = SimpleNamespace(
        ActivationType=SimpleNamespace(SIGMOID=10, RELU=12),
    )
    sigmoid_layer_nocast = MagicMock(name=sigmoid_name)
    sigmoid_layer_nocast.name = sigmoid_name
    relu_layer_nocast = MagicMock(name=relu_name)
    relu_layer_nocast.name = relu_name

    assert _is_precision_sensitive_activation(sigmoid_layer_nocast, fake_trt_nocast) is True, (
        "TRT 10.7+ (no IActivationLayer): Sigmoid should be detected via name fallback"
    )
    assert _is_precision_sensitive_activation(relu_layer_nocast, fake_trt_nocast) is False, (
        "TRT 10.7+ (no IActivationLayer): ReLU should not be detected via name fallback"
    )


def test_openvino_int8_ignores_sigmoid():
    """Verify that the OpenVINO INT8 exporter includes Sigmoid in its ignored scope.

    This documents the cross-exporter invariant: both OpenVINO and TensorRT exclude
    Sigmoid from INT8 quantization to preserve confidence-score calibration.
    See https://github.com/ultralytics/ultralytics/issues/24668
    """
    from ultralytics.engine.exporter import Exporter

    # export_openvino is wrapped by @try_export. The inner function is captured
    # in the closure. We iterate the closure cells to find the function object
    # (robust against cell ordering changes).
    outer = Exporter.export_openvino
    inner = None
    for cell in outer.__closure__ or ():
        if callable(cell.cell_contents):
            inner = cell.cell_contents
            break
    assert inner is not None, "Could not find inner function in try_export closure"
    import inspect

    source = inspect.getsource(inner)
    assert "IgnoredScope" in source, "OpenVINO export should use IgnoredScope for INT8"
    assert "Sigmoid" in source, "OpenVINO IgnoredScope should include 'Sigmoid'"


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
    "task, dynamic, quantize, batch, nms, end2end",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, quantize, batch, nms, end2end)
        for task, dynamic, quantize, batch, nms, end2end in product(
            sorted(TASKS), [True, False], [8, 16], [1, 2], [True, False], [True]
        )
        if not ((task == "classify" and nms) or (end2end and nms))
    ],
)
# disable end2end=False test for now due to github runner OOM during openvino tests
def test_export_openvino_matrix(task, dynamic, quantize, batch, nms, end2end):
    """Test YOLO model export to OpenVINO under various configuration matrix conditions."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(
        format="openvino",
        imgsz=32,
        dynamic=dynamic,
        quantize=quantize,
        batch=batch,
        data=TASK2DATA[task],  # use the smallest task datasets for fast INT8 calibration
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
    "task, dynamic, batch, simplify, nms, end2end",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, batch, simplify, nms, end2end)
        for task, dynamic, batch, simplify, nms, end2end in product(
            sorted(TASKS), [True, False], [1, 2], [True, False], [True, False], [True, False]
        )
        if not ((task == "classify" and nms) or (nms and not TORCH_1_13) or (end2end and nms))
    ],
)
def test_export_onnx_matrix(task, dynamic, batch, simplify, nms, end2end):
    """Test YOLO export to ONNX format with various configurations and parameters."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(
        format="onnx",
        imgsz=32,
        dynamic=dynamic,
        batch=batch,
        simplify=simplify,
        nms=nms,
        end2end=end2end,
    )
    r = YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference
    if task == "semantic":
        assert r[0].semantic_mask is not None
        assert r[0].semantic_mask.data.dtype in {torch.uint8, torch.int32}
    Path(file).unlink()  # cleanup


def test_export_onnx_semantic_dnn():
    """Test semantic ONNX class-map output with OpenCV DNN."""
    skip_rpi_semantic("semantic")
    file = YOLO(TASK2MODEL["semantic"]).export(format="onnx", imgsz=32)
    r = YOLO(file).predict(SOURCE, imgsz=32, dnn=True)
    assert r[0].semantic_mask is not None
    Path(file).unlink()


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, dynamic, batch, nms, end2end",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, batch, nms, end2end)
        for task, dynamic, batch, nms, end2end in product(
            sorted(TASKS), [False, True], [1, 2], [True, False], [True, False]
        )
        if not ((task == "classify" and nms) or (end2end and nms))
    ],
)
def test_export_torchscript_matrix(task, dynamic, batch, nms, end2end, tmp_path):
    """Test YOLO model export to TorchScript format under varied configurations."""
    skip_rpi_semantic(task)
    file = YOLO(isolated_model_path(tmp_path, WEIGHTS_DIR / TASK2MODEL[task])).export(
        format="torchscript", imgsz=32, dynamic=dynamic, batch=batch, nms=nms, end2end=end2end
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
    "task, dynamic, quantize, nms, batch, end2end",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, quantize, nms, batch, end2end)
        for task, dynamic, quantize, nms, batch, end2end in product(
            sorted(TASKS), [True, False], [8, 16], [True, False], [1], [True, False]
        )
        if not (task != "detect" and nms)
        and not (dynamic and nms)
        and not (task == "classify" and dynamic)
        and not (end2end and nms)
    ],
)
def test_export_coreml_matrix(task, dynamic, quantize, nms, batch, end2end):
    """Test YOLO export to CoreML format with various parameter configurations."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(
        format="coreml",
        imgsz=32,
        dynamic=dynamic,
        quantize=quantize,
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
    "task, quantize, batch, end2end",
    [  # generate all combinations except for exclusion cases
        (task, quantize, batch, end2end)
        for task, quantize, batch, end2end in product(sorted(TASKS), [8, 16], [1, 2], [True, False])
    ],
)
def test_export_mnn_matrix(task, quantize, batch, end2end):
    """Test YOLO export to MNN format considering various export configurations."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(format="mnn", imgsz=32, quantize=quantize, batch=batch, end2end=end2end)
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.skipif(not TORCH_2_0, reason="NCNN inference causes segfault on PyTorch<2.0")
def test_export_ncnn(isolated_model):
    """Test YOLO export to NCNN format."""
    file = YOLO(isolated_model).export(format="ncnn", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_2_0, reason="NCNN inference causes segfault on PyTorch<2.0")
@pytest.mark.parametrize("task, quantize, batch", list(product(sorted(TASKS), [16], [1])))
def test_export_ncnn_matrix(task, quantize, batch):
    """Test YOLO export to NCNN format considering various export configurations."""
    skip_rpi_semantic(task)
    file = YOLO(TASK2MODEL[task]).export(format="ncnn", imgsz=32, quantize=quantize, batch=batch)
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
    file = model.export(format="imx", imgsz=32, data="coco8.yaml")
    YOLO(file)(SOURCE, imgsz=32)


@pytest.mark.slow
@pytest.mark.skipif(not LINUX or ARM64, reason="RKNN export only supported on non-aarch64 Linux")
@pytest.mark.parametrize("quantize", [8, 16])
def test_export_rknn(isolated_model, quantize):
    """Test YOLO export to RKNN format."""
    file = YOLO(isolated_model).export(format="rknn", imgsz=32, quantize=quantize, data="coco8.yaml")
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
@pytest.mark.skipif(not (MACOS or (LINUX and not ARM64)), reason="LiteRT export only supported on Linux x86 and macOS")
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="litert-torch requires Python>=3.10")
@pytest.mark.parametrize(
    "task, quantize",
    [(task, quantize) for task in sorted(TASKS) for quantize in (None, 8, "w8a16", "w8a32")],
)
def test_export_litert_matrix(task, quantize):
    """Test YOLO export to LiteRT format (FP32, static INT8, static w8a16, and dynamic w8a32) for various tasks."""
    file = Path(YOLO(TASK2MODEL[task]).export(format="litert", imgsz=32, quantize=quantize))
    assert file.is_file() and file.suffix == ".tflite", f"LiteRT export is not a single .tflite for '{task}': {file}"
    # Contract: exports keep float32 graph I/O (int8/int16 stays internal) so downstream runtimes feed/read floats
    # without boundary (de)quantization; an int8/int16 I/O regression would silently break on-device consumers.
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(file))
    interpreter.allocate_tensors()
    io_details = interpreter.get_input_details() + interpreter.get_output_details()
    assert all(d["dtype"] == np.float32 for d in io_details), (
        f"LiteRT '{task}' quantize={quantize} must keep float32 I/O, got {[d['dtype'] for d in io_details]}"
    )
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference (also exercises the embedded metadata)
    file.unlink()  # cleanup


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
