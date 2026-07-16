# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import os
import shutil
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from itertools import product
from pathlib import Path
from types import SimpleNamespace

if sys.platform == "win32":
    os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX2")

import pytest
import torch

from tests import MODEL, SOURCE
from tests.conftest import isolated_model_path
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS, _handle_deprecation, get_cfg
from ultralytics.engine.exporter import EXPORT_ENVS, Exporter, export_formats, validate_args
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
from ultralytics.utils.export.engine import best_onnx_opset, modelopt_quantize_onnx, torch2onnx
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
    file = YOLO(isolated_model).export(format="torchscript", imgsz=32, end2end=end2end)
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


def test_best_onnx_opset_caps_int8_only(monkeypatch):
    """Check opset>=21 is capped for ONNX Runtime INT8 quantization, not normal ONNX export."""
    from ultralytics.utils.export import engine

    class _Defs:
        @staticmethod
        def onnx_opset_version():
            return 25

    monkeypatch.setattr(engine, "TORCH_2_4", True)
    monkeypatch.setattr(engine, "TORCH_2_9", False)
    monkeypatch.setattr(engine.torch.onnx.utils, "_constants", SimpleNamespace(ONNX_MAX_OPSET=23), raising=False)
    onnx = SimpleNamespace(defs=_Defs())
    assert best_onnx_opset(onnx) == 22
    assert best_onnx_opset(onnx, cuda=True) == 20
    assert best_onnx_opset(onnx, quantize=8) == 20


def test_onnx_int8_quantize_excludes_non_weighted_ops(monkeypatch):
    """Check ONNX INT8 keeps only weighted ops quantized while preserving the string return contract."""
    import onnx
    import onnxruntime.quantization as ort_quantization

    from ultralytics.utils.export.onnx import onnx_int8_quantize

    calls = {}
    graph = SimpleNamespace(
        node=[
            SimpleNamespace(name="conv", op_type="Conv"),
            SimpleNamespace(name="pool", op_type="MaxPool"),
            SimpleNamespace(name="sigmoid", op_type="Sigmoid"),
        ]
    )

    monkeypatch.setattr(onnx, "load", lambda _: SimpleNamespace(graph=graph))
    monkeypatch.setattr(ort_quantization, "quantize_static", lambda *args, **kwargs: calls.update(kwargs))
    result = onnx_int8_quantize(Path("model.onnx"), Path("model_int8.onnx"), [], lambda x: x)
    assert result == "model_int8.onnx"
    assert calls["nodes_to_exclude"] == ["pool", "sigmoid"]


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
    assert _handle_deprecation({"half": True, "quantize": None})["quantize"] is None  # explicit quantize wins
    assert _handle_deprecation({"half": True, "quantize": 8})["quantize"] == 8  # explicit quantize still wins


def test_benchmark_forwards_legacy_precision(monkeypatch):
    """model.benchmark(half=True) must reach the benchmark call as quantize=16, not silently run FP32."""
    import ultralytics.utils.benchmarks as bm

    captured = {}
    monkeypatch.setattr(bm, "benchmark", lambda **kw: captured.update(kw) or {})
    YOLO(MODEL).benchmark(half=True, format="onnx", data="coco8.yaml")
    assert captured["quantize"] == 16, f"legacy half was dropped: quantize={captured.get('quantize')}"


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


def test_int8_calibration_validates_split():
    """Check INT8 calibration rejects dataset splits that do not exist."""
    exporter = object.__new__(Exporter)
    exporter.model = SimpleNamespace(task="obb")
    exporter.args = SimpleNamespace(data="coco8.yaml", split="trainval")
    exporter.imgsz = [32]
    with pytest.raises(FileNotFoundError, match="trainval"):
        exporter.get_int8_calibration_dataloader()


def test_export_rknn_batch_expansion(monkeypatch, tmp_path):
    """Check RKNN calibrates batch 1 before Toolkit expands to the requested batch."""
    calls = {}
    monkeypatch.setattr(
        "ultralytics.utils.export.rknn.onnx2rknn", lambda **kwargs: calls.update(kwargs) or kwargs["output_dir"]
    )
    monkeypatch.setattr("ultralytics.engine.exporter.file_size", lambda _: 1)

    image = tmp_path / "image.jpg"
    exporter = SimpleNamespace(
        args=SimpleNamespace(opset=None, quantize=8, name="rk3588", batch=8),
        im=torch.zeros(8, 3, 32, 32),
        file=tmp_path / "model.pt",
        metadata={},
        get_int8_calibration_dataloader=lambda prefix: SimpleNamespace(dataset=SimpleNamespace(im_files=[image])),
    )
    exporter.export_onnx = lambda: calls.update(onnx_batch=len(exporter.im)) or tmp_path / "model.onnx"
    Exporter.export_rknn(exporter)
    assert calls["onnx_batch"] == 1
    assert calls["batch"] == 8


@pytest.mark.parametrize("one2one", [False, True])
def test_export_hailo_compiles_hef(monkeypatch, tmp_path, one2one):
    """Check Hailo export compiles the static ONNX graph and writes deploy metadata."""
    calls = {}

    class ClientRunner:
        def __init__(self, hw_arch):
            calls["hw_arch"] = hw_arch

        def translate_onnx_model(self, path, name, end_node_names):
            calls.update(path=path, name=name, end_nodes=end_node_names)

        def get_hn_model(self):
            layers = [SimpleNamespace(name=f"model/conv{i}") for i in range(6)]
            return SimpleNamespace(get_output_layers=lambda: layers)

        def load_model_script(self, script):
            calls["script"] = script

        def optimize(self, calibration):
            calls["calibration"] = list(calibration())

        def compile(self):
            return b"hef"

    monkeypatch.setitem(sys.modules, "hailo_sdk_client", SimpleNamespace(ClientRunner=ClientRunner))
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(
            data=SimpleNamespace(Dataset=SimpleNamespace(from_generator=lambda generator, **kwargs: generator())),
            TensorSpec=lambda **kwargs: kwargs,
            float32=torch.float32,
        ),
    )
    monkeypatch.setattr("ultralytics.engine.exporter.file_size", lambda _: 1)
    onnx_file = tmp_path / "model.onnx"
    onnx_file.touch()
    output_dir = tmp_path / "model_hailo_model"
    output_dir.mkdir()
    (output_dir / "stale.txt").touch()
    head = SimpleNamespace(cv2=[None] * 3, one2one_cv2=[None] * 3, stride=(8, 16, 32))
    exporter = SimpleNamespace(
        args=SimpleNamespace(name="hailo15h", opset=None, conf=0, iou=0.7),
        model=SimpleNamespace(model=[None, head], names={0: "item"}, end2end=one2one),
        file=tmp_path / "model.pt",
        imgsz=[32, 32],
        metadata={"task": "detect"},
        export_onnx=lambda: onnx_file,
        get_int8_calibration_dataloader=lambda prefix: [{"img": torch.zeros(2, 3, 32, 32, dtype=torch.uint8)}],
    )

    output_dir = Path(Exporter.export_hailo(exporter))

    assert calls["hw_arch"] == "hailo15h"
    assert calls["name"] == "model"
    assert len(calls["calibration"]) == 2
    assert calls["calibration"][0].shape == (32, 32, 3)
    expected_nodes = (
        ["/model.1/one2one_cv2.0/one2one_cv2.0.2/Conv", "/model.1/one2one_cv2.1/one2one_cv2.1.2/Conv"]
        if one2one
        else ["/model.1/cv2.0/cv2.0.2/Conv", "/model.1/cv3.0/cv3.0.2/Conv"]
    )
    assert calls["end_nodes"][:2] == expected_nodes
    assert (output_dir / "model.hef").read_bytes() == b"hef"
    assert not (output_dir / "stale.txt").exists()
    if one2one:
        assert "nms_postprocess" not in calls["script"]
        assert "output_layer6], precision_mode=a16_w16)" in calls["script"]
        assert not (output_dir / "nms_config.json").exists()
    else:
        assert "change_output_activation(conv1, sigmoid)" in calls["script"]
        assert "nms_postprocess" in calls["script"]
        assert "allocator_param(width_splitter_defuse=disabled)" in calls["script"]
        assert '"classes": 1' in (output_dir / "nms_config.json").read_text()
        assert '"nms_scores_th": 0' in (output_dir / "nms_config.json").read_text()
    assert (output_dir / "metadata.yaml").is_file()
    assert not onnx_file.exists()


@pytest.mark.parametrize(
    "model,kwargs,error",
    [
        ("yolov10n.yaml", {}, "YOLOv8, YOLO11, and YOLO26"),
        ("yolo26n.yaml", {"end2end": False}, "requires end2end=True"),
    ],
)
def test_export_hailo_rejects_unsupported_configurations(monkeypatch, model, kwargs, error):
    """Check Hailo export rejects unsupported detection heads and YOLO26 output paths."""
    monkeypatch.setattr("ultralytics.engine.exporter.LINUX", True)
    monkeypatch.setattr("ultralytics.engine.exporter.ARM64", False)
    with pytest.raises(ValueError, match=error):
        YOLO(model).export(format="hailo", imgsz=32, **kwargs)


def test_modelopt_quantize_onnx_excludes_sigmoid(monkeypatch):
    """Check ModelOpt INT8 keeps Sigmoid unquantized to preserve confidence calibration (#24668)."""
    import onnx

    calls = {}
    graph = SimpleNamespace(input=[SimpleNamespace(name="images")])
    monkeypatch.setattr("ultralytics.utils.export.engine.check_requirements", lambda *args, **kwargs: None)
    monkeypatch.setitem(
        sys.modules, "modelopt.onnx.quantization", SimpleNamespace(quantize=lambda *a, **k: calls.update(k))
    )
    monkeypatch.setattr(onnx, "load", lambda *args, **kwargs: SimpleNamespace(graph=graph))
    modelopt_quantize_onnx("model.onnx", quantize=8, dataset=[{"img": torch.zeros(1, 3, 8, 8)}])
    assert calls["op_types_to_exclude"] == ["Sigmoid"]


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
@pytest.mark.parametrize("format", ["coreml", "mlmodel"])
def test_export_coreml(isolated_model, format, monkeypatch, tmp_path):
    """Test YOLO export to CoreML format and check for errors."""
    from ultralytics.utils.export import coreml

    quantize, torch2coreml = [], coreml.torch2coreml

    def capture_quantize(*args, **kwargs):
        quantize.append(kwargs["quantize"])
        return torch2coreml(*args, **kwargs)

    monkeypatch.setattr(coreml, "torch2coreml", capture_quantize)
    # Capture stdout and stderr
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        file = YOLO(isolated_model_path(tmp_path, WEIGHTS_DIR / "yolo11n.pt")).export(
            format=format, nms=True, imgsz=32, iou=0.42, conf=0.24
        )
        import coremltools as ct

        spec = ct.utils.load_spec(str(file))
        metadata = spec.description.metadata
        assert metadata.author and metadata.shortDescription and metadata.license and metadata.versionString
        assert metadata.userDefined["IoU threshold"] == "0.42"
        assert metadata.userDefined["Confidence threshold"] == "0.24"
        assert all(key in metadata.userDefined for key in ("names", "stride", "task"))
        assert next(iter(spec.pipeline.models[1].nonMaximumSuppression.stringClassLabels.vector)) == "person"
        assert [output.name for output in spec.description.output] == ["confidence", "coordinates"]
        if MACOS:
            file = YOLO(isolated_model).export(format="coreml", imgsz=32)
            YOLO(file)(SOURCE, imgsz=32)  # model prediction only supported on macOS for nms=False models

    # Check captured output for errors
    output = stdout.getvalue() + stderr.getvalue()
    assert quantize[0] == (16 if format == "coreml" else None)
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
        import coremltools as ct

        shape = ct.models.MLModel(str(file)).get_spec().description.output[0].type.multiArrayType.shape
        assert shape[-2] == 300
        if MACOS:
            YOLO(file)(SOURCE, imgsz=160)

    output = stdout.getvalue() + stderr.getvalue()
    assert "Error" not in output, f"RTDETR CoreML export produced errors: {output}"
    assert "You will not be able to run predict()" not in output, "RTDETR CoreML export has predict() error"


@pytest.mark.parametrize(
    "model, expected_nms",
    [("yolo11n.yaml", True), ("yolo11n-seg.yaml", False), ("yolo11n-pose.yaml", False)],
)
def test_export_coreml_nms_detect_only(model, expected_nms, monkeypatch):
    """Test CoreML 'nms=True' stays enabled for detect but warns and is forced off for other tasks."""
    captured = {}
    warnings = []

    def stub(self):
        captured["nms"] = self.args.nms
        captured["metadata_nms"] = self.metadata["args"]["nms"]

    monkeypatch.setattr(Exporter, "export_coreml", stub)  # skip the actual CoreML export
    monkeypatch.setattr("ultralytics.engine.exporter.LOGGER.warning", warnings.append)
    YOLO(model).export(format="coreml", nms=True, imgsz=32)
    assert captured["nms"] is expected_nms
    assert captured["metadata_nms"] is expected_nms
    assert any("only supported for detect models" in warning for warning in warnings) is not expected_nms


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


@pytest.mark.parametrize(
    "model,kwargs,error",
    [
        ("yolo11n.yaml", {"batch": 2, "dynamic": True, "nms": True}, "combining"),
        ("yolo11n-seg.yaml", {"nms": True}, "only supports detect and pose"),
        ("yolo11n-obb.yaml", {"nms": True}, "only supports detect and pose"),
    ],
)
def test_export_mnn_rejects_unsupported_nms(model, kwargs, error):
    """Test MNN rejects NMS combinations that fail or lose task outputs at runtime."""
    with pytest.raises(ValueError, match=error):
        YOLO(model).export(format="mnn", imgsz=32, **kwargs)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model,task,kwargs",
    [
        ("yolo11n.yaml", "detect", {"batch": 2, "dynamic": True}),
        ("yolo11n.yaml", "detect", {"nms": True}),
        ("yolo11n-pose.yaml", "pose", {"nms": True}),
    ],
)
def test_export_mnn_options(model, task, kwargs):
    """Test MNN dynamic shapes and supported embedded NMS tasks through inference."""
    batch = kwargs.get("batch", 1)
    file = YOLO(model).export(format="mnn", imgsz=32, **kwargs)
    assert len(YOLO(file, task=task)([SOURCE] * batch, imgsz=64 if kwargs.get("dynamic") else 32)) == batch
    Path(file).unlink()


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
@pytest.mark.parametrize("quantize,batch", [(8, 8), (16, 1)])
def test_export_rknn(isolated_model, quantize, batch):
    """Test YOLO export to RKNN format."""
    file = YOLO(isolated_model).export(format="rknn", imgsz=32, quantize=quantize, batch=batch, data="coco8.yaml")
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
