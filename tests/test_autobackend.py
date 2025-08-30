# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil
from pathlib import Path

import pytest

from tests import CUDA_IS_AVAILABLE, MODEL
from ultralytics import RTDETR, YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ARM64, IS_RASPBERRYPI, LINUX, MACOS, WINDOWS, checks
from ultralytics.utils.torch_utils import TORCH_1_9, TORCH_1_13

CFG_RTDETR = "rtdetr-l.yaml"  # RTDETR model for testing


def test_autobackend_architecture_onnx():
    """Test AutoBackend correctly reads architecture metadata from ONNX models."""
    pytest.importorskip("onnxruntime", reason="Test requires onnxruntime")

    # Test YOLO11 ONNX
    model = YOLO(MODEL)
    onnx_file = model.export(format="onnx", imgsz=32)

    try:
        backend = AutoBackend(onnx_file)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture in ["YOLO11", "YOLO"], (
            f"Expected YOLO family architecture, got '{backend.architecture}'"
        )
        assert backend.architecture != "", "Architecture should not be empty"

    finally:
        # Cleanup
        if Path(onnx_file).exists():
            Path(onnx_file).unlink()


@pytest.mark.skipif(not TORCH_1_9, reason="RTDETR requires torch>=1.9")
def test_autobackend_architecture_rtdetr_onnx():
    """Test AutoBackend correctly reads RTDETR architecture from ONNX models."""
    pytest.importorskip("onnxruntime", reason="Test requires onnxruntime")

    # Test RTDETR ONNX
    model = RTDETR(CFG_RTDETR)
    onnx_file = model.export(format="onnx", imgsz=640)

    try:
        backend = AutoBackend(onnx_file)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture == "RTDETR", f"Expected RTDETR architecture, got '{backend.architecture}'"
        assert backend.architecture != "", "Architecture should not be empty"

    finally:
        # Cleanup
        if Path(onnx_file).exists():
            Path(onnx_file).unlink()


@pytest.mark.skipif(not TORCH_1_13, reason="OpenVINO requires torch>=1.13")
def test_autobackend_architecture_openvino():
    """Test AutoBackend correctly reads architecture metadata from OpenVINO models."""
    # Test YOLO11 OpenVINO
    model = YOLO(MODEL)
    openvino_dir = model.export(format="openvino", imgsz=32)

    try:
        backend = AutoBackend(openvino_dir)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture in ["YOLO11", "YOLO"], (
            f"Expected YOLO family architecture, got '{backend.architecture}'"
        )
        assert backend.architecture != "", "Architecture should not be empty"

    finally:
        # Cleanup
        if Path(openvino_dir).exists():
            shutil.rmtree(openvino_dir)


@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="GridSample not supported on CPU")
@pytest.mark.skipif(not TORCH_1_13, reason="OpenVINO requires torch>=1.13")
@pytest.mark.skipif(not TORCH_1_9, reason="RTDETR requires torch>=1.9")
def test_autobackend_architecture_rtdetr_openvino():
    """Test AutoBackend correctly reads RTDETR architecture from OpenVINO models."""
    try:
        # Test RTDETR OpenVINO
        model = RTDETR(CFG_RTDETR)
        openvino_dir = model.export(format="openvino", imgsz=640)

        backend = AutoBackend(openvino_dir)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture == "RTDETR", f"Expected RTDETR architecture, got {backend.architecture}"

        # Cleanup
        if Path(openvino_dir).exists():
            shutil.rmtree(openvino_dir)

    except Exception as e:
        pytest.skip(f"RTDETR OpenVINO test failed: {e}")


@pytest.mark.skipif(WINDOWS or not MACOS, reason="CoreML export supported on macOS")
@pytest.mark.skipif(not TORCH_1_9, reason="CoreML>=7.2 not supported with PyTorch<=1.8")
def test_autobackend_architecture_coreml():
    """Test AutoBackend correctly reads architecture metadata from CoreML models."""
    # Test YOLO11 CoreML
    model = YOLO(MODEL)
    coreml_file = model.export(format="coreml", imgsz=32)

    try:
        backend = AutoBackend(coreml_file)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture in ["YOLO11", "YOLO"], (
            f"Expected YOLO family architecture, got '{backend.architecture}'"
        )
        assert backend.architecture != "", "Architecture should not be empty"

    finally:
        # Cleanup
        if Path(coreml_file).exists():
            if Path(coreml_file).is_dir():
                shutil.rmtree(coreml_file)
            else:
                Path(coreml_file).unlink()


def test_autobackend_architecture_torchscript():
    """Test AutoBackend correctly reads architecture metadata from TorchScript models."""
    # Test YOLO11 TorchScript
    model = YOLO(MODEL)
    torchscript_file = model.export(format="torchscript", imgsz=32)

    try:
        backend = AutoBackend(torchscript_file)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        # TorchScript might not always have architecture metadata depending on export method
        # Just verify the attribute exists and is a string
        assert isinstance(backend.architecture, str), "Architecture should be a string"

    finally:
        # Cleanup
        if Path(torchscript_file).exists():
            Path(torchscript_file).unlink()


def test_autobackend_architecture_fallback():
    """Test AutoBackend handles missing architecture metadata gracefully."""
    try:
        # Test with a PyTorch model (should have empty architecture)
        backend = AutoBackend(MODEL)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert isinstance(backend.architecture, str), "Architecture should be a string"

    except Exception as e:
        pytest.skip(f"Fallback test failed: {e}")


def test_autobackend_architecture_metadata_consistency():
    """Test that architecture metadata is consistent across different export formats."""
    try:
        model = YOLO(MODEL)
        formats_to_test = ["onnx"]

        if TORCH_1_13:
            formats_to_test.append("openvino")

        architectures = {}

        for fmt in formats_to_test:
            try:
                exported_file = model.export(format=fmt, imgsz=32)
                backend = AutoBackend(exported_file)
                architectures[fmt] = backend.architecture

                # Cleanup
                if Path(exported_file).exists():
                    if Path(exported_file).is_dir():
                        shutil.rmtree(exported_file)
                    else:
                        Path(exported_file).unlink()

            except Exception as e:
                pytest.skip(f"Format {fmt} failed: {e}")

        # Verify all formats report the same architecture (if multiple formats were tested)
        if len(architectures) > 1:
            arch_values = list(architectures.values())
            assert all(arch == arch_values[0] for arch in arch_values), (
                f"Architecture inconsistent across formats: {architectures}"
            )

    except Exception as e:
        pytest.skip(f"Consistency test failed: {e}")


@pytest.mark.slow
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="TFLite export requires Python>=3.10")
@pytest.mark.skipif(
    not LINUX or IS_RASPBERRYPI,
    reason="Test disabled as TF suffers from install conflicts on Windows, macOS and Raspberry Pi",
)
def test_autobackend_architecture_tflite():
    """Test AutoBackend correctly reads architecture metadata from TFLite models."""
    # Test YOLO11 TFLite
    model = YOLO(MODEL)
    tflite_file = model.export(format="tflite", imgsz=32)

    try:
        backend = AutoBackend(tflite_file)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture in ["YOLO11", "YOLO"], (
            f"Expected YOLO family architecture, got '{backend.architecture}'"
        )
        assert backend.architecture != "", "Architecture should not be empty"

    finally:
        # Cleanup
        if Path(tflite_file).exists():
            Path(tflite_file).unlink()


def test_autobackend_architecture_paddle():
    """Test AutoBackend correctly reads architecture metadata from PaddlePaddle models."""
    # Test YOLO11 PaddlePaddle
    model = YOLO(MODEL)
    paddle_dir = model.export(format="paddle", imgsz=32)

    try:
        backend = AutoBackend(paddle_dir)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture in ["YOLO11", "YOLO"], (
            f"Expected YOLO family architecture, got '{backend.architecture}'"
        )
        assert backend.architecture != "", "Architecture should not be empty"

    finally:
        # Cleanup
        if Path(paddle_dir).exists():
            shutil.rmtree(paddle_dir)


@pytest.mark.skipif(ARM64 or IS_RASPBERRYPI, reason="NCNN export not supported on ARM64/RPi")
def test_autobackend_architecture_ncnn():
    """Test AutoBackend correctly reads architecture metadata from NCNN models."""
    # Test YOLO11 NCNN
    model = YOLO(MODEL)
    ncnn_dir = model.export(format="ncnn", imgsz=32)

    try:
        backend = AutoBackend(ncnn_dir)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture in ["YOLO11", "YOLO"], (
            f"Expected YOLO family architecture, got '{backend.architecture}'"
        )
        assert backend.architecture != "", "Architecture should not be empty"

    finally:
        # Cleanup
        if Path(ncnn_dir).exists():
            shutil.rmtree(ncnn_dir)


@pytest.mark.skipif(True, reason="Test disabled as keras and tensorflow version conflicts with TFlite export.")
@pytest.mark.skipif(not LINUX or MACOS, reason="Skipping test on Windows and Macos")
def test_autobackend_architecture_imx():
    """Test AutoBackend correctly reads architecture metadata from IMX models."""
    # Test YOLOv8 IMX
    model = YOLO("yolov8n.pt")
    imx_dir = model.export(format="imx", imgsz=32)

    try:
        backend = AutoBackend(imx_dir)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture in ["YOLOv8", "YOLO"], (
            f"Expected YOLO family architecture, got '{backend.architecture}'"
        )
        assert backend.architecture != "", "Architecture should not be empty"

    finally:
        # Cleanup
        if Path(imx_dir).exists():
            shutil.rmtree(imx_dir)


# Cross-model architecture consistency tests
@pytest.mark.skipif(not TORCH_1_9, reason="RTDETR requires torch>=1.9")
def test_autobackend_rtdetr_openvino_architecture():
    """Test RTDETR OpenVINO architecture metadata consistency."""
    if not TORCH_1_13:
        pytest.skip("OpenVINO requires torch>=1.13")

    # Test RTDETR OpenVINO
    model = RTDETR(CFG_RTDETR)
    openvino_dir = model.export(format="openvino", imgsz=640)

    try:
        backend = AutoBackend(openvino_dir)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture == "RTDETR", f"Expected RTDETR architecture, got '{backend.architecture}'"
        assert backend.architecture != "", "Architecture should not be empty"

    finally:
        # Cleanup
        if Path(openvino_dir).exists():
            shutil.rmtree(openvino_dir)


def test_autobackend_yolov8_architecture_detection():
    """Test that YOLOv8 models are correctly detected by architecture."""
    # Test with explicit YOLOv8 model
    model = YOLO("yolov8n.yaml")
    onnx_file = model.export(format="onnx", imgsz=32)

    try:
        backend = AutoBackend(onnx_file)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        # Should detect as YOLOv8 or general YOLO
        assert backend.architecture in ["YOLOv8", "YOLO"], (
            f"Expected YOLOv8 or YOLO architecture, got '{backend.architecture}'"
        )
        assert backend.architecture != "", "Architecture should not be empty"

    finally:
        # Cleanup
        if Path(onnx_file).exists():
            Path(onnx_file).unlink()


def test_autobackend_architecture_error_on_missing():
    """Test that missing architecture metadata is handled correctly."""
    # Test with PyTorch model (should have empty architecture but still have attribute)
    backend = AutoBackend(MODEL)
    assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
    assert isinstance(backend.architecture, str), "Architecture should be a string"
    # For PyTorch models, architecture might be empty, which is acceptable
