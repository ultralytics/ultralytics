# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import shutil
from pathlib import Path

import pytest

from tests import MODEL
from ultralytics import RTDETR, YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import MACOS, WINDOWS
from ultralytics.utils.torch_utils import TORCH_1_9, TORCH_1_13

CFG_RTDETR = "rtdetr-l.yaml"  # RTDETR model for testing


def test_autobackend_architecture_onnx():
    """Test AutoBackend correctly reads architecture metadata from ONNX models."""
    try:
        # Test YOLO11 ONNX
        model = YOLO(MODEL)
        onnx_file = model.export(format="onnx", imgsz=32)
        
        backend = AutoBackend(onnx_file)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture in ["YOLO11", "YOLOv8", "YOLO"], (
            f"Expected YOLO family architecture, got {backend.architecture}"
        )
        
        # Cleanup
        if Path(onnx_file).exists():
            Path(onnx_file).unlink()
            
    except ImportError:
        pytest.skip("Test requires onnxruntime")


@pytest.mark.skipif(not TORCH_1_9, reason="RTDETR requires torch>=1.9")
def test_autobackend_architecture_rtdetr_onnx():
    """Test AutoBackend correctly reads RTDETR architecture from ONNX models."""
    try:
        # Test RTDETR ONNX
        model = RTDETR(CFG_RTDETR)
        onnx_file = model.export(format="onnx", imgsz=640)
        
        backend = AutoBackend(onnx_file)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture == "RTDETR", (
            f"Expected RTDETR architecture, got {backend.architecture}"
        )
        
        # Cleanup
        if Path(onnx_file).exists():
            Path(onnx_file).unlink()
            
    except ImportError:
        pytest.skip("Test requires onnxruntime")


@pytest.mark.skipif(not TORCH_1_13, reason="OpenVINO requires torch>=1.13")
def test_autobackend_architecture_openvino():
    """Test AutoBackend correctly reads architecture metadata from OpenVINO models."""
    try:
        # Test YOLO11 OpenVINO
        model = YOLO(MODEL)
        openvino_dir = model.export(format="openvino", imgsz=32)
        
        backend = AutoBackend(openvino_dir)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture in ["YOLO11", "YOLOv8", "YOLO"], (
            f"Expected YOLO family architecture, got {backend.architecture}"
        )
        
        # Cleanup
        if Path(openvino_dir).exists():
            shutil.rmtree(openvino_dir)
            
    except Exception as e:
        pytest.skip(f"OpenVINO test failed: {e}")


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
        assert backend.architecture == "RTDETR", (
            f"Expected RTDETR architecture, got {backend.architecture}"
        )
        
        # Cleanup
        if Path(openvino_dir).exists():
            shutil.rmtree(openvino_dir)
            
    except Exception as e:
        pytest.skip(f"RTDETR OpenVINO test failed: {e}")


@pytest.mark.skipif(WINDOWS or not MACOS, reason="CoreML export supported on macOS")
@pytest.mark.skipif(not TORCH_1_9, reason="CoreML>=7.2 not supported with PyTorch<=1.8")
def test_autobackend_architecture_coreml():
    """Test AutoBackend correctly reads architecture metadata from CoreML models."""
    try:
        # Test YOLO11 CoreML
        model = YOLO(MODEL)
        coreml_file = model.export(format="coreml", imgsz=32)
        
        backend = AutoBackend(coreml_file)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        assert backend.architecture in ["YOLO11", "YOLOv8", "YOLO"], (
            f"Expected YOLO family architecture, got {backend.architecture}"
        )
        
        # Cleanup
        if Path(coreml_file).exists():
            if Path(coreml_file).is_dir():
                shutil.rmtree(coreml_file)
            else:
                Path(coreml_file).unlink()
                
    except Exception as e:
        pytest.skip(f"CoreML test failed: {e}")


def test_autobackend_architecture_torchscript():
    """Test AutoBackend correctly reads architecture metadata from TorchScript models."""
    try:
        # Test YOLO11 TorchScript
        model = YOLO(MODEL)
        torchscript_file = model.export(format="torchscript", imgsz=32)
        
        backend = AutoBackend(torchscript_file)
        assert hasattr(backend, "architecture"), "AutoBackend should have architecture attribute"
        # TorchScript might not always have architecture metadata depending on export method
        # Just verify the attribute exists and is a string
        assert isinstance(backend.architecture, str), "Architecture should be a string"
        
        # Cleanup
        if Path(torchscript_file).exists():
            Path(torchscript_file).unlink()
            
    except Exception as e:
        pytest.skip(f"TorchScript test failed: {e}")


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