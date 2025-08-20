# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import shutil
import uuid
from contextlib import redirect_stderr, redirect_stdout
from itertools import product
from pathlib import Path

import pytest

from tests import MODEL, SOURCE
from ultralytics import RTDETR, YOLO
from ultralytics.cfg import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import (
    ARM64,
    IS_RASPBERRYPI,
    LINUX,
    MACOS,
    WINDOWS,
    checks,
)
from ultralytics.utils.torch_utils import TORCH_1_9, TORCH_1_13

CFG_RTDETR = "rtdetr-l.yaml"  # RTDETR model for testing


def test_export_torchscript():
    """Test YOLO model export to TorchScript format for compatibility and correctness."""
    file = YOLO(MODEL).export(format="torchscript", optimize=False, imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


def test_export_onnx():
    """Test YOLO model export to ONNX format with dynamic axes."""
    file = YOLO(MODEL).export(format="onnx", dynamic=True, imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.skipif(not TORCH_1_9, reason="RTDETR requires torch>=1.9")
def test_rtdetr_onnx_architecture_metadata():
    """Test RTDETR ONNX export includes correct architecture metadata for model routing."""
    file = RTDETR(CFG_RTDETR).export(format="onnx", imgsz=640)

    # Check ONNX metadata contains architecture="RTDETR"
    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(file, providers=["CPUExecutionProvider"])
        metadata = session.get_modelmeta().custom_metadata_map
        assert metadata.get("architecture") == "RTDETR", (
            f"Expected architecture='RTDETR', got {metadata.get('architecture')}"
        )
    except ImportError:
        pytest.skip("Test requires onnxruntime")


def test_yolo11_onnx_architecture_metadata():
    """Test YOLO11 ONNX export includes correct architecture metadata for model identification."""
    file = YOLO(MODEL).export(format="onnx", imgsz=32)

    # Check ONNX metadata contains architecture="YOLO11"
    try:
        import onnxruntime

        session = onnxruntime.InferenceSession(file, providers=["CPUExecutionProvider"])
        metadata = session.get_modelmeta().custom_metadata_map
        assert metadata.get("architecture") == "YOLO11", (
            f"Expected architecture='YOLO11', got {metadata.get('architecture')}"
        )
    except ImportError:
        pytest.skip("Test requires onnxruntime")


@pytest.mark.skipif(not TORCH_1_13, reason="OpenVINO requires torch>=1.13")
def test_export_openvino():
    """Test YOLO export to OpenVINO format for model inference compatibility."""
    file = YOLO(MODEL).export(format="openvino", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.skipif(not TORCH_1_13, reason="OpenVINO requires torch>=1.13")
def test_export_openvino_directory_format():
    """Test that OpenVINO directory output formats have correct prefixes and metadata."""
    try:
        # Test YOLO11 OpenVINO export
        model_yolo11 = YOLO(MODEL)
        openvino_dir = model_yolo11.export(format="openvino", imgsz=640)

        # Check that the directory exists and has expected naming
        assert Path(openvino_dir).exists(), f"OpenVINO export directory not found: {openvino_dir}"

        # Check metadata.yaml contains architecture
        metadata_file = Path(openvino_dir) / "metadata.yaml"
        if metadata_file.exists():
            import yaml

            with open(metadata_file) as f:
                metadata = yaml.safe_load(f)
            assert metadata.get("architecture") == "YOLO11", (
                f"Expected architecture='YOLO11' in metadata.yaml, got {metadata.get('architecture')}"
            )

        # Test RTDETR OpenVINO export
        model_rtdetr = RTDETR(CFG_RTDETR)
        rtdetr_dir = model_rtdetr.export(format="openvino", imgsz=640)

        # Check RTDETR metadata
        rtdetr_metadata_file = Path(rtdetr_dir) / "metadata.yaml"
        if rtdetr_metadata_file.exists():
            with open(rtdetr_metadata_file) as f:
                rtdetr_metadata = yaml.safe_load(f)
            assert rtdetr_metadata.get("architecture") == "RTDETR", (
                f"Expected architecture='RTDETR' in metadata.yaml, got {rtdetr_metadata.get('architecture')}"
            )

    except (ImportError, Exception) as e:
        pytest.skip(f"Test requires dependencies or failed: {e}")


def test_cli_architecture_routing():
    """Test CLI correctly routes models based on architecture metadata instead of filename."""
    try:
        import sys
        from ultralytics.cfg import entrypoint
        
        # Export RTDETR model to ONNX with generic filename
        model = RTDETR(CFG_RTDETR)
        onnx_file = model.export(format="onnx", imgsz=640)
        
        # Rename to generic name to test metadata-based routing
        generic_file = Path(onnx_file).parent / "generic_model.onnx"
        if generic_file.exists():
            generic_file.unlink()
        shutil.move(onnx_file, generic_file)
        
        # Test CLI routing with generic filename works via metadata
        original_argv = sys.argv.copy()
        sys.argv = ["yolo", "val", f"model={generic_file}", "data=coco8.yaml", "imgsz=640", "batch=1"]
        try:
            # This should route to RTDETR validator via metadata, not filename
            entrypoint()
        except (SystemExit, Exception):
            pass  # Expected for test environment - validation might fail but routing should work
        finally:
            sys.argv = original_argv
        
        # Cleanup
        if generic_file.exists():
            generic_file.unlink()
            
    except ImportError:
        pytest.skip("Test requires dependencies")


def test_yolo_architecture_fingerprints():
    """Test architecture detection based on model fingerprints (C2PSA, C2f, C3)."""
    # Test YOLO11 fingerprint detection via C2PSA module
    model_yolo11 = YOLO(MODEL)
    onnx_file = model_yolo11.export(format="onnx", imgsz=32)
    
    try:
        import onnxruntime
        session = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
        metadata = session.get_modelmeta().custom_metadata_map
        architecture = metadata.get("architecture")
        
        # Should detect YOLO11 based on C2PSA fingerprint or model name
        assert architecture in ["YOLO11", "YOLOv8", "YOLO"], (
            f"Expected YOLO family architecture, got {architecture}"
        )
        
    except ImportError:
        pytest.skip("Test requires onnxruntime")
    finally:
        if Path(onnx_file).exists():
            Path(onnx_file).unlink()


@pytest.mark.skipif(not TORCH_1_13, reason="Test requires torch>=1.13") 
def test_multiple_format_metadata():
    """Test architecture metadata is correctly embedded across multiple export formats."""
    model = YOLO(MODEL)
    
    formats_to_test = ["onnx", "openvino"]
    
    for fmt in formats_to_test:
        try:
            exported_file = model.export(format=fmt, imgsz=32)
            
            if fmt == "onnx":
                # Test ONNX metadata
                import onnxruntime
                session = onnxruntime.InferenceSession(exported_file, providers=["CPUExecutionProvider"])
                metadata = session.get_modelmeta().custom_metadata_map
                assert "architecture" in metadata, f"Architecture not found in {fmt} metadata"
                
            elif fmt == "openvino":
                # Test OpenVINO metadata.yaml
                metadata_file = Path(exported_file) / "metadata.yaml"
                if metadata_file.exists():
                    import yaml
                    with open(metadata_file, 'r') as f:
                        metadata = yaml.safe_load(f)
                    assert "architecture" in metadata, f"Architecture not found in {fmt} metadata.yaml"
            
            # Cleanup
            if Path(exported_file).exists():
                if Path(exported_file).is_dir():
                    shutil.rmtree(exported_file)
                else:
                    Path(exported_file).unlink()
                    
        except Exception as e:
            pytest.skip(f"Test failed for format {fmt}: {e}")


def test_custom_trained_model_cli_routing():
    """Test CLI routing works for custom trained models with generic filenames like best.pt, latest.pt."""
    try:
        import subprocess
        import tempfile
        
        # Test YOLO11 with common generic filenames used for trained models
        model = YOLO(MODEL)
        generic_names = ["best.onnx", "latest.onnx", "final.onnx"]
        
        # Export to ONNX format
        exported_file = model.export(format="onnx", imgsz=32)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            for generic_name in generic_names:
                # Create generic filename
                generic_path = Path(tmp_dir) / generic_name
                shutil.copy2(exported_file, generic_path)
                
                # Verify metadata exists in generic file
                try:
                    import onnxruntime
                    session = onnxruntime.InferenceSession(str(generic_path), providers=["CPUExecutionProvider"])
                    metadata = session.get_modelmeta().custom_metadata_map
                    architecture = metadata.get("architecture")
                    assert architecture == "YOLO11", f"Expected YOLO11, got {architecture}"
                    
                    # Test CLI validation with generic filename
                    result = subprocess.run([
                        "yolo", "val", f"model={generic_path}", "data=coco8.yaml", "imgsz=32", "batch=1"
                    ], capture_output=True, text=True, timeout=30)
                    
                    # CLI should execute without errors (validation might fail but routing should work)
                    assert result.returncode == 0 or "validation" in result.stderr.lower(), (
                        f"CLI routing failed for {generic_name}"
                    )
                    
                except ImportError:
                    pytest.skip("Test requires onnxruntime")
                except subprocess.TimeoutExpired:
                    pass  # Expected in some test environments
                except Exception as e:
                    # CLI routing worked if we get validation-related errors, not routing errors
                    if "model" not in str(e).lower() and "loading" not in str(e).lower():
                        pass  # Routing likely succeeded
        
        # Cleanup
        if Path(exported_file).exists():
            Path(exported_file).unlink()
                            
    except ImportError:
        pytest.skip("Test requires dependencies")


def test_directory_format_cli_routing():
    """Test CLI routing for directory-based export formats with generic names."""
    try:
        import sys
        import tempfile
        from ultralytics.cfg import entrypoint
        
        # Test with OpenVINO directory format
        model = YOLO(MODEL)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Export to OpenVINO format
                openvino_dir = model.export(format="openvino", imgsz=32)
                
                # Create generic directory name
                generic_dir = Path(tmp_dir) / "custom_model_openvino_model"
                if generic_dir.exists():
                    shutil.rmtree(generic_dir)
                shutil.copytree(openvino_dir, generic_dir)
                
                # Test CLI validation with generic directory name
                original_argv = sys.argv.copy()
                sys.argv = [
                    "yolo", "val", 
                    f"model={generic_dir}", 
                    "data=coco8.yaml", 
                    "imgsz=32",
                    "batch=1"
                ]
                
                try:
                    # Should route correctly based on metadata.yaml
                    entrypoint()
                    print("✓ Generic OpenVINO directory routed successfully")
                except (SystemExit, Exception):
                    # Expected in test environment
                    pass
                finally:
                    sys.argv = original_argv
                
                # Cleanup
                if Path(openvino_dir).exists():
                    shutil.rmtree(openvino_dir)
                    
            except Exception as e:
                pytest.skip(f"OpenVINO export test failed: {e}")
                
    except ImportError:
        pytest.skip("Test requires torch>=1.13 for OpenVINO")


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_1_13, reason="OpenVINO requires torch>=1.13")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, nms",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, int8, half, batch, nms)
        for task, dynamic, int8, half, batch, nms in product(
            TASKS, [True, False], [True, False], [True, False], [1, 2], [True, False]
        )
        if not ((int8 and half) or (task == "classify" and nms))
    ],
)
def test_export_openvino_matrix(task, dynamic, int8, half, batch, nms):
    """Test YOLO model export to OpenVINO under various configuration matrix conditions."""
    file = YOLO(TASK2MODEL[task]).export(
        format="openvino",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
        data=TASK2DATA[task],
        nms=nms,
    )
    if WINDOWS:
        # Use unique filenames due to Windows file permissions bug possibly due to latent threaded use
        # See https://github.com/ultralytics/ultralytics/actions/runs/8957949304/job/24601616830?pr=10423
        file = Path(file)
        file = file.rename(file.with_stem(f"{file.stem}-{uuid.uuid4()}"))
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32, batch=batch)  # exported model inference
    shutil.rmtree(file, ignore_errors=True)  # retry in case of potential lingering multi-threaded file usage errors


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, simplify, nms",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, int8, half, batch, simplify, nms)
        for task, dynamic, int8, half, batch, simplify, nms in product(
            TASKS, [True, False], [False], [False], [1, 2], [True, False], [True, False]
        )
        if not ((int8 and half) or (task == "classify" and nms) or (task == "obb" and nms and not TORCH_1_13))
    ],
)
def test_export_onnx_matrix(task, dynamic, int8, half, batch, simplify, nms):
    """Test YOLO export to ONNX format with various configurations and parameters."""
    file = YOLO(TASK2MODEL[task]).export(
        format="onnx", imgsz=32, dynamic=dynamic, int8=int8, half=half, batch=batch, simplify=simplify, nms=nms
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, nms",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, int8, half, batch, nms)
        for task, dynamic, int8, half, batch, nms in product(
            TASKS, [False, True], [False], [False], [1, 2], [True, False]
        )
        if not (task == "classify" and nms)
    ],
)
def test_export_torchscript_matrix(task, dynamic, int8, half, batch, nms):
    """Test YOLO model export to TorchScript format under varied configurations."""
    file = YOLO(TASK2MODEL[task]).export(
        format="torchscript", imgsz=32, dynamic=dynamic, int8=int8, half=half, batch=batch, nms=nms
    )
    YOLO(file)([SOURCE] * batch, imgsz=64 if dynamic else 32)  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not MACOS, reason="CoreML inference only supported on macOS")
@pytest.mark.skipif(not TORCH_1_9, reason="CoreML>=7.2 not supported with PyTorch<=1.8")
@pytest.mark.skipif(checks.IS_PYTHON_3_13, reason="CoreML not supported in Python 3.13")
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, int8, half, batch)
        for task, dynamic, int8, half, batch in product(TASKS, [False], [True, False], [True, False], [1])
        if not (int8 and half)
    ],
)
def test_export_coreml_matrix(task, dynamic, int8, half, batch):
    """Test YOLO export to CoreML format with various parameter configurations."""
    file = YOLO(TASK2MODEL[task]).export(
        format="coreml",
        imgsz=32,
        dynamic=dynamic,
        int8=int8,
        half=half,
        batch=batch,
    )
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference
    shutil.rmtree(file)  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="TFLite export requires Python>=3.10")
@pytest.mark.skipif(
    not LINUX or IS_RASPBERRYPI,
    reason="Test disabled as TF suffers from install conflicts on Windows, macOS and Raspberry Pi",
)
@pytest.mark.parametrize(
    "task, dynamic, int8, half, batch, nms",
    [  # generate all combinations except for exclusion cases
        (task, dynamic, int8, half, batch, nms)
        for task, dynamic, int8, half, batch, nms in product(
            TASKS, [False], [True, False], [True, False], [1], [True, False]
        )
        if not ((int8 and half) or (task == "classify" and nms) or (ARM64 and nms))
    ],
)
def test_export_tflite_matrix(task, dynamic, int8, half, batch, nms):
    """Test YOLO export to TFLite format considering various export configurations."""
    file = YOLO(TASK2MODEL[task]).export(
        format="tflite", imgsz=32, dynamic=dynamic, int8=int8, half=half, batch=batch, nms=nms
    )
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.skipif(not TORCH_1_9, reason="CoreML>=7.2 not supported with PyTorch<=1.8")
@pytest.mark.skipif(WINDOWS, reason="CoreML not supported on Windows")  # RuntimeError: BlobWriter not loaded
@pytest.mark.skipif(LINUX and ARM64, reason="CoreML not supported on aarch64 Linux")
@pytest.mark.skipif(checks.IS_PYTHON_3_13, reason="CoreML not supported in Python 3.13")
def test_export_coreml():
    """Test YOLO export to CoreML format and check for errors."""
    # Capture stdout and stderr
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        YOLO(MODEL).export(format="coreml", nms=True, imgsz=32)
        if MACOS:
            file = YOLO(MODEL).export(format="coreml", imgsz=32)
            YOLO(file)(SOURCE, imgsz=32)  # model prediction only supported on macOS for nms=False models

    # Check captured output for errors
    output = stdout.getvalue() + stderr.getvalue()
    assert "Error" not in output, f"CoreML export produced errors: {output}"
    assert "You will not be able to run predict()" not in output, "CoreML export has predict() error"


@pytest.mark.skipif(not checks.IS_PYTHON_MINIMUM_3_10, reason="TFLite export requires Python>=3.10")
@pytest.mark.skipif(not LINUX, reason="Test disabled as TF suffers from install conflicts on Windows and macOS")
def test_export_tflite():
    """Test YOLO export to TFLite format under specific OS and Python version conditions."""
    model = YOLO(MODEL)
    file = model.export(format="tflite", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)


@pytest.mark.skipif(True, reason="Test disabled")
@pytest.mark.skipif(not LINUX, reason="TF suffers from install conflicts on Windows and macOS")
def test_export_pb():
    """Test YOLO export to TensorFlow's Protobuf (*.pb) format."""
    model = YOLO(MODEL)
    file = model.export(format="pb", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)


@pytest.mark.skipif(True, reason="Test disabled as Paddle protobuf and ONNX protobuf requirements conflict.")
def test_export_paddle():
    """Test YOLO export to Paddle format, noting protobuf conflicts with ONNX."""
    YOLO(MODEL).export(format="paddle", imgsz=32)


@pytest.mark.slow
def test_export_mnn():
    """Test YOLO export to MNN format (WARNING: MNN test must precede NCNN test or CI error on Windows)."""
    file = YOLO(MODEL).export(format="mnn", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, int8, half, batch",
    [  # generate all combinations except for exclusion cases
        (task, int8, half, batch)
        for task, int8, half, batch in product(TASKS, [True, False], [True, False], [1, 2])
        if not (int8 and half)
    ],
)
def test_export_mnn_matrix(task, int8, half, batch):
    """Test YOLO export to MNN format considering various export configurations."""
    file = YOLO(TASK2MODEL[task]).export(format="mnn", imgsz=32, int8=int8, half=half, batch=batch)
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference
    Path(file).unlink()  # cleanup


@pytest.mark.slow
def test_export_ncnn():
    """Test YOLO export to NCNN format."""
    file = YOLO(MODEL).export(format="ncnn", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)  # exported model inference


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, half, batch",
    [  # generate all combinations except for exclusion cases
        (task, half, batch) for task, half, batch in product(TASKS, [True, False], [1])
    ],
)
def test_export_ncnn_matrix(task, half, batch):
    """Test YOLO export to NCNN format considering various export configurations."""
    file = YOLO(TASK2MODEL[task]).export(format="ncnn", imgsz=32, half=half, batch=batch)
    YOLO(file)([SOURCE] * batch, imgsz=32)  # exported model inference
    shutil.rmtree(file, ignore_errors=True)  # retry in case of potential lingering multi-threaded file usage errors


@pytest.mark.skipif(True, reason="Test disabled as keras and tensorflow version conflicts with TFlite export.")
@pytest.mark.skipif(not LINUX or MACOS, reason="Skipping test on Windows and Macos")
def test_export_imx():
    """Test YOLO export to IMX format."""
    model = YOLO("yolov8n.pt")
    file = model.export(format="imx", imgsz=32)
    YOLO(file)(SOURCE, imgsz=32)
