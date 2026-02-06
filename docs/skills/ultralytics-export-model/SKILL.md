---
name: ultralytics-export-model
description: Export YOLO models to various formats (ONNX, TensorRT, CoreML, TFLite, etc.) for deployment. Use when the user needs to deploy models to production environments, edge devices, or optimize for inference speed.
license: AGPL-3.0
metadata:
  author: Burhan-Q
  version: "1.0"
  ultralytics-version: ">=8.4.11"
---

# Export YOLO Model

## When to use this skill

Use this skill when you need to:
- Deploy YOLO models to production environments
- Optimize models for specific hardware (GPU, CPU, mobile, edge devices)
- Convert models to framework-specific formats (ONNX, TensorRT, TFLite, CoreML, etc.)
- Reduce model size and improve inference speed

## Prerequisites

- Python ≥3.8 with PyTorch ≥1.8 installed
- `ultralytics` package installed
    - Cloned repo install or package install
    - `uv pip install ultralytics --upgrade` OR `pip install ultralytics --upgrade`
- Trained YOLO model (`.pt` file)
- Format-specific dependencies (see below)

## Supported Export Formats

| Format | Flag | Use Case | Requirements |
|--------|------|----------|--------------|
| PyTorch | `.pt` | Original format | PyTorch |
| TorchScript | `torchscript` | C++ deployment | PyTorch |
| ONNX | `onnx` | Framework-agnostic, wide compatibility | `onnx` |
| OpenVINO | `openvino` | Intel hardware optimization | `openvino` |
| TensorRT | `engine` | NVIDIA GPU optimization | `tensorrt` |
| CoreML | `coreml` | iOS/macOS deployment | `coremltools` |
| TF SavedModel | `saved_model` | TensorFlow serving | `tensorflow` |
| TF GraphDef | `pb` | TensorFlow frozen graph | `tensorflow` |
| TFLite | `tflite` | Mobile/edge devices | `tensorflow` |
| TF Edge TPU | `edgetpu` | Google Coral devices | `tensorflow`, Edge TPU compiler |
| TF.js | `tfjs` | Web browser deployment | `tensorflowjs` |
| PaddlePaddle | `paddle` | Baidu PaddlePaddle framework | `paddle2onnx` |
| NCNN | `ncnn` | Mobile CPU optimization | `ncnn` |

## Export Workflow

### 1. Basic Export

**Python API:**

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo26n.pt")

# Export to ONNX
model.export(format="onnx")  # creates yolo26n.onnx

# The exported model path is returned
path = model.export(format="onnx")
print(f"Exported to: {path}")
```

**CLI:**

```bash
yolo export model=yolo26n.pt format=onnx
```

### 2. Export with Options

**Python API:**

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Export to ONNX with custom options
model.export(
    format="onnx",
    imgsz=640,  # input image size
    half=False,  # FP16 quantization
    dynamic=False,  # dynamic input shapes
    simplify=True,  # simplify ONNX model
    opset=12,  # ONNX opset version
)
```

**CLI:**

```bash
yolo export model=yolo26n.pt format=onnx imgsz=640 half=False dynamic=False simplify=True opset=12
```

### 3. Format-Specific Exports

#### ONNX (Recommended for Deployment)

```python
model.export(
    format="onnx",
    imgsz=640,
    simplify=True,  # simplify for better compatibility
    opset=12,  # ONNX opset version (11-17 supported)
)
```

#### TensorRT (NVIDIA GPUs)

```python
# Requires TensorRT SDK installed
model.export(
    format="engine",
    imgsz=640,
    half=True,  # FP16 precision for faster inference
    workspace=4,  # max workspace size in GB
    device=0,  # GPU device
)
```

#### CoreML (iOS/macOS)

```python
model.export(
    format="coreml",
    imgsz=640,
    nms=True,  # include NMS in model
)
```

#### TFLite (Mobile/Edge)

```python
model.export(
    format="tflite",
    imgsz=640,
    int8=True,  # INT8 quantization for smaller model
)
```

#### OpenVINO (Intel Hardware)

```python
model.export(
    format="openvino",
    imgsz=640,
    half=True,  # FP16 precision
)
```

## Export Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `format` | Export format | `torchscript` |
| `imgsz` | Input image size | 640 |
| `keras` | Use Keras for TF exports | False |
| `optimize` | TorchScript mobile optimization | False |
| `half` | FP16 quantization | False |
| `int8` | INT8 quantization | False |
| `dynamic` | Dynamic input shapes | False |
| `simplify` | Simplify ONNX model | False |
| `opset` | ONNX opset version | None |
| `workspace` | TensorRT workspace size (GB) | 4 |
| `nms` | Add NMS to CoreML model | False |

## Running Exported Models

### ONNX

```python
from ultralytics import YOLO

# Load ONNX model
model = YOLO("yolo26n.onnx")

# Run inference
results = model("image.jpg")
```

### TensorRT

```python
from ultralytics import YOLO

# Load TensorRT model
model = YOLO("yolo26n.engine")

# Run inference (automatically uses TensorRT)
results = model("image.jpg")
```

### CoreML

```python
from ultralytics import YOLO

# Load CoreML model
model = YOLO("yolo26n.mlpackage")

# Run inference (macOS/iOS only)
results = model("image.jpg")
```

## Model Size & Speed Comparison

Typical export results for YOLO26n on COCO:

| Format | Size | CPU Speed | GPU Speed |
|--------|------|-----------|-----------|
| PyTorch (.pt) | 9.4 MB | ~40ms | ~1.7ms |
| ONNX | 9.3 MB | ~38ms | ~1.7ms |
| TensorRT (FP16) | 4.7 MB | N/A | ~0.9ms |
| CoreML | 9.4 MB | ~35ms | N/A |
| TFLite (FP16) | 4.7 MB | ~45ms | N/A |
| TFLite (INT8) | 2.4 MB | ~40ms | N/A |

*Speeds measured on Amazon EC2 P4d instance for GPU, Intel Xeon for CPU*

## Common Issues

**Export Fails:**
- Install format-specific dependencies (`pip install onnx`, etc.)
- Try `simplify=True` for ONNX exports
- Reduce `opset` version if compatibility issues occur

**Exported Model Accuracy Lower:**
- Disable quantization (`half=False`, `int8=False`)
- Validate exported model vs original
- Try different export settings

**Slow Exported Model:**
- Enable `half=True` for FP16 precision (GPU only)
- Use `int8=True` for mobile deployment
- Try `simplify=True` for ONNX

## Next Steps

- Run inference with exported model: see `ultralytics-run-inference` skill
- Deploy to production environment
- Benchmark exported model performance
- [Ultralytics Platform](https://platform.ultralytics.com) also can export models trained using end-to-end cloud infrastructure to all available formats.

## References

- [Ultralytics Export Docs](https://docs.ultralytics.com/modes/export/)
- [TensorRT Guide](https://docs.ultralytics.com/integrations/tensorrt/)
- [CoreML Guide](https://docs.ultralytics.com/integrations/coreml/)
- [ONNX Guide](https://docs.ultralytics.com/integrations/onnx/)