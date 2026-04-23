---
comments: true
description: Export any PyTorch model (timm, torchvision, or custom) to ONNX, OpenVINO, CoreML, TensorFlow SavedModel, TorchScript, NCNN, MNN, PaddlePaddle, and ExecuTorch with Ultralytics export utilities.
keywords: export PyTorch model, convert PyTorch to ONNX, PyTorch to CoreML, PyTorch to OpenVINO, PyTorch to TensorFlow, non-YOLO export, timm export, torchvision export, TorchScript export, NCNN export, MNN export, PaddlePaddle export, ExecuTorch export, TFLite export, Ultralytics export utilities, torch.nn.Module export, model conversion, model deployment, pytorch deployment
---

# How to Export Non-YOLO PyTorch Models with Ultralytics

Deploying PyTorch models to production usually means juggling a different exporter for every target: `torch.onnx.export` for ONNX, `coremltools` for Apple devices, `onnx2tf` for TensorFlow, `pnnx` for NCNN, and so on. Each tool has its own API, dependency quirks, and output conventions.

Ultralytics ships standalone export utilities that wrap multiple backends behind one consistent interface. You can export any `torch.nn.Module`, including [timm](https://github.com/huggingface/pytorch-image-models) image models, [torchvision](https://pytorch.org/vision/) classifiers and detectors, or your own custom architectures, to [ONNX](../integrations/onnx.md), [TorchScript](../integrations/torchscript.md), [OpenVINO](../integrations/openvino.md), [CoreML](../integrations/coreml.md), [NCNN](../integrations/ncnn.md), [PaddlePaddle](../integrations/paddlepaddle.md), [MNN](../integrations/mnn.md), [ExecuTorch](../integrations/executorch.md), and [TensorFlow SavedModel](../integrations/tf-savedmodel.md) without learning each backend separately.

## Why Use Ultralytics for Non-YOLO Export?

- **One API across 10 formats:** learn a single calling convention instead of a dozen.
- **Shared utility surface:** the export helpers live under `ultralytics.utils.export`, so once the backend packages are installed you can keep the same calling pattern across formats.
- **Same code path as YOLO exports:** the same helpers power every Ultralytics YOLO export.
- **FP16 and INT8 quantization** built in for formats that support it (OpenVINO, CoreML, MNN, NCNN).
- **Works on CPU:** no GPU required for the export step itself, so you can run it locally on any laptop.

## Quick Start

The fastest path is a two-line export to [ONNX](../integrations/onnx.md) with no YOLO code and no setup beyond `pip install ultralytics onnx timm`:

```python
import timm
import torch

from ultralytics.utils.export import torch2onnx

model = timm.create_model("resnet18", pretrained=True).eval()
torch2onnx(model, torch.randn(1, 3, 224, 224), output_file="resnet18.onnx")
```

## Supported Export Formats

The `torch2*` functions take a standard `torch.nn.Module` and an example input tensor. MNN, TF SavedModel, and TF Frozen Graph go through an intermediate ONNX or Keras artifact. No YOLO-specific attributes are required in either case.

| Format          | Function              | Install                                                             | Output                         |
| --------------- | --------------------- | ------------------------------------------------------------------- | ------------------------------ |
| ONNX            | `torch2onnx()`        | `pip install onnx`                                                  | `.onnx` file                   |
| TorchScript     | `torch2torchscript()` | included with PyTorch                                               | `.torchscript` file            |
| OpenVINO        | `torch2openvino()`    | `pip install openvino`                                              | `_openvino_model/` directory   |
| CoreML          | `torch2coreml()`      | `pip install coremltools`                                           | `.mlpackage`                   |
| TF SavedModel   | `onnx2saved_model()`  | [see detailed requirements below](#export-to-tensorflow-savedmodel) | `_saved_model/` directory      |
| TF Frozen Graph | `keras2pb()`          | [see detailed requirements below](#export-to-tensorflow-savedmodel) | `.pb` file                     |
| NCNN            | `torch2ncnn()`        | `pip install ncnn pnnx`                                             | `_ncnn_model/` directory       |
| MNN             | `onnx2mnn()`          | `pip install MNN`                                                   | `.mnn` file                    |
| PaddlePaddle    | `torch2paddle()`      | `pip install paddlepaddle x2paddle`                                 | `_paddle_model/` directory     |
| ExecuTorch      | `torch2executorch()`  | `pip install executorch`                                            | `_executorch_model/` directory |

!!! note "ONNX as an intermediate format"

    [MNN](../integrations/mnn.md), [TF SavedModel](../integrations/tf-savedmodel.md), and TF Frozen Graph exports go through ONNX as an intermediate step. Export to ONNX first, then convert.

!!! tip "Embedding metadata"

    Several export functions accept an optional `metadata` dictionary (e.g., `torch2torchscript(..., metadata={"author": "me"})`) that embeds custom key-value pairs into the exported artifact where the format supports it.

## Step-by-Step Examples

Every example below uses the same setup, a pretrained ResNet-18 from timm in evaluation mode:

```python
import timm
import torch

model = timm.create_model("resnet18", pretrained=True).eval()
im = torch.randn(1, 3, 224, 224)
```

!!! warning "Always call `model.eval()` before exporting"

    Dropout, [batch normalization](https://www.ultralytics.com/glossary/batch-normalization), and other train-only layers behave differently during inference. Skipping `.eval()` produces exports with incorrect outputs.

### Export to ONNX

```python
from ultralytics.utils.export import torch2onnx

torch2onnx(model, im, output_file="resnet18.onnx")
```

For dynamic batch size, pass a `dynamic` dictionary:

```python
torch2onnx(model, im, output_file="resnet18_dyn.onnx", dynamic={"images": {0: "batch_size"}})
```

The default opset is `14` and the default input name is `"images"`. Override with the `opset`, `input_names`, or `output_names` arguments.

### Export to TorchScript

No extra dependencies needed. Uses `torch.jit.trace` under the hood.

```python
from ultralytics.utils.export import torch2torchscript

torch2torchscript(model, im, output_file="resnet18.torchscript")
```

### Export to OpenVINO

```python
from ultralytics.utils.export import torch2openvino

ov_model = torch2openvino(model, im, output_dir="resnet18_openvino_model")
```

The directory contains a fixed-name `model.xml` and `model.bin` pair:

```
resnet18_openvino_model/
├── model.xml
└── model.bin
```

Pass `dynamic=True` for dynamic input shapes, `half=True` for FP16, or `int8=True` for INT8 quantization. INT8 additionally requires a `calibration_dataset` argument.

Requires `openvino>=2024.0.0` (or `>=2025.2.0` on macOS 15.4+) and `torch>=2.1`.

### Export to CoreML

```python
import coremltools as ct

from ultralytics.utils.export import torch2coreml

inputs = [ct.TensorType("input", shape=(1, 3, 224, 224))]
ct_model = torch2coreml(model, inputs, im, output_file="resnet18.mlpackage")
```

For [classification](https://www.ultralytics.com/glossary/image-classification) models, pass a list of class names to `classifier_names` to add a classification head to the CoreML model.

Requires `coremltools>=9.0`, `torch>=1.11`, and `numpy<=2.3.5`. Not supported on Windows.

!!! warning "`BlobWriter not loaded` error"

    `coremltools>=9.0` ships wheels for Python 3.10–3.13 on macOS and Linux. On newer Python versions the native C extension fails to load. Use Python 3.10–3.13 for CoreML export.

### Export to TensorFlow SavedModel

TF SavedModel export goes through ONNX as an intermediate step:

```python
from ultralytics.utils.export import onnx2saved_model, torch2onnx

torch2onnx(model, im, output_file="resnet18.onnx")
keras_model = onnx2saved_model("resnet18.onnx", output_dir="resnet18_saved_model")
```

The function returns a Keras model and also generates TFLite files (`.tflite`) inside the output directory:

```
resnet18_saved_model/
├── saved_model.pb
├── variables/
├── resnet18_float32.tflite
├── resnet18_float16.tflite
└── resnet18_int8.tflite
```

Requirements:

- `tensorflow>=2.0.0,<=2.19.0`
- `onnx2tf>=1.26.3,<1.29.0`
- `tf_keras<=2.19.0`
- `sng4onnx>=1.0.1`
- `onnx_graphsurgeon>=0.3.26` (install with `--extra-index-url https://pypi.ngc.nvidia.com`)
- `ai-edge-litert>=1.2.0,<1.4.0` on macOS (`ai-edge-litert>=1.2.0` on other platforms)
- `onnxslim>=0.1.71`
- `onnx>=1.12.0,<2.0.0`
- `protobuf>=5`

### Export to TensorFlow Frozen Graph

Continuing from the SavedModel export above, convert the returned Keras model to a frozen `.pb` graph:

```python
from pathlib import Path

from ultralytics.utils.export import keras2pb

keras2pb(keras_model, output_file=Path("resnet18_saved_model/resnet18.pb"))
```

### Export to NCNN

```python
from ultralytics.utils.export import torch2ncnn

torch2ncnn(model, im, output_dir="resnet18_ncnn_model")
```

The directory contains fixed-name param and bin files along with a Python wrapper:

```
resnet18_ncnn_model/
├── model.ncnn.param
├── model.ncnn.bin
└── model_ncnn.py
```

`torch2ncnn()` checks for `ncnn` and `pnnx` on first use.

### Export to MNN

MNN export requires an ONNX file as input. Export to ONNX first, then convert:

```python
from ultralytics.utils.export import onnx2mnn, torch2onnx

torch2onnx(model, im, output_file="resnet18.onnx")
onnx2mnn("resnet18.onnx", output_file="resnet18.mnn")
```

Supports `half=True` for FP16 and `int8=True` for INT8 quantization. Requires `MNN>=2.9.6` and `torch>=1.10`.

### Export to PaddlePaddle

```python
from ultralytics.utils.export import torch2paddle

torch2paddle(model, im, output_dir="resnet18_paddle_model")
```

The directory contains the PaddlePaddle model and parameter files:

```
resnet18_paddle_model/
├── model.pdmodel
└── model.pdiparams
```

Requires `x2paddle` and the correct PaddlePaddle distribution for your platform:

- `paddlepaddle-gpu>=3.0.0,<3.3.0` on CUDA
- `paddlepaddle==3.0.0` on ARM64 CPU
- `paddlepaddle>=3.0.0,<3.3.0` on other CPUs

Not supported on NVIDIA Jetson.

### Export to ExecuTorch

```python
from ultralytics.utils.export import torch2executorch

torch2executorch(model, im, output_dir="resnet18_executorch_model")
```

The exported `.pte` file is saved inside the output directory:

```
resnet18_executorch_model/
└── model.pte
```

Requires `torch>=2.9.0` and a matching ExecuTorch runtime (`pip install executorch`). For runtime usage, see the [ExecuTorch integration](../integrations/executorch.md).

## Verify Your Exported Model

After exporting, verify numerical parity with the original PyTorch model before shipping. A quick smoke test with `ONNXBackend` from `ultralytics.nn.backends` compares outputs and flags tracing or quantization errors early:

```python
import numpy as np
import timm
import torch

from ultralytics.nn.backends import ONNXBackend

model = timm.create_model("resnet18", pretrained=True).eval()
im = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    pytorch_output = model(im).numpy()

onnx_model = ONNXBackend("resnet18.onnx", device=torch.device("cpu"))
onnx_output = onnx_model.forward(im)[0]

diff = np.abs(pytorch_output - onnx_output).max()
print(f"Max difference: {diff:.6f}")  # should be < 1e-5
```

!!! tip "Expected difference"

    For FP32 exports, the max absolute difference should be under `1e-5`. Larger differences point to unsupported ops, incorrect input shape, or a model not in eval mode. FP16 and INT8 exports have looser tolerances. Validate on real data instead of random tensors.

For other runtimes, the input tensor name may differ. OpenVINO, for example, uses the model's forward-argument name (typically `x` for generic models), while `torch2onnx` defaults to `"images"`.

## Known Limitations

- **Multi-input support is uneven**: `torch2onnx` and `torch2openvino` accept a tuple or list of example tensors for models with multiple inputs. `torch2torchscript`, `torch2coreml`, `torch2ncnn`, `torch2paddle`, and `torch2executorch` assume a single input tensor.
- **ExecuTorch needs `flatc`**: The ExecuTorch runtime requires the FlatBuffers compiler. Install with `brew install flatbuffers` on macOS or `apt install flatbuffers-compiler` on Ubuntu.
- **No inference via Ultralytics**: Exported non-YOLO models cannot be loaded back through `YOLO()` for inference. Use the native runtime for each format ([ONNX Runtime](../integrations/onnx.md), [OpenVINO Runtime](../integrations/openvino.md), etc.).
- **YOLO-only formats**: [Axelera](../integrations/axelera.md) and [Sony IMX500](../integrations/sony-imx500.md) exports require YOLO-specific model attributes and are not available for generic models.
- **Platform-specific formats**: [TensorRT](../integrations/tensorrt.md) requires an NVIDIA GPU. [RKNN](../integrations/rockchip-rknn.md) requires the `rknn-toolkit2` SDK (Linux only). [Edge TPU](../integrations/edge-tpu.md) requires the `edgetpu_compiler` binary (Linux only).

## FAQ

### What models can I export with Ultralytics?

Any `torch.nn.Module`. This includes models from timm, torchvision, or any custom PyTorch model. The model must be in evaluation mode (`model.eval()`) before export. ONNX and OpenVINO additionally accept a tuple of example tensors for multi-input models.

### Which export formats work without a GPU?

All supported formats (TorchScript, ONNX, OpenVINO, CoreML, TF SavedModel, TF Frozen Graph, NCNN, PaddlePaddle, MNN, ExecuTorch) can export on CPU. No GPU is required for the export process itself. TensorRT is the only format that requires an NVIDIA GPU.

### What Ultralytics version do I need?

Use Ultralytics `>=8.4.38`, which includes the `ultralytics.utils.export` module and the standardized `output_file`/`output_dir` arguments.

### Can I export a torchvision model to CoreML for iOS deployment?

Yes. torchvision classifiers, detectors, and segmentation models export to `.mlpackage` via `torch2coreml`. For image classification models, pass a list of class names to `classifier_names` to bake in a classification head. Run the export on macOS or Linux. CoreML is not supported on Windows. See the [CoreML integration](../integrations/coreml.md) for iOS deployment details.

### Can I quantize my exported model to INT8 or FP16?

Yes, for several formats. Pass `half=True` for FP16 or `int8=True` for INT8 when exporting to OpenVINO, CoreML, MNN, or NCNN. INT8 in OpenVINO additionally requires a `calibration_dataset` argument for [post-training quantization](https://www.ultralytics.com/glossary/model-quantization). See each format's integration page for quantization trade-offs.
