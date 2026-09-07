---
title: Export Any PyTorch Model to ONNX & More
comments: true
description: Export any PyTorch model (timm, torchvision, or custom) to ONNX, OpenVINO, CoreML, TensorFlow, and more through one Ultralytics API, with no per-backend code.
keywords: export PyTorch model, convert PyTorch to ONNX, PyTorch to CoreML, PyTorch to OpenVINO, PyTorch to TensorFlow, non-YOLO export, timm export, torchvision export, TorchScript export, NCNN export, MNN export, PaddlePaddle export, ExecuTorch export, LiteRT export, Ultralytics export utilities, torch.nn.Module export, model conversion, model deployment, PyTorch deployment
---

# How to Export Non-YOLO PyTorch Models with Ultralytics

Ultralytics ships standalone export utilities under [`ultralytics.utils.export`](../reference/utils/export/engine.md) that wrap multiple backends behind one consistent interface. You can export any `torch.nn.Module`, including [timm](https://github.com/huggingface/pytorch-image-models) image models, [torchvision](https://docs.pytorch.org/vision/) classifiers and detectors, or your own custom architectures, to [ONNX](../integrations/onnx.md), [TorchScript](../integrations/torchscript.md), [OpenVINO](../integrations/openvino.md), [CoreML](../integrations/coreml.md), [NCNN](../integrations/ncnn.md), [PaddlePaddle](../integrations/paddlepaddle.md), [MNN](../integrations/mnn.md), [ExecuTorch](../integrations/executorch.md), and [TensorFlow SavedModel](../integrations/tf-savedmodel.md) without learning each backend separately.

Deploying PyTorch models to production usually means juggling a different exporter for every target: `torch.onnx.export` for ONNX, `coremltools` for Apple devices, `onnx2tf` for TensorFlow, `pnnx` for NCNN, and so on. Each tool has its own API, dependency quirks, and output conventions. These utilities collapse that into a single calling pattern.

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

| Format          | Function                                                          | Install                                                             | Output                         |
| --------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------ |
| ONNX            | [`torch2onnx()`](../reference/utils/export/engine.md)             | `pip install onnx`                                                  | `.onnx` file                   |
| TorchScript     | [`torch2torchscript()`](../reference/utils/export/torchscript.md) | included with PyTorch                                               | `.torchscript` file            |
| OpenVINO        | [`torch2openvino()`](../reference/utils/export/openvino.md)       | `pip install openvino`                                              | `_openvino_model/` directory   |
| CoreML          | [`torch2coreml()`](../reference/utils/export/coreml.md)           | `pip install coremltools`                                           | `.mlpackage`                   |
| TF SavedModel   | [`onnx2saved_model()`](../reference/utils/export/tensorflow.md)   | [see detailed requirements below](#export-to-tensorflow-savedmodel) | `_saved_model/` directory      |
| TF Frozen Graph | [`keras2pb()`](../reference/utils/export/tensorflow.md)           | [see detailed requirements below](#export-to-tensorflow-savedmodel) | `.pb` file                     |
| NCNN            | [`torch2ncnn()`](../reference/utils/export/ncnn.md)               | `pip install ncnn pnnx`                                             | `_ncnn_model/` directory       |
| MNN             | [`onnx2mnn()`](../reference/utils/export/mnn.md)                  | `pip install MNN`                                                   | `.mnn` file                    |
| PaddlePaddle    | [`torch2paddle()`](../reference/utils/export/paddle.md)           | `pip install paddlepaddle x2paddle`                                 | `_paddle_model/` directory     |
| ExecuTorch      | [`torch2executorch()`](../reference/utils/export/executorch.md)   | `pip install executorch`                                            | `_executorch_model/` directory |

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

```text
resnet18_openvino_model/
├── model.xml
└── model.bin
```

Pass `dynamic=True` for dynamic input shapes, `quantize=16` for FP16, or `quantize=8` for INT8 quantization. INT8 additionally requires a `calibration_dataset` argument.

Requires `openvino>=2024.0.0` (or `>=2025.2.0` on macOS 15.4+) and `torch>=2.1`.

### Export to CoreML

```python
import coremltools as ct

from ultralytics.utils.export import torch2coreml

inputs = [ct.TensorType("input", shape=(1, 3, 224, 224))]
ct_model = torch2coreml(model, inputs, im, classifier_names=None, output_file="resnet18.mlpackage")
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

The function returns a Keras model and also generates FP32 and FP16 [LiteRT](../integrations/litert.md) files (`.tflite`) inside the output directory:

```text
resnet18_saved_model/
├── saved_model.pb
├── variables/
├── assets/
├── fingerprint.pb
├── resnet18_float32.tflite
└── resnet18_float16.tflite
```

Pass `quantize=8` to add an INT8 `.tflite` alongside them.

Requirements:

- `tensorflow>=2.0.0,<=2.19.0`
- `onnx2tf>=1.26.3,<1.29.0`
- `tf_keras<=2.19.0`
- `sng4onnx>=1.0.1`
- `onnx_graphsurgeon>=0.3.26`
- `ai-edge-litert>=1.2.0,<1.4.0` on macOS (`ai-edge-litert>=1.2.0` on other platforms)
- `onnxslim>=0.1.82`
- `onnx>=1.12.0,<2.0.0`
- `protobuf>=5`

### Export to TensorFlow Frozen Graph

Continuing from the SavedModel export above, convert the returned `keras_model` to a frozen `.pb` graph:

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

```text
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

Supports `quantize=16` for FP16 and `quantize=8` for INT8 quantization. Requires `MNN>=2.9.6` and `torch>=1.10`.

### Export to PaddlePaddle

```python
from ultralytics.utils.export import torch2paddle

torch2paddle(model, im, output_dir="resnet18_paddle_model")
```

The directory contains the PaddlePaddle model and parameter files:

```text
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

```text
resnet18_executorch_model/
└── model.pte
```

Requires `torch>=2.9.0` and a matching ExecuTorch runtime (`pip install executorch`). For runtime usage, see the [ExecuTorch integration](../integrations/executorch.md).

## Verify Your Exported Model

After exporting, verify numerical parity with the original PyTorch model before shipping. A quick smoke test with [`ONNXBackend`](../reference/nn/backends/onnx.md) from `ultralytics.nn.backends` compares outputs and flags tracing or quantization errors early:

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
onnx_output = onnx_model(im)[0]

diff = np.abs(pytorch_output - onnx_output).max()
print(f"Max difference: {diff:.6f}")  # ~1e-6 for an FP32 ONNX export
```

!!! tip "Expected difference"

    The tolerance is per format, not global. On a ResNet-18 the FP32 exports land near `1e-6` for ONNX, TF SavedModel and LiteRT, and at exactly `0` for TorchScript. NCNN is the outlier at roughly `1e-2`: its CPU runtime enables FP16 packing and arithmetic by default, so an FP32 export still runs in half precision. A difference far above the format's own baseline points to unsupported ops, a wrong input shape, or a model not in eval mode. FP16 and INT8 exports have looser tolerances. Validate on real data instead of random tensors.

For other runtimes, the input tensor name may differ. OpenVINO, for example, uses the model's forward-argument name (typically `x` for generic models), while `torch2onnx` defaults to `"images"`.

## Run Your Exported Model

Exported non-YOLO models load back through the normal `YOLO()` API. The exports above carry no Ultralytics task or input-size metadata, so pass `task` explicitly and `imgsz` matching the example tensor you exported with:

```python
from ultralytics import YOLO

results = YOLO("resnet18.onnx", task="classify")("path/to/image.jpg", imgsz=224)
print(results[0].probs.top1)
```

`imgsz` matters when the export has a fixed input shape: the ONNX and TF SavedModel exports above reject the default of 640. The TorchScript and NCNN exports above do accept other sizes, but neither exporter guarantees it: both trace from the example tensor, so a model that flattens into a `Linear` layer stays fixed. Check your own export.

The value is then rounded up to a multiple of the model stride, which is 32 without metadata. A fixed-shape export at 200x200 is therefore fed 224x224 and rejected even though `imgsz=200` matches it. For input sizes that are not multiples of 32, call the backend directly.

### Calling a Backend Directly

For raw tensors without Ultralytics [preprocessing](https://www.ultralytics.com/glossary/data-preprocessing) and post-processing, use the per-format classes in [`ultralytics.nn.backends`](../reference/nn/backends/base.md), as the [verification example](#verify-your-exported-model) above does. Each takes the exported artifact and a device, and is callable:

| Format                      | Backend                                                       | Input layout |
| --------------------------- | ------------------------------------------------------------- | ------------ |
| ONNX                        | [`ONNXBackend`](../reference/nn/backends/onnx.md)             | BCHW         |
| TorchScript                 | [`TorchScriptBackend`](../reference/nn/backends/pytorch.md)   | BCHW         |
| OpenVINO                    | [`OpenVINOBackend`](../reference/nn/backends/openvino.md)     | BCHW         |
| CoreML                      | [`CoreMLBackend`](../reference/nn/backends/coreml.md)         | BHWC         |
| TF SavedModel, Frozen Graph | [`TensorFlowBackend`](../reference/nn/backends/tensorflow.md) | BHWC         |
| LiteRT                      | [`LiteRTBackend`](../reference/nn/backends/litert.md)         | BCHW         |
| NCNN                        | [`NCNNBackend`](../reference/nn/backends/ncnn.md)             | BCHW         |
| PaddlePaddle                | [`PaddleBackend`](../reference/nn/backends/paddle.md)         | BCHW         |
| MNN                         | [`MNNBackend`](../reference/nn/backends/mnn.md)               | BCHW         |
| ExecuTorch                  | [`ExecuTorchBackend`](../reference/nn/backends/executorch.md) | BCHW         |

`TensorFlowBackend` covers two formats and defaults to `format="saved_model"`, so pass `format="pb"` for a frozen graph.

Three things the `YOLO()` route handles for you and a direct call does not:

- **Input layout**: `CoreMLBackend` and `TensorFlowBackend` expect BHWC. Transpose first with `im.permute(0, 2, 3, 1)`; a BCHW tensor raises a shape mismatch.
- **Autograd**: wrap calls in `torch.inference_mode()`. `TorchScriptBackend` returns a tensor that still carries a [gradient](https://www.ultralytics.com/glossary/gradient-descent) graph.
- **Post-processing**: without metadata a backend leaves `task` as `None` and `names` empty. `LiteRTBackend` still denormalizes any 3-D output by image size on the assumption it holds YOLO boxes, which is wrong for a non-YOLO model with a 3-D output. Two-dimensional outputs such as classifier logits are unaffected.

## Known Limitations

- **Multi-input support is uneven**: `torch2onnx` and `torch2openvino` accept a tuple or list of example tensors for models with multiple inputs. `torch2torchscript`, `torch2coreml`, `torch2ncnn`, `torch2paddle`, and `torch2executorch` assume a single input tensor.
- **ExecuTorch needs `flatc`**: The ExecuTorch runtime requires the FlatBuffers compiler. Install with `brew install flatbuffers` on macOS or `apt install flatbuffers-compiler` on Ubuntu.
- **No embedded metadata**: the exports above carry no Ultralytics task or input-size metadata, so `YOLO()` cannot infer either and needs both passed explicitly. See [Run Your Exported Model](#run-your-exported-model).
- **YOLO-only formats**: [Axelera](../integrations/axelera.md) and [Sony IMX500](../integrations/sony-imx500.md) exports require YOLO-specific model attributes and are not available for generic models.
- **Platform-specific formats**: [TensorRT](../integrations/tensorrt.md) requires an NVIDIA GPU. [RKNN](../integrations/rockchip-rknn.md) requires the `rknn-toolkit2` SDK (Linux only). [Edge TPU](../integrations/edge-tpu.md) requires the `edgetpu_compiler` binary (Linux only).

## Conclusion

These utilities take any PyTorch model from a plain `torch.nn.Module` to a deployment-ready ONNX, OpenVINO, CoreML, TensorFlow, or mobile-runtime artifact through one consistent API. Pick the format that matches your target hardware, [verify numerical parity](#verify-your-exported-model) against the original model, then follow the matching [integration guide](../integrations/index.md) for runtime-specific deployment steps.

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

Yes, for several formats. Pass `quantize=16` for FP16 or `quantize=8` for INT8 when exporting to OpenVINO, CoreML, MNN, or NCNN. INT8 in OpenVINO additionally requires a `calibration_dataset` argument for [post-training quantization](https://www.ultralytics.com/glossary/model-quantization). See each format's integration page for quantization trade-offs.

### How do I verify an exported model matches the original?

Run the original PyTorch model and the exported model on the same input, then compare outputs. Load the exported file with the matching backend (for example, [`ONNXBackend`](../reference/nn/backends/onnx.md) for ONNX) and check the maximum absolute difference. Judge the gap against the format's own baseline. For the ResNet-18 example above, FP32 ONNX, TF SavedModel and LiteRT sit near `1e-6`, TorchScript at `0`, and NCNN near `1e-2` because its CPU runtime defaults to FP16. A much larger gap points to unsupported ops, a wrong input shape, or a model not in eval mode. See [Verify Your Exported Model](#verify-your-exported-model) for a runnable example.
