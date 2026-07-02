---
comments: true
description: Learn how to export your YOLO26 model to various formats like ONNX, TensorRT, and CoreML. Achieve maximum compatibility and performance.
keywords: YOLO26, Model Export, ONNX, TensorRT, CoreML, Ultralytics, AI, Machine Learning, Inference, Deployment
---

# Model Export with Ultralytics YOLO

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

The ultimate goal of training a model is to deploy it for real-world applications. Export mode in Ultralytics YOLO26 offers a versatile range of options for exporting your trained model to different formats, making it deployable across various platforms and devices. This comprehensive guide aims to walk you through the nuances of model exporting, showcasing how to achieve maximum compatibility and performance.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/KGHYU-MKYeE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Export Ultralytics YOLO26 in different formats for Deployment | ONNX, TensorRT, CoreML 🚀
</p>

## Why Choose YOLO26's Export Mode?

- **Versatility:** Export to multiple formats including [ONNX](../integrations/onnx.md), [TensorRT](../integrations/tensorrt.md), [CoreML](../integrations/coreml.md), and more.
- **Performance:** Gain up to 5x GPU speedup with TensorRT and 3x CPU speedup with ONNX or [OpenVINO](../integrations/openvino.md).
- **Compatibility:** Make your model universally deployable across numerous hardware and software environments.
- **Ease of Use:** Simple CLI and Python API for quick and straightforward model exporting.

### Key Features of Export Mode

Here are some of the standout functionalities:

- **One-Click Export:** Simple commands for exporting to different formats.
- **Batch Export:** Export batched-inference capable models.
- **Optimized Inference:** Exported models are optimized for quicker inference times.
- **Tutorial Videos:** In-depth guides and tutorials for a smooth exporting experience.

!!! tip

    * Export to [ONNX](../integrations/onnx.md) or [OpenVINO](../integrations/openvino.md) for up to 3x CPU speedup.
    * Export to [TensorRT](../integrations/tensorrt.md) for up to 5x GPU speedup.

## Usage Examples

Export a YOLO26n model to a different format like ONNX or TensorRT. See the Arguments section below for a full list of export arguments.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx      # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

## Arguments

This table details the configurations and options available for exporting YOLO models to different formats. These settings are critical for optimizing the exported model's performance, size, and compatibility across various platforms and environments. Proper configuration ensures that the model is ready for deployment in the intended application with optimal efficiency.

{% include "macros/export-args.md" %}

Adjusting these parameters allows for customization of the export process to fit specific requirements, such as deployment environment, hardware constraints, and performance targets. Selecting the appropriate format and settings is essential for achieving the best balance between model size, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy).

## Export Formats

Available YOLO26 export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n.onnx`. Usage examples are shown for your model after export completes. Models can also be exported directly from the browser on [Ultralytics Platform](https://platform.ultralytics.com) without any local setup.

{% include "macros/export-table.md" %}

## Quantization Options

Use the `quantize` argument to request the export precision. String values are case-insensitive, and Ultralytics canonicalizes accepted aliases before export:

| Request values                     | Canonical value | Meaning                                                                         |
| ---------------------------------- | --------------- | ------------------------------------------------------------------------------- |
| `8`, `"8"`, `"int8"`, `"w8a8"`     | `8`             | INT8 weights and activations                                                    |
| `16`, `"16"`, `"fp16"`, `"w16a16"` | `16`            | FP16 weights and activations                                                    |
| `32`, `"32"`, `"fp32"`, `"w32a32"` | `32`            | FP32 export; same precision as leaving `quantize` unset                         |
| `"w8a16"`                          | `"w8a16"`       | INT8 weights with 16-bit activations (FP16; INT16 on LiteRT)                    |
| `"w8a32"`                          | `"w8a32"`       | INT8 weights with FP32 activations (LiteRT dynamic INT8, no calibration needed) |

The legacy `half=True` and `int8=True` flags are still accepted with deprecation warnings and forward to `quantize=16` and `quantize=8`.

Not every export format supports every precision. Explicit `quantize` requests either produce that precision or fail before export:

| Format        | FP32 (`32`/unset) | FP16 (`16`)       | INT8 (`8`) | W8A16 (`"w8a16"`) | Notes                                                                                                                                                                                                                                                   |
| ------------- | ----------------- | ----------------- | ---------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch       | ✅                | N/A               | N/A        | N/A               | Native training/checkpoint format.                                                                                                                                                                                                                      |
| TorchScript   | ✅                | ✅ GPU only       | ❌         | ❌                | FP16 TorchScript export requires `device=0`; CPU export is FP32.                                                                                                                                                                                        |
| ONNX          | ✅                | ✅                | ✅         | ❌                | INT8 uses ONNX Runtime static quantization and calibration data.                                                                                                                                                                                        |
| OpenVINO      | ✅                | ✅                | ✅         | ❌                | INT8 uses NNCF post-training quantization.                                                                                                                                                                                                              |
| TensorRT      | ✅                | ✅                | ✅         | ❌                | INT8 needs representative calibration data.                                                                                                                                                                                                             |
| CoreML        | ✅                | ✅                | ✅         | ✅                | CoreML INT8 is weight quantization; W8A16 uses INT8 weights with FP16 activations.                                                                                                                                                                      |
| TF SavedModel | ✅                | ❌                | ✅         | ❌                | INT8 export uses TensorFlow calibration.                                                                                                                                                                                                                |
| TF GraphDef   | ✅                | ❌                | ❌         | ❌                | No export-time precision conversion.                                                                                                                                                                                                                    |
| Edge TPU      | ❌                | ❌                | ✅ auto    | ❌                | Edge TPU requires INT8; it is auto-enabled when unset.                                                                                                                                                                                                  |
| PaddlePaddle  | ✅                | ❌                | ❌         | ❌                | No export-time precision conversion.                                                                                                                                                                                                                    |
| MNN           | ✅                | ✅                | ✅         | ❌                | INT8 is weight quantization through MNN conversion.                                                                                                                                                                                                     |
| NCNN          | ✅                | ✅                | ❌         | ❌                | Mobile/embedded runtime format.                                                                                                                                                                                                                         |
| IMX500        | ❌                | ❌                | ✅ auto    | ✅                | IMX500 requires quantization; INT8 is auto-enabled when unset.                                                                                                                                                                                          |
| RKNN          | ❌                | ✅ chip-dependent | ✅         | ❌                | RK3588/RK3576/RK3566/RK3568/RK3562/RK2118/RV1126B support FP16 or INT8; RV1103/RV1106 variants are INT8-only.                                                                                                                                           |
| ExecuTorch    | ✅                | ❌                | ❌         | ❌                | No export-time precision conversion.                                                                                                                                                                                                                    |
| Axelera       | ❌                | ❌                | ✅ auto    | ❌                | Axelera export requires INT8; it is auto-enabled when unset.                                                                                                                                                                                            |
| DEEPX         | ❌                | ❌                | ✅ auto    | ❌                | DEEPX export requires INT8; it is auto-enabled when unset.                                                                                                                                                                                              |
| Qualcomm QNN  | ❌                | ❌                | ❌         | ✅ auto           | QNN HTP export is fixed to INT8 weights with 16-bit activations.                                                                                                                                                                                        |
| LiteRT        | ✅                | ❌                | ✅         | ✅                | Static INT8 (`8`) and `"w8a16"` (int8 weights + **int16** activations) use calibration data; also supports `"w8a32"` dynamic INT8 (no calibration). `quantize=16` is not a separate export; an FP32 model runs in FP16 at runtime via the GPU delegate. |

For INT8 and W8A16 exports, provide representative calibration data with `data`, such as `data="coco8.yaml"`, unless the target integration documents a default or auto-enabled behavior. The LiteRT `"w8a32"` (dynamic INT8) scheme needs no calibration data.

## FAQ

### How do I export a YOLO26 model to ONNX format?

Exporting a YOLO26 model to ONNX format is straightforward with Ultralytics. It provides both Python and CLI methods for exporting models.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx      # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

For more details on the process, including advanced options like handling different input sizes, refer to the [ONNX integration guide](../integrations/onnx.md).

### What are the benefits of using TensorRT for model export?

Using TensorRT for model export offers significant performance improvements. YOLO26 models exported to TensorRT can achieve up to a 5x GPU speedup, making it ideal for real-time inference applications.

- **Versatility:** Optimize models for a specific hardware setup.
- **Speed:** Achieve faster inference through advanced optimizations.
- **Compatibility:** Integrate smoothly with NVIDIA hardware.

To learn more about integrating TensorRT, see the [TensorRT integration guide](../integrations/tensorrt.md).

### How do I enable INT8 quantization when exporting my YOLO26 model?

INT8 quantization is an excellent way to compress the model and speed up inference, especially on edge devices. Here's how you can enable INT8 quantization:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # Load a model
        model.export(format="onnx", quantize=8, data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx quantize=8 data=coco8.yaml # export ONNX model with INT8 quantization
        ```

INT8 quantization can be applied to formats such as [ONNX](../integrations/onnx.md), [TensorRT](../integrations/tensorrt.md), [OpenVINO](../integrations/openvino.md), [CoreML](../integrations/coreml.md), and [Rockchip RKNN](../integrations/rockchip-rknn.md). For optimal quantization results, provide a representative [dataset](../datasets/index.md) using the `data` parameter. See [Quantization Options](#quantization-options) for accepted `quantize` values and supported formats.

### Why is dynamic input size important when exporting models?

Dynamic input size allows the exported model to handle varying image dimensions, providing flexibility and optimizing processing efficiency for different use cases. When exporting to formats like [ONNX](../integrations/onnx.md) or [TensorRT](../integrations/tensorrt.md), enabling dynamic input size ensures that the model can adapt to different input shapes seamlessly.

To enable this feature, use the `dynamic=True` flag during export:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx dynamic=True
        ```

Dynamic input sizing is particularly useful for applications where input dimensions may vary, such as video processing or when handling images from different sources.

### What are the key export arguments to consider for optimizing model performance?

Understanding and configuring export arguments is crucial for optimizing model performance:

- **`format:`** The target format for the exported model (e.g., `onnx`, `torchscript`, `tensorflow`).
- **`imgsz:`** Desired image size for the model input (e.g., `640` or `(height, width)`).
- **`quantize:`** Quantization precision, such as `8`/`"int8"`, `16`/`"fp16"`, `32`/`"fp32"`, or the mixed weight/activation schemes `"w8a16"` and `"w8a32"` (LiteRT dynamic INT8) on supported formats. See [Quantization Options](#quantization-options).
- **`optimize:`** Applies specific optimizations for mobile or constrained environments.

For deployment on specific hardware platforms, consider using specialized export formats like [TensorRT](../integrations/tensorrt.md) for NVIDIA GPUs, [CoreML](../integrations/coreml.md) for Apple devices, or [Edge TPU](../integrations/edge-tpu.md) for Google Coral devices.

### What do the output tensors represent in exported YOLO models?

When you export a YOLO model to formats like ONNX or TensorRT, the output tensor structure depends on the model task. Understanding these outputs is important for custom inference implementations.

For **YOLO26 detection models** (e.g., `yolo26n.pt`), end-to-end export is enabled by default in formats that support it, so the output is shaped like `(batch_size, max_detections, 6)` with `[x1, y1, x2, y2, confidence, class_id]` values. With the default `max_det=300`, this is commonly `(batch_size, 300, 6)`. Some constrained formats automatically fall back to the traditional output layout when end-to-end operators are unsupported.

For non-end-to-end detection models, or YOLO26 models exported with `end2end=False`, the output is typically a single tensor shaped like `(batch_size, 4 + num_classes, num_predictions)` where the channels represent box coordinates plus per-class scores, and `num_predictions` depends on the export input resolution (and can be dynamic).

For **segmentation models** (e.g., `yolo26n-seg.pt`), you'll typically get two outputs: the first tensor shaped like `(batch_size, 4 + num_classes + mask_dim, num_predictions)` (boxes, class scores, and mask coefficients), and the second tensor shaped like `(batch_size, mask_dim, proto_h, proto_w)` containing mask prototypes used with the coefficients to generate instance masks. Sizes depend on the export input resolution (and can be dynamic).

For **pose models** (e.g., `yolo26n-pose.pt`), the output tensor is typically shaped like `(batch_size, 4 + num_classes + keypoint_dims, num_predictions)`, where `keypoint_dims` depends on the pose specification (e.g., number of keypoints and whether confidence is included), and `num_predictions` depends on the export input resolution (and can be dynamic).

The examples in the [ONNX inference examples](https://github.com/ultralytics/ultralytics/tree/main/examples) demonstrate how to process these outputs for each model type.

### Is there an official Ultralytics C++ inference API?

Ultralytics does not currently provide a dedicated C++ inference API for YOLO models. For C++ deployments, export the
model to a runtime format such as [ONNX](../integrations/onnx.md), [TensorRT](../integrations/tensorrt.md),
[TorchScript](../integrations/torchscript.md), or [MNN](../integrations/mnn.md), then load the exported artifact with
that runtime's native C++ API.

For example, export a detection model with `yolo export model=yolo26n.pt format=onnx` and run the `.onnx` file with
ONNX Runtime C++, or export with `format=engine` and run the TensorRT engine from a TensorRT C++ application. When you
use custom C++ post-processing, match the output tensor layout for your task and export settings; YOLO26 end-to-end
detection exports usually return `(batch, max_det, 6)`, while non-end-to-end exports return raw prediction tensors that
require external post-processing.

### Why is `output0` FP32 when exporting quantized models with `end2end=True`?

When exporting with `quantize=16` (FP16) or `quantize=8` (INT8), most tensors are converted to lower precision to reduce model size and improve performance. However, when `end2end=True` is enabled, post-processing (including class indices) is embedded directly in the exported graph.

The `output0` tensor contains class indices, which are internally represented as floating-point values. FP16 cannot reliably represent integer values above 2048 due to its limited mantissa precision. To avoid potential precision loss or incorrect class IDs, `output0` is intentionally kept in FP32.

This behavior is expected and also applies to lower-precision or quantized exports where class index fidelity must be preserved.

If full FP16 outputs are required, export with `end2end=False` and perform post-processing externally.

### Why does `end2end=False` on YOLO26 still use the one-to-one head?

YOLO26 models are trained with end-to-end detection using two heads: a one-to-one head (used for NMS-free inference) and a one-to-many head (used as an auxiliary training signal). During training, the one-to-many head's loss weight decays from 0.8 to 0.1 over the training epochs, so by the end of training the one-to-one head is the well-calibrated one while the one-to-many head produces poorly calibrated scores.

When you export with `end2end=False` (e.g., for formats that don't support the topk post-processing, or to run custom NMS), the exporter automatically uses the one-to-one head's output with external NMS rather than the decayed one-to-many head. This ensures the exported model uses the best-trained head regardless of the `end2end` setting. A warning is logged during export to inform you of this behavior.

See [issue #24668](https://github.com/ultralytics/ultralytics/issues/24668) for the detailed analysis.
