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

| Argument       | Type              | Default         | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| -------------- | ----------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`       | `str`             | `'torchscript'` | Target format for the exported model, such as `'onnx'`, `'torchscript'`, `'engine'` (TensorRT), or others. Each format enables compatibility with different [deployment environments](https://docs.ultralytics.com/modes/export).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `name`         | `str`             | `None`          | Hardware target name for the formats that require one: Hailo architecture (`'hailo8'`, `'hailo8l'`, `'hailo10h'`, `'hailo15h'`, `'hailo15l'`; defaults to `'hailo8l'`), Rockchip RKNN chip (defaults to `'rk3588'`), Huawei Ascend SoC (a CANN `--soc_version`; defaults to `'Ascend310B4'`), or Qualcomm QNN HTP target (defaults to `'73'`). Distinct from the `project`/`name` run-naming pair used by other modes.                                                                                                                                                                                                                                                                                                                                      |
| `imgsz`        | `int` or `tuple`  | `640`           | Desired image size for the model input. Can be an integer for square images (e.g., `640` for 640×640) or a tuple `(height, width)` for specific dimensions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `keras`        | `bool`            | `False`         | Enables export to Keras format for [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) SavedModel, providing compatibility with TensorFlow serving and APIs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `optimize`     | `bool`            | `False`         | Enables higher compiler optimization for DEEPX, reducing inference latency while increasing compilation time.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `quantize`     | `int` or `str`    | `None`          | Quantization precision: `16` (FP16, reduces model size and can speed up inference on supported hardware) or `8` (INT8/PTQ, further compresses the model with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for [edge devices](https://www.ultralytics.com/blog/understanding-the-real-world-applications-of-edge-ai); needs calibration `data`/`fraction`); `32`/unset is FP32. Export formats that support mixed weight/activation precision also accept the `'w8a8'`/`'w16a16'`/`'w8a16'`/`'w8a32'` notation. Replaces the deprecated `half`/`int8` flags (`half=True` → `16`, `int8=True` → `8`, still accepted with a deprecation warning). Only precisions supported by the target format are allowed (see below). |
| `dynamic`      | `bool`            | `False`         | Allows dynamic input sizes for TorchScript, ONNX, OpenVINO, TensorRT, and CoreML exports, enhancing flexibility in handling varying image dimensions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `simplify`     | `bool`            | `True`          | Simplifies the intermediate ONNX graph with `onnxslim` for the exports that build one (see [Export Formats](https://docs.ultralytics.com/modes/export)), potentially improving performance and compatibility with inference engines.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `opset`        | `int`             | `None`          | Specifies the ONNX opset version for the exports that build an ONNX graph (see [Export Formats](https://docs.ultralytics.com/modes/export)), for compatibility with different [ONNX](https://docs.ultralytics.com/integrations/onnx) parsers and runtimes. If not set, uses the latest supported version.                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `workspace`    | `float` or `None` | `None`          | Sets the maximum workspace size in GiB for [TensorRT](https://docs.ultralytics.com/integrations/tensorrt) optimizations, balancing memory usage and performance. Use `None` for auto-allocation by TensorRT up to device maximum.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `nms`          | `bool`            | `False`         | Adds Non-Maximum Suppression (NMS) to the exported model when supported (see [Export Formats](https://docs.ultralytics.com/modes/export)), improving detection post-processing efficiency. Not available for end2end models. For CoreML, only supported for detection models.                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `conf`         | `float`           | `None`          | Confidence threshold used wherever export-time NMS is generated: `nms=True` exports; Hailo's non-end-to-end detect exports; and IMX's detect, pose, and segment exports, which force `nms=True` internally. Defaults to `0.25` when unset, except for IMX exports, which default to `0.001`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `iou`          | `float`           | `0.7`           | [IoU](https://www.ultralytics.com/glossary/intersection-over-union-iou) threshold used wherever export-time NMS is generated: `nms=True` exports; Hailo's non-end-to-end detect exports; and IMX's detect, pose, and segment exports, which force `nms=True` internally.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `max_det`      | `int`             | `300`           | Maximum number of detections kept in the exported model's output. Applies to `nms=True` exports on every format except CoreML, whose NMS pipeline has no detection cap, plus NMS-free end-to-end detection exports (YOLO26, YOLOv10, clamped to the number of available anchors) and IMX's detect, pose, and segment exports.                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `agnostic_nms` | `bool`            | `False`         | Enables class-agnostic NMS wherever export-time NMS is generated through the standard `nms=True` pipeline, including CoreML's own NMS stage, suppressing lower-scoring overlapping boxes across different classes rather than only within the same class. Not honored by Hailo's or IMX's own generated NMS configs, which have no class-agnostic option and remain class-aware regardless of this flag. Also baked into NMS-free end-to-end exports (YOLO26, YOLOv10), where it only prevents the same detection from appearing under multiple class labels (IoU=1.0 duplicates), not IoU-threshold suppression between distinct boxes.                                                                                                                    |
| `batch`        | `int`             | `1`             | Specifies export model batch inference size or the maximum number of images the exported model will process concurrently in `predict` mode. For Edge TPU exports, this is automatically set to 1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `device`       | `str`             | `None`          | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`), Huawei Ascend NPU (`device=npu` or `device=npu:0`), or DLA for NVIDIA Jetson (`device=dla:0` or `device=dla:1`). TensorRT exports automatically use GPU, but TensorRT 11.0 does not support DLA.                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `verbose`      | `bool`            | `True`          | Raises the TensorRT builder log to VERBOSE severity during `format='engine'` export. Other export formats ignore it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `data`         | `str`             | `None`          | Path to the [dataset](https://docs.ultralytics.com/datasets) YAML, essential for INT8 quantization calibration; classification instead takes a dataset directory or a built-in dataset name. If not specified with INT8 enabled, Ultralytics selects a task-specific calibration dataset where required, or falls back to the default dataset for the model task.                                                                                                                                                                                                                                                                                                                                                                                           |
| `split`        | `str`             | `'val'`         | Dataset split (`'train'`, `'val'`, or `'test'`) used to build the INT8 quantization calibration dataloader from `data`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `fraction`     | `float`           | `1.0`           | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `end2end`      | `bool`            | `None`          | Overrides the end-to-end mode in YOLO models that support NMS-free inference (YOLO26, YOLOv10). Setting it to `False` lets you export these models to be compatible with the traditional NMS-based postprocessing pipeline. See the [End-to-End Detection guide](../guides/end2end-detection.md) for details.                                                                                                                                                                                                                                                                                                                                                                                                                                               |

Adjusting these parameters allows for customization of the export process to fit specific requirements, such as deployment environment, hardware constraints, and performance targets. Selecting the appropriate format and settings is essential for achieving the best balance between model size, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy).

## Export Formats

Available YOLO26 export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n.onnx`. Usage examples are shown for your model after export completes. Models can also be exported directly from the browser on [Ultralytics Platform](../platform/train/models.md#export-model) without any local setup.

| Format | `format` Argument | Model | Metadata | Arguments |
| ---------------------------------------------------------- | ----------------- | ------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| [PyTorch](https://pytorch.org/) | - | `yolo26n.pt` | ✅ | - |
| [TorchScript](../integrations/torchscript.md) | `torchscript` | `yolo26n.torchscript` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [ONNX](../integrations/onnx.md) | `onnx` | `yolo26n.onnx` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [OpenVINO](../integrations/openvino.md) | `openvino` | `yolo26n_openvino_model/` | ✅ | `imgsz`, `quantize`, `dynamic`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TensorRT](../integrations/tensorrt.md) | `engine` | `yolo26n.engine` | ✅ | `imgsz`, `quantize`, `dynamic`, `simplify`, `opset`, `workspace`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [CoreML](../integrations/coreml.md) | `coreml` | `yolo26n.mlpackage` | ✅ | `imgsz`, `dynamic`, `quantize`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device` |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model` | `yolo26n_saved_model/` | ✅ | `imgsz`, `keras`, `quantize`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [TF GraphDef](../integrations/tf-graphdef.md) | `pb` | `yolo26n.pb` | ❌ | `imgsz`, `opset`, `batch`, `device` |
| [TF Edge TPU](../integrations/edge-tpu.md) | `edgetpu` | `yolo26n_edgetpu.tflite` | ✅ | `imgsz`, `quantize`, `opset`, `data`, `fraction`, `device` |
| [PaddlePaddle](../integrations/paddlepaddle.md) | `paddle` | `yolo26n_paddle_model/` | ✅ | `imgsz`, `batch`, `device` |
| [MNN](../integrations/mnn.md) | `mnn` | `yolo26n.mnn` | ✅ | `imgsz`, `batch`, `dynamic`, `quantize`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [NCNN](../integrations/ncnn.md) | `ncnn` | `yolo26n_ncnn_model/` | ✅ | `imgsz`, `quantize`, `batch`, `device` |
| [IMX500](../integrations/sony-imx500.md):material-information-outline:{ title="IMX format is currently only supported for YOLOv8n, YOLO11n models" } | `imx` | `yolo26n_imx_model/` | ✅ | `imgsz`, `quantize`, `data`, `fraction`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `device` |
| [RKNN](../integrations/rockchip-rknn.md) | `rknn` | `yolo26n_rknn_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [ExecuTorch](../integrations/executorch.md) | `executorch` | `yolo26n_executorch_model/` | ✅ | `imgsz`, `batch`, `device` |
| [Axelera](../integrations/axelera.md) | `axelera` | `yolo26n_axelera_model/` | ✅ | `imgsz`, `batch`, `quantize`, `data`, `fraction`, `device` |
| [DEEPX](../integrations/deepx.md) | `deepx` | `yolo26n_deepx_model/` | ✅ | `imgsz`, `quantize`, `simplify`, `opset`, `data`, `optimize`, `device` |
| [Qualcomm QNN](../integrations/qnn.md) | `qnn` | `yolo26n_qnn.onnx` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `simplify`, `opset`, `data`, `fraction`, `device` |
| [LiteRT](../integrations/litert.md) | `litert` | `yolo26n.tflite` | ✅ | `imgsz`, `quantize`, `batch`, `data`, `fraction`, `device` |
| [Hailo](../integrations/hailo.md) | `hailo` | `yolo26n_hailo_model/` | ✅ | `imgsz`, `name`, `quantize`, `data`, `fraction`, `simplify`, `conf`, `iou` |
| [Huawei Ascend](../integrations/ascend.md) | `ascend` | `yolo26n_ascend_model/` | ✅ | `imgsz`, `batch`, `name`, `quantize`, `opset`, `simplify`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" } |

## Quantization Options

Use the `quantize` argument to request the export precision. String values are case-insensitive, and Ultralytics canonicalizes accepted aliases before export:

| Request values                     | Canonical value | Meaning                                                                         |
| ---------------------------------- | --------------- | ------------------------------------------------------------------------------- |
| `8`, `"8"`, `"int8"`, `"w8a8"`     | `8`             | INT8 weights and activations                                                    |
| `16`, `"16"`, `"fp16"`, `"w16a16"` | `16`            | FP16 weights and activations                                                    |
| `32`, `"32"`, `"fp32"`, `"w32a32"` | `32`            | FP32 export; same as unset except CoreML NMS ML Programs, which default to FP16 |
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
| CoreML        | ✅¹               | ✅                | ✅         | ✅                | CoreML INT8 is weight quantization; W8A16 uses INT8 weights with FP16 activations. ¹Unset NMS ML Programs default to FP16.                                                                                                                              |
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
| Huawei Ascend | ❌                | ✅ auto           | ❌         | ❌                | Ascend AI Core convolutions accept only FP16/INT8 inputs, so ATC compiles FP16; it is auto-enabled when unset.                                                                                                                                          |

For INT8 and W8A16 exports, provide representative calibration data with `data`, such as `data="coco8.yaml"`, unless the target integration documents a default or auto-enabled behavior. The LiteRT `"w8a32"` (dynamic INT8) scheme needs no calibration data.

## What's Next

Find your deployment target's integration guide — [ONNX](../integrations/onnx.md), [TensorRT](../integrations/tensorrt.md), [CoreML](../integrations/coreml.md), and more are on the [full integrations list](../integrations/index.md) — for how to run the exported model.

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
- **`optimize:`** Enables higher compiler optimization for DEEPX exports.

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
