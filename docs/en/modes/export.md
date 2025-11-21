---
comments: true
description: Learn how to export your YOLO11 model to various formats like ONNX, TensorRT, and CoreML. Achieve maximum compatibility and performance.
keywords: YOLO11, Model Export, ONNX, TensorRT, CoreML, Ultralytics, AI, Machine Learning, Inference, Deployment
---

# Model Export with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

The ultimate goal of training a model is to deploy it for real-world applications. Export mode in Ultralytics YOLO11 offers a versatile range of options for exporting your trained model to different formats, making it deployable across various platforms and devices. This comprehensive guide aims to walk you through the nuances of model exporting, showcasing how to achieve maximum compatibility and performance.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How To Export Custom Trained Ultralytics YOLO Model and Run Live Inference on Webcam.
</p>

## Why Choose YOLO11's Export Mode?

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

Export a YOLO11n model to a different format like ONNX or TensorRT. See the Arguments section below for a full list of export arguments.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx      # export official model
        yolo export model=path/to/best.pt format=onnx # export custom trained model
        ```

## Arguments

This table details the configurations and options available for exporting YOLO models to different formats. These settings are critical for optimizing the exported model's performance, size, and compatibility across various platforms and environments. Proper configuration ensures that the model is ready for deployment in the intended application with optimal efficiency.

| Argument    | Type              | Default         | Description                                                                                                                                                                                                                                                                                                                                                |
| ----------- | ----------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`    | `str`             | `'torchscript'` | Target format for the exported model, such as `'onnx'`, `'torchscript'`, `'engine'` (TensorRT), or others. Each format enables compatibility with different [deployment environments](https://docs.ultralytics.com/modes/export/).                                                                                                                         |
| `imgsz`     | `int` or `tuple`  | `640`           | Desired image size for the model input. Can be an integer for square images (e.g., `640` for 640×640) or a tuple `(height, width)` for specific dimensions.                                                                                                                                                                                                |
| `keras`     | `bool`            | `False`         | Enables export to Keras format for [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) SavedModel, providing compatibility with TensorFlow serving and APIs.                                                                                                                                                                                     |
| `optimize`  | `bool`            | `False`         | Applies optimization for mobile devices when exporting to TorchScript, potentially reducing model size and improving [inference](https://docs.ultralytics.com/modes/predict/) performance. Not compatible with NCNN format or CUDA devices.                                                                                                                |
| `half`      | `bool`            | `False`         | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware. Not compatible with INT8 quantization or CPU-only exports. Only available for certain formats, e.g. ONNX (see below).                                                                                                         |
| `int8`      | `bool`            | `False`         | Activates INT8 quantization, further compressing the model and speeding up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for [edge devices](https://www.ultralytics.com/blog/understanding-the-real-world-applications-of-edge-ai). When used with TensorRT, performs post-training quantization (PTQ). |
| `dynamic`   | `bool`            | `False`         | Allows dynamic input sizes for ONNX, TensorRT, and OpenVINO exports, enhancing flexibility in handling varying image dimensions. Automatically set to `True` when using TensorRT with INT8.                                                                                                                                                                |
| `simplify`  | `bool`            | `True`          | Simplifies the model graph for ONNX exports with `onnxslim`, potentially improving performance and compatibility with inference engines.                                                                                                                                                                                                                   |
| `opset`     | `int`             | `None`          | Specifies the ONNX opset version for compatibility with different [ONNX](https://docs.ultralytics.com/integrations/onnx/) parsers and runtimes. If not set, uses the latest supported version.                                                                                                                                                             |
| `workspace` | `float` or `None` | `None`          | Sets the maximum workspace size in GiB for [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) optimizations, balancing memory usage and performance. Use `None` for auto-allocation by TensorRT up to device maximum.                                                                                                                         |
| `nms`       | `bool`            | `False`         | Adds Non-Maximum Suppression (NMS) to the exported model when supported (see [Export Formats](https://docs.ultralytics.com/modes/export/)), improving detection post-processing efficiency. Not available for end2end models.                                                                                                                              |
| `batch`     | `int`             | `1`             | Specifies export model batch inference size or the maximum number of images the exported model will process concurrently in `predict` mode. For Edge TPU exports, this is automatically set to 1.                                                                                                                                                          |
| `device`    | `str`             | `None`          | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`) or DLA for NVIDIA Jetson (`device=dla:0` or `device=dla:1`). TensorRT exports automatically use GPU.                                                                                                                                        |
| `data`      | `str`             | `'coco8.yaml'`  | Path to the [dataset](https://docs.ultralytics.com/datasets/) configuration file (default: `coco8.yaml`), essential for INT8 quantization calibration. If not specified with INT8 enabled, a default dataset will be assigned.                                                                                                                             |
| `fraction`  | `float`           | `1.0`           | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used.                                                                                           |

Adjusting these parameters allows for customization of the export process to fit specific requirements, such as deployment environment, hardware constraints, and performance targets. Selecting the appropriate format and settings is essential for achieving the best balance between model size, speed, and [accuracy](https://www.ultralytics.com/glossary/accuracy).

## Export Formats

Available YOLO11 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolo11n.onnx`. Usage examples are shown for your model after export completes.

| Format                                                                                                                                              | `format` Argument | Model                       | Metadata | Arguments                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | --------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                                                                                                                     | -                 | `yolo11n.pt`                | ✅       | -                                                                                                                                                                                                           |
| [TorchScript](../integrations/torchscript.md)                                                                                                       | `torchscript`     | `yolo11n.torchscript`       | ✅       | `imgsz`, `half`, `dynamic`, `optimize`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device`                                          |
| [ONNX](../integrations/onnx.md)                                                                                                                     | `onnx`            | `yolo11n.onnx`              | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device`                                 |
| [OpenVINO](../integrations/openvino.md)                                                                                                             | `openvino`        | `yolo11n_openvino_model/`   | ✅       | `imgsz`, `half`, `dynamic`, `int8`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device`                          |
| [TensorRT](../integrations/tensorrt.md)                                                                                                             | `engine`          | `yolo11n.engine`            | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device` |
| [CoreML](../integrations/coreml.md)                                                                                                                 | `coreml`          | `yolo11n.mlpackage`         | ✅       | `imgsz`, `dynamic`, `half`, `int8`, `nms`:material-information-outline:{ title="conf, iou are also available when nms=True" }, `batch`, `device`                                                            |
| [TF SavedModel](../integrations/tf-savedmodel.md)                                                                                                   | `saved_model`     | `yolo11n_saved_model/`      | ✅       | `imgsz`, `keras`, `int8`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device`                                                        |
| [TF GraphDef](../integrations/tf-graphdef.md)                                                                                                       | `pb`              | `yolo11n.pb`                | ❌       | `imgsz`, `batch`, `device`                                                                                                                                                                                  |
| [TF Lite](../integrations/tflite.md)                                                                                                                | `tflite`          | `yolo11n.tflite`            | ✅       | `imgsz`, `half`, `int8`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `data`, `fraction`, `device`                                     |
| [TF Edge TPU](../integrations/edge-tpu.md)                                                                                                          | `edgetpu`         | `yolo11n_edgetpu.tflite`    | ✅       | `imgsz`, `device`                                                                                                                                                                                           |
| [TF.js](../integrations/tfjs.md)                                                                                                                    | `tfjs`            | `yolo11n_web_model/`        | ✅       | `imgsz`, `half`, `int8`, `nms`:material-information-outline:{ title="conf, iou, agnostic_nms are also available when nms=True" }, `batch`, `device`                                                         |
| [PaddlePaddle](../integrations/paddlepaddle.md)                                                                                                     | `paddle`          | `yolo11n_paddle_model/`     | ✅       | `imgsz`, `batch`, `device`                                                                                                                                                                                  |
| [MNN](../integrations/mnn.md)                                                                                                                       | `mnn`             | `yolo11n.mnn`               | ✅       | `imgsz`, `batch`, `int8`, `half`, `device`                                                                                                                                                                  |
| [NCNN](../integrations/ncnn.md)                                                                                                                     | `ncnn`            | `yolo11n_ncnn_model/`       | ✅       | `imgsz`, `half`, `batch`, `device`                                                                                                                                                                          |
| [IMX500](../integrations/sony-imx500.md):material-information-outline:{ title="imx format only supported for YOLOv8n and yolo11n model currently" } | `imx`             | `yolo11n_imx_model/`        | ✅       | `imgsz`, `int8`, `data`, `fraction`, `device`                                                                                                                                                               |
| [RKNN](../integrations/rockchip-rknn.md)                                                                                                            | `rknn`            | `yolo11n_rknn_model/`       | ✅       | `imgsz`, `batch`, `name`, `device`                                                                                                                                                                          |
| [ExecuTorch](../integrations/executorch.md)                                                                                                         | `executorch`      | `yolo11n_executorch_model/` | ✅       | `imgsz`, `device`                                                                                                                                                                                           |

## FAQ

### How do I export a YOLO11 model to ONNX format?

Exporting a YOLO11 model to ONNX format is straightforward with Ultralytics. It provides both Python and CLI methods for exporting models.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx      # export official model
        yolo export model=path/to/best.pt format=onnx # export custom trained model
        ```

For more details on the process, including advanced options like handling different input sizes, refer to the [ONNX integration guide](../integrations/onnx.md).

### What are the benefits of using TensorRT for model export?

Using TensorRT for model export offers significant performance improvements. YOLO11 models exported to TensorRT can achieve up to a 5x GPU speedup, making it ideal for real-time inference applications.

- **Versatility:** Optimize models for a specific hardware setup.
- **Speed:** Achieve faster inference through advanced optimizations.
- **Compatibility:** Integrate smoothly with NVIDIA hardware.

To learn more about integrating TensorRT, see the [TensorRT integration guide](../integrations/tensorrt.md).

### How do I enable INT8 quantization when exporting my YOLO11 model?

INT8 quantization is an excellent way to compress the model and speed up inference, especially on edge devices. Here's how you can enable INT8 quantization:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")  # Load a model
        model.export(format="engine", int8=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=engine int8=True # export TensorRT model with INT8 quantization
        ```

INT8 quantization can be applied to various formats, such as [TensorRT](../integrations/tensorrt.md), [OpenVINO](../integrations/openvino.md), and [CoreML](../integrations/coreml.md). For optimal quantization results, provide a representative [dataset](https://docs.ultralytics.com/datasets/) using the `data` parameter.

### Why is dynamic input size important when exporting models?

Dynamic input size allows the exported model to handle varying image dimensions, providing flexibility and optimizing processing efficiency for different use cases. When exporting to formats like [ONNX](../integrations/onnx.md) or [TensorRT](../integrations/tensorrt.md), enabling dynamic input size ensures that the model can adapt to different input shapes seamlessly.

To enable this feature, use the `dynamic=True` flag during export:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=onnx dynamic=True
        ```

Dynamic input sizing is particularly useful for applications where input dimensions may vary, such as video processing or when handling images from different sources.

### What are the key export arguments to consider for optimizing model performance?

Understanding and configuring export arguments is crucial for optimizing model performance:

- **`format:`** The target format for the exported model (e.g., `onnx`, `torchscript`, `tensorflow`).
- **`imgsz:`** Desired image size for the model input (e.g., `640` or `(height, width)`).
- **`half:`** Enables FP16 quantization, reducing model size and potentially speeding up inference.
- **`optimize:`** Applies specific optimizations for mobile or constrained environments.
- **`int8:`** Enables INT8 quantization, highly beneficial for [edge AI](https://www.ultralytics.com/blog/deploying-computer-vision-applications-on-edge-ai-devices) deployments.

For deployment on specific hardware platforms, consider using specialized export formats like [TensorRT](../integrations/tensorrt.md) for NVIDIA GPUs, [CoreML](../integrations/coreml.md) for Apple devices, or [Edge TPU](../integrations/edge-tpu.md) for Google Coral devices.
