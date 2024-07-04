---
comments: true
description: Learn how to export your YOLOv8 model to various formats like ONNX, TensorRT, and CoreML. Achieve maximum compatibility and performance.
keywords: YOLOv8, Model Export, ONNX, TensorRT, CoreML, Ultralytics, AI, Machine Learning, Inference, Deployment
---

# Model Export with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

The ultimate goal of training a model is to deploy it for real-world applications. Export mode in Ultralytics YOLOv8 offers a versatile range of options for exporting your trained model to different formats, making it deployable across various platforms and devices. This comprehensive guide aims to walk you through the nuances of model exporting, showcasing how to achieve maximum compatibility and performance.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How To Export Custom Trained Ultralytics YOLOv8 Model and Run Live Inference on Webcam.
</p>

## Why Choose YOLOv8's Export Mode?

- **Versatility:** Export to multiple formats including ONNX, TensorRT, CoreML, and more.
- **Performance:** Gain up to 5x GPU speedup with TensorRT and 3x CPU speedup with ONNX or OpenVINO.
- **Compatibility:** Make your model universally deployable across numerous hardware and software environments.
- **Ease of Use:** Simple CLI and Python API for quick and straightforward model exporting.

### Key Features of Export Mode

Here are some of the standout functionalities:

- **One-Click Export:** Simple commands for exporting to different formats.
- **Batch Export:** Export batched-inference capable models.
- **Optimized Inference:** Exported models are optimized for quicker inference times.
- **Tutorial Videos:** In-depth guides and tutorials for a smooth exporting experience.

!!! Tip "Tip"

    * Export to [ONNX](../integrations/onnx.md) or [OpenVINO](../integrations/openvino.md) for up to 3x CPU speedup.
    * Export to [TensorRT](../integrations/tensorrt.md) for up to 5x GPU speedup.

## Usage Examples

Export a YOLOv8n model to a different format like ONNX or TensorRT. See Arguments section below for a full list of export arguments.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

## Arguments

This table details the configurations and options available for exporting YOLO models to different formats. These settings are critical for optimizing the exported model's performance, size, and compatibility across various platforms and environments. Proper configuration ensures that the model is ready for deployment in the intended application with optimal efficiency.

| Argument    | Type             | Default         | Description                                                                                                                                                      |
| ----------- | ---------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`    | `str`            | `'torchscript'` | Target format for the exported model, such as `'onnx'`, `'torchscript'`, `'tensorflow'`, or others, defining compatibility with various deployment environments. |
| `imgsz`     | `int` or `tuple` | `640`           | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                |
| `keras`     | `bool`           | `False`         | Enables export to Keras format for TensorFlow SavedModel, providing compatibility with TensorFlow serving and APIs.                                              |
| `optimize`  | `bool`           | `False`         | Applies optimization for mobile devices when exporting to TorchScript, potentially reducing model size and improving performance.                                |
| `half`      | `bool`           | `False`         | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                     |
| `int8`      | `bool`           | `False`         | Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices.                     |
| `dynamic`   | `bool`           | `False`         | Allows dynamic input sizes for ONNX and TensorRT exports, enhancing flexibility in handling varying image dimensions.                                            |
| `simplify`  | `bool`           | `False`         | Simplifies the model graph for ONNX exports with `onnxslim`, potentially improving performance and compatibility.                                                |
| `opset`     | `int`            | `None`          | Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses the latest supported version.                      |
| `workspace` | `float`          | `4.0`           | Sets the maximum workspace size in GiB for TensorRT optimizations, balancing memory usage and performance.                                                       |
| `nms`       | `bool`           | `False`         | Adds Non-Maximum Suppression (NMS) to the CoreML export, essential for accurate and efficient detection post-processing.                                         |
| `batch`     | `int`            | `1`             | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                          |

Adjusting these parameters allows for customization of the export process to fit specific requirements, such as deployment environment, hardware constraints, and performance targets. Selecting the appropriate format and settings is essential for achieving the best balance between model size, speed, and accuracy.

## Export Formats

Available YOLOv8 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolov8n.onnx`. Usage examples are shown for your model after export completes.

| Format                                            | `format` Argument | Model                     | Metadata | Arguments                                                            |
| ------------------------------------------------- | ----------------- | ------------------------- | -------- | -------------------------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                   | -                 | `yolov8n.pt`              | ✅       | -                                                                    |
| [TorchScript](../integrations/torchscript.md)     | `torchscript`     | `yolov8n.torchscript`     | ✅       | `imgsz`, `optimize`, `batch`                                         |
| [ONNX](../integrations/onnx.md)                   | `onnx`            | `yolov8n.onnx`            | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `batch`             |
| [OpenVINO](../integrations/openvino.md)           | `openvino`        | `yolov8n_openvino_model/` | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TensorRT](../integrations/tensorrt.md)           | `engine`          | `yolov8n.engine`          | ✅       | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `batch` |
| [CoreML](../integrations/coreml.md)               | `coreml`          | `yolov8n.mlpackage`       | ✅       | `imgsz`, `half`, `int8`, `nms`, `batch`                              |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model`     | `yolov8n_saved_model/`    | ✅       | `imgsz`, `keras`, `int8`, `batch`                                    |
| [TF GraphDef](../integrations/tf-graphdef.md)     | `pb`              | `yolov8n.pb`              | ❌       | `imgsz`, `batch`                                                     |
| [TF Lite](../integrations/tflite.md)              | `tflite`          | `yolov8n.tflite`          | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [TF Edge TPU](../integrations/edge-tpu.md)        | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅       | `imgsz`                                                              |
| [TF.js](../integrations/tfjs.md)                  | `tfjs`            | `yolov8n_web_model/`      | ✅       | `imgsz`, `half`, `int8`, `batch`                                     |
| [PaddlePaddle](../integrations/paddlepaddle.md)   | `paddle`          | `yolov8n_paddle_model/`   | ✅       | `imgsz`, `batch`                                                     |
| [NCNN](../integrations/ncnn.md)                   | `ncnn`            | `yolov8n_ncnn_model/`     | ✅       | `imgsz`, `half`, `batch`                                             |



## FAQ

### How do I export my YOLOv8 model to ONNX format using Ultralytics?

To export your YOLOv8 model to ONNX format, you can use either Python or the command line interface (CLI). Below are the steps for each method:

**Python:**
```python
from ultralytics import YOLO

# Load a custom trained model
model = YOLO("path/to/best.pt")

# Export the model to ONNX format
model.export(format="onnx")
```

**CLI:**
```bash
yolo export model=path/to/best.pt format=onnx
```
Exporting to [ONNX](../integrations/onnx.md) can provide up to a 3x CPU speedup, making it ideal for deployment in resource-constrained environments.

### What are the benefits of using TensorRT for YOLOv8 model exports?

Using TensorRT to export your YOLOv8 model offers several benefits:
- **Performance:** Gain up to 5x GPU speedup, significantly enhancing inference speed.
- **Efficiency:** Optimizes model execution for NVIDIA hardware, reducing latency.
- **Compatibility:** Integrates well with various deployment environments, especially those utilizing NVIDIA GPUs.

For step-by-step instructions, follow our [TensorRT export guide](../integrations/tensorrt.md).

### Can I export my YOLOv8 model with FP16 precision and what are the advantages?

Yes, you can export your YOLOv8 model with FP16 (half-precision) using the `half` argument. FP16 quantization reduces the model size and can speed up inference on supported hardware. This is especially useful for deployment on devices with limited computational power.

Example:
```python
model.export(format="onnx", half=True)
```
Using FP16 precision is detailed under the [ONNX integration](../integrations/onnx.md) section.

### How do I specify the input image size when exporting a YOLOv8 model?

You can set the desired input image size using the `imgsz` argument during the export process. This can be an integer for square images (e.g., 640) or a tuple for specific dimensions (e.g., (320, 640)).

Example:
```python
model.export(format="onnx", imgsz=(320, 640))
```
Setting image size appropriately ensures optimal performance and compatibility, as explained in the [Arguments](#arguments) section.

### Why should I use dynamic input sizes when exporting to ONNX or TensorRT formats?

Using dynamic input sizes allows the exported model to handle varying image dimensions, improving flexibility and usability in different deployment scenarios.

Example:
```python
model.export(format="engine", dynamic=True)
```
This feature is particularly beneficial for applications where input sizes are not constant, as described in the [Export Formats](#export-formats) section.