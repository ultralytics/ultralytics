---
comments: true
description: Step-by-step guide on exporting your YOLOv8 models to various format like ONNX, TensorRT, CoreML and more for deployment. Explore now!.
keywords: YOLO, YOLOv8, Ultralytics, Model export, ONNX, TensorRT, CoreML, TensorFlow SavedModel, OpenVINO, PyTorch, export model
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

    * Export to ONNX or OpenVINO for up to 3x CPU speedup.
    * Export to TensorRT for up to 5x GPU speedup.

## Usage Examples

Export a YOLOv8n model to a different format like ONNX or TensorRT. See Arguments section below for a full list of export arguments.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom trained model

        # Export the model
        model.export(format='onnx')
        ```

    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

## Arguments

This table details the configurations and options available for exporting YOLO models to different formats. These settings are critical for optimizing the exported model's performance, size, and compatibility across various platforms and environments. Proper configuration ensures that the model is ready for deployment in the intended application with optimal efficiency.

| Argument    | Type             | Default         | Description                                                                                                                                                      |
|-------------|------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `format`    | `str`            | `'torchscript'` | Target format for the exported model, such as `'onnx'`, `'torchscript'`, `'tensorflow'`, or others, defining compatibility with various deployment environments. |
| `imgsz`     | `int` or `tuple` | `640`           | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                |
| `keras`     | `bool`           | `False`         | Enables export to Keras format for TensorFlow SavedModel, providing compatibility with TensorFlow serving and APIs.                                              |
| `optimize`  | `bool`           | `False`         | Applies optimization for mobile devices when exporting to TorchScript, potentially reducing model size and improving performance.                                |
| `half`      | `bool`           | `False`         | Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware.                                     |
| `int8`      | `bool`           | `False`         | Activates INT8 quantization, further compressing the model and speeding up inference with minimal accuracy loss, primarily for edge devices.                     |
| `dynamic`   | `bool`           | `False`         | Allows dynamic input sizes for ONNX and TensorRT exports, enhancing flexibility in handling varying image dimensions.                                            |
| `simplify`  | `bool`           | `False`         | Simplifies the model graph for ONNX exports, potentially improving performance and compatibility.                                                                |
| `opset`     | `int`            | `None`          | Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses the latest supported version.                      |
| `workspace` | `float`          | `4.0`           | Sets the maximum workspace size in GB for TensorRT optimizations, balancing memory usage and performance.                                                        |
| `nms`       | `bool`           | `False`         | Adds Non-Maximum Suppression (NMS) to the CoreML export, essential for accurate and efficient detection post-processing.                                         |

Adjusting these parameters allows for customization of the export process to fit specific requirements, such as deployment environment, hardware constraints, and performance targets. Selecting the appropriate format and settings is essential for achieving the best balance between model size, speed, and accuracy.

## Export Formats

Available YOLOv8 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`.

| Format                                                             | `format` Argument | Model                     | Metadata | Arguments                                           |
|--------------------------------------------------------------------|-------------------|---------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](../integrations/openvino.md)                            | `openvino`        | `yolov8n_openvino_model/` | ✅        | `imgsz`, `half`, `int8`                             |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n_saved_model/`    | ✅        | `imgsz`, `keras`, `int8`                            |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n_web_model/`      | ✅        | `imgsz`, `half`, `int8`                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n_paddle_model/`   | ✅        | `imgsz`                                             |
| [NCNN](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |
