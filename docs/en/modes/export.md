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

| Key         | Default Value   | Description                                                                                                                                                       |
|-------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `format`    | `'torchscript'` | Specifies the export format. Supported values include `'torchscript'`, `'onnx'`, `'coreml'`, `'engine'` (TensorRT), `'saved_model'` (TensorFlow SavedModel), etc. |
| `imgsz`     | `640`           | Defines the image size for export. Accepts an integer for square images or a tuple `(height, width)` for non-square images.                                       |
| `keras`     | `False`         | When exporting to TensorFlow SavedModel, setting this to `True` utilizes Keras for the export process.                                                            |
| `optimize`  | `False`         | Applies to TorchScript exports, enabling optimization for mobile deployments.                                                                                     |
| `half`      | `False`         | Enables half-precision (FP16) quantization for the exported model, reducing size and potentially increasing inference speed on compatible hardware.               |
| `int8`      | `False`         | Activates INT8 quantization, further reducing model size and increasing inference speed at the cost of precision. Useful for edge devices.                        |
| `dynamic`   | `False`         | For ONNX and TensorRT formats, enables dynamic axes, allowing variable input sizes for inference.                                                                 |
| `simplify`  | `False`         | Simplifies the model structure for ONNX and TensorRT formats, potentially improving efficiency and compatibility.                                                 |
| `opset`     | `None`          | Specifies the ONNX opset version for export. If not set, uses the latest supported version. Useful for ensuring compatibility with older ONNX parsers.            |
| `workspace` | `4`             | Defines the maximum workspace size in GB for TensorRT exports, affecting the optimization process and memory usage.                                               |
| `nms`       | `False`         | When exporting to CoreML, adds a Non-Maximum Suppression (NMS) layer to the model, useful for filtering overlapping detections.                                   |

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
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |
