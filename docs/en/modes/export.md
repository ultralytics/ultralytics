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

### How do I export a YOLOv8 model to ONNX format?

Exporting a YOLOv8 model to ONNX format is straightforward with Ultralytics. It provides both Python and CLI methods for exporting models.

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

For more details on the process, including advanced options like handling different input sizes, refer to the [ONNX](../integrations/onnx.md) section.

### What are the benefits of using TensorRT for model export?

Using TensorRT for model export offers significant performance improvements. YOLOv8 models exported to TensorRT can achieve up to a 5x GPU speedup, making it ideal for real-time inference applications.

- **Versatility:** Optimize models for a specific hardware setup.
- **Speed:** Achieve faster inference through advanced optimizations.
- **Compatibility:** Integrate smoothly with NVIDIA hardware.

To learn more about integrating TensorRT, see the [TensorRT](../integrations/tensorrt.md) integration guide.

### How do I enable INT8 quantization when exporting my YOLOv8 model?

INT8 quantization is an excellent way to compress the model and speed up inference, especially on edge devices. Here's how you can enable INT8 quantization:

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")  # Load a model
        model.export(format="onnx", int8=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx int8=True   # export model with INT8 quantization
        ```

INT8 quantization can be applied to various formats, such as TensorRT and CoreML. More details can be found in the [Export](../modes/export.md) section.

### Why is dynamic input size important when exporting models?

Dynamic input size allows the exported model to handle varying image dimensions, providing flexibility and optimizing processing efficiency for different use cases. When exporting to formats like ONNX or TensorRT, enabling dynamic input size ensures that the model can adapt to different input shapes seamlessly.

To enable this feature, use the `dynamic=True` flag during export:

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx dynamic=True
        ```

For additional context, refer to the [dynamic input size configuration](#arguments).

### What are the key export arguments to consider for optimizing model performance?

Understanding and configuring export arguments is crucial for optimizing model performance:

- **`format:`** The target format for the exported model (e.g., `onnx`, `torchscript`, `tensorflow`).
- **`imgsz:`** Desired image size for the model input (e.g., `640` or `(height, width)`).
- **`half:`** Enables FP16 quantization, reducing model size and potentially speeding up inference.
- **`optimize:`** Applies specific optimizations for mobile or constrained environments.
- **`int8:`** Enables INT8 quantization, highly beneficial for edge deployments.

For a detailed list and explanations of all the export arguments, visit the [Export Arguments](#arguments) section.
