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
  <iframe width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
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

!!! tip "Tip"

    * Export to ONNX or OpenVINO for up to 3x CPU speedup.
    * Export to TensorRT for up to 5x GPU speedup.

## Usage Examples

Export a YOLOv8n model to a different format like ONNX or TensorRT. See Arguments section below for a full list of export arguments.

!!! example ""

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

Export settings for YOLO models refer to the various configurations and options used to save or export the model for use in other environments or platforms. These settings can affect the model's performance, size, and compatibility with different systems. Some common YOLO export settings include the format of the exported model file (e.g. ONNX, TensorFlow SavedModel), the device on which the model will be run (e.g. CPU, GPU), and the presence of additional features such as masks or multiple labels per box. Other factors that may affect the export process include the specific task the model is being used for and the requirements or constraints of the target environment or platform. It is important to carefully consider and configure these settings to ensure that the exported model is optimized for the intended use case and can be used effectively in the target environment.

| Key         | Value           | Description                                          |
|-------------|-----------------|------------------------------------------------------|
| `format`    | `'torchscript'` | format to export to                                  |
| `imgsz`     | `640`           | image size as scalar or (h, w) list, i.e. (640, 480) |
| `keras`     | `False`         | use Keras for TF SavedModel export                   |
| `optimize`  | `False`         | TorchScript: optimize for mobile                     |
| `half`      | `False`         | FP16 quantization                                    |
| `int8`      | `False`         | INT8 quantization                                    |
| `dynamic`   | `False`         | ONNX/TensorRT: dynamic axes                          |
| `simplify`  | `False`         | ONNX/TensorRT: simplify model                        |
| `opset`     | `None`          | ONNX: opset version (optional, defaults to latest)   |
| `workspace` | `4`             | TensorRT: workspace size (GB)                        |
| `nms`       | `False`         | CoreML: add NMS                                      |

## Export Formats

Available YOLOv8 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`.

| Format                      | `format=`Argument | Model                     | Metadata           | Available Modifiers                                 |
|-----------------------------|-------------------|:--------------------------|:------------------:|-----------------------------------------------------|
| [PyTorch][pytorch]          | -                 | `yolov8n.pt`              | :white_check_mark: | -                                                   |
| [TorchScript][torchscript]  | `torchscript`     | `yolov8n.torchscript`     | :white_check_mark: | `imgsz`, `optimize`                                 |
| [ONNX][onnx]                | `onnx`            | `yolov8n.onnx`            | :white_check_mark: | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO][openvino]        | `openvino`        | `yolov8n_openvino_model/` | :white_check_mark: | `imgsz`, `half`                                     |
| [TensorRT][tensorrt]        | `engine`          | `yolov8n.engine`          | :white_check_mark: | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML][coreml]            | `coreml`          | `yolov8n.mlpackage`       | :white_check_mark: | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel][tf_saved]   | `saved_model`     | `yolov8n_saved_model/`    | :white_check_mark: | `imgsz`, `keras`                                    |
| [TF GraphDef][tf_graph]     | `pb`              | `yolov8n.pb`              | :x:                | `imgsz`                                             |
| [TF Lite][tf_lite]          | `tflite`          | `yolov8n.tflite`          | :white_check_mark: | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU][tf_edge_tpu]  | `edgetpu`         | `yolov8n_edgetpu.tflite`  | :white_check_mark: | `imgsz`                                             |
| [TF.js][tf_js]              | `tfjs`            | `yolov8n_web_model/`      | :white_check_mark: | `imgsz`                                             |
| [PaddlePaddle][paddle]      | `paddle`          | `yolov8n_paddle_model/`   | :white_check_mark: | `imgsz`                                             |
| [ncnn][ncnn]                | `ncnn`            | `yolov8n_ncnn_model/`     | :white_check_mark: | `imgsz`, `half`                                     |


[pytorch]:     https://pytorch.org/
[torchscript]: https://pytorch.org/docs/stable/jit.html
[onnx]:        https://onnx.ai/
[openvino]:    https://docs.openvino.ai/latest/index.html
[tensorrt]:    https://developer.nvidia.com/tensorrt
[coreml]:      https://github.com/apple/coremltools
[tf_saved]:    https://www.tensorflow.org/guide/saved_model
[tf_graph]:    https://www.tensorflow.org/api_docs/python/tf/Graph
[tf_lite]:     https://www.tensorflow.org/lite
[tf_edge_tpu]: https://coral.ai/docs/edgetpu/models-intro/
[tf_js]:       https://www.tensorflow.org/js
[paddle]:      https://github.com/PaddlePaddle
[ncnn]:        https://github.com/Tencent/ncnn