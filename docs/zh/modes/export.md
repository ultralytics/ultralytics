---
comments: true
description: 如何逐步指导您将 YOLOv8 模型导出到各种格式，如 ONNX、TensorRT、CoreML 等以进行部署。现在就探索！
keywords: YOLO, YOLOv8, Ultralytics, 模型导出, ONNX, TensorRT, CoreML, TensorFlow SavedModel, OpenVINO, PyTorch, 导出模型
---

# Ultralytics YOLO 的模型导出

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO 生态系统和集成">

## 引言

训练模型的最终目标是将其部署到现实世界的应用中。Ultralytics YOLOv8 的导出模式提供了多种选项，用于将您训练好的模型导出到不同的格式，从而可以在各种平台和设备上部署。本综合指南旨在带您逐步了解模型导出的细节，展示如何实现最大的兼容性和性能。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/WbomGeoOT_k?si=aGmuyooWftA0ue9X"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>如何导出自定义训练的 Ultralytics YOLOv8 模型并在网络摄像头上实时推理。
</p>

## 为什么选择 YOLOv8 的导出模式？

- **多功能性：** 支持导出到多种格式，包括 ONNX、TensorRT、CoreML 等。
- **性能：** 使用 TensorRT 可实现高达 5 倍 GPU 加速，使用 ONNX 或 OpenVINO 可实现高达 3 倍 CPU 加速。
- **兼容性：** 使您的模型可以在众多硬件和软件环境中广泛部署。
- **易用性：** 简单的 CLI 和 Python API，快速直接地进行模型导出。

### 导出模式的关键特性

以下是一些突出的功能：

- **一键导出：** 用于导出到不同格式的简单命令。
- **批量导出：** 支持批推理能力的模型导出。
- **优化推理：** 导出的模型针对更快的推理时间进行优化。
- **教学视频：** 提供深入指导和教学，确保流畅的导出体验。

!!! Tip "提示"

    * 导出到 ONNX 或 OpenVINO，以实现高达 3 倍的 CPU 加速。
    * 导出到 TensorRT，以实现高达 5 倍的 GPU 加速。

## 使用示例

将 YOLOv8n 模型导出为 ONNX 或 TensorRT 等不同格式。查看下面的参数部分，了解完整的导出参数列表。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolov8n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义训练的模型

        # 导出模型
        model.export(format="onnx")
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx  # 导出官方模型
        yolo export model=path/to/best.pt format=onnx  # 导出自定义训练的模型
        ```

## 参数

YOLO 模型的导出设置是指用于在其他环境或平台中使用模型时保存或导出模型的各种配置和选项。这些设置会影响模型的性能、大小和与不同系统的兼容性。一些常见的 YOLO 导出设置包括导出的模型文件格式（例如 ONNX、TensorFlow SavedModel）、模型将在哪个设备上运行（例如 CPU、GPU）以及是否包含附加功能，如遮罩或每个框多个标签。其他可能影响导出过程的因素包括模型用途的具体细节以及目标环境或平台的要求或限制。重要的是要仔细考虑和配置这些设置，以确保导出的模型针对预期用例经过优化，并且可以在目标环境中有效使用。

| 键          | 值              | 描述                                                |
| ----------- | --------------- | --------------------------------------------------- |
| `format`    | `'torchscript'` | 导出的格式                                          |
| `imgsz`     | `640`           | 图像尺寸，可以是标量或 (h, w) 列表，比如 (640, 480) |
| `keras`     | `False`         | 使用 Keras 导出 TF SavedModel                       |
| `optimize`  | `False`         | TorchScript：为移动设备优化                         |
| `half`      | `False`         | FP16 量化                                           |
| `int8`      | `False`         | INT8 量化                                           |
| `dynamic`   | `False`         | ONNX/TensorRT：动态轴                               |
| `simplify`  | `False`         | ONNX/TensorRT：简化模型                             |
| `opset`     | `None`          | ONNX：opset 版本（可选，默认为最新版本）            |
| `workspace` | `4`             | TensorRT：工作区大小（GB）                          |
| `nms`       | `False`         | CoreML：添加 NMS                                    |

## 导出格式

下表中提供了可用的 YOLOv8 导出格式。您可以使用 `format` 参数导出任何格式的模型，比如 `format='onnx'` 或 `format='engine'`。

| 格式                                                               | `format` 参数 | 模型                      | 元数据 | 参数                                                |
| ------------------------------------------------------------------ | ------------- | ------------------------- | ------ | --------------------------------------------------- |
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n.pt`              | ✅     | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n.torchscript`     | ✅     | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n.onnx`            | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n_openvino_model/` | ✅     | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n.engine`          | ✅     | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n.mlpackage`       | ✅     | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n_saved_model/`    | ✅     | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n.pb`              | ❌     | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n.tflite`          | ✅     | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n_edgetpu.tflite`  | ✅     | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n_web_model/`      | ✅     | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n_paddle_model/`   | ✅     | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n_ncnn_model/`     | ✅     | `imgsz`, `half`                                     |
