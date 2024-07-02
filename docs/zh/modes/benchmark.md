---
comments: 真
description: 了解如何评估YOLOv8在各种导出格式下的速度和准确性，获取mAP50-95、accuracy_top5等指标的洞察。
keywords: Ultralytics, YOLOv8, 基准测试, 速度分析, 准确性分析, mAP50-95, accuracy_top5, ONNX, OpenVINO, TensorRT, YOLO导出格式
---

# 使用Ultralytics YOLO进行模型基准测试

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO生态系统和集成">

## 介绍

一旦您的模型经过训练和验证，下一个合乎逻辑的步骤是评估它在各种实际场景中的性能。Ultralytics YOLOv8的基准模式通过提供一个健壮的框架来评估模型在一系列导出格式中的速度和准确性，为此目的服务。

## 为什么基准测试至关重要？

- **明智的决策：** 洞察速度和准确性之间的权衡。
- **资源分配：** 理解不同的导出格式在不同硬件上的性能表现。
- **优化：** 了解哪种导出格式为您的特定用例提供最佳性能。
- **成本效益：** 根据基准测试结果，更有效地利用硬件资源。

### 基准模式的关键指标

- **mAP50-95：** 用于物体检测、分割和姿态估计。
- **accuracy_top5：** 用于图像分类。
- **推断时间：** 处理每张图片的时间（毫秒）。

### 支持的导出格式

- **ONNX：** 为了最佳的CPU性能
- **TensorRT：** 为了最大化的GPU效率
- **OpenVINO：** 针对Intel硬件的优化
- **CoreML、TensorFlow SavedModel 等：** 满足多样化部署需求。

!!! 技巧 "提示"

    * 导出到ONNX或OpenVINO可实现高达3倍CPU速度提升。
    * 导出到TensorRT可实现高达5倍GPU速度提升。

## 使用示例

在所有支持的导出格式上运行YOLOv8n基准测试，包括ONNX、TensorRT等。更多导出参数的完整列表请见下方的参数部分。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # 在GPU上进行基准测试
        benchmark(model='yolov8n.pt', data='coco8.yaml', imgsz=640, half=False, device=0)
        ```
    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

## 参数

参数如 `model`、`data`、`imgsz`、`half`、`device` 和 `verbose` 等，为用户提供了灵活性，以便根据具体需求微调基准测试，并轻松比较不同导出格式的性能。

| 键         | 值       | 描述                                                 |
|-----------|---------|----------------------------------------------------|
| `model`   | `None`  | 模型文件路径，如 yolov8n.pt, yolov8n.yaml                  |
| `data`    | `None`  | 引用基准测试数据集的YAML路径（标记为 `val`）                        |
| `imgsz`   | `640`   | 图像大小作为标量或（h, w）列表，如 (640, 480)                     |
| `half`    | `False` | FP16量化                                             |
| `int8`    | `False` | INT8量化                                             |
| `device`  | `None`  | 运行设备，如 cuda device=0 或 device=0,1,2,3 或 device=cpu |
| `verbose` | `False` | 错误时不继续（布尔值），或验证阈值下限（浮点数）                           |

## 导出格式

基准测试将尝试在下方列出的所有可能的导出格式上自动运行。

| 格式                                                                 | `format` 参数   | 模型                        | 元数据 | 参数                                                  |
|--------------------------------------------------------------------|---------------|---------------------------|-----|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -             | `yolov8n.pt`              | ✅   | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript` | `yolov8n.torchscript`     | ✅   | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`        | `yolov8n.onnx`            | ✅   | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`    | `yolov8n_openvino_model/` | ✅   | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`      | `yolov8n.engine`          | ✅   | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`      | `yolov8n.mlpackage`       | ✅   | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n_saved_model/`    | ✅   | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n.pb`              | ❌   | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n.tflite`          | ✅   | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n_edgetpu.tflite`  | ✅   | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n_web_model/`      | ✅   | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`      | `yolov8n_paddle_model/`   | ✅   | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`        | `yolov8n_ncnn_model/`     | ✅   | `imgsz`, `half`                                     |

在[导出](https://docs.ultralytics.com/modes/export/)页面查看完整的 `export` 详情。
