---
comments: true
description: 指南 - 验证 YOLOv8 模型。了解如何使用验证设置和指标评估您的 YOLO 模型的性能，包括 Python 和 CLI 示例。
keywords: Ultralytics, YOLO 文档, YOLOv8, 验证, 模型评估, 超参数, 准确率, 指标, Python, CLI
---

# 使用 Ultralytics YOLO 进行模型验证

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO 生态系统和集成">

## 引言

在机器学习流程中，验证是一个关键步骤，让您能够评估训练模型的质量。Ultralytics YOLOv8 的 Val 模式提供了一整套强大的工具和指标，用于评估您的目标检测模型的性能。本指南作为一个完整资源，用于理解如何有效使用 Val 模式来确保您的模型既准确又可靠。

## 为什么要使用 Ultralytics YOLO 进行验证？

以下是使用 YOLOv8 的 Val 模式的好处：

- **精确性：** 获取准确的指标，如 mAP50、mAP75 和 mAP50-95，全面评估您的模型。
- **便利性：** 利用内置功能记住训练设置，简化验证过程。
- **灵活性：** 使用相同或不同的数据集和图像尺寸验证您的模型。
- **超参数调优：** 使用验证指标来调整您的模型以获得更好的性能。

### Val 模式的主要特点

以下是 YOLOv8 的 Val 模式提供的显著功能：

- **自动化设置：** 模型记住其训练配置，以便直接进行验证。
- **多指标支持：** 根据一系列准确度指标评估您的模型。
- **CLI 和 Python API：** 根据您的验证偏好选择命令行界面或 Python API。
- **数据兼容性：** 与训练阶段使用的数据集以及自定义数据集无缝协作。

!!! Tip "提示"

    * YOLOv8 模型会自动记住其训练设置，因此您可以很容易地仅使用 `yolo val model=yolov8n.pt` 或 `model('yolov8n.pt').val()` 在原始数据集上并以相同图像大小验证模型。

## 使用示例

在 COCO128 数据集上验证训练过的 YOLOv8n 模型的准确性。由于 `model` 保留了其训练的 `data` 和参数作为模型属性，因此无需传递任何参数。有关完整的导出参数列表，请参阅下面的参数部分。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO("yolov8n.pt")  # 加载官方模型
        model = YOLO("path/to/best.pt")  # 加载自定义模型

        # 验证模型
        metrics = model.val()  # 无需参数，数据集和设置记忆
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # 包含每个类别的map50-95列表
        ```
    === "CLI"

        ```bash
        yolo detect val model=yolov8n.pt  # 验证官方模型
        yolo detect val model=path/to/best.pt  # 验证自定义模型
        ```

## 参数

YOLO 模型的验证设置是指用于评估模型在验证数据集上性能的各种超参数和配置。这些设置会影响模型的性能、速度和准确性。一些常见的 YOLO 验证设置包括批处理大小、在训练期间验证频率以及用于评估模型性能的指标。其他可能影响验证过程的因素包括验证数据集的大小和组成以及模型用于特定任务的特性。仔细调整和实验这些设置很重要，以确保模型在验证数据集上表现良好并且检测和预防过拟合。

| 键            | 值      | 描述                                                   |
| ------------- | ------- | ------------------------------------------------------ |
| `data`        | `None`  | 数据文件的路径，例如 coco128.yaml                      |
| `imgsz`       | `640`   | 输入图像的大小，以整数表示                             |
| `batch`       | `16`    | 每批图像的数量（AutoBatch 为 -1）                      |
| `save_json`   | `False` | 将结果保存至 JSON 文件                                 |
| `save_hybrid` | `False` | 保存混合版本的标签（标签 + 额外预测）                  |
| `conf`        | `0.001` | 用于检测的对象置信度阈值                               |
| `iou`         | `0.6`   | NMS（非极大抑制）用的交并比（IoU）阈值                 |
| `max_det`     | `300`   | 每张图像的最大检测数量                                 |
| `half`        | `True`  | 使用半精度（FP16）                                     |
| `device`      | `None`  | 运行所用的设备，例如 cuda device=0/1/2/3 或 device=cpu |
| `dnn`         | `False` | 使用 OpenCV DNN 进行 ONNX 推理                         |
| `plots`       | `False` | 在训练期间显示图表                                     |
| `rect`        | `False` | 矩形验证，每批图像为了最小填充整齐排列                 |
| `split`       | `val`   | 用于验证的数据集分割，例如 'val'、'test' 或 'train'    |

|
