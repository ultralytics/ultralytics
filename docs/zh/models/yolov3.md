---
comments: true
description: 了解YOLOv3、YOLOv3-Ultralytics和YOLOv3u的概述。了解它们的关键功能、用途和支持的目标检测任务。
keywords: YOLOv3、YOLOv3-Ultralytics、YOLOv3u、目标检测、推理、训练、Ultralytics
---

# YOLOv3、YOLOv3-Ultralytics和YOLOv3u

## 概述

本文介绍了三个紧密相关的目标检测模型，分别是[YOLOv3](https://pjreddie.com/darknet/yolo/)、[YOLOv3-Ultralytics](https://github.com/ultralytics/yolov3)和[YOLOv3u](https://github.com/ultralytics/ultralytics)。

1. **YOLOv3：** 这是第三版 You Only Look Once (YOLO) 目标检测算法。YOLOv3 在前作的基础上进行了改进，引入了多尺度预测和三种不同尺寸的检测核，提高了检测准确性。

2. **YOLOv3-Ultralytics：** 这是 Ultralytics 对 YOLOv3 模型的实现。它在复现了原始 YOLOv3 架构的基础上，提供了额外的功能，如对更多预训练模型的支持和更简单的定制选项。

3. **YOLOv3u：** 这是 YOLOv3-Ultralytics 的更新版本，它引入了 YOLOv8 模型中使用的无锚、无物体性能分离头。YOLOv3u 保留了 YOLOv3 的主干和颈部架构，但使用了来自 YOLOv8 的更新检测头。

![Ultralytics YOLOv3](https://raw.githubusercontent.com/ultralytics/assets/main/yolov3/banner-yolov3.png)

## 关键功能

- **YOLOv3：** 引入了三种不同尺度的检测，采用了三种不同尺寸的检测核：13x13、26x26 和 52x52。这显著提高了对不同大小对象的检测准确性。此外，YOLOv3 还为每个边界框添加了多标签预测和更好的特征提取网络。

- **YOLOv3-Ultralytics：** Ultralytics 对 YOLOv3 的实现具有与原始模型相同的性能，但增加了对更多预训练模型、额外训练方法和更简单的定制选项的支持。这使得它在实际应用中更加通用和易用。

- **YOLOv3u：** 这个更新的模型采用了来自 YOLOv8 的无锚、无物体性能分离头。通过消除预定义的锚框和物体性能分数的需求，检测头设计可以提高模型对不同大小和形状的对象的检测能力。这使得 YOLOv3u 在目标检测任务中更加强大和准确。

## 支持的任务和模式

YOLOv3 系列，包括 YOLOv3、YOLOv3-Ultralytics 和 YOLOv3u，专门用于目标检测任务。这些模型以在各种实际场景中平衡准确性和速度而闻名。每个变体都提供了独特的功能和优化，使其适用于各种应用场景。

这三个模型都支持一套全面的模式，确保在模型部署和开发的各个阶段具备多种功能。这些模式包括[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)，为用户提供了有效的目标检测完整工具。

| 模型类型           | 支持的任务                     | 推理 | 验证 | 训练 | 导出 |
| ------------------ | ------------------------------ | ---- | ---- | ---- | ---- |
| YOLOv3             | [目标检测](../tasks/detect.md) | ✅   | ✅   | ✅   | ✅   |
| YOLOv3-Ultralytics | [目标检测](../tasks/detect.md) | ✅   | ✅   | ✅   | ✅   |
| YOLOv3u            | [目标检测](../tasks/detect.md) | ✅   | ✅   | ✅   | ✅   |

该表格提供了每个 YOLOv3 变体的能力一览，突显了它们的多功能性和适用性，以用于目标检测工作流程中的各种任务和操作模式。

## 用法示例

以下示例提供了简单的 YOLOv3 训练和推理示例。有关这些和其他模式的完整文档，请参阅 [Predict](../modes/predict.md)、[Train](../modes/train.md)、[Val](../modes/val.md) 和 [Export](../modes/export.md) 文档页面。

!!! Example "示例"

    === "Python"

        可以将预先训练的 PyTorch `*.pt` 模型以及配置 `*.yaml` 文件传递给 `YOLO()` 类，以在 Python 中创建模型实例：

        ```python
        from ultralytics import YOLO

        # 加载一个经过 COCO 预训练的 YOLOv3n 模型
        model = YOLO("yolov3n.pt")

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上训练模型100个epoch
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 YOLOv3n 模型对 'bus.jpg' 图像进行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        可以直接使用命令行界面 (CLI) 来运行模型：

        ```bash
        # 加载一个经过 COCO 预训练的 YOLOv3n 模型，并在 COCO8 示例数据集上训练100个epoch
        yolo train model=yolov3n.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载一个经过 COCO 预训练的 YOLOv3n 模型，并对 'bus.jpg' 图像进行推理
        yolo predict model=yolov3n.pt source=path/to/bus.jpg
        ```

## 引用和致谢

如果您在研究中使用 YOLOv3，请引用原始的 YOLO 论文和 Ultralytics 的 YOLOv3 仓库：

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{redmon2018yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal={arXiv preprint arXiv:1804.02767},
          year={2018}
        }
        ```

感谢 Joseph Redmon 和 Ali Farhadi 开发了原始的 YOLOv3 模型。
