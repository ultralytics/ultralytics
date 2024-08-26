---
comments: true
description: 探索YOLOv8的激动人心功能，这是我们实时目标检测器的最新版本！了解高级架构、预训练模型和精确度与速度的最佳平衡如何使YOLOv8成为您进行目标检测任务的理想选择。
keywords: YOLOv8，Ultralytics，实时目标检测器，预训练模型，文档，目标检测，YOLO系列，高级架构，精确度，速度
---

# YOLOv8

## 概述

YOLOv8是YOLO系列实时目标检测器的最新版本，以其在准确度和速度方面的卓越性能而闻名。在构建在之前YOLO版本的基础上，YOLOv8引入了新功能和优化，使其成为各种应用领域中各种目标检测任务的理想选择。

![Ultralytics YOLOv8](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## 主要功能

- **先进的主干和中间架构：** YOLOv8采用最先进的主干和中间架构，提供了更好的特征提取和目标检测性能。
- **无锚分割Ultralytics头：** YOLOv8采用无锚分割的Ultralytics头，相比于基于锚点的方法，可以提供更高的准确性和更高效的检测过程。
- **优化的准确度和速度平衡：** YOLOv8专注于在准确度和速度之间维持最佳平衡，适用于各种实时目标检测任务。
- **多种预训练模型：** YOLOv8提供了一系列预训练模型，以满足各种任务和性能要求，更容易找到适合特定用例的模型。

## 支持的任务和模式

YOLOv8系列提供了多种模型，每个模型专门用于计算机视觉中的特定任务。这些模型旨在满足各种要求，从目标检测到更复杂的任务，如实例分割、姿态/关键点检测和分类。

YOLOv8系列的每个变体都针对其相应的任务进行了优化，确保高性能和准确性。此外，这些模型与各种操作模式兼容，包括[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)，便于在部署和开发的不同阶段使用。

| 模型        | 文件名                                                                                                         | 任务                            | 推理 | 验证 | 训练 | 导出 |
| ----------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------- | ---- | ---- | ---- | ---- |
| YOLOv8      | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt`                                               | [检测](../tasks/detect.md)      | ✅   | ✅   | ✅   | ✅   |
| YOLOv8-seg  | `yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt`                           | [实例分割](../tasks/segment.md) | ✅   | ✅   | ✅   | ✅   |
| YOLOv8-pose | `yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` | [姿态/关键点](../tasks/pose.md) | ✅   | ✅   | ✅   | ✅   |
| YOLOv8-cls  | `yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt`                           | [分类](../tasks/classify.md)    | ✅   | ✅   | ✅   | ✅   |

这个表格提供了YOLOv8模型变种的概览，突出了它们在特定任务中的适用性，以及它们与各种操作模式（如推理、验证、训练和导出）的兼容性。它展示了YOLOv8系列的多功能性和鲁棒性，使它们适用于计算机视觉中各种应用。

## 性能指标

!!! Performance

    === "检测（COCO）"

        有关在[COCO](https://docs.ultralytics.com/datasets/detect/coco/)上训练的这些模型的用法示例，请参见[Detection Docs](https://docs.ultralytics.com/tasks/detect/)，其中包括80个预训练的类别。

        | 模型                                                                                   | 大小<br><sup>(pixels) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
        | ---------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------------------- | ---------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

    === "检测（Open Images V7）"

        有关在[Open Image V7](https://docs.ultralytics.com/datasets/detect/open-images-v7/)上训练的这些模型的用法示例，请参见[Detection Docs](https://docs.ultralytics.com/tasks/detect/)，其中包括600个预训练的类别。

        | 模型                                                                                        | 大小<br><sup>(pixels) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
        | ------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ------------------------------ | ---------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-oiv7.pt) | 640                   | 18.4                 | 142.4                          | 1.21                                | 3.5                | 10.5              |
        | [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-oiv7.pt) | 640                   | 27.7                 | 183.1                          | 1.40                                | 11.4               | 29.7              |
        | [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-oiv7.pt) | 640                   | 33.6                 | 408.5                          | 2.26                                | 26.2               | 80.6              |
        | [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-oiv7.pt) | 640                   | 34.9                 | 596.9                          | 2.43                                | 44.1               | 167.4             |
        | [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-oiv7.pt) | 640                   | 36.3                 | 860.6                          | 3.56                                | 68.7               | 260.6             |

    === "分割（COCO）"

        有关在[COCO](https://docs.ultralytics.com/datasets/segment/coco/)上训练的这些模型的用法示例，请参见[Segmentation Docs](https://docs.ultralytics.com/tasks/segment/)，其中包括80个预训练的类别。

        | 模型                                                                                         | 大小<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
        | -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ---------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
        | [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
        | [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
        | [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
        | [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |

    === "分类（ImageNet）"

        有关在[ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/)上训练的这些模型的用法示例，请参见[Classification Docs](https://docs.ultralytics.com/tasks/classify/)，其中包括1000个预训练的类别。

        | 模型                                                                                           | 大小<br><sup>(pixels) | 准确率<br><sup>top1 | 准确率<br><sup>top5 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
        | ---------------------------------------------------------------------------------------------- | --------------------- | ------------------ | ------------------ | ------------------------------ | ---------------------------------- | ------------------ | ------------------------ |
        | [YOLOv8n-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-cls.pt) | 224                   | 66.6               | 87.0               | 12.9                           | 0.31                                | 2.7                | 4.3                      |
        | [YOLOv8s-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-cls.pt) | 224                   | 72.3               | 91.1               | 23.4                           | 0.35                                | 6.4                | 13.5                     |
        | [YOLOv8m-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-cls.pt) | 224                   | 76.4               | 93.2               | 85.4                           | 0.62                                | 17.0               | 42.7                     |
        | [YOLOv8l-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-cls.pt) | 224                   | 78.0               | 94.1               | 163.0                          | 0.87                                | 37.5               | 99.7                     |
        | [YOLOv8x-cls](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-cls.pt) | 224                   | 78.4               | 94.3               | 232.0                          | 1.01                                | 57.4               | 154.8                    |

    === "姿态（COCO）"

        有关在[COCO](https://docs.ultralytics.com/datasets/pose/coco/)上训练的这些模型的用法示例，请参见[Pose Estimation Docs](https://docs.ultralytics.com/tasks/segment/)，其中包括1个预训练的类别，'person'。

        | 模型                                                                                                | 大小<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数<br><sup>(M) | FLOPs<br><sup>(B) |
        | ---------------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ---------------------------------- | ------------------ | ----------------- |
        | [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt)       | 640                   | 50.4                  | 80.1               | 131.8                          | 1.18                                | 3.3                | 9.2               |
        | [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt)       | 640                   | 60.0                  | 86.2               | 233.2                          | 1.42                                | 11.6               | 30.2              |
        | [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-pose.pt)       | 640                   | 65.0                  | 88.8               | 456.3                          | 2.00                                | 26.4               | 81.0              |
        | [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-pose.pt)       | 640                   | 67.6                  | 90.0               | 784.5                          | 2.59                                | 44.4               | 168.6             |
        | [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose.pt)       | 640                   | 69.2                  | 90.2               | 1607.1                         | 3.73                                | 69.4               | 263.2             |
        | [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-pose-p6.pt) | 1280                  | 71.6                  | 91.2               | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

## 用法示例

这个示例提供了关于YOLOv8训练和推理的简单示例。有关这些和其他[模式](../modes/index.md)的完整文档，请参见[Predict](../modes/predict.md)，[Train](../modes/train.md)，[Val](../modes/val.md)和[Export](../modes/export.md)文档页面。

请注意，以下示例是针对用于目标检测的YOLOv8 [Detect](../tasks/detect.md)模型。有关其他支持的任务，请参见[Segment](../tasks/segment.md)、[Classify](../tasks/classify.md)和[Pose](../tasks/pose.md)文档。

!!! Example "示例"

    === "Python"

        可以将PyTorch预训练的`*.pt`模型和配置`*.yaml`文件传递给`YOLO()`类，在python中创建一个模型实例：

        ```python
        from ultralytics import YOLO

        # 加载一个在COCO预训练的YOLOv8n模型
        model = YOLO("yolov8n.pt")

        # 显示模型信息（可选）
        model.info()

        # 使用COCO8示例数据集训练模型100个epoch
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用YOLOv8n模型在'bus.jpg'图片上运行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        可以使用CLI命令直接运行模型：

        ```bash
        # 加载一个在COCO预训练的YOLOv8n模型，并在COCO8示例数据集上训练100个epoch
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载一个在COCO预训练的YOLOv8n模型，并在'bus.jpg'图片上运行推理
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 引用和致谢

如果您在工作中使用YOLOv8模型或此存储库中的其他软件，请使用以下格式进行引用：

!!! Quote "引用"

    === "BibTeX"

        ```bibtex
        @software{yolov8_ultralytics,
          author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
          title = {Ultralytics YOLOv8},
          version = {8.0.0},
          year = {2023},
          url = {https://github.com/ultralytics/ultralytics},
          orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
          license = {AGPL-3.0}
        }
        ```

请注意，DOI正在等待中，DOI将在可用时添加到引用中。YOLOv8模型根据[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)和[企业许可证](https://ultralytics.com/license)提供。
