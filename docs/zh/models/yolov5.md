---
comments: true
description: 发现YOLOv5u，它是YOLOv5模型的改进版本，具有更好的准确性和速度之间的平衡，并为各种目标检测任务提供了许多预训练模型。
keywords: YOLOv5u、目标检测、预训练模型、Ultralytics、推断、验证、YOLOv5、YOLOv8、无锚点、无物体检测、实时应用、机器学习
---

# YOLOv5

## 概述

YOLOv5u是目标检测方法的一种进步。YOLOv5u源于Ultralytics开发的[YOLOv5](https://github.com/ultralytics/yolov5)模型的基础架构，它集成了无锚点、无物体检测分离头的新特性，这一特性在[YOLOv8](yolov8.md)模型中首次引入。通过采用这种适应性更强的检测机制，YOLOv5u改进了模型的架构，从而在目标检测任务中实现了更好的准确性和速度的平衡。根据实证结果和其衍生特性，YOLOv5u为那些在研究和实际应用中寻求强大解决方案的人提供了一种高效的选择。

![Ultralytics YOLOv5](https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png)

## 主要特性

- **无锚点分离Ultralytics头部**: 传统的目标检测模型依靠预定义的锚点框来预测目标位置，而YOLOv5u改变了这种方法。采用无锚点分离Ultralytics头部的方式，它确保了更灵活、适应性更强的检测机制，从而在各种场景中提高了性能。

- **优化的准确性和速度之间的平衡**: 速度和准确性通常是相互制约的。但是YOLOv5u挑战了这种平衡。它提供了一个校准平衡，确保在保持准确性的同时实现实时检测。这一特性对于需要快速响应的应用非常重要，比如自动驾驶车辆、机器人和实时视频分析。

- **丰富的预训练模型**: YOLOv5u提供了多种预训练模型。无论你专注于推断、验证还是训练，都有一个量身定制的模型等待着你。这种多样性确保你不仅仅使用“一刀切”的解决方案，而是使用一个专门为你的独特挑战进行了精细调整的模型。

## 支持的任务和模式

具有各种预训练权重的YOLOv5u模型在[目标检测](../tasks/detect.md)任务中表现出色。它们支持全面的模式，适用于从开发到部署的各种应用场景。

| 模型类型    | 预训练权重                                                                                                                       | 任务                         | 推断 | 验证 | 训练 | 导出 |
|---------|-----------------------------------------------------------------------------------------------------------------------------|----------------------------|----|----|----|----|
| YOLOv5u | `yolov5nu`, `yolov5su`, `yolov5mu`, `yolov5lu`, `yolov5xu`, `yolov5n6u`, `yolov5s6u`, `yolov5m6u`, `yolov5l6u`, `yolov5x6u` | [目标检测](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅  |

该表详细介绍了YOLOv5u模型的变体，突出了它们在目标检测任务和各种操作模式（如[推断](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)）方面的适用性。这种全面的支持确保用户可以充分发挥YOLOv5u模型在各种目标检测场景中的能力。

## 性能指标

!!! Performance

    === "检测"

    请参阅[检测文档](https://docs.ultralytics.com/tasks/detect/)，以了解在[COCO](https://docs.ultralytics.com/datasets/detect/coco/)上训练的这些模型的用法示例，其中包括80个预训练类别。

    | 模型                                                                                       | YAML                                                                                                           | 大小<br><sup>（像素） | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>（毫秒） | 速度<br><sup>A100 TensorRT<br>（毫秒） | 参数数<br><sup>（百万） | FLOPs<br><sup>（十亿） |
    |---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------|----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
    | [yolov5nu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5nu.pt)   | [yolov5n.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 34.3                 | 73.6                           | 1.06                                | 2.6                | 7.7               |
    | [yolov5su.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5su.pt)   | [yolov5s.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 43.0                 | 120.7                          | 1.27                                | 9.1                | 24.0              |
    | [yolov5mu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5mu.pt)   | [yolov5m.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 49.0                 | 233.9                          | 1.86                                | 25.1               | 64.2              |
    | [yolov5lu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5lu.pt)   | [yolov5l.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 52.2                 | 408.4                          | 2.50                                | 53.2               | 135.0             |
    | [yolov5xu.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5xu.pt)   | [yolov5x.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5.yaml)     | 640                   | 53.2                 | 763.2                          | 3.81                                | 97.2               | 246.4             |
    |                                                                                             |                                                                                                                |                       |                      |                                |                                     |                    |                   |
    | [yolov5n6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n6u.pt) | [yolov5n6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 42.1                 | 211.0                          | 1.83                                | 4.3                | 7.8               |
    | [yolov5s6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5s6u.pt) | [yolov5s6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 48.6                 | 422.6                          | 2.34                                | 15.3               | 24.6              |
    | [yolov5m6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5m6u.pt) | [yolov5m6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 53.6                 | 810.9                          | 4.36                                | 41.2               | 65.7              |
    | [yolov5l6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5l6u.pt) | [yolov5l6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 55.7                 | 1470.9                         | 5.47                                | 86.1               | 137.4             |
    | [yolov5x6u.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5x6u.pt) | [yolov5x6.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v5/yolov5-p6.yaml) | 1280                  | 56.8                 | 2436.5                         | 8.98                                | 155.4              | 250.7             |

## 使用示例

这个示例提供了YOLOv5训练和推断的简单示例。有关这些和其他[模式](../modes/index.md)的完整文档，请参阅[预测](../modes/predict.md)、[训练](../modes/train.md)、[验证](../modes/val.md)和[导出](../modes/export.md)的文档页面。

!!! Example "示例"

    === "Python"

        PyTorch预训练的`*.pt`模型，以及配置`*.yaml`文件可以传递给`YOLO()`类，以在python中创建一个模型实例：

        ```python
        from ultralytics import YOLO

        # 加载一个在COCO数据集上预训练的YOLOv5n模型
        model = YOLO('yolov5n.pt')

        # 显示模型信息（可选）
        model.info()

        # 使用COCO8示例数据集对模型进行100个时期的训练
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 使用YOLOv5n模型对'bus.jpg'图像进行推断
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        可以使用CLI命令直接运行模型：

        ```bash
        # 加载一个在COCO数据集上预训练的YOLOv5n模型，并在COCO8示例数据集上进行100个时期的训练
        yolo train model=yolov5n.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载一个在COCO数据集上预训练的YOLOv5n模型，并在'bus.jpg'图像上进行推断
        yolo predict model=yolov5n.pt source=path/to/bus.jpg
        ```

## 引用和致谢

如果您在您的研究中使用了YOLOv5或YOLOv5u，请引用Ultralytics的YOLOv5存储库，引用方式如下：

!!! Quote ""

    === "BibTeX"
        ```bibtex
        @software{yolov5,
          title = {Ultralytics YOLOv5},
          author = {Glenn Jocher},
          year = {2020},
          version = {7.0},
          license = {AGPL-3.0},
          url = {https://github.com/ultralytics/yolov5},
          doi = {10.5281/zenodo.3908559},
          orcid = {0000-0001-5950-6979}
        }
        ```

请注意，YOLOv5模型提供[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)和[企业](https://ultralytics.com/license)许可证。
