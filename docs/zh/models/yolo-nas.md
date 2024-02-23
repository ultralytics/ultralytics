---
comments: true
description: 探索详细的YOLO-NAS文档，这是一个更高级的物体检测模型。了解其特点、预训练模型、与Ultralytics Python API的使用等内容。
keywords: YOLO-NAS, Deci AI, 物体检测, 深度学习, 神经架构搜索, Ultralytics Python API, YOLO模型, 预训练模型, 量化, 优化, COCO, Objects365, Roboflow 100
---

# YOLO-NAS

## 概述

由Deci AI开发，YOLO-NAS是一种开创性的物体检测基础模型。它是先进的神经架构搜索技术的产物，经过精心设计以解决之前YOLO模型的局限性。YOLO-NAS在量化支持和准确性-延迟权衡方面取得了重大改进，代表了物体检测领域的重大飞跃。

![模型示例图像](https://learnopencv.com/wp-content/uploads/2023/05/yolo-nas_COCO_map_metrics.png)
**YOLO-NAS概览。** YOLO-NAS采用量化感知块和选择性量化实现最佳性能。当将模型转换为INT8量化版本时，模型会经历较小的精度损失，比其他模型有显著改进。这些先进技术使得YOLO-NAS成为具有前所未有的物体检测能力和出色性能的卓越架构。

### 主要特点

- **量化友好基本块：** YOLO-NAS引入了一种新的基本块，对量化友好，解决了之前YOLO模型的一个重要局限性。
- **高级训练和量化：** YOLO-NAS利用先进的训练方案和训练后量化以提高性能。
- **AutoNAC优化和预训练：** YOLO-NAS利用AutoNAC优化，并在著名数据集（如COCO、Objects365和Roboflow 100）上进行了预训练。这种预训练使其非常适合生产环境中的下游物体检测任务。

## 预训练模型

通过Ultralytics提供的预训练YOLO-NAS模型，体验下一代物体检测的强大功能。这些模型旨在在速度和准确性方面提供出色的性能。根据您的需求，可以选择各种选项：

| 模型               | mAP   | 延迟（ms） |
|------------------|-------|--------|
| YOLO-NAS S       | 47.5  | 3.21   |
| YOLO-NAS M       | 51.55 | 5.85   |
| YOLO-NAS L       | 52.22 | 7.87   |
| YOLO-NAS S INT-8 | 47.03 | 2.36   |
| YOLO-NAS M INT-8 | 51.0  | 3.78   |
| YOLO-NAS L INT-8 | 52.1  | 4.78   |

每个模型变体均旨在在均衡平均精度（mAP）和延迟之间提供平衡，帮助您为性能和速度都进行优化的物体检测任务。

## 用法示例

通过我们的`ultralytics` python包，Ultralytics使得将YOLO-NAS模型集成到您的Python应用程序中变得容易。该包提供了一个用户友好的Python API，以简化流程。

以下示例展示了如何使用`ultralytics`包与YOLO-NAS模型进行推理和验证：

### 推理和验证示例

这个示例中，我们在COCO8数据集上验证YOLO-NAS-s。

!!! 例子

    以下示例为YOLO-NAS提供了简单的推理和验证代码。有关处理推理结果的方法，请参见[Predict](../modes/predict.md)模式。有关使用其他模式的YOLO-NAS的方法，请参见[Val](../modes/val.md)和[Export](../modes/export.md)。`ultralytics`包中的YOLO-NAS不支持训练。

    === "Python"

        可以将预训练的PyTorch `*.pt`模型文件传递给`NAS()`类以在python中创建一个模型实例：

        ```python
        from ultralytics import NAS

        # 加载一个在COCO上预训练的YOLO-NAS-s模型
        model = NAS('yolo_nas_s.pt')

        # 显示模型信息（可选）
        model.info()

        # 在COCO8示例数据集上验证模型
        results = model.val(data='coco8.yaml')

        # 使用YOLO-NAS-s模型对'bus.jpg'图像进行推理
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        可以使用CLI命令直接运行模型：

        ```bash
        # 加载一个在COCO上预训练的YOLO-NAS-s模型，并验证其在COCO8示例数据集上的性能
        yolo val model=yolo_nas_s.pt data=coco8.yaml

        # 加载一个在COCO上预训练的YOLO-NAS-s模型，并对'bus.jpg'图像进行推理
        yolo predict model=yolo_nas_s.pt source=path/to/bus.jpg
        ```

## 支持的任务和模式

我们提供了三种类型的YOLO-NAS模型：Small (s)、Medium (m)和Large (l)。每种类型都旨在满足不同的计算和性能需求：

- **YOLO-NAS-s：** 针对计算资源有限但效率至关重要的环境进行了优化。
- **YOLO-NAS-m：** 提供平衡的方法，适用于具有更高准确性的通用物体检测。
- **YOLO-NAS-l：** 面向需要最高准确性的场景，计算资源不是限制因素。

下面是每个模型的详细信息，包括它们的预训练权重链接、支持的任务以及与不同操作模式的兼容性。

| 模型类型       | 预训练权重链接                                                                                       | 支持的任务                      | 推理 | 验证 | 训练 | 导出 |
|------------|-----------------------------------------------------------------------------------------------|----------------------------|----|----|----|----|
| YOLO-NAS-s | [yolo_nas_s.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_s.pt) | [物体检测](../tasks/detect.md) | ✅  | ✅  | ❌  | ✅  |
| YOLO-NAS-m | [yolo_nas_m.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_m.pt) | [物体检测](../tasks/detect.md) | ✅  | ✅  | ❌  | ✅  |
| YOLO-NAS-l | [yolo_nas_l.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo_nas_l.pt) | [物体检测](../tasks/detect.md) | ✅  | ✅  | ❌  | ✅  |

## 引用和致谢

如果您在研究或开发工作中使用了YOLO-NAS，请引用SuperGradients：

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{supergradients,
              doi = {10.5281/ZENODO.7789328},
              url = {https://zenodo.org/record/7789328},
              author = {Aharon,  Shay and {Louis-Dupont} and {Ofri Masad} and Yurkova,  Kate and {Lotem Fridman} and {Lkdci} and Khvedchenya,  Eugene and Rubin,  Ran and Bagrov,  Natan and Tymchenko,  Borys and Keren,  Tomer and Zhilko,  Alexander and {Eran-Deci}},
              title = {Super-Gradients},
              publisher = {GitHub},
              journal = {GitHub repository},
              year = {2021},
        }
        ```

我们向Deci AI的[SuperGradients](https://github.com/Deci-AI/super-gradients/)团队表示感谢，他们致力于创建和维护这个对计算机视觉社区非常有价值的资源。我们相信YOLO-NAS凭借其创新的架构和卓越的物体检测能力，将成为开发者和研究人员的重要工具。

*keywords: YOLO-NAS, Deci AI, 物体检测, 深度学习, 神经架构搜索, Ultralytics Python API, YOLO模型, SuperGradients, 预训练模型, 量化友好基本块, 高级训练方案, 训练后量化, AutoNAC优化, COCO, Objects365, Roboflow 100*
