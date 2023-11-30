---
comments: true
description: 探索美团YOLOv6，一种在速度和准确性之间取得平衡的最先进的物体检测模型。深入了解功能、预训练模型和Python使用方法。
keywords: 美团YOLOv6、物体检测、Ultralytics、YOLOv6文档、双向连接、锚辅助训练、预训练模型、实时应用
---

# 美团YOLOv6

## 概述

[美团](https://about.meituan.com/) YOLOv6是一种最先进的物体检测器，速度和准确性兼具，成为实时应用的热门选择。该模型在架构和训练方案上引入了几项重要改进，包括双向连接模块（BiC）、锚辅助训练（AAT）策略以及改进了的主干和颈部设计，使其在COCO数据集上达到了最先进的准确性。

![美团YOLOv6](https://user-images.githubusercontent.com/26833433/240750495-4da954ce-8b3b-41c4-8afd-ddb74361d3c2.png)
![模型示例图片](https://user-images.githubusercontent.com/26833433/240750557-3e9ec4f0-0598-49a8-83ea-f33c91eb6d68.png)
**YOLOv6概述。** 模型架构图显示了经过重新设计的网络组件和训练策略，这些策略导致了显著的性能提升。（a）YOLOv6的颈部（N和S）。（b）BiC模块的结构。（c）SimCSPSPPF块。([来源](https://arxiv.org/pdf/2301.05586.pdf)).

### 主要功能

- **双向连接（BiC）模块：** YOLOv6在检测器的颈部引入了双向连接（BiC）模块，增强了定位信号，提供了性能增益，并且几乎没有降低速度。
- **锚辅助训练（AAT）策略：** 该模型提出了锚辅助训练（AAT）以享受基于锚点和无锚点范例的双重优势，同时不影响推理效率。
- **增强的主干和颈部设计：** 通过在主干和颈部中增加一个阶段，该模型在高分辨率输入下在COCO数据集上实现了最先进的性能。
- **自我蒸馏策略：** 实施了一种新的自我蒸馏策略，以提升YOLOv6的较小模型的性能，在训练过程中增强辅助回归分支，并在推理过程中将其删除，以避免明显的速度下降。

## 性能指标

YOLOv6提供了具有不同尺度的各种预训练模型：

- YOLOv6-N：在NVIDIA Tesla T4 GPU上，COCO val2017上的AP为37.5%，帧率为1187 FPS。
- YOLOv6-S：AP为45.0%，帧率为484 FPS。
- YOLOv6-M：AP为50.0%，帧率为226 FPS。
- YOLOv6-L：AP为52.8%，帧率为116 FPS。
- YOLOv6-L6：实时场景中的最先进准确性。

YOLOv6还提供了适用于不同精度和移动平台的量化模型。

## 使用示例

以下示例提供了简单的YOLOv6训练和推理示例。有关这些示例和其他[模式](../modes/index.md)的完整文档，请参阅[Predict](../modes/predict.md)、[Train](../modes/train.md)、[Val](../modes/val.md)和[Export](../modes/export.md)的文档页面。

!!! 例子

    === "Python"

        在Python中，可以将PyTorch预训练的`*.pt`模型以及配置文件`*.yaml`传递给`YOLO()`类，以创建一个模型实例：

        ```python
        from ultralytics import YOLO

        # 从头开始构建一个YOLOv6n模型
        model = YOLO('yolov6n.yaml')

        # 显示模型信息（可选）
        model.info()

        # 使用COCO8示例数据集对模型进行100个epoch的训练
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 使用YOLOv6n模型对'bus.jpg'图像进行推理
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        也可以使用CLI命令直接运行模型：

        ```bash
        # 从头开始构建YOLOv6n模型，并在COCO8示例数据集上进行100个epoch的训练
        yolo train model=yolov6n.yaml data=coco8.yaml epochs=100 imgsz=640

        # 从头开始构建YOLOv6n模型，并对'bus.jpg'图像进行推理
        yolo predict model=yolov6n.yaml source=path/to/bus.jpg
        ```

## 支持的任务和模式

YOLOv6系列提供了一系列模型，每个模型都针对高性能[物体检测](../tasks/detect.md)进行了优化。这些模型适用于各种计算需求和准确性要求，使其在广泛的应用中具备多样性。

| 模型类型      | 预训练权重          | 支持的任务                      | 推理 | 验证 | 训练 | 导出 |
|-----------|----------------|----------------------------|----|----|----|----|
| YOLOv6-N  | `yolov6-n.pt`  | [物体检测](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅  |
| YOLOv6-S  | `yolov6-s.pt`  | [物体检测](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅  |
| YOLOv6-M  | `yolov6-m.pt`  | [物体检测](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅  |
| YOLOv6-L  | `yolov6-l.pt`  | [物体检测](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅  |
| YOLOv6-L6 | `yolov6-l6.pt` | [物体检测](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅  |

这个表格详细介绍了YOLOv6模型的各个变体，突出了它们在物体检测任务中的能力以及它们与各种操作模式（如[推理](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)）的兼容性。这种全面的支持确保用户可以在各种物体检测场景中充分利用YOLOv6模型的能力。

## 引用和致谢

我们要感谢这些作者在实时物体检测领域的重要贡献：

!!! 引文 ""

    === "BibTeX"

        ```bibtex
        @misc{li2023yolov6,
              title={YOLOv6 v3.0: A Full-Scale Reloading},
              author={Chuyi Li and Lulu Li and Yifei Geng and Hongliang Jiang and Meng Cheng and Bo Zhang and Zaidan Ke and Xiaoming Xu and Xiangxiang Chu},
              year={2023},
              eprint={2301.05586},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

    原始的YOLOv6论文可以在[arXiv](https://arxiv.org/abs/2301.05586)上找到。作者已经将他们的作品公开，并且代码可以在[GitHub](https://github.com/meituan/YOLOv6)上访问。我们对他们在推动该领域的努力以及使他们的工作为更广泛的社区所接触到的努力表示感谢。
