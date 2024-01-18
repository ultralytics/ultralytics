---
comments: true
description: 了解百度的RT-DETR，一种基于Vision Transformers的高效灵活的实时目标检测器，包括预训练模型的特性和优势。
keywords: RT-DETR、Baidu、Vision Transformers、目标检测、实时表现、CUDA、TensorRT、IoU感知的查询选择、Ultralytics、Python API、PaddlePaddle
---

# 百度的RT-DETR：基于Vision Transformers的实时目标检测器

## 概览

百度开发的实时检测变换器（RT-DETR）是一种尖端的端到端目标检测器，具有实时性能和高准确性。它利用Vision Transformers (ViT) 的强大功能，通过解耦内部尺度交互和跨尺度融合，高效处理多尺度特征。RT-DETR非常灵活适应各种推断速度的调整，支持使用不同的解码器层而无需重新训练。该模型在CUDA和TensorRT等加速后端上表现出色，超越了许多其他实时目标检测器。

![模型示例图像](https://user-images.githubusercontent.com/26833433/238963168-90e8483f-90aa-4eb6-a5e1-0d408b23dd33.png)
**百度的RT-DETR概览** 百度的RT-DETR模型架构图显示了骨干网的最后三个阶段{S3, S4, S5}作为编码器输入。高效的混合编码器通过内部尺度特征交互（AIFI）和跨尺度特征融合模块（CCFM）将多尺度特征转换为图像特征序列。采用IoU感知的查询选择来选择一定数量的图像特征作为解码器的初始对象查询。最后，解码器通过辅助预测头迭代优化对象查询，生成框和置信度得分。（[文章来源](https://arxiv.org/pdf/2304.08069.pdf)）

### 主要特点

- **高效的混合编码器：** 百度的RT-DETR使用高效的混合编码器，通过解耦内部尺度交互和跨尺度融合来处理多尺度特征。这种独特的Vision Transformers架构降低了计算成本，实现实时目标检测。
- **IoU感知的查询选择：** 百度的RT-DETR利用IoU感知的查询选择改进了对象查询的初始化。这使得模型能够聚焦于场景中最相关的对象，提高了检测准确性。
- **灵活的推断速度：** 百度的RT-DETR支持使用不同的解码器层灵活调整推断速度，无需重新训练。这种适应性有助于在各种实时目标检测场景中实际应用。

## 预训练模型

Ultralytics Python API提供了不同尺度的预训练PaddlePaddle RT-DETR模型：

- RT-DETR-L: 在COCO val2017上达到53.0%的AP，在T4 GPU上达到114 FPS
- RT-DETR-X: 在COCO val2017上达到54.8%的AP，在T4 GPU上达到74 FPS

## 使用示例

此示例提供了简单的RT-DETR训练和推断示例。有关这些和其他[模式](../modes/index.md)的完整文档，请参阅[预测](../modes/predict.md)、[训练](../modes/train.md)、[验证](../modes/val.md)和[导出](../modes/export.md)文档页面。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import RTDETR

        # 加载预训练的COCO RT-DETR-l模型
        model = RTDETR('rtdetr-l.pt')

        # 显示模型信息（可选）
        model.info()

        # 使用COCO8示例数据集对模型进行100个epoch的训练
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 使用RT-DETR-l模型在'bus.jpg'图像上运行推断
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        ```bash
        # 加载预训练的COCO RT-DETR-l模型，并在COCO8示例数据集上进行100个epoch的训练
        yolo train model=rtdetr-l.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载预训练的COCO RT-DETR-l模型，并在'bus.jpg'图像上运行推断
        yolo predict model=rtdetr-l.pt source=path/to/bus.jpg
        ```

## 支持的任务和模式

该表格提供了各个模型类型、具体的预训练权重、各个模型支持的任务以及支持的各种模式（[训练](../modes/train.md)、[验证](../modes/val.md)、[预测](../modes/predict.md)、[导出](../modes/export.md)），其中✅表示支持。

| 模型类型                | 预训练权重         | 支持的任务                      | 推断 | 验证 | 训练 | 导出 |
|---------------------|---------------|----------------------------|----|----|----|----|
| RT-DETR-Large       | `rtdetr-l.pt` | [目标检测](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅  |
| RT-DETR-Extra-Large | `rtdetr-x.pt` | [目标检测](../tasks/detect.md) | ✅  | ✅  | ✅  | ✅  |

## 引用和致谢

如果你在研究或开发中使用了百度的RT-DETR，请引用[原始论文](https://arxiv.org/abs/2304.08069)：

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{lv2023detrs,
              title={DETRs Beat YOLOs on Real-time Object Detection},
              author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
              year={2023},
              eprint={2304.08069},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们要感谢百度和[PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection)团队为计算机视觉社区创建和维护了这个宝贵的资源。非常感谢他们使用基于Vision Transformers的实时目标检测器RT-DETR在该领域做出的贡献。

*keywords: RT-DETR、Transformer、ViT、Vision Transformers、Baidu RT-DETR、PaddlePaddle、Paddle Paddle RT-DETR，实时目标检测、基于Vision Transformers的目标检测、预训练的PaddlePaddle RT-DETR模型、百度RT-DETR的使用、Ultralytics Python API*
