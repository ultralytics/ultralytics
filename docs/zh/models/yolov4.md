---
comments: true
description: 通过我们详细的YOLOv4指南，探索最先进的实时目标检测器。了解其建筑亮点，创新功能和应用示例。
keywords: ultralytics, YOLOv4, 目标检测, 神经网络, 实时检测, 目标检测器, 机器学习
---

# YOLOv4：高速和精确的目标检测

欢迎来到Ultralytics关于YOLOv4的文档页面，YOLOv4是由Alexey Bochkovskiy于2020年在 [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) 发布的最先进的实时目标检测器。YOLOv4旨在提供速度和准确性的最佳平衡，使其成为许多应用的优秀选择。

![YOLOv4架构图](https://user-images.githubusercontent.com/26833433/246185689-530b7fe8-737b-4bb0-b5dd-de10ef5aface.png)
**YOLOv4架构图**。展示了YOLOv4的复杂网络设计，包括主干，颈部和头部组件以及它们相互连接的层，以实现最佳的实时目标检测。

## 简介

YOLOv4代表You Only Look Once版本4。它是为解决之前YOLO版本（如[YOLOv3](yolov3.md)）和其他目标检测模型的局限性而开发的实时目标检测模型。与其他基于卷积神经网络（CNN）的目标检测器不同，YOLOv4不仅适用于推荐系统，还可用于独立的进程管理和减少人工输入。它在传统图形处理单元（GPU）上的操作可以以经济实惠的价格进行大规模使用，并且设计为在常规GPU上实时工作，仅需要一个这样的GPU进行训练。

## 架构

YOLOv4利用了几个创新功能，这些功能共同优化其性能。这些功能包括加权残差连接（WRC），跨阶段部分连接（CSP），交叉mini-Batch归一化（CmBN），自适应对抗训练（SAT），Mish激活函数，Mosaic数据增强，DropBlock正则化和CIoU损失。这些功能的组合可以实现最先进的结果。

典型的目标检测器由几个部分组成，包括输入、主干、颈部和头部。YOLOv4的主干是在ImageNet上预训练的，用于预测对象的类别和边界框。主干可以来自多个模型，包括VGG、ResNet、ResNeXt或DenseNet。检测器的颈部部分用于从不同阶段收集特征图，通常包括几条自底向上的路径和几条自顶向下的路径。头部部分用于进行最终的目标检测和分类。

## 免费赠品

YOLOv4还使用了称为“免费赠品”的方法，这些方法在训练过程中提高模型的准确性，而不增加推理成本。数据增强是目标检测中常用的一种免费赠品技术，它增加了输入图像的变异性，以提高模型的鲁棒性。一些数据增强的例子包括光度失真（调整图像的亮度、对比度、色调、饱和度和噪音）和几何失真（添加随机缩放、裁剪、翻转和旋转）。这些技术帮助模型更好地应对不同类型的图像。

## 特点和性能

YOLOv4被设计为在目标检测中具有最佳速度和准确性。YOLOv4的架构包括CSPDarknet53作为主干，PANet作为颈部，以及YOLOv3作为检测头。这种设计使得YOLOv4能够以令人印象深刻的速度进行目标检测，适用于实时应用。YOLOv4在准确性方面也表现出色，在目标检测基准测试中取得了最先进的结果。

## 使用示例

截至撰写本文时，Ultralytics当前不支持YOLOv4模型。因此，任何有兴趣使用YOLOv4的用户需要直接参考YOLOv4 GitHub存储库中的安装和使用说明。

以下是使用YOLOv4的典型步骤的简要概述：

1. 访问YOLOv4 GitHub存储库：[https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)。

2. 按照README文件中提供的说明进行安装。这通常涉及克隆存储库，安装必要的依赖项，并设置任何必要的环境变量。

3. 安装完成后，您可以根据存储库提供的使用说明训练和使用模型。这通常涉及准备您的数据集、配置模型参数、训练模型，然后使用训练好的模型进行目标检测。

请注意，具体的步骤可能因您的特定用例和YOLOv4存储库的当前状态而有所不同。因此，强烈建议直接参考YOLOv4 GitHub存储库中提供的说明。

对于Ultralytics不支持YOLOv4的情况，我们感到非常抱歉，我们将努力更新本文档，以包括使用Ultralytics支持的YOLOv4的示例。

## 结论

YOLOv4是一种强大而高效的目标检测模型，它在速度和准确性之间取得了平衡。它在训练过程中使用独特的功能和免费赠品技术，使其在实时目标检测任务中表现出色。任何具备常规GPU的人都可以进行YOLOv4的训练和使用，使其对于各种应用具有可访问性和实用性。

## 引文和致谢

我们要感谢YOLOv4的作者对实时目标检测领域的重要贡献：

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{bochkovskiy2020yolov4,
              title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
              author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
              year={2020},
              eprint={2004.10934},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

YOLOv4的原始论文可以在[arXiv](https://arxiv.org/abs/2004.10934)上找到。作者已经公开了他们的工作，代码库可以在[GitHub](https://github.com/AlexeyAB/darknet)上获取。我们赞赏他们在推动该领域方面的努力，并使他们的工作对广大社区产生影响。
