---
comments: true
description: 探索YOLOv7，一个实时物体检测器。了解其卓越的速度，令人印象深刻的精确度和独特的可训练无需付费优化聚焦点。
keywords: YOLOv7，实时物体检测器，State-of-the-art，Ultralytics，MS COCO数据集，模型重新参数化，动态标签分配，扩展缩放，复合缩放
---

# YOLOv7：可训练无需付费

YOLOv7是一种实时物体检测器的最新技术，其速度和准确度超过了目前已知的所有物体检测器，速度范围在5 FPS到160 FPS之间。在GPU V100上，它在所有已知实时物体检测器中具有最高的准确度（56.8％AP），且帧率达到30 FPS或更高。此外，YOLOv7在速度和准确度方面也优于其他物体检测器，如YOLOR，YOLOX，缩放后的YOLOv4，YOLOv5等等。该模型是从头开始使用MS COCO数据集进行训练的，而没有使用其他数据集或预训练权重。YOLOv7的源代码可在GitHub上获得。

![YOLOv7与SOTA物体检测器的比较](https://github.com/ultralytics/ultralytics/assets/26833433/5e1e0420-8122-4c79-b8d0-2860aa79af92)
**最先进物体检测器的比较**。从表2的结果可以看出，所提出的方法在速度和准确度的均衡上最佳。将YOLOv7-tiny-SiLU与YOLOv5-N（r6.1）进行比较，我们的方法在AP上快了127 FPS，准确度提高了10.7％。此外，YOLOv7在161 FPS的帧率下具有51.4％的AP，而具有相同AP的PPYOLOE-L仅具有78 FPS的帧率。在参数使用方面，YOLOv7比PPYOLOE-L少了41％。将YOLOv7-X与114 FPS的推理速度与YOLOv5-L（r6.1）的99 FPS的推理速度进行比较，YOLOv7-X可以提高3.9％的AP。如果将YOLOv7-X与类似规模的YOLOv5-X（r6.1）进行比较，YOLOv7-X的推理速度比YOLOv5-X快31 FPS。此外，就参数和计算量而言，与YOLOv5-X（r6.1）相比，YOLOv7-X减少了22％的参数和8％的计算量，但AP提高了2.2％（[来源](https://arxiv.org/pdf/2207.02696.pdf)）。

## 概述

实时物体检测是许多计算机视觉系统的重要组件，包括多目标跟踪，自动驾驶，机器人技术和医学图像分析等。近年来，实时物体检测的发展一直致力于设计高效的架构，并提高各种CPU，GPU和神经处理单元（NPU）的推理速度。YOLOv7支持移动GPU和GPU设备，从边缘到云端。

与传统的实时物体检测器侧重于架构优化不同，YOLOv7引入了对训练过程优化的关注。这包括模块和优化方法，旨在提高目标检测的准确性而不增加推理成本，这个概念被称为“可训练无需付费”。

## 主要特性

YOLOv7引入了几个关键特性：

1. **模型重新参数化**：YOLOv7提出了一种计划好的重新参数化模型，它是一种适用于不同网络中的层的策略，具有梯度传播路径的概念。

2. **动态标签分配**：对多个输出层的模型进行训练会遇到一个新问题：“如何为不同分支的输出分配动态目标？”为了解决这个问题，YOLOv7引入了一种新的标签分配方法，称为粗到细的引导式标签分配。

3. **扩展和复合缩放**：YOLOv7提出了适用于实时物体检测器的“扩展”和“复合缩放”方法，可以有效利用参数和计算。

4. **效率**：YOLOv7提出的方法可以有效地减少最先进实时物体检测器的约40％的参数和50％的计算量，并具有更快的推理速度和更高的检测准确度。

## 使用示例

截至撰写本文时，Ultralytics当前不支持YOLOv7模型。因此，任何希望使用YOLOv7的用户都需要直接参考YOLOv7 GitHub存储库中的安装和使用说明。

这是您可能采取的使用YOLOv7的典型步骤的简要概述：

1. 访问YOLOv7 GitHub存储库：[https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)。

2. 按照README文件中提供的说明进行安装。这通常涉及克隆存储库，安装必要的依赖项，并设置任何必要的环境变量。

3. 安装完成后，您可以根据存储库中提供的使用说明训练和使用模型。这通常涉及准备数据集，配置模型参数，训练模型，然后使用训练好的模型执行物体检测。

请注意，具体的步骤可能因您的特定用例和YOLOv7存储库的当前状态而有所不同。因此，强烈建议直接参考YOLOv7 GitHub存储库中提供的说明。

我们对这可能造成的任何不便表示歉意，并将努力更新此文档以提供针对Ultralytics的YOLOv7支持的使用示例。

## 引用和致谢

我们要感谢YOLOv7的作者在实时物体检测领域做出的重大贡献：

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{wang2022yolov7,
          title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
          author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
          journal={arXiv preprint arXiv:2207.02696},
          year={2022}
        }
        ```

YOLOv7的原始论文可以在[arXiv](https://arxiv.org/pdf/2207.02696.pdf)上找到。作者已将其工作公开，并且代码库可在[GitHub](https://github.com/WongKinYiu/yolov7)中访问。我们感谢他们在推动该领域发展并使其工作对广大社区可访问的努力。
