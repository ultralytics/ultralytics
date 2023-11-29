---
comments: true
description: 探索Ultralytics YOLOv8的完整指南，这是一个高速、高精度的目标检测和图像分割模型。包括安装、预测、训练教程等。
keywords: Ultralytics, YOLOv8, 目标检测, 图像分割, 机器学习, 深度学习, 计算机视觉, YOLOv8安装, YOLOv8预测, YOLOv8训练, YOLO历史, YOLO许可
---

# Ultralytics 中文文档

<div align="center">
  <p>
    <a href="https://yolovision.ultralytics.com" target="_blank">
    <img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
  </p>
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.instagram.com/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="Ultralytics Instagram"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
  <br>
  <br>
  <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
  <a href="https://codecov.io/github/ultralytics/ultralytics"><img src="https://codecov.io/github/ultralytics/ultralytics/branch/main/graph/badge.svg?token=HHW7IIVFVY" alt="Ultralytics Code Coverage"></a>
  <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
  <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
  <br>
  <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
  <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
</div>

介绍 [Ultralytics](https://ultralytics.com) [YOLOv8](https://github.com/ultralytics/ultralytics)，这是备受好评的实时目标检测和图像分割模型的最新版本。YOLOv8基于深度学习和计算机视觉的前沿进展，提供了无与伦比的速度和准确性表现。它的精简设计使其适用于各种应用，并且可以轻松适应不同的硬件平台，从边缘设备到云API。

探索YOLOv8文档，这是一个全面的资源，旨在帮助您理解并利用其功能和能力。无论您是经验丰富的机器学习从业者还是新入行者，该中心旨在最大化YOLOv8在您的项目中的潜力。

## 从哪里开始

- **安装** `ultralytics` 并通过 pip 在几分钟内开始运行 &nbsp; [:material-clock-fast: 开始使用](quickstart.md){ .md-button }
- **预测** 使用YOLOv8预测新的图像和视频 &nbsp; [:octicons-image-16: 在图像上预测](modes/predict.md){ .md-button }
- **训练** 在您自己的自定义数据集上训练新的YOLOv8模型 &nbsp; [:fontawesome-solid-brain: 训练模型](modes/train.md){ .md-button }
- **探索** YOLOv8的任务，如分割、分类、姿态和跟踪 &nbsp; [:material-magnify-expand: 探索任务](tasks/index.md){ .md-button }

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 在<a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb" target="_blank">Google Colab</a>中如何训练您的自定义数据集上的YOLOv8模型。
</p>

## YOLO：简史

[YOLO](https://arxiv.org/abs/1506.02640) (You Only Look Once)，由华盛顿大学的Joseph Redmon和Ali Farhadi开发的流行目标检测和图像分割模型，于2015年推出，由于其高速和准确性而迅速流行。

- [YOLOv2](https://arxiv.org/abs/1612.08242) 在2016年发布，通过引入批量归一化、锚框和维度聚类来改进了原始模型。
- [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) 在2018年推出，进一步增强了模型的性能，使用了更高效的主干网络、多个锚点和空间金字塔池化。
- [YOLOv4](https://arxiv.org/abs/2004.10934) 在2020年发布，引入了Mosaic数据增强、新的无锚检测头和新的损失函数等创新功能。
- [YOLOv5](https://github.com/ultralytics/yolov5) 进一步改进了模型的性能，并增加了新功能，如超参数优化、集成实验跟踪和自动导出到常用的导出格式。
- [YOLOv6](https://github.com/meituan/YOLOv6) 在2022年由[美团](https://about.meituan.com/)开源，现在正在该公司的许多自动送货机器人中使用。
- [YOLOv7](https://github.com/WongKinYiu/yolov7) 在COCO关键点数据集上添加了额外的任务，如姿态估计。
- [YOLOv8](https://github.com/ultralytics/ultralytics) 是Ultralytics的YOLO的最新版本。作为一种前沿、最先进(SOTA)的模型，YOLOv8在之前版本的成功基础上引入了新功能和改进，以提高性能、灵活性和效率。YOLOv8支持全范围的视觉AI任务，包括[检测](https://docs.ultralytics.com/tasks/detect/), [分割](https://docs.ultralytics.com/tasks/segment/), [姿态估计](https://docs.ultralytics.com/tasks/pose/), [跟踪](https://docs.ultralytics.com/modes/track/), 和[分类](https://docs.ultralytics.com/tasks/classify/)。这种多功能性使用户能够利用YOLOv8的功能应对多种应用和领域的需求。

## YOLO许可证：Ultralytics YOLO是如何授权的？

Ultralytics提供两种许可选项以适应不同的使用场景：

- **AGPL-3.0许可证**：这种[OSI-approved](https://opensource.org/licenses/)开源许可证非常适合学生和爱好者，促进了开放的合作和知识共享。更多详细信息请参阅[LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)文件。
- **企业许可证**：这种许可证设计用于商业用途，允许将Ultralytics软件和AI模型无缝集成到商业商品和服务中，绕过AGPL-3.0的开源要求。如果您的场景涉及将我们的解决方案嵌入到商业产品中，请通过[Ultralytics Licensing](https://ultralytics.com/license)联系我们。

我们的授权策略旨在确保我们的开源项目的任何改进都能回馈到社区。我们十分珍视开源原则❤️，我们的使命是确保我们的贡献能够以对所有人有益的方式被利用和拓展。

---

**注意**：我们正在努力为我们的文档页面提供中文文档，并希望在接下来的几个月内发布。请密切关注我们的更新，并感谢您的耐心等待🙏。
