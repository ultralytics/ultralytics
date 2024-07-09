---
comments: true
description: Discover Ultralytics YOLOv8 - the latest in real-time object detection and image segmentation. Learn its features and maximize its potential in your projects.
keywords: Ultralytics, YOLOv8, object detection, image segmentation, deep learning, computer vision, AI, machine learning, documentation, tutorial
---

<div align="center">
<a href="https://github.com/ultralytics/assets/releases/tag/v8.2.0" target="_blank"><img width="1024%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
<a href="https://docs.ultralytics.com/zh/">中文</a> |
<a href="https://docs.ultralytics.com/ko/">한국어</a> |
<a href="https://docs.ultralytics.com/ja/">日本語</a> |
<a href="https://docs.ultralytics.com/ru/">Русский</a> |
<a href="https://docs.ultralytics.com/de/">Deutsch</a> |
<a href="https://docs.ultralytics.com/fr/">Français</a> |
<a href="https://docs.ultralytics.com/es/">Español</a> |
<a href="https://docs.ultralytics.com/pt/">Português</a> |
<a href="https://docs.ultralytics.com/tr/">Türkçe</a> |
<a href="https://docs.ultralytics.com/vi/">Tiếng Việt</a> |
<a href="https://docs.ultralytics.com/hi/">हिन्दी</a> |
<a href="https://docs.ultralytics.com/ar/">العربية</a>
<br>
<br>
<a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
<a href="https://codecov.io/github/ultralytics/ultralytics"><img src="https://codecov.io/github/ultralytics/ultralytics/branch/main/graph/badge.svg?token=HHW7IIVFVY" alt="Ultralytics Code Coverage"></a>
<a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
<a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
<a href="https://ultralytics.com/discord"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
<br>
<a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
<a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
<a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
</div>

介绍 [Ultralytics](https://ultralytics.com) [YOLOv8](https://github.com/ultralytics/ultralytics)，广受赞誉的实时目标检测和图像分割模型的最新版本。YOLOv8建立在深度学习和计算机视觉的前沿技术之上，在速度和准确性方面提供了无与伦比的性能。其流线型设计使其适用于各种应用，并可轻松适应不同的硬件平台，从边缘设备到云API。

探索YOLOv8文档，这是一个全面的资源，旨在帮助您了解和利用其特性和功能。无论您是经验丰富的机器学习从业者还是该领域的新手，该中心都旨在最大限度地发挥YOLOv8在您的项目中的潜力

<div align="center">
  <br>
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/discord"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>

## 从哪里开始

- **安装** `ultralytics` 使用pip并在几分钟内启动并运行 &nbsp; [:material-clock-fast: 开始](quickstart.md){ .md-button }
- **预测** 使用YOLOv8预测新图像和视频 &nbsp; [:octicons-image-16: 图像预测](modes/predict.md){ .md-button }
- **训练** 在您自己的自定义数据集上使用新的YOLOv8模型 &nbsp; [:fontawesome-solid-brain: 训练模型](modes/train.md){ .md-button }
- **任务** YOLOv8任务，如分段、分类、姿势估计和跟踪 &nbsp; [:material-magnify-expand: 探索任务](tasks/index.md){ .md-button }
- **NEW 🚀 探索** 具有高级语义和SQL搜索的数据集 &nbsp; [:material-magnify-expand: 探索数据集](datasets/explorer/index.md){ .md-button }

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看:</strong> 如何在自定义数据集上训练YOLOv8模型 <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb" target="_blank">Google Colab</a>.
</p>

## YOLO: 简史

[YOLO](https://arxiv.org/abs/1506.02640) (You Only Look Once)，华盛顿大学的约瑟夫·雷德蒙（Joseph Redmon）和阿里·法哈迪（Ali Farhadi）开发了一种流行的目标检测和图像分割模型。YOLO于2015年推出，因其高速和准确性而迅速流行起来。

- [YOLOv2](https://arxiv.org/abs/1612.08242)，2016年发布，通过合并批量归一化、锚框和维度聚类改进了原始模型。
- [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)，该模型于2018年推出，使用更高效的骨干网络、多个锚点和空间金字塔池进一步增强了模型的性能。
- [YOLOv4](https://arxiv.org/abs/2004.10934) 于2020年发布，引入了Mosaic数据增强、新的无锚检测头和新的损失函数等创新。
- [YOLOv5](https://github.com/ultralytics/yolov5) 进一步提高了模型的性能，并增加了超参数优化、集成实验跟踪和自动导出到流行导出格式等新功能。
- [YOLOv6](https://github.com/meituan/YOLOv6) 由[美团](https://about.meituan.com/)于2022年开源，并用于该公司的许多自动送货机器人。
- [YOLOv7](https://github.com/WongKinYiu/yolov7) 添加了其他任务，例如对COCO关键点数据集进行姿态估计。
- [YOLOv8](https://github.com/ultralytics/ultralytics) 是Ultralytics的YOLO的最新版本。作为尖端、最先进的（SOTA）模型，YOLOv8建立在以前版本的成功基础上，引入了新功能和改进，以增强性能、灵活性和效率。YOLOv8支持全方位的视觉AI任务，包括[检测](tasks/detect.md)，[分割](tasks/segment.md)，[姿态估计](tasks/pose.md)，[跟踪](modes/track.md), 和[分类](tasks/classify.md)。这种多功能性使用户能够在不同的应用和领域中利用YOLOv8的功能。
- [YOLOv9](models/yolov9.md) 引入了可编程梯度信息（PGI）和广义高效层聚合网络（GELAN）等创新方法。
- [YOLOv10](models/yolov10.md) 由[清华大学](https://www.tsinghua.edu.cn/en/)的研究人员使用[Ultralytics](https://ultralytics.com/) [Python 包](https://pypi.org/project/ultralytics/)创建。此版本通过引入消除非最大抑制（NMS）要求的端到端检测头，提供了实时[目标检测](tasks/detect.md)改进。

## YOLO许可证: Ultralytics YOLO如何获得许可？

Ultralytics提供两种许可选项以适应不同的情况：

- **AGPL-3.0 许可证**: 这种[经OSI批准](https://opensource.org/licenses/)开源许可证非常适合学生和爱好者，促进开放式协作和知识共享。有关详细信息，请参阅[LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)文件。
- **企业许可证**: 该许可证专为商业用途而设计，允许将 Ultralytics软件和AI模型无缝集成到商业商品和服务中，绕过AGPL-3.0的开源要求。如果您的方案涉及将我们的解决方案嵌入到商业产品中，请通过[Ultralytics 许可](https://ultralytics.com/license)联系我们。

我们的许可策略旨在确保对开源项目的任何改进都返回给社区。我们将开源原则铭记于心❤️，我们的使命是保证我们的贡献能够以有益于所有人的方式得到利用和扩展。

## 常见问题

### 什么是Ultralytics YOLO，它如何改进目标检测？

Ultralytics YOLO是广受赞誉的YOLO（You Only Look Once）系列的最新进展，用于实时物体检测和图像分割。它基于以前的版本，引入了新功能和改进，以增强性能、灵活性和效率。YOLOv8支持检测、分割、姿态估计、跟踪、分类等多种[视觉AI 任务](tasks/index.md)。其最先进的架构确保了卓越的速度和准确性，使其适用于各种应用，包括边缘设备和云API。

### 如何开始YOLO安装和设置？

YOLO的入门既快速又简单。您可以使用pip安装Ultralytics软件包，并在几分钟内启动并运行。下面是一个基本的安装命令:

```bash
pip install ultralytics
```

有关全面的分步指南，请访问我们的[快速入门指南](quickstart.md)。此资源将帮助您了解安装说明、初始设置和运行您的第一个模型。

### 如何在数据集上训练自定义YOLO模型？

在数据集上训练自定义YOLO模型涉及几个详细步骤:

1. 准备带注释的数据集。
2. 在YAML文件中配置训练参数。
3. 使用`yolo train`命令开始训练。

下面是一个示例命令:

```bash
yolo train model=yolov8n.pt data=coco128.yaml epochs=100 imgsz=640
```

有关详细演练，请查看我们的[训练模型](modes/train.md)指南，其中包含用于优化训练过程的示例和提示。

### Ultralytics YOLO有哪些许可选项？

Ultralytics为YOLO提供两种许可选项:

- **AGPL-3.0 许可证**: 此开源许可证非常适合教育和非商业用途，促进开放协作。
- **企业许可证**: 这是为商业应用而设计的，允许将Ultralytics软件无缝集成到商业产品中，而不受AGPL-3.0许可证的限制。

有关更多详细信息，请访问我们的[许可](https://ultralytics.com/license)页面。

### Ultralytics YOLO如何用于实时对象跟踪？

Ultralytics YOLO支持高效且可定制的多对象跟踪。要利用跟踪功能，您可以使用`yolo track`命令，如下所示:

```bash
yolo track model=yolov8n.pt source=video.mp4
```

有关设置和运行对象跟踪的详细指南，请查看我们的[跟踪模式](modes/track.md)文档，其中解释了实时场景中的配置和实际应用。
