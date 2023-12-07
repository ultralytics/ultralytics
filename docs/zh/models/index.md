---
comments: true
description: 探索 Ultralytics 支持的 YOLO 系列、SAM、MobileSAM、FastSAM、YOLO-NAS 和 RT-DETR 模型多样化的范围。提供 CLI 和 Python 使用的示例以供入门。
keywords: Ultralytics, 文档, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, 模型, 架构, Python, CLI
---

# Ultralytics 支持的模型

欢迎来到 Ultralytics 的模型文档！我们支持多种模型，每种模型都针对特定任务进行了优化，如[对象检测](/../tasks/detect.md)、[实例分割](/../tasks/segment.md)、[图像分类](/../tasks/classify.md)、[姿态估计](/../tasks/pose.md)和[多对象追踪](/../modes/track.md)。如果您有兴趣将您的模型架构贡献给 Ultralytics，请查看我们的[贡献指南](/../help/contributing.md)。

!!! Note "笔记"

    Ultralytics 团队正忙于将文档翻译成多种语言。本页面上的链接目前可能会导向英文文档页面，因为我们正在努力扩展多语言文档支持。感谢您的耐心等待 🙏！

## 特色模型

以下是一些关键支持的模型：

1. **[YOLOv3](/../models/yolov3.md)**：YOLO 模型系列的第三个版本，最初由 Joseph Redmon 提出，以其高效的实时对象检测能力而闻名。
2. **[YOLOv4](/../models/yolov4.md)**：YOLOv3 的 darknet 本地更新，由 Alexey Bochkovskiy 在 2020 年发布。
3. **[YOLOv5](/../models/yolov5.md)**：Ultralytics 改进的 YOLO 架构版本，与之前的版本相比提供了更好的性能和速度折中选择。
4. **[YOLOv6](/../models/yolov6.md)**：由 [美团](https://about.meituan.com/) 在 2022 年发布，并在公司众多自主配送机器人中使用。
5. **[YOLOv7](/../models/yolov7.md)**：YOLOv4 作者在 2022 年发布的更新版 YOLO 模型。
6. **[YOLOv8](/../models/yolov8.md)**：YOLO 系列的最新版本，具备增强的功能，如实例分割、姿态/关键点估计和分类。
7. **[Segment Anything Model (SAM)](/../models/sam.md)**：Meta's Segment Anything Model (SAM)。
8. **[Mobile Segment Anything Model (MobileSAM)](/../models/mobile-sam.md)**：由庆熙大学为移动应用程序打造的 MobileSAM。
9. **[Fast Segment Anything Model (FastSAM)](/../models/fast-sam.md)**：中国科学院自动化研究所图像与视频分析组的 FastSAM。
10. **[YOLO-NAS](/../models/yolo-nas.md)**：YOLO 神经架构搜索 (NAS) 模型。
11. **[Realtime Detection Transformers (RT-DETR)](/../models/rtdetr.md)**：百度 PaddlePaddle 实时检测变换器 (RT-DETR) 模型。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>仅使用几行代码运行 Ultralytics YOLO 模型。
</p>

## 入门：使用示例

!!! Example "示例"

    === "Python"

        PyTorch 预训练的 `*.pt` 模型以及配置 `*.yaml` 文件都可以传递给 `YOLO()`、`SAM()`、`NAS()` 和 `RTDETR()` 类来在 Python 中创建模型实例：

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上训练模型 100 个周期
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # 使用 YOLOv8n 模型对 'bus.jpg' 图像进行推理
        results = model('path/to/bus.jpg')
        ```

    === "CLI"

        CLI 命令可直接运行模型：

        ```bash
        # 加载 COCO 预训练的 YOLOv8n 模型，并在 COCO8 示例数据集上训练它 100 个周期
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载 COCO 预训练的 YOLOv8n 模型，并对 'bus.jpg' 图像进行推理
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 贡献新模型

有兴趣将您的模型贡献给 Ultralytics 吗？太好了！我们始终欢迎扩展我们的模型组合。

1. **Fork 仓库**：首先 Fork [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)。

2. **克隆您的 Fork**：将您的 Fork 克隆到本地机器上，并创建一个新分支进行工作。

3. **实现您的模型**：按照我们在[贡献指南](/../help/contributing.md)中提供的编码标准和指南添加您的模型。

4. **彻底测试**：确保彻底测试您的模型，无论是独立还是作为整个管道的一部分。

5. **创建 Pull Request**：一旦您对您的模型感到满意，请创建一个到主仓库的 Pull Request 以便审查。

6. **代码审查与合并**：经审查，如果您的模型符合我们的标准，它将被合并到主仓库中。

有关详细步骤，请参阅我们的[贡献指南](/../help/contributing.md)。
