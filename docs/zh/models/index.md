---
comments: true
description: 探索 Ultralytics 支持的多样化 YOLO 系列、SAM、MobileSAM、FastSAM、YOLO-NAS 和 RT-DETR 模型。开启您的 CLI 和 Python 使用示例之旅。
keywords: Ultralytics, 文档, YOLO, SAM, MobileSAM, FastSAM, YOLO-NAS, RT-DETR, 模型, 架构, Python, CLI
---

# Ultralytics 支持的模型

欢迎来到 Ultralytics 的模型文档！我们提供多种模型的支持，每种模型都针对特定任务量身定做，如[对象检测](../tasks/detect.md)、[实例分割](../tasks/segment.md)、[图像分类](../tasks/classify.md)、[姿态估计](../tasks/pose.md)以及[多对象跟踪](../modes/track.md)。如果您有兴趣将您的模型架构贡献给 Ultralytics，请查看我们的[贡献指南](../../help/contributing.md)。

!!! Note "注意"

    🚧 我们的多语言文档目前正在建设中，我们正在努力进行完善。感谢您的耐心等待！🙏

## 特色模型

以下是一些关键模型的介绍：

1. **[YOLOv3](yolov3.md)**：由 Joseph Redmon 最初开发的 YOLO 模型家族的第三版，以其高效的实时对象检测能力而闻名。
2. **[YOLOv4](yolov4.md)**：由 Alexey Bochkovskiy 在 2020 年发布的 YOLOv3 的 darknet 原生更新版本。
3. **[YOLOv5](yolov5.md)**：Ultralytics 改进的 YOLO 架构版本，与先前版本相比，提供了更好的性能和速度权衡。
4. **[YOLOv6](yolov6.md)**：由[美团](https://about.meituan.com/)在 2022 年发布，用于公司多个自主送货机器人中。
5. **[YOLOv7](yolov7.md)**：YOLOv4 作者在 2022 年发布的更新版 YOLO 模型。
6. **[YOLOv8](yolov8.md) NEW 🚀**：YOLO 家族的最新版本，具备实例分割、姿态/关键点估计和分类等增强能力。
7. **[Segment Anything Model (SAM)](sam.md)**：Meta 的 Segment Anything Model (SAM)。
8. **[Mobile Segment Anything Model (MobileSAM)](mobile-sam.md)**：由庆熙大学开发的移动应用 MobileSAM。
9. **[Fast Segment Anything Model (FastSAM)](fast-sam.md)**：中国科学院自动化研究所图像与视频分析组开发的 FastSAM。
10. **[YOLO-NAS](yolo-nas.md)**：YOLO 神经网络结构搜索 (NAS) 模型。
11. **[Realtime Detection Transformers (RT-DETR)](rtdetr.md)**：百度 PaddlePaddle 实时检测变换器 (RT-DETR) 模型。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 使用 Ultralytics YOLO 模型在几行代码中运行。
</p>

## 入门：使用示例

此示例提供了简单的 YOLO 训练和推理示例。有关这些和其他[模式](../modes/index.md)的完整文档，请查看[Predict](../modes/predict.md)、[Train](../modes/train.md)、[Val](../modes/val.md) 和 [Export](../modes/export.md) 文档页面。

请注意，以下示例适用于对象检测的 YOLOv8 [Detect](../tasks/detect.md) 模型。有关其他支持任务的详细信息，请查看[Segment](../tasks/segment.md)、[Classify](../tasks/classify.md) 和 [Pose](../tasks/pose.md) 文档。

!!! Example "示例"

    === "Python"

        可将 PyTorch 预训练的 `*.pt` 模型以及配置文件 `*.yaml` 传入 `YOLO()`、`SAM()`、`NAS()` 和 `RTDETR()` 类，以在 Python 中创建模型实例：

        ```python
        from ultralytics import YOLO

        # 加载 COCO 预训练的 YOLOv8n 模型
        model = YOLO("yolov8n.pt")

        # 显示模型信息（可选）
        model.info()

        # 在 COCO8 示例数据集上训练模型 100 个周期
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # 使用 YOLOv8n 模型对 'bus.jpg' 图像进行推理
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        CLI 命令可直接运行模型：

        ```bash
        # 加载 COCO 预训练的 YOLOv8n 模型，并在 COCO8 示例数据集上训练 100 个周期
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # 加载 COCO 预训练的 YOLOv8n 模型，并对 'bus.jpg' 图像进行推理
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## 贡献新模型

有兴趣将您的模型贡献给 Ultralytics 吗？太好了！我们始终欢迎扩展我们的模型投资组合。

1. **Fork 仓库**：从 Fork [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics) 开始。

2. **克隆您的 Fork**：将您的 Fork 克隆到您的本地机器，并创建一个新的分支进行工作。

3. **实现您的模型**：按照我们在[贡献指南](../../help/contributing.md)中提供的编码标准和指南添加您的模型。

4. **彻底测试**：确保彻底测试您的模型，无论是独立测试还是作为流水线的一部分。

5. **创建拉取请求**：一旦您对您的模型满意，就创建一个拉取请求以供主仓库审查。

6. **代码审查与合并**：经过审查，如果您的模型符合我们的标准，它将被合并到主仓库中。

有关详细步骤，请参阅我们的[贡献指南](../../help/contributing.md)。
