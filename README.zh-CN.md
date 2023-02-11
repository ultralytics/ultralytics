<div align="center">
  <p>
    <a href="https://ultralytics.com/yolov8" target="_blank">
      <img width="850" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png"></a>
  </p>

[English](README.md) | [简体中文](README.zh-CN.md)
<br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"/></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
  <br>

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 是由 [Ultralytics](https://ultralytics.com) 开发的一个前沿的 SOTA 模型。它在以前成功的 YOLO 版本基础上，引入了新的功能和改进，进一步提升了其性能和灵活性。YOLOv8 基于快速、准确和易于使用的设计理念，使其成为广泛的目标检测、图像分割和图像分类任务的绝佳选择。

如果要申请企业许可证，请填写 [Ultralytics 许可](https://ultralytics.com/license)。

<div align="center">
    <a href="https://github.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.producthunt.com/@glenn_jocher" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-producthunt.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.facebook.com/ultralytics" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-facebook.png" width="2%" alt="" /></a>
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
    <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
      <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="2%" alt="" /></a>
  </div>
</div>

## <div align="center">文档</div>

有关训练、测试和部署的完整文档见[YOLOv8 Docs](https://docs.ultralytics.com)。请参阅下面的快速入门示例。

<details open>
<summary>安装</summary>

Pip 安装包含所有 [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) 的 ultralytics 包，环境要求 [**3.10>=Python>=3.7**](https://www.python.org/)，且 [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/)。

```bash
pip install ultralytics
```

</details>

<details open>
<summary>使用方法</summary>

YOLOv8 可以直接在命令行界面（CLI）中使用 `yolo` 命令运行：

```bash
yolo predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg"
```

`yolo`可以用于各种任务和模式，并接受额外的参数，例如 `imgsz=640`。参见 YOLOv8 [文档](https://docs.ultralytics.com)中可用`yolo`[参数](https://docs.ultralytics.com/cfg/)的完整列表。

```bash
yolo task=detect    mode=train    model=yolov8n.pt        args...
          classify       predict        yolov8n-cls.yaml  args...
          segment        val            yolov8n-seg.yaml  args...
                         export         yolov8n.pt        format=onnx  args...
```

YOLOv8 也可以在 Python 环境中直接使用，并接受与上面 CLI 例子中相同的[参数](https://docs.ultralytics.com/cfg/)：

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（推荐用于训练）

# Use the model
results = model.train(data="coco128.yaml", epochs=3)  # 训练模型
results = model.val()  # 在验证集上评估模型性能
results = model("https://ultralytics.com/images/bus.jpg")  # 预测图像
success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
```

[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models) 会从 Ultralytics [发布页](https://github.com/ultralytics/ultralytics/releases) 自动下载。

### 已知问题 / 待办事项

我们仍在努力完善 YOLOv8 的几个部分！我们的目标是尽快完成这些工作，使 YOLOv8 的功能设置达到YOLOv5 的水平，包括对所有相同格式的导出和推理。我们还在写一篇 YOLOv8 的论文，一旦完成，我们将提交给 [arxiv.org](https://arxiv.org)。

- [ ] TensorFlow 导出
- [ ] DDP 恢复训练
- [ ] [arxiv.org](https://arxiv.org) 论文

</details>

## <div align="center">模型</div>

所有 YOLOv8 的预训练模型都可以在这里找到。目标检测和分割模型是在 COCO 数据集上预训练的，而分类模型是在 ImageNet 数据集上预训练的。

第一次使用时，[模型](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models) 会从 Ultralytics [发布页](https://github.com/ultralytics/ultralytics/releases) 自动下载。

<details open><summary>目标检测</summary>

| 模型                                                                                   | 尺寸<br><sup>（像素） | mAP<sup>val<br>50-95 | 推理速度<br><sup>CPU ONNX<br>(ms) | 推理速度<br><sup>A100 TensorRT<br>(ms) | 参数量<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------- | -------------------- | ----------------------------- | ---------------------------------- | --------------- | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640             | 37.3                 | 80.4                          | 0.99                               | 3.2             | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640             | 44.9                 | 128.4                         | 1.20                               | 11.2            | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640             | 50.2                 | 234.7                         | 1.83                               | 25.9            | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640             | 52.9                 | 375.2                         | 2.39                               | 43.7            | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640             | 53.9                 | 479.1                         | 3.53                               | 68.2            | 257.8             |

- **mAP<sup>val</sup>** 结果都在 [COCO val2017](http://cocodataset.org) 数据集上，使用单模型单尺度测试得到。
  <br>复现命令 `yolo val detect data=coco.yaml device=0`
- **推理速度**使用 COCO 验证集图片推理时间进行平均得到，测试环境使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例。
  <br>复现命令 `yolo val detect data=coco128.yaml batch=1 device=0/cpu`

</details>

<details><summary>实例分割</summary>

| 模型                                                                                       | 尺寸<br><sup>（像素） | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | 推理速度<br><sup>CPU ONNX<br>(ms) | 推理速度<br><sup>A100 TensorRT<br>(ms) | 参数量<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------- | --------------- | -------------------- | --------------------- | ----------------------------- | ---------------------------------- | --------------- | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640             | 36.7                 | 30.5                  | 96.1                          | 1.21                               | 3.4             | 12.6              |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640             | 44.6                 | 36.8                  | 155.7                         | 1.47                               | 11.8            | 42.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640             | 49.9                 | 40.8                  | 317.0                         | 2.18                               | 27.3            | 110.2             |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640             | 52.3                 | 42.6                  | 572.4                         | 2.79                               | 46.0            | 220.5             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640             | 53.4                 | 43.4                  | 712.1                         | 4.02                               | 71.8            | 344.1             |

- **mAP<sup>val</sup>**  结果都在 [COCO val2017](http://cocodataset.org) 数据集上，使用单模型单尺度测试得到。
  <br>复现命令 `yolo val segment data=coco.yaml device=0`
- **推理速度**使用 COCO 验证集图片推理时间进行平均得到，测试环境使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例。
  <br>复现命令 `yolo val segment data=coco128-seg.yaml batch=1 device=0/cpu`

</details>

<details><summary>分类</summary>

| 模型                                                                                       | 尺寸<br><sup>（像素） | acc<br><sup>top1 | acc<br><sup>top5 | 推理速度<br><sup>CPU ONNX<br>(ms) | 推理速度<br><sup>A100 TensorRT<br>(ms) | 参数量<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| ---------------------------------------------------------------------------------------- | --------------- | ---------------- | ---------------- | ----------------------------- | ---------------------------------- | --------------- | ------------------------ |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt) | 224             | 66.6             | 87.0             | 12.9                          | 0.31                               | 2.7             | 4.3                      |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt) | 224             | 72.3             | 91.1             | 23.4                          | 0.35                               | 6.4             | 13.5                     |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt) | 224             | 76.4             | 93.2             | 85.4                          | 0.62                               | 17.0            | 42.7                     |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-cls.pt) | 224             | 78.0             | 94.1             | 163.0                         | 0.87                               | 37.5            | 99.7                     |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt) | 224             | 78.4             | 94.3             | 232.0                         | 1.01                               | 57.4            | 154.8                    |

- **acc** 都在 [ImageNet](https://www.image-net.org/) 数据集上，使用单模型单尺度测试得到。
  <br>复现命令 `yolo val classify data=path/to/ImageNet device=0`
- **推理速度**使用 ImageNet 验证集图片推理时间进行平均得到，测试环境使用 [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) 实例。
  <br>复现命令 `yolo val classify data=path/to/ImageNet batch=1 device=0/cpu`

</details>

## <div align="center">模块集成</div>

<br>
<a href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png"></a>
<br>
<br>

<div align="center">
  <a href="https://roboflow.com/?ref=ultralytics">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-roboflow.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://cutt.ly/yolov5-readme-clearml">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-clearml.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://bit.ly/yolov5-readme-comet">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-comet.png" width="10%" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="" />
  <a href="https://bit.ly/yolov5-neuralmagic">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-neuralmagic.png" width="10%" /></a>
</div>

|                                      Roboflow                                      |                                 ClearML ⭐ 新                                 |                                     Comet ⭐ 新                                     |                                    Neural Magic ⭐ 新                                    |
| :--------------------------------------------------------------------------------: | :-------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: |
| 将您的自定义数据集进行标注并直接导出到 YOLOv8 以进行训练 [Roboflow](https://roboflow.com/?ref=ultralytics) | 自动跟踪、可视化甚至远程训练 YOLOv8 [ClearML](https://cutt.ly/yolov5-readme-clearml)（开源！） | 永远免费，[Comet](https://bit.ly/yolov5-readme-comet)可让您保存 YOLOv8 模型、恢复训练以及交互式可视化和调试预测 | 使用 [Neural Magic DeepSparse](https://bit.ly/yolov5-neuralmagic)，运行 YOLOv8 推理的速度最高可提高6倍 |

## <div align="center">Ultralytics HUB</div>

[Ultralytics HUB](https://bit.ly/ultralytics_hub) 是我们⭐ **新**的无代码解决方案，用于可视化数据集，训练 YOLOv8🚀 模型，并以无缝体验方式部署到现实世界。现在开始**免费**! 还可以通过下载 [Ultralytics App](https://ultralytics.com/app_install) 在你的 iOS 或 Android 设备上运行 YOLOv8 模型!

<a href="https://bit.ly/ultralytics_hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png"></a>

## <div align="center">贡献</div>

我们喜欢您的意见或建议！我们希望尽可能简单和透明地为 YOLOv8 做出贡献。请看我们的 [贡献指南](CONTRIBUTING.md) ，并填写 [调查问卷](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) 向我们发送您的体验反馈。感谢我们所有的贡献者！

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" /></a>

## <div align="center">License</div>

- YOLOv8 在两种不同的 License 下可用：
  - **GPL-3.0 License**： 查看 [License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 文件的详细信息。
  - **企业License**：在没有 GPL-3.0 开源要求的情况下为商业产品开发提供更大的灵活性。典型用例是将 Ultralytics 软件和 AI 模型嵌入到商业产品和应用程序中。在以下位置申请企业许可证 [Ultralytics 许可](https://ultralytics.com/license) 。

## <div align="center">联系我们</div>

若发现 YOLOv8 的 Bug 或有功能需求，请访问 [GitHub 问题](https://github.com/ultralytics/ultralytics/issues)。如需专业支持，请 [联系我们](https://ultralytics.com/contact)。

<br>
<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.producthunt.com/@glenn_jocher" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-producthunt.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.facebook.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-facebook.png" width="3%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="3%" alt="" /></a>
</div>
