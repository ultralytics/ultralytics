# this repo is a fork of ultralytics, site:https://github.com/ultralytics.com

## added new features:

1. use this source code train models on own datasets
2. in order to modify the model, you need to modify the model.py file
3. send train information to redis
4. ultralytics's tree
   ├── docker # 官方的 docker 版本（不用 docker 的话不用管）
   ├── docs # 官方文档
   ├── examples # 官方实例
   ├── test # 测试文件
   └── ultralytics # 核心的模型代码文件和其他工具组件
   ├── assets # 感觉好像没啥用的
   ├── cfg # --- 参数 ---
   │ ├── datasets # 数据集的参数
   │ ├── models # 各类模型的网络结构参数 rt-detr,v3,v5,v6,v8,v9,v10,v11
   │ ├── trackers # ？ 部分参数文件
   │ └── default.yaml # ！ 训练器默认会加载的参数
   ├── data # --- 数据处理 ---
   │ ├── explorer # 可视化
   │ ├── scripts # 获取数据的脚本文件
   │ ├── augment.py # ！ 所有的数据增强类和实现
   │ └── **_.py # 各类数据读取器、数据处理等相关代码
   ├── engine # --- 最底层的类定义 ---
   │ ├── model.py # 模型基类
   │ ├── trainer.py # 训练器基类
   │ └── _**.py # 其他各种基类
   ├── hub # ？
   ├── models # --- 外层的类定义 ---
   │ ├── fastsam # fast SAM 模型
   │ ├── nas # NAS 模型
   │ ├── rtdetr # RT-DETR 模型
   │ ├── sam # SAM 模型
   │ │ └── modules # SAM中用到的网络结构的实现，encoder、decoder等
   │ ├── utils # 各网络中公用的基础组件
   │ └── yolo # YOLO 模型
   │ ├── classify # 分类任务
   │ ├── detect # 检测任务
   │ ├── obb # Oriented Bounding Box 3D包围盒
   │ ├── pose # 姿态估计
   │ └── segment # 分割任务
   ├── nn # --- 公用的神经网络组件 ---
   │ └── modules # 公用的基础网络结构的实现
   ├── solutions # --- 一些偏专用的解决方法 ---
   ├── trackers # ？
   │ └── utils # 该任务中公用的基础组件
   └── utils # --- 在架构各阶段用到的公用的基础组件 ---
   └── callbacks # ？ 回调函数、可视化等基础组件

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/events/yolovision" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLO Vision banner"></a>
  </p>

[中文](https://docs.ultralytics.com/zh) | [한국어](https://docs.ultralytics.com/ko) | [日本語](https://docs.ultralytics.com/ja) | [Русский](https://docs.ultralytics.com/ru) | [Deutsch](https://docs.ultralytics.com/de) | [Français](https://docs.ultralytics.com/fr) | [Español](https://docs.ultralytics.com/es) | [Português](https://docs.ultralytics.com/pt) | [Türkçe](https://docs.ultralytics.com/tr) | [Tiếng Việt](https://docs.ultralytics.com/vi) | [العربية](https://docs.ultralytics.com/ar) <br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://pepy.tech/projects/ultralytics"><img src="https://static.pepy.tech/badge/ultralytics" alt="Ultralytics Downloads"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="Ultralytics YOLO Citation"></a>
    <a href="https://discord.com/invite/ultralytics"><img alt="Ultralytics Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <a href="https://reddit.com/r/ultralytics"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run Ultralytics on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Ultralytics In Colab"></a>
    <a href="https://www.kaggle.com/models/ultralytics/yolo11"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open Ultralytics In Kaggle"></a>
    <a href="https://mybinder.org/v2/gh/ultralytics/ultralytics/HEAD?labpath=examples%2Ftutorial.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Open Ultralytics In Binder"></a>
</div>
<br>

[Ultralytics](https://www.ultralytics.com/) [YOLO11](https://github.com/ultralytics/ultralytics) is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLO11 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection and tracking, instance segmentation, image classification and pose estimation tasks.

We hope that the resources here will help you get the most out of YOLO. Please browse the Ultralytics <a href="https://docs.ultralytics.com/">Docs</a> for details, raise an issue on <a href="https://github.com/ultralytics/ultralytics/issues/new/choose">GitHub</a> for support, questions, or discussions, become a member of the Ultralytics <a href="https://discord.com/invite/ultralytics">Discord</a>, <a href="https://reddit.com/r/ultralytics">Reddit</a> and <a href="https://community.ultralytics.com/">Forums</a>!

To request an Enterprise License please complete the form at [Ultralytics Licensing](https://www.ultralytics.com/license).

<a href="https://docs.ultralytics.com/models/yolo11/" target="_blank">
  <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png" alt="YOLO11 performance plots">
</a>

<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="2%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="2%" alt="Ultralytics Discord"></a>
</div>
</div>

## <div align="center">Documentation</div>

See below for a quickstart install and usage examples, and see our [Docs](https://docs.ultralytics.com/) for full documentation on training, validation, prediction and deployment.

<details open>
<summary>Install</summary>

Pip install the Ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/) [![Ultralytics Downloads](https://static.pepy.tech/badge/ultralytics)](https://www.pepy.tech/projects/ultralytics) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

```bash
pip install ultralytics
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/ultralytics), [Docker](https://hub.docker.com/r/ultralytics/ultralytics), and Git, please refer to the [Quickstart Guide](https://docs.ultralytics.com/quickstart/).

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics) [![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics) [![Ultralytics Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)

</details>

<details open>
<summary>Usage</summary>

### CLI

YOLO may be used directly in the Command Line Interface (CLI) with a `yolo` command:

```bash
yolo predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` can be used for a variety of tasks and modes and accepts additional arguments, e.g. `imgsz=640`. See the YOLO [CLI Docs](https://docs.ultralytics.com/usage/cli/) for examples.

### Python

YOLO may also be used directly in a Python environment, and accepts the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
```

See YOLO [Python Docs](https://docs.ultralytics.com/usage/python/) for more examples.

</details>

## <div align="center">Models</div>

YOLO11 [Detect](https://docs.ultralytics.com/tasks/detect/), [Segment](https://docs.ultralytics.com/tasks/segment/) and [Pose](https://docs.ultralytics.com/tasks/pose/) models pretrained on the [COCO](https://docs.ultralytics.com/datasets/detect/coco/) dataset are available here, as well as YOLO11 [Classify](https://docs.ultralytics.com/tasks/classify/) models pretrained on the [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/) dataset. [Track](https://docs.ultralytics.com/modes/track/) mode is available for all Detect, Segment and Pose models. All [Models](https://docs.ultralytics.com/models/) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

<a href="https://docs.ultralytics.com/tasks/" target="_blank">
    <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-tasks-banner.avif" alt="Ultralytics YOLO supported tasks">
</a>
<br>
<br>

<details open><summary>Detection (COCO)</summary>

See [Detection Docs](https://docs.ultralytics.com/tasks/detect/) for usage examples with these models trained on [COCO](https://docs.ultralytics.com/datasets/detect/coco/), which include 80 pre-trained classes.

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | 640                   | 39.5                 | 56.1 ± 0.8                     | 1.5 ± 0.0                           | 2.6                | 6.5               |
| [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) | 640                   | 47.0                 | 90.0 ± 1.2                     | 2.5 ± 0.0                           | 9.4                | 21.5              |
| [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt) | 640                   | 51.5                 | 183.2 ± 2.0                    | 4.7 ± 0.1                           | 20.1               | 68.0              |
| [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) | 640                   | 53.4                 | 238.6 ± 1.4                    | 6.2 ± 0.1                           | 25.3               | 86.9              |
| [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) | 640                   | 54.7                 | 462.8 ± 6.7                    | 11.3 ± 0.2                          | 56.9               | 194.9             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val detect data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val detect data=coco.yaml batch=1 device=0|cpu`

</details>

<details><summary>Segmentation (COCO)</summary>

See [Segmentation Docs](https://docs.ultralytics.com/tasks/segment/) for usage examples with these models trained on [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/), which include 80 pre-trained classes.

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLO11n-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) | 640                   | 38.9                 | 32.0                  | 65.9 ± 1.1                     | 1.8 ± 0.0                           | 2.9                | 10.4              |
| [YOLO11s-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) | 640                   | 46.6                 | 37.8                  | 117.6 ± 4.9                    | 2.9 ± 0.0                           | 10.1               | 35.5              |
| [YOLO11m-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) | 640                   | 51.5                 | 41.5                  | 281.6 ± 1.2                    | 6.3 ± 0.1                           | 22.4               | 123.3             |
| [YOLO11l-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt) | 640                   | 53.4                 | 42.9                  | 344.2 ± 3.2                    | 7.8 ± 0.2                           | 27.6               | 142.2             |
| [YOLO11x-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) | 640                   | 54.7                 | 43.8                  | 664.5 ± 3.2                    | 15.8 ± 0.7                          | 62.1               | 319.0             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val segment data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val segment data=coco.yaml batch=1 device=0|cpu`

</details>

<details><summary>Classification (ImageNet)</summary>

See [Classification Docs](https://docs.ultralytics.com/tasks/classify/) for usage examples with these models trained on [ImageNet](https://docs.ultralytics.com/datasets/classify/imagenet/), which include 1000 pretrained classes.

| Model                                                                                        | size<br><sup>(pixels) | acc<br><sup>top1 | acc<br><sup>top5 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) at 640 |
| -------------------------------------------------------------------------------------------- | --------------------- | ---------------- | ---------------- | ------------------------------ | ----------------------------------- | ------------------ | ------------------------ |
| [YOLO11n-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt) | 224                   | 70.0             | 89.4             | 5.0 ± 0.3                      | 1.1 ± 0.0                           | 1.6                | 3.3                      |
| [YOLO11s-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-cls.pt) | 224                   | 75.4             | 92.7             | 7.9 ± 0.2                      | 1.3 ± 0.0                           | 5.5                | 12.1                     |
| [YOLO11m-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-cls.pt) | 224                   | 77.3             | 93.9             | 17.2 ± 0.4                     | 2.0 ± 0.0                           | 10.4               | 39.3                     |
| [YOLO11l-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-cls.pt) | 224                   | 78.3             | 94.3             | 23.2 ± 0.3                     | 2.8 ± 0.0                           | 12.9               | 49.4                     |
| [YOLO11x-cls](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-cls.pt) | 224                   | 79.5             | 94.9             | 41.4 ± 0.9                     | 3.8 ± 0.0                           | 28.4               | 110.4                    |

- **acc** values are model accuracies on the [ImageNet](https://www.image-net.org/) dataset validation set. <br>Reproduce by `yolo val classify data=path/to/ImageNet device=0`
- **Speed** averaged over ImageNet val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val classify data=path/to/ImageNet batch=1 device=0|cpu`

</details>

<details><summary>Pose (COCO)</summary>

See [Pose Docs](https://docs.ultralytics.com/tasks/pose/) for usage examples with these models trained on [COCO-Pose](https://docs.ultralytics.com/datasets/pose/coco/), which include 1 pre-trained class, person.

| Model                                                                                          | size<br><sup>(pixels) | mAP<sup>pose<br>50-95 | mAP<sup>pose<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ---------------------------------------------------------------------------------------------- | --------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLO11n-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt) | 640                   | 50.0                  | 81.0               | 52.4 ± 0.5                     | 1.7 ± 0.0                           | 2.9                | 7.6               |
| [YOLO11s-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt) | 640                   | 58.9                  | 86.3               | 90.5 ± 0.6                     | 2.6 ± 0.0                           | 9.9                | 23.2              |
| [YOLO11m-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt) | 640                   | 64.9                  | 89.4               | 187.3 ± 0.8                    | 4.9 ± 0.1                           | 20.9               | 71.7              |
| [YOLO11l-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-pose.pt) | 640                   | 66.1                  | 89.9               | 247.7 ± 1.1                    | 6.4 ± 0.1                           | 26.2               | 90.7              |
| [YOLO11x-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt) | 640                   | 69.5                  | 91.1               | 488.0 ± 13.9                   | 12.1 ± 0.2                          | 58.8               | 203.3             |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO Keypoints val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val pose data=coco-pose.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val pose data=coco-pose.yaml batch=1 device=0|cpu`

</details>

<details><summary>OBB (DOTAv1)</summary>

See [OBB Docs](https://docs.ultralytics.com/tasks/obb/) for usage examples with these models trained on [DOTAv1](https://docs.ultralytics.com/datasets/obb/dota-v2/#dota-v10/), which include 15 pre-trained classes.

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>test<br>50 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>T4 TensorRT10<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | ------------------ | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLO11n-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt) | 1024                  | 78.4               | 117.6 ± 0.8                    | 4.4 ± 0.0                           | 2.7                | 17.2              |
| [YOLO11s-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt) | 1024                  | 79.5               | 219.4 ± 4.0                    | 5.1 ± 0.0                           | 9.7                | 57.5              |
| [YOLO11m-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-obb.pt) | 1024                  | 80.9               | 562.8 ± 2.9                    | 10.1 ± 0.4                          | 20.9               | 183.5             |
| [YOLO11l-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-obb.pt) | 1024                  | 81.0               | 712.5 ± 5.0                    | 13.5 ± 0.6                          | 26.2               | 232.0             |
| [YOLO11x-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-obb.pt) | 1024                  | 81.3               | 1408.6 ± 7.7                   | 28.6 ± 1.0                          | 58.8               | 520.2             |

- **mAP<sup>test</sup>** values are for single-model multiscale on [DOTAv1](https://captain-whu.github.io/DOTA/index.html) dataset. <br>Reproduce by `yolo val obb data=DOTAv1.yaml device=0 split=test` and submit merged results to [DOTA evaluation](https://captain-whu.github.io/DOTA/evaluation.html).
- **Speed** averaged over DOTAv1 val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val obb data=DOTAv1.yaml batch=1 device=0|cpu`

</details>

## <div align="center">Integrations</div>

Our key integrations with leading AI platforms extend the functionality of Ultralytics' offerings, enhancing tasks like dataset labeling, training, visualization, and model management. Discover how Ultralytics, in collaboration with [W&B](https://docs.wandb.ai/guides/integrations/ultralytics/), [Comet](https://bit.ly/yolov8-readme-comet), [Roboflow](https://roboflow.com/?ref=ultralytics) and [OpenVINO](https://docs.ultralytics.com/integrations/openvino/), can optimize your AI workflow.

<a href="https://www.ultralytics.com/hub" target="_blank">
    <img width="100%" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics active learning integrations">
</a>
<br>
<br>

<div align="center">
  <a href="https://www.ultralytics.com/hub">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-ultralytics-hub.png" width="10%" alt="Ultralytics HUB logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://docs.wandb.ai/guides/integrations/ultralytics/">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-wb.png" width="10%" alt="ClearML logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://bit.ly/yolov8-readme-comet">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-comet.png" width="10%" alt="Comet ML logo"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="15%" height="0" alt="space">
  <a href="https://bit.ly/yolov5-neuralmagic">
    <img src="https://github.com/ultralytics/assets/raw/main/partners/logo-neuralmagic.png" width="10%" alt="NeuralMagic logo"></a>
</div>

|                                                         Ultralytics HUB 🚀                                                         |                                                               W&B                                                               |                                                                       Comet ⭐ NEW                                                                        |                                              Neural Magic                                              |
| :--------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| Streamline YOLO workflows: Label, train, and deploy effortlessly with [Ultralytics HUB](https://www.ultralytics.com/hub). Try now! | Track experiments, hyperparameters, and results with [Weights & Biases](https://docs.wandb.ai/guides/integrations/ultralytics/) | Free forever, [Comet](https://bit.ly/yolov5-readme-comet) lets you save YOLO11 models, resume training, and interactively visualize and debug predictions | Run YOLO11 inference up to 6x faster with [Neural Magic DeepSparse](https://bit.ly/yolov5-neuralmagic) |

## <div align="center">Ultralytics HUB</div>

Experience seamless AI with [Ultralytics HUB](https://www.ultralytics.com/hub) ⭐, the all-in-one solution for data visualization, YOLO11 🚀 model training and deployment, without any coding. Transform images into actionable insights and bring your AI visions to life with ease using our cutting-edge platform and user-friendly [Ultralytics App](https://www.ultralytics.com/app-install). Start your journey for **Free** now!

<a href="https://www.ultralytics.com/hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png" alt="Ultralytics HUB preview image"></a>

## <div align="center">Contribute</div>

We love your input! Ultralytics YOLO would not be possible without help from our community. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) to get started, and fill out our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) to send us feedback on your experience. Thank you 🙏 to all our contributors!

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->

<a href="https://github.com/ultralytics/ultralytics/graphs/contributors">
<img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" alt="Ultralytics open-source contributors"></a>

## <div align="center">License</div>

Ultralytics offers two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license) open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details.
- **Enterprise License**: Designed for commercial use, this license permits seamless integration of Ultralytics software and AI models into commercial goods and services, bypassing the open-source requirements of AGPL-3.0. If your scenario involves embedding our solutions into a commercial offering, reach out through [Ultralytics Licensing](https://www.ultralytics.com/license).

## <div align="center">Contact</div>

For Ultralytics bug reports and feature requests please visit [GitHub Issues](https://github.com/ultralytics/ultralytics/issues). Become a member of the Ultralytics [Discord](https://discord.com/invite/ultralytics), [Reddit](https://www.reddit.com/r/ultralytics/), or [Forums](https://community.ultralytics.com/) for asking questions, sharing projects, learning discussions, or for help with all things Ultralytics!

<br>
<div align="center">
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
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
