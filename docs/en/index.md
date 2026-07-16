---
title: YOLO Object Detection & Segmentation
comments: true
description: Discover Ultralytics YOLO - the latest in real-time object detection and image segmentation. Learn about its features and maximize its potential in your projects.
keywords: Ultralytics, YOLO, YOLO26, YOLO11, object detection, image segmentation, deep learning, computer vision, AI, machine learning, documentation, tutorial
---

<div align="center">
<br><br>
<a href="https://platform.ultralytics.com/ultralytics/yolo26?utm_source=docs&utm_medium=referral&utm_campaign=platform_launch&utm_content=banner&utm_term=ultralytics_docs" target="_blank"><img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
<br><br>
</div>

<p align="center">
<a href="https://docs.ultralytics.com/zh">中文</a> ·
<a href="https://docs.ultralytics.com/ko">한국어</a> ·
<a href="https://docs.ultralytics.com/ja">日本語</a> ·
<a href="https://docs.ultralytics.com/ru">Русский</a> ·
<a href="https://docs.ultralytics.com/de">Deutsch</a> ·
<a href="https://docs.ultralytics.com/fr">Français</a> ·
<a href="https://docs.ultralytics.com/es">Español</a> ·
<a href="https://docs.ultralytics.com/pt">Português</a> ·
<a href="https://docs.ultralytics.com/tr">Türkçe</a> ·
<a href="https://docs.ultralytics.com/vi">Tiếng Việt</a> ·
<a href="https://docs.ultralytics.com/ar">العربية</a>
</p>

<div align="center">
<br>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://clickpy.clickhouse.com/dashboard/ultralytics"><img src="https://static.pepy.tech/badge/ultralytics" alt="Ultralytics Downloads"></a>
    <a href="https://discord.com/invite/ultralytics"><img alt="Ultralytics Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a>
    <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a>
    <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run Ultralytics on Gradient"></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Ultralytics In Colab"></a>
    <a href="https://www.kaggle.com/models/ultralytics/yolo26"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open Ultralytics In Kaggle"></a>
    <a href="https://mybinder.org/v2/gh/ultralytics/ultralytics/HEAD?labpath=examples%2Ftutorial.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Open Ultralytics In Binder"></a>
<br><br>
</div>

# Ultralytics YOLO Docs

[Ultralytics](https://www.ultralytics.com) YOLO is a family of real-time computer vision models for object detection, instance segmentation, semantic segmentation, classification, pose estimation, oriented bounding boxes, and tracking, available through one Python package and CLI. YOLO26 is built on deep learning and computer vision advancements, featuring end-to-end NMS-free inference and optimized edge deployment. Its streamlined design makes it suitable for various applications and easily adaptable to different hardware platforms, from edge devices to cloud APIs. For stable production workloads, both [YOLO26](models/yolo26.md) and [YOLO11](models/yolo11.md) are recommended.

Explore the Ultralytics Docs, a comprehensive resource covering the YOLO package and CLI as well as the [Ultralytics Platform](platform/index.md), which adds data annotation, cloud training, and deployment on top of the same models. Whether you are a seasoned machine learning practitioner or new to the field, this hub aims to help you get the most out of YOLO in your projects.

Request an Enterprise License for commercial use at [Ultralytics Licensing](https://www.ultralytics.com/license?utm_source=docs.ultralytics.com&utm_medium=referral&utm_content=license_inline_link).

!!! tip "🚀 New: Knowledge Distillation"

    Train smaller YOLO models with guidance from a larger teacher model — no extra inference cost, just better accuracy.

    [:octicons-arrow-right-24: Learn more](guides/knowledge-distillation.md)

## Get Started in Two Commands

```bash
# Install the ultralytics package from PyPI
pip install ultralytics

# Detect objects in an image with a pretrained YOLO26 model
yolo predict model=yolo26n.pt source='https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg'
```

The model weights and the example image download automatically, and the annotated result is saved to `runs/detect/predict`.

See the [Quickstart](quickstart.md) guide for the full installation and usage reference.

## What Do You Want to Do?

<div class="grid cards" markdown>

- :fontawesome-solid-brain:{ .lg .middle } &nbsp; **Train a model on your own dataset**

    ***

    Fine-tune a pretrained YOLO26 model on your own dataset, tuning augmentation and hyperparameters for multi-GPU training

    ***

    [:octicons-arrow-right-24: Train a custom model](modes/train.md)

- :material-image:{ .lg .middle } &nbsp; **Run a model on your images or video**

    ***

    Load a pretrained model and get bounding boxes, masks, or keypoints in a few lines of Python or a single CLI command

    ***

    [:octicons-arrow-right-24: Predict on new data](modes/predict.md)

- :material-play-circle:{ .lg .middle } &nbsp; **Track objects across video frames**

    ***

    Track objects across video frames with a persistent ID using BoT-SORT or ByteTrack, built into YOLO26's predict pipeline

    ***

    [:octicons-arrow-right-24: Multi-object tracking](modes/track.md)

- :material-apps:{ .lg .middle } &nbsp; **Run a ready-made vision application**

    ***

    Ready-made vision apps for object counting, heatmaps, queue management, security alarms, and workouts, no training required

    ***

    [:octicons-arrow-right-24: Explore Solutions](solutions/index.md)

- :material-rocket-launch:{ .lg .middle } &nbsp; **Deploy your model**

    ***

    Export trained models to ONNX, TensorRT, or OpenVINO for fast inference on edge devices, mobile hardware, and cloud servers

    ***

    [:octicons-arrow-right-24: Export and deploy](modes/export.md)

- :material-magnify-expand:{ .lg .middle } &nbsp; **Pick the right model**

    ***

    Compare YOLO26, YOLO11, SAM 3, RT-DETR, and every other supported architecture by speed, accuracy, and use case

    ***

    [:octicons-arrow-right-24: Browse all models](models/index.md)

- :material-book-open-variant:{ .lg .middle } &nbsp; **Look up the Python API**

    ***

    Look up classes, functions, and method signatures for the Python API, auto-generated from source on every new release

    ***

    [:octicons-arrow-right-24: API Reference](reference/index.md)

- :rocket:{ .lg .middle } &nbsp; **What's new: YOLO26**

    ***

    Ultralytics' newest model family delivers NMS-free, end-to-end inference with higher accuracy and lower latency than YOLO11

    ***

    [:octicons-arrow-right-24: Meet YOLO26](models/yolo26.md)

</div>

## How These Docs Are Organized

Every `yolo` command follows one grammar, `yolo TASK MODE ARGS`, and these docs are organized around the same three parts, plus one shortcut:

- **[Task](tasks/index.md)** answers _what_ you want from an image: [detection](tasks/detect.md), [instance segmentation](tasks/segment.md), [semantic segmentation](tasks/semantic.md), [classification](tasks/classify.md), [pose estimation](tasks/pose.md), or [oriented boxes](tasks/obb.md).
- **[Mode](modes/index.md)** answers _how_ you use a model: [train](modes/train.md), [validate](modes/val.md), [predict](modes/predict.md), [export](modes/export.md), [track](modes/track.md), or [benchmark](modes/benchmark.md).
- **[Models](models/index.md)** answers _which_ network runs it: YOLO26, YOLO11, SAM 3, RT-DETR, and every other supported architecture.
- **[Solutions](solutions/index.md)** is the shortcut: a finished application, like object counting or a security alarm, that skips Task and Mode entirely.

Everything else supports that grammar: [Datasets](datasets/index.md) supplies what each Task trains on, [Guides](guides/index.md) is a broad collection of in-depth how-tos spanning hardware deployment, hyperparameter tuning, dataset conversion, and full project walkthroughs, [Integrations](integrations/index.md) connects the pipeline to the training and deployment tools you already use, and the [Reference](reference/index.md) section documents every class and function in the Python API.

Beyond the Python package, two more surfaces run on the same models: the [Ultralytics Platform](platform/index.md) for cloud annotation, training, and deployment, and [Ultralytics Inference](inference/index.md), a standalone Rust library and CLI for running exported models without a Python runtime.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/7lZa3Yi2kbo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train a YOLO26 model on Your Custom Dataset in <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb" target="_blank">Google Colab</a>.
</p>

## YOLO Licenses: How is Ultralytics YOLO licensed?

<a href="https://www.ultralytics.com/license?utm_source=docs.ultralytics.com&utm_medium=referral&utm_content=license_banner" target="_blank" rel="noopener noreferrer">
<img width="100%" style="border-radius:.4rem" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-license.avif" alt="Ultralytics Enterprise License banner"></a>

Ultralytics offers two licensing options to accommodate diverse use cases:

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license/agpl-3.0) open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details.
- **Enterprise License**: For development and production use, this license enables seamless integration of Ultralytics software and AI models into business products and services, including internal tools, automated workflows, and production deployments, bypassing the open-source requirements of AGPL-3.0. To get started, please contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

Our licensing strategy is designed to ensure that any improvements to our open-source projects are returned to the community. We believe in open source, and our mission is to ensure that our contributions can be used and expanded in ways that benefit everyone.

## FAQ

### What is Ultralytics YOLO and how does it improve object detection?

Ultralytics YOLO is the acclaimed YOLO (You Only Look Once) series for real-time object detection and image segmentation. The latest model, [YOLO26](models/yolo26.md), builds on previous versions by introducing end-to-end NMS-free inference and optimized edge deployment. YOLO supports various [vision AI tasks](tasks/index.md) such as [detection](tasks/detect.md), [instance segmentation](tasks/segment.md), [semantic segmentation](tasks/semantic.md), [pose estimation](tasks/pose.md), [tracking](modes/track.md), and [classification](tasks/classify.md). Its efficient architecture ensures excellent speed and accuracy, making it suitable for diverse applications, including edge devices and cloud APIs.

### How can I get started with YOLO installation and setup?

Getting started with YOLO is quick and straightforward. Install the Ultralytics package from [pip](https://pypi.org/project/ultralytics/) with `pip install ultralytics`, then run your first prediction with `yolo predict model=yolo26n.pt` — the model weights download automatically. For comprehensive instructions covering conda, Docker, and installation from source, visit the [Quickstart](quickstart.md) page.

### How can I train a custom YOLO model on my dataset?

Training a custom YOLO model on your dataset involves a few detailed steps:

1. Prepare your annotated dataset and describe it in a dataset YAML file.
2. Load a pretrained model, for example `YOLO("yolo26n.pt")` in Python.
3. Start training with `model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)`, or from the command line with `yolo detect train data=path/to/dataset.yaml epochs=100 imgsz=640`.

For a detailed walkthrough, check out our [Train a Model](modes/train.md) guide, which includes examples and tips for optimizing your training process.

### What are the licensing options available for Ultralytics YOLO?

Ultralytics offers two licensing options for YOLO:

- **AGPL-3.0 License**: This open-source license is ideal for educational and non-commercial use, promoting open collaboration.
- **Enterprise License**: For development and production use, including internal tools, automated workflows, and production deployments, bypassing the open-source requirements of AGPL-3.0.

For more details, visit our [Licensing](https://www.ultralytics.com/license) page.

### How can Ultralytics YOLO be used for real-time object tracking?

Ultralytics YOLO supports efficient and customizable multi-object tracking. Call `model.track(source="path/to/video.mp4")` in Python, or run `yolo track source=path/to/video.mp4` from the command line — both work with video files, live streams, and webcam input. For a detailed guide on setting up and running object tracking, check our [Track Mode](modes/track.md) documentation, which explains the configuration and practical applications in real-time scenarios.

<div align="center">
  <br>
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img width="3%" src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" alt="">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
