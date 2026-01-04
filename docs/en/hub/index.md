---
comments: true
description: Discover Ultralytics HUB, the all-in-one web tool for training and deploying YOLO models. Get started quickly with pre-trained models and user-friendly features.
keywords: Ultralytics HUB, YOLO models, train YOLO, YOLOv5, YOLOv8, YOLO11, object detection, model deployment, machine learning, deep learning, AI tools, dataset upload, model training
---

# Ultralytics HUB

<div align="center">
<a href="https://www.ultralytics.com/hub" target="_blank"><img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub.avif" alt="Ultralytics HUB banner"></a>
<a href="https://docs.ultralytics.com/zh/hub/">中文</a> |
<a href="https://docs.ultralytics.com/ko/hub/">한국어</a> |
<a href="https://docs.ultralytics.com/ja/hub/">日本語</a> |
<a href="https://docs.ultralytics.com/ru/hub/">Русский</a> |
<a href="https://docs.ultralytics.com/de/hub/">Deutsch</a> |
<a href="https://docs.ultralytics.com/fr/hub/">Français</a> |
<a href="https://docs.ultralytics.com/es/hub/">Español</a> |
<a href="https://docs.ultralytics.com/pt/hub/">Português</a> |
<a href="https://docs.ultralytics.com/tr/hub/">Türkçe</a> |
<a href="https://docs.ultralytics.com/vi/hub/">Tiếng Việt</a> |
<a href="https://docs.ultralytics.com/ar/hub/">العربية</a>
<br>
<br>

<a href="https://github.com/ultralytics/hub/actions/workflows/ci.yml"><img src="https://github.com/ultralytics/hub/actions/workflows/ci.yml/badge.svg" alt="CI CPU"></a> <a href="https://colab.research.google.com/github/ultralytics/hub/blob/main/hub.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://discord.com/invite/ultralytics"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a> <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a> <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>

</div>

👋 Hello from the [Ultralytics](https://www.ultralytics.com/) Team! We've been working hard these last few months to launch [Ultralytics HUB](https://www.ultralytics.com/hub), a new web tool for training and deploying all your YOLOv5, YOLOv8, and YOLO11 🚀 models from one spot!

We hope that the resources here will help you get the most out of HUB. Please browse the HUB <a href="https://docs.ultralytics.com/">Docs</a> for details, raise an issue on <a href="https://github.com/ultralytics/hub/issues/new/choose">GitHub</a> for support, and join our <a href="https://discord.com/invite/ultralytics">Discord</a> community for questions and discussions!

<div align="center">
  <br>
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>

## Introduction

[Ultralytics HUB](https://www.ultralytics.com/hub) is designed to be user-friendly and intuitive, allowing users to quickly upload their datasets and train new YOLO models. It also offers a range of pre-trained models to choose from, making it extremely easy for users to get started. Once a model is trained, it can be effortlessly previewed in the [Ultralytics HUB App](app/index.md) before being deployed for real-time classification, [object detection](https://www.ultralytics.com/glossary/object-detection), and [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) tasks.

<p align="center">
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/qE-dfbB5Sis"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to train Ultralytics YOLO11 on Custom Dataset using Ultralytics HUB | HUB Datasets 🚀
</p>

We hope that the resources here will help you get the most out of HUB. Please browse the HUB <a href="https://docs.ultralytics.com/hub/">Docs</a> for details, raise an issue on <a href="https://github.com/ultralytics/hub/issues/new/choose">GitHub</a> for support, and join our <a href="https://discord.com/invite/ultralytics">Discord</a> community for questions and discussions!

- [**Quickstart**](quickstart.md): Start training and deploying models in seconds.
- [**Datasets**](datasets.md): Learn how to prepare and upload your datasets.
- [**Projects**](projects.md): Group your models into projects for improved organization.
- [**Models**](models.md): Train models and export them to various formats for deployment.
- [**Pro**](pro.md): Level up your experience by becoming a Pro user.
- [**Cloud Training**](cloud-training.md): Understand how to train models using our Cloud Training solution.
- [**Inference API**](inference-api.md): Understand how to use our Inference API.
- [**Teams**](teams.md): Collaborate effortlessly with your team.
- [**Integrations**](integrations.md): Explore different integration options.
- [**Ultralytics HUB App**](app/index.md): Learn about the Ultralytics HUB App, which allows you to run models directly on your mobile device.
    - [**iOS**](app/ios.md): Explore CoreML acceleration on iPhones and iPads.
    - [**Android**](app/android.md): Explore TFLite acceleration on Android devices.

## FAQ

### How do I get started with Ultralytics HUB for training YOLO models?

To get started with [Ultralytics HUB](https://www.ultralytics.com/hub), follow these steps:

1. **Sign Up:** Create an account on [Ultralytics HUB](https://www.ultralytics.com/hub).
2. **Upload Dataset:** Navigate to the [Datasets](datasets.md) section to upload your custom dataset.
3. **Train Model:** Go to the [Models](models.md) section and select a pre-trained YOLOv5, YOLOv8, or YOLO11 model to start training.
4. **Deploy Model:** Once trained, preview and deploy your model using the [Ultralytics HUB App](app/index.md) for real-time tasks.

For a detailed guide, refer to the [Quickstart](quickstart.md) page.

### What are the benefits of using Ultralytics HUB over other AI platforms?

[Ultralytics HUB](https://www.ultralytics.com/hub) offers several unique benefits:

- **User-Friendly Interface:** Intuitive design for easy dataset uploads and model training.
- **Pre-Trained Models:** Access to a variety of pre-trained YOLO models including YOLOv5, YOLOv8, and YOLO11.
- **Cloud Training:** Seamless cloud training capabilities, detailed on the [Cloud Training](cloud-training.md) page.
- **Real-Time Deployment:** Effortlessly deploy models for real-time applications using the [Ultralytics HUB App](app/index.md).
- **Team Collaboration:** Collaborate with your team efficiently through the [Teams](teams.md) feature.
- **No-Code Solution:** Train and deploy advanced computer vision models without writing a single line of code.

Explore more about the advantages in our [Ultralytics HUB Blog](https://www.ultralytics.com/blog/ultralytics-hub-a-game-changer-for-computer-vision).

### Can I use Ultralytics HUB for object detection on mobile devices?

Yes, Ultralytics HUB supports object detection on mobile devices. You can run YOLO models on both iOS and Android devices using the Ultralytics HUB App. For more details:

- **iOS:** Learn about CoreML acceleration on iPhones and iPads in the [iOS](app/ios.md) section.
- **Android:** Explore TFLite acceleration on Android devices in the [Android](app/android.md) section.

### How do I manage and organize my projects in Ultralytics HUB?

Ultralytics HUB allows you to manage and organize your projects efficiently. You can group your models into projects for better organization. To learn more:

- Visit the [Projects](projects.md) page for detailed instructions on creating, editing, and managing projects.
- Use the [Teams](teams.md) feature to collaborate with team members and share resources.

### What integrations are available with Ultralytics HUB?

Ultralytics HUB offers seamless integrations with various platforms to enhance your [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) workflows. Some key integrations include:

- **Roboflow:** For dataset management and model training. Learn more on the [Integrations](integrations.md) page.
- **Google Colab:** Efficiently train models using Google Colab's cloud-based environment. Detailed steps are available in the [Google Colab](https://docs.ultralytics.com/integrations/google-colab/) section.
- **Weights & Biases:** For enhanced experiment tracking and visualization. Explore the [Weights & Biases](https://docs.ultralytics.com/integrations/weights-biases/) integration.

For a complete list of integrations, refer to the [Integrations](integrations.md) page.
