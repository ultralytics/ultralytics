---
comments: true
description: Gain insights into training and deploying your YOLOv5 and YOLOv8 models with Ultralytics HUB. Explore pre-trained models, templates and various integrations.
keywords: Ultralytics HUB, YOLOv5, YOLOv8, model training, model deployment, pretrained models, model integrations
---

# Ultralytics HUB

<a href="https://bit.ly/ultralytics_hub" target="_blank">
  <img width="100%" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png" alt="Ultralytics HUB preview image"></a>
<br>
<div align="center">
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
  <a href="https://github.com/ultralytics/hub/actions/workflows/ci.yaml">
    <img src="https://github.com/ultralytics/hub/actions/workflows/ci.yaml/badge.svg" alt="CI CPU"></a>
  <a href="https://colab.research.google.com/github/ultralytics/hub/blob/master/hub.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

👋 Hello from the [Ultralytics](https://ultralytics.com/) Team! We've been working hard these last few months to launch [Ultralytics HUB](https://bit.ly/ultralytics_hub), a new web tool for training and deploying all your YOLOv5 and YOLOv8 🚀 models from one spot!

## Introduction

HUB is designed to be user-friendly and intuitive, with a drag-and-drop interface that allows users to easily upload their data and train new models quickly. It offers a range of pre-trained models and templates to choose from, making it easy for users to get started with training their own models. Once a model is trained, it can be easily deployed and used for real-time object detection, instance segmentation and classification tasks.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/lveF9iCMIzc?si=_Q4WB5kMB5qNe7q6"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Train Your Custom YOLO Models In A Few Clicks with Ultralytics HUB.
</p>

We hope that the resources here will help you get the most out of HUB. Please browse the HUB <a href="https://docs.ultralytics.com/hub">Docs</a> for details, raise an issue on <a href="https://github.com/ultralytics/hub/issues/new/choose">GitHub</a> for support, and join our <a href="https://ultralytics.com/discord">Discord</a> community for questions and discussions!

- [**Quickstart**](quickstart.md). Start training and deploying YOLO models with HUB in seconds.
- [**Datasets: Preparing and Uploading**](datasets.md). Learn how to prepare and upload your datasets to HUB in YOLO format.
- [**Projects: Creating and Managing**](projects.md). Group your models into projects for improved organization.
- [**Models: Training and Exporting**](models.md). Train YOLOv5 and YOLOv8 models on your custom datasets and export them to various formats for deployment.
- [**Integrations: Options**](integrations.md). Explore different integration options for your trained models, such as TensorFlow, ONNX, OpenVINO, CoreML, and PaddlePaddle.
- [**Ultralytics HUB App**](app/index.md). Learn about the Ultralytics App for iOS and Android, which allows you to run models directly on your mobile device.
    - [**iOS**](app/ios.md). Learn about YOLO CoreML models accelerated on Apple's Neural Engine on iPhones and iPads.
    - [**Android**](app/android.md). Explore TFLite acceleration on mobile devices.
- [**Inference API**](inference-api.md). Understand how to use the Inference API for running your trained models in the cloud to generate predictions.
