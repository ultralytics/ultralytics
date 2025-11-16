---
comments: true
description: Discover the Ultralytics HUB App for running YOLO models on iOS and Android devices with hardware acceleration for real-time object detection.
keywords: Ultralytics HUB App, YOLO models, mobile app, iOS, Android, hardware acceleration, YOLOv5, YOLOv8, YOLO11, neural engine, GPU, NNAPI, real-time object detection
---

# Ultralytics HUB App

<a href="https://www.ultralytics.com/hub" target="_blank">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-hub.avif" alt="Ultralytics HUB preview image"></a>
<br>
<div align="center">
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
  <br>
  <br>
  <a href="https://apps.apple.com/xk/app/ultralytics-hub/id1583935240" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/master/app/app-store.svg" width="15%" alt="Apple App store"></a>
  <a href="https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/master/app/google-play.svg" width="15%" alt="Google Play store"></a>&nbsp;
</div>

Welcome to the Ultralytics HUB App! This powerful mobile application allows you to run Ultralytics YOLO models including YOLOv5, YOLOv8, and YOLO11 directly on your [iOS](https://apps.apple.com/xk/app/ultralytics-hub/id1583935240) and [Android](https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app) devices. The HUB App leverages hardware acceleration features like Apple's Neural Engine (ANE) on iOS or Android GPU and [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) API (NNAPI) delegates to deliver impressive real-time performance on your mobile device.

## Features

- **Run Ultralytics YOLO Models**: Experience the power of YOLO models on your mobile device for real-time [object detection](https://www.ultralytics.com/glossary/object-detection), [image segmentation](https://www.ultralytics.com/glossary/image-segmentation), and [image recognition](https://www.ultralytics.com/glossary/image-recognition) tasks.
- **Hardware Acceleration**: Benefit from Apple ANE on iOS devices or Android GPU and NNAPI delegates for optimized performance and efficiency.
- **Custom Model Training**: Train custom models with the [Ultralytics HUB platform](https://www.ultralytics.com/hub) and preview them live using the HUB App.
- **Mobile Compatibility**: The HUB App supports both iOS and Android devices, bringing the power of YOLO models to a wide range of users.
- **Real-time Performance**: Achieve impressive inference speeds of up to 30 frames per second on modern devices.
- **Model Quantization**: Models are optimized with FP16 or INT8 quantization for faster mobile inference without significant accuracy loss.

## Getting Started

Getting started with the Ultralytics HUB App is simple:

1. Download the app from the [App Store](https://apps.apple.com/xk/app/ultralytics-hub/id1583935240) (iOS) or [Google Play](https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app) (Android)
2. Sign in with your Ultralytics account (or create one if you don't have it)
3. Select a pre-trained model or one of your custom models
4. Start detecting objects in real-time using your device's camera

## App Documentation

- [**iOS**](ios.md): Learn about YOLO CoreML models accelerated on Apple's Neural Engine for iPhones and iPads.
- [**Android**](android.md): Explore TFLite acceleration on Android mobile devices.

## Integration with Ultralytics HUB

The Ultralytics HUB App is fully integrated with the [Ultralytics HUB platform](https://docs.ultralytics.com/hub/), allowing you to:

- Train custom models in the cloud without coding knowledge
- Manage your datasets, projects, and models in one place
- Preview and test your trained models directly on your mobile device
- Deploy models for various applications across different platforms

Get started today by downloading the Ultralytics HUB App on your mobile device and unlock the potential of YOLO models on-the-go. For more information on training, deploying, and using your custom models with the Ultralytics HUB platform, check out our comprehensive [HUB documentation](../index.md).

## FAQ

### What models can I run on the Ultralytics HUB App?

The Ultralytics HUB App supports running YOLOv5, YOLOv8, and YOLO11 models. You can use pre-trained models or train your own custom models using the [Ultralytics HUB platform](https://www.ultralytics.com/blog/how-to-train-and-deploy-yolo11-using-ultralytics-hub).

### How is model performance optimized for mobile devices?

Models are optimized through quantization (FP16 or INT8) and by leveraging hardware acceleration features like Apple's Neural Engine on iOS devices or GPU and NNAPI delegates on Android devices. This enables real-time inference while maintaining good accuracy.

### Can I use my custom-trained models on the app?

Yes! You can train custom models using the [Ultralytics HUB cloud training](https://docs.ultralytics.com/hub/cloud-training/) feature and then deploy them directly to the HUB App for testing and use on your mobile device.
