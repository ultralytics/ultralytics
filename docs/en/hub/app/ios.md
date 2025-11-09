---
comments: true
description: Discover the Ultralytics iOS App for running YOLO models on your iPhone or iPad. Achieve fast, real-time object detection with Apple Neural Engine.
keywords: Ultralytics, iOS App, YOLO models, real-time object detection, Apple Neural Engine, Core ML, FP16 quantization, INT8 quantization, machine learning
---

# Ultralytics iOS App: Real-time Object Detection with YOLO Models

<a href="https://www.ultralytics.com/hub" target="_blank">
  <img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-android-app-detection.avif" alt="Ultralytics HUB preview image"></a>
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
</div>

The Ultralytics iOS App is a powerful tool that allows you to run YOLO models directly on your iPhone or iPad for real-time object detection. This app utilizes the Apple Neural Engine and Core ML for model optimization and acceleration, enabling fast and efficient object detection.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/AIvrQ7y0aLo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Getting Started with the Ultralytics HUB App (IOS & Android)
</p>

## Quantization and Acceleration

To achieve real-time performance on your iOS device, YOLO models are quantized to either FP16 or INT8 [precision](https://www.ultralytics.com/glossary/precision). Quantization is a process that reduces the numerical precision of the model's weights and biases, thus reducing the model's size and the amount of computation required. This results in faster inference times without significantly affecting the model's [accuracy](https://www.ultralytics.com/glossary/accuracy).

### FP16 Quantization

FP16 (or half-precision) quantization converts the model's 32-bit floating-point numbers to 16-bit floating-point numbers. This reduces the model's size by half and speeds up the inference process, while maintaining a good balance between accuracy and performance.

### INT8 Quantization

INT8 (or 8-bit integer) quantization further reduces the model's size and computation requirements by converting its 32-bit floating-point numbers to 8-bit integers. This quantization method can result in a significant speedup, but it may lead to a slight reduction in accuracy.

## Apple Neural Engine

The Apple Neural Engine (ANE) is a dedicated hardware component integrated into Apple's A-series and M-series chips. It's designed to accelerate [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) tasks, particularly for [neural networks](https://www.ultralytics.com/glossary/neural-network-nn), allowing for faster and more efficient execution of your YOLO models.

By combining quantized YOLO models with the Apple Neural Engine, the Ultralytics iOS App achieves real-time object detection on your iOS device without compromising on accuracy or performance.

| Release Year | iPhone Name                                          | Chipset Name                                          | Node Size | ANE TOPs |
| ------------ | ---------------------------------------------------- | ----------------------------------------------------- | --------- | -------- |
| 2017         | [iPhone X](https://en.wikipedia.org/wiki/IPhone_X)   | [A11 Bionic](https://en.wikipedia.org/wiki/Apple_A11) | 10 nm     | 0.6      |
| 2018         | [iPhone XS](https://en.wikipedia.org/wiki/IPhone_XS) | [A12 Bionic](https://en.wikipedia.org/wiki/Apple_A12) | 7 nm      | 5        |
| 2019         | [iPhone 11](https://en.wikipedia.org/wiki/IPhone_11) | [A13 Bionic](https://en.wikipedia.org/wiki/Apple_A13) | 7 nm      | 6        |
| 2020         | [iPhone 12](https://en.wikipedia.org/wiki/IPhone_12) | [A14 Bionic](https://en.wikipedia.org/wiki/Apple_A14) | 5 nm      | 11       |
| 2021         | [iPhone 13](https://en.wikipedia.org/wiki/IPhone_13) | [A15 Bionic](https://en.wikipedia.org/wiki/Apple_A15) | 5 nm      | 15.8     |
| 2022         | [iPhone 14](https://en.wikipedia.org/wiki/IPhone_14) | [A16 Bionic](https://en.wikipedia.org/wiki/Apple_A16) | 4 nm      | 17.0     |
| 2023         | [iPhone 15](https://en.wikipedia.org/wiki/IPhone_15) | [A17 Pro](https://en.wikipedia.org/wiki/Apple_A17)    | 3 nm      | 35.0     |

Please note that this list includes iPhone models from 2017 onwards, and the ANE TOPs values are approximate.

## CoreML Integration

The Ultralytics iOS App leverages [CoreML](https://docs.ultralytics.com/integrations/coreml/), Apple's foundational machine learning framework, to optimize YOLO models for iOS devices. CoreML provides several advantages:

- **On-device processing**: All inference happens locally on your device, ensuring data privacy and eliminating the need for internet connectivity
- **Hardware acceleration**: Automatically utilizes the Apple Neural Engine, CPU, and GPU for optimal performance
- **Seamless integration**: Works natively with iOS camera and system frameworks

CoreML converts YOLO models to a format that's optimized for Apple devices, allowing for efficient execution while maintaining detection accuracy.

## Getting Started with the Ultralytics iOS App

To get started with the Ultralytics iOS App, follow these steps:

1. Download the Ultralytics App from the [App Store](https://apps.apple.com/xk/app/ultralytics-hub/id1583935240).

2. Launch the app on your iOS device and sign in with your Ultralytics account. If you don't have an account yet, create one at [Ultralytics HUB](https://hub.ultralytics.com/).

3. Once signed in, you will see a list of your trained YOLO models. Select a model to use for object detection.

4. Grant the app permission to access your device's camera.

5. Point your device's camera at objects you want to detect. The app will display bounding boxes and class labels in real-time as it detects objects.

6. Explore the app's settings to adjust the detection threshold, enable or disable specific object classes, and more.

With the Ultralytics iOS App, you can now leverage the power of YOLO models for real-time object detection on your iPhone or iPad, powered by the Apple Neural Engine and optimized with FP16 or INT8 quantization.
