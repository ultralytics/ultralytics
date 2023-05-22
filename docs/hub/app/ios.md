---
comments: true
description: Get started with the Ultralytics iOS app and run YOLO models in real-time for object detection on your iPhone or iPad with the Apple Neural Engine.
---

# Ultralytics iOS App: Real-time Object Detection with YOLO Models

The Ultralytics iOS App is a powerful tool that allows you to run YOLO models directly on your iPhone or iPad for real-time object detection. This app utilizes the Apple Neural Engine and Core ML for model optimization and acceleration, enabling fast and efficient object detection.

## Quantization and Acceleration

To achieve real-time performance on your iOS device, YOLO models are quantized to either FP16 or INT8 precision. Quantization is a process that reduces the numerical precision of the model's weights and biases, thus reducing the model's size and the amount of computation required. This results in faster inference times without significantly affecting the model's accuracy.

### FP16 Quantization

FP16 (or half-precision) quantization converts the model's 32-bit floating-point numbers to 16-bit floating-point numbers. This reduces the model's size by half and speeds up the inference process, while maintaining a good balance between accuracy and performance.

### INT8 Quantization

INT8 (or 8-bit integer) quantization further reduces the model's size and computation requirements by converting its 32-bit floating-point numbers to 8-bit integers. This quantization method can result in a significant speedup, but it may lead to a slight reduction in accuracy.

## Apple Neural Engine

The Apple Neural Engine (ANE) is a dedicated hardware component integrated into Apple's A-series and M-series chips. It's designed to accelerate machine learning tasks, particularly for neural networks, allowing for faster and more efficient execution of your YOLO models.

By combining quantized YOLO models with the Apple Neural Engine, the Ultralytics iOS App achieves real-time object detection on your iOS device without compromising on accuracy or performance.

| Release Year | iPhone Name                                          | Chipset Name                                          | Node Size | ANE TOPs |
|--------------|------------------------------------------------------|-------------------------------------------------------|-----------|----------|
| 2017         | [iPhone X](https://en.wikipedia.org/wiki/IPhone_X)   | [A11 Bionic](https://en.wikipedia.org/wiki/Apple_A11) | 10 nm     | 0.6      |
| 2018         | [iPhone XS](https://en.wikipedia.org/wiki/IPhone_XS) | [A12 Bionic](https://en.wikipedia.org/wiki/Apple_A12) | 7 nm      | 5        |
| 2019         | [iPhone 11](https://en.wikipedia.org/wiki/IPhone_11) | [A13 Bionic](https://en.wikipedia.org/wiki/Apple_A13) | 7 nm      | 6        |
| 2020         | [iPhone 12](https://en.wikipedia.org/wiki/IPhone_12) | [A14 Bionic](https://en.wikipedia.org/wiki/Apple_A14) | 5 nm      | 11       |
| 2021         | [iPhone 13](https://en.wikipedia.org/wiki/IPhone_13) | [A15 Bionic](https://en.wikipedia.org/wiki/Apple_A15) | 5 nm      | 15.8     |
| 2022         | [iPhone 14](https://en.wikipedia.org/wiki/IPhone_14) | [A16 Bionic](https://en.wikipedia.org/wiki/Apple_A16) | 4 nm      | 17.0     |

Please note that this list only includes iPhone models from 2017 onwards, and the ANE TOPs values are approximate.

## Getting Started with the Ultralytics iOS App

To get started with the Ultralytics iOS App, follow these steps:

1. Download the Ultralytics App from the [App Store](https://apps.apple.com/xk/app/ultralytics/id1583935240).

2. Launch the app on your iOS device and sign in with your Ultralytics account. If you don't have an account yet, create one [here](https://hub.ultralytics.com/).

3. Once signed in, you will see a list of your trained YOLO models. Select a model to use for object detection.

4. Grant the app permission to access your device's camera.

5. Point your device's camera at objects you want to detect. The app will display bounding boxes and class labels in real-time as it detects objects.

6. Explore the app's settings to adjust the detection threshold, enable or disable specific object classes, and more.

With the Ultralytics iOS App, you can now leverage the power of YOLO models for real-time object detection on your iPhone or iPad, powered by the Apple Neural Engine and optimized with FP16 or INT8 quantization.