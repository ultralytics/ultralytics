---
comments: true
description: Experience real-time object detection on Android with Ultralytics. Leverage YOLO models for efficient and fast object identification. Download now!.
keywords: Ultralytics, Android app, real-time object detection, YOLO models, TensorFlow Lite, FP16 quantization, INT8 quantization, hardware delegates, mobile AI, download app
---

# Ultralytics Android App: Real-time Object Detection with YOLO Models

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
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
  <br>
  <br>
  <a href="https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app" style="text-decoration:none;">
    <img src="https://raw.githubusercontent.com/ultralytics/assets/master/app/google-play.svg" width="15%" alt="Google Play store"></a>&nbsp;
</div>

The Ultralytics Android App is a powerful tool that allows you to run YOLO models directly on your Android device for real-time object detection. This app utilizes [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Lite for model optimization and various hardware delegates for acceleration, enabling fast and efficient object detection.

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

To achieve real-time performance on your Android device, YOLO models are quantized to either FP16 or INT8 [precision](https://www.ultralytics.com/glossary/precision). Quantization is a process that reduces the numerical precision of the model's weights and biases, thus reducing the model's size and the amount of computation required. This results in faster inference times without significantly affecting the model's [accuracy](https://www.ultralytics.com/glossary/accuracy).

### FP16 Quantization

FP16 (or half-precision) quantization converts the model's 32-bit floating-point numbers to 16-bit floating-point numbers. This reduces the model's size by half and speeds up the inference process, while maintaining a good balance between accuracy and performance.

### INT8 Quantization

INT8 (or 8-bit integer) quantization further reduces the model's size and computation requirements by converting its 32-bit floating-point numbers to 8-bit integers. This quantization method can result in a significant speedup, but it may lead to a slight reduction in [mean average precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) due to the lower numerical precision.

!!! tip "mAP Reduction in INT8 Models"

    The reduced numerical precision in INT8 models can lead to some loss of information during the quantization process, which may result in a slight decrease in mAP. However, this trade-off is often acceptable considering the substantial performance gains offered by INT8 quantization.

## Delegates and Performance Variability

Different delegates are available on Android devices to accelerate model inference. These delegates include CPU, [GPU](https://ai.google.dev/edge/litert/android/gpu), [Hexagon](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide) and [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide). The performance of these delegates varies depending on the device's hardware vendor, product line, and specific chipsets used in the device.

1. **CPU**: The default option, with reasonable performance on most devices.
2. **GPU**: Utilizes the device's GPU for faster inference. It can provide a significant performance boost on devices with powerful GPUs.
3. **Hexagon**: Leverages Qualcomm's Hexagon DSP for faster and more efficient processing. This option is available on devices with Qualcomm Snapdragon processors.
4. **NNAPI**: The Android [Neural Networks](https://www.ultralytics.com/glossary/neural-network-nn) API (NNAPI) serves as an abstraction layer for running ML models on Android devices. NNAPI can utilize various hardware accelerators, such as CPU, GPU, and dedicated AI chips (e.g., Google's Edge TPU, or the Pixel Neural Core).

Here's a table showing the primary vendors, their product lines, popular devices, and supported delegates:

| Vendor                                    | Product Lines                                                                        | Popular Devices                                                                                                                                                              | Delegates Supported      |
| ----------------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| [Qualcomm](https://www.qualcomm.com/)     | [Snapdragon (e.g., 800 series)](https://www.qualcomm.com/snapdragon/overview)        | [Samsung Galaxy S21](https://www.samsung.com/us/mobile/phones/galaxy-s/), [OnePlus 9](https://www.oneplus.com/9), [Google Pixel 6](https://store.google.com/product/pixel_6) | CPU, GPU, Hexagon, NNAPI |
| [Samsung](https://www.samsung.com/)       | [Exynos (e.g., Exynos 2100)](https://www.samsung.com/semiconductor/minisite/exynos/) | [Samsung Galaxy S21 (Global version)](https://www.samsung.com/us/mobile/phones/galaxy-s/)                                                                                    | CPU, GPU, NNAPI          |
| [MediaTek](https://i.mediatek.com/)       | [Dimensity (e.g., Dimensity 1200)](https://i.mediatek.com/dimensity-1200)            | [Realme GT](https://www.realme.com/global/realme-gt), [Xiaomi Redmi Note](https://www.mi.com/global/phone/redmi/note-list)                                                   | CPU, GPU, NNAPI          |
| [HiSilicon](https://www.hisilicon.com/cn) | [Kirin (e.g., Kirin 990)](https://www.hisilicon.com/en/products/Kirin)               | [Huawei P40 Pro](https://consumer.huawei.com/en/phones/), [Huawei Mate 30 Pro](https://consumer.huawei.com/en/phones/)                                                       | CPU, GPU, NNAPI          |
| [NVIDIA](https://www.nvidia.com/)         | [Tegra (e.g., Tegra X1)](https://developer.nvidia.com/content/tegra-x1)              | [NVIDIA Shield TV](https://www.nvidia.com/en-us/shield/shield-tv/), [Nintendo Switch](https://www.nintendo.com/switch/)                                                      | CPU, GPU, NNAPI          |

Please note that the list of devices mentioned is not exhaustive and may vary depending on the specific chipsets and device models. Always test your models on your target devices to ensure compatibility and optimal performance.

Keep in mind that the choice of delegate can affect performance and model compatibility. For example, some models may not work with certain delegates, or a delegate may not be available on a specific device. As such, it's essential to test your model and the chosen delegate on your target devices for the best results.

## Getting Started with the Ultralytics Android App

To get started with the Ultralytics Android App, follow these steps:

1. Download the Ultralytics App from the [Google Play Store](https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app).

2. Launch the app on your Android device and sign in with your Ultralytics account. If you don't have an account yet, create one [here](https://hub.ultralytics.com/).

3. Once signed in, you will see a list of your trained YOLO models. Select a model to use for object detection.

4. Grant the app permission to access your device's camera.

5. Point your device's camera at objects you want to detect. The app will display bounding boxes and class labels in real-time as it detects objects.

6. Explore the app's settings to adjust the detection threshold, enable or disable specific object classes, and more.

With the Ultralytics Android App, you now have the power of real-time object detection using YOLO models right at your fingertips. Enjoy exploring the app's features and optimizing its settings to suit your specific use cases.
