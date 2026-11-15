---
title: Deploy Ultralytics YOLO on MediaTek Dimensity with LiteRT
comments: true
description: Deploy Ultralytics YOLO26 on MediaTek Dimensity Android phones with the LiteRT export and the official Flutter plugin, with measured CPU, GPU, and prototype NPU results.
keywords: MediaTek, Dimensity 9500, MediaTek NPU, LiteRT, TFLite, Android, YOLO26, Flutter, edge AI, Ultralytics
---

# Deploy Ultralytics YOLO on MediaTek Dimensity with LiteRT

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/829fcdeccbb9afee17bb36dfdc85207d.avif" alt="MediaTek Dimensity chip on a circuit board">
</p>

Ultralytics YOLO runs on [MediaTek Dimensity](https://www.mediatek.com/products/smartphones/mediatek-dimensity-9500) Android phones through the existing [LiteRT export](litert.md). There is no MediaTek-specific export format: export a standard `.tflite` model and run it on Android with the official [Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app), which uses LiteRT for CPU and GPU inference.

## Export

LiteRT export is supported on **Linux x86_64** and **macOS**, not on Linux aarch64 hosts such as NVIDIA Jetson. Export classification models at `imgsz=224` and all other tasks at `imgsz=640`.

```bash
pip install ultralytics
```

!!! example "Export to LiteRT"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="litert", imgsz=640)  # creates 'yolo26n.tflite' (FP32)
        model.export(format="litert", quantize="w8a32", imgsz=640)  # creates 'yolo26n_w8a32.tflite'
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=litert imgsz=640
        yolo export model=yolo26n.pt format=litert quantize=w8a32 imgsz=640
        ```

You can check an exported model on the host with `YOLO("yolo26n.tflite")("https://ultralytics.com/images/bus.jpg")`. This runs LiteRT locally and does not select or test a MediaTek NPU. See the [LiteRT guide](litert.md#usage) for prediction, validation, and all export arguments.

## Deploy on Android

The [Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) runs LiteRT `.tflite` exports on Android with camera and single-image inference for all seven YOLO26 tasks. Official `w8a32` assets are published in the [yolo-flutter-app `v0.7.0` release](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.7.0).

Try the [Ultralytics YOLO Android app](https://play.google.com/store/apps/details?id=com.ultralytics.yolo) on Google Play. The previews below show the app on iOS.

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/4d2098fa5d7bc0e5a5aa28660da41eb4.avif" alt="Ultralytics YOLO app previews on iOS showing segmentation, pose, depth, oriented boxes, and detection">
</p>

<p align="center">
  <a href="https://play.google.com/store/apps/details?id=com.ultralytics.yolo"><img width="160" src="https://raw.githubusercontent.com/ultralytics/assets/main/app/google-play.svg" alt="Get it on Google Play"></a>
</p>

To run your exported model, copy `yolo26n_w8a32.tflite` into `assets/models/` in a Flutter app and add the plugin and asset to `pubspec.yaml`:

```yaml
dependencies:
    flutter:
        sdk: flutter
    ultralytics_yolo: ^0.7.1

flutter:
    assets:
        - assets/models/yolo26n_w8a32.tflite
```

Run `flutter pub get`, then use the model for camera inference in `lib/main.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() => runApp(
      MaterialApp(
        home: Scaffold(
          body: YOLOView(modelPath: 'assets/models/yolo26n_w8a32.tflite'),
        ),
      ),
    );
```

Set Android's minimum SDK to 24, add camera permission, and run `flutter run` on the phone. See the [plugin installation guide](https://github.com/ultralytics/yolo-flutter-app/blob/main/doc/install.md#android-configuration) for the Android setup and the [quickstart](https://github.com/ultralytics/yolo-flutter-app/blob/main/doc/quickstart.md) for single-image inference. The plugin attempts GPU inference and falls back to CPU if the device or model cannot use the GPU.

!!! warning "MediaTek NPU support is a prototype"

    On MediaTek devices, the released Flutter plugin supports LiteRT CPU and GPU inference, but not the MediaTek NPU. The NPU results below were measured with a prototype, unreleased NPU build of the plugin and the LiteRT `mediatek` 2.3.0 NPU runtime, using FP32 exports (no `quantize`). The `w8a32` assets do not compile fully on the NPU.

## Measured Performance

The [Xiaomi 17T Pro results in the LiteRT guide](litert.md#xiaomi-17t-pro) cover all seven YOLO26n tasks on a **MediaTek Dimensity 9500** (MT6993) with MediaTek NPU 990 and Android 16 / API 36. Each value is the mean of 15 calls after 3 warmups. CPU and GPU measurements used `ultralytics_yolo` `0.7.1` and the official `v0.7.0` `w8a32` assets; the prototype NPU measurements used FP32 exports. YOLO26n detection takes 31.4 ms on the CPU, 25.6 ms on the GPU, and 11.0 ms on the prototype NPU path. Depth and OBB have no NPU result and run on the GPU instead.

Compare more devices and models on the [Ultralytics mobile benchmarks page](https://www.ultralytics.com/benchmarks?tab=mobile).

## MediaTek Genio

These results are for **Dimensity Android phones** using LiteRT. MediaTek **Genio** IoT platforms use a separate Yocto and NeuroPilot software stack. Ultralytics does not validate or document Genio deployment, and the Dimensity results above do not apply to Genio boards.

## FAQ

### How do I export YOLO for a MediaTek Dimensity phone?

Use `model.export(format="litert", imgsz=640)` or `yolo export model=yolo26n.pt format=litert imgsz=640`. Use `imgsz=224` for classification. The resulting `.tflite` file can run with the [Flutter plugin](#deploy-on-android); there is no separate MediaTek export format.

### Can I run YOLO on the MediaTek NPU with the released Flutter plugin?

The released plugin supports LiteRT CPU and GPU inference. The NPU results above used an unreleased prototype with FP32 models; the official `w8a32` assets do not compile fully on the NPU.

### Do the Dimensity benchmarks apply to MediaTek Genio boards?

No. The measurements are from an Android phone, while Genio uses a different software stack. Ultralytics has not validated Genio deployment.
