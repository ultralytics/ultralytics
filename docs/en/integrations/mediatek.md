---
title: MediaTek Dimensity Deployment with LiteRT
comments: true
description: Deploy Ultralytics YOLO26 on MediaTek Dimensity Android phones with the LiteRT export and the official Flutter plugin, with measured CPU, GPU, and prototype NPU results.
keywords: MediaTek, Dimensity 9500, MediaTek NPU, LiteRT, TFLite, Android, YOLO26, Flutter, edge AI, Ultralytics
---

# MediaTek Dimensity Deployment with LiteRT

Ultralytics YOLO runs on **MediaTek Dimensity** Android phones through the existing [LiteRT export](litert.md). There is no MediaTek-specific export format: export a standard `.tflite` model and run it on Android with the official [Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app), which uses LiteRT for CPU and GPU inference.

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

!!! warning "MediaTek NPU support is a prototype"

    The released Flutter plugin uses LiteRT CPU and GPU only. The NPU results below were measured with a prototype, unreleased NPU build of the plugin and the LiteRT `mediatek` 2.3.0 NPU runtime, using FP32 exports (no `quantize`). The `w8a32` assets do not compile fully on the NPU.

## Measured Performance

The [Xiaomi 17T Pro results in the LiteRT guide](litert.md#xiaomi-17t-pro) cover all seven YOLO26n tasks on a **MediaTek Dimensity 9500** (MT6993) with MediaTek NPU 990 and Android 16 / API 36. Each value is the mean of 15 calls after 3 warmups, using `ultralytics_yolo` `0.7.1` and the official `v0.7.0` assets. YOLO26n detection takes 31.4 ms on the CPU, 25.6 ms on the GPU, and 11.0 ms on the prototype NPU path. Depth and OBB have no NPU result and run on the GPU instead.

## MediaTek Genio

These results are for **Dimensity Android phones** using LiteRT. MediaTek **Genio** IoT platforms use a separate Yocto and NeuroPilot software stack. Ultralytics does not validate or document Genio deployment, and the Dimensity results above do not apply to Genio boards.
