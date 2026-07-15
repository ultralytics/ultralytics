---
title: Export YOLO to TFLite for Edge Devices
comments: true
description: Learn how to convert YOLO26 models to TFLite for edge device deployment. Optimize performance and ensure seamless execution on various platforms.
keywords: YOLO26, TFLite, model export, TensorFlow Lite, edge devices, deployment, Ultralytics, machine learning, on-device inference, model optimization
---

# A Guide on YOLO26 Model Export to TFLite for Deployment

!!! warning "Deprecated — replaced by LiteRT"

    As of **Ultralytics 8.4.83**, the standalone `tflite` export format has been removed and replaced by the unified **[Google LiteRT](litert.md)** format. LiteRT (_Lite Runtime_) is the next generation and new name for TensorFlow Lite, and it exports the **same `.tflite` model** — now covering mobile, embedded, edge, and browser deployment in one format.

    `format="tflite"` still works but emits a deprecation warning and exports a LiteRT model instead. **Use [`format="litert"`](litert.md)** going forward; for current export instructions and options, see the **[LiteRT export guide](litert.md)**.

<p align="center">
  <img width="75%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/tflite-logo.avif" alt="TensorFlow Lite edge deployment framework">
</p>

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on edge devices or embedded devices requires a format that can ensure seamless performance.

The TensorFlow Lite or TFLite export format allows you to optimize your [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models for tasks like [object detection](https://www.ultralytics.com/glossary/object-detection) and [image classification](https://www.ultralytics.com/glossary/image-classification) in edge device-based applications. In this guide, we'll walk through the steps for converting your models to the TFLite format, making it easier for your models to perform well on various edge devices.

## Why Should You Export to TFLite?

Introduced by Google in May 2017 as part of their TensorFlow framework, [TensorFlow Lite](https://developers.google.com/edge/litert), or TFLite for short, is an open-source deep learning framework designed for on-device inference, also known as [edge computing](https://www.ultralytics.com/glossary/edge-computing). It gives developers the necessary tools to execute their trained models on mobile, embedded, and IoT devices, as well as traditional computers.

TensorFlow Lite is compatible with a wide range of platforms, including embedded Linux, Android, iOS, and microcontrollers (MCUs). Exporting your model to TFLite makes your applications faster, more reliable, and capable of running offline.

## Key Features of TFLite Models

TFLite models offer a wide range of key features that enable on-device machine learning by helping developers run their models on mobile, embedded, and edge devices:

- **On-device Optimization**: TFLite optimizes for on-device ML, reducing latency by processing data locally, enhancing privacy by not transmitting personal data, and minimizing model size to save space.

- **Multiple Platform Support**: TFLite offers extensive platform compatibility, supporting Android, iOS, embedded Linux, and microcontrollers.

- **Diverse Language Support**: TFLite is compatible with various programming languages, including Java, Swift, Objective-C, C++, and Python.

- **High Performance**: Achieves superior performance through hardware acceleration and model optimization.

## Measured Performance (historical)

!!! note "Before/after reference for the format migration"

    These TFLite numbers are kept as a **historical before/after record** for the onnx2tf-TFLite → LiteRT migration: the legacy onnx2tf **INT8 TFLite** export below versus the new **[LiteRT](litert.md) w8a32** export (see the [LiteRT Measured Performance table](litert.md#measured-performance)). They are shared with the Google LiteRT team to show where the new litert-torch format still regresses against the format it replaced — see [Format regressions](#format-regressions-vs-litert) below.

Per-task before/after on the Adreno GPU of a [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) (Qualcomm Snapdragon 8 Elite Gen 5, SM8850), measured through the [Ultralytics Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) `0.6.8`: the legacy onnx2tf **INT8 TFLite** assets (NHWC, input `images`) versus the new **w8a32 LiteRT** assets (NCHW, input `args_0`), both run on LiteRT 2.x in the same back-to-back sweep at the shipped Android `imgsz`. Each cell is the **total time** (preprocessing + inference + postprocessing) with the per-stage split beneath it; both formats compiled fully on the GPU.

| Model        | Task     | size<br><sup>(pixels)</sup> | Before<br><sup>onnx2tf INT8 TFLite<br>(ms)</sup> | After<br><sup>w8a32 LiteRT<br>(ms)</sup> |
| ------------ | -------- | --------------------------- | ------------------------------------------------ | ---------------------------------------- |
| YOLO26n      | Detect   | 640                         | 14.0<br><sup>1.8 / 8.1 / 4.2</sup>               | **13.5**<br><sup>1.9 / 8.1 / 3.5</sup>   |
| YOLO26n-seg  | Segment  | 640                         | 30.1<br><sup>1.9 / 20.3 / 8.0</sup>              | **28.6**<br><sup>1.8 / 20.1 / 6.7</sup>  |
| YOLO26n-sem  | Semantic | 640                         | **26.4**<br><sup>1.9 / 16.4 / 8.1</sup>          | 32.9<br><sup>1.8 / 23.0 / 8.2</sup>      |
| YOLO26n-cls  | Classify | 224                         | 3.5<br><sup>0.9 / 2.2 / 0.4</sup>                | **3.2**<br><sup>1.0 / 2.2 / 0.1</sup>    |
| YOLO26n-pose | Pose     | 640                         | 17.4<br><sup>2.4 / 9.9 / 5.1</sup>               | **14.0**<br><sup>1.9 / 9.3 / 2.8</sup>   |
| YOLO26n-obb  | OBB      | 640                         | 13.9<br><sup>3.0 / 8.3 / 2.7</sup>               | **13.0**<br><sup>2.9 / 7.9 / 2.3</sup>   |

w8a32 LiteRT matches or beats the legacy onnx2tf INT8 format on five of six tasks in total latency. **Semantic remains the format regression** because the w8a32 NCHW logits cost more inference time than the legacy NHWC logits, even after preprocessing cleanup. The legacy onnx2tf models run unchanged on LiteRT 2.x alongside the new NCHW exports. The official Android LiteRT assets are hosted on the [yolo-flutter-app `v0.6.6` release](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.6.6), with the detailed benchmark record in [the Flutter performance doc](https://github.com/ultralytics/yolo-flutter-app/blob/main/doc/performance.md).

### Format regressions vs LiteRT

Same-device YOLO26n detect on the Adreno GPU of a [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) — legacy onnx2tf INT8 TFLite versus the four LiteRT quantization formats, all measured in one sustained run (so **inference** is the comparable, format-dependent metric):

| Android format                    | GPU inference (ms) | GPU-compiles |
| --------------------------------- | ------------------ | ------------ |
| onnx2tf INT8 (legacy TFLite)      | **8.6**            | yes          |
| LiteRT w8a32 (new official)       | 8.4                | yes          |
| LiteRT INT8 (`quantize=8`)        | 11.0               | yes          |
| LiteRT FP32                       | 8.8                | yes          |
| LiteRT w8a16 (`quantize="w8a16"`) | (CPU fallback)     | no — fails   |

Issues for the Google LiteRT / litert-torch team, surfaced migrating production Android assets from onnx2tf TFLite to LiteRT:

1. **NCHW layout makes consumers layout-aware.** litert-torch traces the PyTorch model and emits **NCHW** `[1,3,H,W]` with a float input, whereas the onnx2tf TFLite export was **NHWC** `[1,H,W,3]` — matching the camera/bitmap layout. The current Flutter plugin writes planar CHW directly during RGB packing, avoiding a separate HWC→CHW transpose, but simpler consumers still need either direct planar packing or an extra transpose.
2. **`quantize="w8a16"` does not compile on the GPU (OpenCL) delegate** and silently falls back to a CPU path that is ~40× slower (~660 ms vs ~17 ms), making the int16-activation format unusable for GPU deployment.
3. **Static INT8 (`quantize=8`) is the slowest GPU format** — ~11 ms vs ~8.6 ms for the equivalent legacy onnx2tf INT8 model, i.e. LiteRT's own INT8 path regresses against the format it replaced. Dynamic-range **w8a32** is the only LiteRT format that matches the old INT8 speed, which is why it is now shipped.
4. **Semantic models export as raw NCHW logits with no in-graph ArgMax option,** forcing a cache-unfriendly host-side argmax over `[1, C, H, W]` (each class plane is a full H×W apart). The onnx2tf, CoreML, and QNN paths can emit a compact class map instead.
5. **Output tensors were renamed `output_0`, `output_1`, …** (vs onnx2tf `Identity`, `Identity_1`, …), which silently broke runtime output-shape lookup until the consumer added the new names.

The corresponding **LiteRT w8a32** numbers (the format now shipped) are on the [LiteRT page](litert.md#measured-performance).
