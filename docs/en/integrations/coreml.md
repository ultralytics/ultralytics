---
comments: true
description: Export Ultralytics YOLO26 models to CoreML for fast on-device inference on the Apple Neural Engine across iPhone, iPad, and Mac. Step-by-step export, quantization, and deployment guide.
keywords: CoreML export, Core ML, YOLO26 CoreML, Apple Neural Engine, ANE, mlpackage, ML Program, Vision framework, iOS object detection, macOS machine learning, Swift, quantization, INT8, FP16, on-device AI, Ultralytics
---

# CoreML Export for YOLO26 Models

Apple ships dedicated AI silicon — the Neural Engine — in every modern iPhone, iPad, and Mac, and [CoreML](https://developer.apple.com/documentation/coreml) is the only way to program it. Exporting [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models to CoreML turns a trained `.pt` checkpoint into a native `.mlpackage` that runs all six YOLO tasks on-device at single-digit milliseconds, with no network connection and no data leaving the device.

!!! tip "Run YOLO on the Apple Neural Engine today with the official mobile apps"

    The official [Ultralytics YOLO iOS SDK](https://github.com/ultralytics/yolo-ios-app) and [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) run CoreML exports on the Apple Neural Engine out of the box — real-time camera inference, single-image prediction, and automatic model download for all six YOLO26 tasks. For Android NPU deployment, see the [Qualcomm QNN integration](qnn.md).

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/hfSK3Mk5P0I"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Export Ultralytics YOLO26 to CoreML for 2x Fast Inference on Apple Devices 🚀
</p>

## What is CoreML?

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/coreml-overview.avif" alt="Apple CoreML deployment pipeline">
</p>

CoreML (styled "Core ML" by Apple) is Apple's on-device [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) framework. It loads models in the modern **ML Program** format — the `.mlpackage` bundle the Ultralytics exporter produces — and schedules them across the device's CPU, GPU, and **Apple Neural Engine (ANE)**, the dedicated NPU in every Apple-silicon chip. Because everything runs locally, inference works offline, adds no network latency, and keeps user data on the device.

CoreML integrates directly with Apple's [Vision framework](https://developer.apple.com/documentation/vision), which handles image scaling and orientation on the way into the model — this is how the Ultralytics iOS SDK feeds camera frames to YOLO with effectively zero preprocessing cost.

## Why Export YOLO26 to CoreML?

- **Neural Engine speed**: YOLO26n detection runs end-to-end in **3.8 ms** on an iPhone 17 Pro for single images, and ~16 ms/frame in sustained real-time camera use (see the table and notes below) — comfortably real-time with headroom for the rest of your app.
- **NMS-free by design**: YOLO26 is [end-to-end](https://www.ultralytics.com/glossary/non-maximum-suppression-nms), so the exported graph needs no NMS pipeline and decode is sub-millisecond. Older models like YOLO11 can embed a CoreML NMS pipeline with `nms=True`.
- **Private and offline**: All computation stays on the device — no cloud round-trips, no API keys, full [data privacy](https://www.ultralytics.com/glossary/data-privacy).
- **One export, the whole ecosystem**: The same `.mlpackage` runs on iOS, iPadOS, macOS, watchOS, tvOS, and visionOS, and powers the official Ultralytics [iOS SDK](https://github.com/ultralytics/yolo-ios-app) and [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app).

## Measured Performance

End-to-end single-image inference for the official YOLO26n INT8 CoreML models on an iPhone 17 Pro (Apple A19, iOS 26.5). Each cell shows the **total time** (preprocessing + inference + postprocessing, excluding annotation) with the per-stage split beneath it. On iOS, Vision performs input scaling inside the inference request, so preprocessing is reported as 0 and its cost is included in inference.

| Model        | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>`.cpuOnly`<br>(ms)</sup> | Neural Engine<br><sup>`.cpuAndNeuralEngine`<br>(ms)</sup> |
| ------------ | -------- | --------------------------- | ------------------------------------ | --------------------------------------------------------- |
| YOLO26n      | Detect   | 640                         | 9.1<br><sup>0.0 / 9.1 / 0.0</sup>    | **3.8**<br><sup>0.0 / 3.8 / 0.0</sup>                     |
| YOLO26n-seg  | Segment  | 640                         | 12.3<br><sup>0.0 / 12.1 / 0.2</sup>  | **4.8**<br><sup>0.0 / 4.5 / 0.3</sup>                     |
| YOLO26n-sem  | Semantic | 1024<sup>1</sup>            | 21.8<br><sup>0.0 / 21.0 / 0.8</sup>  | **12.1**<br><sup>0.0 / 11.3 / 0.8</sup>                   |
| YOLO26n-cls  | Classify | 224                         | 2.2<br><sup>0.0 / 2.2 / 0.0</sup>    | **2.0**<br><sup>0.0 / 2.0 / 0.0</sup>                     |
| YOLO26n-pose | Pose     | 640                         | 12.0<br><sup>0.0 / 11.9 / 0.0</sup>  | **3.8**<br><sup>0.0 / 3.8 / 0.0</sup>                     |
| YOLO26n-obb  | OBB      | 1024                        | 21.7<br><sup>0.0 / 21.7 / 0.0</sup>  | **7.2**<br><sup>0.0 / 7.2 / 0.0</sup>                     |

- <sup>1</sup> Semantic CoreML exports embed the ArgMax in the graph and return a compact full-resolution class map (`[1, 1024, 1024]`) instead of float logits, so the postprocess is a sub-millisecond color sweep and masks render pixel-sharp.
- **Speed** values are **single-image burst latencies** — the mean of 15 runs after 3 warmup runs on [bus.jpg](https://ultralytics.com/images/bus.jpg), measured through the [iOS SDK's](https://github.com/ultralytics/yolo-ios-app) per-stage timing via the [Flutter plugin's](https://github.com/ultralytics/yolo-flutter-app) benchmark harness in profile mode (optimized native code). Sustained real-time camera operation runs higher (full-sensor letterboxing every frame plus thermal settling): YOLO26n detect measures ~16 ms/frame in the live camera app on the same device — see the [iOS SDK performance doc](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md) for steady-state profiling.
- The matching Snapdragon CPU/GPU/NPU table is in the [Qualcomm QNN integration](qnn.md).

## Exporting YOLO26 Models to CoreML

### Installation

To install the required package, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO26
        pip install ultralytics
        ```

The `coremltools` converter is installed automatically on first export. Export runs on macOS or x86 Linux; for detailed instructions and best practices, check our [installation guide](../quickstart.md) and the [Common Issues guide](../guides/yolo-common-issues.md).

### Usage

The CoreML format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Inference and validation with CoreML run on macOS only. Export your model, then load the exported model to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export to CoreML (FP16 by default); int8=True matches the official app models
        model.export(format="coreml", int8=True)  # creates 'yolo26n.mlpackage'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to CoreML format with INT8 weight quantization
        yolo export model=yolo26n.pt format=coreml int8=True # creates 'yolo26n.mlpackage'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported CoreML model (macOS)
        model = YOLO("yolo26n.mlpackage")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported CoreML model
        yolo predict model=yolo26n.mlpackage source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported CoreML model (macOS)
        model = YOLO("yolo26n.mlpackage")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported CoreML model
        yolo val model=yolo26n.mlpackage data=coco8.yaml
        ```

### Export Arguments

| Argument  | Type             | Default    | Description                                                                                                                             |
| --------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format`  | `str`            | `'coreml'` | Target format for the exported model, defining compatibility with various deployment environments.                                      |
| `imgsz`   | `int` or `tuple` | `640`      | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.       |
| `half`    | `bool`           | `False`    | Enables FP16 weight quantization, halving model size with negligible accuracy impact — a good default for the Neural Engine.            |
| `int8`    | `bool`           | `False`    | Enables INT8 weight quantization for the smallest models; the official Ultralytics app models ship as INT8.                             |
| `nms`     | `bool`           | `False`    | Embeds a CoreML NMS pipeline. Not needed for NMS-free YOLO26; use for earlier models like YOLO11.                                       |
| `dynamic` | `bool`           | `False`    | Allows dynamic input sizes, enhancing flexibility in handling varying image dimensions.                                                 |
| `batch`   | `int`            | `1`        | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `device`  | `str`            | `None`     | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                         |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Targeting the Neural Engine

CoreML chooses hardware via `MLModelConfiguration.computeUnits`. The Ultralytics iOS SDK defaults to `.cpuAndNeuralEngine` on iOS 16+ rather than `.all`: in a real-time camera app the GPU is already busy compositing the preview and overlays, so excluding it avoids contention and frame-time jitter while the ANE does the heavy lifting. Pin `.cpuOnly` only for compatibility testing — the table above shows what it costs.

## Deploying Exported YOLO26 CoreML Models

The fastest path is the official [Ultralytics YOLO iOS SDK](https://github.com/ultralytics/yolo-ios-app), the same Swift package that powers the Ultralytics iOS app and the [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app). It resolves official model names automatically, downloads and caches the `.mlpackage`, and returns fully decoded results:

```swift
import UltralyticsYOLO

// Loads the official INT8 model (downloaded and cached on first use), then runs inference
let yolo = YOLO("yolo26n", task: .detect) { result in
    if case .success(let model) = result {
        let results = model(uiImage)  // boxes, labels, confidences, timing
    }
}
```

For camera apps, drop in the SDK's `YOLOView` for real-time inference with native overlays, or use the [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) for cross-platform apps that share one codebase with Android.

Integrating a raw `.mlpackage` yourself is also straightforward with Apple's stack — load it with `MLModel`, wrap it in a `VNCoreMLRequest`, and feed images through `VNImageRequestHandler`. These resources cover the details:

- **[Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app)**: Apple's guide to bundling and calling a CoreML model.
- **[CoreML Tools](https://apple.github.io/coremltools/docs-guides/)**: Conversion, quantization, and optimization reference for the `coremltools` toolchain that powers this export.
- **[Xcode Core ML Performance Reports](https://developer.apple.com/videos/)**: Per-layer device placement and latency profiling for your exact model and device.

Ship the model either embedded in the app bundle (instant availability, ideal for nano/small models) or downloaded on first run and cached (smaller binary, easy model updates) — the official apps use the second approach with the [GitHub release assets](https://github.com/ultralytics/yolo-ios-app/releases).

## Recommended Workflow

1. **Train** your model with Ultralytics [Train mode](../modes/train.md), or start from the official YOLO26 weights
2. **Export** with `model.export(format="coreml", int8=True)` on macOS or x86 Linux
3. **Verify** accuracy with `model.val()` on a Mac, and profile with an Xcode Core ML Performance Report on your target device
4. **Deploy** with the iOS SDK, the Flutter plugin, or your own Vision integration, targeting `.cpuAndNeuralEngine`

## Summary

In this guide, you learned how to export Ultralytics YOLO26 models to CoreML's `.mlpackage` format, quantize them for the Apple Neural Engine, and deploy them at single-digit-millisecond latencies — either through the official iOS SDK and Flutter plugin or your own Vision integration. For other deployment targets, browse the [integration guide page](../integrations/index.md), and compare formats with [Benchmark mode](../modes/benchmark.md).

## FAQ

### How do I export YOLO26 models to CoreML format?

Run `model.export(format="coreml")` in Python or `yolo export model=yolo26n.pt format=coreml` from the CLI on macOS or x86 Linux. Add `int8=True` to match the official app models. The export produces a `yolo26n.mlpackage` ML Program ready for Xcode, the iOS SDK, or the Flutter plugin.

### Do I need `nms=True` when exporting YOLO26?

No. YOLO26 is NMS-free end-to-end, so the exported graph already emits final detections and decode costs well under a millisecond. The `nms=True` option exists for earlier models such as YOLO11, where it embeds a CoreML NMS pipeline so your app does not have to implement suppression.

### Which precision should I use — FP16 or INT8?

The official Ultralytics app models ship as INT8, which minimizes download size and runs at the speeds in the table above. `half=True` (FP16) is a conservative alternative with essentially no accuracy loss. Validate your exact export with `model.val()` on a Mac before shipping.

### How do I make sure inference runs on the Neural Engine?

Set `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine` (the iOS SDK default on iOS 16+). Avoid `.all` in camera apps — the GPU is busy compositing the preview, and scheduling inference there causes frame-time jitter. Confirm placement with an Xcode Core ML Performance Report.

### Can I run and validate CoreML models with the Ultralytics CLI?

Yes, on macOS: `yolo predict model=yolo26n.mlpackage source=image.jpg` and `yolo val model=yolo26n.mlpackage data=coco8.yaml` work like any other format. CoreML execution requires Apple hardware, so these modes are unavailable on Linux and Windows.

### What is the fastest way to get YOLO26 running in an iOS or Flutter app?

Use the official [Ultralytics YOLO iOS SDK](https://github.com/ultralytics/yolo-ios-app) (Swift Package) or the [Flutter plugin](https://github.com/ultralytics/yolo-flutter-app). Both load official models by name with automatic download and caching, run them on the Neural Engine, and include complete real-time camera UIs — the measured performance table above was produced with exactly this stack.
