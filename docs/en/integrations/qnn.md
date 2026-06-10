---
comments: true
description: Export Ultralytics YOLO models to Qualcomm QNN for fast on-device inference on Snapdragon Hexagon NPU, Adreno GPU, and CPU. Step-by-step Qualcomm Snapdragon export guide.
keywords: Qualcomm QNN, Qualcomm export, Snapdragon export, export YOLO to Qualcomm, QNN export, YOLO Snapdragon, YOLO on Snapdragon, Qualcomm AI Hub, Hexagon NPU, Hexagon HTP, Qualcomm AI Engine Direct, QAIRT, SNPE, onnxruntime-qnn, ONNX Runtime QNN, Snapdragon NPU, on-device inference, edge AI deployment, Ultralytics YOLO, model export
---

# Qualcomm QNN Export for Ultralytics YOLO Models

Deploying computer vision models on Qualcomm Snapdragon devices requires a model format tuned for the Qualcomm AI Engine Direct (QNN) runtime. Exporting [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models to the QNN format lets you run accelerated, on-device inference across Snapdragon CPU, Adreno GPU, and Hexagon NPU hardware found in billions of mobile phones, laptops, automotive systems, and IoT devices. This guide walks through how to export YOLO to Qualcomm QNN and deploy it for fast, low-power inference on Snapdragon hardware.

!!! tip "Run YOLO on Snapdragon NPUs today with the official mobile apps"

    The official [Ultralytics Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) runs QNN exports on the Hexagon NPU out of the box — real-time camera inference, single-image prediction, and automatic model download for all six YOLO26 tasks. For iOS deployment, see the [Ultralytics YOLO iOS SDK](https://github.com/ultralytics/yolo-ios-app) and the [CoreML integration](coreml.md).

## What is Qualcomm QNN?

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/qnn_cover.avif" alt="Qualcomm QNN on-device inference">
</p>

[Qualcomm AI Engine Direct](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) — commonly referred to as **QNN** and distributed as part of the Qualcomm AI Runtime (QAIRT) SDK — is Qualcomm's low-level inference stack for [Snapdragon](https://www.qualcomm.com/products/mobile/snapdragon) processors. It provides a unified API with backend-specific libraries that target the Snapdragon CPU, the Adreno GPU, and the Hexagon Tensor Processor (HTP), the dedicated [neural network](https://www.ultralytics.com/glossary/neural-network-nn) processing unit (NPU) inside modern Snapdragon SoCs. QNN gives developers full-stack access to these Snapdragon AI accelerators and is the modern successor to the older [Snapdragon Neural Processing Engine (SNPE)](https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai) SDK. It powers on-device AI across the Snapdragon 8 Gen 2, 8 Gen 3, and 8 Elite mobile platforms, Snapdragon X laptops, and automotive and XR products.

## Why Export to Qualcomm QNN?

Snapdragon is the most widely deployed mobile compute platform in the world. Exporting Ultralytics YOLO to the Qualcomm QNN format unlocks the dedicated AI hardware on these devices:

- **Hexagon NPU acceleration**: Running YOLO on the Hexagon Tensor Processor delivers dramatically higher throughput and lower power than CPU inference — ideal for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference) and always-on computer vision on Snapdragon.
- **On-device and offline**: QNN inference runs entirely on the Snapdragon device, so there are no cloud round-trips, latency stays low, and data never leaves the device.
- **Quantized efficiency**: QNN export [quantizes](https://www.ultralytics.com/glossary/model-quantization) YOLO to INT8 weights with 16-bit activations, the Hexagon NPU's preferred accuracy/performance balance, shrinking model size and maximizing frames per second on battery-powered hardware.
- **One format, many devices**: A single Qualcomm QNN export targets Snapdragon CPU, Adreno GPU, and Hexagon NPU across the Snapdragon 8 Gen 2, 8 Gen 3, and 8 Elite families and beyond.
- **Production-ready Qualcomm AI stack**: QNN (Qualcomm AI Engine Direct / QAIRT) is Qualcomm's current, actively maintained on-device AI runtime and the recommended replacement for SNPE.

## QNN Export Format

Ultralytics compiles YOLO models to QNN **locally** using the [ONNX Runtime](https://onnxruntime.ai/) QNN Execution Provider (the pip-installable `onnxruntime-qnn` package, which bundles the QAIRT libraries). The exporter converts your model to [ONNX](onnx.md), **quantizes it** with calibration data to 16-bit activations and INT8 weights (the recommended balance for the Hexagon NPU), then initializes an ONNX Runtime session with context-binary caching enabled — this compiles the quantized graph into a **QNN context binary** embedded in `<model>_qnn.onnx`. No Qualcomm account, cloud upload, or separate SDK download is required.

Unlike the cloud-based [Qualcomm AI Hub](https://aihub.qualcomm.com/), which compiles and profiles models on Qualcomm-hosted Snapdragon devices and requires a Qualcomm account, the Ultralytics QNN export runs entirely on your own machine with a single `export(format="qnn")` call. You get the same QNN/QAIRT runtime target — Snapdragon CPU, Adreno GPU, and Hexagon NPU — without sign-up, upload limits, or queue times, and it drops straight into the standard YOLO export workflow.

The exported `*_qnn.onnx` file is self-contained: it embeds the QNN context binary and ONNX metadata such as class names, image size, and task.

## Key Features of QNN Models

- **Quantization**: The model is quantized to 16-bit activations and INT8 weights with the ONNX Runtime QNN QDQ flow and a calibration dataset, the Hexagon NPU's recommended accuracy/performance balance. Learn more about [model quantization](https://www.ultralytics.com/glossary/model-quantization).
- **Fully Local Compilation**: The context binary is generated entirely on your host machine — no Qualcomm account, API token, or cloud upload.
- **Full Snapdragon Acceleration**: Run inference on the Hexagon NPU (HTP), Adreno GPU, or CPU through a single unified runtime.
- **Broad Device Reach**: Target the wide range of Snapdragon platforms shipping in phones, PCs (Windows on Snapdragon), automotive, XR, and embedded products.
- **Precompiled Context Binary**: Shipping a context binary minimizes on-device graph compilation, reducing model load latency on the target.
- **Self-Contained Output**: The exported ONNX file includes the precompiled QNN context binary and metadata for straightforward deployment.

## Measured Performance

End-to-end single-image inference for the official YOLO26n models on a Xiaomi 17 phone powered by the Qualcomm Snapdragon 8 Elite Gen 5 (SM8850) — Qualcomm Oryon CPU, Adreno GPU, and Hexagon NPU (HTP v81). Each cell shows the **total time** (preprocessing + inference + postprocessing, excluding annotation) with the per-stage split beneath it. CPU and GPU run INT8 TFLite via LiteRT; the NPU runs QNN context binaries (INT8 weights, 16-bit activations).

| Model        | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>INT8 TFLite<br>(ms)</sup> | GPU Adreno<br><sup>INT8 TFLite<br>(ms)</sup> | NPU Hexagon<br><sup>QNN A16W8<br>(ms)</sup> |
| ------------ | -------- | --------------------------- | -------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| YOLO26n      | Detect   | 640                         | 53.3<br><sup>3.6 / 47.4 / 2.4</sup>    | 17.2<br><sup>3.6 / 9.1 / 4.5</sup>            | **11.3**<br><sup>3.5 / 5.6 / 2.2</sup>        |
| YOLO26n-seg  | Segment  | 640                         | 76.0<br><sup>3.6 / 64.7 / 7.7</sup>    | 23.9<br><sup>3.6 / 11.8 / 8.6</sup>           | **21.3**<br><sup>3.5 / 7.9 / 10.0</sup>       |
| YOLO26n-sem  | Semantic | 1024                        | 66.6<br><sup>3.6 / 46.3 / 16.8</sup>   | **37.7**<br><sup>3.6 / 17.4 / 16.7</sup>      | 49.1<sup>1</sup><br><sup>8.8 / 20.8 / 19.5</sup> |
| YOLO26n-cls  | Classify | 224                         | 5.2<br><sup>0.8 / 4.0 / 0.5</sup>      | 4.5<br><sup>1.6 / 2.2 / 0.7</sup>             | **2.4**<br><sup>1.1 / 0.6 / 0.7</sup>         |
| YOLO26n-pose | Pose     | 640                         | 57.7<br><sup>3.5 / 52.4 / 1.8</sup>    | 15.2<br><sup>3.6 / 9.7 / 1.9</sup>            | **10.8**<br><sup>3.5 / 5.6 / 1.8</sup>        |
| YOLO26n-obb  | OBB      | 1024                        | 50.3<br><sup>3.6 / 45.4 / 1.3</sup>    | **13.9**<br><sup>3.8 / 8.2 / 1.8</sup>        | 21.0<br><sup>8.8 / 10.9 / 1.3</sup>           |

- **Speed** values are **single-image burst latencies** — the mean of 15 runs after 3 warmup runs on [bus.jpg](https://ultralytics.com/images/bus.jpg), measured with the [Flutter plugin's](https://github.com/ultralytics/yolo-flutter-app) on-device benchmark harness on a thermally rested device. Sustained real-time camera frame times run higher (per-frame capture letterboxing plus thermal settling); use the app's on-screen pre/inference/post breakdown for steady-state numbers on your device.
- <sup>1</sup> Semantic QNN uses the in-graph ArgMax class-map output from this release, which replaced erratic 123-1065 ms logits decoding with a stable ~49 ms; the GPU remains slightly faster for semantic at 1024px.

## Supported Tasks

QNN export supports the standard task set available in each model family, including YOLO26 semantic segmentation.

| Task                                                                  | Supported |
| :-------------------------------------------------------------------- | :-------- |
| [Object Detection](https://docs.ultralytics.com/tasks/detect/)        | ✅        |
| [Instance Segmentation](https://docs.ultralytics.com/tasks/segment/)  | ✅        |
| [Semantic Segmentation](https://docs.ultralytics.com/tasks/semantic/) | ✅        |
| [Pose Estimation](https://docs.ultralytics.com/tasks/pose/)           | ✅        |
| [OBB Detection](https://docs.ultralytics.com/tasks/obb/)              | ✅        |
| [Classification](https://docs.ultralytics.com/tasks/classify/)        | ✅        |

## Export to QNN: Converting Your YOLO Model

Export an Ultralytics YOLO model to QNN format for deployment on Snapdragon hardware. The context binary is finalized for a target Hexagon Tensor Processor (HTP) architecture, which you select with the `name` argument — the same argument used to target a chip in [RKNN export](rockchip-rknn.md).

### Supported HTP Architectures

Pass the target architecture via `name` (e.g. `name="73"`). Valid values:

| `name` | Hexagon HTP | Snapdragon platform                   |
| :----- | :---------- | :------------------------------------ |
| `68`   | v68         | Snapdragon 888                        |
| `69`   | v69         | Snapdragon 8 Gen 1 / 8+ Gen 1         |
| `73`   | v73         | Snapdragon 8 Gen 2, X Elite (default) |
| `75`   | v75         | Snapdragon 8 Gen 3                    |
| `79`   | v79         | Snapdragon 8 Elite                    |
| `81`   | v81         | Snapdragon 8 Elite Gen 5              |

!!! note "Platform support"

    QNN export uses the `onnxruntime-qnn` package. Prebuilt wheels are published for **Windows (x64 and ARM64)** and **Linux ARM64 (aarch64)**; on **Linux x86-64** build ONNX Runtime from source with `--use_qnn` (no prebuilt wheel is published, and macOS is not a supported QNN host). QNN context-binary generation runs on an x64 host — Windows x64 or Linux x86-64 — and does not require a Snapdragon device for the export step.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO
        pip install ultralytics
        ```

The `onnxruntime-qnn` package (which provides the ONNX Runtime QNN Execution Provider and bundles the QAIRT libraries) is installed automatically on first export. For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

The QNN format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Inference and validation run on Qualcomm Snapdragon hardware through ONNX Runtime's QNN Execution Provider (the same `onnxruntime-qnn` package used for export). Export your model, then load the exported model on a Snapdragon device to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export to Qualcomm QNN format (INT8, enforced automatically), targeting an HTP architecture via 'name'
        # 'name' can be one of 68, 69, 73, 75, 79, 81 (Snapdragon 888, 8 Gen 1, 8 Gen 2, 8 Gen 3, 8 Elite, 8 Elite Gen 5)
        model.export(format="qnn", name="73")  # creates 'yolo26n_qnn.onnx'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to Qualcomm QNN format for the target HTP architecture
        # 'name' can be one of 68, 69, 73, 75, 79, 81 (Snapdragon 888, 8 Gen 1, 8 Gen 2, 8 Gen 3, 8 Elite, 8 Elite Gen 5)
        yolo export model=yolo26n.pt format=qnn name=73 # creates 'yolo26n_qnn.onnx'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported QNN model (on a Snapdragon device with onnxruntime-qnn)
        model = YOLO("yolo26n_qnn.onnx")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported QNN model
        yolo predict model=yolo26n_qnn.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported QNN model (on a Snapdragon device with onnxruntime-qnn)
        model = YOLO("yolo26n_qnn.onnx")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported QNN model
        yolo val model=yolo26n_qnn.onnx data=coco8.yaml
        ```

### Export Arguments

| Argument   | Type             | Default        | Description                                                                                                                                                                                                |
| :--------- | :--------------- | :------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'qnn'`        | Target format for the exported model, defining compatibility with the Qualcomm QNN runtime.                                                                                                                |
| `imgsz`    | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)`.                                                                                                  |
| `batch`    | `int`            | `1`            | Specifies the export model batch size, which is baked into the generated QNN context binary.                                                                                                               |
| `name`     | `str`            | `'73'`         | Target Hexagon HTP architecture version: `68`, `69`, `73`, `75`, `79`, or `81` (Snapdragon 888, 8 Gen 1, 8 Gen 2, 8 Gen 3, 8 Elite, 8 Elite Gen 5). The context binary is finalized for this architecture. |
| `int8`     | `bool`           | `True`         | Enables INT8 quantization. Required for QNN HTP export — automatically set to `True` if not specified.                                                                                                     |
| `data`     | `str`            | `'coco8.yaml'` | Dataset configuration file used for INT8 calibration. Specifies the calibration image source.                                                                                                              |
| `fraction` | `float`          | `1.0`          | Fraction of the calibration dataset to use for INT8 quantization.                                                                                                                                          |
| `device`   | `str`            | `None`         | Specifies the device for the ONNX export step: GPU (`device=0`) or CPU (`device=cpu`).                                                                                                                     |

!!! note "Precision"

    QNN export quantizes the model to **16-bit activations and INT8 weights** — the recommended accuracy/performance balance for the Hexagon NPU — using the [ONNX Runtime QDQ quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) flow with calibration images from `data`. `int8=True` is enforced automatically.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Output Structure

After a successful export, a self-contained ONNX file is created:

    yolo26n_qnn.onnx   # ONNX wrapping the precompiled QNN context binary and metadata

The `yolo26n_qnn.onnx` file embeds the QNN context binary and is loaded by ONNX Runtime with the QNN Execution Provider on the Snapdragon device. It also carries model metadata such as class names, image size, and task in ONNX `metadata_props`.

## Deploying Exported YOLO QNN Models

QNN models run on Qualcomm Snapdragon hardware, making on-device [model deployment](https://www.ultralytics.com/glossary/model-deployment) straightforward. On a Snapdragon device with `onnxruntime-qnn` installed, run the exported model directly with the Ultralytics API (`yolo predict`/`yolo val`, see [Usage](#usage) above) — Ultralytics loads the context binary through the [ONNX Runtime QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) and selects the HTP (NPU), GPU, or CPU backend.

For custom pipelines, you can also load the context-binary [ONNX](https://onnx.ai/) directly with ONNX Runtime. `onnxruntime-qnn` is a plugin Execution Provider, so register it at runtime:

```python
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# On the Snapdragon device, register the QNN plugin EP and select its device(s)
ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]

options = ort.SessionOptions()
options.add_provider_for_devices(devices, {"backend_path": qnn_ep.get_qnn_htp_path()})
session = ort.InferenceSession("yolo26n_qnn.onnx", sess_options=options)
outputs = session.run(None, {"images": input_tensor})  # input_tensor: float32 NCHW
```

Because the QNN context binary is precompiled, the session loads quickly without recompiling the graph on-device.

## Recommended Workflow

1. **Train** your model using Ultralytics [Train Mode](../modes/train.md)
2. **Export** to QNN format using `model.export(format="qnn")` on a supported platform (Windows x64 or ARM64, or Linux ARM64)
3. **Deploy** the exported `*_qnn.onnx` file to your Snapdragon device
4. **Run** inference with ONNX Runtime and the QNN Execution Provider, selecting the HTP, GPU, or CPU backend

## Real-World Applications

YOLO models running on Qualcomm Snapdragon hardware are well suited for a wide range of [edge AI](https://www.ultralytics.com/glossary/edge-ai) applications:

- **Smartphones**: Real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and scene understanding in camera and photo apps with NPU acceleration.
- **Windows on Snapdragon**: On-device computer vision in Copilot+ PCs without offloading to the cloud.
- **Automotive**: Driver monitoring, occupant detection, and ADAS features on Snapdragon Digital Chassis platforms.
- **XR and Wearables**: Low-power, low-latency perception for AR/VR headsets and smart glasses.
- **IoT and Robotics**: Efficient vision inference on Snapdragon-powered cameras, drones, and embedded systems.

## Summary

In this guide, you've learned how to export Ultralytics YOLO models to the Qualcomm QNN format **locally** with the ONNX Runtime QNN Execution Provider. The export pipeline converts your model to ONNX, then compiles it into a QNN context binary on your host machine — no Qualcomm account or cloud required — producing a `*_qnn.onnx` file optimized for Snapdragon CPU, Adreno GPU, and Hexagon NPU hardware via the QNN/QAIRT runtime.

The combination of [Ultralytics YOLO](https://www.ultralytics.com/yolo) and Qualcomm's on-device AI stack provides an effective solution for running advanced [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) workloads across the broad Snapdragon ecosystem.

For other on-device and mobile deployment targets, see the related [ONNX](onnx.md), [CoreML](coreml.md), [NCNN](ncnn.md), [TFLite](tflite.md), [ExecuTorch](executorch.md), [RKNN](rockchip-rknn.md), [Sony IMX500](sony-imx500.md), and [TensorRT](tensorrt.md) export guides. To compare formats before shipping, use [Benchmark mode](../modes/benchmark.md). For the full list of formats and options, visit the [Export mode](../modes/export.md) documentation and the [integrations guide page](../integrations/index.md).

## FAQ

### How do I export my Ultralytics YOLO model to QNN format?

You can export your model using the `export()` method in Python or via the CLI with `format="qnn"`. The export first creates an ONNX model, then compiles it locally into a QNN context binary using the ONNX Runtime QNN Execution Provider. The `onnxruntime-qnn` package is installed automatically on first export.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="qnn")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=qnn
        ```

### Do I need a Qualcomm account or cloud access?

No. QNN export runs entirely on your local machine using the `onnxruntime-qnn` package, which bundles the QAIRT libraries. No Qualcomm account, API token, or network access is required.

### How does Ultralytics QNN export compare to Qualcomm AI Hub?

[Qualcomm AI Hub](https://aihub.qualcomm.com/) is Qualcomm's cloud service for compiling, profiling, and benchmarking models on hosted Snapdragon devices, and it requires a Qualcomm account. Ultralytics QNN export targets the same QNN/QAIRT runtime (Snapdragon CPU, Adreno GPU, and Hexagon NPU) but compiles the context binary **locally** with the ONNX Runtime QNN Execution Provider — no account, no upload, and no queue. It is the fastest way to go from a `.pt` model to a Snapdragon-ready build directly inside the standard YOLO export workflow.

### Which platforms can I export on?

`onnxruntime-qnn` provides prebuilt wheels for **Windows (x64 and ARM64)** and **Linux ARM64 (aarch64)**; on **Linux x86-64** build ONNX Runtime from source with `--use_qnn` (no prebuilt wheel is published, and macOS is not a supported QNN host). Context-binary generation runs on an x64 host — Windows x64 or Linux x86-64 — and does not require a physical Snapdragon device.

### How do I run YOLO on a Qualcomm Snapdragon NPU?

Export with `model.export(format="qnn")`, copy the resulting `yolo26n_qnn.onnx` file to your Snapdragon device, and run `yolo predict model=yolo26n_qnn.onnx source=image.jpg` (or `yolo val`). Ultralytics loads the context binary through the ONNX Runtime QNN Execution Provider and runs it on the Hexagon NPU — see [Deploying Exported YOLO QNN Models](#deploying-exported-yolo-qnn-models).

### What is the difference between QNN and SNPE?

QNN (Qualcomm AI Engine Direct, part of the QAIRT SDK) is Qualcomm's current inference stack and the recommended replacement for the older Snapdragon Neural Processing Engine (SNPE) SDK. New deployments should target QNN.

### Can I run a QNN model with `yolo predict` and `yolo val`?

Yes, on a Qualcomm Snapdragon device with `onnxruntime-qnn` installed — `YOLO("yolo26n_qnn.onnx")` loads the context binary through the QNN Execution Provider and runs `predict`/`val` like any other format. On an x86 host without QNN hardware the model cannot execute, since the context binary targets the Snapdragon NPU.

### What is the output of a QNN export?

The export creates a self-contained context-binary ONNX file (e.g., `yolo26n_qnn.onnx`) with class names, image size, task, and other model metadata embedded in ONNX `metadata_props`.
