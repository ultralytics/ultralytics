---
title: TFLite Model Export for Edge Devices (Deprecated)
comments: true
description: Historical guide to legacy YOLO TFLite model export and edge deployment. Use LiteRT for new exports.
keywords: YOLO26, TFLite, model export, TensorFlow Lite, edge devices, deployment, Ultralytics, machine learning, on-device inference, model optimization
---

# TFLite Model Export for Deployment (Deprecated)

!!! warning "Deprecated — replaced by LiteRT"

    As of **Ultralytics 8.4.83**, the standalone `tflite` export format has been removed and replaced by the unified **[Google LiteRT](litert.md)** format. LiteRT (_Lite Runtime_) is the next generation and new name for TensorFlow Lite, and it exports the **same `.tflite` model** — now covering mobile, embedded, edge, and browser deployment in one format.

    `format="tflite"` still works but emits a deprecation warning and exports a LiteRT model instead. **Use [`format="litert"`](litert.md)** going forward; for current export instructions and options, see the **[LiteRT export guide](litert.md)**.

<p align="center">
  <img width="75%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/tflite-logo.avif" alt="TensorFlow Lite edge deployment framework">
</p>

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on edge devices or embedded devices requires a format that can ensure seamless performance.

The former TensorFlow Lite or TFLite export format optimized [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models for tasks like [object detection](https://www.ultralytics.com/glossary/object-detection) and [image classification](https://www.ultralytics.com/glossary/image-classification) in edge applications. This guide preserves the legacy TFLite deployment context; use [LiteRT](litert.md) for new exports.

## Why Was TFLite Used for Export?

Introduced by Google in May 2017 as part of their TensorFlow framework, [TensorFlow Lite](https://developers.google.com/edge/litert), or TFLite for short, was an open-source deep learning framework designed for on-device inference, also known as [edge computing](https://www.ultralytics.com/glossary/edge-computing). It gave developers tools to execute trained models on mobile, embedded, and IoT devices, as well as traditional computers.

TensorFlow Lite supported a wide range of platforms, including embedded Linux, Android, iOS, and microcontrollers (MCUs). TFLite exports enabled applications to run models locally and offline.

## Key Features of TFLite Models

TFLite models offer a wide range of key features that enable on-device machine learning by helping developers run their models on mobile, embedded, and edge devices:

- **On-device Optimization**: TFLite optimizes for on-device ML, reducing latency by processing data locally, enhancing privacy by not transmitting personal data, and minimizing model size to save space.

- **Multiple Platform Support**: TFLite offers extensive platform compatibility, supporting Android, iOS, embedded Linux, and microcontrollers.

- **Diverse Language Support**: TFLite is compatible with various programming languages, including Java, Swift, Objective-C, C++, and Python.

- **High Performance**: Achieves superior performance through hardware acceleration and model optimization.

## Measured Performance (historical)

!!! note "Frozen before/after migration record"

    These results were recorded with the Ultralytics Flutter plugin `0.6.8` and LiteRT `2.1.5` on Android 16/API 36
    on a [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) with 12 GB LPDDR5X memory. Its 3 nm
    [Snapdragon 8 Elite Gen 5](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5)
    (SM8850) has an 8-core Qualcomm Oryon CPU (2 Prime cores up to 4.6 GHz and 6 Performance cores up to 3.62 GHz),
    Adreno GPU, and Hexagon NPU. These tables compare the legacy onnx2tf INT8 TFLite assets with candidate
    litert-torch exports evaluated during the migration. The release assets were subsequently overwritten with the
    standardized exports, so these numbers do **not** describe the current `v0.6.6` asset bytes. See the
    [LiteRT performance tables](litert.md#measured-performance) for current measurements.

Both formats ran in the same back-to-back GPU sweep at the input sizes shown. Each cell is the total time
(preprocessing + inference + postprocessing), with the per-stage split beneath it; native logs confirmed that both
formats delegated the complete graph to LiteRT OpenCL (`LITERT_CL`) on the Adreno GPU.

| Model        | Task     | size<br><sup>(pixels)</sup> | Legacy<br><sup>onnx2tf INT8 TFLite<br>(ms)</sup> | Candidate<br><sup>w8a32 LiteRT<br>(ms)</sup> |
| ------------ | -------- | --------------------------- | ------------------------------------------------ | -------------------------------------------- |
| YOLO26n      | Detect   | 640                         | 14.0<br><sup>1.8 / 8.1 / 4.2</sup>               | **13.5**<br><sup>1.9 / 8.1 / 3.5</sup>       |
| YOLO26n-seg  | Segment  | 640                         | 30.1<br><sup>1.9 / 20.3 / 8.0</sup>              | **28.6**<br><sup>1.8 / 20.1 / 6.7</sup>      |
| YOLO26n-sem  | Semantic | 640                         | **26.4**<br><sup>1.9 / 16.4 / 8.1</sup>          | 32.9<br><sup>1.8 / 23.0 / 8.2</sup>          |
| YOLO26n-cls  | Classify | 224                         | 3.5<br><sup>0.9 / 2.2 / 0.4</sup>                | **3.2**<br><sup>1.0 / 2.2 / 0.1</sup>        |
| YOLO26n-pose | Pose     | 640                         | 17.4<br><sup>2.4 / 9.9 / 5.1</sup>               | **14.0**<br><sup>1.9 / 9.3 / 2.8</sup>       |
| YOLO26n-obb  | OBB      | 640                         | 13.9<br><sup>3.0 / 8.3 / 2.7</sup>               | **13.0**<br><sup>2.9 / 7.9 / 2.3</sup>       |

The same migration run also compared YOLO26n detect quantization formats. Inference is the comparable,
format-dependent metric:

| Android format                    | GPU inference (ms) | GPU-compiles |
| --------------------------------- | ------------------ | ------------ |
| onnx2tf INT8 (legacy TFLite)      | **8.6**            | yes          |
| LiteRT w8a32 (candidate)          | 8.4                | yes          |
| LiteRT INT8 (`quantize=8`)        | 11.0               | yes          |
| LiteRT FP32                       | 8.8                | yes          |
| LiteRT w8a16 (`quantize="w8a16"`) | CPU fallback       | no           |

In this run, w8a16 fell back to CPU at about 660 ms total versus about 17 ms for the GPU formats. These results
motivated shipping `w8a32`, but delegate compatibility and performance depend on the device and runtime version.
Benchmark the target device and confirm accelerator placement in its runtime logs.

## Deployment Options in TFLite

Before we look at the LiteRT replacement export example, let's understand how TFLite models are normally used.

TFLite offers various on-device deployment options for machine learning models, including:

- **Deploying with Android and iOS**: Both Android and iOS applications with TFLite can analyze edge-based camera feeds and sensors to detect and identify objects. TFLite also offers native iOS libraries written in [Swift](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/swift) and [Objective-C](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/objc). The architecture diagram below shows the process of deploying a trained model onto Android and iOS platforms using TensorFlow Lite.

 <p align="center">
  <img width="75%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/architecture-diagram-tflite-deployment.avif" alt="TensorFlow Lite deployment architecture for mobile">
</p>

- **Implementing with Embedded Linux**: If running inferences on a [Raspberry Pi](https://www.raspberrypi.com/) using the [Ultralytics Guide](../guides/raspberry-pi.md) does not meet the speed requirements for your use case, you can use an exported TFLite model to accelerate inference times. Additionally, it's possible to further improve performance by utilizing a [Coral Edge TPU device](https://developers.google.com/coral).

- **Deploying with Microcontrollers**: TFLite models can also be deployed on microcontrollers and other devices with only a few kilobytes of memory. The core runtime just fits in 16 KB on an Arm Cortex M3 and can run many basic models. It doesn't require operating system support, any standard C or C++ libraries, or dynamic memory allocation.

## Replace TFLite Export with LiteRT

For new exports, convert your model to LiteRT. The resulting model keeps the `.tflite` file extension.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO26
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO26, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

All [Ultralytics YOLO26 models](../models/index.md) are designed to support export out of the box, making it easy to integrate them into your preferred deployment workflow. You can [view the full list of supported export formats and configuration options](../modes/export.md) to choose the best setup for your application.

The replacement LiteRT format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Export your model, then load the exported `.tflite` model to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to LiteRT format
        model.export(format="litert", imgsz=640)  # use imgsz=224 for classification
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to LiteRT format
        yolo export model=yolo26n.pt format=litert imgsz=640 # use imgsz=224 for classification
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported TFLite model
        model = YOLO("yolo26n.tflite")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported TFLite model
        yolo predict model=yolo26n.tflite source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported TFLite model
        model = YOLO("yolo26n.tflite")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported TFLite model
        yolo val model=yolo26n.tflite data=coco8.yaml
        ```

### Export Arguments

| Argument   | Type             | Default        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------- | ---------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'litert'`     | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                                            |
| `imgsz`    | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                                                                                                                                                                                                                                                             |
| `quantize` | `int` or `str`   | `None`         | Quantization precision: `8` (static INT8, int8 weights + int8 activations; needs calibration `data`/`fraction`), `'w8a16'` (static, int8 weights + int16 activations; needs calibration `data`/`fraction`), `'w8a32'` (dynamic INT8, int8 weights + FP32 activations; no calibration needed), or `32`/unset (FP32). FP16 is not exported separately — an FP32 model runs in FP16 automatically on GPU delegates. Replaces the deprecated `half`/`int8` flags. |
| `batch`    | `int`            | `1`            | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode.                                                                                                                                                                                                                                                                                                                       |
| `data`     | `str`            | `'coco8.yaml'` | Path to the [dataset](../datasets/index.md) configuration file (default: `coco8.yaml`), essential for quantization.                                                                                                                                                                                                                                                                                                                                           |
| `fraction` | `float`          | `1.0`          | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used.                                                                                                                                                                                              |
| `device`   | `str`            | `None`         | Specifies the device for exporting: CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                                                                                                                                                                                                                                                                                                                                                                 |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying Exported YOLO26 TFLite Models

After exporting your Ultralytics YOLO26 model to LiteRT format, you can deploy the resulting `.tflite` model. The primary and recommended first step for running a TFLite model is to use the `YOLO("model.tflite")` method, as outlined in the previous usage code snippet. However, for in-depth instructions on deploying your TFLite models in various other settings, take a look at the following resources:

- **[Android](https://developers.google.com/edge/litert/android)**: A quick start guide for integrating [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Lite into Android applications, providing easy-to-follow steps for setting up and running [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models.

- **[iOS](https://developers.google.com/edge/litert/ios/quickstart)**: Check out this detailed guide for developers on integrating and deploying TensorFlow Lite models in iOS applications, offering step-by-step instructions and resources.

- **[End-To-End Examples](https://github.com/tensorflow/examples/tree/master/lite/examples)**: This page provides an overview of various TensorFlow Lite examples, showcasing practical applications and tutorials designed to help developers implement TensorFlow Lite in their machine learning projects on mobile and edge devices.

## Summary

This guide preserves the legacy TFLite deployment workflow. For new exports, use LiteRT to create `.tflite` models for edge computing environments.

For further details on usage, visit the [TFLite official documentation](https://developers.google.com/edge/litert).

Also, if you're curious about other Ultralytics YOLO26 integrations, check out our [integration guide page](../integrations/index.md). You'll find plenty of helpful information and insights there.

## FAQ

### How do I replace a TFLite export with LiteRT?

For a new export, use the LiteRT format. First, install the required package using:

```bash
pip install ultralytics
```

Then, use the following code snippet to export your model:

```python
from ultralytics import YOLO

# Load a YOLO26 model
model = YOLO("yolo26n.pt")

# Export the model to LiteRT format
model.export(format="litert", imgsz=640)  # use imgsz=224 for classification
```

For CLI users, you can achieve this with:

```bash
yolo export model=yolo26n.pt format=litert imgsz=640 # use imgsz=224 for classification
```

For more details, visit the [Ultralytics export guide](../modes/export.md).

### What are the benefits of using TensorFlow Lite for YOLO26 model deployment?

TensorFlow Lite (TFLite) is an open-source [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) framework designed for on-device inference, making it ideal for deploying YOLO26 models on mobile, embedded, and IoT devices. Key benefits include:

- **On-device optimization**: Minimize latency and enhance privacy by processing data locally.
- **Platform compatibility**: Supports Android, iOS, embedded Linux, and MCU.
- **Performance**: Utilizes hardware acceleration to optimize model speed and efficiency.

To learn more, check out the [TFLite guide](https://developers.google.com/edge/litert).

### Is it possible to run YOLO26 TFLite models on Raspberry Pi?

Yes, you can run YOLO26 TFLite models on Raspberry Pi to improve inference speeds. First, export your model to LiteRT format as explained above. Then, use a tool like TensorFlow Lite Interpreter to execute the model on your Raspberry Pi.

For further optimizations, you might consider using [Coral Edge TPU](https://developers.google.com/coral). For detailed steps, refer to our [Raspberry Pi deployment guide](../guides/raspberry-pi.md) and the [Edge TPU integration guide](../integrations/edge-tpu.md).

### Can I use TFLite models on microcontrollers for YOLO26 predictions?

Yes, TFLite supports deployment on microcontrollers with limited resources. TFLite's core runtime requires only 16 KB of memory on an Arm Cortex M3 and can run basic YOLO26 models. This makes it suitable for deployment on devices with minimal computational power and memory.

To get started, visit the [TFLite Micro for Microcontrollers guide](https://developers.google.com/edge/litert/microcontrollers/overview).

### What platforms are compatible with TFLite exported YOLO26 models?

TensorFlow Lite provides extensive platform compatibility, allowing you to deploy YOLO26 models on a wide range of devices, including:

- **Android and iOS**: Native support through TFLite Android and iOS libraries.
- **Embedded Linux**: Ideal for single-board computers such as Raspberry Pi.
- **Microcontrollers**: Suitable for MCUs with constrained resources.

For more information on deployment options, see our detailed [deployment guide](#deploying-exported-yolo26-tflite-models).

### How do I troubleshoot common issues during YOLO26 model export to LiteRT?

If you encounter errors while exporting YOLO26 models to LiteRT, common solutions include:

- **Check package compatibility**: Ensure you're using compatible versions of Ultralytics, `litert-torch`, and `ai-edge-litert`. Refer to our [installation guide](../quickstart.md).
- **Model support**: Verify that the specific YOLO26 model supports LiteRT export by checking the Ultralytics [export documentation page](../modes/export.md).
- **Quantization issues**: When using INT8 quantization, make sure your dataset path is correctly specified in the `data` parameter.

For additional troubleshooting tips, visit our [Common Issues guide](../guides/yolo-common-issues.md).
