---
title: YOLO26 on Raspberry Pi with Coral Edge TPU
comments: true
description: Accelerate Ultralytics YOLO26 inference on a Raspberry Pi with the Coral Edge TPU. Step-by-step guide to install the runtime, export to Edge TPU format, and run fast low-power inference.
keywords: Coral Edge TPU, Raspberry Pi, YOLO26, Ultralytics, TensorFlow Lite, tflite-runtime, edge AI, low-power inference, edgetpu export, USB Accelerator, embedded AI, ML inference
---

# How to Run Ultralytics YOLO26 on a Raspberry Pi with a Coral Edge TPU

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/edge-tpu-usb-accelerator-and-pi.avif" alt="Raspberry Pi with Edge TPU accelerator">
</p>

A [Raspberry Pi](raspberry-pi.md) is a power-efficient, affordable platform for running computer vision at the edge, but on-device inference is slow even with optimized formats like [ONNX](../integrations/onnx.md) or [OpenVINO](../integrations/openvino.md). Pairing the Pi with a Coral Edge TPU coprocessor offloads inference to dedicated hardware and dramatically speeds it up. This guide shows you how to install the runtime, export an Ultralytics YOLO26 model to the Edge TPU format, and run accelerated inference.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/w4yHORvDBw0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Inference on Raspberry Pi using Google Coral Edge TPU
</p>

## Why Use a Coral Edge TPU?

The Coral Edge TPU is a compact device that adds an Edge TPU coprocessor to your system, enabling low-power, high-performance ML inference for TensorFlow Lite models. It is a great fit for embedded and mobile deployments where a CPU alone cannot keep up:

- **Faster inference** — the Edge TPU accelerates quantized models far beyond what the Raspberry Pi CPU achieves on its own.
- **Low power draw** — it delivers high throughput per watt, ideal for battery- or solar-powered deployments.
- **Plug-and-play** — the [USB Accelerator](https://developers.google.com/coral) connects over USB 3.0, so no extra hardware integration is required.

!!! note "Updated runtime for current TensorFlow Lite"

    The [official Coral guide](https://gweb-coral-full.uc.r.appspot.com/docs/accelerator/get-started/) is outdated: the original Coral runtime builds no longer work with current TensorFlow Lite runtime versions, and the project saw no updates between 2021 and 2025. This guide uses an actively maintained Edge TPU runtime and the latest `tflite-runtime` so the accelerator works on a current Raspberry Pi OS install.

## Prerequisites

- [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) (2GB or more recommended) or [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/) (Recommended)
- [Raspberry Pi OS](https://www.raspberrypi.com/software/) Bullseye/Bookworm (64-bit) with desktop (Recommended)
- [Coral USB Accelerator](https://developers.google.com/coral)
- A non-ARM platform (Google Colab, an x86_64 Linux machine, or the [Ultralytics Docker container](docker-quickstart.md)) for exporting the model, since the Edge TPU compiler is not available on ARM

This guide assumes you already have a working Raspberry Pi OS install with `ultralytics` and its dependencies installed. If not, follow the [quickstart guide](../quickstart.md) first.

With the prerequisites ready, the workflow has three steps: [install the Edge TPU runtime](#install-the-edge-tpu-runtime) on the Pi, [export your model](#export-your-model-to-edge-tpu-format) on a non-ARM machine, and [run inference](#run-inference-on-the-edge-tpu) back on the Pi.

## Install the Edge TPU Runtime

The runtime ships in several builds, so pick the one that matches your operating system. The high-frequency build runs the Edge TPU at a higher clock speed for better performance, but it can cause thermal throttling — use some form of cooling if you choose it.

| Raspberry Pi OS | High frequency mode | Version to download                        |
| --------------- | :-----------------: | ------------------------------------------ |
| Bullseye 32bit  |         No          | `libedgetpu1-std_ ... .bullseye_armhf.deb` |
| Bullseye 64bit  |         No          | `libedgetpu1-std_ ... .bullseye_arm64.deb` |
| Bullseye 32bit  |         Yes         | `libedgetpu1-max_ ... .bullseye_armhf.deb` |
| Bullseye 64bit  |         Yes         | `libedgetpu1-max_ ... .bullseye_arm64.deb` |
| Bookworm 32bit  |         No          | `libedgetpu1-std_ ... .bookworm_armhf.deb` |
| Bookworm 64bit  |         No          | `libedgetpu1-std_ ... .bookworm_arm64.deb` |
| Bookworm 32bit  |         Yes         | `libedgetpu1-max_ ... .bookworm_armhf.deb` |
| Bookworm 64bit  |         Yes         | `libedgetpu1-max_ ... .bookworm_arm64.deb` |

[Download the latest version from here](https://github.com/feranick/libedgetpu/releases), then install the `.deb` package:

```bash
sudo dpkg -i path/to/package.deb
```

After installing the runtime, plug your Coral Edge TPU into a USB 3.0 port on the Raspberry Pi so the new `udev` rule can take effect.

???+ warning "Remove any existing runtime first"

    If you already have the Coral Edge TPU runtime installed, uninstall it before installing a new build.

    ```bash
    # If you installed the standard version
    sudo apt remove libedgetpu1-std

    # If you installed the high-frequency version
    sudo apt remove libedgetpu1-max
    ```

## Export Your Model to Edge TPU Format

To use the Edge TPU, convert your model to a compatible format. Run the export on a non-ARM platform — Google Colab, an x86_64 Linux machine, the official [Ultralytics Docker container](docker-quickstart.md), or [Ultralytics Platform](../platform/quickstart.md) — since the Edge TPU compiler is not available on ARM. See the [Export mode](../modes/export.md) for the available arguments.

!!! example "Exporting the model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/model.pt")  # Load an official model or custom model

        # Export the model
        model.export(format="edgetpu")
        ```

    === "CLI"

        ```bash
        yolo export model=path/to/model.pt format=edgetpu # Export an official model or custom model
        ```

The exported model is saved in the `<model_name>_saved_model/` folder as `<model_name>_full_integer_quant_edgetpu.tflite`.

!!! warning "Keep the `_edgetpu.tflite` suffix"

    The file name must end with `_edgetpu.tflite`. If you rename it to anything else, Ultralytics will load it as a plain TensorFlow Lite model instead of detecting the Edge TPU and the accelerator will not be used.

## Run Inference on the Edge TPU

Before running the model, install the correct libraries on the Raspberry Pi. If TensorFlow is already installed, uninstall it first:

```bash
pip uninstall tensorflow tensorflow-aarch64
```

Then install or update `tflite-runtime`:

```bash
pip install -U tflite-runtime
```

Now you can run inference:

!!! example "Running the model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/<model_name>_full_integer_quant_edgetpu.tflite")  # Load an official model or custom model

        # Run Prediction
        model.predict("path/to/source.png")
        ```

    === "CLI"

        ```bash
        yolo predict model=path/to/ source=path/to/source.png < model_name > _full_integer_quant_edgetpu.tflite # Load an official model or custom model
        ```

Find full prediction-mode details on the [Predict](../modes/predict.md) page.

!!! note "Inference with multiple Edge TPUs"

    If you have multiple Edge TPUs, you can select a specific one with the `device` argument.

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/<model_name>_full_integer_quant_edgetpu.tflite")  # Load an official model or custom model

        # Run Prediction
        model.predict("path/to/source.png")  # Inference defaults to the first TPU

        model.predict("path/to/source.png", device="tpu:0")  # Select the first TPU

        model.predict("path/to/source.png", device="tpu:1")  # Select the second TPU
        ```

## Benchmarks

The figures below were measured with Raspberry Pi OS Bookworm 64-bit and a USB Coral Edge TPU. They show inference time only (pre-/postprocessing excluded) and serve as a relative reference for the acceleration the Edge TPU provides across Pi models and modes.

!!! note "About these numbers"

    These benchmarks were recorded with YOLOv8 models. Absolute inference times vary by model version and image size, but the relative speedups between Pi models and clock modes hold.

=== "Raspberry Pi 4B 2GB"

    | Image Size | Model   | Standard Inference Time (ms) | High-Frequency Inference Time (ms) |
    |------------|---------|------------------------------|------------------------------------|
    | 320        | YOLOv8n | 32.2                         | 26.7                               |
    | 320        | YOLOv8s | 47.1                         | 39.8                               |
    | 512        | YOLOv8n | 73.5                         | 60.7                               |
    | 512        | YOLOv8s | 149.6                        | 125.3                              |

=== "Raspberry Pi 5 8GB"

    | Image Size | Model   | Standard Inference Time (ms) | High Frequency Inference Time (ms) |
    |------------|---------|------------------------------|------------------------------------|
    | 320        | YOLOv8n | 22.2                         | 16.7                               |
    | 320        | YOLOv8s | 40.1                         | 32.2                               |
    | 512        | YOLOv8n | 53.5                         | 41.6                               |
    | 512        | YOLOv8s | 132.0                        | 103.3                              |

On average:

- The Raspberry Pi 5 is 22% faster with the standard mode than the Raspberry Pi 4B.
- The Raspberry Pi 5 is 30.2% faster with the high-frequency mode than the Raspberry Pi 4B.
- The high-frequency mode is 28.4% faster than the standard mode.

## Conclusion

A Coral Edge TPU turns a Raspberry Pi into a capable, low-power inference device for Ultralytics YOLO26. Export your model on a non-ARM machine, keep the `_edgetpu.tflite` suffix, and run it with `tflite-runtime` on the Pi to get accelerated edge inference. For more deployment options, see the [Raspberry Pi](raspberry-pi.md) guide.

## FAQ

### What is a Coral Edge TPU and how does it enhance Raspberry Pi's performance with Ultralytics YOLO26?

The Coral Edge TPU is a compact device that adds an Edge TPU coprocessor to your system. This coprocessor enables low-power, high-performance ML inference, particularly optimized for TensorFlow Lite models. On a Raspberry Pi, it accelerates inference well beyond what the CPU achieves alone, which significantly boosts performance for Ultralytics YOLO26 models.

### How do I install the Coral Edge TPU runtime on a Raspberry Pi?

Download the appropriate `.deb` package for your Raspberry Pi OS version from [this link](https://github.com/feranick/libedgetpu/releases), then install it:

```bash
sudo dpkg -i path/to/package.deb
```

Make sure to uninstall any previous Coral Edge TPU runtime versions by following the steps in the [Install the Edge TPU Runtime](#install-the-edge-tpu-runtime) section.

### Can I export my Ultralytics YOLO26 model to be compatible with Coral Edge TPU?

Yes. Run the export on Google Colab, an x86_64 Linux machine, or the [Ultralytics Docker container](docker-quickstart.md); you can also use [Ultralytics Platform](../platform/quickstart.md). Here is how to export with Python and CLI:

!!! example "Exporting the model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/model.pt")  # Load an official model or custom model

        # Export the model
        model.export(format="edgetpu")
        ```

    === "CLI"

        ```bash
        yolo export model=path/to/model.pt format=edgetpu # Export an official model or custom model
        ```

For more information, refer to the [Export mode](../modes/export.md) documentation.

### What should I do if TensorFlow is already installed on my Raspberry Pi, but I want to use tflite-runtime instead?

If you have TensorFlow installed and need to switch to `tflite-runtime`, uninstall TensorFlow first:

```bash
pip uninstall tensorflow tensorflow-aarch64
```

Then install or update `tflite-runtime`:

```bash
pip install -U tflite-runtime
```

For detailed instructions, refer to the [Run Inference on the Edge TPU](#run-inference-on-the-edge-tpu) section.

### How do I run inference with an exported YOLO26 model on a Raspberry Pi using the Coral Edge TPU?

After exporting your YOLO26 model to an Edge TPU-compatible format, run inference with the following snippets. The model file must keep the `_edgetpu.tflite` suffix so Ultralytics loads it on the Edge TPU:

!!! example "Running the model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/<model_name>_full_integer_quant_edgetpu.tflite")  # Load an official model or custom model

        # Run Prediction
        model.predict("path/to/source.png")
        ```

    === "CLI"

        ```bash
        yolo predict model=path/to/ source=path/to/source.png < model_name > _full_integer_quant_edgetpu.tflite # Load an official model or custom model
        ```

Comprehensive details on prediction mode are on the [Predict](../modes/predict.md) page.
