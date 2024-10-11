---
comments: true
description: Learn how to boost your Raspberry Pi's ML performance using Coral Edge TPU with Ultralytics YOLO11. Follow our detailed setup and installation guide.
keywords: Coral Edge TPU, Raspberry Pi, YOLO11, Ultralytics, TensorFlow Lite, ML inference, machine learning, AI, installation guide, setup tutorial
---

# Coral Edge TPU on a Raspberry Pi with Ultralytics YOLO11 🚀

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/edge-tpu-usb-accelerator-and-pi.avif" alt="Raspberry Pi single board computer with USB Edge TPU accelerator">
</p>

## What is a Coral Edge TPU?

The Coral Edge TPU is a compact device that adds an Edge TPU coprocessor to your system. It enables low-power, high-performance ML inference for [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Lite models. Read more at the [Coral Edge TPU home page](https://coral.ai/products/accelerator).

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

## Boost Raspberry Pi Model Performance with Coral Edge TPU

Many people want to run their models on an embedded or mobile device such as a Raspberry Pi, since they are very power efficient and can be used in many different applications. However, the inference performance on these devices is usually poor even when using formats like [onnx](../integrations/onnx.md) or [openvino](../integrations/openvino.md). The Coral Edge TPU is a great solution to this problem, since it can be used with a Raspberry Pi and accelerate inference performance greatly.

## Edge TPU on Raspberry Pi with TensorFlow Lite (New)⭐

The [existing guide](https://coral.ai/docs/accelerator/get-started/) by Coral on how to use the Edge TPU with a Raspberry Pi is outdated, and the current Coral Edge TPU runtime builds do not work with the current TensorFlow Lite runtime versions anymore. In addition to that, Google seems to have completely abandoned the Coral project, and there have not been any updates between 2021 and 2024. This guide will show you how to get the Edge TPU working with the latest versions of the TensorFlow Lite runtime and an updated Coral Edge TPU runtime on a Raspberry Pi single board computer (SBC).

## Prerequisites

- [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) (2GB or more recommended) or [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/) (Recommended)
- [Raspberry Pi OS](https://www.raspberrypi.com/software/) Bullseye/Bookworm (64-bit) with desktop (Recommended)
- [Coral USB Accelerator](https://coral.ai/products/accelerator/)
- A non-ARM based platform for exporting an Ultralytics [PyTorch](https://www.ultralytics.com/glossary/pytorch) model

## Installation Walkthrough

This guide assumes that you already have a working Raspberry Pi OS install and have installed `ultralytics` and all dependencies. To get `ultralytics` installed, visit the [quickstart guide](../quickstart.md) to get setup before continuing here.

### Installing the Edge TPU runtime

First, we need to install the Edge TPU runtime. There are many different versions available, so you need to choose the right version for your operating system.

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

[Download the latest version from here](https://github.com/feranick/libedgetpu/releases).

After downloading the file, you can install it with the following command:

```bash
sudo dpkg -i path/to/package.deb
```

After installing the runtime, you need to plug in your Coral Edge TPU into a USB 3.0 port on your Raspberry Pi. This is because, according to the official guide, a new `udev` rule needs to take effect after installation.

???+ warning "Important"

    If you already have the Coral Edge TPU runtime installed, uninstall it using the following command.

    ```bash
    # If you installed the standard version
    sudo apt remove libedgetpu1-std

    # If you installed the high frequency version
    sudo apt remove libedgetpu1-max
    ```

## Export your model to a Edge TPU compatible model

To use the Edge TPU, you need to convert your model into a compatible format. It is recommended that you run export on Google Colab, x86_64 Linux machine, using the official [Ultralytics Docker container](docker-quickstart.md), or using [Ultralytics HUB](../hub/quickstart.md), since the Edge TPU compiler is not available on ARM. See the [Export Mode](../modes/export.md) for the available arguments.

!!! note "Exporting the model"

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
        yolo export model=path/to/model.pt format=edgetpu  # Export an official model or custom model
        ```

The exported model will be saved in the `<model_name>_saved_model/` folder with the name `<model_name>_full_integer_quant_edgetpu.tflite`.

## Running the model

After exporting your model, you can run inference with it using the following code:

!!! note "Running the model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/edgetpu_model.tflite")  # Load an official model or custom model

        # Run Prediction
        model.predict("path/to/source.png")
        ```

    === "CLI"

        ```bash
        yolo predict model=path/to/edgetpu_model.tflite source=path/to/source.png  # Load an official model or custom model
        ```

Find comprehensive information on the [Predict](../modes/predict.md) page for full prediction mode details.

???+ warning "Important"

    You should run the model using `tflite-runtime` and not `tensorflow`.
    If `tensorflow` is installed, uninstall tensorflow with the following command:

    ```bash
    pip uninstall tensorflow tensorflow-aarch64
    ```

    Then install/update `tflite-runtime`:

    ```
    pip install -U tflite-runtime
    ```

    If you want a `tflite-runtime` wheel for `tensorflow` 2.15.0 download it from [here](https://github.com/feranick/TFlite-builds/releases) and install it using `pip` or your package manager of choice.

## FAQ

### What is a Coral Edge TPU and how does it enhance Raspberry Pi's performance with Ultralytics YOLO11?

The Coral Edge TPU is a compact device designed to add an Edge TPU coprocessor to your system. This coprocessor enables low-power, high-performance [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) inference, particularly optimized for TensorFlow Lite models. When using a Raspberry Pi, the Edge TPU accelerates ML model inference, significantly boosting performance, especially for Ultralytics YOLO11 models. You can read more about the Coral Edge TPU on their [home page](https://coral.ai/products/accelerator).

### How do I install the Coral Edge TPU runtime on a Raspberry Pi?

To install the Coral Edge TPU runtime on your Raspberry Pi, download the appropriate `.deb` package for your Raspberry Pi OS version from [this link](https://github.com/feranick/libedgetpu/releases). Once downloaded, use the following command to install it:

```bash
sudo dpkg -i path/to/package.deb
```

Make sure to uninstall any previous Coral Edge TPU runtime versions by following the steps outlined in the [Installation Walkthrough](#installation-walkthrough) section.

### Can I export my Ultralytics YOLO11 model to be compatible with Coral Edge TPU?

Yes, you can export your Ultralytics YOLO11 model to be compatible with the Coral Edge TPU. It is recommended to perform the export on Google Colab, an x86_64 Linux machine, or using the [Ultralytics Docker container](docker-quickstart.md). You can also use Ultralytics HUB for exporting. Here is how you can export your model using Python and CLI:

!!! note "Exporting the model"

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
        yolo export model=path/to/model.pt format=edgetpu  # Export an official model or custom model
        ```

For more information, refer to the [Export Mode](../modes/export.md) documentation.

### What should I do if TensorFlow is already installed on my Raspberry Pi but I want to use tflite-runtime instead?

If you have TensorFlow installed on your Raspberry Pi and need to switch to `tflite-runtime`, you'll need to uninstall TensorFlow first using:

```bash
pip uninstall tensorflow tensorflow-aarch64
```

Then, install or update `tflite-runtime` with the following command:

```bash
pip install -U tflite-runtime
```

For a specific wheel, such as TensorFlow 2.15.0 `tflite-runtime`, you can download it from [this link](https://github.com/feranick/TFlite-builds/releases) and install it using `pip`. Detailed instructions are available in the section on running the model [Running the Model](#running-the-model).

### How do I run inference with an exported YOLO11 model on a Raspberry Pi using the Coral Edge TPU?

After exporting your YOLO11 model to an Edge TPU-compatible format, you can run inference using the following code snippets:

!!! note "Running the model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/edgetpu_model.tflite")  # Load an official model or custom model

        # Run Prediction
        model.predict("path/to/source.png")
        ```

    === "CLI"

        ```bash
        yolo predict model=path/to/edgetpu_model.tflite source=path/to/source.png  # Load an official model or custom model
        ```

Comprehensive details on full prediction mode features can be found on the [Predict Page](../modes/predict.md).
