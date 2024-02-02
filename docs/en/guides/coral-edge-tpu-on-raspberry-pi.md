---
comments: true
description: Using Ultralytics with a Coral Edge TPU on a Raspberry Pi for increased inference performance.
keywords: Ultralytics, YOLOv8, Object Detection, Coral, Edge Tpu, Raspberry Pi
---

# Coral Edge TPU on a Raspberry Pi with Ultralytics YoloV8. 🚀

## What is a Coral Edge TPU?

The Coral USB Accelerator is a compact device that adds an Edge TPU coprocessor to your system.
It enables low-power, high-performance ML inferencing for TensorFlow Lite models.
You can read more about it [here](https://coral.ai/products/accelerator).

## Why?

Many people want to run their models on a device such as a Raspberry Pi, since they are very power efficient and can be
used in many different applications. However, the inference performance on these kinds of devices is usually poor even
when using formats like onnx or openvino. The Coral Edge TPU is a great solution to this problem, since it can be used
with a Raspberry Pi and accelerate inference performance greatly.

## Why this guide?

The [current guide](https://coral.ai/docs/accelerator/get-started/) by Coral on how to use the Edge TPU with a Raspberry
Pi is outdated, and the current Coral Edge TPU runtime builds do not work with the current tensorflow lite runtime
versions anymore.
In addition to that, Google seems to have completely abandoned the coral project,
and there has not been an update in over three years.
This guide will show you how to get the Edge TPU working with the latest versions of the tensorflow lite runtime and an
updated Coral Edge TPU runtime.

## Prerequisites

- [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) (2GB or more recommended)
  or [Raspberry Pi 5](https://www.raspberrypi.com/products/raspberry-pi-5/) (Recommended)
- [Raspberry Pi OS](https://www.raspberrypi.com/software/) Bullseye/Bookworm (64-bit) with desktop (Recommended)
- [Coral USB Accelerator](https://coral.ai/products/accelerator/)

## Installation

This guide assumes that you already have ultralytics installed.
If not, head to the [quickstart guide](https://docs.ultralytics.com/quickstart/) to get started before continuing here.

### Installing the Edge TPU runtime

First, we need to install the Edge TPU runtime.
There are many different versions available, so you need to choose the version for your device and operating system.

| Raspberry Pi Os | High frequency mode | Version to download                      |
|-----------------|---------------------|------------------------------------------|
| Bullseye 32bit  | No                  | libedgetpu1-std_ ... .bullseye_armhf.deb |
| Bullseye 64bit  | No                  | libedgetpu1-std_ ... .bullseye_arm64.deb |
| Bullseye 32bit  | Yes                 | libedgetpu1-max_ ... .bullseye_armhf.deb |
| Bullseye 64bit  | Yes                 | libedgetpu1-max_ ... .bullseye_arm64.deb |
| Bookworm 32bit  | No                  | libedgetpu1-std_ ... .bookworm_armhf.deb |
| Bookworm 64bit  | No                  | libedgetpu1-std_ ... .bookworm_arm64.deb |
| Bookworm 32bit  | Yes                 | libedgetpu1-max_ ... .bookworm_armhf.deb |
| Bookworm 64bit  | Yes                 | libedgetpu1-max_ ... .bookworm_arm64.deb |

Download the latest version from [here](https://github.com/feranick/libedgetpu/releases).

After downloading the file, you can install it with the following command:

```bash
sudo dpkg -i <path to file>
```

After installing the runtime, you need to plug in your Coral Edge TPU into a USB 3.0 port on your Raspberry Pi.
This is because, according to the official guide, a new `udev` rule needs to take effect after installation.

???+ warning "Important"

    If you already have the Coral Edge TPU runtime installed, uninstall it using the following command.
    ```bash
    # If you installed the standard version
    sudo apt remove libedgetpu1-std 

    # If you installed the high frequency version
    sudo apt remove libedgetpu1-max 
    ```

## Exporting your model to a edge tpu compatible model

To use the Edge TPU, you need to convert your model to a format that is compatible with it.
It is recommended that you export on Google Colab or an x86_64 linux machine,
since the Edge TPU compiler is not available on arm.

!!! Exporting the model

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('path/to/model')  # Load a official model or custom model

        # Export the model
        model.export(format='edgetpu')
        ```

    === "CLI"

        ```bash
        yolo export model=path/to/model format=edgetpu  # Export a official model or custom model
        ```

The exported model will be saved in the `<model_name>_saved_model/` folder with the
name `<model_name>_full_integer_quant_edgetpu.tflite`.

## Running the model

After exporting your model, you can run it using the following code:

!!! Running the model

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('path/to/model')  # Load a official model or custom model

        # Run Prediction
        model.predict()
        ```

For a more detailed guide on how to use the predict mode, head to
the [prediction guide](https://docs.ultralytics.com/modes/predict/).

???+ warning "Important"

    You should run the model through tflite-runtime and not tensorflow.
    Uninstall tensorflow with the following command:

    ```bash
    pip uninstall tensorflow tensorflow-aarch64
    ```

    Then install/update tflite-runtime:

    ```
    pip install -U tflite-runtime
    ```

    If you want a tflite-runtime wheel for tensorflow 2.15.0 download it from [here](https://github.com/feranick/TFlite-builds/releases) and install it through pip.