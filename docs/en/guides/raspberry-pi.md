---
comments: true
description: Learn how to deploy Ultralytics YOLO11 on Raspberry Pi with our comprehensive guide. Get performance benchmarks, setup instructions, and best practices.
keywords: Ultralytics, YOLO11, Raspberry Pi, setup, guide, benchmarks, computer vision, object detection, NCNN, Docker, camera modules
---

# Quick Start Guide: Raspberry Pi with Ultralytics YOLO11

This comprehensive guide provides a detailed walkthrough for deploying Ultralytics YOLO11 on [Raspberry Pi](https://www.raspberrypi.com/) devices. Additionally, it showcases performance benchmarks to demonstrate the capabilities of YOLO11 on these small and powerful devices.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yul4gq_LrOI"
    title="Introducing Raspberry Pi 5" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Raspberry Pi 5 updates and improvements.
</p>

!!! note

    This guide has been tested with Raspberry Pi 4 and Raspberry Pi 5 running the latest [Raspberry Pi OS Bookworm (Debian 12)](https://www.raspberrypi.com/software/operating-systems/). Using this guide for older Raspberry Pi devices such as the Raspberry Pi 3 is expected to work as long as the same Raspberry Pi OS Bookworm is installed.

## What is Raspberry Pi?

Raspberry Pi is a small, affordable, single-board computer. It has become popular for a wide range of projects and applications, from hobbyist home automation to industrial uses. Raspberry Pi boards are capable of running a variety of operating systems, and they offer GPIO (General Purpose Input/Output) pins that allow for easy integration with sensors, actuators, and other hardware components. They come in different models with varying specifications, but they all share the same basic design philosophy of being low-cost, compact, and versatile.

## Raspberry Pi Series Comparison

|                   | Raspberry Pi 3                         | Raspberry Pi 4                         | Raspberry Pi 5                         |
| ----------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| CPU               | Broadcom BCM2837, Cortex-A53 64Bit SoC | Broadcom BCM2711, Cortex-A72 64Bit SoC | Broadcom BCM2712, Cortex-A76 64Bit SoC |
| CPU Max Frequency | 1.4GHz                                 | 1.8GHz                                 | 2.4GHz                                 |
| GPU               | Videocore IV                           | Videocore VI                           | VideoCore VII                          |
| GPU Max Frequency | 400Mhz                                 | 500Mhz                                 | 800Mhz                                 |
| Memory            | 1GB LPDDR2 SDRAM                       | 1GB, 2GB, 4GB, 8GB LPDDR4-3200 SDRAM   | 4GB, 8GB LPDDR4X-4267 SDRAM            |
| PCIe              | N/A                                    | N/A                                    | 1xPCIe 2.0 Interface                   |
| Max Power Draw    | 2.5A@5V                                | 3A@5V                                  | 5A@5V (PD enabled)                     |

## What is Raspberry Pi OS?

[Raspberry Pi OS](https://www.raspberrypi.com/software/) (formerly known as Raspbian) is a Unix-like operating system based on the Debian GNU/Linux distribution for the Raspberry Pi family of compact single-board computers distributed by the Raspberry Pi Foundation. Raspberry Pi OS is highly optimized for the Raspberry Pi with ARM CPUs and uses a modified LXDE desktop environment with the Openbox stacking window manager. Raspberry Pi OS is under active development, with an emphasis on improving the stability and performance of as many Debian packages as possible on Raspberry Pi.

## Flash Raspberry Pi OS to Raspberry Pi

The first thing to do after getting your hands on a Raspberry Pi is to flash a micro-SD card with Raspberry Pi OS, insert into the device and boot into the OS. Follow along with detailed [Getting Started Documentation by Raspberry Pi](https://www.raspberrypi.com/documentation/computers/getting-started.html) to prepare your device for first use.

## Set Up Ultralytics

There are two ways of setting up Ultralytics package on Raspberry Pi to build your next [Computer Vision](https://www.ultralytics.com/glossary/computer-vision-cv) project. You can use either of them.

- [Start with Docker](#start-with-docker)
- [Start without Docker](#start-without-docker)

### Start with Docker

The fastest way to get started with Ultralytics YOLO11 on Raspberry Pi is to run with pre-built docker image for Raspberry Pi.

Execute the below command to pull the Docker container and run on Raspberry Pi. This is based on [arm64v8/debian](https://hub.docker.com/r/arm64v8/debian) docker image which contains Debian 12 (Bookworm) in a Python3 environment.

```bash
t=ultralytics/ultralytics:latest-arm64 && sudo docker pull $t && sudo docker run -it --ipc=host $t
```

After this is done, skip to [Use NCNN on Raspberry Pi section](#use-ncnn-on-raspberry-pi).

### Start without Docker

#### Install Ultralytics Package

Here we will install Ultralytics package on the Raspberry Pi with optional dependencies so that we can export the [PyTorch](https://www.ultralytics.com/glossary/pytorch) models to other different formats.

1. Update packages list, install pip and upgrade to latest

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. Install `ultralytics` pip package with optional dependencies

    ```bash
    pip install ultralytics[export]
    ```

3. Reboot the device

    ```bash
    sudo reboot
    ```

## Use NCNN on Raspberry Pi

Out of all the model export formats supported by Ultralytics, [NCNN](https://docs.ultralytics.com/integrations/ncnn/) delivers the best inference performance when working with Raspberry Pi devices because NCNN is highly optimized for mobile/ embedded platforms (such as ARM architecture). Therefor our recommendation is to use NCNN with Raspberry Pi.

## Convert Model to NCNN and Run Inference

The YOLO11n model in PyTorch format is converted to NCNN to run inference with the exported model.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Export the model to NCNN format
        model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

        # Load the exported NCNN model
        ncnn_model = YOLO("yolo11n_ncnn_model")

        # Run inference
        results = ncnn_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to NCNN format
        yolo export model=yolo11n.pt format=ncnn  # creates 'yolo11n_ncnn_model'

        # Run inference with the exported model
        yolo predict model='yolo11n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
        ```

!!! tip

    For more details about supported export options, visit the [Ultralytics documentation page on deployment options](https://docs.ultralytics.com/guides/model-deployment-options/).

## Raspberry Pi 5 YOLO11 Benchmarks

YOLO11 benchmarks were run by the Ultralytics team on nine different model formats measuring speed and [accuracy](https://www.ultralytics.com/glossary/accuracy): PyTorch, TorchScript, ONNX, OpenVINO, TF SavedModel, TF GraphDef, TF Lite, PaddlePaddle, NCNN. Benchmarks were run on a Raspberry Pi 5 at FP32 [precision](https://www.ultralytics.com/glossary/precision) with default input image size of 640.

### Comparison Chart

We have only included benchmarks for YOLO11n and YOLO11s models because other models sizes are too big to run on the Raspberry Pis and does not offer decent performance.

<div style="text-align: center;">
    <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/rpi-yolo11-benchmarks.avif" alt="YOLO11 benchmarks on RPi 5">
</div>

### Detailed Comparison Table

The below table represents the benchmark results for two different models (YOLO11n, YOLO11s) across nine different formats (PyTorch, TorchScript, ONNX, OpenVINO, TF SavedModel, TF GraphDef, TF Lite, PaddlePaddle, NCNN), running on a Raspberry Pi 5, giving us the status, size, mAP50-95(B) metric, and inference time for each combination.

!!! tip "Performance"

    === "YOLO11n"

        | Format        | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |---------------|--------|-------------------|-------------|------------------------|
        | PyTorch       | ✅      | 5.4               | 0.61        | 524.828                |
        | TorchScript   | ✅      | 10.5              | 0.6082      | 666.874                |
        | ONNX          | ✅      | 10.2              | 0.6082      | 181.818                |
        | OpenVINO      | ✅      | 10.4              | 0.6082      | 530.224                |
        | TF SavedModel | ✅      | 25.8              | 0.6082      | 405.964                |
        | TF GraphDef   | ✅      | 10.3              | 0.6082      | 473.558                |
        | TF Lite       | ✅      | 10.3              | 0.6082      | 324.158                |
        | PaddlePaddle  | ✅      | 20.4              | 0.6082      | 644.312                |
        | NCNN          | ✅      | 10.2              | 0.6106      | 93.938                 |

    === "YOLO11s"

        | Format        | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |---------------|--------|-------------------|-------------|------------------------|
        | PyTorch       | ✅      | 18.4              | 0.7526      | 1226.426               |
        | TorchScript   | ✅      | 36.5              | 0.7416      | 1507.95                |
        | ONNX          | ✅      | 36.3              | 0.7416      | 415.24                 |
        | OpenVINO      | ✅      | 36.4              | 0.7416      | 1167.102               |
        | TF SavedModel | ✅      | 91.1              | 0.7416      | 776.14                 |
        | TF GraphDef   | ✅      | 36.4              | 0.7416      | 1014.396               |
        | TF Lite       | ✅      | 36.4              | 0.7416      | 845.934                |
        | PaddlePaddle  | ✅      | 72.5              | 0.7416      | 1567.824               |
        | NCNN          | ✅      | 36.2              | 0.7419      | 197.358                |

## Reproduce Our Results

To reproduce the above Ultralytics benchmarks on all [export formats](../modes/export.md), run this code:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Benchmark YOLO11n speed and accuracy on the COCO8 dataset for all all export formats
        results = model.benchmarks(data="coco8.yaml", imgsz=640)
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO11n speed and accuracy on the COCO8 dataset for all all export formats
        yolo benchmark model=yolo11n.pt data=coco8.yaml imgsz=640
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results use a dataset with a large number of images, i.e. `data='coco8.yaml' (4 val images), or `data='coco.yaml'` (5000 val images).

## Use Raspberry Pi Camera

When using Raspberry Pi for Computer Vision projects, it can be essentially to grab real-time video feeds to perform inference. The onboard MIPI CSI connector on the Raspberry Pi allows you to connect official Raspberry PI camera modules. In this guide, we have used a [Raspberry Pi Camera Module 3](https://www.raspberrypi.com/products/camera-module-3/) to grab the video feeds and perform inference using YOLO11 models.

!!! tip

    Learn more about the [different camera modules offered by Raspberry Pi](https://www.raspberrypi.com/documentation/accessories/camera.html) and also [how to get started with the Raspberry Pi camera modules](https://www.raspberrypi.com/documentation/computers/camera_software.html#introducing-the-raspberry-pi-cameras).

!!! note

    Raspberry Pi 5 uses smaller CSI connectors than the Raspberry Pi 4 (15-pin vs 22-pin), so you will need a [15-pin to 22pin adapter cable](https://www.raspberrypi.com/products/camera-cable/) to connect to a Raspberry Pi Camera.

### Test the Camera

Execute the following command after connecting the camera to the Raspberry Pi. You should see a live video feed from the camera for about 5 seconds.

```bash
rpicam-hello
```

!!! tip

    Learn more about [`rpicam-hello` usage on official Raspberry Pi documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-hello)

### Inference with Camera

There are 2 methods of using the Raspberry Pi Camera to inference YOLO11 models.

!!! usage

    === "Method 1"

        We can use `picamera2`which comes pre-installed with Raspberry Pi OS to access the camera and inference YOLO11 models.

        !!! example

            === "Python"

                ```python
                import cv2
                from picamera2 import Picamera2

                from ultralytics import YOLO

                # Initialize the Picamera2
                picam2 = Picamera2()
                picam2.preview_configuration.main.size = (1280, 720)
                picam2.preview_configuration.main.format = "RGB888"
                picam2.preview_configuration.align()
                picam2.configure("preview")
                picam2.start()

                # Load the YOLO11 model
                model = YOLO("yolo11n.pt")

                while True:
                    # Capture frame-by-frame
                    frame = picam2.capture_array()

                    # Run YOLO11 inference on the frame
                    results = model(frame)

                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()

                    # Display the resulting frame
                    cv2.imshow("Camera", annotated_frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) == ord("q"):
                        break

                # Release resources and close windows
                cv2.destroyAllWindows()
                ```

    === "Method 2"

        We need to initiate a TCP stream with `rpicam-vid` from the connected camera so that we can use this stream URL as an input when we are inferencing later. Execute the following command to start the TCP stream.

        ```bash
        rpicam-vid -n -t 0 --inline --listen -o tcp://127.0.0.1:8888
        ```

        Learn more about [`rpicam-vid` usage on official Raspberry Pi documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-vid)

        !!! example

            === "Python"

                ```python
                from ultralytics import YOLO

                # Load a YOLO11n PyTorch model
                model = YOLO("yolo11n.pt")

                # Run inference
                results = model("tcp://127.0.0.1:8888")
                ```

            === "CLI"

                ```bash
                yolo predict model=yolo11n.pt source="tcp://127.0.0.1:8888"
                ```

!!! tip

    Check our document on [Inference Sources](https://docs.ultralytics.com/modes/predict/#inference-sources) if you want to change the image/ video input type

## Best Practices when using Raspberry Pi

There are a couple of best practices to follow in order to enable maximum performance on Raspberry Pis running YOLO11.

1. Use an SSD

    When using Raspberry Pi for 24x7 continued usage, it is recommended to use an SSD for the system because an SD card will not be able to withstand continuous writes and might get broken. With the onboard PCIe connector on the Raspberry Pi 5, now you can connect SSDs using an adapter such as the [NVMe Base for Raspberry Pi 5](https://shop.pimoroni.com/products/nvme-base).

2. Flash without GUI

    When flashing Raspberry Pi OS, you can choose to not install the Desktop environment (Raspberry Pi OS Lite) and this can save a bit of RAM on the device, leaving more space for computer vision processing.

## Next Steps

Congratulations on successfully setting up YOLO on your Raspberry Pi! For further learning and support, visit [Ultralytics YOLO11 Docs](../index.md) and [Kashmir World Foundation](https://www.kashmirworldfoundation.org/).

## Acknowledgements and Citations

This guide was initially created by Daan Eeltink for Kashmir World Foundation, an organization dedicated to the use of YOLO for the conservation of endangered species. We acknowledge their pioneering work and educational focus in the realm of object detection technologies.

For more information about Kashmir World Foundation's activities, you can visit their [website](https://www.kashmirworldfoundation.org/).

## FAQ

### How do I set up Ultralytics YOLO11 on a Raspberry Pi without using Docker?

To set up Ultralytics YOLO11 on a Raspberry Pi without Docker, follow these steps:

1. Update the package list and install `pip`:
    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```
2. Install the Ultralytics package with optional dependencies:
    ```bash
    pip install ultralytics[export]
    ```
3. Reboot the device to apply changes:
    ```bash
    sudo reboot
    ```

For detailed instructions, refer to the [Start without Docker](#start-without-docker) section.

### Why should I use Ultralytics YOLO11's NCNN format on Raspberry Pi for AI tasks?

Ultralytics YOLO11's NCNN format is highly optimized for mobile and embedded platforms, making it ideal for running AI tasks on Raspberry Pi devices. NCNN maximizes inference performance by leveraging ARM architecture, providing faster and more efficient processing compared to other formats. For more details on supported export options, visit the [Ultralytics documentation page on deployment options](../modes/export.md).

### How can I convert a YOLO11 model to NCNN format for use on Raspberry Pi?

You can convert a PyTorch YOLO11 model to NCNN format using either Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Export the model to NCNN format
        model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

        # Load the exported NCNN model
        ncnn_model = YOLO("yolo11n_ncnn_model")

        # Run inference
        results = ncnn_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to NCNN format
        yolo export model=yolo11n.pt format=ncnn  # creates 'yolo11n_ncnn_model'

        # Run inference with the exported model
        yolo predict model='yolo11n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
        ```

For more details, see the [Use NCNN on Raspberry Pi](#use-ncnn-on-raspberry-pi) section.

### What are the hardware differences between Raspberry Pi 4 and Raspberry Pi 5 relevant to running YOLO11?

Key differences include:

- **CPU**: Raspberry Pi 4 uses Broadcom BCM2711, Cortex-A72 64-bit SoC, while Raspberry Pi 5 uses Broadcom BCM2712, Cortex-A76 64-bit SoC.
- **Max CPU Frequency**: Raspberry Pi 4 has a max frequency of 1.8GHz, whereas Raspberry Pi 5 reaches 2.4GHz.
- **Memory**: Raspberry Pi 4 offers up to 8GB of LPDDR4-3200 SDRAM, while Raspberry Pi 5 features LPDDR4X-4267 SDRAM, available in 4GB and 8GB variants.

These enhancements contribute to better performance benchmarks for YOLO11 models on Raspberry Pi 5 compared to Raspberry Pi 4. Refer to the [Raspberry Pi Series Comparison](#raspberry-pi-series-comparison) table for more details.

### How can I set up a Raspberry Pi Camera Module to work with Ultralytics YOLO11?

There are two methods to set up a Raspberry Pi Camera for YOLO11 inference:

1. **Using `picamera2`**:

    ```python
    import cv2
    from picamera2 import Picamera2

    from ultralytics import YOLO

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (1280, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    model = YOLO("yolo11n.pt")

    while True:
        frame = picam2.capture_array()
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Camera", annotated_frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
    ```

2. **Using a TCP Stream**:

    ```bash
    rpicam-vid -n -t 0 --inline --listen -o tcp://127.0.0.1:8888
    ```

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    results = model("tcp://127.0.0.1:8888")
    ```

For detailed setup instructions, visit the [Inference with Camera](#inference-with-camera) section.
