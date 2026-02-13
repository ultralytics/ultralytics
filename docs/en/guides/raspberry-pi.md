---
comments: true
description: Learn how to deploy Ultralytics YOLO26 on Raspberry Pi with our comprehensive guide. Get performance benchmarks, setup instructions, and best practices.
keywords: Ultralytics, YOLO26, Raspberry Pi, setup, guide, benchmarks, computer vision, object detection, NCNN, Docker, camera modules
---

# Quick Start Guide: Raspberry Pi with Ultralytics YOLO26

This comprehensive guide provides a detailed walkthrough for deploying Ultralytics YOLO26 on [Raspberry Pi](https://www.raspberrypi.com/) devices. Additionally, it showcases performance benchmarks to demonstrate the capabilities of YOLO26 on these small and powerful devices.

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

The fastest way to get started with Ultralytics YOLO26 on Raspberry Pi is to run with pre-built docker image for Raspberry Pi.

Execute the below command to pull the Docker container and run on Raspberry Pi. This is based on [arm64v8/debian](https://hub.docker.com/r/arm64v8/debian) docker image which contains Debian 12 (Bookworm) in a Python3 environment.

```bash
t=ultralytics/ultralytics:latest-arm64
sudo docker pull $t && sudo docker run -it --ipc=host $t
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

Out of all the model export formats supported by Ultralytics, [NCNN](https://docs.ultralytics.com/integrations/ncnn/) delivers the best inference performance when working with Raspberry Pi devices because NCNN is highly optimized for mobile/ embedded platforms (such as ARM architecture).

## Convert Model to NCNN and Run Inference

The YOLO26n model in PyTorch format is converted to NCNN to run inference with the exported model.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26n PyTorch model
        model = YOLO("yolo26n.pt")

        # Export the model to NCNN format
        model.export(format="ncnn")  # creates 'yolo26n_ncnn_model'

        # Load the exported NCNN model
        ncnn_model = YOLO("yolo26n_ncnn_model")

        # Run inference
        results = ncnn_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to NCNN format
        yolo export model=yolo26n.pt format=ncnn # creates 'yolo26n_ncnn_model'

        # Run inference with the exported model
        yolo predict model='yolo26n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
        ```

!!! tip

    For more details about supported export options, visit the [Ultralytics documentation page on deployment options](https://docs.ultralytics.com/guides/model-deployment-options/).

## YOLO26 Performance Improvements over YOLO11

YOLO26 is specifically designed to run on hardware-constrained devices such as the Raspberry Pi 5. Compared to YOLO11n, YOLO26n achieves a ~15% increase in FPS (6.79 → 7.79) while also delivering higher mAP (40.1 vs 39.5) at 640 input size with ONNX-exported models on the Raspberry Pi 5. The table and chart below showcase this comparison.

<figure style="text-align: center;">
    <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolo26-vs-yolo11-rpi5-onnx-benchmarks.avif" alt="YOLO26 benchmarks on RPi 5">
    <figcaption style="font-style: italic; color: gray;">Benchmarked with Ultralytics 8.4.14</figcaption>
</figure>

!!! tip "Performance"

    === "YOLO26 (ONNX)"

        | Model   	| mAP50-95(B) 	| Inference time (ms/im) 	|
        |---------	|-------------	|------------------------	|
        | YOLO26n 	| 40.1        	| 128.42                 	|
        | YOLO26s 	| 47.8        	| 352.84                 	|
        | YOLO26m 	| 52.5        	| 993.78                 	|
        | YOLO26l 	| 54.4        	| 1259.46                	|
        | YOLO26x 	| 56.9        	| 2636.26                	|


    === "YOLO11 (ONNX)"

        | Model   	| mAP50-95(B) 	| Inference time (ms/im) 	|
        |---------	|-------------	|------------------------	|
        | YOLO11n 	| 39.5        	| 147.20                 	|
        | YOLO11s 	| 47.0        	| 366.83                 	|
        | YOLO11m 	| 51.5        	| 997.46                 	|
        | YOLO11l 	| 53.4        	| 1274.95                	|
        | YOLO11x 	| 54.7        	| 2646.76                	|

    Benchmarked with Ultralytics 8.4.14.

## Raspberry Pi 5 YOLO26 Benchmarks

YOLO26 benchmarks were run by the Ultralytics team on ten different model formats measuring speed and [accuracy](https://www.ultralytics.com/glossary/accuracy): PyTorch, TorchScript, ONNX, OpenVINO, TF SavedModel, TF GraphDef, TF Lite, MNN, NCNN, ExecuTorch. Benchmarks were run on a Raspberry Pi 5 at FP32 [precision](https://www.ultralytics.com/glossary/precision) with default input image size of 640.

### Comparison Chart

We have only included benchmarks for YOLO26n and YOLO26s models because other model sizes are too big to run on the Raspberry Pis and do not offer decent performance.

<figure style="text-align: center;">
    <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/raspberry-pi-yolo26-benchmarks.avif" alt="YOLO26 benchmarks on RPi 5">
    <figcaption style="font-style: italic; color: gray;">Benchmarked with Ultralytics 8.4.1</figcaption>
</figure>

### Detailed Comparison Table

The below table represents the benchmark results for two different models (YOLO26n, YOLO26s) across ten different formats (PyTorch, TorchScript, ONNX, OpenVINO, TF SavedModel, TF GraphDef, TF Lite, MNN, NCNN, ExecuTorch), running on a Raspberry Pi 5, giving us the status, size, mAP50-95(B) metric, and inference time for each combination.

!!! tip "Performance"

    === "YOLO26n"

        | Format        | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |---------------|--------|-------------------|-------------|------------------------|
        | PyTorch       | ✅      | 5.3               | 0.4798      | 302.15                |
        | TorchScript   | ✅      | 9.8              | 0.4764      | 357.58                |
        | ONNX          | ✅      | 9.5              | 0.4764      | 130.33                |
        | OpenVINO      | ✅      | 9.6              | 0.4818      | 70.74                 |
        | TF SavedModel | ✅      | 24.6              | 0.4764      | 213.58                |
        | TF GraphDef   | ✅      | 9.5              | 0.4764      | 213.5                |
        | TF Lite       | ✅      | 9.9              | 0.4764      | 251.41                |
        | MNN           | ✅      | 9.4              | 0.4784      | 90.89                |
        | NCNN          | ✅      | 9.4              | 0.4805      | 67.69                 |
        | ExecuTorch    | ✅      | 9.4              | 0.4764      | 148.36                 |

    === "YOLO26s"

        | Format        | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |---------------|--------|-------------------|-------------|------------------------|
        | PyTorch       | ✅      | 19.5              | 0.5740      | 836.54                 |
        | TorchScript   | ✅      | 36.8              | 0.5665      | 1032.25               |
        | ONNX          | ✅      | 36.5              | 0.5665      | 351.96                |
        | OpenVINO      | ✅      | 36.7              | 0.5654      | 158.6                |
        | TF SavedModel | ✅      | 92.2               | 0.5665      | 507.6                |
        | TF GraphDef   | ✅      | 36.5              | 0.5665      | 525.64                 |
        | TF Lite       | ✅      | 36.9               | 0.5665      | 805.3               |
        | MNN           | ✅      | 36.4              | 0.5644      | 236.47                |
        | NCNN          | ✅      | 36.4              | 0.5697      | 168.47                |
        | ExecuTorch    | ✅      | 36.5              | 0.5665      | 388.72                |

    Benchmarked with Ultralytics 8.4.1

    !!! note

        Inference time does not include pre/ post-processing.

### Reproduce Our Results

To reproduce the above Ultralytics benchmarks on all [export formats](../modes/export.md), run this code:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26n PyTorch model
        model = YOLO("yolo26n.pt")

        # Benchmark YOLO26n speed and accuracy on the COCO128 dataset for all export formats
        results = model.benchmark(data="coco128.yaml", imgsz=640)
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO26n speed and accuracy on the COCO128 dataset for all export formats
        yolo benchmark model=yolo26n.pt data=coco128.yaml imgsz=640
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results, use a dataset with a large number of images, e.g., `data='coco.yaml'` (5000 val images).

## Use Raspberry Pi Camera

When using Raspberry Pi for Computer Vision projects, it can be essential to grab real-time video feeds to perform inference. The onboard MIPI CSI connector on the Raspberry Pi allows you to connect official Raspberry PI camera modules. In this guide, we have used a [Raspberry Pi Camera Module 3](https://www.raspberrypi.com/products/camera-module-3/) to grab the video feeds and perform inference using YOLO26 models.

!!! tip

    Learn more about the [different camera modules offered by Raspberry Pi](https://www.raspberrypi.com/documentation/accessories/camera.html) and also [how to get started with the Raspberry Pi camera modules](https://www.raspberrypi.com/documentation/computers/camera_software.html#introducing-the-raspberry-pi-cameras).

!!! note

    Raspberry Pi 5 uses smaller CSI connectors than the Raspberry Pi 4 (15-pin vs 22-pin), so you will need a [15-pin to 22-pin adapter cable](https://www.raspberrypi.com/products/camera-cable/) to connect to a Raspberry Pi Camera.

### Test the Camera

Execute the following command after connecting the camera to the Raspberry Pi. You should see a live video feed from the camera for about 5 seconds.

```bash
rpicam-hello
```

!!! tip

    Learn more about [`rpicam-hello` usage on official Raspberry Pi documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html#rpicam-hello)

### Inference with Camera

There are 2 methods of using the Raspberry Pi Camera to run inference on YOLO26 models.

!!! usage

    === "Method 1"

        We can use `picamera2` which comes pre-installed with Raspberry Pi OS to access the camera and run inference on YOLO26 models.

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

                # Load the YOLO26 model
                model = YOLO("yolo26n.pt")

                while True:
                    # Capture frame-by-frame
                    frame = picam2.capture_array()

                    # Run YOLO26 inference on the frame
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

                # Load a YOLO26n PyTorch model
                model = YOLO("yolo26n.pt")

                # Run inference
                results = model("tcp://127.0.0.1:8888")
                ```

            === "CLI"

                ```bash
                yolo predict model=yolo26n.pt source="tcp://127.0.0.1:8888"
                ```

!!! tip

    Check our document on [Inference Sources](https://docs.ultralytics.com/modes/predict/#inference-sources) if you want to change the image/video input type

## Best Practices when using Raspberry Pi

There are a couple of best practices to follow in order to enable maximum performance on Raspberry Pis running YOLO26.

1. Use an SSD

    When using Raspberry Pi for 24x7 continued usage, it is recommended to use an SSD for the system because an SD card will not be able to withstand continuous writes and might get broken. With the onboard PCIe connector on the Raspberry Pi 5, now you can connect SSDs using an adapter such as the [NVMe Base for Raspberry Pi 5](https://shop.pimoroni.com/products/nvme-base).

2. Flash without GUI

    When flashing Raspberry Pi OS, you can choose to not install the Desktop environment (Raspberry Pi OS Lite) and this can save a bit of RAM on the device, leaving more space for computer vision processing.

3. Overclock Raspberry Pi

    If you want a little boost in performance while running Ultralytics YOLO26 models on Raspberry Pi 5, you can overclock the CPU from its base 2.4GHz to 2.9GHz and the GPU from 800MHz to 1GHz. If the system becomes unstable or crashes, reduce the overclock values by 100MHz increments. Ensure proper cooling is in place, as overclocking increases heat generation and may lead to thermal throttling.

    a. Upgrade the software

    ```bash
    sudo apt update && sudo apt dist-upgrade
    ```

    b. Open to edit the configuration file

    ```bash
    sudo nano /boot/firmware/config.txt
    ```

    c. Add the following lines at the bottom

    ```bash
    arm_freq=3000
    gpu_freq=1000
    force_turbo=1
    ```

    d. Save and exit by pressing CTRL + X, then Y, and hit ENTER

    e. Reboot the Raspberry Pi

## Next Steps

You have successfully set up YOLO on your Raspberry Pi. For further learning and support, visit [Ultralytics YOLO26 Docs](../index.md) and [Kashmir World Foundation](https://www.kashmirworldfoundation.org/).

## Acknowledgments and Citations

This guide was initially created by Daan Eeltink for Kashmir World Foundation, an organization dedicated to the use of YOLO for the conservation of endangered species. We acknowledge their pioneering work and educational focus in the realm of object detection technologies.

For more information about Kashmir World Foundation's activities, you can visit their [website](https://www.kashmirworldfoundation.org/).

## FAQ

### How do I set up Ultralytics YOLO26 on a Raspberry Pi without using Docker?

To set up Ultralytics YOLO26 on a Raspberry Pi without Docker, follow these steps:

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

### Why should I use Ultralytics YOLO26's NCNN format on Raspberry Pi for AI tasks?

Ultralytics YOLO26's NCNN format is highly optimized for mobile and embedded platforms, making it ideal for running AI tasks on Raspberry Pi devices. NCNN maximizes inference performance by leveraging ARM architecture, providing faster and more efficient processing compared to other formats. For more details on supported export options, visit the [Ultralytics documentation page on deployment options](https://docs.ultralytics.com/guides/model-deployment-options/).

### How can I convert a YOLO26 model to NCNN format for use on Raspberry Pi?

You can convert a PyTorch YOLO26 model to NCNN format using either Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26n PyTorch model
        model = YOLO("yolo26n.pt")

        # Export the model to NCNN format
        model.export(format="ncnn")  # creates 'yolo26n_ncnn_model'

        # Load the exported NCNN model
        ncnn_model = YOLO("yolo26n_ncnn_model")

        # Run inference
        results = ncnn_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to NCNN format
        yolo export model=yolo26n.pt format=ncnn # creates 'yolo26n_ncnn_model'

        # Run inference with the exported model
        yolo predict model='yolo26n_ncnn_model' source='https://ultralytics.com/images/bus.jpg'
        ```

For more details, see the [Use NCNN on Raspberry Pi](#use-ncnn-on-raspberry-pi) section.

### What are the hardware differences between Raspberry Pi 4 and Raspberry Pi 5 relevant to running YOLO26?

Key differences include:

- **CPU**: Raspberry Pi 4 uses Broadcom BCM2711, Cortex-A72 64-bit SoC, while Raspberry Pi 5 uses Broadcom BCM2712, Cortex-A76 64-bit SoC.
- **Max CPU Frequency**: Raspberry Pi 4 has a max frequency of 1.8GHz, whereas Raspberry Pi 5 reaches 2.4GHz.
- **Memory**: Raspberry Pi 4 offers up to 8GB of LPDDR4-3200 SDRAM, while Raspberry Pi 5 features LPDDR4X-4267 SDRAM, available in 4GB and 8GB variants.

These enhancements contribute to better performance benchmarks for YOLO26 models on Raspberry Pi 5 compared to Raspberry Pi 4. Refer to the [Raspberry Pi Series Comparison](#raspberry-pi-series-comparison) table for more details.

### How can I set up a Raspberry Pi Camera Module to work with Ultralytics YOLO26?

There are two methods to set up a Raspberry Pi Camera for YOLO26 inference:

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

    model = YOLO("yolo26n.pt")

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

    model = YOLO("yolo26n.pt")
    results = model("tcp://127.0.0.1:8888")
    ```

For detailed setup instructions, visit the [Inference with Camera](#inference-with-camera) section.

## Raspberry Pi AI Kit (Hailo-8L) 

The Raspberry Pi AI Kit integrates a 13 TOPS Hailo-8L neural network accelerator. This setup allows for high-performance, real-time inference with YOLO26 while offloading the primary CPU.

To help you get started quickly, we provide a dedicated repository with pre-compiled models, inference scripts, and a full export pipeline.

### Prerequisites

Before using the YOLO26-Hailo repository, you must install the HailoRT software stack.

1.  **Install HailoRT**: Follow the [official HailoRT installation guide for Linux](https://hailo.ai/developer-zone/documentation/hailort-v4-23-0/?sp_referrer=install/install.html#ubuntu-installer-requirements). Ensure you install the Python package for your Python version (e.g., Python 3.10).
2.  **Activate Environment**: After installation, activate the HailoRT virtual environment.

    ```bash
    source <path_to_hailort_venv>/bin/activate
    ```

### Quick Start (Recommended)

You do **not** need to export or compile models yourself to get started. We provide pre-compiled HEF (Hailo Executable Format) files for YOLO26n, s, m, and l.

1.  **Clone the repository:**

    Inside your activated HailoRT environment:

    ```bash
    git clone https://github.com/DanielDubinsky/yolo26_hailo.git
    cd yolo26_hailo
    pip install -r requirements.txt
    ```

2.  **Download Pre-compiled Models:**

    You can download all models or a specific variant (e.g., 'n' for YOLO26n):

    ```bash
    # Download YOLO26n only
    bash scripts/download_hef.sh n
    
    # Or download all variants (n, s, m, l)
    bash scripts/download_hef.sh
    ```

3.  **Run Inference (Python):**

    Use the `detect_image.py` script to run inference on an image:

    ```bash
    python python/detect_image.py input.jpg --hef models/yolo26n.hef
    ```

4.  **Run Inference (C++):**

    For maximum performance, you can compile and run the C++ inference tools:

    ```bash
    cd cpp
    make
    ./detect_image ../input.jpg ../models/yolo26n.hef
    ```

### Performance Benchmarks (Raspberry Pi 5 + Hailo-8L)

YOLO26 models on Hailo-8L achieve significant speedups compared to CPU execution, with minimal accuracy loss due to quantization.

| Model | CPU mAP (FP32) | CPU FPS | Hailo mAP (INT8) | Hailo FPS | Speedup | Accuracy Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n** | 0.402 | 6.50 | 0.371 | 86.5 | 13.3x | 92.3% |
| **YOLO26s** | 0.477 | 2.62 | 0.424 | 37.5 | 14.3x | 88.9% |
| **YOLO26m** | 0.525 | 0.88 | 0.441 | 23.4 | 26.6x | 84.0% |
| **YOLO26l** | 0.541 | 0.74 | 0.473 | 17.9 | 24.2x | 87.4% |

*Benchmarks tested on COCO val2017. Quantization is calibrated using random images from the COCO train2017 dataset.*

### Advanced: Manual Model Export

If you prefer to manually export the model instead of using our automated repository, follow these steps to converting YOLO26n to HEF for Hailo-8L.

#### 1. Software Versions

Ensure you are using the following software versions to reproduce our results:

-   **Ultralytics**: `8.4.7`
-   **Hailo Dataflow Compiler (DFC)**: `v3.33.0`
-   **HailoRT**: `4.23.0`

#### 2. Export to ONNX

First, export the YOLO26n model to ONNX format. We use `opset=11` and enable simplification.

```bash
yolo export model=yolo26n.pt format=onnx opset=11 simplify=True
```

#### 3. Convert ONNX to HAR

Convert the ONNX model to the Hailo Archive (HAR) format. You must specify the exact start and end nodes to ensure the correct graph is extracted.

```bash
hailo parser onnx yolo26n.onnx \
    --hw-arch hailo8l \
    --start-node-names images \
    --end-node-names /model.23/one2one_cv3.0/one2one_cv3.0.2/Conv /model.23/one2one_cv3.1/one2one_cv3.1.2/Conv /model.23/one2one_cv3.2/one2one_cv3.2.2/Conv /model.23/one2one_cv2.0/one2one_cv2.0.2/Conv /model.23/one2one_cv2.1/one2one_cv2.1.2/Conv /model.23/one2one_cv2.2/one2one_cv2.2.2/Conv
```

Select 'n' when prompted "Would you like to add nms postprocess command to the model script? (y/n)".

#### 4. Prepare Calibration Data

The Hailo optimization process requires calibration data in `.npy` format. You will need a set of images (typically from COCO `val2017` or `train2017`), preprocessed to the model's input size (640x640) with letterboxing and compiled into loose `.npy` files.

You can use the following Python script to prepare your calibration data:

```python
import cv2
import numpy as np
import os
from pathlib import Path

def preprocess(src_dir, dst_dir, target_size=640):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Iterate over images
    for img_path in src_path.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None: continue
            
        # 1. Resize with Letterbox
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # 2. Pad to target_size (centered) and fill with gray (114)
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # 3. Convert BGR to RGB
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # 4. Save as .npy
        np.save(dst_path / f"{img_path.stem}.npy", padded)

# Usage
preprocess('data/images', 'data/images_npy')
```

#### 5. Optimize and Quantize

To optimize the model for the Hailo-8L (8-bit quantization), usage of a model script (`.alls`) is required.

Create a file named `yolo26n.alls` with the following content:

```text
# Step 1: Normalize input (typical for YOLO)
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])

# Step 2: Solve the 'Agent infeasible' error by increasing effort
performance_param(compiler_optimization_level=max)

# Step 3: Global optimization level (0 is fastest, 2 is standard)
model_optimization_flavor(optimization_level=2)

# 3. Enable Multi-Context for the Hailo-8L
context_switch_param(mode=allowed)

model_optimization_config(calibration, calibset_size=1024)

pre_quantization_optimization(activation_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])
pre_quantization_optimization(weights_clipping, layers={*}, mode=percentile, clipping_values=[0.01, 99.99])
```



Then, run the optimization command (ensure you have a directory `data/images_npy/` with calibration images, typically 1024 random images from COCO train)

```bash
hailo optimize yolo26n.har \
    --calib-set-path data/images_npy/ \
    --model-script yolo26n.alls
```

#### 6. Compile to HEF

Finally, compile the optimized HAR file into the Hailo Executable Format (HEF).

```bash
hailo compiler yolo26n_quantized.har
```

#### 7. Inference Details

-   **Input**: The compiled model expects **640x640 RGB** images. The normalization layer in the `.alls` script (`div 255`) handles the conversion from `[0, 255]` to `[0, 1]`.
-   **Output**: The model outputs raw feature maps which must be post-processed on the host CPU.

For full implementation details, including the post-processing code, please refer to the [YOLO26-Hailo repository](https://github.com/DanielDubinsky/yolo26_hailo).