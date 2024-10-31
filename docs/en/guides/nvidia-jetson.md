---
comments: true
description: Learn to deploy Ultralytics YOLO11 on NVIDIA Jetson devices with our detailed guide. Explore performance benchmarks and maximize AI capabilities.
keywords: Ultralytics, YOLO11, NVIDIA Jetson, JetPack, AI deployment, performance benchmarks, embedded systems, deep learning, TensorRT, computer vision
---

# Quick Start Guide: NVIDIA Jetson with Ultralytics YOLO11

This comprehensive guide provides a detailed walkthrough for deploying Ultralytics YOLO11 on [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) devices. Additionally, it showcases performance benchmarks to demonstrate the capabilities of YOLO11 on these small and powerful devices.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/mUybgOlSxxA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Setup NVIDIA Jetson with Ultralytics YOLO11
</p>

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/nvidia-jetson-ecosystem.avif" alt="NVIDIA Jetson Ecosystem">

!!! note

    This guide has been tested with both [Seeed Studio reComputer J4012](https://www.seeedstudio.com/reComputer-J4012-p-5586.html) which is based on NVIDIA Jetson Orin NX 16GB running the latest stable JetPack release of [JP6.0](https://developer.nvidia.com/embedded/jetpack-sdk-60), JetPack release of [JP5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) and [Seeed Studio reComputer J1020 v2](https://www.seeedstudio.com/reComputer-J1020-v2-p-5498.html) which is based on NVIDIA Jetson Nano 4GB running JetPack release of [JP4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461). It is expected to work across all the NVIDIA Jetson hardware lineup including latest and legacy.

## What is NVIDIA Jetson?

NVIDIA Jetson is a series of embedded computing boards designed to bring accelerated AI (artificial intelligence) computing to edge devices. These compact and powerful devices are built around NVIDIA's GPU architecture and are capable of running complex AI algorithms and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models directly on the device, without needing to rely on [cloud computing](https://www.ultralytics.com/glossary/cloud-computing) resources. Jetson boards are often used in robotics, autonomous vehicles, industrial automation, and other applications where AI inference needs to be performed locally with low latency and high efficiency. Additionally, these boards are based on the ARM64 architecture and runs on lower power compared to traditional GPU computing devices.

## NVIDIA Jetson Series Comparison

[Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) is the latest iteration of the NVIDIA Jetson family based on NVIDIA Ampere architecture which brings drastically improved AI performance when compared to the previous generations. Below table compared few of the Jetson devices in the ecosystem.

|                   | Jetson AGX Orin 64GB                                              | Jetson Orin NX 16GB                                              | Jetson Orin Nano 8GB                                          | Jetson AGX Xavier                                           | Jetson Xavier NX                                              | Jetson Nano                                   |
| ----------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| AI Performance    | 275 TOPS                                                          | 100 TOPS                                                         | 40 TOPs                                                       | 32 TOPS                                                     | 21 TOPS                                                       | 472 GFLOPS                                    |
| GPU               | 2048-core NVIDIA Ampere architecture GPU with 64 Tensor Cores     | 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores    | 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores | 512-core NVIDIA Volta architecture GPU with 64 Tensor Cores | 384-core NVIDIA Volta™ architecture GPU with 48 Tensor Cores | 128-core NVIDIA Maxwell™ architecture GPU    |
| GPU Max Frequency | 1.3 GHz                                                           | 918 MHz                                                          | 625 MHz                                                       | 1377 MHz                                                    | 1100 MHz                                                      | 921MHz                                        |
| CPU               | 12-core NVIDIA Arm® Cortex A78AE v8.2 64-bit CPU 3MB L2 + 6MB L3 | 8-core NVIDIA Arm® Cortex A78AE v8.2 64-bit CPU 2MB L2 + 4MB L3 | 6-core Arm® Cortex®-A78AE v8.2 64-bit CPU 1.5MB L2 + 4MB L3 | 8-core NVIDIA Carmel Arm®v8.2 64-bit CPU 8MB L2 + 4MB L3   | 6-core NVIDIA Carmel Arm®v8.2 64-bit CPU 6MB L2 + 4MB L3     | Quad-Core Arm® Cortex®-A57 MPCore processor |
| CPU Max Frequency | 2.2 GHz                                                           | 2.0 GHz                                                          | 1.5 GHz                                                       | 2.2 GHz                                                     | 1.9 GHz                                                       | 1.43GHz                                       |
| Memory            | 64GB 256-bit LPDDR5 204.8GB/s                                     | 16GB 128-bit LPDDR5 102.4GB/s                                    | 8GB 128-bit LPDDR5 68 GB/s                                    | 32GB 256-bit LPDDR4x 136.5GB/s                              | 8GB 128-bit LPDDR4x 59.7GB/s                                  | 4GB 64-bit LPDDR4 25.6GB/s"                   |

For a more detailed comparison table, please visit the **Technical Specifications** section of [official NVIDIA Jetson page](https://developer.nvidia.com/embedded/jetson-modules).

## What is NVIDIA JetPack?

[NVIDIA JetPack SDK](https://developer.nvidia.com/embedded/jetpack) powering the Jetson modules is the most comprehensive solution and provides full development environment for building end-to-end accelerated AI applications and shortens time to market. JetPack includes Jetson Linux with bootloader, Linux kernel, Ubuntu desktop environment, and a complete set of libraries for acceleration of GPU computing, multimedia, graphics, and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). It also includes samples, documentation, and developer tools for both host computer and developer kit, and supports higher level SDKs such as DeepStream for streaming video analytics, Isaac for robotics, and Riva for conversational AI.

## Flash JetPack to NVIDIA Jetson

The first step after getting your hands on an NVIDIA Jetson device is to flash NVIDIA JetPack to the device. There are several different way of flashing NVIDIA Jetson devices.

1. If you own an official NVIDIA Development Kit such as the Jetson Orin Nano Developer Kit, you can [download an image and prepare an SD card with JetPack for booting the device](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit).
2. If you own any other NVIDIA Development Kit, you can [flash JetPack to the device using SDK Manager](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html).
3. If you own a Seeed Studio reComputer J4012 device, you can [flash JetPack to the included SSD](https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/) and if you own a Seeed Studio reComputer J1020 v2 device, you can [flash JetPack to the eMMC/ SSD](https://wiki.seeedstudio.com/reComputer_J2021_J202_Flash_Jetpack/).
4. If you own any other third party device powered by the NVIDIA Jetson module, it is recommended to follow [command-line flashing](https://docs.nvidia.com/jetson/archives/r35.5.0/DeveloperGuide/IN/QuickStart.html).

!!! note

    For methods 3 and 4 above, after flashing the system and booting the device, please enter "sudo apt update && sudo apt install nvidia-jetpack -y" on the device terminal to install all the remaining JetPack components needed.

## JetPack Support Based on Jetson Device

The below table highlights NVIDIA JetPack versions supported by different NVIDIA Jetson devices.

|                   | JetPack 4 | JetPack 5 | JetPack 6 |
| ----------------- | --------- | --------- | --------- |
| Jetson Nano       | ✅        | ❌        | ❌        |
| Jetson TX2        | ✅        | ❌        | ❌        |
| Jetson Xavier NX  | ✅        | ✅        | ❌        |
| Jetson AGX Xavier | ✅        | ✅        | ❌        |
| Jetson AGX Orin   | ❌        | ✅        | ✅        |
| Jetson Orin NX    | ❌        | ✅        | ✅        |
| Jetson Orin Nano  | ❌        | ✅        | ✅        |

## Quick Start with Docker

The fastest way to get started with Ultralytics YOLO11 on NVIDIA Jetson is to run with pre-built docker images for Jetson. Refer to the table above and choose the JetPack version according to the Jetson device you own.

=== "JetPack 4"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack4
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

=== "JetPack 5"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack5
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

=== "JetPack 6"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack6
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

After this is done, skip to [Use TensorRT on NVIDIA Jetson section](#use-tensorrt-on-nvidia-jetson).

## Start with Native Installation

For a native installation without Docker, please refer to the steps below.

### Run on JetPack 6.x

#### Install Ultralytics Package

Here we will install Ultralytics package on the Jetson with optional dependencies so that we can export the [PyTorch](https://www.ultralytics.com/glossary/pytorch) models to other different formats. We will mainly focus on [NVIDIA TensorRT exports](../integrations/tensorrt.md) because TensorRT will make sure we can get the maximum performance out of the Jetson devices.

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

#### Install PyTorch and Torchvision

The above ultralytics installation will install Torch and Torchvision. However, these 2 packages installed via pip are not compatible to run on Jetson platform which is based on ARM64 architecture. Therefore, we need to manually install pre-built PyTorch pip wheel and compile/ install Torchvision from source.

Install `torch 2.3.0` and `torchvision 0.18` according to JP6.0

```bash
sudo apt-get install libopenmpi-dev libopenblas-base libomp-dev -y
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
```

Visit the [PyTorch for Jetson page](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) to access all different versions of PyTorch for different JetPack versions. For a more detailed list on the PyTorch, Torchvision compatibility, visit the [PyTorch and Torchvision compatibility page](https://github.com/pytorch/vision).

#### Install `onnxruntime-gpu`

The [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) package hosted in PyPI does not have `aarch64` binaries for the Jetson. So we need to manually install this package. This package is needed for some of the exports.

All different `onnxruntime-gpu` packages corresponding to different JetPack and Python versions are listed [here](https://elinux.org/Jetson_Zoo#ONNX_Runtime). However, here we will download and install `onnxruntime-gpu 1.18.0` with `Python3.10` support.

```bash
wget https://nvidia.box.com/shared/static/48dtuob7meiw6ebgfsfqakc9vse62sg4.whl -O onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl
pip install onnxruntime_gpu-1.18.0-cp310-cp310-linux_aarch64.whl
```

!!! note

    `onnxruntime-gpu` will automatically revert back the numpy version to latest. So we need to reinstall numpy to `1.23.5` to fix an issue by executing:

    `pip install numpy==1.23.5`

### Run on JetPack 5.x

#### Install Ultralytics Package

Here we will install Ultralytics package on the Jetson with optional dependencies so that we can export the PyTorch models to other different formats. We will mainly focus on [NVIDIA TensorRT exports](../integrations/tensorrt.md) because TensorRT will make sure we can get the maximum performance out of the Jetson devices.

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

#### Install PyTorch and Torchvision

The above ultralytics installation will install Torch and Torchvision. However, these 2 packages installed via pip are not compatible to run on Jetson platform which is based on ARM64 architecture. Therefore, we need to manually install pre-built PyTorch pip wheel and compile/ install Torchvision from source.

1. Uninstall currently installed PyTorch and Torchvision

    ```bash
    pip uninstall torch torchvision
    ```

2. Install PyTorch 2.1.0 according to JP5.1.3

    ```bash
    sudo apt-get install -y libopenblas-base libopenmpi-dev
    wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl -O torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
    pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
    ```

3. Install Torchvision v0.16.2 according to PyTorch v2.1.0

    ```bash
    sudo apt install -y libjpeg-dev zlib1g-dev
    git clone https://github.com/pytorch/vision torchvision
    cd torchvision
    git checkout v0.16.2
    python3 setup.py install --user
    ```

Visit the [PyTorch for Jetson page](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) to access all different versions of PyTorch for different JetPack versions. For a more detailed list on the PyTorch, Torchvision compatibility, visit the [PyTorch and Torchvision compatibility page](https://github.com/pytorch/vision).

#### Install `onnxruntime-gpu`

The [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) package hosted in PyPI does not have `aarch64` binaries for the Jetson. So we need to manually install this package. This package is needed for some of the exports.

All different `onnxruntime-gpu` packages corresponding to different JetPack and Python versions are listed [here](https://elinux.org/Jetson_Zoo#ONNX_Runtime). However, here we will download and install `onnxruntime-gpu 1.17.0` with `Python3.8` support.

```bash
wget https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6kpcuwzd.whl -O onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
```

!!! note

    `onnxruntime-gpu` will automatically revert back the numpy version to latest. So we need to reinstall numpy to `1.23.5` to fix an issue by executing:

    `pip install numpy==1.23.5`

## Use TensorRT on NVIDIA Jetson

Out of all the model export formats supported by Ultralytics, TensorRT delivers the best inference performance when working with NVIDIA Jetson devices and our recommendation is to use TensorRT with Jetson. We also have a detailed document on TensorRT [here](../integrations/tensorrt.md).

### Convert Model to TensorRT and Run Inference

The YOLO11n model in PyTorch format is converted to TensorRT to run inference with the exported model.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Export the model to TensorRT
        model.export(format="engine")  # creates 'yolo11n.engine'

        # Load the exported TensorRT model
        trt_model = YOLO("yolo11n.engine")

        # Run inference
        results = trt_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to TensorRT format
        yolo export model=yolo11n.pt format=engine  # creates 'yolo11n.engine'

        # Run inference with the exported model
        yolo predict model=yolo11n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

### Use NVIDIA Deep Learning Accelerator (DLA)

[NVIDIA Deep Learning Accelerator (DLA)](https://developer.nvidia.com/deep-learning-accelerator) is a specialized hardware component built into NVIDIA Jetson devices that optimizes deep learning inference for energy efficiency and performance. By offloading tasks from the GPU (freeing it up for more intensive processes), DLA enables models to run with lower power consumption while maintaining high throughput, ideal for embedded systems and real-time AI applications.

The following Jetson devices are equipped with DLA hardware:

- Jetson Orin NX 16GB
- Jetson AGX Orin Series
- Jetson AGX Xavier Series
- Jetson Xavier NX Series

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Export the model to TensorRT with DLA enabled (only works with FP16 or INT8)
        model.export(format="engine", device="dla:0", half=True)  # dla:0 or dla:1 corresponds to the DLA cores

        # Load the exported TensorRT model
        trt_model = YOLO("yolo11n.engine")

        # Run inference
        results = trt_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to TensorRT format with DLA enabled (only works with FP16 or INT8)
        yolo export model=yolo11n.pt format=engine device="dla:0" half=True  # dla:0 or dla:1 corresponds to the DLA cores

        # Run inference with the exported model on the DLA
        yolo predict model=yolo11n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    Visit the [Export page](../modes/export.md#arguments) to access additional arguments when exporting models to different model formats

## NVIDIA Jetson Orin YOLO11 Benchmarks

YOLO11 benchmarks were run by the Ultralytics team on 10 different model formats measuring speed and [accuracy](https://www.ultralytics.com/glossary/accuracy): PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, TF SavedModel, TF GraphDef, TF Lite, PaddlePaddle, NCNN. Benchmarks were run on Seeed Studio reComputer J4012 powered by Jetson Orin NX 16GB device at FP32 [precision](https://www.ultralytics.com/glossary/precision) with default input image size of 640.

### Comparison Chart

Even though all model exports are working with NVIDIA Jetson, we have only included **PyTorch, TorchScript, TensorRT** for the comparison chart below because, they make use of the GPU on the Jetson and are guaranteed to produce the best results. All the other exports only utilize the CPU and the performance is not as good as the above three. You can find benchmarks for all exports in the section after this chart.

<div style="text-align: center;">
    <img src="https://github.com/ultralytics/docs/releases/download/0/nvidia-jetson-benchmarks.avif" alt="NVIDIA Jetson Ecosystem">
</div>

### Detailed Comparison Table

The below table represents the benchmark results for five different models (YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x) across ten different formats (PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, TF SavedModel, TF GraphDef, TF Lite, PaddlePaddle, NCNN), giving us the status, size, mAP50-95(B) metric, and inference time for each combination.

!!! performance

    === "YOLO11n"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.6176      | 19.80                  |
        | TorchScript     | ✅      | 10.5              | 0.6100      | 13.30                  |
        | ONNX            | ✅      | 10.2              | 0.6082      | 67.92                  |
        | OpenVINO        | ✅      | 10.4              | 0.6082      | 118.21                 |
        | TensorRT (FP32) | ✅      | 14.1              | 0.6100      | 7.94                   |
        | TensorRT (FP16) | ✅      | 8.3               | 0.6082      | 4.80                   |
        | TensorRT (INT8) | ✅      | 6.6               | 0.3256      | 4.17                   |
        | TF SavedModel   | ✅      | 25.8              | 0.6082      | 185.88                 |
        | TF GraphDef     | ✅      | 10.3              | 0.6082      | 256.66                 |
        | TF Lite         | ✅      | 10.3              | 0.6082      | 284.64                 |
        | PaddlePaddle    | ✅      | 20.4              | 0.6082      | 477.41                 |
        | NCNN            | ✅      | 10.2              | 0.6106      | 32.18                  |

    === "YOLO11s"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.7526      | 20.20                  |
        | TorchScript     | ✅      | 36.5              | 0.7416      | 23.42                  |
        | ONNX            | ✅      | 36.3              | 0.7416      | 162.01                 |
        | OpenVINO        | ✅      | 36.4              | 0.7416      | 159.61                 |
        | TensorRT (FP32) | ✅      | 40.3              | 0.7416      | 13.93                  |
        | TensorRT (FP16) | ✅      | 21.7              | 0.7416      | 7.47                   |
        | TensorRT (INT8) | ✅      | 13.6              | 0.3179      | 5.66                   |
        | TF SavedModel   | ✅      | 91.1              | 0.7416      | 316.46                 |
        | TF GraphDef     | ✅      | 36.4              | 0.7416      | 506.71                 |
        | TF Lite         | ✅      | 36.4              | 0.7416      | 842.97                 |
        | PaddlePaddle    | ✅      | 72.5              | 0.7416      | 1172.57                |
        | NCNN            | ✅      | 36.2              | 0.7419      | 66.00                  |

    === "YOLO11m"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.7595      | 36.70                  |
        | TorchScript     | ✅      | 77.3              | 0.7643      | 50.95                  |
        | ONNX            | ✅      | 76.9              | 0.7643      | 416.34                 |
        | OpenVINO        | ✅      | 77.1              | 0.7643      | 370.99                 |
        | TensorRT (FP32) | ✅      | 81.5              | 0.7640      | 30.49                  |
        | TensorRT (FP16) | ✅      | 42.2              | 0.7658      | 14.93                  |
        | TensorRT (INT8) | ✅      | 24.3              | 0.4118      | 10.32                  |
        | TF SavedModel   | ✅      | 192.7             | 0.7643      | 597.08                 |
        | TF GraphDef     | ✅      | 77.0              | 0.7643      | 1016.12                |
        | TF Lite         | ✅      | 77.0              | 0.7643      | 2494.60                |
        | PaddlePaddle    | ✅      | 153.8             | 0.7643      | 3218.99                |
        | NCNN            | ✅      | 76.8              | 0.7691      | 192.77                 |

    === "YOLO11l"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.7475      | 47.6                   |
        | TorchScript     | ✅      | 97.6              | 0.7250      | 66.36                  |
        | ONNX            | ✅      | 97.0              | 0.7250      | 532.58                 |
        | OpenVINO        | ✅      | 97.3              | 0.7250      | 477.55                 |
        | TensorRT (FP32) | ✅      | 101.6             | 0.7250      | 38.71                  |
        | TensorRT (FP16) | ✅      | 52.6              | 0.7265      | 19.35                  |
        | TensorRT (INT8) | ✅      | 31.6              | 0.3856      | 13.50                  |
        | TF SavedModel   | ✅      | 243.3             | 0.7250      | 895.24                 |
        | TF GraphDef     | ✅      | 97.2              | 0.7250      | 1301.19                |
        | TF Lite         | ✅      | 97.2              | 0.7250      | 3202.93                |
        | PaddlePaddle    | ✅      | 193.9             | 0.7250      | 4206.98                |
        | NCNN            | ✅      | 96.9              | 0.7252      | 225.75                 |

    === "YOLO11x"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.8288      | 85.60                  |
        | TorchScript     | ✅      | 218.1             | 0.8308      | 121.67                 |
        | ONNX            | ✅      | 217.5             | 0.8308      | 1073.14                |
        | OpenVINO        | ✅      | 217.8             | 0.8308      | 955.60                 |
        | TensorRT (FP32) | ✅      | 221.6             | 0.8307      | 75.84                  |
        | TensorRT (FP16) | ✅      | 113.1             | 0.8295      | 35.75                  |
        | TensorRT (INT8) | ✅      | 62.2              | 0.4783      | 22.23                  |
        | TF SavedModel   | ✅      | 545.0             | 0.8308      | 1497.40                |
        | TF GraphDef     | ✅      | 217.8             | 0.8308      | 2552.42                |
        | TF Lite         | ✅      | 217.8             | 0.8308      | 7044.58                |
        | PaddlePaddle    | ✅      | 434.9             | 0.8308      | 8386.73                |
        | NCNN            | ✅      | 217.3             | 0.8304      | 486.36                 |

[Explore more benchmarking efforts by Seeed Studio](https://www.seeedstudio.com/blog/2023/03/30/yolov8-performance-benchmarks-on-nvidia-jetson-devices) running on different versions of NVIDIA Jetson hardware.

## Reproduce Our Results

To reproduce the above Ultralytics benchmarks on all export [formats](../modes/export.md) run this code:

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

## Best Practices when using NVIDIA Jetson

When using NVIDIA Jetson, there are a couple of best practices to follow in order to enable maximum performance on the NVIDIA Jetson running YOLO11.

1. Enable MAX Power Mode

    Enabling MAX Power Mode on the Jetson will make sure all CPU, GPU cores are turned on.

    ```bash
    sudo nvpmodel -m 0
    ```

2. Enable Jetson Clocks

    Enabling Jetson Clocks will make sure all CPU, GPU cores are clocked at their maximum frequency.

    ```bash
    sudo jetson_clocks
    ```

3. Install Jetson Stats Application

    We can use jetson stats application to monitor the temperatures of the system components and check other system details such as view CPU, GPU, RAM utilization, change power modes, set to max clocks, check JetPack information

    ```bash
    sudo apt update
    sudo pip install jetson-stats
    sudo reboot
    jtop
    ```

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/jetson-stats-application.avif" alt="Jetson Stats">

## Next Steps

Congratulations on successfully setting up YOLO11 on your NVIDIA Jetson! For further learning and support, visit more guide at [Ultralytics YOLO11 Docs](../index.md)!

## FAQ

### How do I deploy Ultralytics YOLO11 on NVIDIA Jetson devices?

Deploying Ultralytics YOLO11 on NVIDIA Jetson devices is a straightforward process. First, flash your Jetson device with the NVIDIA JetPack SDK. Then, either use a pre-built Docker image for quick setup or manually install the required packages. Detailed steps for each approach can be found in sections [Quick Start with Docker](#quick-start-with-docker) and [Start with Native Installation](#start-with-native-installation).

### What performance benchmarks can I expect from YOLO11 models on NVIDIA Jetson devices?

YOLO11 models have been benchmarked on various NVIDIA Jetson devices showing significant performance improvements. For example, the TensorRT format delivers the best inference performance. The table in the [Detailed Comparison Table](#detailed-comparison-table) section provides a comprehensive view of performance metrics like mAP50-95 and inference time across different model formats.

### Why should I use TensorRT for deploying YOLO11 on NVIDIA Jetson?

TensorRT is highly recommended for deploying YOLO11 models on NVIDIA Jetson due to its optimal performance. It accelerates inference by leveraging the Jetson's GPU capabilities, ensuring maximum efficiency and speed. Learn more about how to convert to TensorRT and run inference in the [Use TensorRT on NVIDIA Jetson](#use-tensorrt-on-nvidia-jetson) section.

### How can I install PyTorch and Torchvision on NVIDIA Jetson?

To install PyTorch and Torchvision on NVIDIA Jetson, first uninstall any existing versions that may have been installed via pip. Then, manually install the compatible PyTorch and Torchvision versions for the Jetson's ARM64 architecture. Detailed instructions for this process are provided in the [Install PyTorch and Torchvision](#install-pytorch-and-torchvision) section.

### What are the best practices for maximizing performance on NVIDIA Jetson when using YOLO11?

To maximize performance on NVIDIA Jetson with YOLO11, follow these best practices:

1. Enable MAX Power Mode to utilize all CPU and GPU cores.
2. Enable Jetson Clocks to run all cores at their maximum frequency.
3. Install the Jetson Stats application for monitoring system metrics.

For commands and additional details, refer to the [Best Practices when using NVIDIA Jetson](#best-practices-when-using-nvidia-jetson) section.
