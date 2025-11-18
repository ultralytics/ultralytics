---
comments: true
description: Learn to deploy Ultralytics YOLO11 on NVIDIA Jetson devices with our detailed guide. Explore performance benchmarks and maximize AI capabilities.
keywords: Ultralytics, YOLO11, NVIDIA Jetson, JetPack, AI deployment, performance benchmarks, embedded systems, deep learning, TensorRT, computer vision
---

# Quick Start Guide: NVIDIA Jetson with Ultralytics YOLO11

This comprehensive guide provides a detailed walkthrough for deploying Ultralytics YOLO11 on [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) devices. Additionally, it showcases performance benchmarks to demonstrate the capabilities of YOLO11 on these small and powerful devices.

!!! tip "New product support"

    We have updated this guide with the latest [NVIDIA Jetson AGX Thor Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor) which delivers up to 2070 FP4 TFLOPS of AI compute and 128 GB of memory with power configurable between 40 W and 130 W. It delivers over 7.5x higher AI compute than NVIDIA Jetson AGX Orin, with 3.5x better energy efficiency to seamlessly run the most popular AI models.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/BPYkGt3odNk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to use Ultralytics YOLO11 on NVIDIA JETSON Devices
</p>

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/nvidia-jetson-ecosystem.avif" alt="NVIDIA Jetson Ecosystem">

!!! note

    This guide has been tested with [NVIDIA Jetson AGX Thor Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor) running the latest stable JetPack release of [JP7.0](https://developer.nvidia.com/embedded/jetpack/downloads), [NVIDIA Jetson AGX Orin Developer Kit (64GB)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin) running JetPack release of [JP6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62), [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit) running JetPack release of [JP6.1](https://developer.nvidia.com/embedded/jetpack-sdk-61), [Seeed Studio reComputer J4012](https://www.seeedstudio.com/reComputer-J4012-p-5586.html) which is based on NVIDIA Jetson Orin NX 16GB running JetPack release of [JP6.0](https://developer.nvidia.com/embedded/jetpack-sdk-60)/ JetPack release of [JP5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) and [Seeed Studio reComputer J1020 v2](https://www.seeedstudio.com/reComputer-J1020-v2-p-5498.html) which is based on NVIDIA Jetson Nano 4GB running JetPack release of [JP4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461). It is expected to work across all the NVIDIA Jetson hardware lineup including latest and legacy.

## What is NVIDIA Jetson?

NVIDIA Jetson is a series of embedded computing boards designed to bring accelerated AI (artificial intelligence) computing to edge devices. These compact and powerful devices are built around NVIDIA's GPU architecture and are capable of running complex AI algorithms and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models directly on the device, without needing to rely on [cloud computing](https://www.ultralytics.com/glossary/cloud-computing) resources. Jetson boards are often used in robotics, autonomous vehicles, industrial automation, and other applications where AI inference needs to be performed locally with low latency and high efficiency. Additionally, these boards are based on the ARM64 architecture and runs on lower power compared to traditional GPU computing devices.

## NVIDIA Jetson Series Comparison

[NVIDIA Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/) is the latest iteration of the NVIDIA Jetson family based on NVIDIA Blackwell architecture which brings drastically improved AI performance when compared to the previous generations. The table below compares a few of the Jetson devices in the ecosystem.

|                   | Jetson AGX Thor                                                  | Jetson AGX Orin 64GB                                              | Jetson Orin NX 16GB                                              | Jetson Orin Nano Super                                        | Jetson AGX Xavier                                           | Jetson Xavier NX                                              | Jetson Nano                                   |
| ----------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| AI Performance    | 2070 TFLOPS                                                      | 275 TOPS                                                          | 100 TOPS                                                         | 67 TOPS                                                       | 32 TOPS                                                     | 21 TOPS                                                       | 472 GFLOPS                                    |
| GPU               | 2560-core NVIDIA Blackwell architecture GPU with 96 Tensor Cores | 2048-core NVIDIA Ampere architecture GPU with 64 Tensor Cores     | 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores    | 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores | 512-core NVIDIA Volta architecture GPU with 64 Tensor Cores | 384-core NVIDIA Volta™ architecture GPU with 48 Tensor Cores | 128-core NVIDIA Maxwell™ architecture GPU    |
| GPU Max Frequency | 1.57 GHz                                                         | 1.3 GHz                                                           | 918 MHz                                                          | 1020 MHz                                                      | 1377 MHz                                                    | 1100 MHz                                                      | 921MHz                                        |
| CPU               | 14-core Arm® Neoverse®-V3AE 64-bit CPU 1MB L2 + 16MB L3        | 12-core NVIDIA Arm® Cortex A78AE v8.2 64-bit CPU 3MB L2 + 6MB L3 | 8-core NVIDIA Arm® Cortex A78AE v8.2 64-bit CPU 2MB L2 + 4MB L3 | 6-core Arm® Cortex®-A78AE v8.2 64-bit CPU 1.5MB L2 + 4MB L3 | 8-core NVIDIA Carmel Arm®v8.2 64-bit CPU 8MB L2 + 4MB L3   | 6-core NVIDIA Carmel Arm®v8.2 64-bit CPU 6MB L2 + 4MB L3     | Quad-Core Arm® Cortex®-A57 MPCore processor |
| CPU Max Frequency | 2.6 GHz                                                          | 2.2 GHz                                                           | 2.0 GHz                                                          | 1.7 GHz                                                       | 2.2 GHz                                                     | 1.9 GHz                                                       | 1.43GHz                                       |
| Memory            | 128GB 256-bit LPDDR5X 273GB/s                                    | 64GB 256-bit LPDDR5 204.8GB/s                                     | 16GB 128-bit LPDDR5 102.4GB/s                                    | 8GB 128-bit LPDDR5 102 GB/s                                   | 32GB 256-bit LPDDR4x 136.5GB/s                              | 8GB 128-bit LPDDR4x 59.7GB/s                                  | 4GB 64-bit LPDDR4 25.6GB/s                    |

For a more detailed comparison table, please visit the **Compare Specifications** section of [official NVIDIA Jetson page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems).

## What is NVIDIA JetPack?

[NVIDIA JetPack SDK](https://developer.nvidia.com/embedded/jetpack) powering the Jetson modules is the most comprehensive solution and provides full development environment for building end-to-end accelerated AI applications and shortens time to market. JetPack includes Jetson Linux with bootloader, Linux kernel, Ubuntu desktop environment, and a complete set of libraries for acceleration of GPU computing, multimedia, graphics, and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). It also includes samples, documentation, and developer tools for both host computer and developer kit, and supports higher level SDKs such as [DeepStream](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson/) for streaming video analytics, Isaac for robotics, and Riva for conversational AI.

## Flash JetPack to NVIDIA Jetson

The first step after getting your hands on an NVIDIA Jetson device is to flash NVIDIA JetPack to the device. There are several different way of flashing NVIDIA Jetson devices.

1. If you own an official NVIDIA Development Kit such as the Jetson AGX Thor Developer Kit, you can [download an image and prepare a bootable USB stick to flash JetPack to the included SSD](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html).
2. If you own an official NVIDIA Development Kit such as the Jetson Orin Nano Developer Kit, you can [download an image and prepare an SD card with JetPack for booting the device](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit).
3. If you own any other NVIDIA Development Kit, you can [flash JetPack to the device using SDK Manager](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html).
4. If you own a Seeed Studio reComputer J4012 device, you can [flash JetPack to the included SSD](https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/) and if you own a Seeed Studio reComputer J1020 v2 device, you can [flash JetPack to the eMMC/ SSD](https://wiki.seeedstudio.com/reComputer_J2021_J202_Flash_Jetpack/).
5. If you own any other third party device powered by the NVIDIA Jetson module, it is recommended to follow [command-line flashing](https://docs.nvidia.com/jetson/archives/r35.5.0/DeveloperGuide/IN/QuickStart.html).

!!! note

    For methods 1, 4 and 5 above, after flashing the system and booting the device, please enter "sudo apt update && sudo apt install nvidia-jetpack -y" on the device terminal to install all the remaining JetPack components needed.

## JetPack Support Based on Jetson Device

The below table highlights NVIDIA JetPack versions supported by different NVIDIA Jetson devices.

|                   | JetPack 4 | JetPack 5 | JetPack 6 | JetPack 7 |
| ----------------- | --------- | --------- | --------- | --------- |
| Jetson Nano       | ✅        | ❌        | ❌        | ❌        |
| Jetson TX2        | ✅        | ❌        | ❌        | ❌        |
| Jetson Xavier NX  | ✅        | ✅        | ❌        | ❌        |
| Jetson AGX Xavier | ✅        | ✅        | ❌        | ❌        |
| Jetson AGX Orin   | ❌        | ✅        | ✅        | ❌        |
| Jetson Orin NX    | ❌        | ✅        | ✅        | ❌        |
| Jetson Orin Nano  | ❌        | ✅        | ✅        | ❌        |
| Jetson AGX Thor   | ❌        | ❌        | ❌        | ✅        |

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

=== "JetPack 7"

    Coming soon! Stay tuned!

After this is done, skip to [Use TensorRT on NVIDIA Jetson section](#use-tensorrt-on-nvidia-jetson).

## Start with Native Installation

For a native installation without Docker, please refer to the steps below.

### Run on JetPack 7.0

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

The above ultralytics installation will install Torch and Torchvision. However, these 2 packages installed via pip are not compatible to run on Jetson AGX Thor which comes with JetPack 7.0 and CUDA 13. Therefore, we need to manually install them.

Install `torch` and `torchvision` according to JP7.0

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

#### Install `onnxruntime-gpu`

The [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) package hosted in PyPI does not have `aarch64` binaries for the Jetson. So we need to manually install this package. This package is needed for some of the exports.

Here we will download and install `onnxruntime-gpu 1.24.0` with `Python3.12` support.

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl
```

### Run on JetPack 6.1

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

The above ultralytics installation will install Torch and Torchvision. However, these two packages installed via pip are not compatible with the Jetson platform, which is based on ARM64 architecture. Therefore, we need to manually install a pre-built PyTorch pip wheel and compile or install Torchvision from source.

Install `torch 2.5.0` and `torchvision 0.20` according to JP6.1

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

!!! note

    Visit the [PyTorch for Jetson page](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) to access all different versions of PyTorch for different JetPack versions. For a more detailed list on the PyTorch, Torchvision compatibility, visit the [PyTorch and Torchvision compatibility page](https://github.com/pytorch/vision).

Install [`cuSPARSELt`](https://developer.nvidia.com/cusparselt-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_network) to fix a dependency issue with `torch 2.5.0`

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```

#### Install `onnxruntime-gpu`

The [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) package hosted in PyPI does not have `aarch64` binaries for the Jetson. So we need to manually install this package. This package is needed for some of the exports.

You can find all available `onnxruntime-gpu` packages—organized by JetPack version, Python version, and other compatibility details—in the [Jetson Zoo ONNX Runtime compatibility matrix](https://elinux.org/Jetson_Zoo#ONNX_Runtime). Here we will download and install `onnxruntime-gpu 1.20.0` with `Python3.10` support.

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

!!! note

    `onnxruntime-gpu` will automatically revert back the numpy version to latest. So we need to reinstall numpy to `1.23.5` to fix an issue by executing:

    `pip install numpy==1.23.5`

### Run on JetPack 5.1.2

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

The above ultralytics installation will install Torch and Torchvision. However, these two packages installed via pip are not compatible with the Jetson platform, which is based on ARM64 architecture. Therefore, we need to manually install a pre-built PyTorch pip wheel and compile or install Torchvision from source.

1. Uninstall currently installed PyTorch and Torchvision

    ```bash
    pip uninstall torch torchvision
    ```

2. Install `torch 2.2.0` and `torchvision 0.17.2` according to JP5.1.2

    ```bash
    pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.2.0-cp38-cp38-linux_aarch64.whl
    pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl
    ```

!!! note

    Visit the [PyTorch for Jetson page](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) to access all different versions of PyTorch for different JetPack versions. For a more detailed list on the PyTorch, Torchvision compatibility, visit the [PyTorch and Torchvision compatibility page](https://github.com/pytorch/vision).

#### Install `onnxruntime-gpu`

The [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) package hosted in PyPI does not have `aarch64` binaries for the Jetson. So we need to manually install this package. This package is needed for some of the exports.

You can find all available `onnxruntime-gpu` packages—organized by JetPack version, Python version, and other compatibility details—in the [Jetson Zoo ONNX Runtime compatibility matrix](https://elinux.org/Jetson_Zoo#ONNX_Runtime). Here we will download and install `onnxruntime-gpu 1.17.0` with `Python3.8` support.

```bash
wget https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6kpcuwzd.whl -O onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
```

!!! note

    `onnxruntime-gpu` will automatically revert back the numpy version to latest. So we need to reinstall numpy to `1.23.5` to fix an issue by executing:

    `pip install numpy==1.23.5`

## Use TensorRT on NVIDIA Jetson

Among all the model export formats supported by Ultralytics, TensorRT offers the highest inference performance on NVIDIA Jetson devices, making it our top recommendation for Jetson deployments. For setup instructions and advanced usage, see our [dedicated TensorRT integration guide](../integrations/tensorrt.md).

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
        yolo export model=yolo11n.pt format=engine # creates 'yolo11n.engine'

        # Run inference with the exported model
        yolo predict model=yolo11n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    Visit the [Export page](../modes/export.md#arguments) to access additional arguments when exporting models to different model formats

### Use NVIDIA Deep Learning Accelerator (DLA)

[NVIDIA Deep Learning Accelerator (DLA)](https://developer.nvidia.com/deep-learning-accelerator) is a specialized hardware component built into NVIDIA Jetson devices that optimizes deep learning inference for energy efficiency and performance. By offloading tasks from the GPU (freeing it up for more intensive processes), DLA enables models to run with lower power consumption while maintaining high throughput, ideal for embedded systems and real-time AI applications.

The following Jetson devices are equipped with DLA hardware:

| Jetson Device            | DLA Cores | DLA Max Frequency |
| ------------------------ | --------- | ----------------- |
| Jetson AGX Orin Series   | 2         | 1.6 GHz           |
| Jetson Orin NX 16GB      | 2         | 614 MHz           |
| Jetson Orin NX 8GB       | 1         | 614 MHz           |
| Jetson AGX Xavier Series | 2         | 1.4 GHz           |
| Jetson Xavier NX Series  | 2         | 1.1 GHz           |

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
        # Once DLA core number is specified at export, it will use the same core at inference
        yolo export model=yolo11n.pt format=engine device="dla:0" half=True # dla:0 or dla:1 corresponds to the DLA cores

        # Run inference with the exported model on the DLA
        yolo predict model=yolo11n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    When using DLA exports, some layers may not be supported to run on DLA and will fall back to the GPU for execution. This fallback can introduce additional latency and impact the overall inference performance. Therefore, DLA is not primarily designed to reduce inference latency compared to TensorRT running entirely on the GPU. Instead, its primary purpose is to increase throughput and improve energy efficiency.

## NVIDIA Jetson YOLO11 Benchmarks

YOLO11 benchmarks were run by the Ultralytics team on 11 different model formats measuring speed and [accuracy](https://www.ultralytics.com/glossary/accuracy): PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, TF SavedModel, TF GraphDef, TF Lite, MNN, NCNN, ExecuTorch. Benchmarks were run on NVIDIA Jetson AGX Thor Developer Kit, NVIDIA Jetson AGX Orin Developer Kit (64GB), NVIDIA Jetson Orin Nano Super Developer Kit and Seeed Studio reComputer J4012 powered by Jetson Orin NX 16GB device at FP32 [precision](https://www.ultralytics.com/glossary/precision) with default input image size of 640.

### Comparison Charts

Even though all model exports are working with NVIDIA Jetson, we have only included **PyTorch, TorchScript, TensorRT** for the comparison chart below because, they make use of the GPU on the Jetson and are guaranteed to produce the best results. All the other exports only utilize the CPU and the performance is not as good as the above three. You can find benchmarks for all exports in the section after this chart.

#### NVIDIA Jetson AGX Thor Developer Kit

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-agx-thor-benchmarks-coco128.avif" alt="Jetson AGX Thor Benchmarks">
    <figcaption style="font-style: italic; color: gray;">Benchmarked with Ultralytics 8.3.226</figcaption>
</figure>

#### NVIDIA Jetson AGX Orin Developer Kit (64GB)

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-agx-orin-benchmarks-coco128.avif" alt="Jetson AGX Orin Benchmarks">
    <figcaption style="font-style: italic; color: gray;">Benchmarked with Ultralytics 8.3.157</figcaption>
</figure>

#### NVIDIA Jetson Orin Nano Super Developer Kit

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-orin-nano-super-benchmarks-coco128.avif" alt="Jetson Orin Nano Super Benchmarks">
    <figcaption style="font-style: italic; color: gray;">Benchmarked with Ultralytics 8.3.157</figcaption>
</figure>

#### NVIDIA Jetson Orin NX 16GB

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-orin-nx-16-benchmarks-coco128.avif" alt="Jetson Orin NX 16GB Benchmarks">
    <figcaption style="font-style: italic; color: gray;">Benchmarked with Ultralytics 8.3.157</figcaption>
</figure>

### Detailed Comparison Tables

The below table represents the benchmark results for five different models (YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x) across 11 different formats (PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, TF SavedModel, TF GraphDef, TF Lite, MNN, NCNN, ExecuTorch), giving us the status, size, mAP50-95(B) metric, and inference time for each combination.

#### NVIDIA Jetson AGX Thor Developer Kit

!!! tip "Performance"

    === "YOLO11n"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.5070      | 4.1                    |
        | TorchScript     | ✅      | 10.5              | 0.5083      | 3.61                   |
        | ONNX            | ✅      | 10.2              | 0.5076      | 4.8                    |
        | OpenVINO        | ✅      | 10.4              | 0.5058      | 16.48                  |
        | TensorRT (FP32) | ✅      | 12.6              | 0.5077      | 1.70                   |
        | TensorRT (FP16) | ✅      | 7.7               | 0.5075      | 1.20                   |
        | TensorRT (INT8) | ✅      | 6.2               | 0.4858      | 1.29                   |
        | TF SavedModel   | ✅      | 25.7              | 0.5076      | 40.35                  |
        | TF GraphDef     | ✅      | 10.3              | 0.5076      | 40.55                  |
        | TF Lite         | ✅      | 10.3              | 0.5075      | 206.74                 |
        | MNN             | ✅      | 10.1              | 0.5075      | 23.47                  |
        | NCNN            | ✅      | 10.2              | 0.5041      | 22.05                  |
        | ExecuTorch      | ✅      | 10.2              | 0.5075      | 34.28                  |

    === "YOLO11s"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.5770      | 6.10                  |
        | TorchScript     | ✅      | 36.6              | 0.5783      | 5.33                   |
        | ONNX            | ✅      | 36.3              | 0.5783      | 7.01                   |
        | OpenVINO        | ✅      | 36.4              | 0.5809      | 33.08                  |
        | TensorRT (FP32) | ✅      | 40.1              | 0.5784      | 2.57                   |
        | TensorRT (FP16) | ✅      | 20.8              | 0.5796      | 1.55                   |
        | TensorRT (INT8) | ✅      | 12.7              | 0.5514      | 1.50                   |
        | TF SavedModel   | ✅      | 90.8              | 0.5782      | 80.55                  |
        | TF GraphDef     | ✅      | 36.3              | 0.5782      | 80.82                  |
        | TF Lite         | ✅      | 36.3              | 0.5782      | 615.29                 |
        | MNN             | ✅      | 36.2              | 0.5790      | 54.12                  |
        | NCNN            | ✅      | 36.3              | 0.5806      | 40.76                  |
        | ExecuTorch      | ✅      | 36.2              | 0.5782      | 67.21                  |

    === "YOLO11m"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.6250      | 11.4                   |
        | TorchScript     | ✅      | 77.3              | 0.6304      | 10.16                  |
        | ONNX            | ✅      | 76.9              | 0.6304      | 12.35                  |
        | OpenVINO        | ✅      | 77.1              | 0.6284      | 77.81                  |
        | TensorRT (FP32) | ✅      | 80.7              | 0.6305      | 5.29                   |
        | TensorRT (FP16) | ✅      | 41.3              | 0.6294      | 2.42                   |
        | TensorRT (INT8) | ✅      | 23.7              | 0.6133      | 2.20                   |
        | TF SavedModel   | ✅      | 192.4             | 0.6306      | 184.66                 |
        | TF GraphDef     | ✅      | 76.9              | 0.6306      | 187.91                 |
        | TF Lite         | ✅      | 76.9              | 0.6306      | 1845.09                |
        | MNN             | ✅      | 76.8              | 0.6298      | 143.52                 |
        | NCNN            | ✅      | 76.9              | 0.6308      | 95.86                  |
        | ExecuTorch      | ✅      | 76.9              | 0.6306      | 167.94                 |

    === "YOLO11l"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.6370      | 14.0                   |
        | TorchScript     | ✅      | 97.6              | 0.6409      | 13.77                  |
        | ONNX            | ✅      | 97.0              | 0.6410      | 16.37                  |
        | OpenVINO        | ✅      | 97.3              | 0.6377      | 98.86                  |
        | TensorRT (FP32) | ✅      | 101.0             | 0.6396      | 6.71                   |
        | TensorRT (FP16) | ✅      | 51.5              | 0.6358      | 3.26                   |
        | TensorRT (INT8) | ✅      | 29.7              | 0.6190      | 3.21                   |
        | TF SavedModel   | ✅      | 242.7             | 0.6409      | 246.93                 |
        | TF GraphDef     | ✅      | 97.0              | 0.6409      | 251.84                 |
        | TF Lite         | ✅      | 97.0              | 0.6409      | 2383.45                |
        | MNN             | ✅      | 96.9              | 0.6361      | 176.53                 |
        | NCNN            | ✅      | 97.0              | 0.6373      | 118.05                 |
        | ExecuTorch      | ✅      | 97.0              | 0.6409      | 211.46                 |

    === "YOLO11x"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.6990      | 21.70                  |
        | TorchScript     | ✅      | 218.1             | 0.6900      | 20.99                  |
        | ONNX            | ✅      | 217.5             | 0.6900      | 24.07                  |
        | OpenVINO        | ✅      | 217.8             | 0.6872      | 187.33                 |
        | TensorRT (FP32) | ✅      | 220.0             | 0.6902      | 11.70                  |
        | TensorRT (FP16) | ✅      | 114.6             | 0.6881      | 5.10                   |
        | TensorRT (INT8) | ✅      | 59.9              | 0.6857      | 4.53                   |
        | TF SavedModel   | ✅      | 543.9             | 0.6900      | 489.91                 |
        | TF GraphDef     | ✅      | 217.5             | 0.6900      | 503.21                 |
        | TF Lite         | ✅      | 217.5             | 0.6900      | 5164.31                |
        | MNN             | ✅      | 217.3             | 0.6905      | 350.37                 |
        | NCNN            | ✅      | 217.5             | 0.6901      | 230.63                 |
        | ExecuTorch      | ✅      | 217.4             | 0.6900      | 419.9                  |

    Benchmarked with Ultralytics 8.3.226

    !!! note

        Inference time does not include pre/ post-processing.

#### NVIDIA Jetson AGX Orin Developer Kit (64GB)

!!! tip "Performance"

    === "YOLO11n"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.5101      | 9.40                   |
        | TorchScript     | ✅      | 10.5              | 0.5083      | 11.00                  |
        | ONNX            | ✅      | 10.2              | 0.5077      | 48.32                  |
        | OpenVINO        | ✅      | 10.4              | 0.5058      | 27.24                  |
        | TensorRT (FP32) | ✅      | 12.1              | 0.5085      | 3.93                   |
        | TensorRT (FP16) | ✅      | 8.3               | 0.5063      | 2.55                   |
        | TensorRT (INT8) | ✅      | 5.4               | 0.4719      | 2.18                   |
        | TF SavedModel   | ✅      | 25.9              | 0.5077      | 66.87                  |
        | TF GraphDef     | ✅      | 10.3              | 0.5077      | 65.68                  |
        | TF Lite         | ✅      | 10.3              | 0.5077      | 272.92                 |
        | MNN             | ✅      | 10.1              | 0.5059      | 36.33                  |
        | NCNN            | ✅      | 10.2              | 0.5031      | 28.51                  |

    === "YOLO11s"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.5783      | 12.10                  |
        | TorchScript     | ✅      | 36.5              | 0.5782      | 11.01                  |
        | ONNX            | ✅      | 36.3              | 0.5782      | 107.54                 |
        | OpenVINO        | ✅      | 36.4              | 0.5810      | 55.03                  |
        | TensorRT (FP32) | ✅      | 38.1              | 0.5781      | 6.52                   |
        | TensorRT (FP16) | ✅      | 21.4              | 0.5803      | 3.65                   |
        | TensorRT (INT8) | ✅      | 12.1              | 0.5735      | 2.81                   |
        | TF SavedModel   | ✅      | 91.0              | 0.5782      | 132.73                 |
        | TF GraphDef     | ✅      | 36.4              | 0.5782      | 134.96                 |
        | TF Lite         | ✅      | 36.3              | 0.5782      | 798.21                 |
        | MNN             | ✅      | 36.2              | 0.5777      | 82.35                  |
        | NCNN            | ✅      | 36.2              | 0.5784      | 56.07                  |

    === "YOLO11m"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.6265      | 22.20                  |
        | TorchScript     | ✅      | 77.3              | 0.6307      | 21.47                  |
        | ONNX            | ✅      | 76.9              | 0.6307      | 270.89                 |
        | OpenVINO        | ✅      | 77.1              | 0.6284      | 129.10                 |
        | TensorRT (FP32) | ✅      | 78.8              | 0.6306      | 12.53                  |
        | TensorRT (FP16) | ✅      | 41.9              | 0.6305      | 6.25                   |
        | TensorRT (INT8) | ✅      | 23.2              | 0.6291      | 4.69                   |
        | TF SavedModel   | ✅      | 192.7             | 0.6307      | 299.95                 |
        | TF GraphDef     | ✅      | 77.1              | 0.6307      | 310.58                 |
        | TF Lite         | ✅      | 77.0              | 0.6307      | 2400.54                |
        | MNN             | ✅      | 76.8              | 0.6308      | 213.56                 |
        | NCNN            | ✅      | 76.8              | 0.6284      | 141.18                 |

    === "YOLO11l"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.6364      | 27.70                   |
        | TorchScript     | ✅      | 97.6              | 0.6399      | 27.94                  |
        | ONNX            | ✅      | 97.0              | 0.6409      | 345.47                 |
        | OpenVINO        | ✅      | 97.3              | 0.6378      | 161.93                 |
        | TensorRT (FP32) | ✅      | 99.1              | 0.6406      | 16.11                  |
        | TensorRT (FP16) | ✅      | 52.6              | 0.6376      | 8.08                   |
        | TensorRT (INT8) | ✅      | 30.8              | 0.6208      | 6.12                   |
        | TF SavedModel   | ✅      | 243.1             | 0.6409      | 390.78                 |
        | TF GraphDef     | ✅      | 97.2              | 0.6409      | 398.76                 |
        | TF Lite         | ✅      | 97.1              | 0.6409      | 3037.05                |
        | MNN             | ✅      | 96.9              | 0.6372      | 265.46                 |
        | NCNN            | ✅      | 96.9              | 0.6364      | 179.68                 |

    === "YOLO11x"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.7005      | 44.40                  |
        | TorchScript     | ✅      | 218.1             | 0.6898      | 47.49                  |
        | ONNX            | ✅      | 217.5             | 0.6900      | 682.98                 |
        | OpenVINO        | ✅      | 217.8             | 0.6876      | 298.15                 |
        | TensorRT (FP32) | ✅      | 219.6             | 0.6904      | 28.50                  |
        | TensorRT (FP16) | ✅      | 112.2             | 0.6887      | 13.55                  |
        | TensorRT (INT8) | ✅      | 60.0              | 0.6574      | 9.40                   |
        | TF SavedModel   | ✅      | 544.3             | 0.6900      | 749.85                 |
        | TF GraphDef     | ✅      | 217.7             | 0.6900      | 753.86                 |
        | TF Lite         | ✅      | 217.6             | 0.6900      | 6603.27                |
        | MNN             | ✅      | 217.3             | 0.6868      | 519.77                 |
        | NCNN            | ✅      | 217.3             | 0.6849      | 298.58                 |

    Benchmarked with Ultralytics 8.3.157

    !!! note

        Inference time does not include pre/ post-processing.

#### NVIDIA Jetson Orin Nano Super Developer Kit

!!! tip "Performance"

    === "YOLO11n"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.5101      | 13.70                  |
        | TorchScript     | ✅      | 10.5              | 0.5082      | 13.69                  |
        | ONNX            | ✅      | 10.2              | 0.5081      | 14.47                  |
        | OpenVINO        | ✅      | 10.4              | 0.5058      | 56.66                  |
        | TensorRT (FP32) | ✅      | 12.0              | 0.5081      | 7.44                   |
        | TensorRT (FP16) | ✅      | 8.2               | 0.5061      | 4.53                   |
        | TensorRT (INT8) | ✅      | 5.4               | 0.4825      | 3.70                   |
        | TF SavedModel   | ✅      | 25.9              | 0.5077      | 116.23                 |
        | TF GraphDef     | ✅      | 10.3              | 0.5077      | 114.92                 |
        | TF Lite         | ✅      | 10.3              | 0.5077      | 340.75                 |
        | MNN             | ✅      | 10.1              | 0.5059      | 76.26                  |
        | NCNN            | ✅      | 10.2              | 0.5031      | 45.03                  |

    === "YOLO11s"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.5790      | 20.90                  |
        | TorchScript     | ✅      | 36.5              | 0.5781      | 21.22                  |
        | ONNX            | ✅      | 36.3              | 0.5781      | 25.07                  |
        | OpenVINO        | ✅      | 36.4              | 0.5810      | 122.98                 |
        | TensorRT (FP32) | ✅      | 37.9              | 0.5783      | 13.02                  |
        | TensorRT (FP16) | ✅      | 21.8              | 0.5779      | 6.93                   |
        | TensorRT (INT8) | ✅      | 12.2              | 0.5735      | 5.08                   |
        | TF SavedModel   | ✅      | 91.0              | 0.5782      | 250.65                 |
        | TF GraphDef     | ✅      | 36.4              | 0.5782      | 252.69                 |
        | TF Lite         | ✅      | 36.3              | 0.5782      | 998.68                 |
        | MNN             | ✅      | 36.2              | 0.5781      | 188.01                 |
        | NCNN            | ✅      | 36.2              | 0.5784      | 101.37                 |

    === "YOLO11m"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.6266      | 46.50                  |
        | TorchScript     | ✅      | 77.3              | 0.6307      | 47.95                  |
        | ONNX            | ✅      | 76.9              | 0.6307      | 53.06                  |
        | OpenVINO        | ✅      | 77.1              | 0.6284      | 301.63                 |
        | TensorRT (FP32) | ✅      | 78.8              | 0.6305      | 27.86                  |
        | TensorRT (FP16) | ✅      | 41.7              | 0.6309      | 13.50                  |
        | TensorRT (INT8) | ✅      | 23.2              | 0.6291      | 9.12                   |
        | TF SavedModel   | ✅      | 192.7             | 0.6307      | 622.24                 |
        | TF GraphDef     | ✅      | 77.1              | 0.6307      | 628.74                 |
        | TF Lite         | ✅      | 77.0              | 0.6307      | 2997.93                |
        | MNN             | ✅      | 76.8              | 0.6299      | 509.96                 |
        | NCNN            | ✅      | 76.8              | 0.6284      | 292.99                 |

    === "YOLO11l"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.6364      | 56.50                  |
        | TorchScript     | ✅      | 97.6              | 0.6409      | 62.51                  |
        | ONNX            | ✅      | 97.0              | 0.6399      | 68.35                  |
        | OpenVINO        | ✅      | 97.3              | 0.6378      | 376.03                 |
        | TensorRT (FP32) | ✅      | 99.2              | 0.6396      | 35.59                  |
        | TensorRT (FP16) | ✅      | 52.1              | 0.6361      | 17.48                  |
        | TensorRT (INT8) | ✅      | 30.9              | 0.6207      | 11.87                  |
        | TF SavedModel   | ✅      | 243.1             | 0.6409      | 807.47                 |
        | TF GraphDef     | ✅      | 97.2              | 0.6409      | 822.88                 |
        | TF Lite         | ✅      | 97.1              | 0.6409      | 3792.23                |
        | MNN             | ✅      | 96.9              | 0.6372      | 631.16                 |
        | NCNN            | ✅      | 96.9              | 0.6364      | 350.46                 |

    === "YOLO11x"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.7005      | 90.00                  |
        | TorchScript     | ✅      | 218.1             | 0.6901      | 113.40                 |
        | ONNX            | ✅      | 217.5             | 0.6901      | 122.94                 |
        | OpenVINO        | ✅      | 217.8             | 0.6876      | 713.1                  |
        | TensorRT (FP32) | ✅      | 219.5             | 0.6904      | 66.93                  |
        | TensorRT (FP16) | ✅      | 112.2             | 0.6892      | 32.58                  |
        | TensorRT (INT8) | ✅      | 61.5              | 0.6612      | 19.90                  |
        | TF SavedModel   | ✅      | 544.3             | 0.6900      | 1605.4                 |
        | TF GraphDef     | ✅      | 217.8             | 0.6900      | 2961.8                 |
        | TF Lite         | ✅      | 217.6             | 0.6900      | 8234.86                |
        | MNN             | ✅      | 217.3             | 0.6893      | 1254.18                |
        | NCNN            | ✅      | 217.3             | 0.6849      | 725.50                 |

    Benchmarked with Ultralytics 8.3.157

    !!! note

        Inference time does not include pre/ post-processing.

#### NVIDIA Jetson Orin NX 16GB

!!! tip "Performance"

    === "YOLO11n"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.5101      | 12.90                  |
        | TorchScript     | ✅      | 10.5              | 0.5082      | 13.17                  |
        | ONNX            | ✅      | 10.2              | 0.5081      | 15.43                  |
        | OpenVINO        | ✅      | 10.4              | 0.5058      | 39.80                  |
        | TensorRT (FP32) | ✅      | 11.8              | 0.5081      | 7.94                   |
        | TensorRT (FP16) | ✅      | 8.1               | 0.5085      | 4.73                   |
        | TensorRT (INT8) | ✅      | 5.4               | 0.4786      | 3.90                   |
        | TF SavedModel   | ✅      | 25.9              | 0.5077      | 88.48                  |
        | TF GraphDef     | ✅      | 10.3              | 0.5077      | 86.67                  |
        | TF Lite         | ✅      | 10.3              | 0.5077      | 302.55                 |
        | MNN             | ✅      | 10.1              | 0.5059      | 52.73                  |
        | NCNN            | ✅      | 10.2              | 0.5031      | 32.04                  |

    === "YOLO11s"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.5790      | 21.70                  |
        | TorchScript     | ✅      | 36.5              | 0.5781      | 22.71                  |
        | ONNX            | ✅      | 36.3              | 0.5781      | 26.49                  |
        | OpenVINO        | ✅      | 36.4              | 0.5810      | 84.73                  |
        | TensorRT (FP32) | ✅      | 37.8              | 0.5783      | 13.77                  |
        | TensorRT (FP16) | ✅      | 21.2              | 0.5796      | 7.31                   |
        | TensorRT (INT8) | ✅      | 12.0              | 0.5735      | 5.33                   |
        | TF SavedModel   | ✅      | 91.0              | 0.5782      | 185.06                 |
        | TF GraphDef     | ✅      | 36.4              | 0.5782      | 186.45                 |
        | TF Lite         | ✅      | 36.3              | 0.5782      | 882.58                 |
        | MNN             | ✅      | 36.2              | 0.5775      | 126.36                 |
        | NCNN            | ✅      | 36.2              | 0.5784      | 66.73                  |

    === "YOLO11m"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.6266      | 45.00                  |
        | TorchScript     | ✅      | 77.3              | 0.6307      | 51.87                  |
        | ONNX            | ✅      | 76.9              | 0.6307      | 56.00                  |
        | OpenVINO        | ✅      | 77.1              | 0.6284      | 202.69                 |
        | TensorRT (FP32) | ✅      | 78.7              | 0.6305      | 30.38                  |
        | TensorRT (FP16) | ✅      | 41.8              | 0.6302      | 14.48                  |
        | TensorRT (INT8) | ✅      | 23.2              | 0.6291      | 9.74                   |
        | TF SavedModel   | ✅      | 192.7             | 0.6307      | 445.58                 |
        | TF GraphDef     | ✅      | 77.1              | 0.6307      | 460.94                 |
        | TF Lite         | ✅      | 77.0              | 0.6307      | 2653.65                |
        | MNN             | ✅      | 76.8              | 0.6308      | 339.38                 |
        | NCNN            | ✅      | 76.8              | 0.6284      | 187.64                 |

    === "YOLO11l"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.6364      | 56.60                  |
        | TorchScript     | ✅      | 97.6              | 0.6409      | 66.72                  |
        | ONNX            | ✅      | 97.0              | 0.6399      | 71.92                  |
        | OpenVINO        | ✅      | 97.3              | 0.6378      | 254.17                 |
        | TensorRT (FP32) | ✅      | 99.2              | 0.6406      | 38.89                  |
        | TensorRT (FP16) | ✅      | 51.9              | 0.6363      | 18.59                  |
        | TensorRT (INT8) | ✅      | 30.9              | 0.6207      | 12.60                  |
        | TF SavedModel   | ✅      | 243.1             | 0.6409      | 575.98                 |
        | TF GraphDef     | ✅      | 97.2              | 0.6409      | 583.79                 |
        | TF Lite         | ✅      | 97.1              | 0.6409      | 3353.41                |
        | MNN             | ✅      | 96.9              | 0.6367      | 421.33                 |
        | NCNN            | ✅      | 96.9              | 0.6364      | 228.26                 |

    === "YOLO11x"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.7005      | 98.50                  |
        | TorchScript     | ✅      | 218.1             | 0.6901      | 123.03                 |
        | ONNX            | ✅      | 217.5             | 0.6901      | 129.55                 |
        | OpenVINO        | ✅      | 217.8             | 0.6876      | 483.44                 |
        | TensorRT (FP32) | ✅      | 219.6             | 0.6904      | 75.92                  |
        | TensorRT (FP16) | ✅      | 112.1             | 0.6885      | 35.78                  |
        | TensorRT (INT8) | ✅      | 61.6              | 0.6592      | 21.60                  |
        | TF SavedModel   | ✅      | 544.3             | 0.6900      | 1120.43                |
        | TF GraphDef     | ✅      | 217.7             | 0.6900      | 1172.35                |
        | TF Lite         | ✅      | 217.6             | 0.6900      | 7283.63                |
        | MNN             | ✅      | 217.3             | 0.6877      | 840.16                 |
        | NCNN            | ✅      | 217.3             | 0.6849      | 474.41                 |

    Benchmarked with Ultralytics 8.3.157

    !!! note

        Inference time does not include pre/ post-processing.

[Explore more benchmarking efforts by Seeed Studio](https://www.seeedstudio.com/blog/2023/03/30/yolov8-performance-benchmarks-on-nvidia-jetson-devices/) running on different versions of NVIDIA Jetson hardware.

## Reproduce Our Results

To reproduce the above Ultralytics benchmarks on all export [formats](../modes/export.md) run this code:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all all export formats
        results = model.benchmark(data="coco128.yaml", imgsz=640)
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all all export formats
        yolo benchmark model=yolo11n.pt data=coco128.yaml imgsz=640
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results use a dataset with a large number of images, i.e. `data='coco.yaml'` (5000 val images).

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

YOLO11 models have been benchmarked on various NVIDIA Jetson devices showing significant performance improvements. For example, the TensorRT format delivers the best inference performance. The table in the [Detailed Comparison Tables](#detailed-comparison-tables) section provides a comprehensive view of performance metrics like mAP50-95 and inference time across different model formats.

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
