---
comments: true
description: Learn to deploy Ultralytics YOLO11 on NVIDIA Jetson devices with our detailed guide. Explore performance benchmarks and maximize AI capabilities.
keywords: Ultralytics, YOLO11, NVIDIA Jetson, JetPack, AI deployment, performance benchmarks, embedded systems, deep learning, TensorRT, computer vision
---

# Quick Start Guide: NVIDIA Jetson with Ultralytics YOLO11

This comprehensive guide provides a detailed walkthrough for deploying Ultralytics YOLO11 on [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) devices. Additionally, it showcases performance benchmarks to demonstrate the capabilities of YOLO11 on these small and powerful devices.

!!! tip "New product support"

    We have updated this guide with the latest [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit) which delivers up to 67 TOPS of AI performance — a 1.7X improvement over its predecessor — to seamlessly run the most popular AI models.

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

    This guide has been tested with [NVIDIA Jetson AGX Orin Developer Kit (64GB)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin) running the latest stable JetPack release of [JP6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62), [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit) running JetPack release of [JP6.1](https://developer.nvidia.com/embedded/jetpack-sdk-61), [Seeed Studio reComputer J4012](https://www.seeedstudio.com/reComputer-J4012-p-5586.html) which is based on NVIDIA Jetson Orin NX 16GB running JetPack release of [JP6.0](https://developer.nvidia.com/embedded/jetpack-sdk-60)/ JetPack release of [JP5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) and [Seeed Studio reComputer J1020 v2](https://www.seeedstudio.com/reComputer-J1020-v2-p-5498.html) which is based on NVIDIA Jetson Nano 4GB running JetPack release of [JP4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461). It is expected to work across all the NVIDIA Jetson hardware lineup including latest and legacy.

## What is NVIDIA Jetson?

NVIDIA Jetson is a series of embedded computing boards designed to bring accelerated AI (artificial intelligence) computing to edge devices. These compact and powerful devices are built around NVIDIA's GPU architecture and are capable of running complex AI algorithms and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models directly on the device, without needing to rely on [cloud computing](https://www.ultralytics.com/glossary/cloud-computing) resources. Jetson boards are often used in robotics, autonomous vehicles, industrial automation, and other applications where AI inference needs to be performed locally with low latency and high efficiency. Additionally, these boards are based on the ARM64 architecture and runs on lower power compared to traditional GPU computing devices.

## NVIDIA Jetson Series Comparison

[Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) is the latest iteration of the NVIDIA Jetson family based on NVIDIA Ampere architecture which brings drastically improved AI performance when compared to the previous generations. Below table compared few of the Jetson devices in the ecosystem.

|                   | Jetson AGX Orin 64GB                                              | Jetson Orin NX 16GB                                              | Jetson Orin Nano Super                                        | Jetson AGX Xavier                                           | Jetson Xavier NX                                              | Jetson Nano                                   |
| ----------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| AI Performance    | 275 TOPS                                                          | 100 TOPS                                                         | 67 TOPs                                                       | 32 TOPS                                                     | 21 TOPS                                                       | 472 GFLOPS                                    |
| GPU               | 2048-core NVIDIA Ampere architecture GPU with 64 Tensor Cores     | 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores    | 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores | 512-core NVIDIA Volta architecture GPU with 64 Tensor Cores | 384-core NVIDIA Volta™ architecture GPU with 48 Tensor Cores | 128-core NVIDIA Maxwell™ architecture GPU    |
| GPU Max Frequency | 1.3 GHz                                                           | 918 MHz                                                          | 1020 MHz                                                      | 1377 MHz                                                    | 1100 MHz                                                      | 921MHz                                        |
| CPU               | 12-core NVIDIA Arm® Cortex A78AE v8.2 64-bit CPU 3MB L2 + 6MB L3 | 8-core NVIDIA Arm® Cortex A78AE v8.2 64-bit CPU 2MB L2 + 4MB L3 | 6-core Arm® Cortex®-A78AE v8.2 64-bit CPU 1.5MB L2 + 4MB L3 | 8-core NVIDIA Carmel Arm®v8.2 64-bit CPU 8MB L2 + 4MB L3   | 6-core NVIDIA Carmel Arm®v8.2 64-bit CPU 6MB L2 + 4MB L3     | Quad-Core Arm® Cortex®-A57 MPCore processor |
| CPU Max Frequency | 2.2 GHz                                                           | 2.0 GHz                                                          | 1.7 GHz                                                       | 2.2 GHz                                                     | 1.9 GHz                                                       | 1.43GHz                                       |
| Memory            | 64GB 256-bit LPDDR5 204.8GB/s                                     | 16GB 128-bit LPDDR5 102.4GB/s                                    | 8GB 128-bit LPDDR5 102 GB/s                                   | 32GB 256-bit LPDDR4x 136.5GB/s                              | 8GB 128-bit LPDDR4x 59.7GB/s                                  | 4GB 64-bit LPDDR4 25.6GB/s"                   |

For a more detailed comparison table, please visit the **Technical Specifications** section of [official NVIDIA Jetson page](https://developer.nvidia.com/embedded/jetson-modules).

## What is NVIDIA JetPack?

[NVIDIA JetPack SDK](https://developer.nvidia.com/embedded/jetpack) powering the Jetson modules is the most comprehensive solution and provides full development environment for building end-to-end accelerated AI applications and shortens time to market. JetPack includes Jetson Linux with bootloader, Linux kernel, Ubuntu desktop environment, and a complete set of libraries for acceleration of GPU computing, multimedia, graphics, and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). It also includes samples, documentation, and developer tools for both host computer and developer kit, and supports higher level SDKs such as [DeepStream](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson/) for streaming video analytics, Isaac for robotics, and Riva for conversational AI.

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

The above ultralytics installation will install Torch and Torchvision. However, these 2 packages installed via pip are not compatible to run on Jetson platform which is based on ARM64 architecture. Therefore, we need to manually install pre-built PyTorch pip wheel and compile/ install Torchvision from source.

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

The above ultralytics installation will install Torch and Torchvision. However, these 2 packages installed via pip are not compatible to run on Jetson platform which is based on ARM64 architecture. Therefore, we need to manually install pre-built PyTorch pip wheel and compile/ install Torchvision from source.

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

## NVIDIA Jetson Orin YOLO11 Benchmarks

YOLO11 benchmarks were run by the Ultralytics team on 10 different model formats measuring speed and [accuracy](https://www.ultralytics.com/glossary/accuracy): PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, TF SavedModel, TF GraphDef, TF Lite, PaddlePaddle, NCNN. Benchmarks were run on NVIDIA Jetson AGX Orin Developer Kit (64GB), NVIDIA Jetson Orin Nano Super Developer Kit and Seeed Studio reComputer J4012 powered by Jetson Orin NX 16GB device at FP32 [precision](https://www.ultralytics.com/glossary/precision) with default input image size of 640.

### Comparison Charts

Even though all model exports are working with NVIDIA Jetson, we have only included **PyTorch, TorchScript, TensorRT** for the comparison chart below because, they make use of the GPU on the Jetson and are guaranteed to produce the best results. All the other exports only utilize the CPU and the performance is not as good as the above three. You can find benchmarks for all exports in the section after this chart.

#### NVIDIA Jetson AGX Orin Developer Kit (64GB)

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-agx-orin-benchmarks.avif" alt="Jetson AGX Orin Benchmarks">
    <figcaption style="font-style: italic; color: gray;">Benchmarked with Ultralytics 8.3.133</figcaption>
</figure>

#### NVIDIA Jetson Orin Nano Super Developer Kit

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-orin-nano-super-benchmarks.avif" alt="Jetson Orin Nano Super Benchmarks">
    <figcaption style="font-style: italic; color: gray;">Benchmarked with Ultralytics 8.3.51</figcaption>
</figure>

#### NVIDIA Jetson Orin NX 16GB

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-orin-nx-16-benchmarks.avif" alt="Jetson Orin NX 16GB Benchmarks">
    <figcaption style="font-style: italic; color: gray;">Benchmarked with Ultralytics 8.3.51</figcaption>
</figure>

### Detailed Comparison Tables

The below table represents the benchmark results for five different models (YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x) across ten different formats (PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, TF SavedModel, TF GraphDef, TF Lite, PaddlePaddle, NCNN), giving us the status, size, mAP50-95(B) metric, and inference time for each combination.

#### NVIDIA Jetson AGX Orin Developer Kit (64GB)

!!! tip "Performance"

    === "YOLO11n"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.6176      | 9.4                    |
        | TorchScript     | ✅      | 10.5              | 0.6100      | 11.43                  |
        | ONNX            | ✅      | 10.2              | 0.6100      | 10.25                  |
        | OpenVINO        | ✅      | 10.4              | 0.6091      | 29.05                  |
        | TensorRT (FP32) | ✅      | 12.0              | 0.6100      | 3.94                   |
        | TensorRT (FP16) | ✅      | 8.1               | 0.6096      | 2.60                   |
        | TensorRT (INT8) | ✅      | 5.4               | 0.6050      | 2.45                   |
        | TF SavedModel   | ✅      | 25.9              | 0.6082      | 219.06                 |
        | TF GraphDef     | ✅      | 10.3              | 0.6082      | 230.01                 |
        | TF Lite         | ✅      | 10.3              | 0.6082      | 274.16                 |
        | PaddlePaddle    | ✅      | 20.4              | 0.6082      | 431.55                 |
        | MNN             | ✅      | 10.1              | 0.6099      | 37.11                  |
        | NCNN            | ✅      | 10.2              | 0.6101      | 28.22                  |

    === "YOLO11s"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.7526      | 12.0                   |
        | TorchScript     | ✅      | 36.5              | 0.7400      | 11.35                  |
        | ONNX            | ✅      | 36.3              | 0.7400      | 13.64                  |
        | OpenVINO        | ✅      | 36.4              | 0.7391      | 59.97                  |
        | TensorRT (FP32) | ✅      | 38.0              | 0.7400      | 6.52                   |
        | TensorRT (FP16) | ✅      | 21.5              | 0.7406      | 3.68                   |
        | TensorRT (INT8) | ✅      | 12.2              | 0.7230      | 3.01                   |
        | TF SavedModel   | ✅      | 91.0              | 0.7400      | 282.34                 |
        | TF GraphDef     | ✅      | 36.4              | 0.7400      | 376.11                 |
        | TF Lite         | ✅      | 36.3              | 0.7400      | 801.09                 |
        | PaddlePaddle    | ✅      | 72.5              | 0.7400      | 1074.64                |
        | MNN             | ✅      | 36.2              | 0.7396      | 84.02                  |
        | NCNN            | ✅      | 36.2              | 0.7380      | 58.17                 |

    === "YOLO11m"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.7598      | 21.7                   |
        | TorchScript     | ✅      | 77.3              | 0.7643      | 22.92                  |
        | ONNX            | ✅      | 76.9              | 0.7643      | 25.24                  |
        | OpenVINO        | ✅      | 77.1              | 0.7642      | 136.59                 |
        | TensorRT (FP32) | ✅      | 78.8              | 0.7640      | 12.69                  |
        | TensorRT (FP16) | ✅      | 41.8              | 0.7653      | 6.79                   |
        | TensorRT (INT8) | ✅      | 23.2              | 0.4194      | 5.12                   |
        | TF SavedModel   | ✅      | 192.7             | 0.7643      | 489.91                 |
        | TF GraphDef     | ✅      | 77.1              | 0.7643      | 716.25                 |
        | TF Lite         | ✅      | 77.0              | 0.7643      | 2402.99                |
        | PaddlePaddle    | ✅      | 153.8             | 0.7643      | 2881.60                |
        | MNN             | ✅      | 76.8              | 0.7649      | 215.16                 |
        | NCNN            | ✅      | 76.8              | 0.7650      | 142.53                 |

    === "YOLO11l"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.7475      | 27.7                   |
        | TorchScript     | ✅      | 97.6              | 0.7250      | 27.95                  |
        | ONNX            | ✅      | 97.0              | 0.7250      | 32.0                   |
        | OpenVINO        | ✅      | 97.3              | 0.7226      | 174.37                 |
        | TensorRT (FP32) | ✅      | 99.3              | 0.7250      | 16.31                  |
        | TensorRT (FP16) | ✅      | 52.0              | 0.7265      | 8.11                   |
        | TensorRT (INT8) | ✅      | 31.0              | 0.4033      | 6.60                   |
        | TF SavedModel   | ✅      | 243.2             | 0.7250      | 683.88                 |
        | TF GraphDef     | ✅      | 97.2              | 0.7250      | 1042.83                |
        | TF Lite         | ✅      | 97.1              | 0.7250      | 3027.16                |
        | PaddlePaddle    | ✅      | 194.1             | 0.7250      | 3775.25                |
        | MNN             | ✅      | 96.9              | 0.7206      | 266.75                 |
        | NCNN            | ✅      | 96.9              | 0.7216      | 174.21                 |

    === "YOLO11x"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.8288      | 44.2                   |
        | TorchScript     | ✅      | 218.1             | 0.8308      | 47.81                  |
        | ONNX            | ✅      | 217.5             | 0.8308      | 79.71                  |
        | OpenVINO        | ✅      | 217.8             | 0.8285      | 329.7                  |
        | TensorRT (FP32) | ✅      | 219.7             | 0.8307      | 28.4                   |
        | TensorRT (FP16) | ✅      | 112.2             | 0.8248      | 13.66                  |
        | TensorRT (INT8) | ✅      | 61.7              | 0.4854      | 9.78                   |
        | TF SavedModel   | ✅      | 544.4             | 0.8308      | 1027.82                |
        | TF GraphDef     | ✅      | 217.7             | 0.8308      | 1902.75                |
        | TF Lite         | ✅      | 217.6             | 0.8308      | 6616.01                |
        | PaddlePaddle    | ✅      | 435.0             | 0.8308      | 7589.03                |
        | MNN             | ✅      | 217.3             | 0.8286      | 522.32                 |
        | NCNN            | ✅      | 217.3             | 0.8277      | 301.36                 |

    Benchmarked with Ultralytics 8.3.133

#### NVIDIA Jetson Orin Nano Super Developer Kit

!!! tip "Performance"

    === "YOLO11n"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.6176      | 21.3                   |
        | TorchScript     | ✅      | 10.5              | 0.6100      | 13.40                  |
        | ONNX            | ✅      | 10.2              | 0.6100      | 7.94                   |
        | OpenVINO        | ✅      | 10.4              | 0.6091      | 57.36                  |
        | TensorRT (FP32) | ✅      | 11.9              | 0.6082      | 7.60                   |
        | TensorRT (FP16) | ✅      | 8.3               | 0.6096      | 4.91                   |
        | TensorRT (INT8) | ✅      | 5.6               | 0.3180      | 3.91                   |
        | TF SavedModel   | ✅      | 25.8              | 0.6082      | 223.98                 |
        | TF GraphDef     | ✅      | 10.3              | 0.6082      | 289.95                 |
        | TF Lite         | ✅      | 10.3              | 0.6082      | 328.29                 |
        | PaddlePaddle    | ✅      | 20.4              | 0.6082      | 530.46                 |
        | MNN             | ✅      | 10.1              | 0.6120      | 74.75                  |
        | NCNN            | ✅      | 10.2              | 0.6106      | 46.12                  |

    === "YOLO11s"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.7526      | 22.00                  |
        | TorchScript     | ✅      | 36.5              | 0.7400      | 21.35                  |
        | ONNX            | ✅      | 36.3              | 0.7400      | 13.91                  |
        | OpenVINO        | ✅      | 36.4              | 0.7391      | 126.95                 |
        | TensorRT (FP32) | ✅      | 38.0              | 0.7400      | 13.29                  |
        | TensorRT (FP16) | ✅      | 21.3              | 0.7431      | 7.30                   |
        | TensorRT (INT8) | ✅      | 12.2              | 0.3243      | 5.25                   |
        | TF SavedModel   | ✅      | 91.1              | 0.7400      | 406.73                 |
        | TF GraphDef     | ✅      | 36.4              | 0.7400      | 629.80                 |
        | TF Lite         | ✅      | 36.4              | 0.7400      | 953.98                 |
        | PaddlePaddle    | ✅      | 72.5              | 0.7400      | 1311.67                |
        | MNN             | ✅      | 36.2              | 0.7392      | 187.66                 |
        | NCNN            | ✅      | 36.2              | 0.7403      | 122.02                 |

    === "YOLO11m"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.7598      | 33.00                  |
        | TorchScript     | ✅      | 77.3              | 0.7643      | 48.17                  |
        | ONNX            | ✅      | 76.9              | 0.7641      | 29.31                  |
        | OpenVINO        | ✅      | 77.1              | 0.7642      | 313.49                 |
        | TensorRT (FP32) | ✅      | 78.7              | 0.7641      | 28.21                  |
        | TensorRT (FP16) | ✅      | 41.8              | 0.7653      | 13.99                  |
        | TensorRT (INT8) | ✅      | 23.2              | 0.4194      | 9.58                   |
        | TF SavedModel   | ✅      | 192.7             | 0.7643      | 802.30                 |
        | TF GraphDef     | ✅      | 77.0              | 0.7643      | 1335.42                |
        | TF Lite         | ✅      | 77.0              | 0.7643      | 2842.42                |
        | PaddlePaddle    | ✅      | 153.8             | 0.7643      | 3644.29                |
        | MNN             | ✅      | 76.8              | 0.7648      | 503.90                 |
        | NCNN            | ✅      | 76.8              | 0.7674      | 298.78                 |

    === "YOLO11l"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.7475      | 43.00                  |
        | TorchScript     | ✅      | 97.6              | 0.7250      | 62.94                  |
        | ONNX            | ✅      | 97.0              | 0.7250      | 36.33                  |
        | OpenVINO        | ✅      | 97.3              | 0.7226      | 387.72                 |
        | TensorRT (FP32) | ✅      | 99.1              | 0.7250      | 35.59                  |
        | TensorRT (FP16) | ✅      | 52.0              | 0.7265      | 17.57                  |
        | TensorRT (INT8) | ✅      | 31.0              | 0.4033      | 12.37                  |
        | TF SavedModel   | ✅      | 243.3             | 0.7250      | 1116.20                |
        | TF GraphDef     | ✅      | 97.2              | 0.7250      | 1603.32                |
        | TF Lite         | ✅      | 97.2              | 0.7250      | 3607.51                |
        | PaddlePaddle    | ✅      | 193.9             | 0.7250      | 4890.90                |
        | MNN             | ✅      | 96.9              | 0.7222      | 619.04                 |
        | NCNN            | ✅      | 96.9              | 0.7252      | 352.85                 |

    === "YOLO11x"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.8288      | 81.00                  |
        | TorchScript     | ✅      | 218.1             | 0.8308      | 113.49                 |
        | ONNX            | ✅      | 217.5             | 0.8308      | 75.20                  |
        | OpenVINO        | ✅      | 217.8             | 0.8285      | 508.12                 |
        | TensorRT (FP32) | ✅      | 219.5             | 0.8307      | 67.32                  |
        | TensorRT (FP16) | ✅      | 112.2             | 0.8248      | 32.94                  |
        | TensorRT (INT8) | ✅      | 61.7              | 0.4854      | 20.72                  |
        | TF SavedModel   | ✅      | 545.0             | 0.8308      | 1048.8                 |
        | TF GraphDef     | ✅      | 217.8             | 0.8308      | 2961.8                 |
        | TF Lite         | ✅      | 217.8             | 0.8308      | 7898.8                 |
        | PaddlePaddle    | ✅      | 434.8             | 0.8308      | 9903.68                |
        | MNN             | ✅      | 217.3             | 0.8308      | 1242.97                |
        | NCNN            | ✅      | 217.3             | 0.8304      | 850.05                 |

    Benchmarked with Ultralytics 8.3.51

#### NVIDIA Jetson Orin NX 16GB

!!! tip "Performance"

    === "YOLO11n"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.6176      | 19.50                  |
        | TorchScript     | ✅      | 10.5              | 0.6100      | 13.03                  |
        | ONNX            | ✅      | 10.2              | 0.6100      | 8.44                   |
        | OpenVINO        | ✅      | 10.4              | 0.6091      | 40.83                  |
        | TensorRT (FP32) | ✅      | 11.9              | 0.6100      | 8.05                   |
        | TensorRT (FP16) | ✅      | 8.2               | 0.6096      | 4.85                   |
        | TensorRT (INT8) | ✅      | 5.5               | 0.3180      | 4.37                   |
        | TF SavedModel   | ✅      | 25.8              | 0.6082      | 185.39                 |
        | TF GraphDef     | ✅      | 10.3              | 0.6082      | 244.85                 |
        | TF Lite         | ✅      | 10.3              | 0.6082      | 289.77                 |
        | PaddlePaddle    | ✅      | 20.4              | 0.6082      | 476.52                 |
        | MNN             | ✅      | 10.1              | 0.6120      | 53.37                  |
        | NCNN            | ✅      | 10.2              | 0.6106      | 33.55                  |

    === "YOLO11s"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.7526      | 19.00                  |
        | TorchScript     | ✅      | 36.5              | 0.7400      | 22.90                  |
        | ONNX            | ✅      | 36.3              | 0.7400      | 14.44                  |
        | OpenVINO        | ✅      | 36.4              | 0.7391      | 88.70                  |
        | TensorRT (FP32) | ✅      | 37.9              | 0.7400      | 14.13                  |
        | TensorRT (FP16) | ✅      | 21.6              | 0.7406      | 7.55                   |
        | TensorRT (INT8) | ✅      | 12.2              | 0.3243      | 5.63                   |
        | TF SavedModel   | ✅      | 91.1              | 0.7400      | 317.61                 |
        | TF GraphDef     | ✅      | 36.4              | 0.7400      | 515.99                 |
        | TF Lite         | ✅      | 36.4              | 0.7400      | 838.85                 |
        | PaddlePaddle    | ✅      | 72.5              | 0.7400      | 1170.07                |
        | MNN             | ✅      | 36.2              | 0.7413      | 125.23                 |
        | NCNN            | ✅      | 36.2              | 0.7403      | 68.13                  |

    === "YOLO11m"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.7598      | 36.50                  |
        | TorchScript     | ✅      | 77.3              | 0.7643      | 52.55                  |
        | ONNX            | ✅      | 76.9              | 0.7640      | 31.16                  |
        | OpenVINO        | ✅      | 77.1              | 0.7642      | 208.57                 |
        | TensorRT (FP32) | ✅      | 78.7              | 0.7640      | 30.72                  |
        | TensorRT (FP16) | ✅      | 41.5              | 0.7651      | 14.45                  |
        | TensorRT (INT8) | ✅      | 23.3              | 0.4194      | 10.19                  |
        | TF SavedModel   | ✅      | 192.7             | 0.7643      | 590.11                 |
        | TF GraphDef     | ✅      | 77.0              | 0.7643      | 998.57                 |
        | TF Lite         | ✅      | 77.0              | 0.7643      | 2486.11                |
        | PaddlePaddle    | ✅      | 153.8             | 0.7643      | 3236.09                |
        | MNN             | ✅      | 76.8              | 0.7661      | 335.78                 |
        | NCNN            | ✅      | 76.8              | 0.7674      | 188.43                 |

    === "YOLO11l"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.7475      | 46.6                   |
        | TorchScript     | ✅      | 97.6              | 0.7250      | 66.54                  |
        | ONNX            | ✅      | 97.0              | 0.7250      | 39.55                  |
        | OpenVINO        | ✅      | 97.3              | 0.7226      | 262.44                 |
        | TensorRT (FP32) | ✅      | 99.2              | 0.7250      | 38.68                  |
        | TensorRT (FP16) | ✅      | 51.9              | 0.7265      | 18.53                  |
        | TensorRT (INT8) | ✅      | 30.9              | 0.4033      | 13.36                  |
        | TF SavedModel   | ✅      | 243.3             | 0.7250      | 850.25                 |
        | TF GraphDef     | ✅      | 97.2              | 0.7250      | 1324.60                |
        | TF Lite         | ✅      | 97.2              | 0.7250      | 3191.24                |
        | PaddlePaddle    | ✅      | 193.9             | 0.7250      | 4204.97                |
        | MNN             | ✅      | 96.9              | 0.7225      | 414.41                 |
        | NCNN            | ✅      | 96.9              | 0.7252      | 237.74                 |

    === "YOLO11x"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.8288      | 86.00                  |
        | TorchScript     | ✅      | 218.1             | 0.8308      | 122.43                 |
        | ONNX            | ✅      | 217.5             | 0.8307      | 77.50                  |
        | OpenVINO        | ✅      | 217.8             | 0.8285      | 508.12                 |
        | TensorRT (FP32) | ✅      | 219.5             | 0.8307      | 76.44                  |
        | TensorRT (FP16) | ✅      | 112.0             | 0.8309      | 35.99                  |
        | TensorRT (INT8) | ✅      | 61.6              | 0.4854      | 22.32                  |
        | TF SavedModel   | ✅      | 545.0             | 0.8308      | 1470.06                |
        | TF GraphDef     | ✅      | 217.8             | 0.8308      | 2549.78                |
        | TF Lite         | ✅      | 217.8             | 0.8308      | 7025.44                |
        | PaddlePaddle    | ✅      | 434.8             | 0.8308      | 8364.89                |
        | MNN             | ✅      | 217.3             | 0.8289      | 827.13                 |
        | NCNN            | ✅      | 217.3             | 0.8304      | 490.29                 |

    Benchmarked with Ultralytics 8.3.51

[Explore more benchmarking efforts by Seeed Studio](https://www.seeedstudio.com/blog/2023/03/30/yolov8-performance-benchmarks-on-nvidia-jetson-devices/) running on different versions of NVIDIA Jetson hardware.

## Reproduce Our Results

To reproduce the above Ultralytics benchmarks on all export [formats](../modes/export.md) run this code:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Benchmark YOLO11n speed and accuracy on the COCO8 dataset for all all export formats
        results = model.benchmark(data="coco8.yaml", imgsz=640)
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO11n speed and accuracy on the COCO8 dataset for all all export formats
        yolo benchmark model=yolo11n.pt data=coco8.yaml imgsz=640
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results use a dataset with a large number of images, i.e. `data='coco128.yaml'` (128 val images), or `data='coco.yaml'` (5000 val images).

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
