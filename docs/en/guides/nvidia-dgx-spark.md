---
comments: true
description: Learn to deploy Ultralytics YOLO26 on NVIDIA DGX Spark with our detailed guide. Explore performance benchmarks and maximize AI capabilities on this compact desktop AI supercomputer.
keywords: Ultralytics, YOLO26, NVIDIA DGX Spark, AI deployment, performance benchmarks, deep learning, TensorRT, computer vision, GB10 Grace Blackwell
---

# Quick Start Guide: NVIDIA DGX Spark with Ultralytics YOLO26

This comprehensive guide provides a detailed walkthrough for deploying Ultralytics YOLO26 on [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), NVIDIA's compact desktop AI supercomputer. Additionally, it showcases performance benchmarks to demonstrate the capabilities of YOLO26 on this powerful system.

<p align="center">
  <img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/nvidia-dgx-spark.avif" alt="NVIDIA DGX Spark AI workstation overview">
</p>

!!! note

    This guide has been tested with NVIDIA DGX Spark Founders Edition running DGX OS based on Ubuntu. It is expected to work with the latest DGX OS releases.

## What is NVIDIA DGX Spark?

NVIDIA DGX Spark is a compact desktop AI supercomputer powered by the NVIDIA GB10 Grace Blackwell Superchip. It delivers up to 1 petaFLOP of AI computing performance with FP4 precision, making it ideal for developers, researchers, and data scientists who need powerful AI capabilities in a desktop form factor.

### Key Specifications

| Specification    | Details                                                                                 |
| ---------------- | --------------------------------------------------------------------------------------- |
| AI Performance   | Up to 1 PFLOP (FP4)                                                                     |
| GPU              | NVIDIA Blackwell Architecture with 5th Generation Tensor Cores, 4th Generation RT Cores |
| CPU              | 20-core Arm processor (10 Cortex-X925 + 10 Cortex-A725)                                 |
| Memory           | 128 GB LPDDR5x unified system memory, 256-bit interface, 4266 MHz, 273 GB/s bandwidth   |
| Storage          | 1 TB or 4 TB NVMe M.2 with self-encryption                                              |
| Network          | 1x RJ-45 (10 GbE), ConnectX-7 Smart NIC, Wi-Fi 7, Bluetooth 5.4                         |
| Connectivity     | 4x USB Type-C, 1x HDMI 2.1a, HDMI multichannel audio                                    |
| Video Processing | 1x NVENC, 1x NVDEC                                                                      |

### DGX OS

[NVIDIA DGX OS](https://docs.nvidia.com/dgx/dgx-os-7-user-guide/introduction.html) is a customized Linux distribution that provides a stable, tested, and supported operating system foundation for running AI, machine learning, and analytics applications on DGX systems. It includes:

- A robust Linux foundation optimized for AI workloads
- Pre-configured drivers and system settings for NVIDIA hardware
- Security updates and system maintenance capabilities
- Compatibility with the broader NVIDIA software ecosystem

DGX OS follows a regular release schedule with updates typically provided twice per year (around February and August), with additional security patches provided between major releases.

### DGX Dashboard

DGX Spark comes with a built-in [DGX Dashboard](https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html) that provides:

- **Real-time System Monitoring**: Overview of the system's current operational metrics
- **System Updates**: Ability to apply updates directly from the dashboard
- **System Settings**: Change device name and other configurations
- **Integrated JupyterLab**: Access local Jupyter Notebooks for development

<p align="center">
  <img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/nvidia-dgx-dashboard.avif" alt="NVIDIA DGX management dashboard interface">
</p>

#### Accessing the Dashboard

=== "Locally"

    Click the "Show Apps" button in the bottom left corner of the Ubuntu desktop, then select "DGX Dashboard" to open it in your browser.

=== "Remotely via SSH"

    ```bash
    # Open an SSH tunnel
    ssh -L 11000:localhost:11000 username@spark-abcd.local

    # Then open in browser
    # http://localhost:11000
    ```

=== "Remotely via NVIDIA Sync"

    After connecting with NVIDIA Sync, click the "DGX Dashboard" button to open the dashboard at `http://localhost:11000`.

!!! tip "Integrated JupyterLab"

    The dashboard includes an integrated JupyterLab instance that automatically creates a virtual environment and installs recommended packages when started. Each user account is assigned a dedicated port for JupyterLab access.

## Quick Start with Docker

The fastest way to get started with Ultralytics YOLO26 on NVIDIA DGX Spark is to run with pre-built docker images. The same Docker image that supports Jetson AGX Thor (JetPack 7.0) works on DGX Spark with DGX OS.

```bash
t=ultralytics/ultralytics:latest-nvidia-arm64
sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t
```

After this is done, skip to [Use TensorRT on NVIDIA DGX Spark section](#use-tensorrt-on-nvidia-dgx-spark).

## Start with Native Installation

For a native installation without Docker, follow these steps.

### Install Ultralytics Package

Here we will install Ultralytics package on DGX Spark with optional dependencies so that we can export the [PyTorch](https://www.ultralytics.com/glossary/pytorch) models to other different formats. We will mainly focus on [NVIDIA TensorRT exports](../integrations/tensorrt.md) because TensorRT will make sure we can get the maximum performance out of the DGX Spark.

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

### Install PyTorch and Torchvision

The above ultralytics installation will install Torch and Torchvision. However, these packages installed via pip may not be fully optimized for the DGX Spark's ARM64 architecture with CUDA 13. Therefore, we recommend installing the CUDA 13 compatible versions:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

!!! info

    When running PyTorch 2.9.1 on NVIDIA DGX Spark, you may encounter the following `UserWarning` when initializing CUDA (e.g. running `yolo checks`, `yolo predict`, etc.):

    ```text
    UserWarning: Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
    Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
    ```

    This warning can be safely ignored. To address this permanently, a fix has been submitted in PyTorch PR [#164590](https://github.com/pytorch/pytorch/pull/164590) which will be included in the PyTorch 2.10 release.

### Install `onnxruntime-gpu`

The [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) package hosted in PyPI does not have `aarch64` binaries for ARM64 systems. So we need to manually install this package. This package is needed for some of the exports.

Here we will download and install `onnxruntime-gpu 1.24.0` with `Python3.12` support.

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl
```

## Use TensorRT on NVIDIA DGX Spark

Among all the model export formats supported by Ultralytics, TensorRT offers the highest inference performance on NVIDIA DGX Spark, making it our top recommendation for deployments. For setup instructions and advanced usage, see our [dedicated TensorRT integration guide](../integrations/tensorrt.md).

### Convert Model to TensorRT and Run Inference

The YOLO26n model in PyTorch format is converted to TensorRT to run inference with the exported model.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26n PyTorch model
        model = YOLO("yolo26n.pt")

        # Export the model to TensorRT
        model.export(format="engine")  # creates 'yolo26n.engine'

        # Load the exported TensorRT model
        trt_model = YOLO("yolo26n.engine")

        # Run inference
        results = trt_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to TensorRT format
        yolo export model=yolo26n.pt format=engine # creates 'yolo26n.engine'

        # Run inference with the exported model
        yolo predict model=yolo26n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    Visit the [Export page](../modes/export.md#arguments) to access additional arguments when exporting models to different model formats

## NVIDIA DGX Spark YOLO11 Benchmarks

YOLO11 benchmarks were run by the Ultralytics team on multiple model formats measuring speed and [accuracy](https://www.ultralytics.com/glossary/accuracy): PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, TF SavedModel, TF GraphDef, TF Lite, MNN, NCNN, ExecuTorch. Benchmarks were run on NVIDIA DGX Spark at FP32 [precision](https://www.ultralytics.com/glossary/precision) with default input image size of 640.

### Detailed Comparison Table

The below table represents the benchmark results for five different models (YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x) across multiple formats, giving us the status, size, mAP50-95(B) metric, and inference time for each combination.

!!! tip "Performance"

    === "YOLO11n"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.5071      | 2.67                   |
        | TorchScript     | ✅      | 10.5              | 0.5083      | 2.62                   |
        | ONNX            | ✅      | 10.2              | 0.5074      | 5.92                   |
        | OpenVINO        | ✅      | 10.4              | 0.5058      | 14.95                  |
        | TensorRT (FP32) | ✅      | 12.8              | 0.5085      | 1.95                   |
        | TensorRT (FP16) | ✅      | 7.0               | 0.5068      | 1.01                   |
        | TensorRT (INT8) | ✅      | 18.6              | 0.4880      | 1.62                   |
        | TF SavedModel   | ✅      | 25.7              | 0.5076      | 36.39                  |
        | TF GraphDef     | ✅      | 10.3              | 0.5076      | 41.06                  |
        | TF Lite         | ✅      | 10.3              | 0.5075      | 64.36                  |
        | MNN             | ✅      | 10.1              | 0.5075      | 12.14                  |
        | NCNN            | ✅      | 10.2              | 0.5041      | 12.31                  |
        | ExecuTorch      | ✅      | 10.2              | 0.5075      | 27.61                  |

    === "YOLO11s"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.5767      | 5.38                   |
        | TorchScript     | ✅      | 36.5              | 0.5781      | 5.48                   |
        | ONNX            | ✅      | 36.3              | 0.5784      | 8.17                   |
        | OpenVINO        | ✅      | 36.4              | 0.5809      | 27.12                  |
        | TensorRT (FP32) | ✅      | 39.8              | 0.5783      | 3.59                   |
        | TensorRT (FP16) | ✅      | 20.1              | 0.5800      | 1.85                   |
        | TensorRT (INT8) | ✅      | 17.5              | 0.5664      | 1.88                   |
        | TF SavedModel   | ✅      | 90.8              | 0.5782      | 66.63                  |
        | TF GraphDef     | ✅      | 36.3              | 0.5782      | 71.67                  |
        | TF Lite         | ✅      | 36.3              | 0.5782      | 187.36                 |
        | MNN             | ✅      | 36.2              | 0.5775      | 27.05                  |
        | NCNN            | ✅      | 36.2              | 0.5806      | 26.26                  |
        | ExecuTorch      | ✅      | 36.2              | 0.5782      | 54.73                  |

    === "YOLO11m"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.6254      | 11.14                  |
        | TorchScript     | ✅      | 77.3              | 0.6304      | 12.00                  |
        | ONNX            | ✅      | 76.9              | 0.6304      | 13.83                  |
        | OpenVINO        | ✅      | 77.1              | 0.6284      | 62.44                  |
        | TensorRT (FP32) | ✅      | 79.9              | 0.6305      | 6.96                   |
        | TensorRT (FP16) | ✅      | 40.6              | 0.6313      | 3.14                   |
        | TensorRT (INT8) | ✅      | 26.6              | 0.6204      | 3.30                   |
        | TF SavedModel   | ✅      | 192.4             | 0.6306      | 139.85                 |
        | TF GraphDef     | ✅      | 76.9              | 0.6306      | 146.76                 |
        | TF Lite         | ✅      | 76.9              | 0.6306      | 568.18                 |
        | MNN             | ✅      | 76.8              | 0.6306      | 67.67                  |
        | NCNN            | ✅      | 76.8              | 0.6308      | 60.49                  |
        | ExecuTorch      | ✅      | 76.9              | 0.6306      | 120.37                 |

    === "YOLO11l"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.6366      | 13.95                  |
        | TorchScript     | ✅      | 97.6              | 0.6399      | 15.67                  |
        | ONNX            | ✅      | 97.0              | 0.6399      | 16.62                  |
        | OpenVINO        | ✅      | 97.3              | 0.6377      | 78.80                  |
        | TensorRT (FP32) | ✅      | 99.2              | 0.6407      | 8.86                   |
        | TensorRT (FP16) | ✅      | 50.8              | 0.6350      | 3.85                   |
        | TensorRT (INT8) | ✅      | 32.5              | 0.6224      | 4.52                   |
        | TF SavedModel   | ✅      | 242.7             | 0.6409      | 187.45                 |
        | TF GraphDef     | ✅      | 97.0              | 0.6409      | 193.92                 |
        | TF Lite         | ✅      | 97.0              | 0.6409      | 728.61                 |
        | MNN             | ✅      | 96.9              | 0.6369      | 85.21                  |
        | NCNN            | ✅      | 96.9              | 0.6373      | 77.62                  |
        | ExecuTorch      | ✅      | 97.0              | 0.6409      | 153.56                 |

    === "YOLO11x"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.6992      | 23.19                  |
        | TorchScript     | ✅      | 218.1             | 0.6900      | 25.75                  |
        | ONNX            | ✅      | 217.5             | 0.6900      | 27.43                  |
        | OpenVINO        | ✅      | 217.8             | 0.6872      | 149.44                 |
        | TensorRT (FP32) | ✅      | 222.7             | 0.6902      | 13.87                  |
        | TensorRT (FP16) | ✅      | 111.1             | 0.6883      | 6.19                   |
        | TensorRT (INT8) | ✅      | 62.9              | 0.6793      | 6.62                   |
        | TF SavedModel   | ✅      | 543.9             | 0.6900      | 335.10                 |
        | TF GraphDef     | ✅      | 217.5             | 0.6900      | 348.86                 |
        | TF Lite         | ✅      | 217.5             | 0.6900      | 1578.66                |
        | MNN             | ✅      | 217.3             | 0.6874      | 168.95                 |
        | NCNN            | ✅      | 217.4             | 0.6901      | 132.13                 |
        | ExecuTorch      | ✅      | 217.4             | 0.6900      | 297.17                 |

    Benchmarked with Ultralytics 8.3.249

## Reproduce Our Results

To reproduce the above Ultralytics benchmarks on all export [formats](../modes/export.md) run this code:

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

## Best Practices for NVIDIA DGX Spark

When using NVIDIA DGX Spark, there are a couple of best practices to follow in order to enable maximum performance running YOLO26.

1. **Monitor System Performance**

    Use NVIDIA's monitoring tools to track GPU and CPU utilization:

    ```bash
    nvidia-smi
    ```

2. **Optimize Memory Usage**

    With 128GB of unified memory, DGX Spark can handle large batch sizes and models. Consider increasing batch size for improved throughput:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.engine")
    results = model.predict(source="path/to/images", batch=16)
    ```

3. **Use TensorRT with FP16 or INT8**

    For best performance, export models with FP16 or INT8 precision:

    ```bash
    yolo export model=yolo26n.pt format=engine half=True # FP16
    yolo export model=yolo26n.pt format=engine int8=True # INT8
    ```

## System Updates (Founders Edition)

Keeping your DGX Spark Founders Edition up to date is crucial for performance and security. NVIDIA provides two primary methods for updating the system OS, drivers, and firmware.

### Using DGX Dashboard (Recommended)

The [DGX Dashboard](https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html) is the recommended way to perform system updates ensuring compatibility. It allows you to:

- View available system updates
- Install security patches and system updates
- Manage NVIDIA driver and firmware updates

### Manual System Updates

For advanced users, updates can be performed manually via terminal:

```bash
sudo apt update
sudo apt dist-upgrade
sudo fwupdmgr refresh
sudo fwupdmgr upgrade
sudo reboot
```

!!! warning

    Ensure your system is connected to a stable power source and you have backed up critical data before performing updates.

## Next Steps

For further learning and support, see the [Ultralytics YOLO26 Docs](../index.md).

## FAQ

### How do I deploy Ultralytics YOLO26 on NVIDIA DGX Spark?

Deploying Ultralytics YOLO26 on NVIDIA DGX Spark is straightforward. You can use the pre-built Docker image for quick setup or manually install the required packages. Detailed steps for each approach can be found in sections [Quick Start with Docker](#quick-start-with-docker) and [Start with Native Installation](#start-with-native-installation).

### What performance can I expect from YOLO26 on NVIDIA DGX Spark?

YOLO26 models deliver excellent performance on DGX Spark thanks to the GB10 Grace Blackwell Superchip. The TensorRT format provides the best inference performance. Check the [Detailed Comparison Table](#detailed-comparison-table) section for specific benchmark results across different model sizes and formats.

### Why should I use TensorRT for YOLO26 on DGX Spark?

TensorRT is highly recommended for deploying YOLO26 models on DGX Spark due to its optimal performance. It accelerates inference by leveraging the Blackwell GPU capabilities, ensuring maximum efficiency and speed. Learn more in the [Use TensorRT on NVIDIA DGX Spark](#use-tensorrt-on-nvidia-dgx-spark) section.

### How does DGX Spark compare to Jetson devices for YOLO26?

DGX Spark offers significantly more compute power than Jetson devices with up to 1 PFLOP of AI performance and 128GB unified memory, compared to Jetson AGX Thor's 2070 TFLOPS and 128GB memory. DGX Spark is designed as a desktop AI supercomputer, while Jetson devices are embedded systems optimized for edge deployment.

### Can I use the same Docker image for DGX Spark and Jetson AGX Thor?

Yes! The `ultralytics/ultralytics:latest-nvidia-arm64` Docker image supports both NVIDIA DGX Spark (with DGX OS) and Jetson AGX Thor (with JetPack 7.0), as both use ARM64 architecture with CUDA 13 and similar software stacks.
