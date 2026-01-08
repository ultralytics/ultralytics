---
comments: true
description: Learn to deploy Ultralytics YOLO11 on NVIDIA DGX Spark with our detailed guide. Explore performance benchmarks and maximize AI capabilities on this compact desktop AI supercomputer.
keywords: Ultralytics, YOLO11, NVIDIA DGX Spark, AI deployment, performance benchmarks, deep learning, TensorRT, computer vision, GB10 Grace Blackwell
---

# Quick Start Guide: NVIDIA DGX Spark with Ultralytics YOLO11

This comprehensive guide provides a detailed walkthrough for deploying Ultralytics YOLO11 on [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), NVIDIA's compact desktop AI supercomputer. Additionally, it showcases performance benchmarks to demonstrate the capabilities of YOLO11 on this powerful system.

<p align="center">
  <img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/nvidia-dgx-spark.avif" alt="NVIDIA DGX Spark">
</p>

!!! note

    This guide has been tested with NVIDIA DGX Spark running DGX OS based on Ubuntu. It is expected to work with the latest DGX OS releases.

## What is NVIDIA DGX Spark?

NVIDIA DGX Spark is a compact desktop AI supercomputer powered by the NVIDIA GB10 Grace Blackwell Superchip. It delivers up to 1 petaflop of AI computing performance with FP4 precision, making it ideal for developers, researchers, and data scientists who need powerful AI capabilities in a desktop form factor.

### Key Specifications

| Specification  | Details                                              |
| -------------- | ---------------------------------------------------- |
| AI Performance | Up to 1 PFLOP (FP4)                                  |
| GPU            | NVIDIA Blackwell GPU with 1TB/s memory bandwidth     |
| CPU            | NVIDIA Grace CPU (Arm Neoverse V2 cores)             |
| Memory         | 128 GB unified LPDDR5X memory                        |
| Storage        | Up to 4TB NVMe SSD                                   |
| Connectivity   | USB4 (40Gbps), USB 3.2, 10GbE, WiFi 7, Bluetooth 5.3 |
| OS             | DGX OS (Ubuntu-based Linux)                          |
| Power          | Compact desktop design                               |

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

#### Accessing the Dashboard

=== "Locally"

    Click the "Show Apps" button in the bottom left corner of the Ubuntu desktop, then select "DGX Dashboard" to open it in your browser.

=== "Remotely via SSH"

    ```bash
    # Open an SSH tunnel
    ssh -L 11000:localhost:11000 <username>@<IP or spark-abcd.local>

    # Then open in browser
    # http://localhost:11000
    ```

=== "Remotely via NVIDIA Sync"

    After connecting with NVIDIA Sync, click the "DGX Dashboard" button to open the dashboard at `http://localhost:11000`.

!!! tip "Integrated JupyterLab"

    The dashboard includes an integrated JupyterLab instance that automatically creates a virtual environment and installs recommended packages when started. Each user account is assigned a dedicated port for JupyterLab access.

## Quick Start with Docker

The fastest way to get started with Ultralytics YOLO11 on NVIDIA DGX Spark is to run with pre-built docker images. The same Docker image that supports Jetson AGX Thor (JetPack 7.0) works on DGX Spark with DGX OS.

```bash
t=ultralytics/ultralytics:latest-nvidia-arm64
sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t
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

### Install `onnxruntime-gpu`

The [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) package hosted in PyPI does not have `aarch64` binaries for ARM64 systems. So we need to manually install this package. This package is needed for some of the exports.

Here we will download and install `onnxruntime-gpu 1.24.0` with `Python3.12` support.

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl
```

## Use TensorRT on NVIDIA DGX Spark

Among all the model export formats supported by Ultralytics, TensorRT offers the highest inference performance on NVIDIA DGX Spark, making it our top recommendation for deployments. For setup instructions and advanced usage, see our [dedicated TensorRT integration guide](../integrations/tensorrt.md).

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

## NVIDIA DGX Spark YOLO11 Benchmarks

YOLO11 benchmarks were run by the Ultralytics team on multiple model formats measuring speed and [accuracy](https://www.ultralytics.com/glossary/accuracy): PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, TF SavedModel, TF GraphDef, TF Lite, MNN, NCNN, ExecuTorch. Benchmarks were run on NVIDIA DGX Spark at FP32 [precision](https://www.ultralytics.com/glossary/precision) with default input image size of 640.

### Comparison Chart

Even though all model exports work on NVIDIA DGX Spark, we have only included **PyTorch, TorchScript, TensorRT** for the comparison chart below because they make use of the GPU and are guaranteed to produce the best results. All the other exports only utilize the CPU and the performance is not as good as the above three. You can find benchmarks for all exports in the section after this chart.

<figure style="text-align: center;">
    <img src="https://github.com/ultralytics/assets/releases/download/v0.0.0/dgx-spark-benchmarks-coco128.avif" alt="DGX Spark Benchmarks">
    <figcaption style="font-style: italic; color: gray;">Benchmarked with Ultralytics 8.3.x</figcaption>
</figure>

### Detailed Comparison Table

The below table represents the benchmark results for five different models (YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x) across multiple formats, giving us the status, size, mAP50-95(B) metric, and inference time for each combination.

!!! tip "Performance"

    === "YOLO11n"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 5.4               | 0.50xx      | X.X                    |
        | TorchScript     | ✅      | 10.5              | 0.50xx      | X.X                    |
        | ONNX            | ✅      | 10.2              | 0.50xx      | X.X                    |
        | OpenVINO        | ✅      | 10.4              | 0.50xx      | X.X                    |
        | TensorRT (FP32) | ✅      | 12.6              | 0.50xx      | X.X                    |
        | TensorRT (FP16) | ✅      | 7.7               | 0.50xx      | X.X                    |
        | TensorRT (INT8) | ✅      | 6.2               | 0.48xx      | X.X                    |
        | TF SavedModel   | ✅      | 25.7              | 0.50xx      | X.X                    |
        | TF GraphDef     | ✅      | 10.3              | 0.50xx      | X.X                    |
        | TF Lite         | ✅      | 10.3              | 0.50xx      | X.X                    |
        | MNN             | ✅      | 10.1              | 0.50xx      | X.X                    |
        | NCNN            | ✅      | 10.2              | 0.50xx      | X.X                    |
        | ExecuTorch      | ✅      | 10.2              | 0.50xx      | X.X                    |

    === "YOLO11s"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 18.4              | 0.57xx      | X.X                    |
        | TorchScript     | ✅      | 36.6              | 0.57xx      | X.X                    |
        | ONNX            | ✅      | 36.3              | 0.57xx      | X.X                    |
        | OpenVINO        | ✅      | 36.4              | 0.58xx      | X.X                    |
        | TensorRT (FP32) | ✅      | 40.1              | 0.57xx      | X.X                    |
        | TensorRT (FP16) | ✅      | 20.8              | 0.57xx      | X.X                    |
        | TensorRT (INT8) | ✅      | 12.7              | 0.55xx      | X.X                    |
        | TF SavedModel   | ✅      | 90.8              | 0.57xx      | X.X                    |
        | TF GraphDef     | ✅      | 36.3              | 0.57xx      | X.X                    |
        | TF Lite         | ✅      | 36.3              | 0.57xx      | X.X                    |
        | MNN             | ✅      | 36.2              | 0.57xx      | X.X                    |
        | NCNN            | ✅      | 36.3              | 0.58xx      | X.X                    |
        | ExecuTorch      | ✅      | 36.2              | 0.57xx      | X.X                    |

    === "YOLO11m"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 38.8              | 0.62xx      | X.X                    |
        | TorchScript     | ✅      | 77.3              | 0.63xx      | X.X                    |
        | ONNX            | ✅      | 76.9              | 0.63xx      | X.X                    |
        | OpenVINO        | ✅      | 77.1              | 0.62xx      | X.X                    |
        | TensorRT (FP32) | ✅      | 80.7              | 0.63xx      | X.X                    |
        | TensorRT (FP16) | ✅      | 41.3              | 0.62xx      | X.X                    |
        | TensorRT (INT8) | ✅      | 23.7              | 0.61xx      | X.X                    |
        | TF SavedModel   | ✅      | 192.4             | 0.63xx      | X.X                    |
        | TF GraphDef     | ✅      | 76.9              | 0.63xx      | X.X                    |
        | TF Lite         | ✅      | 76.9              | 0.63xx      | X.X                    |
        | MNN             | ✅      | 76.8              | 0.62xx      | X.X                    |
        | NCNN            | ✅      | 76.9              | 0.63xx      | X.X                    |
        | ExecuTorch      | ✅      | 76.9              | 0.63xx      | X.X                    |

    === "YOLO11l"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 49.0              | 0.63xx      | X.X                    |
        | TorchScript     | ✅      | 97.6              | 0.64xx      | X.X                    |
        | ONNX            | ✅      | 97.0              | 0.64xx      | X.X                    |
        | OpenVINO        | ✅      | 97.3              | 0.63xx      | X.X                    |
        | TensorRT (FP32) | ✅      | 101.0             | 0.63xx      | X.X                    |
        | TensorRT (FP16) | ✅      | 51.5              | 0.63xx      | X.X                    |
        | TensorRT (INT8) | ✅      | 29.7              | 0.61xx      | X.X                    |
        | TF SavedModel   | ✅      | 242.7             | 0.64xx      | X.X                    |
        | TF GraphDef     | ✅      | 97.0              | 0.64xx      | X.X                    |
        | TF Lite         | ✅      | 97.0              | 0.64xx      | X.X                    |
        | MNN             | ✅      | 96.9              | 0.63xx      | X.X                    |
        | NCNN            | ✅      | 97.0              | 0.63xx      | X.X                    |
        | ExecuTorch      | ✅      | 97.0              | 0.64xx      | X.X                    |

    === "YOLO11x"

        | Format          | Status | Size on disk (MB) | mAP50-95(B) | Inference time (ms/im) |
        |-----------------|--------|-------------------|-------------|------------------------|
        | PyTorch         | ✅      | 109.3             | 0.69xx      | X.X                    |
        | TorchScript     | ✅      | 218.1             | 0.69xx      | X.X                    |
        | ONNX            | ✅      | 217.5             | 0.69xx      | X.X                    |
        | OpenVINO        | ✅      | 217.8             | 0.68xx      | X.X                    |
        | TensorRT (FP32) | ✅      | 220.0             | 0.69xx      | X.X                    |
        | TensorRT (FP16) | ✅      | 114.6             | 0.68xx      | X.X                    |
        | TensorRT (INT8) | ✅      | 59.9              | 0.68xx      | X.X                    |
        | TF SavedModel   | ✅      | 543.9             | 0.69xx      | X.X                    |
        | TF GraphDef     | ✅      | 217.5             | 0.69xx      | X.X                    |
        | TF Lite         | ✅      | 217.5             | 0.69xx      | X.X                    |
        | MNN             | ✅      | 217.3             | 0.69xx      | X.X                    |
        | NCNN            | ✅      | 217.5             | 0.69xx      | X.X                    |
        | ExecuTorch      | ✅      | 217.4             | 0.69xx      | X.X                    |

    Benchmarked with Ultralytics 8.3.x

    !!! note

        Inference time does not include pre/post-processing. Replace X.X values with your actual benchmark results.

## Reproduce Our Results

To reproduce the above Ultralytics benchmarks on all export [formats](../modes/export.md) run this code:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all export formats
        results = model.benchmark(data="coco128.yaml", imgsz=640)
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all export formats
        yolo benchmark model=yolo11n.pt data=coco128.yaml imgsz=640
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results, use a dataset with a large number of images, e.g., `data='coco.yaml'` (5000 val images).

## Best Practices for NVIDIA DGX Spark

When using NVIDIA DGX Spark, there are a couple of best practices to follow in order to enable maximum performance running YOLO11.

1. **Monitor System Performance**

    Use NVIDIA's monitoring tools to track GPU and CPU utilization:

    ```bash
    nvidia-smi
    ```

2. **Optimize Memory Usage**

    With 128GB of unified memory, DGX Spark can handle large batch sizes and models. Consider increasing batch size for improved throughput:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.engine")
    results = model.predict(source="path/to/images", batch=16)
    ```

3. **Use TensorRT with FP16 or INT8**

    For best performance, export models with FP16 or INT8 precision:

    ```bash
    yolo export model=yolo11n.pt format=engine half=True # FP16
    yolo export model=yolo11n.pt format=engine int8=True # INT8
    ```

## Next Steps

For further learning and support, see the [Ultralytics YOLO11 Docs](../index.md).

## FAQ

### How do I deploy Ultralytics YOLO11 on NVIDIA DGX Spark?

Deploying Ultralytics YOLO11 on NVIDIA DGX Spark is straightforward. You can use the pre-built Docker image for quick setup or manually install the required packages. Detailed steps for each approach can be found in sections [Quick Start with Docker](#quick-start-with-docker) and [Start with Native Installation](#start-with-native-installation).

### What performance can I expect from YOLO11 on NVIDIA DGX Spark?

YOLO11 models deliver excellent performance on DGX Spark thanks to the GB10 Grace Blackwell Superchip. The TensorRT format provides the best inference performance. Check the [Detailed Comparison Table](#detailed-comparison-table) section for specific benchmark results across different model sizes and formats.

### Why should I use TensorRT for YOLO11 on DGX Spark?

TensorRT is highly recommended for deploying YOLO11 models on DGX Spark due to its optimal performance. It accelerates inference by leveraging the Blackwell GPU capabilities, ensuring maximum efficiency and speed. Learn more in the [Use TensorRT on NVIDIA DGX Spark](#use-tensorrt-on-nvidia-dgx-spark) section.

### How does DGX Spark compare to Jetson devices for YOLO11?

DGX Spark offers significantly more compute power than Jetson devices with up to 1 PFLOP of AI performance and 128GB unified memory, compared to Jetson AGX Thor's 2070 TFLOPS and 128GB memory. DGX Spark is designed as a desktop AI supercomputer, while Jetson devices are embedded systems optimized for edge deployment.

### Can I use the same Docker image for DGX Spark and Jetson AGX Thor?

Yes! The `ultralytics/ultralytics:latest-nvidia-arm64` Docker image supports both NVIDIA DGX Spark (with DGX OS) and Jetson AGX Thor (with JetPack 7.0), as both use ARM64 architecture with CUDA 13 and similar software stacks.
