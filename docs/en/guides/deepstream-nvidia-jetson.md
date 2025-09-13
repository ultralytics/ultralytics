---
comments: true
description: Learn how to deploy Ultralytics YOLO11 on NVIDIA Jetson devices using TensorRT and DeepStream SDK. Explore performance benchmarks and maximize AI capabilities.
keywords: Ultralytics, YOLO11, NVIDIA Jetson, JetPack, AI deployment, embedded systems, deep learning, TensorRT, DeepStream SDK, computer vision
---

# Ultralytics YOLO11 on NVIDIA Jetson using DeepStream SDK and TensorRT

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/hvGqrVT2wPg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to use Ultralytics YOLO11 models with NVIDIA Deepstream on Jetson Orin NX 🚀
</p>

This comprehensive guide provides a detailed walkthrough for deploying Ultralytics YOLO11 on [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) devices using DeepStream SDK and TensorRT. Here we use TensorRT to maximize the inference performance on the Jetson platform.

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/deepstream-nvidia-jetson.avif" alt="DeepStream on NVIDIA Jetson">

!!! note

    This guide has been tested with [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit) running the latest stable JetPack release of [JP6.1](https://developer.nvidia.com/embedded/jetpack-sdk-61),
    [Seeed Studio reComputer J4012](https://www.seeedstudio.com/reComputer-J4012-p-5586.html) which is based on NVIDIA Jetson Orin NX 16GB running JetPack release of [JP5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) and [Seeed Studio reComputer J1020 v2](https://www.seeedstudio.com/reComputer-J1020-v2-p-5498.html) which is based on NVIDIA Jetson Nano 4GB running JetPack release of [JP4.6.4](https://developer.nvidia.com/jetpack-sdk-464). It is expected to work across all the NVIDIA Jetson hardware lineup including latest and legacy.

## What is NVIDIA DeepStream?

[NVIDIA's DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) is a complete streaming analytics toolkit based on GStreamer for AI-based multi-sensor processing, video, audio, and image understanding. It's ideal for vision AI developers, software partners, startups, and OEMs building IVA (Intelligent Video Analytics) apps and services. You can now create stream-processing pipelines that incorporate [neural networks](https://www.ultralytics.com/glossary/neural-network-nn) and other complex processing tasks like tracking, video encoding/decoding, and video rendering. These pipelines enable real-time analytics on video, image, and sensor data. DeepStream's multi-platform support gives you a faster, easier way to develop vision AI applications and services on-premise, at the edge, and in the cloud.

## Prerequisites

Before you start to follow this guide:

- Visit our documentation, [Quick Start Guide: NVIDIA Jetson with Ultralytics YOLO11](nvidia-jetson.md) to set up your NVIDIA Jetson device with Ultralytics YOLO11
- Install [DeepStream SDK](https://developer.nvidia.com/deepstream-getting-started) according to the JetPack version
    - For JetPack 4.6.4, install [DeepStream 6.0.1](https://docs.nvidia.com/metropolis/deepstream/6.0.1/dev-guide/text/DS_Quickstart.html)
    - For JetPack 5.1.3, install [DeepStream 6.3](https://docs.nvidia.com/metropolis/deepstream/6.3/dev-guide/text/DS_Quickstart.html)
    - For JetPack 6.1, install [DeepStream 7.1](https://docs.nvidia.com/metropolis/deepstream/7.0/dev-guide/text/DS_Overview.html)

!!! tip

    In this guide we have used the Debian package method of installing DeepStream SDK to the Jetson device. You can also visit the [DeepStream SDK on Jetson (Archived)](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived) to access legacy versions of DeepStream.

## DeepStream Configuration for YOLO11

Here we are using [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) GitHub repository which includes NVIDIA DeepStream SDK support for YOLO models. We appreciate the efforts of marcoslucianops for his contributions!

1.  Install Ultralytics with necessary dependencies

    ```bash
    cd ~
    pip install -U pip
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics
    pip install -e ".[export]" onnxslim
    ```

2.  Clone the DeepStream-Yolo repository

    ```bash
    cd ~
    git clone https://github.com/marcoslucianops/DeepStream-Yolo
    ```

3.  Copy the `export_yolo11.py` file from `DeepStream-Yolo/utils` directory to the `ultralytics` folder

    ```bash
    cp ~/DeepStream-Yolo/utils/export_yolo11.py ~/ultralytics
    cd ultralytics
    ```

4.  Download Ultralytics YOLO11 detection model (.pt) of your choice from [YOLO11 releases](https://github.com/ultralytics/assets/releases). Here we use [yolo11s.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt).

    ```bash
    wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
    ```

    !!! note

        You can also use a [custom trained YOLO11 model](https://docs.ultralytics.com/modes/train/).

5.  Convert model to ONNX

    ```bash
    python3 export_yolo11.py -w yolo11s.pt
    ```

    !!! note "Pass the below arguments to the above command"

        For DeepStream 5.1, remove the `--dynamic` arg and use `opset` 12 or lower. The default `opset` is 17.

        ```bash
        --opset 12
        ```

        To change the inference size (default: 640)

        ```bash
        -s SIZE
        --size SIZE
        -s HEIGHT WIDTH
        --size HEIGHT WIDTH
        ```

        Example for 1280:

        ```bash
        -s 1280
        or
        -s 1280 1280
        ```

        To simplify the ONNX model (DeepStream >= 6.0)

        ```bash
        --simplify
        ```

        To use dynamic batch-size (DeepStream >= 6.1)

        ```bash
        --dynamic
        ```

        To use static batch-size (example for batch-size = 4)

        ```bash
        --batch 4
        ```

6.  Copy the generated `.onnx` model file and `labels.txt` file to the `DeepStream-Yolo` folder

    ```bash
    cp yolo11s.pt.onnx labels.txt ~/DeepStream-Yolo
    cd ~/DeepStream-Yolo
    ```

7.  Set the CUDA version according to the JetPack version installed

    For JetPack 4.6.4:

    ```bash
    export CUDA_VER=10.2
    ```

    For JetPack 5.1.3:

    ```bash
    export CUDA_VER=11.4
    ```

    For Jetpack 6.1:

    ```bash
    export CUDA_VER=12.6
    ```

8.  Compile the library

    ```bash
    make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
    ```

9.  Edit the `config_infer_primary_yolo11.txt` file according to your model (for YOLO11s with 80 classes)

    ```bash
    [property]
    ...
    onnx-file=yolo11s.pt.onnx
    ...
    num-detected-classes=80
    ...
    ```

10. Edit the `deepstream_app_config` file

    ```bash
    ...
    [primary-gie]
    ...
    config-file=config_infer_primary_yolo11.txt
    ```

11. You can also change the video source in `deepstream_app_config` file. Here a default video file is loaded

    ```bash
    ...
    [source0]
    ...
    uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
    ```

### Run Inference

```bash
deepstream-app -c deepstream_app_config.txt
```

!!! note

    It will take a long time to generate the TensorRT engine file before starting the inference. So please be patient.

<div align=center><img width=1000 src="https://github.com/ultralytics/docs/releases/download/0/yolov8-with-deepstream.avif" alt="YOLO11 with deepstream"></div>

!!! tip

    If you want to convert the model to FP16 precision, simply set `model-engine-file=model_b1_gpu0_fp16.engine` and `network-mode=2` inside `config_infer_primary_yolo11.txt`

## INT8 Calibration

If you want to use INT8 precision for inference, you need to follow the steps below

!!! note

    Currently INT8 does not work with TensorRT 10.x. This section of the guide has been tested with TensorRT 8.x which is expected to work.

1.  Set `OPENCV` environment variable

    ```bash
    export OPENCV=1
    ```

2.  Compile the library

    ```bash
    make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
    ```

3.  For COCO dataset, download the [val2017](http://images.cocodataset.org/zips/val2017.zip), extract, and move to `DeepStream-Yolo` folder

4.  Make a new directory for calibration images

    ```bash
    mkdir calibration
    ```

5.  Run the following to select 1000 random images from COCO dataset to run calibration

    ```bash
    for jpg in $(ls -1 val2017/*.jpg | sort -R | head -1000); do
      cp ${jpg} calibration/
    done
    ```

    !!! note

        NVIDIA recommends at least 500 images to get a good [accuracy](https://www.ultralytics.com/glossary/accuracy). On this example, 1000 images are chosen to get better accuracy (more images = more accuracy). You can set it from **head -1000**. For example, for 2000 images, **head -2000**. This process can take a long time.

6.  Create the `calibration.txt` file with all selected images

    ```bash
    realpath calibration/*jpg > calibration.txt
    ```

7.  Set environment variables

    ```bash
    export INT8_CALIB_IMG_PATH=calibration.txt
    export INT8_CALIB_BATCH_SIZE=1
    ```

    !!! note

        Higher INT8_CALIB_BATCH_SIZE values will result in more accuracy and faster calibration speed. Set it according to you GPU memory.

8.  Update the `config_infer_primary_yolo11.txt` file

    From

    ```bash
    ...
    model-engine-file=model_b1_gpu0_fp32.engine
    #int8-calib-file=calib.table
    ...
    network-mode=0
    ...
    ```

    To

    ```bash
    ...
    model-engine-file=model_b1_gpu0_int8.engine
    int8-calib-file=calib.table
    ...
    network-mode=1
    ...
    ```

### Run Inference

```bash
deepstream-app -c deepstream_app_config.txt
```

## MultiStream Setup

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/wWmXKIteRLA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Multiple Streams with DeepStream SDK on Jetson Nano using Ultralytics YOLO11 🎉
</p>

To set up multiple streams under a single deepstream application, you can do the following changes to the `deepstream_app_config.txt` file

1. Change the rows and columns to build a grid display according to the number of streams you want to have. For example, for 4 streams, we can add 2 rows and 2 columns.

    ```bash
    [tiled-display]
    rows=2
    columns=2
    ```

2. Set `num-sources=4` and add `uri` of all the 4 streams

    ```bash
    [source0]
    enable=1
    type=3
    uri=path/to/video1.jpg
    uri=path/to/video2.jpg
    uri=path/to/video3.jpg
    uri=path/to/video4.jpg
    num-sources=4
    ```

### Run Inference

```bash
deepstream-app -c deepstream_app_config.txt
```

<div align=center><img width=1000 src="https://github.com/ultralytics/docs/releases/download/0/multistream-setup.avif" alt="Multistream setup"></div>

## Benchmark Results

The following benchmarks summarizes how YOLO11 models perform at different TensorRT precision levels with an input size of 640x640 on NVIDIA Jetson Orin NX 16GB.

### Comparison Chart

<div align=center><img width=1000 src="https://github.com/ultralytics/assets/releases/download/v0.0.0/jetson-deepstream-benchmarks.avif" alt="Jetson DeepStream Benchmarks Chart"></div>

### Detailed Comparison Table

!!! tip "Performance"

    === "YOLO11n"

        | Format          | Status | Inference time (ms/im) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 8.64                   |
        | TensorRT (FP16) | ✅      | 5.27                   |
        | TensorRT (INT8) | ✅      | 4.54                   |

    === "YOLO11s"

        | Format          | Status | Inference time (ms/im) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 14.53                  |
        | TensorRT (FP16) | ✅      | 7.91                   |
        | TensorRT (INT8) | ✅      | 6.05                   |

    === "YOLO11m"

        | Format          | Status | Inference time (ms/im) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 32.05                  |
        | TensorRT (FP16) | ✅      | 15.55                  |
        | TensorRT (INT8) | ✅      | 10.43                  |

    === "YOLO11l"

        | Format          | Status | Inference time (ms/im) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 39.68                  |
        | TensorRT (FP16) | ✅      | 19.88                  |
        | TensorRT (INT8) | ✅      | 13.64                  |

    === "YOLO11x"

        | Format          | Status | Inference time (ms/im) |
        |-----------------|--------|------------------------|
        | TensorRT (FP32) | ✅      | 80.65                  |
        | TensorRT (FP16) | ✅      | 39.06                  |
        | TensorRT (INT8) | ✅      | 22.83                  |

## Acknowledgements

This guide was initially created by our friends at Seeed Studio, Lakshantha and Elaine.

## FAQ

### How do I set up Ultralytics YOLO11 on an NVIDIA Jetson device?

To set up Ultralytics YOLO11 on an [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) device, you first need to install the [DeepStream SDK](https://developer.nvidia.com/deepstream-getting-started) compatible with your JetPack version. Follow the step-by-step guide in our [Quick Start Guide](nvidia-jetson.md) to configure your NVIDIA Jetson for YOLO11 deployment.

### What is the benefit of using TensorRT with YOLO11 on NVIDIA Jetson?

Using TensorRT with YOLO11 optimizes the model for inference, significantly reducing latency and improving throughput on NVIDIA Jetson devices. TensorRT provides high-performance, low-latency [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) inference through layer fusion, precision calibration, and kernel auto-tuning. This leads to faster and more efficient execution, particularly useful for real-time applications like video analytics and autonomous machines.

### Can I run Ultralytics YOLO11 with DeepStream SDK across different NVIDIA Jetson hardware?

Yes, the guide for deploying Ultralytics YOLO11 with the DeepStream SDK and TensorRT is compatible across the entire NVIDIA Jetson lineup. This includes devices like the Jetson Orin NX 16GB with [JetPack 5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) and the Jetson Nano 4GB with [JetPack 4.6.4](https://developer.nvidia.com/jetpack-sdk-464). Refer to the section [DeepStream Configuration for YOLO11](#deepstream-configuration-for-yolo11) for detailed steps.

### How can I convert a YOLO11 model to ONNX for DeepStream?

To convert a YOLO11 model to ONNX format for deployment with DeepStream, use the `utils/export_yolo11.py` script from the [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) repository.

Here's an example command:

```bash
python3 utils/export_yolo11.py -w yolo11s.pt --opset 12 --simplify
```

For more details on model conversion, check out our [model export section](../modes/export.md).

### What are the performance benchmarks for YOLO on NVIDIA Jetson Orin NX?

The performance of YOLO11 models on NVIDIA Jetson Orin NX 16GB varies based on TensorRT precision levels. For example, YOLO11s models achieve:

- **FP32 Precision**: 14.6 ms/im, 68.5 FPS
- **FP16 Precision**: 7.94 ms/im, 126 FPS
- **INT8 Precision**: 5.95 ms/im, 168 FPS

These benchmarks underscore the efficiency and capability of using TensorRT-optimized YOLO11 models on NVIDIA Jetson hardware. For further details, see our [Benchmark Results](#benchmark-results) section.
