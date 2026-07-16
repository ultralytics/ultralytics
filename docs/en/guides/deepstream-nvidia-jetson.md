---
title: YOLO26 on Jetson: DeepStream & TensorRT
comments: true
description: Learn how to deploy Ultralytics YOLO26 on NVIDIA Jetson devices using TensorRT and DeepStream SDK. Explore performance benchmarks and maximize AI capabilities.
keywords: Ultralytics, YOLO26, NVIDIA Jetson, JetPack, AI deployment, embedded systems, deep learning, TensorRT, DeepStream SDK, computer vision
---

# Ultralytics YOLO26 on NVIDIA Jetson using DeepStream SDK and TensorRT

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/hvGqrVT2wPg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to use Ultralytics YOLO26 models with NVIDIA Deepstream on Jetson Orin NX 🚀
</p>

This comprehensive guide provides a detailed walkthrough for deploying Ultralytics YOLO26 on [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) devices using DeepStream SDK and TensorRT. Here we use [TensorRT](../integrations/tensorrt.md) to maximize the inference performance on the Jetson platform.

This guide walks through [DeepStream configuration for YOLO26](#deepstream-configuration-for-yolo26), [INT8 calibration](#int8-calibration), [multi-stream setup](#multistream-setup), and [benchmark results](#benchmark-results).

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/deepstream-nvidia-jetson.avif" alt="NVIDIA DeepStream SDK on Jetson platform">

!!! note

    This guide has been tested with [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit) running the latest stable JetPack release of [JP6.1](https://developer.nvidia.com/embedded/jetpack-sdk-61),
    [Seeed Studio reComputer J4012](https://www.seeedstudio.com/reComputer-J4012-p-5586.html) which is based on NVIDIA Jetson Orin NX 16GB running JetPack release of [JP5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) and [Seeed Studio reComputer J1020 v2](https://www.seeedstudio.com/reComputer-J1020-v2-p-5498.html) which is based on NVIDIA Jetson Nano 4GB running JetPack release of [JP4.6.4](https://developer.nvidia.com/jetpack-sdk-464). It is expected to work across all the NVIDIA Jetson hardware lineup including latest and legacy.

## What is NVIDIA DeepStream?

[NVIDIA's DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) is a complete streaming analytics toolkit based on GStreamer for AI-based multi-sensor processing, video, audio, and image understanding. It's ideal for vision AI developers, software partners, startups, and OEMs building IVA (Intelligent Video Analytics) apps and services. You can now create stream-processing pipelines that incorporate [neural networks](https://www.ultralytics.com/glossary/neural-network-nn) and other complex processing tasks like tracking, video encoding/decoding, and video rendering. These pipelines enable real-time analytics on video, image, and sensor data. DeepStream's multi-platform support gives you a faster, easier way to develop vision AI applications and services on-premise, at the edge, and in the cloud.

## Prerequisites

Before you start to follow this guide:

- Visit our documentation, [Quick Start Guide: NVIDIA Jetson with Ultralytics YOLO26](nvidia-jetson.md) to set up your NVIDIA Jetson device with Ultralytics YOLO26
- Install [DeepStream SDK](https://developer.nvidia.com/deepstream-getting-started) according to the JetPack version
    - For JetPack 4.6.4, install [DeepStream 6.0.1](https://archive.docs.nvidia.com/metropolis/deepstream/6.0.1/dev-guide/text/DS_Quickstart.html)
    - For JetPack 5.1.3, install [DeepStream 6.3](https://archive.docs.nvidia.com/metropolis/deepstream/6.3/dev-guide/text/DS_Quickstart.html)
    - For JetPack 6.1, install [DeepStream 7.1](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Overview.html)
    - For JetPack 7.0, install [DeepStream 8.0](https://docs.nvidia.com/metropolis/deepstream/8.0/text/DS_Overview.html)
    - For JetPack 7.1, install [DeepStream 9.0](https://docs.nvidia.com/metropolis/deepstream/9.0/text/DS_Overview.html)
    - For JetPack 7.2, install [DeepStream 9.1](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_Overview.html)

!!! tip

    In this guide we have used the Debian package method of installing DeepStream SDK to the Jetson device. You can also visit the [DeepStream SDK on Jetson (Archived)](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived) to access legacy versions of DeepStream.

!!! note "DeepStream source on GitHub"

    NVIDIA hosts the DeepStream SDK source on the [NVIDIA/DeepStream](https://github.com/NVIDIA/DeepStream) GitHub monorepo (the unified source home since DeepStream 9.0), and from DeepStream 9.1 the release packages (`.deb` and `.tar.gz` archives plus Python wheels) are published as GitHub Release assets rather than through NGC only. The SDK is not fully open source: the open-source components still require proprietary NVIDIA runtime libraries, which the source build (`bash build/build.sh`) downloads as release assets. The Debian package method described below remains the recommended path.

## DeepStream Configuration for YOLO26

Here we are using [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) GitHub repository which includes NVIDIA DeepStream SDK support for YOLO models. We appreciate the efforts of marcoslucianops for his contributions!

!!! note

    The `DeepStream-Yolo` build and configuration steps below are validated by its maintainer through DeepStream 8.0. For DeepStream 9.0 and 9.1, NVIDIA also provides an official YOLO integration under [tools/yolo_deepstream](https://github.com/NVIDIA/DeepStream/tree/main/tools/yolo_deepstream) in the DeepStream monorepo. Check the target repository for the YOLO and DeepStream versions it supports before deploying.

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

3.  Copy the `export_yolo26.py` file from `DeepStream-Yolo/utils` directory to the `ultralytics` folder

    ```bash
    cp ~/DeepStream-Yolo/utils/export_yolo26.py ~/ultralytics
    cd ultralytics
    ```

4.  Download Ultralytics YOLO26 detection model (.pt) of your choice from [YOLO26 releases](https://github.com/ultralytics/assets/releases). Here we use [yolo26s.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt).

    ```bash
    wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
    ```

    !!! note

        You can also use a [custom-trained YOLO26 model](../modes/train.md).

5.  Convert model to ONNX

    ```bash
    python3 export_yolo26.py -w yolo26s.pt
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
    cp yolo26s.onnx labels.txt ~/DeepStream-Yolo
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

    For JetPack 6.1:

    ```bash
    export CUDA_VER=12.6
    ```

    For JetPack 7.0:

    ```bash
    export CUDA_VER=13.0
    ```

    For JetPack 7.2:

    ```bash
    export CUDA_VER=13.2
    ```

8.  Compile the library

    ```bash
    make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
    ```

9.  Edit the `config_infer_primary_yolo26.txt` file according to your model (for YOLO26s with 80 classes)

    ```bash
    [property]
    ...
    onnx-file=yolo26s.onnx
    ...
    num-detected-classes=80
    ...
    parse-bbox-func-name=NvDsInferParseYolo
    ...
    ```

    !!! note "YOLO26 accuracy settings"

        YOLO26 resizes the input with center padding and runs without NMS. For the best [accuracy](https://www.ultralytics.com/glossary/accuracy), add the following to the `[property]` section of `config_infer_primary_yolo26.txt`:

        ```bash
        [property]
        ...
        maintain-aspect-ratio=1
        symmetric-padding=1
        cluster-mode=4
        ...
        ```

10. Edit the `deepstream_app_config` file

    ```bash
    ...
    [primary-gie]
    ...
    config-file=config_infer_primary_yolo26.txt
    ```

11. You can also change the video source in `deepstream_app_config` file. Here, a default video file is loaded

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

<div align=center><img width=1000 src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolov8-with-deepstream.avif" alt="YOLO26 with deepstream"></div>

!!! tip

    If you want to convert the model to FP16 precision, simply set `model-engine-file=model_b1_gpu0_fp16.engine` and `network-mode=2` inside `config_infer_primary_yolo26.txt`

## INT8 Calibration

If you want to use INT8 precision for inference, you need to follow the steps below:

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

        Higher INT8_CALIB_BATCH_SIZE values will result in more accuracy and faster calibration speed. Set it according to your GPU memory.

8.  Update the `config_infer_primary_yolo26.txt` file

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

### Run INT8 Inference

Run the same command to build the INT8 engine and start inference:

```bash
deepstream-app -c deepstream_app_config.txt
```

## MultiStream Setup

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/BpSuXSUzEYY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Multi-Stream Inference with Ultralytics YOLO26 using NVIDIA DeepStream on Jetson Orin 🚀
</p>

To set up multiple streams under a single DeepStream application, make the following changes to the `deepstream_app_config.txt` file:

1. Change the rows and columns to build a grid display according to the number of streams you want to have. For example, for 4 streams, we can add 2 rows and 2 columns.

    ```bash
    [tiled-display]
    rows=2
    columns=2
    ```

2. Add a separate `[sourceN]` group for each stream, each with its own `uri` and `num-sources=1`.

    ```bash
    [source0]
    enable=1
    type=3
    uri=file:///path/to/video1.mp4
    num-sources=1

    [source1]
    enable=1
    type=3
    uri=file:///path/to/video2.mp4
    num-sources=1

    [source2]
    enable=1
    type=3
    uri=file:///path/to/video3.mp4
    num-sources=1

    [source3]
    enable=1
    type=3
    uri=file:///path/to/video4.mp4
    num-sources=1
    ```

### Run Multi-Stream Inference

Run the same command to launch all streams in the tiled display:

```bash
deepstream-app -c deepstream_app_config.txt
```

<div align=center><img width=1000 src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/multistream-setup.avif" alt="DeepStream multi-camera streaming configuration"></div>

## Benchmark Results

The following [benchmarks](../modes/benchmark.md) summarize how YOLO11 models perform at different TensorRT precision levels with an input size of 640x640 on NVIDIA Jetson Orin NX 16GB. YOLO26 uses the same DeepStream export and inference workflow described above.

### Comparison Chart

<div align=center><img width=1000 src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/jetson-deepstream-benchmarks.avif" alt="NVIDIA Jetson DeepStream performance benchmarks"></div>

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

## Acknowledgments

This guide was initially created by our friends at Seeed Studio, Lakshantha and Elaine.

## FAQ

### How do I set up Ultralytics YOLO26 on an NVIDIA Jetson device?

To set up Ultralytics YOLO26 on an [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) device, you first need to install the [DeepStream SDK](https://developer.nvidia.com/deepstream-getting-started) compatible with your JetPack version. Follow the step-by-step guide in our [Quick Start Guide](nvidia-jetson.md) to configure your NVIDIA Jetson for YOLO26 deployment.

### What is the benefit of using TensorRT with YOLO26 on NVIDIA Jetson?

Using TensorRT with YOLO26 optimizes the model for inference, significantly reducing latency and improving throughput on NVIDIA Jetson devices. TensorRT provides high-performance, low-latency [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) inference through layer fusion, precision calibration, and kernel auto-tuning. This leads to faster and more efficient execution, particularly useful for real-time applications like video analytics and autonomous machines.

### Can I run Ultralytics YOLO26 with DeepStream SDK across different NVIDIA Jetson hardware?

Yes, the guide for deploying Ultralytics YOLO26 with the DeepStream SDK and TensorRT is compatible across the entire NVIDIA Jetson lineup. This includes devices like the Jetson Orin NX 16GB with [JetPack 5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) and the Jetson Nano 4GB with [JetPack 4.6.4](https://developer.nvidia.com/jetpack-sdk-464). Refer to the section [DeepStream Configuration for YOLO26](#deepstream-configuration-for-yolo26) for detailed steps.

### How can I convert a YOLO26 model to ONNX for DeepStream?

To convert a YOLO26 model to ONNX format for deployment with DeepStream, use the `utils/export_yolo26.py` script from the [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) repository.

Here's an example command:

```bash
python3 utils/export_yolo26.py -w yolo26s.pt --opset 12 --simplify
```

For more details on model conversion, check out our [model export section](../modes/export.md).

### How do I run INT8 inference with YOLO26 on DeepStream?

To run INT8 inference, calibrate the model on a representative image set and switch the DeepStream config to INT8 mode. Download the COCO val2017 images, select around 1000 calibration images, set the `INT8_CALIB_IMG_PATH` and `INT8_CALIB_BATCH_SIZE` environment variables, then update `config_infer_primary_yolo26.txt` with `model-engine-file=model_b1_gpu0_int8.engine`, `int8-calib-file=calib.table`, and `network-mode=1`. See the [INT8 Calibration](#int8-calibration) section for the full steps. INT8 currently requires TensorRT 8.x.

### How do I run multiple camera streams with DeepStream on Jetson?

To process multiple streams in a single DeepStream application, edit the `deepstream_app_config.txt` file to add a tiled-display grid and list each source URI. Set the `rows` and `columns` under `[tiled-display]` to build the grid, add a separate `[sourceN]` group per stream with its own `uri` and `num-sources=1`, and adjust the grid to fit the number of streams. See the [MultiStream Setup](#multistream-setup) section for a complete example.

### What are the performance benchmarks for YOLO on NVIDIA Jetson Orin NX?

The performance of YOLO11 models on NVIDIA Jetson Orin NX 16GB varies based on TensorRT precision levels. For example, YOLO11s models achieve:

- **FP32 Precision**: 14.53 ms/im, 68.8 FPS
- **FP16 Precision**: 7.91 ms/im, 126 FPS
- **INT8 Precision**: 6.05 ms/im, 165 FPS

These benchmarks underscore the efficiency and capability of using TensorRT-optimized YOLO11 models on NVIDIA Jetson hardware. For further details, see our [Benchmark Results](#benchmark-results) section.
