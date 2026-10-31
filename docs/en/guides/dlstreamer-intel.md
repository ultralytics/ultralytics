---
title: YOLO26 on Intel DL Streamer & OpenVINO
comments: true
description: Deploy Ultralytics YOLO26 on Intel Core Ultra Series 3 with the DL Streamer Pipeline Framework and OpenVINO™ for GStreamer video analytics on CPU, GPU, and NPU.
keywords: Ultralytics, YOLO26, Intel, Intel Core Ultra, DL Streamer, OpenVINO, deep learning, inference, computer vision, GStreamer
---

# Ultralytics YOLO26 on Intel Core Ultra Series 3 using DL Streamer Pipeline Framework and OpenVINO™

This guide walks through deploying [Ultralytics YOLO26](https://www.ultralytics.com/yolo/yolo26) on Intel Core Ultra Series 3 platforms using [DL Streamer Pipeline Framework](https://github.com/open-edge-platform/dlstreamer) and [OpenVINO™ toolkit](https://docs.openvino.ai/). Here we use OpenVINO™ to maximize inference performance on Intel CPUs, integrated and discrete GPUs, and NPUs.

<img src="https://cdn.ul.run/i/442cc1f69a8581156d521e16cc671e9e.avif" alt="Intel DL Streamer Pipeline Framework with Ultralytics YOLO26">

**Contents:** [What is Intel DL Streamer?](#what-is-intel-dl-streamer) • [Prerequisites](#prerequisites) • [YOLO26 Model Preparation](#yolo26-model-preparation) • [Running Inference with YOLO26](#running-inference-with-yolo26) • [Multi-Stream Setup](#multi-stream-setup) • [FAQ](#faq)

## What is Intel DL Streamer?

[Deep Learning Streamer (DL Streamer) Pipeline Framework](https://github.com/open-edge-platform/dlstreamer) is an open-source streaming media analytics framework based on the [GStreamer](https://github.com/GStreamer/gstreamer) multimedia framework, designed for creating complex media analytics pipelines in the cloud or at the [edge](https://www.ultralytics.com/glossary/edge-ai).

DL Streamer enables analysis of audio and video streams to detect, classify, track, identify, and count objects, events, and people. It is optimized for Intel hardware and provides interoperability between GStreamer plugins built on various backend libraries:

- **Inference**: [OpenVINO™](https://docs.openvino.ai/) inference engine, optimized for Intel CPU, GPU, and NPU
- **Video Encoding/Decoding**: GPU-acceleration via VA-API
- **Image Processing**: GPU-acceleration via VA-API
- **Metadata**: GStreamer Analytics for structured inference results
- **Ecosystem**: Hundreds of GStreamer plugins for media I/O, muxing/demuxing, codec support, and more

DL Streamer supports many AI models including the full Ultralytics YOLO family (YOLOv5 through YOLO26) all in OpenVINO™ format.

DL Streamer is regularly validated on the systems listed in the [DL Streamer system requirements](https://docs.openedgeplatform.intel.com/dev/edge-ai-libraries/dlstreamer/system_requirements.html).

## Prerequisites

Before you begin, ensure the following are installed and configured on your Intel system:

- **Ubuntu 24.04** with [Intel GPU/NPU drivers](https://docs.openedgeplatform.intel.com/dev/edge-ai-libraries/dlstreamer/install/install_guide_ubuntu.html) installed

- **[Docker Engine](https://docs.docker.com/engine/install/ubuntu/)**

## YOLO26 Model Preparation

DL Streamer uses models in [OpenVINO™ IR format](https://docs.openvino.ai/2026/documentation/openvino-ir-format.html). Ultralytics YOLO26 models are exported from PyTorch to OpenVINO™ IR using the Ultralytics exporter. DL Streamer uses the official [Ultralytics OpenVINO™ integration](../integrations/openvino.md), which provides optimized export and inference capabilities for Intel hardware.

1.  Create the `~/intel/dlstreamer_demo` folder and install OpenVINO™ and Ultralytics in a virtual environment

```bash
mkdir -p ~/intel/dlstreamer_demo && cd ~/intel/dlstreamer_demo
python3 -m venv .dls-venv && source .dls-venv/bin/activate
pip install openvino==2026.2.0 ultralytics==8.4.92
```

2.  Download the PyTorch [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26/yolo26s) model from Ultralytics, convert it to OpenVINO™ IR format, and generate an INT8 precision variant.

```bash
yolo export model=yolo26s.pt format=openvino dynamic=True quantize=8 data=coco128.yaml
```

The model is exported to the `~/intel/dlstreamer_demo/yolo26s_int8_openvino_model` folder.

### Model Precision

DL Streamer supports FP32, FP16, and INT8 precision models. Each requires a separate export step:

```bash
# FP32 (default precision, highest accuracy)
yolo export model=yolo26s.pt format=openvino dynamic=True

# FP16 (recommended for GPU inference, good accuracy/performance balance)
yolo export model=yolo26s.pt format=openvino dynamic=True quantize=16

# INT8 (maximum performance, requires calibration dataset)
yolo export model=yolo26s.pt format=openvino dynamic=True quantize=8 data=coco128.yaml
```

The examples below use the INT8 model exported in the [preparation step](#yolo26-model-preparation). To use FP32 or FP16 models, replace the model path accordingly (e.g. `yolo26s_openvino_model/yolo26s.xml`).

## Running Inference with YOLO26

Before running the DL Streamer YOLO26 detection pipeline, complete the [YOLO26 Model Preparation](#yolo26-model-preparation) step and download the following sample video from Pexels.

```bash
curl -L https://videos.pexels.com/video-files/1192116/1192116-sd_640_360_30fps.mp4 --output ~/intel/dlstreamer_demo/video1.mp4
```

Then run the DL Streamer Docker image in interactive mode.

!!! note

    This command is intended for systems equipped with Intel Core Ultra Series 3 processors with integrated GPU (iGPU) and NPU onboard.

```bash
docker run -it --rm \
  -v ~/intel/dlstreamer_demo:/home/dlstreamer/demo \
  -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
  -e DISPLAY=$DISPLAY \
  -e XDG_RUNTIME_DIR=/tmp \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/dri \
  --group-add $(stat -c "%g" /dev/dri/render*) \
  --device /dev/accel \
  --group-add $(stat -c "%g" /dev/accel/accel*) \
  -e ZE_ENABLE_ALT_DRIVERS=libze_intel_npu.so \
  intel/dlstreamer:latest
```

### INT8 Precision (Maximum Performance)

INT8 [quantization](https://www.ultralytics.com/glossary/model-quantization) delivers the highest throughput by reducing model weights to 8-bit integers. The Ultralytics exporter handles calibration automatically.

### Run YOLO26s with INT8 on GPU

```bash
gst-launch-1.0 filesrc location=/home/dlstreamer/demo/video1.mp4 ! decodebin3 ! gvadetect model=/home/dlstreamer/demo/yolo26s_int8_openvino_model/yolo26s.xml device=GPU ! queue ! vapostproc ! gvawatermark ! gvafpscounter ! autovideosink sync=false
```

<img src="https://cdn.ul.run/i/5bbfdc15b1ea8aa2ee9879db25573123.avif" alt="YOLO26s INT8 detections in a DL Streamer pipeline on an Intel GPU">

### Run YOLO26s with INT8 on GPU and save the output to a video file

```bash
gst-launch-1.0 filesrc location=/home/dlstreamer/demo/video1.mp4 ! decodebin3 ! gvadetect model=/home/dlstreamer/demo/yolo26s_int8_openvino_model/yolo26s.xml device=GPU ! queue ! vapostproc ! gvawatermark ! gvafpscounter ! vah264enc ! h264parse ! mp4mux ! filesink location=/home/dlstreamer/demo/yolo_video1_yolo26s_INT8_GPU.mp4
```

### Run YOLO26s with INT8 on NPU

```bash
gst-launch-1.0 filesrc location=/home/dlstreamer/demo/video1.mp4 ! decodebin3 ! gvadetect model=/home/dlstreamer/demo/yolo26s_int8_openvino_model/yolo26s.xml device=NPU ! queue ! vapostproc ! gvawatermark ! gvafpscounter ! autovideosink sync=false
```

<img src="https://cdn.ul.run/i/79c5839b6ff903dce0feb2aeb257a423.avif" alt="YOLO26s INT8 detections in a DL Streamer pipeline on an Intel NPU">

## Multi-Stream Setup

DL Streamer supports multi-stream processing, where multiple video sources are decoded and inferred simultaneously. You can launch multiple pipelines in parallel using GStreamer's `vacompositor` element to combine multiple streams.

### Running Multiple Pipelines (4 streams) in Parallel (GPU)

```bash
gst-launch-1.0 vacompositor name=comp sink_0::xpos=0 sink_0::ypos=0 sink_1::xpos=660 sink_1::ypos=0 sink_2::xpos=0 sink_2::ypos=380 sink_3::xpos=660 sink_3::ypos=380 ! autovideosink sync=false \
  filesrc location=/home/dlstreamer/demo/video1.mp4 ! decodebin3 ! gvadetect model=/home/dlstreamer/demo/yolo26s_int8_openvino_model/yolo26s.xml device=GPU model-instance-id=inf0 scheduling-policy="latency" ! queue ! gvawatermark ! gvafpscounter ! comp.sink_0 \
  filesrc location=/home/dlstreamer/demo/video1.mp4 ! decodebin3 ! gvadetect model=/home/dlstreamer/demo/yolo26s_int8_openvino_model/yolo26s.xml device=GPU model-instance-id=inf0 scheduling-policy="latency" ! queue ! gvawatermark ! gvafpscounter ! comp.sink_1 \
  filesrc location=/home/dlstreamer/demo/video1.mp4 ! decodebin3 ! gvadetect model=/home/dlstreamer/demo/yolo26s_int8_openvino_model/yolo26s.xml device=GPU model-instance-id=inf0 scheduling-policy="latency" ! queue ! gvawatermark ! gvafpscounter ! comp.sink_2 \
  filesrc location=/home/dlstreamer/demo/video1.mp4 ! decodebin3 ! gvadetect model=/home/dlstreamer/demo/yolo26s_int8_openvino_model/yolo26s.xml device=GPU model-instance-id=inf0 scheduling-policy="latency" ! queue ! gvawatermark ! gvafpscounter ! comp.sink_3
```

<img src="https://cdn.ul.run/i/0caade37bce29c78fda38d4730dde046.avif" alt="Four-stream DL Streamer YOLO26 detection grid on an Intel GPU">

## Additional Resources

- [DL Streamer GitHub Repository](https://github.com/open-edge-platform/dlstreamer)
- [DL Streamer Documentation](https://docs.openedgeplatform.intel.com/dev/edge-ai-libraries/dlstreamer/)
- [DL Streamer Elements](https://docs.openedgeplatform.intel.com/dev/edge-ai-libraries/dlstreamer/elements/elements.html)
- [OpenVINO™ Toolkit](https://docs.openvino.ai/)
- [Ultralytics YOLO26](https://www.ultralytics.com/yolo/yolo26)
- [GStreamer Framework](https://github.com/GStreamer/gstreamer)
- [Supported Models Table](https://docs.openedgeplatform.intel.com/dev/edge-ai-libraries/dlstreamer/supported_models.html)

## FAQ

### How do I set up Ultralytics YOLO26 on an Intel platform with DL Streamer?

Install DL Streamer following the [Installation Guide](https://github.com/open-edge-platform/dlstreamer/blob/main/docs/user-guide/install/install_guide_ubuntu.md), set up the environment with `source /opt/intel/dlstreamer/scripts/setup_dls_env.sh`, install Ultralytics and OpenVINO™, download models using `download_ultralytics_models.sh`. Then run inference with the DL Streamer [sample application](https://github.com/open-edge-platform/dlstreamer/tree/main/samples/gstreamer/gst_launch/detection_with_yolo) `yolo_detect.sh` that provides a ready-to-use script for running YOLO26 inference pipelines.

### What is the benefit of using OpenVINO™ with YOLO26 on Intel hardware?

OpenVINO™ optimizes the YOLO26 model specifically for Intel hardware through techniques such as graph optimization, layer fusion, and hardware-specific kernel tuning. Combined with DL Streamer's VA-API accelerated decode and zero-copy `va-surface-sharing` pre-processing, the full video analytics pipeline achieves significantly higher throughput than unoptimized frameworks.

### Can I run YOLO26 with DL Streamer on different Intel devices?

Yes. DL Streamer supports inference on Intel CPUs (Core, Core Ultra, Xeon), integrated GPUs (Iris Xe, Arc), discrete GPUs (Arc A-Series, B-Series), and NPUs (AI Boost) across multiple Intel platform generations. Change the `device` property of `gvadetect` to `CPU`, `GPU`, or `NPU`.

### How do I choose between FP16 and INT8 precision?

- **FP16** is recommended as the default for GPU inference — it provides near-FP32 accuracy with approximately 2x throughput improvement.
- **INT8** delivers the highest performance (2–3x over FP32) with a small accuracy trade-off and is ideal when maximum throughput is the priority. INT8 models are automatically calibrated during Ultralytics export.

### What YOLO26 tasks are supported?

DL Streamer supports these YOLO26 task variants:

- **Detection**: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x
- **Instance Segmentation**: yolo26s-seg (and all size variants)
- **Classification**: yolo26s-cls (composite pipeline with detection)
- **Pose Estimation**: yolo26s-pose (and all size variants)
- **Oriented Bounding Box (OBB)**: yolo26s-obb (and all size variants)

### How can I export detections as structured data?

Use the `json` output option to write detection results as JSON-lines to a file:

```bash
./yolo_detect.sh yolo26s GPU input_video.mp4 json va-surface-sharing INT8
```

Alternatively, use the `gvametapublish` element in custom pipelines to publish metadata to files, MQTT, or Kafka.
