---
title: YOLO Deployment on Texas Instruments Edge AI (TIDL)
comments: true
description: Deploy Ultralytics YOLO models on Texas Instruments MPU devices using the TI Edge AI TIDL runtime. Step-by-step guide for ONNX export, model preparation, TIDL compilation, and on-device inference.
keywords: Texas Instruments Edge AI, TI TIDL, TI MPU, TDA4x, TIDL Runner, TIDL Tools, YOLO TI deployment, edgeai-modelhub, ONNX TI, TI deep learning, edge AI deployment, on-device inference, industrial AI, embedded vision, YOLO26 TI, YOLO11 TI
---

# Texas Instruments Edge AI Deployment for Ultralytics YOLO Models

TI Edge AI MPUs enable accelerated, low-power inference of Neural Network workloads, using C7&trade; NPU. These include TI TDA4x, AM62A and other TI MPU-family AM6XA devices. For more information see [TI Edge AI developer landing page](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu)

Compiling and deploying Neural Network models on Texas Instruments MPU devices requires compiling models with the [TI Deep Learning (TIDL) Edge AI tools](https://github.com/TexasInstruments/edgeai-tidl-tools).

This page gives information on exporting [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models to ONNX and compiling them with the TI Edge AI toolchain lets you run these on TI devices.

## What is TI Edge AI?

[Texas Instruments Edge AI](https://github.com/TexasInstruments/edgeai) is TI's open-source ecosystem for deploying AI/ML models on TI MPU processors. At its core is the **TI Deep Learning (TIDL)** runtime, which compiles standard ONNX models into device-specific artifacts that execute on the on-chip C7 NPU. The platform includes two compilation toolchains:

- **[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)** — the low-level software tool for fine-grained compilation and inference; suited for advanced users and custom post-processing pipelines.
- **[edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner)** — a high-level CLI and Python API that wraps TIDL Tools with automated benchmarking and accuracy evaluation; the recommended path for most users.

The [TI Edge AI Model Hub](https://github.com/TexasInstruments/edgeai-modelhub) is a companion model repository that provides pre-validated ONNX models with per-model YAML configuration files, ready to be compiled and benchmarked on TI hardware.

For a complete overview of the TI Edge AI MPU platform — architecture, supported devices, and development resources — see the [Edge AI MPU overview](https://github.com/TexasInstruments/edgeai/tree/main/edgeai-mpu); for SDK installation and system setup, refer to the [Edge AI SDK documentation](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md).

## Why Deploy YOLO on TI Devices?

TI MPU processors combine a powerful application CPU with dedicated AI accelerators and are the platform of choice for industrial, automotive, and robotics applications:

- **Dedicated AI accelerators**: The C7 NPU delivers significantly higher throughput than CPU-only inference — important for real-time camera pipelines and always-on monitoring.
- **Industrial-grade reliability**: TI TDA4x and Jacinto-family devices are qualified for automotive (ADAS) and industrial applications, with long supply lifecycles and safety certifications.
- **Standard ONNX input**: TIDL compiles from ONNX, so the Ultralytics standard export workflow feeds directly into the TI toolchain without format conversion.
- **Broad model support**: TIDL supports detection, classification, segmentation, and pose estimation with standard post-processing provided by the TI toolchain.
- **Production ecosystem**: The full TI Edge AI stack — SDK, toolchain, model hub, and reference demos — is maintained as an integrated, production-ready suite.

## Key Features of TI Edge AI Deployment

- **ONNX-first**: Compile any Ultralytics YOLO ONNX export directly with TIDL.
- **Automated shape fixing**: The Model Hub `prepare_model.py` script resolves dynamic tensor dimensions to static shapes, a prerequisite for TIDL compilation.
- **Per-model config YAML**: Every model in the Model Hub ships with a `<model>_model_config.yaml` that specifies pre/post-processing, calibration, and hardware targets — a single file for compile and evaluate.
- **Two-line compile and evaluate**: `tidlrunner-cli compile` and `tidlrunner-cli evaluate` cover the full benchmark pipeline.
- **Multi-device support**: The same ONNX model and model-config file can target any TIDL-compatible MPU device by setting `--target_device` to values like TDA4VH, AM62A.

## Supported Models

The TI Edge AI Model Hub currently provides the following Ultralytics YOLO models validated for TI MPU deployment:

| Model  | Task             | Variants      | Input Size | mAP[.5:.95]% | License  |
| :----- | :--------------- | :------------ | :--------- | :----------- | :------- |
| YOLO26 | Object Detection | n, s, m, l, x | 640×640    | 40.9 – 57.5  | AGPL 3.0 |
| YOLO11 | Object Detection | n, s, m, l, x | 640×640    | 39.5 – 54.7  | AGPL 3.0 |
| YOLOv8 | Object Detection | n, m          | 640×640    | 37.3 – 53.9  | AGPL 3.0 |

## Workflow for compiling and infering pre-trained YOLO Models on TI Hardware

The TI Edge AI deployment workflow is a multi-step pipeline: **get the model** → **compile and evaluate with TIDL**. Detailed instructions, per-model configuration files, and download scripts are available on the [TI Edge AI HuggingFace page](https://huggingface.co/TIEdgeAI/models).

### Step 1 — Setup tidlrunner

Follow the [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) setup guide to install the CLI and configure the target device connection.

### Step 2 — Get the Model

Download the model export script (prepare_model.py) and configuration files from the [TI Edge AI HuggingFace page](https://huggingface.co/TIEdgeAI/models), or clone the [TI Edge AI Model Hub](https://github.com/TexasInstruments/edgeai-modelhub).\
Refer to the model specific `README.md` document to get more details on model export.\
Run `prepare_model.py` for your chosen variant. The script handles the full preparation pipeline — it exports the YOLO model to ONNX, fixes dynamic shapes to static shapes, and validates the graph for TIDL compilation. Full setup instructions are in the Model Hub README.

### Step 3 — Compile with TIDL Runner (on PC)

[edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) is the recommended tool. Each model ships with a `<model>_model_config.yaml`; pass it to `tidlrunner-cli` to compile on your target device.

```bash
# Compile for TI TDA4VH (run from edgeai-tidlrunner path)
cd /path/to/edgeai-tidlrunner
tidlrunner-cli compile --target_device TDA4VH --config_path /path/to/model/<model>_model_config.yaml
```

### Step 4 — Infer with TIDL Runner (on device)

Each model ships with a `<model>_model_config.yaml`; pass it to `tidlrunner-cli` to run inference on your target device.

```bash
# Run infernce (run from edgeai-tidlrunner path)
cd /path/to/edgeai-tidlrunner
tidlrunner-cli infer --target_device TDA4VH --config_path /path/to/model/<model>_model_config.yaml
```

Note: For fine-grained control over the compilation and inferene script, use [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) directly.

## Workflow for compiling and infering your own trained YOLO Models on TI Hardware

1. **Train** your model using Ultralytics [Train Mode](../modes/train.md)
2. **Export** the model to ONNX format using the Ultralytics ONNX export method.
3. **Update** the `<model>_model_config.yaml` file to use the path of the ONNX model that you just exported.
4. **Compile** on PC for your target TI device with `tidlrunner-cli compile`, passing the per-model config YAML.
5. **Infer** on your target TI device with `tidlrunner-cli infer`, passing the per-model config YAML to test that the inference is working correctly on device.
6. **Deploy** on device through the ONNX Runtime APIs with TIDL Offfload (see [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) for more details).

## Real-World Applications

YOLO models running on TI MPU hardware are well suited for a wide range of embedded and industrial vision applications:

- **Automotive ADAS**: Pedestrian detection, lane monitoring, and occupant sensing on Jacinto-family SoCs (TDAx) inside vehicles.
- **Industrial Automation**: High-speed quality inspection and defect detection on factory lines where cloud round-trips are unacceptable.
- **Smart Surveillance**: Real-time multi-camera object detection on edge gateways without central server dependency.
- **Robotics**: On-board perception for autonomous mobile robots (AMRs) and collaborative arms running on TI industrial MPUs (AM6xa).
- **IoT Vision**: Always-on scene monitoring in smart cameras, drones, and embedded systems deployed in the field.

## Summary

In this guide, you have learned how to deploy Ultralytics YOLO models on Texas Instruments MPU hardware using the TI Edge AI TIDL toolchain. The pipeline exports your model to ONNX, prepares it with static shapes using the Model Hub scripts, and compiles it with `tidlrunner-cli` targeting the C7 NPU on the TI TDA4x or other supported TI devices — producing a hardware-optimized artifact ready for on-device inference.

The combination of [Ultralytics YOLO](https://www.ultralytics.com/yolo) and the TI Edge AI platform provides a straightforward path from model training to production deployment on TI's industrial, automotive, and embedded compute hardware.

## FAQ

### How do I deploy a YOLO model on TI Edge AI hardware?

Download the model and configuration files from the [TI Edge AI HuggingFace page](https://huggingface.co/TIEdgeAI/models) or run `prepare_model.py` from the [TI Edge AI Model Hub](https://github.com/TexasInstruments/edgeai-modelhub) — the script handles ONNX export, static shape fixing, and graph validation in one go. Then follow [Step 3](#step-3--compile-with-tidl-runner) and [Step 4](#step-4--evaluate-with-tidl-runner) to compile and evaluate on your target device.

### What is the difference between edgeai-tidl-tools and edgeai-tidlrunner?

[edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) is the low-level TIDL SDK that gives full control over the compilation process, model partitioning, and quantization — suited for advanced users with custom pipelines. [edgeai-tidlrunner](https://github.com/TexasInstruments/edgeai-tidlrunner) is a high-level CLI wrapper around TIDL Tools that automates benchmarking and accuracy evaluation from a single YAML config file, and is the recommended starting point for most users.

### Do I need a TI device to compile a model?

Compilation with `tidlrunner-cli compile` produces device artifacts on the host PC. It does not require a connected or emulated TI target for compilation.

### Where can I find pre-compiled models for TI devices?

The [TI Edge AI HuggingFace page](https://huggingface.co/TIEdgeAI/models) and the [TI Edge AI Model Hub](https://github.com/TexasInstruments/edgeai-modelhub) host pre-validated ONNX models with per-model config YAMLs for YOLO26, YOLO11, and YOLOv8. Models can be downloaded and prepared with the included `prepare_model.py` script, then compiled locally for your target device.

### Which TI devices are supported?

The primary validated target is the **TI TDA4VH** MPU. Other TIDL-compatible TI MPU devices are also supported — change `--target_device` in the `tidlrunner-cli` command to match your hardware. See the [TI Edge AI ecosystem](https://github.com/TexasInstruments/edgeai) for the full list of supported devices.

---

<!-- Footer -->
<div class="footer">
<p>Maintained by Texas Instruments EdgeAI Team &nbsp;|&nbsp; Last Updated August 2026</p>
<p style="margin-top:0.4rem;">
    <a href="https://github.com/TexasInstruments/edgeai/issues">Issues</a>
    <a href="https://github.com/TexasInstruments/edgeai/discussions">Discussions</a>
    <a href="mailto:edgeai-dev@list.ti.com">Contact</a>
</p>
</div>
