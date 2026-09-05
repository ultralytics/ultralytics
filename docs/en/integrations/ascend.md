---
comments: true
description: Train Ultralytics YOLO on Huawei Ascend NPUs and export models to .om format with the CANN ATC compiler for accelerated inference.
keywords: Huawei Ascend, CANN, torch_npu, HCCL, NPU training, ATC, om model, Ascend310B, Atlas 200I, OrangePi AIPro, model export, Ultralytics, YOLO, edge AI, ais_bench, embedded inference
---

# Huawei Ascend Integration for Ultralytics YOLO

Deploying computer vision models on [Huawei Ascend](https://www.hiascend.com/) AI Processors requires compiling them into the `.om` offline model format executed by the Ascend AI Core. Exporting [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models to Ascend lets you run fast, low-power inference on Atlas developer boards and OrangePi AIPro devices used across robotics, industrial inspection, and smart camera deployments.

## What is Huawei Ascend?

Huawei Ascend is a family of AI Processors paired with **CANN** (Compute Architecture for Neural Networks), Huawei's heterogeneous computing stack. CANN ships the **ATC** (Ascend Tensor Compiler), a host-side tool that converts a standard ONNX graph into a hardware-specific `.om` offline model targeting the Ascend AI Core.

Because ATC runs entirely on the host, you can compile models on a regular x86-64 or ARM64 Linux machine without any Ascend hardware attached, then copy the resulting `.om` to the target device for inference.

## Train on Ascend NPUs

Ultralytics supports single- and multi-NPU training with `torch_npu`, including AMP, validation, checkpointing, resume, and
AutoBatch. Select one NPU with `device=npu:0` or multiple NPUs with `device=npu:0,1`; distributed training uses HCCL.
Install compatible CANN, PyTorch, and `torch_npu` versions and source the CANN environment first. See the
[Ascend NPU training section](../modes/train.md#huawei-ascend-npu-training) for installation and usage examples.

## Ascend Export Format

The Ultralytics exporter first writes an intermediate ONNX graph, then invokes ATC to compile it for the SoC you target with the `name` argument. The result is a self-contained model directory holding the `.om` binary and its Ultralytics metadata.

## Key Features of Ascend Models

- **Dedicated AI Core**: Compiled models execute on the Ascend AI Core rather than the host CPU, delivering substantially higher throughput for real-time camera pipelines.
- **Host-side compilation**: ATC needs no attached NPU, so export runs in CI, in containers, or on a laptop.
- **ONNX-first**: The standard [Ultralytics ONNX export](onnx.md) feeds directly into the CANN toolchain with no proprietary intermediate format.
- **Static shapes**: The input shape is baked into the `.om` at compile time, removing runtime shape negotiation overhead.
- **Multi-device targeting**: The same ONNX graph compiles for any supported SoC by changing `name`.

## Supported Tasks

Huawei Ascend export supports all seven Ultralytics tasks. Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

{% include "macros/supported-tasks.md" %}

## Supported Devices

Pass the target SoC with `name`; ATC bakes it into the `.om` as `--soc_version`, so it must match the board you deploy to. Locally you can compile for any SoC your CANN installation provides kernels for. [Ultralytics Platform](https://platform.ultralytics.com) supports the Ascend310P1, Ascend310P3, Ascend310B1, and Ascend310B4 targets.

| `name`        | Devices                           | Local export | Platform |
| :------------ | :-------------------------------- | :----------- | :------- |
| `Ascend310P3` | Atlas 300I Pro, Atlas 300V Pro    | ✅           | ✅       |
| `Ascend310P1` | Atlas 300I                        | ✅           | ✅       |
| `Ascend310B4` | OrangePi AIPro 20T, Atlas 200I A2 | ✅           | ✅       |
| `Ascend310B1` | OrangePi AIPro 8T                 | ✅           | ✅       |

## Export to Ascend: Converting Your YOLO Model

!!! note

    Ascend export requires the CANN toolkit, which is Linux-only. The `atc` compiler must be on your `PATH` — source the CANN environment script before exporting, e.g. `source /usr/local/Ascend/ascend-toolkit/set_env.sh`.

### Installation

Install Ultralytics, then install CANN separately from Huawei.

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO
        pip install ultralytics
        ```

The CANN toolkit is distributed by Huawei and is not installed automatically. Download it from the [Ascend community download page](https://www.hiascend.com/developer/download/community/result?module=cann), or use one of the official [`ascendai/cann` container images](https://hub.docker.com/r/ascendai/cann), which bundle ATC and its dependencies.

!!! warning "Match the toolkit to your target SoC"

    ATC can only compile for SoC families whose operator kernels are installed. A toolkit carrying only `ascend310p` kernels fails with `No supported Ops kernel and engine are found for ... Conv2D` when asked for an `Ascend310B4` target. Install the matching `Ascend-cann-kernels-<soc>` package for the device you deploy to.

### Usage

The Ascend format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Inference and validation run on Ascend hardware. Export your model, then load the exported model to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to Ascend format for an Atlas 200I / OrangePi AIPro board
        model.export(format="ascend", name="Ascend310B4")

        # Load the exported Ascend model and run inference on the device
        ascend_model = YOLO("yolo26n_ascend_model")
        results = ascend_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to Ascend format
        yolo export model=yolo26n.pt format=ascend name=Ascend310B4

        # Run inference with the exported model on the device
        yolo predict model=yolo26n_ascend_model source='https://ultralytics.com/images/bus.jpg'
        ```

### Export Arguments

| Argument   | Type             | Default         | Description                                                                                                                                                                                          |
| :--------- | :--------------- | :-------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'ascend'`      | Target format for the exported model, defining compatibility with Ascend AI Processors.                                                                                                              |
| `name`     | `str`            | `'Ascend310B4'` | Target SoC passed to ATC as `--soc_version`, e.g. `Ascend310P3`. Any SoC your CANN install has kernels for is valid; see [Supported Devices](#supported-devices). Must match your deployment device. |
| `imgsz`    | `int` or `tuple` | `640`           | Desired image size for the model input, baked into the `.om` as a static shape.                                                                                                                      |
| `batch`    | `int`            | `1`             | Static batch size compiled into the offline model.                                                                                                                                                   |
| `quantize` | `int`            | `16`            | Quantization precision. Ascend export is FP16-only and auto-enables `16` if not specified. Replaces the deprecated `half`/`int8` flags.                                                              |
| `opset`    | `int`            | `17`            | ONNX opset for the intermediate graph. Capped at 17, the highest version the CANN ONNX parser accepts.                                                                                               |
| `simplify` | `bool`           | `True`          | Simplifies the intermediate ONNX graph with `onnxslim`.                                                                                                                                              |
| `nms`      | `bool`, optional | `None`          | Select raw output (`None`, default), embedded NMS (`True`), or the NMS-free head (`False`).                                                                                                          |

!!! note "Why is FP32 unavailable?"

    The Ascend AI Core accepts only `DT_FLOAT16` and `DT_INT8` inputs for convolutions, so an FP32 offline model cannot execute on the hardware. Requesting `quantize=32` raises an error rather than silently producing a model that will not run.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Output Structure

After a successful export, a model directory is created with the following layout:

    yolo26n_ascend_model/
    ├── yolo26n_Ascend310B4.om  # Compiled Ascend offline model (AI Core executable)
    └── metadata.yaml           # Model metadata (classes, image size, task, etc.)

The `.om` file is the compiled offline model that the CANN runtime loads on the device. Its name carries the target SoC so the artifact is self-describing; keep one `.om` per directory, since the loader picks the single model it finds there. The `metadata.yaml` contains class names, image size, and other information used by the Ultralytics inference pipeline.

## Deploying Exported YOLO Ascend Models

Once you've successfully exported your Ultralytics YOLO model to Ascend format, the next step is deploying it on Ascend hardware.

### Runtime Installation

Inference requires the CANN runtime on the device plus the `ais_bench` Python package, which wraps CANN's pyACL bindings. Build and install it from the Huawei [Ascend tools repository](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench):

```bash
# On the Ascend device, with CANN installed and sourced
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Build and install the ais_bench inference wheel
git clone https://gitee.com/ascend/tools.git
cd tools/ais-bench_workload/tool/ais_bench
pip3 wheel ./backend -v
pip3 install ./aclruntime-*.whl
pip3 wheel ./ -v
pip3 install ./ais_bench-*.whl
```

Copy the exported `yolo26n_ascend_model` directory to the device, then run inference exactly as you would with any other Ultralytics format.

## Recommended Workflow

1. **Train** your model using Ultralytics [Train Mode](../modes/train.md), or start from a pre-trained checkpoint.
2. **Export** to Ascend on any Linux host with CANN installed, passing the `name` that matches your target SoC.
3. **Copy** the exported model directory to the Ascend device.
4. **Deploy** with the CANN runtime and `ais_bench` for on-device inference.

## Real-World Applications

YOLO models running on Ascend hardware suit a wide range of embedded and industrial vision applications:

- **Industrial Automation**: High-speed quality inspection and defect detection on production lines.
- **Smart Surveillance**: Real-time multi-camera object detection on edge gateways without a central server.
- **Robotics**: On-board perception for autonomous mobile robots and collaborative arms.
- **Smart Cities**: Traffic monitoring and pedestrian analytics on low-power roadside units.

## Summary

In this guide, you have learned how to train Ultralytics YOLO on Huawei Ascend NPUs and export models to the Ascend `.om` format using the CANN ATC compiler. Export runs entirely on the host, produces a self-contained model directory, and targets any supported Ascend SoC through a single `name` argument.

For more details on usage, visit the [official CANN documentation](https://www.hiascend.com/cann).

Also, if you'd like to know more about other Ultralytics YOLO integrations, visit our [integration guide page](index.md) to find plenty of helpful resources.

## FAQ

### How do I export my Ultralytics YOLO model to Ascend format?

Install Ultralytics and the CANN toolkit, source the CANN environment so `atc` is on your `PATH`, then export with `model.export(format="ascend", name="Ascend310B4")` in Python or `yolo export model=yolo26n.pt format=ascend name=Ascend310B4` from the CLI. Set `name` to the SoC of your deployment device.

### Do I need Ascend hardware to export a model?

No. The ATC compiler runs entirely on the host, so compilation works on a standard x86-64 or ARM64 Linux machine with no NPU attached. You only need Ascend hardware to run inference on the exported `.om`.

### Why does my export fail with "No supported Ops kernel and engine are found"?

Your CANN installation does not carry operator kernels for the SoC you requested. Each toolkit bundles kernels for specific SoC families, so install the `Ascend-cann-kernels-<soc>` package matching your target, or use a container image built for that device family.

### Why does Ascend export only support FP16?

Ascend AI Core convolutions accept only `DT_FLOAT16` and `DT_INT8` tensors. An FP32 offline model would fail to execute on the hardware, so the exporter compiles with `--precision_mode=force_fp16` and rejects an explicit `quantize=32` request.

### Which Ascend devices are supported?

Any SoC that your installed CANN toolkit provides kernels for, selected through the `name` argument. See [Supported Devices](#supported-devices) for the four validated targets and their availability on Ultralytics Platform.
