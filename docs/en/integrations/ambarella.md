---
comments: true
description: Deploy Ultralytics YOLO models on Ambarella CVflow SoCs like the CV72 with SpongeTorch compression-aware training, ONNX export, and CVflow toolchain compilation.
keywords: Ambarella, Ambarella YOLO, deploy YOLO on Ambarella, Ambarella object detection, CVflow, Ambarella CVflow, CV72, CV72S, CV75, CV5, CV3-AD, N1, SpongeTorch, SpongeKit, AmbaPB, CVflow toolchain, Cooper Developer Platform, model compression, pruning, quantization, edge AI, NPU, AI camera, security camera, smart camera, Ultralytics, YOLO26, YOLO11, ONNX export, embedded AI, edge deployment
---

# Ambarella CVflow Export for Ultralytics YOLO Models

!!! warning "Preview guide — not yet vendor verified"

    This guide is an early preview and is not yet complete or verified by Ambarella. Commands, compatibility details, and workflow steps may change as vendor feedback becomes available. There is currently no `format="ambarella"` export target; the workflow uses the standard ONNX export (`format="onnx"`) combined with the `amba_config`/`amba_chipset` arguments, then compiles the resulting ONNX model into the deployable AmbaPB format offline with Ambarella's CVflow toolchain.

Deploying [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models on [Ambarella](https://www.ambarella.com/) SoCs requires a model format optimized for the CVflow® AI engine. [This fork of Ultralytics](https://github.com/Ambarella-Inc/ultralytics/tree/amba_v8.4.46) integrates Ambarella's **SpongeTorch** compression toolkit directly into the train, validate, and export pipeline, so you can produce pruned and quantization-optimized models that run efficiently on Ambarella hardware. This guide outlines the current [object detection](https://www.ultralytics.com/glossary/object-detection) workflow: compression-aware training, ONNX export, compilation with the CVflow toolchain, and inference with the compiled AmbaPB model.

!!! note

    This workflow requires proprietary Ambarella toolchain components (`spongetorch`, the CVflow compiler, and `cvflowbackend`) that are not available on PyPI. Register on the [Ambarella Developer Zone](https://www.ambarella.com/developer/) to obtain SDK access through the Cooper™ Developer Platform.

## What is Ambarella CVflow?

[Ambarella](https://www.ambarella.com/) is a Santa Clara-based semiconductor company known for its low-power AI vision SoCs, widely used in IP security cameras, dash cameras, drones, robotics, and automotive systems. Its chips are built around **CVflow®**, a dedicated neural vector processing architecture (the on-chip AI accelerator, or NPU) that delivers high inference throughput at very low power — the CV72S runs 4K security-camera AI workloads under 3 W. Models trained in standard frameworks like [PyTorch](https://www.ultralytics.com/glossary/pytorch) are compiled to CVflow's native format with Ambarella's offline toolchain before deployment.

Current CVflow SoC families and their typical applications:

| SoC family  | Typical applications                                                 |
| ----------- | -------------------------------------------------------------------- |
| CV72 / CV75 | 4K AI security cameras, smart cameras, industrial vision             |
| CV5 / CV52  | Drones, action cameras, robotics, multi-camera systems               |
| CV3-AD      | Automotive ADAS and autonomous driving domain controllers            |
| N1          | On-premise generative AI and multi-stream video analytics appliances |

## Why Deploy YOLO on Ambarella?

- **Performance per watt**: CVflow SoCs are designed for always-on edge AI, running real-time [object detection](https://www.ultralytics.com/glossary/object-detection) within camera-grade power budgets.
- **Compression-aware training**: SpongeTorch applies [pruning](https://www.ultralytics.com/glossary/pruning) and [quantization](https://www.ultralytics.com/glossary/model-quantization)-aware optimization during training, so the model learns to stay accurate while becoming NPU-friendly.
- **Bit-exact host validation**: the compiled AmbaPB model runs through Ultralytics `predict`/`val` on your workstation exactly as it will execute on the chip, so you can measure quantized [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) before touching hardware.
- **Integrated camera pipeline**: Ambarella SoCs combine the AI engine with an ISP and video encoders, making them a single-chip solution for AI cameras.

## Workflow Overview

The pipeline has four stages:

1. **Compression-aware training** — train with a SpongeKit config (`amba_config`) so SpongeTorch applies pruning/quantization progressively during training.
2. **ONNX export** — export the compressed checkpoint with the same `amba_config`, preserving the compression structure in the ONNX graph.
3. **CVflow compilation** — compile the ONNX model to an AmbaPB artifact with the CVflow toolchain.
4. **Inference and validation** — run the compiled `*.ambapb.ckpt.onnx` model through Ultralytics `predict`/`val` via the AmbaPB backend, then deploy on the board.

SpongeTorch training and SpongeTorch-aware export can be replaced by a plain ONNX export if you do not need SpongeTorch's training-time optimizations (see [Exporting Without SpongeTorch](#exporting-without-spongetorch)).

## Prerequisites

### Installation

Install [this Ultralytics fork](https://github.com/Ambarella-Inc/ultralytics/tree/amba_v8.4.46), then install the Ambarella toolchain wheels from the SDK distribution:

!!! Tip "Installation"

    === "CLI"

        ```bash
        # Install this Ultralytics fork from source
        git clone https://github.com/Ambarella-Inc/ultralytics
        cd ultralytics
        git checkout amba_v8.4.46
        pip install -e .

        # Install Ambarella toolchain wheels from the SDK
        pip install /path/to/spongetorch-*.whl
        pip install /path/to/cvflowbackend-*.whl
        ```

The AmbaPB inference backend locates `cvflowbackend` through the CVflow toolchain's `tv2` command (`tv2 -libpath cvflowbackend`), so the toolchain must be installed and on your `PATH` before running inference or validation with compiled models.

### SpongeKit Configuration File

SpongeTorch is driven by a SpongeKit configuration file (protobuf-text format, `.prototxt`) that defines the compression passes to apply: pruning sparsity targets, quantization settings, and the compression schedule. Obtain example configurations and the matching schema documentation from your Ambarella SDK release. Use the training config whenever validation must prepare an unprepared model, and always use the same config when exporting a compressed checkpoint.

## Amba Arguments

Two arguments control the SpongeTorch integration across `train`, `val`, and `export` modes:

| Argument       | Type  | Default | Description                                                                                                                      |
| -------------- | ----- | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `amba_config`  | `str` | `None`  | Path to the SpongeKit config passed to `spongetorch.prepare()`. Enables compression-aware training and SpongeTorch-aware export. |
| `amba_chipset` | `str` | `None`  | Target chipset name passed to `spongetorch.set_target_chipset()`, e.g. `CV72`.                                                   |

The fork also adds a general export argument:

| Argument      | Type  | Default | Description                                                                 |
| ------------- | ----- | ------- | --------------------------------------------------------------------------- |
| `export_file` | `str` | `None`  | Custom export output path/name, e.g. `'/tmp/model.onnx'` or `'model.onnx'`. |

## Compression-Aware Training

Train (or fine-tune) your model with SpongeTorch compression enabled:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.train(
            data="coco8.yaml",
            epochs=100,
            amba_config="config.prototxt",
            amba_chipset="CV72",
        )
        ```

    === "CLI"

        ```bash
        yolo train model=yolo26n.pt data=coco8.yaml epochs=100 \
          amba_config=config.prototxt amba_chipset=CV72
        ```

When `amba_config` is set, the trainer wraps the model and optimizer with `spongetorch.prepare()` at setup. Compression is applied progressively on a step schedule, so the network learns to stay accurate while becoming sparse and quantization-friendly. The trained checkpoint stores SpongeTorch's sparse state (`_orig`/`_mask` tensors), which the export step later requires. The config file is copied into the run directory as `amba_config.prototxt` for reproducibility.

!!! note "Checkpoint gating"

    `best.pt` and `last.pt` are intentionally not saved until the SpongeTorch compression schedule crosses its `end_step` — a half-compressed checkpoint would not be usable. Ensure `epochs` is long enough for the schedule in your config to complete; the log reports when checkpoint saving begins. If training ends before the schedule completes, the final epoch is saved anyway with a warning, but such a checkpoint should not be deployed.

!!! tip "Fine-tune instead of training from scratch"

    For best accuracy, first train your model normally (or start from a pretrained checkpoint), then run a shorter compression fine-tune with `amba_config` on the trained weights.

### Validating the Compressed Checkpoint

Validate accuracy before compiling, using the same config:

!!! example "Usage"

    === "CLI"

        ```bash
        yolo val model=runs/detect/train/weights/best.pt data=coco8.yaml \
          amba_config=config.prototxt amba_chipset=CV72
        ```

The validator re-applies `spongetorch.prepare()` when required and disables Conv+BN fusion so the compression structure is preserved. Compare mAP against your uncompressed baseline; if the [accuracy](https://www.ultralytics.com/glossary/accuracy) drop is too large, adjust the SpongeKit config and retrain.

## Export to ONNX

Export the compressed checkpoint with the **same** `amba_config` used in training:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("runs/detect/train/weights/best.pt")
        model.export(
            format="onnx",
            amba_config="config.prototxt",
            amba_chipset="CV72",
        )
        ```

    === "CLI"

        ```bash
        yolo export model=runs/detect/train/weights/best.pt format=onnx \
          amba_config=config.prototxt amba_chipset=CV72
        ```

The exporter rebuilds the model, re-applies `spongetorch.prepare()` with your config, reloads the sparse checkpoint weights into the prepared structure, and traces to ONNX with Conv+BN fusion disabled — producing a graph in the exact form the CVflow compiler expects.

### Preserve Model Metadata

ONNX export embeds the model task, class names, stride, and input size in the ONNX file, while the AmbaPB backend reads this information from a `metadata.yaml` sidecar next to the compiled model. Unless your CVflow compiler creates this sidecar, extract it from the ONNX model before compilation:

```python
import onnx

from ultralytics.utils import YAML

model = onnx.load("model.onnx")
YAML.save("metadata.yaml", {item.key: item.value for item in model.metadata_props})
```

Keep `metadata.yaml` in the same directory as the compiled `*.ambapb.ckpt.onnx` or `*.ambapb.fastckpt.onnx` file.

!!! warning

    - The checkpoint must contain SpongeTorch compression state. Exporting a plain checkpoint with `amba_config` set raises: *"Checkpoint has no SpongeTorch pruning state... Use a compressed checkpoint from amba training before export."*
    - The config must match the one used during training, or the weight reload fails.

## Compile with the CVflow Toolchain

Compile the exported ONNX model for your target chipset using the CVflow compiler from the SDK, following the SDK's compilation guide. The compiler maps the graph onto the CVflow AI engine (quantization, scheduling, memory planning) and produces the deployable AmbaPB artifact.

!!! note

    For Ultralytics to recognize the compiled model, its filename must end with `.ambapb.ckpt.onnx` or `.ambapb.fastckpt.onnx`.

## Run Inference with the Compiled Model

The compiled AmbaPB model loads directly through the Ultralytics API — [AutoBackend](../reference/nn/autobackend.md) detects the `.ambapb` suffix and routes inference through `cvflowbackend`, executing the model bit-exactly as it will run on the AI engine:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("model.ambapb.ckpt.onnx")

        # Inference
        results = model("https://ultralytics.com/images/bus.jpg")

        # Validation
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        yolo predict model=model.ambapb.ckpt.onnx source='https://ultralytics.com/images/bus.jpg'
        yolo val model=model.ambapb.ckpt.onnx data=coco8.yaml
        ```

This is the final accuracy check before hardware deployment, including all compiler quantization effects. If a `metadata.yaml` file sits next to the compiled model, the backend reads class names, stride, and task information from it. The backend uses CVflow inference mode `acinf` by default; set the environment variable `ULTRALYTICS_AMBAPB_DEBUG=1` to log input/output details for debugging.

## Deploy on the Board

Load the compiled model on your Ambarella device using the Ambarella SDK runtime. Preprocessing and postprocessing must match what the detection model was compiled for: letterboxed RGB input in the `0–255` range (the Ultralytics AmbaPB backend feeds the compiled model `0–255` RGB), and standard YOLO detection decoding on the outputs. Refer to the SDK deployment documentation for runtime APIs.

## Exporting Without SpongeTorch

If you do not need SpongeTorch's training-time pruning and quantization-aware optimizations, the standard Ultralytics pipeline also produces a CVflow-compilable model:

!!! example "Usage"

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx
        ```

Compile the resulting ONNX with the CVflow toolchain, which performs post-training quantization itself. This path trades some NPU performance and quantized accuracy for a simpler workflow with no `spongetorch` dependency at training time.

## Real-World Applications

Ultralytics YOLO models on Ambarella CVflow SoCs power always-on vision at the edge:

- **AI security cameras**: real-time person and vehicle detection on 4K IP cameras within a sub-3 W power budget.
- **Drones and robotics**: onboard object detection and tracking for navigation, inspection, and delivery on CV5-class chips.
- **Automotive**: ADAS perception workloads such as pedestrian and vehicle detection on CV3-AD domain controllers.
- **Industrial and retail analytics**: multi-stream people counting, PPE detection, and shelf monitoring on edge appliances.

## Summary

This preview guide outlined the current workflow to deploy Ultralytics YOLO models on Ambarella CVflow SoCs: compression-aware training with SpongeTorch (`amba_config`/`amba_chipset`), ONNX export of the compressed checkpoint, offline compilation to AmbaPB with the CVflow toolchain, and bit-exact validation of the compiled model through Ultralytics before board deployment.

For other edge AI targets, see the related [Hailo](hailo.md), [Rockchip RKNN](rockchip-rknn.md), [Sony IMX500](sony-imx500.md), [Qualcomm QNN](qnn.md), [DEEPX](deepx.md), and [Axelera](axelera.md) guides. For the full list of export formats, visit the [Export mode](../modes/export.md) documentation and the [integrations page](index.md).

## FAQ

### Can I export a YOLO model directly to Ambarella format with `model.export()`?

No. There is no `format="ambarella"` target. Export to ONNX (optionally with SpongeTorch compression via `amba_config`), then compile the ONNX model to AmbaPB offline with Ambarella's CVflow toolchain from the SDK.

### Which Ambarella chips can run Ultralytics YOLO models?

Any CVflow-based SoC supported by your CVflow toolchain may be targeted, including the CV72/CV75 families for AI cameras, CV5/CV52 for drones and robotics, and CV3-AD for automotive. The `amba_chipset` argument configures SpongeTorch's optimization target; select the matching target separately when compiling. Accepted chipset strings and availability depend on the installed SDK release.

### What is SpongeTorch and do I need it?

SpongeTorch is Ambarella's model compression toolkit, integrated into the Ambarella fork of Ultralytics for pruning and quantization-aware training. It is optional: a plain Ultralytics ONNX export can also be compiled with the CVflow toolchain using post-training quantization, at some cost in NPU performance and quantized accuracy.

### Where do I get the Ambarella SDK, SpongeTorch, and the CVflow toolchain?

They are proprietary and not on PyPI. Register on the [Ambarella Developer Zone](https://www.ambarella.com/developer/) to request SDK access; the `spongetorch` and `cvflowbackend` wheels and the CVflow compiler ship with the SDK distribution.

### How do I check the accuracy of the compiled model before deploying?

Run `yolo val model=model.ambapb.ckpt.onnx data=your_data.yaml` with the Ambarella fork installed. The AmbaPB backend executes the compiled model bit-exactly as it runs on the CVflow AI engine, so the reported mAP includes all compiler quantization effects.
