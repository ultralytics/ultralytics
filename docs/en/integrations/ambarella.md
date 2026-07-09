---
comments: true
description: Learn how to train, compress, and export Ultralytics YOLO models for Ambarella CVflow SoCs using SpongeTorch compression-aware training and the CVFlow toolchain.
keywords: YOLO26, Ambarella, CVflow, SpongeTorch, SpongeKit, AmbaPB, model export, model compression, pruning, quantization, Ultralytics, edge AI, NPU, CV72, embedded devices
---

# Ambarella Export for Ultralytics YOLO Models

!!! warning "Not a direct Ultralytics export format"

    There is no format="ambarella" export target. The workflow uses the standard ONNX export (format="onnx") combined with the amba_config/amba_chipset arguments, and the resulting ONNX model is then compiled into the deployable AmbaPB format offline with Ambarella's CVFlow toolchain.

Deploying [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models on Ambarella SoCs requires a model format optimized for the CVflow® neural processing engine. [This fork of Ultralytics](https://github.com/Ambarella-Inc/ultralytics/tree/amba_v8.4.46) integrates Ambarella's **SpongeTorch** compression toolkit directly into the train, validate, and export pipeline, so you can produce pruned and quantization-optimized models that run efficiently on Ambarella hardware. This guide walks through the complete workflow: compression-aware training, ONNX export, compilation with the CVFlow toolchain, and inference with the compiled AmbaPB model.

!!! note

    This workflow requires proprietary Ambarella toolchain components (`spongetorch`, the CVFlow compiler, and `cvflowbackend`) that are not available on PyPI. Register on the Ambarella Developer Platform to obtain SDK access.

## What is Ambarella CVflow?

Ambarella designs low-power AI vision SoCs built around the CVflow architecture, a dedicated neural vector processing engine that delivers high inference throughput at very low power. CVflow SoCs such as the CV72 power intelligent cameras, robotics, automotive, and industrial edge AI applications. Models trained in standard frameworks like [PyTorch](https://www.ultralytics.com/glossary/pytorch) are compiled to CVflow's native format with Ambarella's offline toolchain before deployment.

## Workflow Overview

The pipeline has four stages:

1. **Compression-aware training** - train with a SpongeKit config (`amba_config`) so SpongeTorch applies pruning/quantization progressively during training.
2. **ONNX export** — export the compressed checkpoint with the same `amba_config`, preserving the compression structure in the ONNX graph.
3. **CVFlow compilation** — compile the ONNX model to an AmbaPB artifact with the CVFlow toolchain.
4. **Inference and validation** — run the compiled `*.ambapb.ckpt.onnx` model through Ultralytics `predict`/`val` via the AmbaPB backend, then deploy on the board.

Stages 1–2 can also be skipped in favor of a plain ONNX export if you do not need SpongeTorch's training-time optimizations (see [Exporting Without SpongeTorch](#exporting-without-spongetorch)).

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

### SpongeKit Configuration File

SpongeTorch is driven by a SpongeKit configuration file (protobuf-text format, `.prototxt`) that defines the compression passes to apply: pruning sparsity targets, quantization settings, and the compression schedule. Example configurations and the config schema documentation ship with the SpongeTorch distribution in the SDK. The same config file must be used for training, validation, and export.

## Amba Arguments

Two arguments control the SpongeTorch integration across `train`, `val`, and `export` modes:

| Argument       | Type  | Default | Description                                                                                                                       |
| -------------- | ----- | ------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `amba_config`  | `str` | `None`  | Path to the SpongeKit config passed to `spongetorch.prepare()`. Enables compression-aware training and SpongeTorch-aware export.  |
| `amba_chipset` | `str` | `None`  | Target chipset name passed to `spongetorch.set_target_chipset()`, e.g. `CV72`.  |

An additional export-only argument is available:

| Argument      | Type  | Default | Description                                                              |
| ------------- | ----- | ------- | ------------------------------------------------------------------------ |
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

    `best.pt` and `last.pt` are intentionally not saved until the SpongeTorch compression schedule crosses its `end_step` - a half-compressed checkpoint would not be usable. Ensure `epochs` is long enough for the schedule in your config to complete; the log reports when checkpoint saving begins.

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

The validator re-applies `spongetorch.prepare()` when required and disables Conv+BN fusion so the compression structure is preserved. Compare mAP against your uncompressed baseline; if the accuracy drop is too large, adjust the SpongeKit config and retrain.

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

The exporter rebuilds the model, re-applies `spongetorch.prepare()` with your config, reloads the sparse checkpoint weights into the prepared structure, and traces to ONNX with Conv+BN fusion disabled - producing a graph in the exact form the CVFlow compiler expects.

!!! warning

    - The checkpoint must contain SpongeTorch compression state. Exporting a plain checkpoint with `amba_config` set raises: *"Checkpoint has no SpongeTorch pruning state... Use a compressed checkpoint from amba training before export."*
    - The config must match the one used during training, or the weight reload fails.

## Compile with the CVFlow Toolchain

Compile the exported ONNX model for your target chipset using the CVFlow compiler from the SDK, following the SDK's compilation guide. The compiler maps the graph onto the CVflow NPU (quantization, scheduling, memory planning) and produces the deployable AmbaPB artifact.

!!! note

    For Ultralytics to recognize the compiled model, its filename must end with `.ambapb.ckpt.onnx` or `.ambapb.fastckpt.onnx`.

## Run Inference with the Compiled Model

The compiled AmbaPB model loads directly through the Ultralytics API - [AutoBackend](../reference/nn/autobackend.md) detects the `.ambapb` suffix and routes inference through `cvflowbackend`, executing the model bit-exactly as it will run on the NPU:

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

This is the final accuracy check before hardware deployment, including all compiler quantization effects. The backend uses CVFlow inference mode `acinf` by default; set the environment variable `ULTRALYTICS_AMBAPB_DEBUG=1` to log input/output details for debugging.

## Deploy on the Board

Load the compiled model on your Ambarella device using the Ambarella SDK runtime. Preprocessing and postprocessing must match Ultralytics conventions: letterboxed RGB input normalized to `0–1`, and standard YOLO detection decoding on the outputs. Refer to the SDK deployment documentation for runtime APIs.

## Exporting Without SpongeTorch

If you do not need SpongeTorch's training-time pruning and quantization-aware optimizations, the standard Ultralytics pipeline also produces a CVFlow-compilable model:

!!! example "Usage"

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx
        ```

Compile the resulting ONNX with the CVFlow toolchain, which performs post-training quantization itself. This path trades some NPU performance and quantized accuracy for a simpler workflow with no `spongetorch` dependency at training time.