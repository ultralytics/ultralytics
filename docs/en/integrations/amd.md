---
title: AMD GPU Inference with Ultralytics YOLO, ROCm and MIGraphX
comments: true
description: Deploy Ultralytics YOLO on AMD GPUs. Export to ONNX and run accelerated inference through the ONNX Runtime MIGraphX execution provider on ROCm, with support for all YOLO26 tasks.
keywords: AMD, ROCm, MIGraphX, MIGraphXExecutionProvider, onnxruntime-ep-migraphx, AMD GPU inference, Radeon, Instinct, ONNX Runtime, Ultralytics, YOLO, YOLO26, model deployment
---

# AMD GPU Inference with Ultralytics YOLO, ROCm and MIGraphX

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on AMD GPUs benefits from a runtime that turns a portable model file into an optimized, hardware-specific program. On AMD hardware that runtime is [MIGraphX](https://github.com/ROCm/AMDMIGraphX), AMD's graph-optimization and inference engine for [ROCm](https://rocm.docs.amd.com/).

By exporting your [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) model to [ONNX](onnx.md) and running it through the ONNX Runtime MIGraphX [execution provider](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html), you get GPU-accelerated inference on AMD Instinct and supported Radeon GPUs with no code changes. The [ONNX backend](onnx.md) detects ROCm, registers the MIGraphX plugin, and selects `MIGraphXExecutionProvider` automatically, so the same `predict` call that runs on NVIDIA GPUs runs on AMD GPUs. A `.pt` model also runs natively on an AMD GPU with `device=0` through PyTorch ROCm, exactly as in [Predict mode](../modes/predict.md); exporting to ONNX adds MIGraphX's graph optimization and a portable, deployment-ready artifact.

## MIGraphX and the ONNX Runtime Execution Provider

[ONNX Runtime](https://onnxruntime.ai/) is a cross-platform inference engine that runs a single ONNX model on many hardware backends through pluggable [execution providers](https://onnxruntime.ai/docs/execution-providers/) (EPs). Each EP maps the ONNX graph onto a specific accelerator: CUDA for NVIDIA GPUs, CoreML for Apple silicon, and **MIGraphX** for AMD GPUs on ROCm.

The MIGraphX EP ships as a loadable plugin package (`onnxruntime-ep-migraphx`) that adds `MIGraphXExecutionProvider` on top of the stock `onnxruntime` module. When Ultralytics runs an ONNX model on a ROCm system, it registers this plugin and hands the graph to MIGraphX, which compiles it into a tuned program for your GPU and executes inference on device.

!!! note "Why AMD GPUs report as CUDA in Ultralytics"

    The ROCm build of PyTorch uses HIP internally but deliberately reuses the `torch.cuda` interfaces, so `torch.cuda.is_available()` returns `True` and AMD GPUs are addressed with standard CUDA-style IDs. Use `device=0` or `device=cuda:0` for AMD GPUs; `rocm` is not a PyTorch device type. See the [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html) for details.

## Why Run YOLO Inference on AMD GPUs with MIGraphX

- **Out-of-the-box GPU acceleration**: On a ROCm host the ONNX backend selects the MIGraphX EP automatically. Without it, ONNX inference on an AMD GPU silently falls back to the CPU, leaving GPU performance unused.
- **No code changes**: Export once to ONNX, then use the standard `predict` and `val` APIs with `device=0`.
- **Portable artifact**: A single `.onnx` file runs on CPUs, NVIDIA GPUs, and AMD GPUs, letting you target multiple platforms from one export.
- **Compiled-program cache**: MIGraphX compiles the graph on the first run and Ultralytics caches the result, so later sessions load fast.
- **Full task coverage**: All YOLO26 tasks run on the MIGraphX EP.

## Key Features of MIGraphX Inference

- **Graph optimization**: MIGraphX applies operator fusion, memory planning, and kernel selection tuned for AMD GPU architectures.
- **Automatic provider selection**: The ONNX backend registers the plugin and picks `MIGraphXExecutionProvider` when ROCm is detected, with a clean CPU fallback otherwise.
- **Zero-copy IO binding**: Inputs and outputs are bound directly to GPU tensors through the DLPack protocol, avoiding host round-trips during inference.
- **Precision options**: Run FP32 or export an FP16 ONNX model for reduced-precision inference.
- **Reproducible deployment**: The full stack — ROCm PyTorch, the MIGraphX plugin, and its libraries — installs through `pip` from AMD's ROCm wheel indexes.

## Supported Tasks

MIGraphX inference supports all seven Ultralytics tasks. Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

{% include "macros/supported-tasks.md" %}

## Export to ONNX for AMD GPU Inference

MIGraphX inference uses the standard [ONNX export](onnx.md). A dedicated `format="migraphx"` export is not yet available; export to ONNX and let the MIGraphX EP compile and run the graph on your AMD GPU.

### Installation

The Python stack installs entirely through `pip` from AMD's ROCm 10 wheel indexes — no `apt` packages or root access are needed for PyTorch, the plugin, or the ROCm runtime libraries, provided the host already has the AMD GPU kernel driver (`amdgpu` / `/dev/kfd`) in place.

!!! tip "Installation"

    ```bash
    # ROCm PyTorch + torchvision; the [device-all] extra pulls the matching GPU kernels and ROCm runtime.
    pip install "torch[device-all]" "torchvision[device-all]" --index-url https://stable.repo.amd.com/rocm/whl-next/

    # Ultralytics
    pip install ultralytics

    # MIGraphX execution provider plugin (pulls stock onnxruntime) plus its C library.
    pip install onnxruntime-ep-migraphx migraphx-libs \
      --extra-index-url https://stable.repo.amd.com/rocm/onnxruntime/whl-next/ \
      --extra-index-url https://stable.repo.amd.com/rocm/migraphx/whl-next/
    ```

If `onnx` or the plugin are missing, Ultralytics installs them automatically on the first ONNX export or inference on a ROCm system. For detailed instructions and best practices, check our [YOLO26 Installation guide](../quickstart.md); if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md).

### Usage

Before diving into the usage instructions, be sure to check out the range of [YOLO26 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

The ONNX format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Inference and validation on an AMD GPU require a ROCm system with the MIGraphX plugin installed. Export your model, then load the exported model to run inference or validate its accuracy on `device=0`.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to ONNX format
        model.export(format="onnx")  # creates 'yolo26n.onnx'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to ONNX format
        yolo export model=yolo26n.pt format=onnx # creates 'yolo26n.onnx'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ONNX model and run inference on an AMD GPU
        model = YOLO("yolo26n.onnx")

        # The MIGraphX execution provider is selected automatically on ROCm
        results = model.predict("https://ultralytics.com/images/bus.jpg", device=0)
        ```

    === "CLI"

        ```bash
        # Run inference with the exported ONNX model on an AMD GPU
        yolo predict model=yolo26n.onnx source='https://ultralytics.com/images/bus.jpg' device=0
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ONNX model
        model = YOLO("yolo26n.onnx")

        # Validate accuracy on the COCO8 dataset on an AMD GPU
        metrics = model.val(data="coco8.yaml", device=0)
        ```

    === "CLI"

        ```bash
        # Validate the exported ONNX model on an AMD GPU
        yolo val model=yolo26n.onnx data=coco8.yaml device=0
        ```

### Export Arguments

MIGraphX inference reuses the ONNX export arguments. The most relevant options for AMD GPU deployment are:

| Argument   | Type             | Default  | Description                                                                                                           |
| :--------- | :--------------- | :------- | :-------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'onnx'` | Target format for the exported model. Use `onnx` for MIGraphX EP inference.                                           |
| `imgsz`    | `int` or `tuple` | `640`    | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)`.             |
| `quantize` | `int` or `str`   | `None`   | Precision of the exported ONNX model: `16` (FP16) for reduced-precision inference; `32`/unset is FP32.                |
| `dynamic`  | `bool`           | `False`  | Allows dynamic input sizes. Static shapes let MIGraphX compile a specialized program and enable zero-copy IO binding. |
| `simplify` | `bool`           | `True`   | Simplifies the model graph with `onnxslim`, potentially improving performance and compatibility.                      |
| `opset`    | `int`            | `None`   | ONNX opset version for compatibility with different runtimes. If not set, uses the latest supported version.          |
| `nms`      | `bool`, optional | `None`   | Select raw output (`None`, default), embedded NMS (`True`), or the NMS-free head (`False`). Embedded NMS (`True`) is not yet supported by the MIGraphX EP ([ROCm/AMDMIGraphX#5246](https://github.com/ROCm/AMDMIGraphX/issues/5246)). |
| `batch`    | `int`            | `1`      | Export batch size, or the max number of images the exported model processes concurrently in `predict` mode.           |
| `device`   | `str`            | `None`   | Device for exporting: GPU (`device=0`), CPU (`device=cpu`).                                                           |

For the full list of export arguments, see the [ONNX integration](onnx.md#export-arguments) and the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying on AMD GPUs with MIGraphX

The wheels target ROCm 10 (MIGraphX 2.17, ONNX Runtime 1.29), so keep the plugin, `migraphx-libs`, and the ROCm runtime on the same ROCm release. The MIGraphX EP is a loadable plugin that adds `MIGraphXExecutionProvider` on top of the stock `onnxruntime` module, so there is no package conflict with `onnxruntime` or `onnxruntime-gpu`.

!!! note "Compiled-program cache"

    The MIGraphX EP compiles the graph on the first session, which dominates initial load time. Ultralytics caches the compiled program per model under the [Ultralytics config directory](../quickstart.md#ultralytics-settings) so later loads of the same model skip recompilation. Set `ORT_MIGRAPHX_CACHE_DIR` to override the location. Cache keys lead with the MIGraphX version, so a runtime upgrade recompiles rather than reusing a stale program.

!!! note "Compile time"

    Ultralytics disables MIGraphX Winograd convolution kernels by default (`MIGRAPHX_DISABLE_WINOGRAD=1`) to cut cold-compile time on YOLO graphs with no measurable inference change ([ROCm/AMDMIGraphX#5234](https://github.com/ROCm/AMDMIGraphX/issues/5234)); set `MIGRAPHX_DISABLE_WINOGRAD=0` to re-enable them.

!!! note "Selecting a GPU on multi-GPU hosts"

    Set `HIP_VISIBLE_DEVICES` (for example `HIP_VISIBLE_DEVICES=2`) to expose the chosen GPU as `device=0`. This is the standard ROCm selection mechanism for MIGraphX EP inference.

For a ready-to-run environment, `docker/Dockerfile-amd` provides a ROCm image with the MIGraphX EP preinstalled, and AMD GPU hardware CI validates the integration on a scheduled job.

## Train on AMD GPUs with PyTorch ROCm

Native training, validation, and prediction on `.pt` models run on AMD GPUs through [PyTorch ROCm](https://pytorch.org/get-started/locally/), independent of the MIGraphX inference path above. Install a ROCm build of PyTorch and use the same device arguments as any other [Ultralytics Train](../modes/train.md) run:

!!! example "ROCm Training"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Train on the first AMD GPU exposed by PyTorch ROCm
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=0)
        ```

    === "CLI"

        ```bash
        # Train on one AMD GPU (use device=0,1 for multiple GPUs)
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=0
        ```

Ultralytics enables [Automatic Mixed Precision (AMP)](https://www.ultralytics.com/glossary/mixed-precision) by default and runs a compatibility check before training; ROCm AMP behavior depends on the PyTorch and ROCm versions, so use `amp=False` when troubleshooting a stack-specific failure.

## Support at a Glance

Support for one AMD product or runtime does not imply support for every AMD accelerator. This table summarizes the current status in the Ultralytics Python package.

| AMD product or runtime                           | Support | Usage or status                                                                                                      |
| :----------------------------------------------- | :------ | :------------------------------------------------------------------------------------------------------------------- |
| AMD Instinct and supported Radeon GPUs with ROCm | ✅      | Train, validate, and run native PyTorch models with `device=0` or `device=cuda:0`.                                   |
| MIGraphX inference                               | ✅      | Run exported ONNX models on AMD GPUs through the MIGraphX EP. All YOLO26 tasks are supported.                        |
| Multi-GPU ROCm                                   | ✅      | Use `device=0,1` or `device=[0, 1]`; distributed execution follows the installed PyTorch ROCm stack.                 |
| ROCm Automatic Mixed Precision (AMP)             | ⚠️      | Available when the installed PyTorch and ROCm versions pass Ultralytics AMP checks; use `amp=False` if incompatible. |
| AMD Docker image and hardware CI                 | ✅      | `docker/Dockerfile-amd` ships the MIGraphX EP; AMD GPU CI runs on a scheduled job.                                   |
| Native `format="migraphx"` export                | ❌ yet  | Not available; use ONNX export plus the MIGraphX EP for AMD GPU inference today.                                     |
| Windows DirectML                                 | ❌      | No DirectML training or prediction backend in the Python package.                                                    |
| Ryzen AI NPU                                     | ❌      | No native NPU integration; external ONNX/Vitis AI workflows are community-managed.                                   |
| AMD CPUs                                         | ✅ CPU  | Use `device=cpu`; standard CPU execution, not an AMD-specific acceleration backend.                                  |

!!! note "Check AMD and PyTorch compatibility first"

    ROCm availability depends on the exact GPU, operating system, ROCm version, and PyTorch build. Confirm your hardware in AMD's [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) before installing.

## Summary

In this guide, you learned how to run Ultralytics YOLO26 inference on AMD GPUs by exporting to ONNX and executing through the ONNX Runtime MIGraphX execution provider on ROCm. The ONNX backend selects `MIGraphXExecutionProvider` automatically, caches the compiled program for fast subsequent loads, and supports all YOLO26 tasks with no code changes. Native training on AMD GPUs is also available through PyTorch ROCm using standard `device=0` selection.

For other deployment targets, browse the [integration guide page](../integrations/index.md), and compare export formats with [Benchmark mode](../modes/benchmark.md).

## FAQ

### How do I run YOLO26 inference on an AMD GPU?

Export your model to ONNX, then run it on a ROCm system with the MIGraphX plugin installed:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Export to ONNX
        model = YOLO("yolo26n.pt")
        model.export(format="onnx")  # creates 'yolo26n.onnx'

        # Run inference on the AMD GPU (MIGraphX EP selected automatically)
        onnx_model = YOLO("yolo26n.onnx")
        results = onnx_model.predict("https://ultralytics.com/images/bus.jpg", device=0)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx
        yolo predict model=yolo26n.onnx source='https://ultralytics.com/images/bus.jpg' device=0
        ```

### Do I need to change my code to use the MIGraphX execution provider?

No. On a ROCm system with `onnxruntime-ep-migraphx` installed, the Ultralytics ONNX backend detects HIP, registers the plugin, and selects `MIGraphXExecutionProvider` automatically. If the plugin is not available, inference falls back to the CPU.

### Why is the first inference slow on AMD GPUs?

MIGraphX compiles the ONNX graph into an optimized program on the first session, which dominates initial load time. Ultralytics caches the compiled program per model under the [Ultralytics config directory](../quickstart.md#ultralytics-settings), so subsequent loads of the same model skip recompilation. Set `ORT_MIGRAPHX_CACHE_DIR` to change where the cache is stored.

### Why does `torch.cuda.is_available()` return `True` on my AMD system?

This is expected. PyTorch ROCm intentionally reuses the `torch.cuda` API and CUDA-style device strings for Python compatibility. Use `device=0` or `device=cuda:0`; the model still executes through HIP and ROCm on the AMD GPU.

### Does Ultralytics support DirectML or Ryzen AI NPUs?

Not through the Python package. DirectML has no training or prediction backend, and Ryzen AI NPUs are not exposed through PyTorch ROCm. Community workflows may export to ONNX and run with AMD's external Ryzen AI or Vitis AI tools, but those runtimes are outside the supported Ultralytics execution path.

### How do I select a specific GPU on a multi-GPU AMD host?

Set `HIP_VISIBLE_DEVICES` to the physical GPU index (for example `HIP_VISIBLE_DEVICES=2`), which exposes the chosen GPU as `device=0` for MIGraphX EP inference.
