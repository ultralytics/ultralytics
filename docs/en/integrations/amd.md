---
title: AMD GPU Training and Inference with Ultralytics YOLO, ROCm and MIGraphX
comments: true
description: Deploy Ultralytics YOLO on AMD GPUs. Export to ONNX and run accelerated inference through the ONNX Runtime MIGraphX execution provider on ROCm, with support for all YOLO26 tasks, and train natively with PyTorch ROCm.
keywords: AMD, ROCm, MIGraphX, MIGraphXExecutionProvider, onnxruntime-ep-migraphx, AMD GPU inference, AMD GPU training, Radeon, Instinct, ONNX Runtime, Ultralytics, YOLO, YOLO26, model deployment
---

# AMD GPU Training and Inference with Ultralytics YOLO, ROCm and MIGraphX

!!! warning "Linux x86_64 only"

    This guide targets **Linux x86_64** hosts with the AMD GPU kernel driver (`amdgpu`) installed, and all commands below are for a Linux shell. The MIGraphX execution provider is published only as Linux x86_64 wheels for Python 3.11 or newer, so native Windows is not supported, and Windows Subsystem for Linux (WSL2) has not been validated yet.

Ultralytics YOLO runs on AMD Instinct and supported Radeon GPUs through [ROCm](https://rocm.docs.amd.com/), AMD's open GPU compute stack. With a ROCm build of PyTorch, you [train](#train-on-amd-gpus-with-pytorch-rocm), validate, and predict with `.pt` models using the same `device=0` argument as on any other GPU, with no code changes.

For deployment, export the trained model to [ONNX](onnx.md) and run it through [MIGraphX](https://github.com/ROCm/AMDMIGraphX), AMD's graph-optimization and inference engine. The Ultralytics [ONNX backend](onnx.md) detects ROCm and selects the ONNX Runtime `MIGraphXExecutionProvider` automatically, so the same `predict` call that runs on NVIDIA GPUs runs on AMD GPUs. Ultralytics tests ONNX export and MIGraphX inference on AMD GPU hardware in continuous integration.

## Installation

The Python stack installs entirely through `pip` from AMD's ROCm wheel indexes, so no `apt` packages or root access are needed for PyTorch, the plugin, or the ROCm runtime libraries, provided the host already has the AMD GPU kernel driver (`amdgpu` / `/dev/kfd`) in place.

!!! tip "Installation"

    === "Auto-detect (recommended)"

        Detect your GPU architecture (for example `gfx1151` on Strix Halo) and install only the PyTorch kernels it needs:

        ```bash
        # Read the GPU architecture from the amdgpu driver, e.g. device-gfx1151
        GPU_ARCH=$(grep -rhs '^gfx_target_version' /sys/class/kfd/kfd/topology/nodes | awk '$2 {printf "device-gfx%d%d%x\n", int($2/10000), int($2/100)%100, $2%100}' | sort -u | paste -sd, -)

        # ROCm PyTorch + torchvision for the detected architecture
        pip install "torch[${GPU_ARCH:?No AMD GPU found}]" "torchvision[$GPU_ARCH]" --index-url https://stable.repo.amd.com/rocm/whl-next/

        # Ultralytics
        pip install ultralytics
        ```

    === "All architectures"

        Install PyTorch kernels for every supported AMD GPU architecture. Use this for container images or environments shared across different GPUs, since it is a much larger download.

        ```bash
        # ROCm PyTorch + torchvision with kernels for all supported architectures
        pip install "torch[device-all]" "torchvision[device-all]" --index-url https://stable.repo.amd.com/rocm/whl-next/

        # Ultralytics
        pip install ultralytics
        ```

Ultralytics installs `onnx` automatically on the first ONNX export. On the first ONNX inference on a ROCm system, it also installs the MIGraphX execution provider plugin ([`onnxruntime-ep-migraphx`](https://stable.repo.amd.com/rocm/onnxruntime/whl-next/onnxruntime-ep-migraphx/)) and the MIGraphX runtime libraries it loads ([`migraphx-libs`](https://stable.repo.amd.com/rocm/migraphx/whl-next/migraphx-libs/)), pinned to versions tested together. For detailed instructions and best practices, check our [YOLO26 Installation guide](../quickstart.md); if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md).

## Train on AMD GPUs with PyTorch ROCm

Native training, validation, and prediction on `.pt` models run on AMD GPUs through [PyTorch ROCm](https://pytorch.org/get-started/locally/), independent of the MIGraphX inference path below. Install a ROCm build of PyTorch as shown in [Installation](#installation) and use the same device arguments as any other [Ultralytics Train](../modes/train.md) run:

!!! note "Why AMD GPUs report as CUDA in Ultralytics"

    The ROCm build of PyTorch uses HIP internally but deliberately reuses the `torch.cuda` interfaces, so `torch.cuda.is_available()` returns `True` and AMD GPUs are addressed with standard CUDA-style IDs. Use `device=0` or `device=cuda:0` for AMD GPUs; `rocm` is not a PyTorch device type. See the [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html) for details.

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

!!! tip "Mixed precision on ROCm"

    Ultralytics enables AMP by default and disables it automatically if a pre-training check finds that mixed-precision results diverge from full precision. ROCm AMP behavior can change with the PyTorch and ROCm versions, so if a run still produces NaN losses or zero mAP, train with `amp=False`:

    ```bash
    yolo detect train data=coco8.yaml model=yolo26n.pt device=0 amp=False
    ```

## Export and Inference with MIGraphX

[ONNX Runtime](https://onnxruntime.ai/) is a cross-platform inference engine that runs a single ONNX model on many hardware backends through pluggable [execution providers](https://onnxruntime.ai/docs/execution-providers/) (EPs). Each EP maps the ONNX graph onto a specific accelerator: CUDA for NVIDIA GPUs, CoreML for Apple silicon, and **MIGraphX** for AMD GPUs on ROCm, which ships as the `onnxruntime-ep-migraphx` plugin and compiles the graph into a tuned program for your GPU.

MIGraphX inference uses the standard [ONNX export](onnx.md). Export the model to ONNX (`format="onnx"`) and Ultralytics inference will automatically use the `MIGraphXExecutionProvider` backend to run the ONNX model on your AMD GPU.

### Key Features of MIGraphX Inference

- **Automatic provider selection**: On a ROCm host the ONNX backend registers the plugin and selects `MIGraphXExecutionProvider` with no code changes. If the plugin is missing or fails to load, inference falls back to the CPU with a warning.
- **Graph optimization**: MIGraphX applies operator fusion, memory planning, and kernel selection tuned for AMD GPU architectures.
- **Zero-copy IO binding**: Inputs and outputs are bound directly to GPU tensors through the DLPack protocol, avoiding host round-trips during inference.
- **Precision options**: Run FP32 or export an FP16 ONNX model for reduced-precision inference.
- **Portable artifact**: A single `.onnx` file runs on CPUs, NVIDIA GPUs, and AMD GPUs, letting you target multiple platforms from one export.
- **Reproducible deployment**: The full stack (ROCm PyTorch, the MIGraphX plugin, and its libraries) installs through `pip` from AMD's ROCm wheel indexes.

### Supported Tasks

MIGraphX inference supports all seven Ultralytics tasks. Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

{% include "macros/supported-tasks.md" %}

### Usage

Before diving into the usage instructions, be sure to check out the range of [YOLO26 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

The ONNX format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Inference and validation on an AMD GPU require a ROCm system with the MIGraphX plugin installed. Export your model, then load the exported model to run inference or validate its accuracy on `device=0`.

!!! warning "Conflict with onnxruntime-gpu"

    `onnxruntime-gpu` and the standard `onnxruntime` package install into the same `onnxruntime` Python module, so whichever is installed last overwrites the other. Uninstalling only one of them leaves the module broken, and older `onnxruntime-gpu` releases can crash MIGraphX inference. If an earlier setup installed `onnxruntime-gpu`, run `pip uninstall -y onnxruntime-gpu onnxruntime`, and Ultralytics reinstalls the standard `onnxruntime` on the next ONNX inference.

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

| Argument   | Type             | Default  | Description                                                                                                                                                                                                                           |
| :--------- | :--------------- | :------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `format`   | `str`            | `'onnx'` | Target format for the exported model. Use `onnx` for MIGraphX EP inference.                                                                                                                                                           |
| `imgsz`    | `int` or `tuple` | `640`    | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)`.                                                                                                                             |
| `quantize` | `int` or `str`   | `None`   | Precision of the exported ONNX model: `16` (FP16) for reduced-precision inference; `32`/unset is FP32.                                                                                                                                |
| `dynamic`  | `bool`           | `False`  | Allows dynamic input sizes. Static shapes let MIGraphX compile a specialized program and enable zero-copy IO binding.                                                                                                                 |
| `simplify` | `bool`           | `True`   | Simplifies the model graph with `onnxslim`, potentially improving performance and compatibility.                                                                                                                                      |
| `opset`    | `int`            | `None`   | ONNX opset version for compatibility with different runtimes. If not set, uses the latest supported version.                                                                                                                          |
| `nms`      | `bool`, optional | `None`   | Select raw output (`None`, default), embedded NMS (`True`), or the NMS-free head (`False`). Embedded NMS (`True`) is not yet supported by the MIGraphX EP ([ROCm/AMDMIGraphX#5246](https://github.com/ROCm/AMDMIGraphX/issues/5246)). |
| `batch`    | `int`            | `1`      | Export batch size, or the max number of images the exported model processes concurrently in `predict` mode.                                                                                                                           |
| `device`   | `str`            | `None`   | Device for exporting: GPU (`device=0`), CPU (`device=cpu`).                                                                                                                                                                           |

For the full list of export arguments, see the [ONNX integration](onnx.md#export-arguments) and the [Ultralytics documentation page on exporting](../modes/export.md).

## Deploying on AMD GPUs with MIGraphX

!!! note "Compiled-program cache"

    The MIGraphX EP compiles the graph on the first session, which dominates initial load time. Ultralytics caches the compiled program per model under the [Ultralytics config directory](../quickstart.md#ultralytics-settings) so later loads of the same model skip recompilation. The cache keeps the 8 most recently used models and evicts older ones, so it cannot grow without bound. Set `ORT_MIGRAPHX_CACHE_DIR` to override the location. Cache keys lead with the MIGraphX version, so a runtime upgrade recompiles rather than reusing a stale program.

!!! note "Compile time"

    Ultralytics disables MIGraphX Winograd convolution kernels by default (`MIGRAPHX_DISABLE_WINOGRAD=1`) to cut cold-compile time on YOLO graphs with no measurable inference change ([ROCm/AMDMIGraphX#5234](https://github.com/ROCm/AMDMIGraphX/issues/5234)); set `MIGRAPHX_DISABLE_WINOGRAD=0` to re-enable them.

For a ready-to-run environment, [`Dockerfile-amd`](https://github.com/ultralytics/ultralytics/blob/main/docker/Dockerfile-amd) builds the [`ultralytics/ultralytics:latest-amd`](https://hub.docker.com/r/ultralytics/ultralytics/tags?name=latest-amd) image with ROCm PyTorch and the MIGraphX EP preinstalled. See [Using GPUs](../guides/docker-quickstart.md#using-gpus) in the Docker Quickstart for the `docker run` flags that expose AMD GPUs to the container.

## Support at a Glance

Support for one AMD product or runtime does not imply support for every AMD accelerator. This table summarizes the current status in the Ultralytics Python package.

| AMD product or runtime                           | Support | Usage or status                                                                                                      |
| :----------------------------------------------- | :------ | :------------------------------------------------------------------------------------------------------------------- |
| AMD Instinct and supported Radeon GPUs with ROCm | ✅      | Train, validate, and run native PyTorch models with `device=0` or `device=cuda:0`.                                   |
| MIGraphX inference                               | ✅      | Run exported ONNX models on AMD GPUs through the MIGraphX EP. All YOLO26 tasks are supported.                        |
| Multi-GPU ROCm                                   | ✅      | Use `device=0,1` or `device=[0, 1]`; distributed execution follows the installed PyTorch ROCm stack.                 |
| ROCm Automatic Mixed Precision (AMP)             | ⚠️      | Available when the installed PyTorch and ROCm versions pass Ultralytics AMP checks; use `amp=False` if incompatible. |
| AMD Docker image                                 | ✅      | [`latest-amd`](../guides/docker-quickstart.md#using-gpus) ships ROCm PyTorch with the MIGraphX EP preinstalled.      |
| Native MIGraphX export                           | ❌      | Export to ONNX with `format="onnx"` and run it on the MIGraphX EP for AMD GPU inference.                             |
| Windows DirectML                                 | ❌      | No DirectML training or prediction backend in the Python package.                                                    |
| Ryzen AI NPU                                     | ❌      | No native NPU integration; external ONNX/Vitis AI workflows are community-managed.                                   |
| AMD CPUs                                         | ✅ CPU  | Use `device=cpu`; standard CPU execution, not an AMD-specific acceleration backend.                                  |

!!! note "Check AMD and PyTorch compatibility first"

    ROCm availability depends on the exact GPU, operating system, ROCm version, and PyTorch build. Confirm your hardware in AMD's [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) before installing.

## Summary

In this guide, you learned how to train Ultralytics YOLO26 models on AMD GPUs with PyTorch ROCm using the standard `device=0` argument, then export them to ONNX for accelerated inference through the ONNX Runtime MIGraphX execution provider. The ONNX backend selects `MIGraphXExecutionProvider` automatically, caches the compiled program for fast subsequent loads, and supports all YOLO26 tasks with no code changes.

For other deployment targets, browse the [integration guide page](../integrations/index.md), and compare export formats with [Benchmark mode](../modes/benchmark.md).

## FAQ

### Can I train YOLO26 on an AMD GPU?

Yes. Install a ROCm build of PyTorch as shown in [Installation](#installation), then train `.pt` models with `device=0`, or `device=0,1` for multiple GPUs. Training runs through PyTorch ROCm and does not use MIGraphX or the ONNX plugin. See [Train on AMD GPUs with PyTorch ROCm](#train-on-amd-gpus-with-pytorch-rocm) for examples.

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

MIGraphX compiles the ONNX graph into an optimized program on the first session, which dominates initial load time. Ultralytics caches the compiled program, so later loads of the same model skip recompilation. See the [compiled-program cache](#deploying-on-amd-gpus-with-migraphx) note for its location and size limit.

### Why does `torch.cuda.is_available()` return `True` on my AMD system?

This is expected. PyTorch ROCm intentionally reuses the `torch.cuda` API and CUDA-style device strings for Python compatibility. Use `device=0` or `device=cuda:0`; the model still executes through HIP and ROCm on the AMD GPU.

### Does Ultralytics support DirectML or Ryzen AI NPUs?

Not through the Python package. DirectML has no training or prediction backend, and Ryzen AI NPUs are not exposed through PyTorch ROCm. Community workflows may export to ONNX and run with AMD's external Ryzen AI or Vitis AI tools, but those runtimes are outside the supported Ultralytics execution path.

### How do I select a specific GPU on a multi-GPU AMD host?

Set `HIP_VISIBLE_DEVICES` to the physical GPU index (for example `HIP_VISIBLE_DEVICES=2`), which exposes the chosen GPU as `device=0` for MIGraphX EP inference.
