---
title: AMD GPU Training with Ultralytics YOLO and ROCm
comments: true
description: Train and run Ultralytics YOLO on supported AMD GPUs with PyTorch ROCm, and understand the current status of MIGraphX, DirectML, and Ryzen AI NPU support.
keywords: AMD, ROCm, Radeon, Instinct, Ryzen AI, MIGraphX, DirectML, PyTorch ROCm, AMD GPU training, Ultralytics, YOLO
---

# AMD GPU Training with Ultralytics YOLO and ROCm

Ultralytics supports training, validation, and inference on compatible AMD GPUs through
[PyTorch ROCm](https://pytorch.org/get-started/locally/). PyTorch intentionally exposes ROCm devices through the same
`torch.cuda` Python API used by CUDA, so Ultralytics does not need a separate `rocm` device type for native PyTorch
models. Install a ROCm build of PyTorch, then select an AMD GPU with the standard `device=0` or `device=cuda:0` syntax.

AMD also offers inference technologies that are separate from PyTorch ROCm. Support for one AMD product does not imply
support for every AMD runtime or accelerator.

## Support at a Glance

This table describes the Ultralytics Python package used for training, validation, export, and prediction.

| AMD product or runtime                           | Ultralytics support | Usage or status                                                                                                                                    |
| :----------------------------------------------- | :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| AMD Instinct and supported Radeon GPUs with ROCm | ✅                  | Train, validate, and run native PyTorch models with `device=0` or `device=cuda:0`.                                                                 |
| Multi-GPU ROCm                                   | ✅                  | Use `device=0,1` or `device=[0, 1]`; distributed execution follows the installed PyTorch ROCm stack.                                               |
| ROCm Automatic Mixed Precision (AMP)             | ⚠️                  | Available when the installed PyTorch and ROCm versions pass Ultralytics AMP checks; use `amp=False` if incompatible.                               |
| [ONNX export](onnx.md)                           | ✅                  | Export is supported, but the ONNX file does not by itself provide an AMD-accelerated runtime.                                                      |
| MIGraphX inference                               | 🚧                  | Not available in the current Python package; implementation work is tracked in [PR #24137](https://github.com/ultralytics/ultralytics/pull/24137). |
| AMD Docker image and AMD hardware CI             | 🚧                  | Also tracked in [PR #24137](https://github.com/ultralytics/ultralytics/pull/24137), not provided by ROCm device selection alone.                   |
| Windows DirectML                                 | ❌ Python           | No DirectML training or prediction backend in the Python package.                                                                                  |
| Ryzen AI NPU                                     | ❌                  | No native Ultralytics NPU integration; external ONNX/Vitis AI workflows are community-managed.                                                     |
| AMD CPUs                                         | ✅ CPU              | Use `device=cpu`; this is standard CPU execution, not an AMD-specific acceleration backend.                                                        |

!!! note "Check AMD and PyTorch compatibility first"

    ROCm availability depends on the exact GPU, operating system, ROCm version, and PyTorch build. Confirm your hardware
    in AMD's [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
    before installing. Ultralytics cannot add support for a device that the installed PyTorch ROCm build does not expose.

## Why ROCm Uses CUDA Device Names

The ROCm build of PyTorch uses HIP internally but deliberately reuses the `torch.cuda` interfaces. For example,
`torch.cuda.is_available()`, `torch.cuda.device_count()`, and `torch.cuda.get_device_name()` work with supported AMD GPUs.
This design lets the same Ultralytics training path serve NVIDIA CUDA and AMD ROCm without a duplicate backend.

See the official [PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html) for details.

!!! important

    In the Python package, use `device=0` or `device=cuda:0` for PyTorch ROCm. Do not use `device=rocm:0`; `rocm` is not
    a PyTorch device type.

## Install PyTorch ROCm

1. Verify that your operating system and GPU appear in the
   [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
2. Use the [PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose the ROCm build matching
   your installed ROCm version.
3. Install Ultralytics after PyTorch:

    ```bash
    pip install ultralytics
    ```

4. Verify that PyTorch sees the AMD GPU:

    ```python
    import torch

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.version.hip)
    ```

`torch.cuda.is_available()` should return `True`, the device name should identify your AMD GPU, and `torch.version.hip`
should report the HIP version supplied by the ROCm build.

## Train on an AMD GPU

Use the same device arguments as standard [Ultralytics Train mode](../modes/train.md).

!!! example "ROCm Training"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Train on the first AMD GPU exposed by PyTorch ROCm
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=0)

        # Train across two AMD GPUs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # Train on one AMD GPU
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=0

        # Train across two AMD GPUs
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 imgsz=640 device=0,1
        ```

[Validation](../modes/val.md) and [prediction](../modes/predict.md) use the same device selection:

```bash
yolo detect val model=path/to/best.pt data=coco8.yaml device=0
yolo predict model=path/to/best.pt source=path/to/image.jpg device=0
```

## AMP Compatibility

Ultralytics enables AMP by default and compares full-precision and mixed-precision results before training. If the check
detects incompatible results, AMP is disabled to prevent NaN losses or zero-mAP training. ROCm AMP behavior can change
with the PyTorch and ROCm versions, so use `amp=False` when troubleshooting a stack-specific failure:

```bash
yolo detect train data=coco8.yaml model=yolo26n.pt device=0 amp=False
```

## Features Not Yet Supported

### MIGraphX

[MIGraphX](https://github.com/ROCm/AMDMIGraphX) is AMD's graph-optimization and inference runtime. Native MIGraphX model
loading is not included in the current Ultralytics Python package. The active implementation, AMD container,
dependencies, tests, and documentation are tracked together in
[PR #24137](https://github.com/ultralytics/ultralytics/pull/24137). Until that work merges and receives AMD hardware
validation, exporting an ONNX model should not be described as native MIGraphX support in the Python package.

### DirectML

The Ultralytics Python package has no DirectML backend for Windows training or prediction. DirectML is distinct from
ROCm, and a working ROCm environment does not enable `device=directml`.

### Ryzen AI NPU

Ryzen AI NPUs are not exposed through PyTorch ROCm and are not native Ultralytics devices. Community workflows may
export YOLO models to ONNX and run them with AMD's external Ryzen AI or Vitis AI tools, but that runtime, conversion, and
hardware compatibility are outside the supported Ultralytics execution path.

## Troubleshooting

### Why does `torch.cuda.is_available()` return `False` on my AMD system?

The installed PyTorch package may be a CPU or CUDA build, or the GPU may not be supported by the active ROCm stack.
Install the matching ROCm build from the PyTorch selector and verify the GPU against AMD's compatibility matrix.

### Why does Ultralytics say CUDA when I have an AMD GPU?

This is expected. PyTorch ROCm intentionally uses the `torch.cuda` API and CUDA-style device strings for Python
compatibility. The model still executes through HIP and ROCm on the AMD GPU.

### Does ONNX export enable MIGraphX or Ryzen AI automatically?

No. ONNX is a portable model format. Accelerated execution still requires a compatible runtime, and native MIGraphX,
DirectML, and Ryzen AI NPU backends are not included in the current Ultralytics Python package.

## Summary

Use a compatible PyTorch ROCm build with `device=0` or `device=cuda:0` for supported AMD GPU training, validation, and
inference. Treat MIGraphX, DirectML, and Ryzen AI NPU as separate capabilities: none is enabled merely by installing
ROCm or exporting an ONNX model.
