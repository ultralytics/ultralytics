---
comments: true
description: Export Ultralytics YOLO26 models to AXERA NPUs such as the AX650 and AX8860 with format=axera. Covers the Pulsar2 toolchain, INT8 and BF16 precision, calibration, tail.onnx, accuracy and deployment with axengine.
keywords: YOLO26, AXERA, AXERA export, AX650, AX8860, Pulsar2, axnn, axmodel, axengine, pyaxengine, NPU, INT8 quantization, BF16, model deployment, edge AI, tail.onnx
---

# AXERA Export

[AXERA](https://www.axera-tech.com/) NPUs, such as the AX650 and AX8860, run models compiled into the `.axmodel` format by Pulsar2, the AXERA compiler. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="axera"` compiles it with Pulsar2 into a `yolo26n_axera_model/` directory, and Ultralytics runs the result on the device through AXERA's `axengine` runtime.

AXERA is a different company from [Axelera AI](axelera.md), which Ultralytics exports to with `format="axelera"`.

## At a Glance

| Property          | Details                                                                                                                                                          |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `axera`                                                                                                                                                          |
| Output            | `yolo26n_axera_model/`, a directory with `yolo26n.axmodel` and `metadata.yaml`, plus [`tail.onnx`](#output-format) for INT8 detect, segment, pose and OBB models |
| Precision         | INT8 by default; `quantize=16` builds BF16 on the AX8860                                                                                                         |
| Export host       | Linux x86-64 or aarch64 with the `pulsar2` compiler from the `axnn` wheel; no AXERA device is needed                                                             |
| Runtime           | [`axengine`](https://github.com/AXERA-TECH/pyaxengine) on an AXERA board or M.2 card                                                                             |

## Quickstart

Pulsar2 ships as `axnn` wheels for one Python version per release on [Hugging Face](https://huggingface.co/AXERA-TECH/Pulsar2), not on PyPI. Pulsar2 9.0 provides `axnn-9.0-cp313` wheels, so install it in a Python 3.13 environment on Linux, then install Ultralytics:

```bash
pip install https://huggingface.co/AXERA-TECH/Pulsar2/resolve/main/9.0/axnn-9.0-cp313-cp313-linux_x86_64.whl
pip install ultralytics
pulsar2 version # version: 9.0
```

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="axera", name="AX650")  # creates 'yolo26n_axera_model/'
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=axera name=AX650 # creates 'yolo26n_axera_model/'
        ```

The export of YOLO26n took about 45 seconds on a server CPU.

## Supported Tasks

All seven Ultralytics tasks export with [YOLO26](../models/yolo26.md) models: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md) and [OBB](../tasks/obb.md). INT8 detection, segmentation, pose and OBB models write a [`tail.onnx`](#output-format); the other tasks compile the whole graph. [Accuracy](#accuracy) was measured for detection, segmentation and pose; the other tasks were checked on one image each.

## Export Arguments

| Argument   | Type                      | Default      | Description                                                                                                                                                                     |
| ---------- | ------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`                     | `'axera'`    | Target format for the exported model.                                                                                                                                           |
| `name`     | `str`                     | `'AX650'`    | Target chip, passed to Pulsar2 as `--target_hardware`, such as `'AX650'` or `'AX8860'`.                                                                                         |
| `quantize` | `int`                     | `8`          | `8` for INT8, or `16` for BF16 weights and activations on the AX8860.                                                                                                           |
| `imgsz`    | `int` or `tuple`          | `640`        | Image size of the model input. The compiled model has a fixed input shape and a batch size of 1.                                                                                |
| `data`     | `str`                     | task default | Dataset YAML whose `val` split, or the `split` you set, provides the calibration images. Defaults to the calibration dataset of the task, such as `coco128.yaml` for detection. |
| `fraction` | `float`, `int`, or `list` | `1.0`        | Calibration subset: a ratio, an image count, or a `[train, val, test]` list. Pulsar2 calibrates on every image of the subset.                                                   |
| `opset`    | `int`                     | `None`       | ONNX opset of the intermediate graph.                                                                                                                                           |
| `simplify` | `bool`                    | `True`       | Simplify the intermediate ONNX graph.                                                                                                                                           |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Calibration

Pulsar2 sets the INT8 range of each layer from the calibration images, so they should look like the images the model will see in deployment. The 4 images of `coco8.yaml` test the export but are too few to calibrate a model: with them, some YOLO26n class scores were clipped to 0.5. Use at least a few dozen images from your own dataset; the [Accuracy](#accuracy) results used 64.

## Output Format

| File              | Contents                                                                                                                            |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `yolo26n.axmodel` | The compiled model. It takes one `uint8` RGB image of shape `(1, H, W, 3)` in 0-255; the division by 255 is part of the model.      |
| `tail.onnx`       | INT8 detect, segment, pose and OBB models only: the decoding of the head outputs into raw predictions, run on the CPU in `float32`. |
| `metadata.yaml`   | Class names, task, image size, stride and export arguments.                                                                         |

INT8 cannot hold the pixel-range box values and the 0-1 class scores of the decoding in one scale, so an INT8 graph ends at the per-level output convolutions of the head and `tail.onnx` finishes the decoding. BF16 models and the other tasks compile the whole graph. The outputs of `tail.onnx`, or of a whole-graph model, are the same raw predictions as the [ONNX export](onnx.md#output-format), so apply NMS to them. Semantic segmentation is the exception: it returns per-class logits instead of a class map.

## Deploy

Install the `axengine` wheel from the [pyaxengine releases](https://github.com/AXERA-TECH/pyaxengine/releases) on the device, copy the exported directory to it, and use it like any Ultralytics model:

```python
from ultralytics import YOLO

model = YOLO("yolo26n_axera_model")
results = model("https://ultralytics.com/images/bus.jpg")
```

The model is compiled for one NPU core, the Pulsar2 default.

## Accuracy

These results come from the Pulsar2 9.0 simulator running the exported models through `YOLO(...).val()`, with the decoding in `tail.onnx` where the model has one and the standard Ultralytics NMS and metrics. The models were calibrated on 64 COCO val2017 images and validated on other COCO val2017 images:

| Model   | Precision    | Validation images | mAP<sup>val</sup> 50-95 | mAP<sup>val</sup> 50 |
| ------- | ------------ | ----------------- | ----------------------- | -------------------- |
| YOLO26n | PyTorch FP32 | 500               | 0.440                   | 0.607                |
| YOLO26n | INT8, AX650  | 500               | 0.432                   | 0.604                |
| YOLO26n | PyTorch FP32 | 200               | 0.447                   | 0.601                |
| YOLO26n | INT8, AX650  | 200               | 0.428                   | 0.590                |
| YOLO26n | BF16, AX8860 | 200               | 0.440                   | 0.604                |

Segmentation and pose on 200 COCO and COCO-Pose val2017 images, INT8 for the AX650, calibrated on 64 other images of the same set:

| Model        | Precision    | Box mAP<sup>val</sup> 50-95 | Mask or pose mAP<sup>val</sup> 50-95 |
| ------------ | ------------ | --------------------------- | ------------------------------------ |
| YOLO26n-seg  | PyTorch FP32 | 0.453                       | 0.385                                |
| YOLO26n-seg  | INT8, AX650  | 0.414                       | 0.357                                |
| YOLO26n-pose | PyTorch FP32 | 0.663                       | 0.579                                |
| YOLO26n-pose | INT8, AX650  | 0.636                       | 0.536                                |

Latency was not measured: the simulator runs on the CPU. Measure it on the device.

## Limitations and Troubleshooting

- **`pulsar2` not found**: the export fails with `AXERA export requires the Pulsar2 'pulsar2' compiler, which was not found on PATH.` Install the `axnn` wheel as in the [Quickstart](#quickstart).
- **`cannot enable executable stack as shared object requires`**: Pulsar2 9.0 fails this way on glibc 2.41, such as Debian 13. It runs on Debian 12, which has glibc 2.36.
- **BF16 on the AX650**: `quantize=16` with `name="AX650"` fails inside Pulsar2 with a `QuantError`. Use INT8, or `name="AX8860"`.
- **Fixed input**: the model runs at its export `imgsz` with a batch size of 1. Export again to change the image size.
- **`axengine` not installed**: `YOLO("yolo26n_axera_model")` fails with `AXERA inference requires the 'axengine' runtime`. Install it on the device as in [Deploy](#deploy).

## FAQ

### Do I need an AXERA device to export a model?

No. The export is a compile step on a Linux PC. Running the model needs an AXERA board or M.2 card with `axengine`.

### Why does the export need calibration data?

The NPU runs INT8 models, and Pulsar2 sets the INT8 range of each layer from the values it sees on real images. Pass a dataset with images like the ones the model will see in deployment; see [Calibration](#calibration).

### Why does an INT8 model come with `tail.onnx`?

The box decoding works with pixel-range values that one INT8 scale cannot hold together with the class scores, so the INT8 graph stops at the head convolutions and `tail.onnx` decodes their outputs on the CPU in floating point. Ultralytics runs it automatically.
