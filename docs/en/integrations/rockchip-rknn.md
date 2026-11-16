---
comments: true
description: Export Ultralytics YOLO26 models to Rockchip RKNN with format=rknn. Covers the name argument that selects the chip, FP16 and INT8 quantization, output files, tail.onnx, and deployment on Rockchip NPUs.
keywords: YOLO26, RKNN, Rockchip, RKNN export, rknn-toolkit2, RK3588, RK3566, RV1106, NPU, INT8 quantization, FP16, Radxa, edge AI, embedded devices, model deployment
---

# Rockchip RKNN Export

RKNN is the model format of the [NPUs](https://www.ultralytics.com/glossary/neural-processing-unit-npu) in Rockchip processors such as the RK3588 and RK3566. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="rknn"` creates a `yolo26n_rknn_model/` directory that holds a `.rknn` file built for one Rockchip chip, which you select with the `name` argument, in FP16 or INT8 precision.

## At a Glance

| Property          | Details                                                                                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `rknn`                                                                                                                                                                     |
| Output            | `yolo26n_rknn_model/`, a directory with `yolo26n-rk3588.rknn` and `metadata.yaml`; INT8 exports of detect, segment, pose, and OBB models add [`tail.onnx`](#output-format) |
| Runs on           | The NPU of one Rockchip chip per export: RK3588, RK3576, RK3566, RK3568, RK3562, RK2118, RV1126B, RV1103, RV1106, RV1103B, or RV1106B, see [Target Chips](#target-chips)   |
| Runtime           | RKNN Toolkit Lite2 (the `rknn-toolkit-lite2` Python package) on the Rockchip device                                                                                        |
| Precision         | FP16 (default) and INT8; INT8 only on RV1103, RV1106, RV1103B, and RV1106B. There is no FP32 export. YOLO26n for the RK3588 is 7.4 MB in FP16 and 4.3 MB in INT8           |
| Export host       | Linux x86-64 with Python 3.12 or older. Export on a PC and copy the exported directory to the Rockchip device                                                              |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#rknn-chip-support) can export a trained model to RKNN for the Rockchip chip you select, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package in a separate virtual environment on an x86-64 Linux machine, together with the `protobuf` and `numpy` versions that the RKNN toolkit needs:

```bash
pip install ultralytics "protobuf<7" "numpy<2.4"
```

The first export installs `onnx`, `onnxslim`, `onnxruntime`, and [`rknn-toolkit2`](https://github.com/airockchip/rknn-toolkit2) with its dependencies if they are missing. With a plain `pip install ultralytics`, the first export also replaces `numpy`, `protobuf`, and `setuptools` when their versions are outside the ranges below, so use a separate environment. When it has to replace `protobuf`, that export [fails](#limitations-and-troubleshooting) and the second one passes. The export has these version requirements:

| Package         | Required version                                                                     |
| --------------- | ------------------------------------------------------------------------------------ |
| Python          | 3.12 or older, the newest version that `rknn-toolkit2` 2.3.2 has builds for          |
| `rknn-toolkit2` | 2.3.2, the only release on PyPI in the required range (2.3.2 or newer and below 2.5) |
| `onnx`          | 1.16.1 or newer and below 1.19.0                                                     |
| `numpy`         | below 2.4                                                                            |
| `protobuf`      | below 7                                                                              |
| `setuptools`    | below 82                                                                             |

For other installation methods, see the [installation guide](../quickstart.md). Then export your model on the PC, copy the exported directory to the Rockchip device, and load it there to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to RKNN format for the RK3588 in FP16
        model.export(format="rknn", name="rk3588")  # creates 'yolo26n_rknn_model/'

        # Or export an INT8-quantized RKNN model with calibration data instead
        model.export(format="rknn", name="rk3588", quantize=8, data="coco8.yaml")  # creates 'yolo26n_rknn_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to RKNN format for the RK3588 in FP16
        yolo export model=yolo26n.pt format=rknn name=rk3588 # creates 'yolo26n_rknn_model/'

        # Or export an INT8-quantized RKNN model with calibration data instead
        yolo export model=yolo26n.pt format=rknn name=rk3588 quantize=8 data=coco8.yaml # creates 'yolo26n_rknn_model/'
        ```

Both exports write the same `yolo26n_rknn_model/` directory, so the second one replaces the first. Predict and Validate run only on a Rockchip device:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported RKNN model on the Rockchip device
        model = YOLO("yolo26n_rknn_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported RKNN model on the Rockchip device
        yolo predict model=yolo26n_rknn_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported RKNN model on the Rockchip device
        model = YOLO("yolo26n_rknn_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported RKNN model on the Rockchip device
        yolo val model=yolo26n_rknn_model data=coco8.yaml
        ```

## Supported Tasks

RKNN export supports all seven Ultralytics tasks with [YOLOv8](../models/yolov8.md), [YOLO11](../models/yolo11.md), and [YOLO26](../models/yolo26.md) models: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

## Export Arguments

| Argument   | Type                      | Default    | Description                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------- | ------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`                     | `'rknn'`   | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                               |
| `name`     | `str`                     | `'rk3588'` | Rockchip chip the model is built for, in upper or lower case. See [Target Chips](#target-chips).                                                                                                                                                                                                                                                                                                                 |
| `imgsz`    | `int` or `tuple`          | `640`      | Image size of the model input: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect, segment, and pose models and for semantic models pretrained on ADE20K, `1024` for OBB models and for semantic models pretrained on Cityscapes, `768` for depth, and `224` for classify. |
| `quantize` | `int` or `str`            | `None`     | Precision: `16` or unset builds FP16, and `8` builds INT8 calibrated on `data`. For the INT8-only chips, unset builds INT8 and `16` raises an error. `32` raises an error, and a checkpoint trained with `quantize=8` cannot be exported to RKNN, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). Replaces the deprecated `half`/`int8` flags.                                          |
| `data`     | `str`                     | `None`     | Dataset YAML used for INT8 calibration; classification instead takes a dataset directory or a built-in dataset name. Calibration uses the `val` split unless `split` is set. If omitted with `quantize=8`, Ultralytics uses the small default dataset for the model task, such as `coco8.yaml` for detection.                                                                                                    |
| `fraction` | `float`, `int`, or `list` | `1.0`      | Calibration subset: a ratio, an image count (integers above `1`), or a `[train, val, test]` list, from which the entry of the calibration split is used.                                                                                                                                                                                                                                                         |
| `batch`    | `int`                     | `1`        | Batch size of the exported model, fixed in the input shape.                                                                                                                                                                                                                                                                                                                                                      |
| `simplify` | `bool`                    | `True`     | Simplifies the intermediate ONNX graph with `onnxslim`.                                                                                                                                                                                                                                                                                                                                                          |
| `opset`    | `int`                     | `None`     | ONNX opset version of the intermediate ONNX graph. Uses `19` when unset, and reduces higher values to `19`.                                                                                                                                                                                                                                                                                                      |
| `nms`      | `bool`, optional          | `None`     | RKNN models always return raw predictions for external NMS. The NMS-free head (`False`) and embedded NMS (`True`) are not available, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).                                                                                                                                                                                                    |
| `device`   | `str`                     | `None`     | Device that traces the intermediate ONNX graph: `cpu` (the default) or `0` for a CUDA GPU.                                                                                                                                                                                                                                                                                                                       |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Target Chips

A `.rknn` file is built for one chip, and `name` selects it. The sizes are for a YOLO26n detection model exported at `imgsz=640` with `quantize` unset:

| `name`    | Precisions | Unset `quantize` builds | YOLO26n file           | Size    |
| --------- | ---------- | ----------------------- | ---------------------- | ------- |
| `rk3588`  | FP16, INT8 | FP16                    | `yolo26n-rk3588.rknn`  | 7.4 MB  |
| `rk3576`  | FP16, INT8 | FP16                    | `yolo26n-rk3576.rknn`  | 10.5 MB |
| `rk3566`  | FP16, INT8 | FP16                    | `yolo26n-rk3566.rknn`  | 6.8 MB  |
| `rk3568`  | FP16, INT8 | FP16                    | `yolo26n-rk3568.rknn`  | 6.8 MB  |
| `rk3562`  | FP16, INT8 | FP16                    | `yolo26n-rk3562.rknn`  | 6.8 MB  |
| `rk2118`  | FP16, INT8 | FP16                    | `yolo26n-rk2118.rknn`  | 6.8 MB  |
| `rv1126b` | FP16, INT8 | FP16                    | `yolo26n-rv1126b.rknn` | 7.1 MB  |
| `rv1103`  | INT8       | INT8                    | `yolo26n-rv1103.rknn`  | 4.2 MB  |
| `rv1106`  | INT8       | INT8                    | `yolo26n-rv1106.rknn`  | 4.2 MB  |
| `rv1103b` | INT8       | INT8                    | `yolo26n-rv1103b.rknn` | 4.2 MB  |
| `rv1106b` | INT8       | INT8                    | `yolo26n-rv1106b.rknn` | 4.2 MB  |

The four INT8-only chips are calibrated on the default `coco8.yaml` in this table, and their exports also write [`tail.onnx`](#output-format).

### Quantization

`quantize` selects the precision of the `.rknn` file. The sizes are for an export at `imgsz=640` and `batch=1`:

| `quantize`    | Precision | Calibration data | Files for YOLO26n and `rk3588`        | Size              |
| ------------- | --------- | ---------------- | ------------------------------------- | ----------------- |
| unset or `16` | FP16      | No               | `yolo26n-rk3588.rknn`                 | 7.4 MB            |
| `8`           | INT8      | Yes              | `yolo26n-rk3588.rknn` and `tail.onnx` | 4.3 MB and 0.2 MB |

- **FP16**: the default for the seven chips that support it. It needs no calibration data.
- **INT8**: `quantize=8` runs calibration images through the model and stores weights and activations as 8-bit integers. The export calibrates on the `val` split of `data`, or on the split you set with `split`, and warns below 300 images. Use `fraction` to limit the number of images.
- **Mixed precision**: on the seven chips that support FP16, an INT8 export lets RKNN Toolkit keep layers in FP16 when quantization changes their output too much (`auto_hybrid`). The four INT8-only chips skip this step.
- **Detection head**: for detect, segment, pose, and OBB models, the INT8 `.rknn` file ends at the output convolutions of the head. The operations that decode those outputs into boxes are saved as [`tail.onnx`](#output-format) and run on the CPU of the device in floating point, because per-tensor INT8 cannot hold the pixel-range values of the box decoding.
- **Depth models**: depth INT8 exports calibrate on the images of a depth dataset that have paired depth maps. Choose calibration images that span the near and far depth range of the deployment scene. With the default `depth8.yaml`, which has 4 validation images, a YOLO26n-depth INT8 model in the rknn-toolkit2 simulator returned depths up to 3.9 for a street image on which the FP16 model returned up to 13.9.
- **On-device status**: INT8 exports of every task were run in the rknn-toolkit2 simulator and compared with the ONNX model on a single image. Their accuracy has not been measured on a dataset or on Rockchip hardware, so validate an INT8 model on your device and your own data before you deploy it.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        model.export(format="rknn", name="rk3588", quantize=8, data="coco.yaml", fraction=0.1)
        ```

    === "CLI"

        ```bash
        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        yolo export model=yolo26n.pt format=rknn name=rk3588 quantize=8 data=coco.yaml fraction=0.1
        ```

## Output Format

The export writes one directory:

| File                  | Contents                                                                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `yolo26n-rk3588.rknn` | The model, with the chip set by `name` in the file name                                                                                  |
| `tail.onnx`           | Only for INT8 detect, segment, pose, and OBB models: the operations that decode the head outputs into raw predictions, to run on the CPU |
| `metadata.yaml`       | Class names, task, image size, batch size, stride, and export arguments, which Ultralytics reads when it loads the model                 |

The intermediate `yolo26n.onnx` file stays next to the directory. The `.rknn` model takes one input with shape `(batch, height, width, 3)`: RGB, channels last (NHWC), `uint8` with values from 0 to 255. The division by 255 is part of the model, so do not scale the image yourself. This differs from the [ONNX export](onnx.md#output-format), which takes channels-first `float32` images scaled to 0-1.

The outputs depend on the task and the precision. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | FP16 output, and `tail.onnx` output for INT8            | INT8 `.rknn` output                                            |
| -------- | --------------- | ------------------------------------------------------- | -------------------------------------------------------------- |
| Detect   | `640`           | `(1, 84, 8400)`                                         | 6 head tensors, decoded by `tail.onnx`                         |
| Segment  | `640`           | `(1, 116, 8400)` and `(1, 32, 160, 160)`                | 9 head tensors and `(1, 32, 160, 160)`, decoded by `tail.onnx` |
| Semantic | `1024`          | `(1, 19, 1024, 1024)`, a score for each class and pixel | Same as FP16                                                   |
| Depth    | `768`           | `(1, 1, 768, 768)`, depth map                           | Same as FP16                                                   |
| Classify | `224`           | `(1, 1000)`, class probabilities                        | Same as FP16                                                   |
| Pose     | `640`           | `(1, 56, 8400)`                                         | 9 head tensors, decoded by `tail.onnx`                         |
| OBB      | `1024`          | `(1, 20, 21504)`                                        | 9 head tensors, decoded by `tail.onnx`                         |

- **Raw predictions**: FP16 models, and `tail.onnx` for INT8 models, return the same raw predictions as a default ONNX export: for each anchor the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS yourself. See the [ONNX output format](onnx.md#output-format) for the layout of each task.
- **INT8 head tensors**: an INT8 detection model returns two tensors for each of the three detection scales, one with box values and one with class scores, and segmentation, pose, and OBB models return one more tensor per scale with their extra values. Feed them to `tail.onnx` in the order the model returns them; a segmentation model also passes its mask prototypes through `tail.onnx` unchanged.
- **Semantic segmentation**: the output holds a score for each class and pixel, not the class map of the ONNX export, so take the argmax over the class axis in your own code.
- **NMS**: there is no NMS-free or embedded-NMS output, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

`YOLO("yolo26n_rknn_model")` reads the layout from `metadata.yaml`, runs `tail.onnx`, and applies NMS, so you only need these details when you run the model in your own code. The keys of `metadata.yaml` include `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, `args`, and, for pose models, `kpt_shape`.

### Run with RKNN Toolkit Lite2

To run the model on the device outside the Ultralytics package, load it with the `rknn-toolkit-lite2` package and, for an INT8 detect, segment, pose, or OBB model, pass its outputs to `tail.onnx` with [ONNX Runtime](https://onnxruntime.ai/):

```python
# Export first: yolo export model=yolo26n.pt format=rknn name=rk3588 quantize=8 data=coco8.yaml
import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite

rknn = RKNNLite()
assert rknn.load_rknn("yolo26n_rknn_model/yolo26n-rk3588.rknn") == 0
assert rknn.init_runtime() == 0

image = np.zeros((1, 640, 640, 3), dtype=np.uint8)  # your letterboxed image: RGB, NHWC, 0-255
outputs = rknn.inference(inputs=[image])  # six head tensors for an INT8 detection model

tail = ort.InferenceSession("yolo26n_rknn_model/tail.onnx", providers=["CPUExecutionProvider"])
feed = {t.name: x.astype(np.float32) for t, x in zip(tail.get_inputs(), outputs)}
predictions = tail.run(None, feed)[0]  # shape (1, 84, 8400), raw predictions that need NMS
assert predictions.shape == (1, 84, 8400), "export a detection model at imgsz=640 with quantize=8"
```

`tail.onnx` has a batch size of 1 whatever the export `batch`, so run it once per image for a model exported with a larger batch. An FP16 model has no `tail.onnx`, and `outputs[0]` already holds the raw predictions.

## Deploy

On the Rockchip device, install the package with `pip install ultralytics` and copy the `yolo26n_rknn_model/` directory to it; the `protobuf` and `numpy` versions from the Quickstart are needed only for the export. `YOLO("yolo26n_rknn_model")` then runs the model on the NPU with the `rknn-toolkit-lite2` Python package, which Ultralytics installs on first use together with `onnxruntime` when the directory holds a `tail.onnx`. The device needs Python 3.12 or older, the newest version that `rknn-toolkit-lite2` has builds for.

- **Rockchip device required**: Ultralytics loads an RKNN model only on 64-bit ARM Linux systems whose device tree `compatible` string ends with one of the [supported chip names](#target-chips). Everywhere else, loading fails, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). `rknn-toolkit-lite2` also has builds for 64-bit ARM Linux only, so on a target that does not meet these conditions, run the `.rknn` file with Rockchip's C API and either run `tail.onnx` or implement the same decoding in your own code.
- **Matching chip**: export with the `name` of the chip on your board.
- **Fixed input shape**: the model always runs at its export `imgsz` and `batch`, which Ultralytics reads from `metadata.yaml`. An `imgsz` passed to `predict` or `val` is ignored. Larger batches are split into chunks of the export batch size and smaller ones are padded, so any number of images works.
- **`device`**: there is no device to select. The `.rknn` model runs through the RKNN runtime, and `tail.onnx` runs with ONNX Runtime on the CPU.

An earlier version of this workflow was tested on a [Radxa Rock 5B](https://radxa.com/products/rock5/5b/) (RK3588) and a [Radxa Zero 3W](https://radxa.com/products/zeros/zero3w/) (RK3566). The current INT8 export has not yet been run on hardware.

### Deployment Guides

- **Board setup**: flash an operating system first, as described in the getting started guides for the [Radxa Rock 5B](https://docs.radxa.com/en/rock5/rock5b) and the [Radxa Zero 3W](https://docs.radxa.com/en/zero/zero3).
- **[RKNN Toolkit2 on GitHub](https://github.com/airockchip/rknn-toolkit2)**: the conversion toolkit, the RKNN Toolkit Lite2 runtime, and the `librknnrt.so` runtime library with its C API.
- **[RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)**: Rockchip's C and Python deployment examples.
- **[Deploy Ultralytics YOLO on Rockchip](https://www.ultralytics.com/blog/deploy-ultralytics-yolo11-on-rockship-for-efficient-edge-ai)**: an overview of the workflow on the Ultralytics blog.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare RKNN with the other export formats.

## Limitations and Troubleshooting

- **Export host**: export on an x86-64 Linux machine. On macOS the export cannot install `rknn-toolkit2` and fails with `No module named 'rknn'`; the package has no macOS or Windows builds. Export on ARM64 Linux, including Rockchip boards, is untested and not supported: Ultralytics tests the RKNN export only on x86-64 Linux.
- **Python version**: `rknn-toolkit2` 2.3.2 has builds for Python 3.12 and older. On a newer Python, create the environment with Python 3.12.
- **First export in a new environment**: in an environment without the `protobuf` version from the Quickstart, the first export replaces `protobuf` 7, which pip installs together with `onnx`, with an older version during the run and fails with `AttributeError: 'google._upb._message.FieldDescriptor' object has no attribute 'label'`. Run the export again, or install `"protobuf<7" "numpy<2.4"` before the first export as in the [Quickstart](#quickstart).
- **RKNN models do not run on a PC**: predicting or validating with `YOLO("yolo26n_rknn_model")` fails with `RKNN inference is only supported on Rockchip devices.` on a PC, and on a board whose `/proc/device-tree/compatible` string does not end with one of the [supported chip names](#target-chips). Print it with `cat /proc/device-tree/compatible`.
- **FP32**: `quantize=32` fails with `quantize=32 (FP32) is not supported; format='rknn' supports quantize=16 (FP16), 8 (INT8).`
- **INT8-only chips**: for `rv1103`, `rv1106`, `rv1103b`, and `rv1106b`, an export without `quantize` logs `Rockchip target 'rv1106' requires INT8 quantization, enabling it.` and builds an INT8 model, and `quantize=16` fails with `Rockchip target 'rv1106' only supports INT8, but got quantize=16.`
- **Missing or wrong `name`**: without `name`, the export logs `Rockchip RKNN export requires a missing 'name' arg for processor type. Using default name='rk3588'.` An unknown value fails with `Invalid processor name 'rk9999' for Rockchip RKNN export.` followed by the valid names.
- **NMS-free head**: RKNN models always return raw predictions that need NMS. With `nms=False`, the export of a YOLO26 or YOLOv10 model logs `This export format does not support end2end models, disabling the end2end branch.` and uses the one-to-many head; for other detection, segmentation, pose, and OBB models, for example YOLO11, it logs `This model has no one-to-one head; using one-to-many outputs.` Classification, semantic segmentation, and depth models ignore `nms=False`.
- **Embedded NMS**: `nms=True` logs `format=rknn does not support embedded NMS; exporting native outputs for external NMS.` and writes the raw output.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='rknn' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends`. Export the checkpoint you started the fine-tuning from with `quantize=8` instead.
- **RT-DETR**: with `rknn-toolkit2` 2.3.2, the export of an [RT-DETR](../models/rtdetr.md) model fails inside the toolkit in FP16 and in INT8, for `rtdetr-l.pt` at `imgsz=640` with `ValueError: operands could not be broadcast together with shapes (256,256) (8400,1)`.
- **Fixed input shape**: `dynamic=True` fails with `argument 'dynamic' is not supported for format='rknn'`. Export again to change the image size or the batch size.
- **Calibration without `data`**: a `quantize=8` export of a detection model logs `INT8 export requires a missing 'data' arg for calibration. Using default 'data=coco8.yaml'.` and calibrates on that small dataset; the other tasks name their own default dataset. Pass `data` with images that represent your deployment data.
- **Calibration size**: the export logs `>300 images recommended for INT8 calibration, found 4 images.` when the calibration split has fewer than 300 images. `coco8.yaml` has only 4 validation images, which is enough to test the export but not to calibrate a model for deployment.
- **Output directory**: every export of one checkpoint writes the same `yolo26n_rknn_model/` directory, whatever its `name`, `quantize`, `imgsz`, or `batch`, and deletes the previous contents first. Rename the directory before you export for a second chip or precision, and keep `_rknn_model` in its name, for example `yolo26n_rk3566_rknn_model/`, because Ultralytics recognizes the format by it.
- **ONNX opset**: an `opset` above 19 logs `rknn-toolkit2 requires opset<=19, setting opset=19.`
- **INT8 models from earlier versions**: re-export INT8 detection, segmentation, pose, and OBB models that were exported before `ultralytics` 8.4.191, the release that added `tail.onnx`. Those models return box and keypoint coordinates relative to the input size, which current releases no longer rescale, so they load without an error and return wrong boxes.
- **Runtime version mismatch**: if the log on the device reports that the RKNN runtime version does not match the RKNN Toolkit version and inference fails, replace `/usr/lib/librknnrt.so` with the official [`librknnrt.so`](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so) file.

    ![RKNN log warning that the model toolkit version does not match the librknnrt runtime version](https://cdn.ul.run/i/90077c647a22b3a29612d2701c6f2108.avif)

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

YOLO26 benchmarks below were run by the Ultralytics team on Radxa Rock 5B based on Rockchip RK3588 with `rknn` model format measuring speed and accuracy.

!!! tip "Performance"

    | Model   | Format | Precision | Status | Size (MB) | mAP<sub>50-95</sub>(B) | Inference time (ms/im) |
    | ------- | ------ | --------- | ------ | --------- | ---------------------- | ---------------------- |
    | YOLO26n | `rknn` | FP16      | ✅     | 7.2       | 0.477                  | 58.5                   |
    | YOLO26n | `rknn` | INT8      | ✅     | 4.1       | 0.463                  | 41.2                   |
    | YOLO26s | `rknn` | FP16      | ✅     | 21.0      | 0.569                  | 93.8                   |
    | YOLO26s | `rknn` | INT8      | ✅     | 12.0      | 0.554                  | 61.9                   |
    | YOLO26m | `rknn` | FP16      | ✅     | 43.0      | 0.608                  | 226.3                  |
    | YOLO26m | `rknn` | INT8      | ✅     | 22.0      | 0.603                  | 122.0                  |
    | YOLO26l | `rknn` | FP16      | ✅     | 53.0      | 0.628                  | 264.7                  |
    | YOLO26l | `rknn` | INT8      | ✅     | 28.0      | 0.617                  | 145.3                  |
    | YOLO26x | `rknn` | FP16      | ✅     | 113.0     | 0.664                  | 655.4                  |
    | YOLO26x | `rknn` | INT8      | ✅     | 58.0      | 0.649                  | 287.4                  |

    Benchmarked with `ultralytics 8.4.60`

    !!! note

        Validation for the above benchmarks was done using the [COCO128](https://platform.ultralytics.com/ultralytics/datasets/coco128) dataset. Inference time does not include pre/post-processing. The INT8 rows will be refreshed once the current INT8 export path (host-side decode and `auto_hybrid`) is validated on Rockchip hardware.

The sizes in the table were recorded with that earlier version; the sizes elsewhere on this page are the `.rknn` file of a current export in MB (10^6 bytes). The [YOLO26 model page](../models/yolo26.md) lists the accuracy of the PyTorch checkpoints on the full COCO validation set.

## About RKNN

[Rockchip](https://www.rock-chips.com/) designs ARM-based systems-on-chip with built-in NPUs, used in single-board computers and embedded products from companies such as Radxa, ASUS, Pine64, Orange Pi, Odroid, Khadas, and Banana Pi. RKNN, short for Rockchip Neural Network, is the model format of those NPUs, and the [RKNN Toolkit](https://github.com/airockchip/rknn-toolkit2) converts models to it on a PC and runs them on the device.

<p align="center">
  <img width="50%" src="https://cdn.ul.run/i/3c3fcbb470dabcabbe1b5c926fee152a.avif" alt="Rockchip RKNN export for NPU deployment">
</p>

## FAQ

### Which `name` should I use for my Rockchip board?

Use the name of the processor on the board, not the name of the board: `rk3588` for a Radxa Rock 5B, for example, or `rk3566` for a Radxa Zero 3W. On the board, `cat /proc/device-tree/compatible` prints the string that Ultralytics reads the chip from. A model exported for one chip is not built for another, so export once per chip. See [Target Chips](#target-chips) for the accepted names.

### Should I export an RKNN model in FP16 or INT8?

Start with FP16, the default. It needs no calibration data and returns its raw predictions without a second model on the CPU. Choose INT8 (`quantize=8`) when you need a smaller and faster model or when the chip is INT8-only: YOLO26n for the RK3588 is 4.3 MB in INT8 against 7.4 MB in FP16, and in the [benchmarks](#benchmarks), which were measured with an earlier export, INT8 ran YOLO26n at 41.2 ms per image against 58.5 ms in FP16. INT8 needs representative [calibration data](#quantization), and for detect, segment, pose, and OBB models it runs the box decoding on the CPU with `tail.onnx`. Measure latency and accuracy on your device before you choose.

### Can I test an RKNN model on my PC before I copy it to the device?

Not with `YOLO()`, which loads RKNN models only on Rockchip devices. To check the model itself on the PC, validate the PyTorch checkpoint or the intermediate `yolo26n.onnx` file that the export leaves next to the directory, with [ONNX Runtime](onnx.md): the RKNN model is converted from that graph. This does not replace a validation on the device, in particular for INT8, because quantization changes the outputs.

### Can I run a YOLO26 model on other edge accelerators?

Yes, with the export format of each accelerator: an RKNN model runs only on Rockchip NPUs. See [TensorRT](tensorrt.md) for NVIDIA Jetson, [Hailo](hailo.md), [Axelera](axelera.md), [DeepX](deepx.md), and [Coral Edge TPU](edge-tpu.md), or the [model deployment options guide](../guides/model-deployment-options.md) for an overview.
