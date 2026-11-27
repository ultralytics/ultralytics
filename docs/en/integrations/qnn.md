---
comments: true
description: Export Ultralytics YOLO26 models to Qualcomm QNN with format=qnn for the Snapdragon Hexagon NPU. Covers onnxruntime-qnn setup, HTP targets, W8A16 calibration, and deployment.
keywords: YOLO26, Qualcomm QNN, QNN export, Snapdragon, Hexagon NPU, HTP, onnxruntime-qnn, ONNX Runtime QNN Execution Provider, EPContext, context binary, W8A16, Dragonwing IQ-8275, Windows on Snapdragon, Qualcomm AI Engine Direct, QAIRT, edge AI
---

# Qualcomm QNN Export

[Qualcomm AI Engine Direct](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk), also called QNN, is Qualcomm's runtime for the Hexagon NPU of Snapdragon and Dragonwing processors. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="qnn"` creates `yolo26n_qnn.onnx`, an ONNX model whose only node holds a QNN context binary, compiled with INT8 weights and 16-bit activations for the Hexagon Tensor Processor (HTP) generation that you select with the `name` argument. The export runs with [ONNX Runtime](onnx.md) on your own computer and needs no Qualcomm account or device.

## At a Glance

| Property          | Details                                                                                                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `format` argument | `qnn`                                                                                                                                                                                                                          |
| Output            | `yolo26n_qnn.onnx`, one ONNX file with an `EPContext` node that embeds the compiled QNN context binary, and the class names, task, and image size as [metadata](#output-format)                                                |
| Runs on           | The Hexagon NPU of the HTP generation set with `name`: Snapdragon phones from the Snapdragon 888 to the 8 Elite Gen 5, Snapdragon X Elite laptops, and Dragonwing IQ-8275, see [Supported HTP Targets](#supported-htp-targets) |
| Runtime           | ONNX Runtime with the QNN Execution Provider: the `onnxruntime-qnn` Python package on Windows and Linux, or ONNX Runtime with QNN in an Android app                                                                            |
| Precision         | W8A16 only: INT8 weights and 16-bit activations, calibrated on images from `data`. YOLO26n is 3.6 MB                                                                                                                           |
| Export host       | Linux x86-64 or ARM64, or Windows x64 or ARM64, with Python 3.11 or newer. Not macOS. Tested on x86-64 Ubuntu 24.04. The export runs on the CPU and needs no GPU and no Snapdragon device                                      |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to QNN for the HTP target you select, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package with the ONNX export dependencies and the QNN Execution Provider:

```bash
pip install "ultralytics[export-base]" onnxruntime-qnn
```

`onnxruntime-qnn` brings the Qualcomm AI Runtime (QAIRT) libraries and installs next to `onnxruntime`, which it requires. With a plain `pip install ultralytics`, the first export installs `onnx`, `onnxslim`, `onnxruntime`, and `onnxruntime-qnn` itself, logs `Restart runtime or rerun command for updates to take effect`, and still completes in the same run. The export has these version requirements:

| Requirement       | Version                                                                                                                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python            | 3.11 to 3.14, the versions with `onnxruntime-qnn` 2.x packages. Exports ran with Python 3.11, 3.12, and 3.14                                                                                        |
| Operating system  | Linux x86-64 with glibc 2.35 or newer, Linux ARM64 with glibc 2.34 or newer, Windows x64 or ARM64. `onnxruntime-qnn` has no packages for macOS, see [Limitations](#limitations-and-troubleshooting) |
| `onnxruntime-qnn` | 2.8.0 for every export on this page, with `onnxruntime` 1.32.0. Ultralytics does not pin a version                                                                                                  |

For other installation methods, see the [installation guide](../quickstart.md). Then export your model:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to QNN format for the Snapdragon 8 Elite Gen 5 (HTP v81)
        model.export(format="qnn", name="81", imgsz=640)  # creates 'yolo26n_qnn.onnx'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to QNN format for the Snapdragon 8 Elite Gen 5 (HTP v81)
        yolo export model=yolo26n.pt format=qnn name=81 imgsz=640 # creates 'yolo26n_qnn.onnx'
        ```

The export writes `yolo26n_qnn.onnx` next to `yolo26n.pt` and fails if that file already exists, so delete or rename it before you export again. Without `data`, it calibrates on the small default dataset of the task, such as the 4 images of `coco8.yaml`, which is enough to test the export; calibrate on images from your deployment data before you ship the model, see [Calibration](#calibration).

Then copy `yolo26n_qnn.onnx` to the Snapdragon device, see [Deploy](#deploy). Predict and Validate also run, slowly, on the x86-64 Linux export host, so you can check the exported file before you copy it. Run each in a new Python process, because a second QNN model in one process fails, see [Limitations](#limitations-and-troubleshooting):

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported QNN model
        model = YOLO("yolo26n_qnn.onnx")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported QNN model
        yolo predict model=yolo26n_qnn.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported QNN model
        model = YOLO("yolo26n_qnn.onnx")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported QNN model
        yolo val model=yolo26n_qnn.onnx data=coco8.yaml
        ```

## Supported Tasks

QNN export supports all seven Ultralytics tasks: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with [YOLO26](../models/yolo26.md), the only family that ships those heads. The YOLO26n checkpoint of every task exported and ran on the export host, and a [YOLO11](../models/yolo11.md) detection model did too. Other model families were not tested.

## Export Arguments

| Argument   | Type                      | Default | Description                                                                                                                                                                                                                                                              |
| ---------- | ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `format`   | `str`                     | `'qnn'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                       |
| `name`     | `str`                     | `'73'`  | HTP generation or SoC the context binary is compiled for, such as `'81'` or `'iq-8275'`. See [Supported HTP Targets](#supported-htp-targets).                                                                                                                            |
| `imgsz`    | `int` or `tuple`          | `640`   | Image size of the model input, fixed in the context binary: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at, such as `1024` for OBB and `224` for classification.                     |
| `quantize` | `int` or `str`            | `None`  | Precision. Unset and `'w8a16'` build the only precision, INT8 weights and 16-bit activations; `8`, `16`, and `32` raise an error, see [Precision](#precision). Replaces the deprecated `half`/`int8` flags.                                                              |
| `data`     | `str`                     | `None`  | Dataset YAML with the calibration images; classification instead takes a dataset directory or a built-in dataset name. Calibration uses the `val` split unless `split` is set. If omitted, Ultralytics uses the small default dataset of the task, such as `coco8.yaml`. |
| `fraction` | `float`, `int`, or `list` | `1.0`   | Calibration subset: a ratio, an image count (integers above `1`), or a `[train, val, test]` list, from which the entry of the calibration split is used.                                                                                                                 |
| `batch`    | `int`                     | `1`     | Batch size of the exported model, fixed in the input shape. Also the batch size used for calibration.                                                                                                                                                                    |
| `nms`      | `bool`, optional          | `None`  | Has no effect: every value writes the raw output of `None`, see [Output Format](#output-format).                                                                                                                                                                         |
| `opset`    | `int`                     | `None`  | ONNX opset version of the intermediate ONNX graph. When unset, the highest opset of your PyTorch version up to `18`, which is `18` with PyTorch 2.4 or newer. The ONNX Runtime quantizer raises it to `21` for the 16-bit activations.                                   |
| `simplify` | `bool`                    | `True`  | Simplifies the intermediate ONNX graph with `onnxslim`.                                                                                                                                                                                                                  |
| `device`   | `str`                     | `None`  | Device that traces the intermediate ONNX graph, `cpu` by default.                                                                                                                                                                                                        |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

!!! note "Image size of the official mobile models"

    The official QNN, [LiteRT](litert.md), and [Core ML](coreml.md) mobile models are exported at `imgsz=640`, and at `imgsz=224` for classification. The Cityscapes semantic, depth, and OBB checkpoints export at a larger size by default, so pass `imgsz=640` to match them. The QNN models of all seven tasks in every size, for HTP v73 and v81, are in the [yolo-flutter-app `v0.7.0` release](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.7.0).

### Supported HTP Targets

The context binary is compiled for one HTP generation, so export with the `name` of your device. The Flutter plugin README lists the Snapdragon 8 Gen 2 or newer as the requirement for the official `_v73` models; other combinations of export and device generation were not tested. `name` accepts the generation with or without a leading `v`, such as `81`, `v81`, or `V81`, or one of the two SoC names, in any case. The sizes are for YOLO26n detection at `imgsz=640`:

| `name`         | Hexagon HTP | Example devices                        | ONNX Runtime option | YOLO26n file |
| -------------- | ----------- | -------------------------------------- | ------------------- | ------------ |
| `68`           | v68         | Snapdragon 888                         | `htp_arch=68`       | 3.7 MB       |
| `69`           | v69         | Snapdragon 8 Gen 1                     | `htp_arch=69`       | 3.8 MB       |
| `73` (default) | v73         | Snapdragon 8 Gen 2, Snapdragon X Elite | `htp_arch=73`       | 3.6 MB       |
| `75`           | v75         | Snapdragon 8 Gen 3                     | `htp_arch=75`       | 3.6 MB       |
| `79`           | v79         | Snapdragon 8 Elite (SM8750)            | `soc_model=69`      | 3.6 MB       |
| `81`           | v81         | Snapdragon 8 Elite Gen 5               | `htp_arch=81`       | 3.6 MB       |
| `iq-8275`      | v75         | Dragonwing IQ-8275                     | `soc_model=82`      | 3.5 MB       |
| `qcs8275`      | v75         | Dragonwing QCS8275                     | `soc_model=82`      | 3.5 MB       |

The Snapdragon devices are examples of each generation, not a complete list. Without `name`, the export logs `Qualcomm QNN export requires a missing 'name' arg for the target Hexagon HTP architecture. Using default name='73' (Snapdragon 8 Gen 2).` Any other value fails with `Invalid Qualcomm QNN target '66'. Valid targets are ('68', '69', '73', '75', '79', '81', 'iq-8275', 'qcs8275').`. The Dragonwing IQ-615 (QCS615) has a Hexagon v66 DSP, which has no QNN target in Ultralytics; export a standard [ONNX](onnx.md) model for its CPU or GPU instead.

!!! example "Export for Dragonwing IQ-8275"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="qnn", name="iq-8275", imgsz=640)  # creates 'yolo26n_qnn.onnx'
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=qnn name=iq-8275 imgsz=640 # creates 'yolo26n_qnn.onnx'
        ```

### Precision

Every QNN export is quantized with ONNX Runtime's QNN QDQ flow to INT8 weights and 16-bit activations (W8A16). There is no FP32 or FP16 export, and no INT8 export with 8-bit activations:

| `quantize`         | Result                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| unset or `'w8a16'` | W8A16 context binary. YOLO26n is 3.6 MB. Without `quantize`, the export logs `qnn export requires INT8 quantization, enabling it.` |
| `8`                | Fails with `quantize=8 (INT8) is not supported; format='qnn' supports quantize='w8a16' (INT8 weights + INT16 activations).`        |
| `16`               | Fails with `quantize=16 (FP16) is not supported; format='qnn' supports quantize='w8a16' (INT8 weights + INT16 activations).`       |
| `32`               | Fails with `quantize=32 (FP32) is not supported; format='qnn' supports quantize='w8a16' (INT8 weights + INT16 activations).`       |

The input and the outputs stay `float32`, except the semantic class map, because the quantization of the input and the dequantization of the outputs run inside the context binary.

### Calibration

The export calibrates the activation ranges on every full batch of the `val` split of `data`, or of the split you set with `split`, and warns `>300 images recommended for INT8 calibration, found 4 images.` below 300 images. Pass `data` with images that look like your deployment data, and use `fraction` to limit their number. On first use, `coco.yaml` downloads the whole COCO dataset, about 20 GB:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        model.export(format="qnn", name="81", imgsz=640, data="coco.yaml", fraction=0.1)
        ```

    === "CLI"

        ```bash
        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        yolo export model=yolo26n.pt format=qnn name=81 imgsz=640 data=coco.yaml fraction=0.1 # creates 'yolo26n_qnn.onnx'
        ```

On the last 500 images of COCO val2017, which the calibration above does not use, YOLO26n reached these scores in `val` on the x86-64 export host, against the PyTorch checkpoint and an FP32 ONNX export on the same images:

| Model                                                           | mAP<sup>val</sup> 50-95 | mAP<sup>val</sup> 50 |
| --------------------------------------------------------------- | ----------------------- | -------------------- |
| PyTorch                                                         | 42.1                    | 57.1                 |
| ONNX FP32                                                       | 42.0                    | 56.9                 |
| QNN, calibrated on `coco8.yaml` (4 images)                      | 38.9                    | 54.6                 |
| QNN, calibrated on `coco.yaml` with `fraction=0.1` (500 images) | 39.1                    | 54.5                 |

Calibrating on 500 COCO images instead of the 4 images of `coco8.yaml` raised mAP<sup>val</sup> 50-95 by 0.2 point and lowered mAP<sup>val</sup> 50 by 0.1 point, and both QNN models scored 2.9 to 3.1 points below the FP32 ONNX export in mAP<sup>val</sup> 50-95. On the small default datasets, depth estimation lost the most: on the 4 validation images of `depth8.yaml`, which were also its calibration images, the QNN depth model reached a delta1 of 56.0% against 89.7% for an FP32 ONNX export. Validate the exported model on your own data before you deploy it.

## Output Format

The export writes one file and keeps the intermediate ONNX model next to it:

| File               | Contents                                                                                                                                                                                                                                                                      |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yolo26n_qnn.onnx` | An ONNX model with one `EPContext` node that embeds the QNN context binary, and the class names, task, image size, batch size, stride, and export arguments as ONNX metadata, which Ultralytics reads when it loads the model                                                 |
| `yolo26n.onnx`     | The FP32 ONNX graph that was quantized, with the same channels-last input. It replaces an earlier `yolo26n.onnx` export of the same checkpoint in that directory. Not needed for deployment: `YOLO()` cannot run it and fails with `Got invalid dimensions for input: images` |

The model takes one input named `images` with shape `(batch, height, width, 3)`: RGB, channels last (NHWC), `float32`, scaled to 0-1. This differs from the channels-first input of an [ONNX export](onnx.md#output-format), and matches the layout of camera frames and of the HTP. The outputs have the shapes of a default ONNX export. The shapes below are for the YOLO26n checkpoints at `imgsz=640`, the size of the official mobile models, and `imgsz=224` for classification, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | `imgsz` | Output                                              |
| -------- | ------- | --------------------------------------------------- |
| Detect   | `640`   | `(1, 84, 8400)`                                     |
| Segment  | `640`   | `(1, 116, 8400)` and `(1, 32, 160, 160)`            |
| Semantic | `640`   | `(1, 640, 640)`, `uint8` class index for each pixel |
| Depth    | `640`   | `(1, 1, 640, 640)`, depth map                       |
| Classify | `224`   | `(1, 1000)`, class probabilities                    |
| Pose     | `640`   | `(1, 56, 8400)`                                     |
| OBB      | `640`   | `(1, 20, 8400)`                                     |

- **Raw output**: for each anchor the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS yourself. See the [ONNX output format](onnx.md#output-format) for the layout of each task. There is no NMS-free or embedded-NMS output: `nms=False` logs `This export format does not support end2end models, disabling the end2end branch.` and `nms=True` logs `format=qnn does not support embedded NMS; using nms=None.`, and both write this raw output.
- **Semantic segmentation**: the output is the class index of every pixel as `uint8` (`int32` for models with more than 256 classes), computed on the NPU, not a score for each class.
- **Segmentation**: the second output holds the mask prototypes.
- **Same layout as ONNX**: on one image per task, every output of the QNN models had the shape and type of the output of an FP32 ONNX export of the same checkpoint. The first output of detection, segmentation, pose, and OBB correlated at 0.999 or higher, and the segmentation mask prototypes at 0.95. The raw outputs differed more for classification (correlation 0.76), semantic segmentation (69% of the pixels agree), and depth estimation (correlation 0.24). In `val` on the small default datasets, depth estimation lost the most, see [Calibration](#calibration).

`YOLO("yolo26n_qnn.onnx")` reads the layout from the ONNX metadata, so you only need these details when you run the model in your own code. The metadata keys include `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, `args`, and, for pose models, `kpt_shape`.

### Run with ONNX Runtime

To run the model outside the Ultralytics package, register the QNN Execution Provider of `onnxruntime-qnn` with ONNX Runtime and point it at its HTP backend library. It needs `onnxruntime-qnn` 2.x, which provides the `onnxruntime_qnn` module. This ran on the x86-64 export host and was not run on a Snapdragon device:

```python
# Export first: yolo export model=yolo26n.pt format=qnn name=81 imgsz=640
import ast

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
options = ort.SessionOptions()
options.add_provider_for_devices(devices, {"backend_path": qnn_ep.get_qnn_htp_path()})
session = ort.InferenceSession("yolo26n_qnn.onnx", sess_options=options)

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # your letterboxed image: RGB, NHWC, 0-1
output = session.run(None, {session.get_inputs()[0].name: image})[0]
assert output.shape == (1, 84, 8400), "export a detection model at imgsz=640"

names = ast.literal_eval(session.get_modelmeta().custom_metadata_map["names"])  # {0: 'person', 1: 'bicycle', ...}
```

## Deploy

On a Windows on Snapdragon laptop, or a Linux ARM64 board whose BSP provides the HTP runtime (see [Qualcomm Linux Boards](#qualcomm-linux-boards)), install the package with `pip install ultralytics onnxruntime-qnn` and copy `yolo26n_qnn.onnx` to it. `YOLO("yolo26n_qnn.onnx")` then registers the QNN Execution Provider and runs the context binary with its HTP backend. Exports made with `onnxruntime-qnn` 2.8.0 for this page were not run on a Snapdragon device; the [Windows measurements](#windows-on-snapdragon-laptop) used earlier v73 exports.

- **Matching HTP**: export with the `name` of the HTP generation of your device.
- **Android apps**: the README of the [Flutter plugin](#ultralytics-flutter-plugin) lists the setup it uses for `_qnn.onnx` models: the `com.microsoft.onnxruntime:onnxruntime-android-qnn:1.29.0` and `com.qualcomm.qti:qnn-runtime:2.51.0` packages, and `useLegacyPackaging = true` in `packagingOptions.jniLibs`, because the Hexagon DSP loader needs real `.so` files. It also says that exports made before Ultralytics 8.4.188 with `onnxruntime-qnn` 2.4 or newer fail to load there with `Input name not found`. This setup was not run with the exports of this page.
- **NPU only**: on a Snapdragon device the context binary runs only on the HTP. For the Adreno GPU or the CPU, export a [LiteRT](litert.md) or a standard [ONNX](onnx.md) model instead; the [Android measurements](#android-phone) compare LiteRT on the CPU and GPU with QNN on the NPU.
- **Fixed input shape**: the model always runs at its export `imgsz` and `batch`, which Ultralytics reads from the ONNX metadata. An `imgsz` passed to `predict` or `val` is ignored. Smaller batches are padded, and a `batch=2` model also ran predictions on one and on three images.
- **One QNN model per process**: load a second QNN model, or run `val` after `predict`, in a new Python process, see [Limitations](#limitations-and-troubleshooting).
- **File name**: keep `_qnn.onnx` at the end of the file name, because Ultralytics recognizes the format by it. A copy named `yolo26n_npu.onnx` loads as a plain ONNX model and fails with `EPContext node generated by 'QNNExecutionProvider' is not compatible with any execution provider added to the session.`

### Ultralytics Flutter Plugin

The [Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) runs QNN exports of all seven YOLO26 tasks on Android, for real-time camera inference and single-image prediction. QNN support is opt-in: enable the QNN runtime and add its ONNX Runtime dependency as described in the plugin README. The [Android measurements](#android-phone) on this page were made with it. For iOS, see the [Ultralytics iOS SDK](https://github.com/ultralytics/yolo-ios-app) and the [Core ML integration](coreml.md).

### Qualcomm Linux Boards

On a Dragonwing board such as the IQ-8275 with Qualcomm Linux or a Yocto BSP, the runtime comes from the board vendor's QAIRT-enabled BSP and is not part of the exported model. For HTP inference the BSP must provide the ONNX Runtime QNN Execution Provider, `libQnnSystem.so`, `libQnnHtp.so`, the HTP stub and skeleton libraries of the board's HTP generation (v75 for the IQ-8275), FastRPC userspace support such as `libcdsprpc.so`, the DSP firmware, and the matching FastRPC kernel drivers. The skeleton library must be on the BSP's DSP library path. GPU execution needs `libQnnGpu.so` and the Adreno userspace driver instead, with a standard ONNX export, because `_qnn.onnx` files target the HTP.

### Deployment Guides

- **[ONNX Runtime QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)**: options, supported operators, and the Android and Windows packages of the provider that loads `_qnn.onnx` files.
- **[Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)**: Qualcomm's reference for QNN and QAIRT.
- **[Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app)**: Android apps with QNN, LiteRT, and Core ML models.
- **[LiteRT](litert.md)** and **[MediaTek Dimensity](mediatek.md)**: Android deployment on the CPU and GPU, and on MediaTek NPUs.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare QNN with the other export formats.

## Limitations and Troubleshooting

- **macOS and Python 3.10**: `onnxruntime-qnn` 2.x has packages only for Python 3.11 to 3.14 on Linux and Windows. On macOS and on Linux with Python 3.10, the installation of `onnxruntime-qnn` fails (pip reports `No matching distribution found for onnxruntime-qnn`), and the export continues, quantizes the model, and then fails with `Failed to load library .../onnxruntime/capi/libonnxruntime_providers_qnn.so`. On Windows x64 with Python 3.10, pip installs the earlier `onnxruntime-qnn` 1.23.2 instead, which was not tested. Export on Linux or Windows with Python 3.11 or newer, or with [Ultralytics Platform](../platform/train/models.md#export-model).
- **Automatic installation turned off**: with `YOLO_AUTOINSTALL=false` and no `onnxruntime-qnn`, the export logs `Ultralytics requirement ['onnxruntime-qnn'] not found, AutoUpdate disabled by YOLO_AUTOINSTALL=False` and fails with the same `Failed to load library` error. Install it as in the [Quickstart](#quickstart).
- **Existing output file**: the export writes `yolo26n_qnn.onnx` next to the `.pt` file, and ONNX Runtime does not overwrite it: when the file exists, the export fails with `Failed to generate EP context model since the file '"yolo26n_qnn.onnx"' exists already. Please remove the EP context model if you want to re-generate it.`, and the Python process can then end with a segmentation fault. Delete or rename the earlier file before you export again, also for a different `name`, `imgsz`, or `data`.
- **Second QNN model in one process**: loading a second `_qnn.onnx` model, or running `val` or an export after `predict` with a QNN model, fails with `EP library is already registered under QNNExecutionProvider`. Predicting several times with the same `YOLO` object works, and so does `predict` right after an export. Use a new Python process for each model, as the `yolo` CLI does.
- **Fixed input shape**: `dynamic=True` fails with `argument 'dynamic' is not supported for format='qnn'`. Export again to change the image size or the batch size.
- **Calibration without `data`**: the export logs `INT8 export requires a missing 'data' arg for calibration. Using default 'data=coco8.yaml'.` and calibrates on the small default dataset of the task. Pass `data` with images that represent your deployment data.
- **Opset warning**: unless you pass `opset=21`, the ONNX Runtime quantizer logs `The original model opset version is 18, which does not support 16-bit integer quantization natively. Please update the model to opset >= 21. Automatically update the model to opset 21. Please verify the quantized model.`, with the opset of your export in place of 18, and raises the graph to opset 21. The export continues and needs no action.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='qnn' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends`. Export the checkpoint you started the fine-tuning from instead.
- **Inference on the export host**: on x86-64 Linux, `onnxruntime-qnn` ran the `name=73` and `name=81` context binaries on the CPU, at about 0.3 s (classification) to 16 s (depth) per image for YOLO26n on a busy server CPU, against milliseconds on a phone. Use it to check an export, not to measure its speed.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Measured Performance

The tables below give the latency of one image on two Snapdragon devices, measured on the published `v0.7.0` HTP v81 models (Android) and on earlier HTP v73 exports (Windows), and the size of exports made on an x86-64 Linux server for this page. Each latency cell shows the total time in milliseconds with the preprocess / inference / postprocess split beneath it, and the fastest column for each model is in bold. The [YOLO26 model page](../models/yolo26.md) lists the accuracy of the PyTorch checkpoints.

### Android Phone

**Hardware:** [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) with 12 GB LPDDR5X memory and Android 17 / API 37. Its 3 nm [Snapdragon 8 Elite Gen 5](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5) (SM8850) has an 8-core Qualcomm Oryon CPU (2 Prime cores up to 4.6 GHz and 6 Performance cores up to 3.62 GHz), Adreno GPU, and Hexagon NPU (HTP v81).

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | NPU<br><sup>QNN W8A16<br>(ms)</sup>    |
| ------------- | -------- | --------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| YOLO26n       | Detect   | 640                         | 36.9<br><sup>1.9 / 32.6 / 2.3</sup>    | 10.6<br><sup>1.8 / 6.2 / 2.6</sup>     | **10.2**<br><sup>1.9 / 6.0 / 2.3</sup> |
| YOLO26n-seg   | Segment  | 640                         | 50.5<br><sup>1.9 / 42.9 / 5.8</sup>    | 18.0<br><sup>1.9 / 9.5 / 6.6</sup>     | **16.2**<br><sup>2.1 / 8.4 / 5.7</sup> |
| YOLO26n-sem   | Semantic | 640                         | 45.7<br><sup>1.9 / 35.7 / 8.2</sup>    | 30.6<br><sup>1.9 / 20.5 / 8.3</sup>    | **10.8**<br><sup>2.0 / 6.2 / 2.6</sup> |
| YOLO26n-depth | Depth    | 640                         | 66.8<br><sup>2.2 / 55.0 / 9.7</sup>    | 21.9<br><sup>2.1 / 10.1 / 9.8</sup>    | **19.3**<br><sup>2.2 / 7.4 / 9.7</sup> |
| YOLO26n-cls   | Classify | 224                         | 3.6<br><sup>0.3 / 3.3 / 0.0</sup>      | 2.0<br><sup>0.4 / 1.4 / 0.2</sup>      | **0.8**<br><sup>0.3 / 0.5 / 0.0</sup>  |
| YOLO26n-pose  | Pose     | 640                         | 39.9<br><sup>2.0 / 36.5 / 1.4</sup>    | 10.0<br><sup>1.8 / 6.7 / 1.4</sup>     | **9.7**<br><sup>1.9 / 5.9 / 1.8</sup>  |
| YOLO26n-obb   | OBB      | 640                         | 36.3<br><sup>2.0 / 33.2 / 1.2</sup>    | 10.1<br><sup>2.6 / 5.9 / 1.6</sup>     | **8.1**<br><sup>1.9 / 5.1 / 1.0</sup>  |

- **Speed** values are **single-image burst latencies**: the mean of 15 runs after 3 warmup runs on `bus.jpg`, measured with the [Flutter plugin's](https://github.com/ultralytics/yolo-flutter-app) `0.7.1` on-device benchmark harness and the official `v0.7.0` assets (Ultralytics 8.4.188 exports). Backend order rotated between tasks in one sequential sweep. Native logs confirmed that every CPU row used LiteRT CPU/XNNPACK, every GPU row delegated the complete graph to LiteRT OpenCL (`LITERT_CL`), and every NPU row used the QNN Hexagon HTP backend.
- The CPU and GPU columns are the `w8a32` [LiteRT](litert.md#xiaomi-17) models of the same release, so the table compares the NPU with the other processors of the same phone.
- The detailed benchmark record is in the [Flutter performance doc](https://github.com/ultralytics/yolo-flutter-app/blob/main/doc/performance.md).
- Compare other Android devices in the [LiteRT integration](litert.md#measured-performance), including the [MediaTek Dimensity guide](mediatek.md#measured-performance), and Apple devices in the [Core ML integration](coreml.md#measured-performance).

### Windows on Snapdragon Laptop

**Hardware:** Lenovo laptop with 32 GB memory and Windows 11. Its [Snapdragon X Elite](https://www.qualcomm.com/laptops/products/snapdragon-x-elite) (X1E78100) has a 12-core Qualcomm Oryon CPU, Adreno GPU, and Hexagon NPU (HTP v73). The exact Lenovo model was not recorded.

This earlier sweep compares the PyTorch FP32 CPU baseline with the QNN models on the Hexagon NPU. It was measured with `ultralytics` 8.4.67, `onnxruntime-qnn` 2.2.0, `torch` 2.10.0+cpu, and Python 3.12.10, on v73 models exported before the 224/640 mobile sizes became the standard, so the semantic and OBB models ran at 1024 pixels and there is no depth row. Each cell shows the full `model.predict()` wall time with the reported preprocess / inference / postprocess split beneath it; the total can include framework overhead outside those three stages.

| Model        | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>PT FP32<br>(ms)</sup>      | NPU Hexagon<br><sup>QNN W8A16<br>(ms)</sup> |
| ------------ | -------- | --------------------------- | -------------------------------------- | ------------------------------------------- |
| YOLO26n      | Detect   | 640                         | 91.4<br><sup>4.3 / 75.2 / 0.1</sup>    | **27.2**<br><sup>4.9 / 19.4 / 0.9</sup>     |
| YOLO26n-seg  | Segment  | 640                         | 138.8<br><sup>4.5 / 127.1 / 2.8</sup>  | **34.3**<br><sup>5.0 / 24.0 / 5.1</sup>     |
| YOLO26n-sem  | Semantic | 1024                        | 295.8<br><sup>9.1 / 189.2 / 94.8</sup> | **133.0**<br><sup>8.8 / 37.4 / 83.9</sup>   |
| YOLO26n-cls  | Classify | 224                         | 15.4<br><sup>3.0 / 9.8 / 0.0</sup>     | **11.7**<br><sup>2.7 / 5.5 / 0.0</sup>      |
| YOLO26n-pose | Pose     | 640                         | 109.6<br><sup>4.6 / 102.9 / 0.2</sup>  | **28.9**<br><sup>5.3 / 23.3 / 0.6</sup>     |
| YOLO26n-obb  | OBB      | 1024                        | 267.8<br><sup>8.1 / 254.6 / 0.1</sup>  | **64.8**<br><sup>8.9 / 54.7 / 0.6</sup>     |

- **Speed** values are **single-image burst latencies**: the mean of 100 runs after 10 warmup runs on `bus.jpg`, measured with `time.perf_counter()` around the full `model.predict()` call on a thermally rested device.
- The Hexagon NPU ran 2.2 to 4.1 times faster than the PyTorch CPU baseline at 640 and 1024 pixels, and 1.3 times faster for the 224-pixel classifier, where the fixed preprocessing time weighs more.

### Model Size

The size of the YOLO26 detection models in MB (10^6 bytes), as the PyTorch checkpoint and as the `_qnn.onnx` file for HTP v81 at `imgsz=640`. The export log prints the size in MiB, as `(3.5 MB)` for YOLO26n v81:

| Model   | PyTorch | QNN v81 |
| ------- | ------- | ------- |
| YOLO26n | 5.5     | 3.6     |
| YOLO26s | 20.4    | 11.1    |
| YOLO26m | 44.3    | 22.8    |
| YOLO26l | 53.2    | 27.9    |
| YOLO26x | 118.7   | 60.6    |

Exported with `ultralytics` 8.4.215, `onnxruntime-qnn` 2.8.0, and `onnxruntime` 1.32.0 on an x86-64 Linux server without Snapdragon hardware. The [Calibration](#calibration) section lists the COCO accuracy of the YOLO26n export.

## About Qualcomm QNN

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/5c35f602ba85ec719d394a1231415225.avif" alt="Qualcomm QNN on-device inference">
</p>

Qualcomm AI Engine Direct (QNN) is part of the Qualcomm AI Runtime (QAIRT) SDK and runs neural networks on the CPU, the Adreno GPU, and the Hexagon NPU of Snapdragon and Dragonwing processors through one API with a backend library for each processor. It replaces the earlier Snapdragon Neural Processing Engine (SNPE) SDK. Ultralytics uses its HTP backend through the ONNX Runtime QNN Execution Provider. For the full reference, see the [Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk).

## FAQ

### Do I need a Qualcomm account or a Snapdragon device to export a model?

No. The export runs with the `onnxruntime-qnn` package from PyPI, which includes the QAIRT libraries, and every export made for this page ran on an x86-64 Linux server without a Qualcomm account, an SDK download, or Snapdragon hardware. You need the device only to run the model at NPU speed.

### How does the Ultralytics export compare to Qualcomm AI Hub?

[Qualcomm AI Hub](https://aihub.qualcomm.com/) is Qualcomm's cloud service for compiling, profiling, and benchmarking models on hosted Snapdragon devices, and it needs a Qualcomm account. The Ultralytics export compiles a QNN context binary for the same Hexagon NPU on your own computer with one `export(format="qnn")` call, with no account or upload, and you measure its speed on your own device.
