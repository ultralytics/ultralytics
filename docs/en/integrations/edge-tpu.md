---
title: Export YOLO26 to Google Coral Edge TPU (TFLite)
comments: true
description: Export Ultralytics YOLO26 models to Google Coral Edge TPU with format=edgetpu. Covers INT8 calibration, the Edge TPU compiler, which imgsz keeps the model on the TPU, output files, the libedgetpu runtime, and Raspberry Pi deployment.
keywords: YOLO26, Coral Edge TPU, Google Coral, Edge TPU export, edgetpu, edgetpu_compiler, _edgetpu.tflite, TFLite, TensorFlow Lite, INT8 quantization, calibration, libedgetpu, tflite-runtime, Coral USB Accelerator, Raspberry Pi, edge AI
---

# Google Coral Edge TPU Export

The [Coral Edge TPU](https://coral.ai/docs/edgetpu/faq/) is a small accelerator chip designed by Google for running [TensorFlow Lite](https://developers.google.com/edge/litert) models that are [quantized](https://www.ultralytics.com/glossary/model-quantization) to 8-bit integers, for example in the [Coral USB Accelerator](https://coral.ai/docs/accelerator/get-started/). Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="edgetpu"` converts it through ONNX and TensorFlow into an INT8 TFLite model, calibrated on images, and compiles that model with the Edge TPU compiler into `yolo26n_saved_model/yolo26n_full_integer_quant_edgetpu.tflite`. The export runs on an x86-64 Linux computer and needs no Coral device.

## At a Glance

| Property          | Details                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `edgetpu`                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Output            | `yolo26n_full_integer_quant_edgetpu.tflite`, one file with the [metadata](#output-format) inside, in a `yolo26n_saved_model/` directory with the intermediate TensorFlow files                                                                                                                                                                                                                                                                      |
| Runs on           | Coral Edge TPU devices, such as the USB Accelerator on a Raspberry Pi or a Linux computer. Operations that the compiler does not map to the Edge TPU run on the CPU of the host: at the default `imgsz=640`, 524 of the 657 operations of YOLO26n, and at 512 pixels 6 of 658, see [Image Size](#image-size). YOLO26m and YOLO26x do not compile at the default 640 pixels, see [Limitations and Troubleshooting](#limitations-and-troubleshooting) |
| Runtime           | The Edge TPU runtime `libedgetpu` with TensorFlow Lite, in Python the `tflite-runtime` or `tensorflow` package                                                                                                                                                                                                                                                                                                                                      |
| Precision         | INT8 only, calibrated on images from `data`. YOLO26n is 4.1 MB                                                                                                                                                                                                                                                                                                                                                                                      |
| Export host       | Linux x86-64 with Python 3.10 to 3.13. Not ARM64 Linux, macOS, or Windows. The export needs no GPU and no Coral device                                                                                                                                                                                                                                                                                                                              |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to TF Edge TPU, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package in a virtual environment on an x86-64 Linux machine:

```bash
python3 -m venv edgetpu-env && source edgetpu-env/bin/activate
pip install ultralytics
```

The first Ultralytics Edge TPU export installs TensorFlow, [`onnx2tf`](https://github.com/PINTO0309/onnx2tf), and the ONNX packages that the conversion uses, 38 packages in the Python 3.12 environment used for this page. It also downloads the Edge TPU compiler 16.0, an 11.7 MB archive from the Ultralytics assets, into the `edgetpu-compiler` folder of the Ultralytics settings directory, by default `~/.config/Ultralytics` on Linux. On Python 3.10 to 3.12, the export installs TensorFlow 2.19.0, which requires NumPy below 2.2.0 and replaced NumPy with 2.1.3. The export installs `protobuf` last, 6.31.1 or newer and below 7.0.0 (6.33.6 in the runs for this page), and with TensorFlow 2.19.0 pip then reports a conflict with it, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). The export logs `Restart runtime or rerun command for updates to take effect` and still completes in the same run.

To install all of this before the first Ultralytics export, for example in a container image that runs with `YOLO_AUTOINSTALL=false` and exports on the CPU, run these commands. They install the packages that the export checks for, `onnx2tf` without its dependencies as the export does, `protobuf` last, and unpack the compiler where the export looks for it. On Python 3.13, use `tensorflow>2.19.0` and `tf_keras>2.19.0` instead of the first two packages:

```bash
pip install "tensorflow<=2.19.0" "tf_keras<=2.19.0" sng4onnx onnx_graphsurgeon ai-edge-litert onnx onnxslim onnxruntime
pip install --no-deps "onnx2tf>=1.26.3,<1.29.0" "protobuf>=6.31.1,<7.0.0"
mkdir -p ~/.config/Ultralytics/edgetpu-compiler
curl -sSL https://github.com/ultralytics/assets/releases/download/v0.0.0/edgetpu-compiler_16.0_amd64.tar.gz | tar -xz -C ~/.config/Ultralytics/edgetpu-compiler
```

With these, an export with `YOLO_AUTOINSTALL=false` installed nothing on Python 3.12 and 3.13. If you set `YOLO_CONFIG_DIR` or `XDG_CONFIG_HOME`, unpack the compiler into the `Ultralytics/edgetpu-compiler` folder under that directory. If the home directory is not writable, Ultralytics falls back to a folder in `/tmp` and logs its path; set `YOLO_CONFIG_DIR` to a writable directory instead and unpack the compiler there. The export has these version requirements:

| Requirement       | Version                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Python            | 3.10 to 3.13. On Python 3.9 the export fails with `No module named 'onnx2tf'`, because `onnx2tf` requires Python 3.10 or newer. On Python 3.14, the first export installed the pre-release TensorFlow 2.22.0rc1, for which pip found no matching `tf_keras` (`ResolutionImpossible`), and failed with `No module named 'tf_keras'`. Exports ran with every version from 3.10 to 3.13 |
| `tensorflow`      | 2.0.0 to 2.19.0 on Python 3.12 and older, newer than 2.19.0 on Python 3.13 and newer, the ranges the export installs. Exports for this page used 2.19.0, and 2.21.0 on Python 3.13                                                                                                                                                                                                   |
| `onnx2tf`         | 1.26.3 or newer and below 1.29.0, installed without its dependencies. All exports for this page used 1.28.8                                                                                                                                                                                                                                                                          |
| `protobuf`        | 6.31.1 or newer and below 7.0.0, checked last. All exports for this page used 6.33.6                                                                                                                                                                                                                                                                                                 |
| Edge TPU compiler | Downloaded by the first export unless an `edgetpu_compiler` is on the `PATH`. All exports for this page used 16.0.384591198, the version in the download                                                                                                                                                                                                                             |
| Operating system  | Linux x86-64, the platform of the Edge TPU compiler. Exports for this page ran on Ubuntu 24.04                                                                                                                                                                                                                                                                                       |

For other installation methods, see the [installation guide](../quickstart.md). Then export an Ultralytics YOLO26 model:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to Edge TPU format
        model.export(format="edgetpu")  # creates 'yolo26n_saved_model/yolo26n_full_integer_quant_edgetpu.tflite'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to Edge TPU format
        yolo export model=yolo26n.pt format=edgetpu # creates 'yolo26n_saved_model/yolo26n_full_integer_quant_edgetpu.tflite'
        ```

On a shared AMD EPYC 9655 server, without a GPU, the default export of YOLO26n took 43 to 152 s with the packages installed, and 64 to 158 s for the first export in a new environment, which also installs them. The server ran other jobs, with a load average of 1 to 85. Most of the time is the TensorFlow conversion; the compiler itself took less than a second at 640 pixels.

Copy the `.tflite` file to the device with the Coral Edge TPU. [Predict](../modes/predict.md) and [Validate](../modes/val.md) with the Ultralytics package run only there, with the [Edge TPU runtime](#runtime-installation). They were not run on a Coral device for this page:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported Edge TPU model on the device with the Coral Edge TPU
        model = YOLO("yolo26n_saved_model/yolo26n_full_integer_quant_edgetpu.tflite")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported Edge TPU model on the device with the Coral Edge TPU
        yolo predict model=yolo26n_saved_model/yolo26n_full_integer_quant_edgetpu.tflite source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported Edge TPU model on the device with the Coral Edge TPU
        model = YOLO("yolo26n_saved_model/yolo26n_full_integer_quant_edgetpu.tflite")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported Edge TPU model on the device with the Coral Edge TPU
        yolo val model=yolo26n_saved_model/yolo26n_full_integer_quant_edgetpu.tflite data=coco8.yaml
        ```

## Supported Tasks

Edge TPU export compiles six of the seven Ultralytics tasks with YOLO26: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md); [depth estimation](../tasks/depth.md) fails at the compiler. For each task, the YOLO26n checkpoint was exported at its default image size and the INT8 model that the compiler takes as input was run on the export host; the Edge TPU files were not run on a Coral device for this page. The last column is the number of operations that the compiler mapped to the Edge TPU, see [Image Size](#image-size):

| Task                             | Default `imgsz` | Compiles | Operations on the Edge TPU                                                    |
| -------------------------------- | --------------- | -------- | ----------------------------------------------------------------------------- |
| [Detect](../tasks/detect.md)     | `640`           | ✅       | 133 of 657                                                                    |
| [Segment](../tasks/segment.md)   | `640`           | ✅       | 133 of 711                                                                    |
| [Semantic](../tasks/semantic.md) | `1024`          | ✅       | 133 of 364                                                                    |
| [Depth](../tasks/depth.md)       | `768`           | ❌       | The compiler stops with `Didn't find op for builtin opcode 'EXP' version '2'` |
| [Classify](../tasks/classify.md) | `224`           | ✅       | 121 of 286                                                                    |
| [Pose](../tasks/pose.md)         | `640`           | ✅       | 133 of 691                                                                    |
| [OBB](../tasks/obb.md)           | `1024`          | ✅       | 133 of 695                                                                    |

On the 500 COCO images of the [benchmarks](#benchmarks), the INT8 YOLO26n-seg model that the compiler takes as input reached 36.1 box and 32.1 mask mAP50-95, against 42.0 and 34.9 for the PyTorch model; the INT8 detection model lost 4.0. On 1,000 ImageNet validation images, the last one of each class, the INT8 YOLO26n-cls model reached 70.6% top-1 and 89.8% top-5 accuracy, against 71.7% and 90.5% for the PyTorch model.

For instance segmentation, the INT8 export scales mask coefficients and prototypes reciprocally to preserve box precision. Decode masks by multiplying the two outputs from the same export. The [ST Edge AI integration](stedgeai.md#supported-tasks) uses the same output convention.

[YOLO11](../models/yolo11.md) detection and segmentation, [YOLOv10](../models/yolov10.md), and [YOLOv8](../models/yolov8.md) detection models compiled too, with 134 of 478, 134 of 518, 116 of 450, and 230 of 257 operations on the Edge TPU at 640 pixels for the nano models. [RT-DETR](../models/rtdetr.md) crashed during the conversion, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

## Export Arguments

| Argument   | Type                      | Default     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------- | ------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `format`   | `str`                     | `'edgetpu'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `imgsz`    | `int` or `tuple`          | `640`       | Image size of the model input, fixed in the `.tflite` file: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect and segment, `640` for semantic models pretrained on ADE20K and `1024` for those pretrained on Cityscapes, `768` for depth, `224` for classify, `640` for pose, and `1024` for OBB. The size decides how much of the model runs on the Edge TPU, see [Image Size](#image-size). |
| `quantize` | `int` or `str`            | `None`      | Precision. Unset, `8`, `'int8'`, and `'w8a8'` build INT8, the only precision; `16` or `'fp16'`, `32` or `'fp32'`, `'w8a16'`, `'w8a32'`, and the legacy `half=True` raise an error, see [Precision](#precision). Replaces the deprecated `half`/`int8` flags.                                                                                                                                                                                                                                                                         |
| `data`     | `str`                     | `None`      | [Dataset](../datasets/index.md) YAML with the calibration images; classification instead takes a dataset directory or a built-in dataset name. If omitted, Ultralytics uses the small default dataset of the task, such as `coco8.yaml` for detection. See [Calibration](#calibration).                                                                                                                                                                                                                                              |
| `fraction` | `float`, `int`, or `list` | `1.0`       | Calibration subset: a ratio, an image count (integers above `1`), or a `[train, val, test]` list, from which the entry of the calibration split is used.                                                                                                                                                                                                                                                                                                                                                                             |
| `split`    | `str`                     | `'val'`     | Dataset split whose images calibrate the model.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `opset`    | `int`                     | `None`      | ONNX opset of the intermediate ONNX model that `onnx2tf` converts to TensorFlow. If not set, the export uses opset 18 (lower with PyTorch older than 2.4), capped at the newest opset that the installed `onnx` supports.                                                                                                                                                                                                                                                                                                            |
| `nms`      | `bool`, optional          | `None`      | Not available for Edge TPU: `True` and `False` both write the raw output, which needs NMS, see [Output Format](#output-format).                                                                                                                                                                                                                                                                                                                                                                                                      |
| `device`   | `str`                     | `None`      | Device that runs the PyTorch model while the export traces it, `cpu` by default. `device=0` keeps the GPUs visible to the TensorFlow conversion, asks for `onnxruntime-gpu` unless another ONNX Runtime build is installed, and failed on the GPU of the export host, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). The Edge TPU compiler runs on the CPU.                                                                                                                                                |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Precision

The Edge TPU runs only models whose tensors are 8-bit integers, so the export always builds a full-integer INT8 model, with INT8 weights, activations, input, and output:

| `quantize`      | Result                                                                                         |
| --------------- | ---------------------------------------------------------------------------------------------- |
| unset           | INT8. The export logs `edgetpu export requires INT8 quantization, enabling it.`                |
| `8` or `'int8'` | INT8, the same model. YOLO26n is 4.1 MB                                                        |
| `16`            | Fails with `quantize=16 (FP16) is not supported; format='edgetpu' supports quantize=8 (INT8).` |
| `32`            | Fails with `quantize=32 (FP32) is not supported; format='edgetpu' supports quantize=8 (INT8).` |
| `'w8a16'`       | Fails with `quantize='w8a16' is not supported; format='edgetpu' supports quantize=8 (INT8).`   |
| `'w8a32'`       | Fails with `quantize='w8a32' is not supported; format='edgetpu' supports quantize=8 (INT8).`   |

### Calibration

The export quantizes the model with full-integer post-training quantization: `onnx2tf` runs the model on calibration images to measure the range of every activation. Ultralytics passes it the `val` split of `data`, or the split you set with `split`. Without `data`, the export uses the small default dataset of the task and logs `INT8 export requires a missing 'data' arg for calibration. Using default 'data=coco8.yaml'.` For detection that dataset has 4 validation images, and the export also logs `>300 images recommended for INT8 calibration, found 4 images.` Pass `data` with images that look like your deployment data, as in this Ultralytics export:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Calibrate on the 128 images of COCO128; replace coco128.yaml with your own dataset
        model.export(format="edgetpu", data="coco128.yaml")
        ```

    === "CLI"

        ```bash
        # Calibrate on the 128 images of COCO128; replace coco128.yaml with your own dataset
        yolo export model=yolo26n.pt format=edgetpu data=coco128.yaml
        ```

- **Number of images**: the Ultralytics export uses every image of the split: 128 with `coco128.yaml`, 64 with `fraction=0.5`, and 10 with `fraction=10`, and it warns below 300 images.
- **More images did not help YOLO26n**: in one export for each set, the INT8 model was not more accurate with 128 or 500 images than with the 4 default images, see [Benchmarks](#benchmarks). Validate the exported model on your own data, see [Test on the Export Host](#test-on-the-export-host).
- **Time**: the calibration images go through the TensorFlow conversion. In one session on the shared server, the export of YOLO26n took 45 s with the 4 default images, 410 s with `coco128.yaml`, and 769 s with 500 images.

### Image Size

The Edge TPU compiler maps the operations of the model to the Edge TPU from the input on. According to the Coral documentation, at the first operation it cannot map, the compiler leaves that operation and everything after it on the CPU, even if later operations are supported. For YOLO26n detection, the image size decides where that happens. The table lists what the compiler reported for exports of YOLO26n detection at each size, with the default calibration. According to the Coral documentation, the Edge TPU has roughly 8 MB of SRAM to cache parameters, part of it reserved; the compiler streams the parameters that do not fit from the host:

| `imgsz`      | Compiles | Operations on the Edge TPU | On-chip memory (MiB) | Off-chip memory (MiB) |
| ------------ | -------- | -------------------------- | -------------------- | --------------------- |
| `256`        | ❌       | —                          | —                    | —                     |
| `320`        | ✅       | 652 of 658                 | 3.27                 | 0.05                  |
| `384`        | ✅       | 652 of 658                 | 3.21                 | 0.07                  |
| `416`        | ❌       | —                          | —                    | —                     |
| `448`        | ✅       | 652 of 658                 | 3.54                 | 0.09                  |
| `480`        | ❌       | —                          | —                    | —                     |
| `512`        | ✅       | 652 of 658                 | 3.20                 | 0.11                  |
| `544`        | ✅       | 133 of 658                 | 1.27                 | 0.00                  |
| `576`        | ✅       | 133 of 658                 | 1.27                 | 0.00                  |
| `608`        | ✅       | 133 of 657                 | 1.27                 | 0.00                  |
| `640`        | ✅       | 133 of 657                 | 1.27                 | 0.00                  |
| `1280`       | ❌       | —                          | —                    | —                     |
| `(480, 640)` | ❌       | —                          | —                    | —                     |

- **320, 384, 448, and 512 pixels**: 652 of 658 operations map to the Edge TPU, including the attention operations (`BATCH_MATMUL`, `SOFTMAX`). The CPU runs the last 6, which reshape and concatenate the class scores and apply the sigmoid to them.
- **544 to 640 pixels, including the default**: the compiler stops at a `TRANSPOSE` operation of the stride-32 feature map, of shape `(1, 256, 20, 20)` at 640 pixels, and logs `Operation is otherwise supported, but not mapped due to some unspecified limitation`. The CPU runs that operation and everything after it, including the attention, so 133 operations run on the Edge TPU: 133 of 657 at 640 pixels.
- **256, 416, 480, and 1280 pixels, and `(480, 640)`**: the export fails at the compiler, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).
- **Other YOLO26n tasks at 512 pixels**: segmentation mapped 700 of 710, semantic segmentation with the Cityscapes checkpoint 363 of 364, and pose estimation 679 of 692 operations to the Edge TPU; the OBB export failed at the compiler.
- **Larger YOLO26 models**: YOLO26s, YOLO26m, and YOLO26l mapped all but 6 operations to the Edge TPU at 320 and 384 pixels (907 of 913, 942 of 948, and 1390 of 1396 at 320 pixels), and stopped early again at 448 pixels. YOLO26x mapped 263 of 1780 operations at 320 pixels and 263 of 1779 at 384, and its export at 448 pixels failed with `Compilation failed: Internal error.` At 512 pixels, YOLO26s mapped 133 of 913 and YOLO26m 167 of 948 operations to the Edge TPU, the export of YOLO26l failed with `Compilation failed due to large activation tensors in model.`, and that of YOLO26x with `Compilation timeout. This is usually caused by large activations.`, which the compiler logs when one attempt takes longer than the 180 s that Ultralytics allows. At the default 640 pixels, YOLO26s mapped 133 of 913 and YOLO26l 263 of 1395 operations, and the exports of YOLO26m and YOLO26x failed with `Compilation failed due to large activation tensors in model.`, see [Benchmarks](#benchmarks).
- **Accuracy**: on the 500 COCO images of the [benchmarks](#benchmarks), YOLO26n reached 39.1 mAP50-95 in PyTorch and 35.6 as the INT8 model at 512 pixels, against 42.1 and 38.1 at 640 pixels. Validate the size you choose on your data, see [Test on the Export Host](#test-on-the-export-host).
- **Other models**: [YOLO11n](../models/yolo11.md) mapped 455 of 478 operations at 320, 384, and 512 pixels and 44 at 480 pixels; [YOLOv8n](../models/yolov8.md) mapped all but 23 at 320, 384, and 512 pixels.

To export an Ultralytics YOLO26n model at a size where all but its last 6 operations run on the Edge TPU, pass `imgsz`:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Export at 512 pixels, where 652 of the 658 operations of YOLO26n map to the Edge TPU
        model.export(format="edgetpu", imgsz=512)  # creates 'yolo26n_saved_model/yolo26n_full_integer_quant_edgetpu.tflite'
        ```

    === "CLI"

        ```bash
        # Export at 512 pixels, where 652 of the 658 operations of YOLO26n map to the Edge TPU
        yolo export model=yolo26n.pt format=edgetpu imgsz=512 # creates 'yolo26n_saved_model/yolo26n_full_integer_quant_edgetpu.tflite'
        ```

## Output Format

The export writes a `yolo26n_saved_model/` directory of 40.3 MB for YOLO26n, next to the `.pt` file. Only the `_edgetpu.tflite` file is needed on the device. Sizes on this page are in MB (10^6 bytes); the export log prints MiB, so it reports 3.9 MB for the 4.1 MB Edge TPU file:

| File                                                                                                      | Size (MB)   | Contents                                                                                                                                              |
| --------------------------------------------------------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yolo26n_full_integer_quant_edgetpu.tflite`                                                               | 4.1         | The compiled model that the Edge TPU runtime and `YOLO()` load                                                                                        |
| `yolo26n_full_integer_quant_edgetpu.log`                                                                  | 0.0         | The operation log of the compiler: each operation type, how many, and whether it maps to the Edge TPU                                                 |
| `yolo26n_full_integer_quant.tflite`                                                                       | 3.9         | The full-integer INT8 model that the compiler takes as input; it runs on the CPU with LiteRT, see [Test on the Export Host](#test-on-the-export-host) |
| `yolo26n_integer_quant.tflite`, `yolo26n_int8.tflite`, `yolo26n_float32.tflite`, `yolo26n_float16.tflite` | 2.9 to 10.0 | Other TFLite files that `onnx2tf` writes on the way. The Edge TPU does not use them                                                                   |
| `saved_model.pb`, `variables/`, `assets/`, `fingerprint.pb`                                               | 10.4        | The TensorFlow SavedModel of the conversion                                                                                                           |
| `metadata.yaml`                                                                                           | 0.0         | Class names, task, image size, batch size, stride, and export arguments                                                                               |

The model takes one input with shape `(1, height, width, 3)`: one RGB image, channels last (NHWC), as `int8`. The quantization parameters of the input are a scale of 1/255 and a zero point of -128, so the `int8` value is the pixel value from 0 to 255 minus 128. This differs from the [ONNX export](onnx.md#output-format), which takes channels-first `float32` images scaled to 0-1. The batch size is always 1. The outputs are `int8` too. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation, and are the same with `nms=None`, `nms=False`, and `nms=True`:

| Task     | Default `imgsz` | Output                                                                   |
| -------- | --------------- | ------------------------------------------------------------------------ |
| Detect   | `640`           | `(1, 84, 8400)`                                                          |
| Segment  | `640`           | `(1, 116, 8400)`, and `(1, 160, 160, 32)` mask prototypes, channels last |
| Semantic | `1024`          | `(1, 1024, 1024, 19)`, a score for each pixel and class, channels last   |
| Depth    | `768`           | Not exported, see [Supported Tasks](#supported-tasks)                    |
| Classify | `224`           | `(1, 1000)`: a probability for each of the 1000 ImageNet classes         |
| Pose     | `640`           | `(1, 56, 8400)`: 4 box values, 1 class score, then 17 keypoints x 3      |
| OBB      | `1024`          | `(1, 20, 21504)`: 4 box values, 15 class scores, then the angle          |

- **Raw output**: shape `(batch, values, anchors)`. For each anchor: the box center x, center y, width, and height, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). The box values and the keypoint x and y are normalized to 0-1, like the scores, so multiply them by the input width and height. Apply NMS yourself.
- **`nms`**: there is no NMS inside the model, and the NMS-free head of YOLO26 is not available. Both `nms=True` and `nms=False` write the raw output with a warning, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).
- **Dequantization**: convert each `int8` output to floats with the scale and zero point that the interpreter reports for it, `(value - zero_point) * scale`, as Ultralytics does.
- **Segmentation**: the 32 mask coefficients are the last 32 values of each anchor, divided by 16 by the export, and the prototypes are multiplied by 16, so their product is the mask as in the PyTorch model. The interpreter lists the mask prototypes `(1, 160, 160, 32)`, channels last, as the first output and the `(1, 116, 8400)` output second; tell them apart by shape, as Ultralytics does.
- **Semantic segmentation**: the output holds class scores, not the class map of the [ONNX export](onnx.md#output-format), so take the argmax over the class axis in your own code.

`YOLO()` reads the layout from the file metadata and scales the coordinates, so you only need these details when you run the model in your own code. The metadata is a `metadata.json` entry that the export appends to each `.tflite` file as a ZIP archive, readable with any ZIP library; its keys include `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, and `args`. The same metadata is in `metadata.yaml`.

## Deploy

On the device, `YOLO("yolo26n_saved_model/yolo26n_full_integer_quant_edgetpu.tflite")` loads the file with the TensorFlow Lite interpreter of the `tflite-runtime` package, or of `tensorflow` when `tflite-runtime` is not installed, and runs it through the Edge TPU delegate of `libedgetpu`. The interpreter runs the operations that the compiler did not map to the Edge TPU on the CPU of the device. The model was not run on a Coral device for this page; the points below follow from the Ultralytics code, and the errors were reproduced on an x86-64 Linux computer without a Coral device:

- **Delegate library**: Ultralytics loads `libedgetpu.so.1` on Linux, `libedgetpu.1.dylib` on macOS, and `edgetpu.dll` on Windows, when the first prediction builds the model. Without the library, that prediction fails with `OSError: libedgetpu.so.1: cannot open shared object file: No such file or directory` on Linux. With the `libedgetpu` 16.0 build from feranick (for TensorFlow 2.19.1) installed on an x86-64 computer without a Coral device, it fails with `ValueError: Failed to load delegate from libedgetpu.so.1`, with both `tflite-runtime` 2.14.0 and `tensorflow` 2.19.0.
- **`device`**: without `device`, or with `device="tpu:0"`, Ultralytics selects the first Edge TPU, and `"tpu:1"` the second; it passes the part after `tpu`, such as `:1`, to the delegate as its `device` option. Pre- and postprocessing run on the CPU.
- **Input conversion**: Ultralytics quantizes each preprocessed image to the `int8` input of the model and dequantizes the outputs, so `predict` and `val` need no extra arguments.
- **Fixed input shape**: the model always runs at its export `imgsz`, which Ultralytics reads from the metadata, and with a batch size of 1. An `imgsz` passed to `predict` or `val` is ignored; export again to change the size. On the export host, the INT8 model of the export predicted a list of three images, ignored `imgsz=320`, and gave the same accuracy with `val` at `batch=4` as at `batch=1`.
- **File name**: Ultralytics recognizes the format by `_edgetpu.tflite` anywhere in the file name: `model_edgetpu.tflite`, `YOLO26N_EDGETPU.TFLITE`, and even `yolo26n_edgetpu.tflite.bak` loaded as Edge TPU models. A name without it, such as `yolo26n.tflite` or `yolo26n_edgetpu_copy.tflite`, loads as a [LiteRT](litert.md) model and fails with `RuntimeError: Encountered unresolved custom op: edgetpu-custom-op.`
- **Files to copy**: only the `_edgetpu.tflite` file. It carries its metadata, so a copy in another directory loaded with the class names and task of the export.

### Test on the Export Host

The `yolo26n_full_integer_quant.tflite` file next to the Edge TPU file is the INT8 model that the compiler took as input. `YOLO()` loads it as a [LiteRT](litert.md) model and runs it on the CPU with the `ai-edge-litert` package, which has packages for Linux on x86-64 and ARM64, macOS on Apple silicon, and Windows, so you can check the accuracy of the quantized Ultralytics YOLO26 model on the export host before you copy the Edge TPU file to the device:

```python
from ultralytics import YOLO

# Load the INT8 model that the export compiled for the Edge TPU, here on the CPU of the export host
model = YOLO("yolo26n_saved_model/yolo26n_full_integer_quant.tflite")

# Validate accuracy on the COCO8 dataset
metrics = model.val(data="coco8.yaml")
```

For YOLO26n, this gave 0.653 mAP50-95 on COCO8, and the model found the bus and the 4 persons in `bus.jpg`, see [Benchmarks](#benchmarks) for the accuracy on COCO. This checks the quantization, not the Edge TPU: the compiled file was not run on a Coral device for this page.

### Runtime Installation

On the device, install the Edge TPU runtime, `libedgetpu`, and a TensorFlow Lite interpreter for Python. Google no longer maintains the runtime: the [`google-coral/libedgetpu`](https://github.com/google-coral/libedgetpu) repository is archived, and the `coral-edgetpu-stable` apt repository that the [Coral setup instructions](https://coral.ai/docs/accelerator/get-started/) use no longer serves the packages. The [Raspberry Pi guide](../guides/coral-edge-tpu-on-raspberry-pi.md#install-the-edge-tpu-runtime) installs the builds of [feranick/libedgetpu](https://github.com/feranick/libedgetpu/releases) instead, which are published for Debian and Ubuntu on x86-64, ARM64, and 32-bit ARM, and for macOS.

For the interpreter, `tflite-runtime` 2.14.0, its last release, has packages for Python 3.8 to 3.11 on Linux x86-64, ARM64, and 32-bit ARM. On newer Python, install `tensorflow`, which Ultralytics uses when `tflite-runtime` is missing; `tensorflow` has no packages for 32-bit ARM. The `libedgetpu` builds name the TensorFlow version they were built with, such as 2.19.1; use a matching TensorFlow Lite interpreter. None of this was installed on a Coral device for this page.

### Deployment Guides

- **[Coral Edge TPU on a Raspberry Pi](../guides/coral-edge-tpu-on-raspberry-pi.md)**: install the Edge TPU runtime on a Raspberry Pi, run an Ultralytics YOLO model on the Coral USB Accelerator, and compare Raspberry Pi 4 and 5.
- **[Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/)**: the compiler options, the on-chip parameter cache, co-compiling several models, and splitting a model across several Edge TPUs.
- **[TensorFlow models on the Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)**: the model requirements and the operations that the Edge TPU supports.
- **[Run inference on the Edge TPU with Python](https://coral.ai/docs/edgetpu/tflite-python/)**: the TensorFlow Lite Python API with the Edge TPU delegate, for code outside the Ultralytics package.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare Edge TPU with the other export formats.

## Limitations and Troubleshooting

- **Export host**: on macOS the export stops with `Edge TPU export only supported on non-aarch64 Linux. See https://coral.ai/docs/edgetpu/compiler` before it installs anything. The same check stops it on Windows and on ARM64 Linux, such as a Raspberry Pi, where it was not run for this page. Export on an x86-64 Linux computer or with [Ultralytics Platform](../platform/train/models.md#export-model).
- **Python 3.9 and 3.14**: the export fails with `No module named 'onnx2tf'` on Python 3.9 and with `No module named 'tf_keras'` on Python 3.14, see [Quickstart](#quickstart). Use Python 3.10 to 3.13.
- **Automatic installation turned off**: with `YOLO_AUTOINSTALL=false` and a plain `pip install ultralytics`, the export logs `Ultralytics requirements ['onnx>=1.12.0,<2.0.0', 'onnxruntime', 'onnxslim>=0.1.82'] not found, AutoUpdate disabled by YOLO_AUTOINSTALL=False` and fails with `No module named 'onnx'`. With the packages installed but no compiler, it fails with `Edge TPU compiler not found and YOLO_AUTOINSTALL=False. Install it from https://coral.ai/docs/edgetpu/compiler/`. The Coral page installs the compiler from an apt repository that no longer serves it; use the commands in [Quickstart](#quickstart) instead.
- **Depth estimation**: the INT8 model of a depth model contains an `EXP` operator of version 2 that the compiler cannot read, and the export fails with `Didn't find op for builtin opcode 'EXP' version '2'`.
- **Image sizes that do not compile**: the export of YOLO26n detection failed at the compiler at `imgsz` 256, 416, 480, 1280, and `(480, 640)`: the compiler logged `Compilation failed!` for each shorter segment that its search tried, and the export ended with the same `returned non-zero exit status 1.` At 1280 pixels the compiler also logged `Compilation failed due to large activation tensors in model.` Running the compiler by hand with its default search step did not help at 256, 416, and 480 pixels. See [Image Size](#image-size) for the sizes that compile.
- **YOLO26m and YOLO26x at the default image size**: their export at 640 pixels failed at the compiler with `Compilation failed due to large activation tensors in model.`, and the export ended with the same `returned non-zero exit status 1.` YOLO26l compiled at 640 pixels but failed with the same message at 512. At 320 and 384 pixels, YOLO26m and YOLO26l compiled with all but 6 operations on the Edge TPU, and YOLO26x with only 263 of 1780 at 320 pixels, see [Image Size](#image-size). The failed export leaves the `_full_integer_quant.tflite` file; [Test on the Export Host](#test-on-the-export-host) shows how to validate it on the CPU.
- **`batch`, `dynamic`, and `simplify`**: `batch` other than 1, `dynamic=True`, and `simplify=False` fail with `argument 'batch' is not supported for format='edgetpu'` and the same message for the other two; the defaults pass. The model has a batch size of 1 and a fixed input size; export again to change the image size.
- **Embedded NMS and the NMS-free head**: `nms=True` logs `format=edgetpu does not support embedded NMS; using nms=None.` For YOLO26, `nms=False` logs `This export format does not support end2end models, disabling the end2end branch.`; for a model without an NMS-free head, such as YOLO11, it logs `This model has no one-to-one head; using one-to-many outputs.` Both write the raw output, which needs NMS, see [Output Format](#output-format).
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='edgetpu' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends. Export a non-QAT checkpoint to this format instead.` Export the checkpoint you started the fine-tuning from instead.
- **Output directory and leftover files**: every export of one checkpoint deletes and rewrites `yolo26n_saved_model/`, whatever its `imgsz` or `data`, so rename the directory before you export a second variant and keep `_edgetpu.tflite` in the file name. During the conversion the export writes an intermediate `yolo26n.onnx` next to the `.pt` file and deletes it afterwards, so it also deletes an [ONNX export](onnx.md) of the same checkpoint in that directory: keep your ONNX export in another directory or export it again. The export also downloads `calibration_image_sample_data_20x128x128x3_float32.npy` (3.9 MB) into the working directory; the Edge TPU file does not need it.
- **Messages in the export log**: with TensorFlow 2.19.0 and `protobuf` 6.33.6, pip reports `tensorflow 2.19.0 requires protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3, but you have protobuf 6.33.6 which is incompatible.` when the export installs `protobuf` (with TensorFlow 2.21.0 on Python 3.13 it reported no conflict), and importing TensorFlow prints `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` five times, in the export log and again when a prediction loads the Edge TPU file through `tensorflow`; neither stops the run. When `libedgetpu` is missing, `AttributeError: 'Delegate' object has no attribute '_library'` follows the `OSError` of the prediction.
- **GPU on the export host: `CUDA_ERROR_INVALID_HANDLE`**: with an NVIDIA RTX PRO 6000 Blackwell GPU visible to the process and TensorFlow 2.19.0 from PyPI, the export with `device=0` failed during the TensorFlow conversion with `CUDA_ERROR_INVALID_HANDLE`. Without `device` and with `device="cpu"`, the export completed with the GPU visible, although TensorFlow still logged that it created a GPU device during the TFLite conversion. The test of the INT8 model in [Test on the Export Host](#test-on-the-export-host) also ran with the GPU visible. Export without `device=0`.
- **RT-DETR**: the export of `rtdetr-l.pt` ended the Python process with a segmentation fault during the INT8 conversion, after the message `error: scale out of expressed type range [1.401298e-45, 3.402823e+38]`.
- **YOLO-NAS models**: the Ultralytics code rejects them with the assertion `Edge TPU export is not supported for YOLO-NAS models.`; this was not run for this page.
- **No Coral device found**: according to the [Coral documentation](https://coral.ai/docs/accelerator/get-started/), `ERROR: Failed to retrieve TPU context` and `ERROR: Node number 0 (edgetpu-custom-op) failed to prepare` mean that the USB Accelerator is not plugged in or that your user account is not in the `plugdev` group. This was not reproduced for this page.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

No Coral device was available for this page, so the speed of the models on the Edge TPU was not measured. The tables below compare, for the YOLO26 detection models exported at the default 640 pixels, the accuracy of the INT8 model that the compiler takes as input, run with LiteRT on the CPU of the export host, an **AMD EPYC 9655** server, with PyTorch and the ONNX export, the file sizes, and what the compiler reported. The Edge TPU runs the compiled form of the same INT8 model; its accuracy on a Coral device was not measured.

=== "Accuracy"

    mAP<sup>val</sup> 50-95 on 500 images of COCO val2017, higher is better. The INT8 columns are the `_full_integer_quant.tflite` file, calibrated on the images named in the header and run on the CPU.

    | Model   | PyTorch | ONNX FP32 | INT8, default calibration (4 images) | INT8, `coco128.yaml` (128 images) | INT8, 500 COCO images |
    | ------- | ------- | --------- | ------------------------------------ | --------------------------------- | --------------------- |
    | YOLO26n | 42.1    | 42.1      | 38.1                                 | 37.3                              | 37.1                  |
    | YOLO26s | 49.5    | 49.5      | 43.4                                 | 42.8                              | 42.9                  |
    | YOLO26m | 53.6    | 53.6      | 47.1                                 | 46.7                              | 46.3                  |
    | YOLO26l | 55.5    | 55.5      | 48.4                                 | 47.3                              | 47.2                  |
    | YOLO26x | 55.7    | 55.7      | 49.2                                 | 47.8                              | 47.8                  |

=== "Size"

    Size of the `.pt` checkpoint, the ONNX file, the INT8 `.tflite` file, and the compiled Edge TPU file in MB (10^6 bytes). The smallest file for each model is in bold.

    | Model   | PyTorch | ONNX FP32 | INT8 `.tflite` | Edge TPU `.tflite` |
    | ------- | ------- | --------- | -------------- | ------------------ |
    | YOLO26n | 5.5     | 9.9       | **3.9**        | 4.1                |
    | YOLO26s | 20.4    | 38.3      | **14.1**       | 16.3               |
    | YOLO26m | 44.3    | 82.0      | **25.3**       | —                  |
    | YOLO26l | 53.2    | 99.6      | **32.0**       | 42.8               |
    | YOLO26x | 118.7   | 223.3     | **70.7**       | —                  |

=== "Edge TPU Compiler"

    What the compiler reported for the exports with the default calibration at `imgsz=640`. On-chip memory holds model parameters on the Edge TPU; the compiler streams the rest from the host.

    | Model   | Operations on the Edge TPU                                            | On-chip memory (MiB) | Off-chip memory (MiB) |
    | ------- | --------------------------------------------------------------------- | -------------------- | --------------------- |
    | YOLO26n | 133 of 657                                                            | 1.27                 | 0.00                  |
    | YOLO26s | 133 of 913                                                            | 4.72                 | 0.00                  |
    | YOLO26m | Fails: `Compilation failed due to large activation tensors in model.` | —                    | —                     |
    | YOLO26l | 263 of 1395                                                           | 6.84                 | 4.52                  |
    | YOLO26x | Fails: `Compilation failed due to large activation tensors in model.` | —                    | —                     |

- **INT8 accuracy**: with the default calibration, the INT8 model was 4.0 to 7.1 mAP below PyTorch. The FP32 `.tflite` file of the same conversion matched the ONNX accuracy for each model in the table, so the difference comes from the quantization.
- **Calibration**: more calibration images did not raise the accuracy: YOLO26n 38.1, 37.3, and 37.1; YOLO26s 43.4, 42.8, and 42.9; YOLO26m 47.1, 46.7, and 46.3; YOLO26l 48.4, 47.3, and 47.2; YOLO26x 49.2, 47.8, and 47.8 mAP with 4, 128, and 500 images.
- **Operations on the Edge TPU**: at 640 pixels, the compiler mapped only the first 133 operations of YOLO26n and YOLO26s and 263 operations of YOLO26l to the Edge TPU and left the rest to the CPU, see [Image Size](#image-size). The exports of YOLO26m and YOLO26x failed at the compiler with `Compilation failed due to large activation tensors in model.`; their INT8 columns are the INT8 file that the export leaves before the compiler.

Measured with `ultralytics` 8.4.220, TensorFlow 2.19.0, `onnx2tf` 1.28.8, the Edge TPU compiler 16.0.384591198, `ai-edge-litert` 2.4.0, PyTorch 2.15.0, and ONNX Runtime 1.32.0 at `batch=1`. PyTorch ran with `rect=False`, at the same 640x640 input as the exported models. Accuracy is on the last 500 images of COCO val2017 in sorted order, so it differs from the values for the full validation set on the [YOLO26 model page](../models/yolo26.md). The 500 calibration images are the first 500 images of COCO train2017 in sorted order.

## About Coral Edge TPU

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/1ae94bc857840372b88ae8bf00492b39.avif" alt="TensorFlow Lite Edge TPU compilation workflow">
</p>

The [Edge TPU](https://coral.ai/docs/edgetpu/faq/) is an ASIC designed by Google for low-power inference. Unlike Cloud [TPUs](https://www.ultralytics.com/glossary/tpu-tensor-processing-unit), which run in Google data centers and suit training large models, it is designed for inference on small, low-power devices. Google rates one Edge TPU at 4 trillion operations per second with 2 watts, and it runs only TensorFlow Lite models. The [Edge TPU compiler](https://coral.ai/docs/edgetpu/compiler/) turns a quantized `.tflite` file into one that the Edge TPU runtime, `libedgetpu`, executes on the accelerator. For the full Edge TPU documentation, see the [Coral documentation](https://coral.ai/docs/); Google's [Coral website](https://developers.google.com/coral) describes its RISC-V based Coral platform for edge AI.

## FAQ

### Does the whole Ultralytics YOLO26 model run on the Coral Edge TPU?

No: when Ultralytics exports YOLO26n at the default 640 pixels, the Edge TPU compiler maps 133 of its 657 operations to the Edge TPU, and the CPU of the device runs the rest, from the attention operations at the stride-32 feature map on. Exported at 320, 384, 448, or 512 pixels, 652 of 658 operations of YOLO26n map to the Edge TPU, and the CPU runs only the last 6. YOLO26s, YOLO26m, and YOLO26l keep all but 6 operations on the Edge TPU, of the sizes tested, only at 320 or 384 pixels, and YOLO26x kept 263 of its 1780 operations there even at 320 pixels, see [Image Size](#image-size). When part of the model runs on the CPU, the compiler output in the export log shows the split in the `Number of operations that will run on Edge TPU` line.

### Can I test an Ultralytics Edge TPU model without a Coral device?

Yes, the INT8 model before compilation: Ultralytics leaves `yolo26n_full_integer_quant.tflite` next to the Edge TPU file, and `YOLO()` runs it on the CPU, see [Test on the Export Host](#test-on-the-export-host). The Edge TPU file itself needs the Edge TPU runtime and a device: without `libedgetpu`, its first prediction fails with `OSError: libedgetpu.so.1: cannot open shared object file: No such file or directory`, and with `libedgetpu` installed but no Coral device attached with `ValueError: Failed to load delegate from libedgetpu.so.1`.

### How many calibration images does an Ultralytics Edge TPU export need?

An Ultralytics Edge TPU export runs with one or more calibration images and warns below 300. For YOLO26n, more images did not raise the accuracy, in one export for each set: 38.1 mAP50-95 with the 4 default images and 37.1 with 500 COCO images, see [Benchmarks](#benchmarks). Calibrate on images from your deployment data, and compare the exported model on your validation set before you deploy it, see [Test on the Export Host](#test-on-the-export-host).

### Can I export an Ultralytics Edge TPU model on a Raspberry Pi or a Mac?

No, the Ultralytics Edge TPU export runs only on Linux x86-64, the platform that the Edge TPU compiler is built for; on macOS, Windows, and ARM64 Linux it stops with `Edge TPU export only supported on non-aarch64 Linux. See https://coral.ai/docs/edgetpu/compiler`. Export on an x86-64 Linux computer or with [Ultralytics Platform](../platform/train/models.md#export-model), then copy the `_edgetpu.tflite` file to the Raspberry Pi, see the [Raspberry Pi guide](../guides/coral-edge-tpu-on-raspberry-pi.md).
