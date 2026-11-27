---
title: Export YOLO26 to TF SavedModel Format
comments: true
description: Export Ultralytics YOLO26 models to TensorFlow SavedModel with format=saved_model. Covers the yolo26n_saved_model directory, NHWC input, output shapes for nms, TensorFlow and TensorFlow Serving inference, CPU threads, and benchmarks.
keywords: YOLO26, TF SavedModel, TensorFlow SavedModel, saved_model export, TensorFlow Serving, onnx2tf, tf.saved_model.load, NHWC, Ultralytics, model deployment
---

# TF SavedModel Export

[TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model) is the standard serialization format of [TensorFlow](https://www.ultralytics.com/glossary/tensorflow): a directory with the graph, the [weights](https://www.ultralytics.com/glossary/model-weights), and the signatures that TensorFlow, [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving), and other TensorFlow tools load without the code that built the model. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="saved_model"` converts it through ONNX with [`onnx2tf`](https://github.com/PINTO0309/onnx2tf) into a `yolo26n_saved_model/` directory with FP32 weights and a channels-last (NHWC) input, which `YOLO()`, TensorFlow in Python, and TensorFlow Serving load.

## At a Glance

| Property          | Details                                                                                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `saved_model`                                                                                                                                                   |
| Output            | `yolo26n_saved_model/`, a directory with the SavedModel, the [metadata](#output-format), and two LiteRT files                                                   |
| Runs on           | CPUs, and GPUs that TensorFlow supports, in Python, TensorFlow Serving, and other TensorFlow runtimes. Ultralytics runs it with TensorFlow in Python on the CPU |
| Runtime           | TensorFlow: the `tensorflow` Python package, the TensorFlow Serving server, or the TensorFlow C and C++ libraries                                               |
| Precision         | FP32 only. YOLO26n is 24.9 MB, including 14.8 MB of LiteRT files                                                                                                |
| Export host       | Linux and macOS, Python 3.10 to 3.13, below 3.13 on macOS (run with 3.12); see [Quickstart](#quickstart). The export needs no GPU                               |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to TF SavedModel, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package in a virtual environment, here on Linux, to export Ultralytics YOLO26 models to TF SavedModel:

```bash
python3 -m venv tf-env && source tf-env/bin/activate
pip install ultralytics
```

The first Ultralytics TF SavedModel export installs TensorFlow, [`onnx2tf`](https://pypi.org/project/onnx2tf/), `tf_keras`, and the ONNX packages that the conversion uses, 38 packages in the Python 3.12 environment used for this page. On Python 3.10 to 3.12, the export installs TensorFlow 2.19.0, which requires NumPy below 2.2.0, and downgraded NumPy to 2.1.3 in every such environment of this page, from 2.5.5 on Python 3.12. The export installs `protobuf` last, 6.31.1 or newer and below 7.0.0 (6.33.6 in the runs for this page), and pip then reports a conflict with TensorFlow 2.19.0, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). The export logs `Restart runtime or rerun command for updates to take effect` and still completes in the same run.

To install all of this on Linux before the first Ultralytics export, for example in a container image that runs with `YOLO_AUTOINSTALL=false`, run these commands. They install the packages that the export checks for, `onnx2tf` without its dependencies as the export does, and `protobuf` last. On Python 3.13, use `tensorflow>2.19.0` and `tf_keras>2.19.0` instead of the first two packages. On macOS, the export checks for `ai-edge-litert` below 1.4.0, so use `"ai-edge-litert<1.4.0"` there; the commands were not run on macOS:

```bash
pip install "tensorflow<=2.19.0" "tf_keras<=2.19.0" sng4onnx onnx_graphsurgeon ai-edge-litert onnx onnxslim onnxruntime
pip install --no-deps "onnx2tf>=1.26.3,<1.29.0" "protobuf>=6.31.1,<7.0.0"
```

With these, an export with `YOLO_AUTOINSTALL=false` installed nothing on Python 3.12 and 3.13. The export has these version requirements:

| Requirement      | Version                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python           | 3.10 to 3.13. On Python 3.9 the export failed with `No module named 'onnx2tf'`, because `onnx2tf` 1.26.3 and newer require Python 3.10 or newer. In the runs for this page on Python 3.14, the first export installed the pre-release TensorFlow 2.22.0rc1, for which pip found no matching `tf_keras` (`ResolutionImpossible`), and failed with `No module named 'tf_keras'`. Exports ran with every version from 3.10 to 3.13 |
| PyTorch          | 1.13 or newer for `nms=True`, which the export checks for. Exports for this page ran with PyTorch 2.15.0, and 2.14.1 on Python 3.10                                                                                                                                                                                                                                                                                             |
| `tensorflow`     | 2.0.0 to 2.19.0 on Python 3.12 and older, newer than 2.19.0 on Python 3.13 and newer, the ranges the export installs. Exports for this page used 2.19.0, and 2.21.0 on Python 3.13                                                                                                                                                                                                                                              |
| `onnx2tf`        | 1.26.3 or newer and below 1.29.0, installed without its dependencies. All exports for this page used 1.28.8                                                                                                                                                                                                                                                                                                                     |
| `protobuf`       | 6.31.1 or newer and below 7.0.0, checked last. All exports for this page used 6.33.6                                                                                                                                                                                                                                                                                                                                            |
| Operating system | Linux and macOS; on macOS with Python 3.13 or newer the export stops, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). Exports for this page ran on Ubuntu 24.04 x86-64 and on macOS 27 with an Apple M4; Windows and ARM64 Linux were not run                                                                                                                                                          |

For other installation methods, see the [installation guide](../quickstart.md). Then export an Ultralytics YOLO26 model and load the exported directory to [run inference](../modes/predict.md) or [validate](../modes/val.md) its accuracy:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to TF SavedModel format
        model.export(format="saved_model")  # creates 'yolo26n_saved_model'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to TF SavedModel format
        yolo export model=yolo26n.pt format=saved_model # creates 'yolo26n_saved_model'
        ```

On a shared AMD EPYC 9655 server, the default export of YOLO26n took 11.9 to 13.3 s in 14 runs with the packages installed, and 28.9 to 35.4 s for the first export in a new environment, which also installs them. The exported Ultralytics YOLO26 model runs on the export host, so you can test it there:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported TF SavedModel model
        model = YOLO("yolo26n_saved_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported TF SavedModel model
        yolo predict model=yolo26n_saved_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported TF SavedModel model
        model = YOLO("yolo26n_saved_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported TF SavedModel model
        yolo val model=yolo26n_saved_model data=coco8.yaml
        ```

## Supported Tasks

Ultralytics TF SavedModel export supports all seven Ultralytics tasks: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with [YOLO26](../models/yolo26.md), the only family that ships those heads. For this page, the YOLO26n checkpoint of every task was exported, run on the export host, and validated on the small dataset of its task (`coco8.yaml`, `coco8-seg.yaml`, `cityscapes8.yaml`, `depth8.yaml`, `imagenet10`, `coco8-pose.yaml`, and `dota8.yaml`). On larger validation sets, the SavedModel reached the accuracy of the PyTorch checkpoint to 0.1 for segmentation (box and mask mAP50-95 42.0 and 34.9 on the last 500 COCO validation images), semantic segmentation (mIoU 71.3 on the 500 Cityscapes validation images), classification (top-1 71.7 on one ImageNet validation image for each of the 1,000 classes), pose estimation (pose mAP50-95 58.7 on the last 500 COCO-Pose validation images), and OBB (mAP50-95 47.2 on the last 500 DOTAv1 validation tiles); detection is in [Benchmarks](#benchmarks), and depth estimation was checked on the 4 images of `depth8.yaml` only. [YOLO11](../models/yolo11.md), [YOLOv10](../models/yolov10.md), [YOLOv8](../models/yolov8.md), [YOLOv5](../models/yolov5.md) (`yolov5nu.pt`), and [RT-DETR](../models/rtdetr.md) detection models were exported and run too.

## Export Arguments

| Argument   | Type             | Default         | Description                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------- | ---------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `format`   | `str`            | `'saved_model'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                   |
| `imgsz`    | `int` or `tuple` | `640`           | Image size of the model input: an integer for square images or a tuple `(height, width)`. Fixed in the SavedModel. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect and segment, `640` for semantic models pretrained on ADE20K and `1024` for those pretrained on Cityscapes, `768` for depth, `224` for classify, `640` for pose, and `1024` for OBB. |
| `quantize` | `int` or `str`   | `None`          | Only `32`, `'fp32'`, `'w32a32'`, or unset (FP32). Every other value fails, see [Precision](#precision). Replaces the deprecated `half`/`int8` flags.                                                                                                                                                                                                                                                                                 |
| `batch`    | `int`            | `1`             | Batch size of the exported model, fixed in the input shape.                                                                                                                                                                                                                                                                                                                                                                          |
| `opset`    | `int`            | `None`          | ONNX opset of the intermediate ONNX model. If not set, the export uses opset 18 (lower with PyTorch older than 2.4), capped at the newest opset that the installed `onnx` supports. For the YOLO26n detection model, opset 8 and 9 failed and 11, 12, 17, 18 (the default), and 20 passed; 10 was not run. RT-DETR needs 16 to 19, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).                          |
| `nms`      | `bool`, optional | `None`          | `None` writes the raw output, which needs NMS for YOLO detection, segmentation, pose, and OBB models; `True` embeds NMS in YOLO detection, segmentation, pose, and OBB models; `False` keeps the NMS-free head of models that have one, such as YOLO26; a model without one, such as YOLO11, logs `This model has no one-to-one head; using one-to-many outputs.` and writes the raw output. See [Output Format](#output-format).    |
| `device`   | `str`            | `None`          | Device that runs the PyTorch model while the export traces it. The exports for this page ran on the CPU; `device=0` failed on the GPU of the export host, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).                                                                                                                                                                                                   |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Precision

TF SavedModel export writes FP32 weights only. Every value of `quantize` other than FP32 stops the export before the conversion:

| `quantize`    | Result                                                                                                      |
| ------------- | ----------------------------------------------------------------------------------------------------------- |
| unset or `32` | FP32 weights. The YOLO26n directory is 24.9 MB                                                              |
| `16`          | Fails with `quantize=16 (FP16) is not supported; format='saved_model' supports quantize=32 or None (FP32).` |
| `8`           | Fails with `quantize=8 (INT8) is not supported; format='saved_model' supports quantize=32 or None (FP32).`  |
| `'w8a16'`     | Fails with `quantize='w8a16' is not supported; format='saved_model' supports quantize=32 or None (FP32).`   |
| `'w8a32'`     | Fails with `quantize='w8a32' is not supported; format='saved_model' supports quantize=32 or None (FP32).`   |

The spellings `'fp32'` and `'w32a32'` export FP32 like `32`, `'fp16'` and `'w16a16'` fail like `16`, and `'int8'` and `'w8a8'` fail like `8`.

- **Calibration**: `data` and `fraction` are not export arguments of this format, and passing them fails with `argument 'data' is not supported for format='saved_model'`.
- **Smaller or integer models**: for a `.tflite` model with INT8 weights or INT8 activations, export [LiteRT](litert.md#quantization) with `format="litert"` and `quantize="w8a32"` or `quantize=8`. For a server that runs FP16 or INT8, see [ONNX](onnx.md) or [TensorRT](tensorrt.md).

## Output Format

The export writes the `yolo26n_saved_model/` directory next to the `.pt` file. In it, `saved_model.pb` holds the graph and, as constants, most of the weights, so `variables/` is small; `onnx2tf` also writes two LiteRT files there. The directory of the YOLO26n detection model holds these files, in MB (10^6 bytes); the export log prints the total in MiB, as `(23.7 MB)` with TensorFlow 2.19.0:

| File                     | Size      | Contents                                                                                      |
| ------------------------ | --------- | --------------------------------------------------------------------------------------------- |
| `saved_model.pb`         | 10.1      | Graph, signature `serving_default`, and constant weights                                      |
| `variables/`             | below 0.1 | `variables.data-00000-of-00001` and `variables.index`, the variables of the graph             |
| `fingerprint.pb`         | below 0.1 | Fingerprint that TensorFlow writes for every SavedModel                                       |
| `assets/`                | empty     | Empty directory that TensorFlow writes for a SavedModel                                       |
| `metadata.yaml`          | below 0.1 | Ultralytics metadata: class names, task, image size, batch size, stride, and export arguments |
| `yolo26n_float32.tflite` | 9.8       | LiteRT model, FP32, written by `onnx2tf`; not needed to run the SavedModel                    |
| `yolo26n_float16.tflite` | 5.0       | LiteRT model with FP16 weights, written by `onnx2tf`; not needed to run the SavedModel        |

The sizes of the whole directory for the official YOLO26n checkpoints at their default export size, in MB:

| Task     | Default `imgsz` | `nms=None` | `nms=False` |
| -------- | --------------- | ---------- | ----------- |
| Detect   | `640`           | 24.9       | 25.8        |
| Segment  | `640`           | 28.1       | 29.3        |
| Semantic | `1024`          | 15.8       | 15.8        |
| Depth    | `768`           | 52.1       | 52.1        |
| Classify | `224`           | 28.3       | 28.3        |
| Pose     | `640`           | 30.2       | 30.8        |
| OBB      | `1024`          | 25.7       | 25.9        |

The model has one input, `images`, with shape `(batch, height, width, 3)`: RGB, channels last (NHWC), `float32`, scaled to 0-1. This is the layout of the ONNX input `(batch, 3, height, width)` transposed, so code that feeds the [ONNX export](onnx.md) must transpose the image before it calls the SavedModel. Ultralytics resizes each image to that size with gray padding (letterbox; classification models are resized and center-cropped instead), and the box coordinates in the output refer to the padded image. The signature is `serving_default`, and the outputs are named `output_0` and, for segmentation, `output_1`. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | `nms=None` (default)                                                | `nms=False`                            | `nms=True`                                                 |
| -------- | --------------- | ------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------- |
| Detect   | `640`           | `(1, 84, 8400)`                                                     | `(1, 300, 6)`                          | `(1, 300, 6)`                                              |
| Segment  | `640`           | `(1, 116, 8400)` and `(1, 160, 160, 32)`                            | `(1, 300, 38)` and `(1, 160, 160, 32)` | `(1, 300, 38)` and `(1, 160, 160, 32)`                     |
| Semantic | `1024`          | `(1, 1024, 1024, 19)`, `float32` score for each class               | Same as `nms=None`                     | Same as `nms=None`                                         |
| Depth    | `768`           | `(1, 768, 768, 1)`, depth map                                       | Same as `nms=None`                     | Same as `nms=None`                                         |
| Classify | `224`           | `(1, 1000)`, class probabilities                                    | Same as `nms=None`                     | Fails, see [Limitations](#limitations-and-troubleshooting) |
| Pose     | `640`           | `(1, 56, 8400)`: 4 box values, 1 class score, then 17 keypoints x 3 | `(1, 300, 57)`                         | `(1, 300, 57)`                                             |
| OBB      | `1024`          | `(1, 20, 21504)`: 4 box values, 15 class scores, then the angle     | `(1, 300, 7)`                          | `(1, 300, 7)`                                              |

- **Raw output (`nms=None`)**: detection, segmentation, pose, and OBB models write shape `(batch, values, anchors)`, the same layout as the [ONNX output](onnx.md#output-format): for each anchor the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS to this output yourself.
- **NMS-free head (`nms=False`)**: YOLO26 detection, segmentation, pose, and OBB models write up to 300 detections per image, each with the box corners x1, y1, x2, y2 in input pixels, the score, the class index, and then the values of the task, so no NMS is needed. OBB rows hold the rotated box center x, center y, width, and height, the score, the class index, and the angle in radians. Keep the rows whose score is above your confidence threshold.
- **Embedded NMS (`nms=True`)**: detection, segmentation, pose, and OBB models write the same layout as `nms=False` after NMS inside the model, with the default `conf=0.25` of the export. Semantic segmentation and depth models log a warning such as `'nms=True' is not valid for semantic models. Forcing 'nms=None'.` and export as with `nms=None`, and classification models fail with this setting.
- **Segmentation**: the mask prototypes `output_1` are channels last, `(1, 160, 160, 32)`; `YOLO()` transposes them to `(1, 32, 160, 160)`, the ONNX layout.
- **Semantic segmentation and depth**: the semantic output is a `float32` score for each class and pixel, channels last, `(1, 1024, 1024, 19)` for the Cityscapes model, where the [ONNX export](onnx.md#output-format) writes a `uint8` class map; take the argmax over the last axis for the class of each pixel. The depth output is a `float32` depth map, `(1, 768, 768, 1)`. Both are the same for every value of `nms`.

The table below compares the outputs on the x86-64 export host with those of the [ONNX](onnx.md) export of the same checkpoint on one image, `bus.jpg`, and for OBB on the aerial image `boats.jpg`, both run through Ultralytics. The `nms=None` column is the largest absolute difference of any output value, where box values are in pixels; the `nms=False` column compares the detections with a score above 0.25:

| Task              | `nms=None` against ONNX                     | `nms=False` against ONNX           |
| ----------------- | ------------------------------------------- | ---------------------------------- |
| Detect            | 0.0036                                      | Same 5 detections, within 0.0002   |
| Segment           | 0.0017                                      | Same 4 detections, within 0.0003   |
| Semantic          | Same class on every pixel                   | Same as `nms=None`                 |
| Depth             | 6.38, mean 0.23, on values from 2.0 to 15.0 | Same as `nms=None`                 |
| Classify          | below 0.0001, same top-1 class              | Same as `nms=None`                 |
| Pose              | 0.0028                                      | Same 4 detections, within 0.0001   |
| OBB (`boats.jpg`) | 0.0058                                      | Same 170 detections, within 0.0001 |

The depth output differs more than the others: on the 4 images of `depth8.yaml`, the SavedModel reached a δ1 of 0.851 against 0.854 for PyTorch and 0.854 for ONNX, and an RMSE of 0.260 against 0.205 and 0.205. Validate a depth model on your own data before you deploy it. On an Apple M4 with macOS, the SavedModel and the ONNX export of YOLO26n both reached a mAP50-95 of 69.2 on `coco8.yaml`; on the x86-64 host, both reached 67.5.

`YOLO("yolo26n_saved_model")` reads the class names, task, image size, batch size, and whether the output is NMS-free from `metadata.yaml`, so you only need these details when you run the model in your own code.

### Run with TensorFlow

The following Python example loads an Ultralytics YOLO26 SavedModel with TensorFlow alone, without PyTorch or the Ultralytics package. It reads the output of an export with `nms=False`, `(1, 300, 6)` for YOLO26n, which needs no NMS, letterboxes the image to 640x640 as Ultralytics does, and maps the boxes back to the original image. Download [`bus.jpg`](https://ultralytics.com/images/bus.jpg) into the working directory first. It ran for this page with TensorFlow 2.19.0 and found the same 5 objects in `bus.jpg` as `YOLO("yolo26n_saved_model")`, with the same classes, scores, and boxes.

```python
import cv2
import numpy as np
import tensorflow as tf

# Export first: yolo export model=yolo26n.pt format=saved_model nms=False
model = tf.saved_model.load("yolo26n_saved_model")
infer = model.signatures["serving_default"]

# Letterbox the image to 640x640 with gray padding, as Ultralytics does
img = cv2.imread("bus.jpg")
h, w = img.shape[:2]
r = min(640 / h, 640 / w)
nh, nw = round(h * r), round(w * r)
top, left = (640 - nh) // 2, (640 - nw) // 2
x = np.full((640, 640, 3), 114, np.uint8)
x[top : top + nh, left : left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
x = x[..., ::-1].astype(np.float32)[None] / 255  # BGR to RGB, NHWC, 0-1

y = infer(images=tf.constant(x))["output_0"].numpy()[0]  # (300, 6): x1, y1, x2, y2, score, class
for x1, y1, x2, y2, score, cls in y[y[:, 4] > 0.25]:
    box = (np.array([x1, y1, x2, y2]) - [left, top, left, top]) / r  # undo the letterbox
    box = box.clip(0, [w, h, w, h])  # clip to the image, as Ultralytics does
    print(int(cls), round(float(score), 2), box.round(1))
```

### Run with TensorFlow Serving

[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) loads a SavedModel from a numbered version directory and serves it over REST on port 8501 and gRPC on port 8500; the gRPC port was not run for this page. To serve an Ultralytics YOLO26 export with the [TensorFlow Serving Docker image](https://www.tensorflow.org/tfx/serving/docker), mount the export directory read-only as version `1`:

```bash
# Export first: yolo export model=yolo26n.pt format=saved_model nms=False
docker run -d --rm --name yolo26n-serving -p 8501:8501 \
  -v "$PWD/yolo26n_saved_model:/models/yolo26n/1:ro" -e MODEL_NAME=yolo26n tensorflow/serving:2.21.0
```

The [REST predict API](https://www.tensorflow.org/tfx/serving/api_rest) takes the letterboxed image as a nested list. This Python client for the Ultralytics YOLO26 model sends `bus.jpg` from the working directory and prints the detections in the coordinates of the original image:

```python
import cv2
import numpy as np
import requests

# Letterbox the image to 640x640 with gray padding, as Ultralytics does
img = cv2.imread("bus.jpg")
h, w = img.shape[:2]
r = min(640 / h, 640 / w)
nh, nw = round(h * r), round(w * r)
top, left = (640 - nh) // 2, (640 - nw) // 2
x = np.full((640, 640, 3), 114, np.uint8)
x[top : top + nh, left : left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
x = x[..., ::-1].astype(np.float32) / 255  # BGR to RGB, HWC, 0-1

resp = requests.post("http://localhost:8501/v1/models/yolo26n:predict", json={"instances": [x.tolist()]})
y = np.array(resp.json()["predictions"][0])  # (300, 6): x1, y1, x2, y2, score, class
for x1, y1, x2, y2, score, cls in y[y[:, 4] > 0.25]:
    box = (np.array([x1, y1, x2, y2]) - [left, top, left, top]) / r  # undo the letterbox
    box = box.clip(0, [w, h, w, h])  # clip to the image, as Ultralytics does
    print(int(cls), round(float(score), 2), box.round(1))
```

With `tensorflow/serving:2.21.0` on the export host, the client printed the same 5 objects as `YOLO("yolo26n_saved_model")`. The server also loaded the directory with its LiteRT files and `metadata.yaml`. Stop the server with `docker stop yolo26n-serving`.

## Deploy

`YOLO("yolo26n_saved_model")` runs the model with TensorFlow in the Python process. This works on the export host too, so you can check a model before you deploy it:

- **CPU threads**: TensorFlow chooses its own thread count, and Ultralytics does not limit it. On the Linux server with 384 logical CPUs, the process ran 778 threads after the Ultralytics backend loaded YOLO26n, and the median of 20 forwards through the Ultralytics backend on a random 640x640 input took 154.1 to 168.4 ms in three runs. Limited with `taskset` to the 8 CPUs of one 8-core CCD of the processor, it ran 26 threads and took 15.6 to 16.1 ms; with 16 CPUs 20.6 to 21.4 ms, with 4 14.5 to 17.4 ms, and with 1 30.7 to 31.3 ms. On a machine with many cores, limit the CPUs of the process, on Linux with `taskset` or Docker `--cpuset-cpus`. Setting the TensorFlow thread counts helps less: limited to 64 CPUs, `tf.config.threading.set_intra_op_parallelism_threads(8)` and `set_inter_op_parallelism_threads(8)` before `YOLO()` lowered the process from 272 to 160 threads, and the median forward took 29.6 to 29.8 ms instead of 32.6 to 32.9 ms in two runs each.
- **`device` and GPUs**: Ultralytics hides the GPUs from TensorFlow and runs the model on the CPU, so `device` does not change where TensorFlow runs. On the export host, with one NVIDIA RTX PRO 6000 Blackwell GPU visible, the prediction with `device="cpu"` and `device=0` found the same 5 objects in `bus.jpg` with TensorFlow 2.19.0 from PyPI. If TensorFlow code in the same process has already started using the GPUs, they stay visible to it; then load the model in a new process, or hide the GPUs with `CUDA_VISIBLE_DEVICES=""`. On an Apple M4, `device="mps"` ran and found the same 5 objects in `bus.jpg` as the CPU.
- **Fixed input shape**: the model runs at its export `imgsz` and `batch`, which Ultralytics reads from `metadata.yaml`. An `imgsz` passed to `predict` or `val` is ignored: YOLO26n exported at 640 gave the same detections with `imgsz=320` and `imgsz=1280`. A list of three images ran on models exported with `batch=1` and `batch=4`, and models exported with `batch=2` and `batch=4` reached the same accuracy on `coco8.yaml` as with `batch=1`.
- **Directory name**: Ultralytics recognizes the format by `_saved_model` in the name of the directory. A copy named `yolo26n_saved_model.bak` or `YOLO26N_SAVED_MODEL` loaded, and one named `detector` or `saved_model` failed with `is not a supported model format`.
- **Files to copy**: `saved_model.pb`, `variables/`, and `metadata.yaml` are enough for `YOLO()`, which ran the model from a directory with only these. Without `metadata.yaml`, the model loads with the class names `class0`, `class1`, and so on; without `variables/`, loading fails. TensorFlow Serving and `tf.saved_model.load` need `saved_model.pb` and `variables/`.
- **LiteRT files in the directory**: `YOLO("yolo26n_saved_model/yolo26n_float32.tflite")` loads, but every box of its raw output ended at the bottom-right corner of `bus.jpg`, because these files write box coordinates in input pixels, as the SavedModel does. For a `.tflite` model, export [LiteRT](litert.md) instead.
- **Several models**: a detection and a segmentation model ran alternately in one process.

### Deployment Guides

- **[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)**: the serving system for SavedModels, with the [Docker guide](https://www.tensorflow.org/tfx/serving/docker) and the [REST API](https://www.tensorflow.org/tfx/serving/api_rest).
- **[SavedModel guide](https://www.tensorflow.org/guide/saved_model)**: how TensorFlow saves, loads, and inspects SavedModels, including `saved_model_cli show`.
- **[Run a TensorFlow SavedModel in Node.js](https://blog.tensorflow.org/2020/01/run-tensorflow-savedmodel-in-nodejs-directly-without-conversion.html)**: a TensorFlow blog post on running a SavedModel in Node.js without conversion.
- **[Deploying on Cloud](https://blog.tensorflow.org/2020/04/how-to-deploy-tensorflow-2-models-on-cloud-ai-platform.html)**: a TensorFlow blog post on deploying a SavedModel on Google Cloud. [Google Cloud](https://cloud.google.com/products/gemini-enterprise-agent-platform), [Amazon SageMaker](https://aws.amazon.com/sagemaker/), and [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/) serve TensorFlow models; these services were not run for this page.
- **[LiteRT](litert.md)**: export a `.tflite` model for phones, embedded boards, and browsers, and **[Raspberry Pi](../guides/raspberry-pi.md)** to run Ultralytics on the board.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare TF SavedModel with the other export formats.

## Limitations and Troubleshooting

- **Automatic installation turned off**: with `YOLO_AUTOINSTALL=false` and a plain `pip install ultralytics`, the export logs `Ultralytics requirements ['onnx>=1.12.0,<2.0.0', 'onnxruntime', 'onnxslim>=0.1.82'] not found, AutoUpdate disabled by YOLO_AUTOINSTALL=False` and fails with `No module named 'onnx'`. Install the packages first with the commands in [Quickstart](#quickstart).
- **Python 3.13: `LLVM ERROR: inconsistency in registered CommandLine options`**: on Python 3.13, with TensorFlow 2.21.0 and PyTorch 2.15.0 from PyPI, importing `torchvision` after TensorFlow stopped the Python process with this error, because `torchvision` loads `triton`, whose LLVM options clash with those of TensorFlow. Ultralytics imports `torchvision` before TensorFlow when it exports or loads the model. If your own code imports TensorFlow before it uses Ultralytics, add `import torchvision` before `import tensorflow`.
- **`protobuf` conflict**: after the first export, pip reports `tensorflow 2.19.0 requires protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3, but you have protobuf 6.33.6 which is incompatible.` on Python 3.10 to 3.12. The export and the inference for this page ran with this combination. With TensorFlow 2.19.0, every import of TensorFlow also logs `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` several times; the exports that logged it completed and gave the same outputs as ONNX.
- **Python 3.9 and 3.14**: the export failed with `No module named 'onnx2tf'` on Python 3.9 and, in the runs for this page, with `No module named 'tf_keras'` on Python 3.14, see [Quickstart](#quickstart).
- **GPU on the export host: `CUDA_ERROR_INVALID_HANDLE`**: with an NVIDIA RTX PRO 6000 Blackwell GPU visible to the process and TensorFlow 2.19.0 from PyPI, the export with `device=0` failed with `CUDA_ERROR_INVALID_HANDLE`. Export without `device=0`.
- **FP16 and INT8**: `quantize=16` fails with `quantize=16 (FP16) is not supported; format='saved_model' supports quantize=32 or None (FP32).`, and so do `8`, `'w8a16'`, and `'w8a32'`. Export [LiteRT](litert.md) for a quantized `.tflite` model.
- **Dynamic shapes**: `dynamic=True` fails with `argument 'dynamic' is not supported for format='saved_model'`. Export again to change the image size or the batch size.
- **`simplify=False`**: fails with `argument 'simplify' is not supported for format='saved_model'`; the export always simplifies the intermediate ONNX model.
- **Minimum opset**: with `opset=8`, the export fails with `ONNX export requires opset>=9 for this model, but got opset=8`, and with `opset=9`, `onnx2tf` fails with `can't split axis of size 8400 into pieces of size [2,2]`. YOLO26n detection exported with opset 11, 12, 17, 18 (the default), and 20; opset 10 and the other tasks were not run.
- **RT-DETR**: `opset` above 19 fails with `RTDETR TensorFlow export requires opset<=19`, and the ONNX step requires 16 or newer. The RT-DETR-L directory is 337.7 MB, and its export took 186.4 s.
- **Accuracy with `nms=True`**: the embedded NMS keeps only boxes above the `conf=0.25` of the export, so `val` reports a lower mAP than for `nms=None` or `nms=False`, as for the ONNX export: on `coco8.yaml`, YOLO26n reached 50.3 with `nms=True`, 67.5 with `nms=None`, and 69.3 with `nms=False`. Pass `conf` to the export to change the threshold.
- **Classification with `nms=True`**: fails with `'nms=True' is not valid for classification models.`
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='saved_model' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends. Export a non-QAT checkpoint to this format instead.` Export the checkpoint you started the fine-tuning from, or export the QAT checkpoint to [ONNX](onnx.md) or [TensorRT](tensorrt.md).
- **Leftover files**: every export leaves the intermediate `yolo26n.onnx` next to the `.pt` file, replacing an existing `yolo26n.onnx`, and the `onnx2tf` sample file `calibration_image_sample_data_20x128x128x3_float32.npy` in the working directory. The SavedModel does not need them.
- **Output directory**: every export of one checkpoint writes `yolo26n_saved_model/` and deletes an existing directory of that name first, whatever its `imgsz`, `batch`, or `nms`. Rename the directory before you export a second variant, and keep `_saved_model` in the name.
- **macOS with Python 3.13 or newer**: according to the Ultralytics code, the export stops with `TensorFlow exports not supported on macOS with Python>=3.13: the ai-edge-litert macOS wheel fails to load (missing libpywrap_litert_common.dylib). TensorFlow export works on Linux Python 3.13.` Use Python 3.12 on macOS. YOLO26n export, prediction, and validation were tested with that version on an Apple M4.
- **Windows and ARM64 Linux**: TensorFlow has packages for both, but exports there were not run for this page.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

TF SavedModel models were benchmarked on the CPU of the export host for this page. The [NVIDIA Jetson guide](../guides/nvidia-jetson.md#nvidia-jetson-yolo26-benchmarks) compares TF SavedModel with the other export formats on Jetson devices.

### FP32 on a Server CPU

The tables below compare PyTorch, the ONNX export, and the TF SavedModel of every YOLO26 detection model on the CPU of the export host, an **AMD EPYC 9655** server shared with other jobs. For the latency table, every process was limited to the 8 CPUs of one 8-core CCD of the processor with `taskset -c 40-47`. The results apply to this x86 setup; TF SavedModel was not benchmarked on a GPU for this page, see [Deploy](#deploy).

=== "Latency"

    Milliseconds per image, lower is better, as the mean of two runs. The fastest format for each model is in bold.

    | Model   | `imgsz` | PyTorch  | ONNX FP32 | TF SavedModel FP32 |
    | ------- | ------- | -------- | --------- | ------------------ |
    | YOLO26n | 640     | **10.1** | 16.4      | 12.7               |
    | YOLO26s | 640     | **26.4** | 29.0      | 26.8               |
    | YOLO26m | 640     | 60.9     | **58.0**  | 65.3               |
    | YOLO26l | 640     | 73.1     | **68.1**  | 81.6               |
    | YOLO26x | 640     | 141.8    | **131.2** | 153.2              |

=== "Accuracy"

    mAP<sup>val</sup> 50-95 on 500 images of COCO val2017, higher is better.

    | Model   | PyTorch | ONNX FP32 | TF SavedModel FP32 |
    | ------- | ------- | --------- | ------------------ |
    | YOLO26n | 42.1    | 42.1      | 42.1               |
    | YOLO26s | 49.5    | 49.5      | 49.5               |
    | YOLO26m | 53.6    | 53.6      | 53.6               |
    | YOLO26l | 55.5    | 55.5      | 55.5               |
    | YOLO26x | 55.7    | 55.7      | 55.7               |

=== "Size"

    Size of the `.pt` checkpoint, the ONNX file, and the whole SavedModel directory, including its two LiteRT files, in MB (10^6 bytes). The smallest for each model is in bold.

    | Model   | PyTorch   | ONNX FP32 | TF SavedModel FP32 |
    | ------- | --------- | --------- | ------------------ |
    | YOLO26n | **5.5**   | 9.9       | 24.9               |
    | YOLO26s | **20.4**  | 38.3      | 95.8               |
    | YOLO26m | **44.3**  | 82.0      | 205.0              |
    | YOLO26l | **53.2**  | 99.6      | 249.1              |
    | YOLO26x | **118.7** | 223.3     | 558.3              |

- **Accuracy**: the TF SavedModel reached the same mAP as PyTorch and ONNX for every model in the table.
- **Latency on 8 CPUs**: through Ultralytics, the TF SavedModel was faster than ONNX for YOLO26n and YOLO26s and took 1.1 to 1.2 times as long as ONNX for YOLO26m to YOLO26x; it took 1.0 to 1.3 times as long as PyTorch.
- **All CPUs**: without `taskset`, on all 384 logical CPUs, YOLO26n took 168.7 ms as a TF SavedModel, against 14.3 ms in PyTorch and 12.3 ms in ONNX, as the mean of two runs; see [Deploy](#deploy) for the threads that TensorFlow ran.

Measured with `ultralytics` 8.4.219, TensorFlow 2.19.0, `onnx2tf` 1.28.8, PyTorch 2.15.0, and ONNX Runtime 1.32.0 at `batch=1`. PyTorch ran with `rect=False`, at the same 640x640 input as the exported models. Accuracy is on the last 500 images of COCO val2017 in sorted order, so it differs from the values for the full validation set on the [YOLO26 model page](../models/yolo26.md). Latency is the inference stage that `val` reports and does not include preprocessing or postprocessing; the PyTorch and ONNX columns were measured again for this page and differ from those on the [MNN page](mnn.md#benchmarks) by up to 8%. The PyTorch and ONNX accuracy cells are 0.1 to 0.2 higher than on the MNN and [PaddlePaddle](paddlepaddle.md#benchmarks) pages, which were measured with `ultralytics` 8.4.218, before 8.4.219 changed the mAP computation to match pycocotools. The two runs differed by up to 13%.

## About TensorFlow SavedModel

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/3f7da5d5bba13367aaf3ba2735b4bb28.avif" alt="TensorFlow SavedModel export format structure">
</p>

[TensorFlow](https://www.tensorflow.org/) is the open-source machine learning framework from Google, released under the Apache 2.0 license. A SavedModel stores a complete TensorFlow program, its graph and its weights, in one directory, and is the format that [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) and the LiteRT converter read, according to the [SavedModel guide](https://www.tensorflow.org/guide/saved_model).

## FAQ

### Can I serve an Ultralytics YOLO26 SavedModel with TensorFlow Serving?

Yes: TensorFlow Serving loads the `yolo26n_saved_model/` directory that Ultralytics writes, mounted as a numbered version directory, and serves its `serving_default` signature over REST, and over gRPC, which was not run for this page. With `tensorflow/serving:2.21.0`, an export with `nms=False` returned the same detections over REST as `YOLO()`, see [Run with TensorFlow Serving](#run-with-tensorflow-serving). The model takes a fixed-size, letterboxed NHWC image, so the client does the preprocessing and maps the boxes back.

### Does an Ultralytics TF SavedModel run on a GPU?

Not with `YOLO()`: Ultralytics hides the GPUs from TensorFlow and runs a TF SavedModel on the CPU, also with `device=0`, unless TensorFlow code in the same process already uses the GPUs, so all results on this page are on the CPU. For NVIDIA GPUs, see the Ultralytics [TensorRT](tensorrt.md) and [ONNX](onnx.md) exports.

### Should I export Ultralytics YOLO26 to TF SavedModel, ONNX, or LiteRT?

Export Ultralytics YOLO26 to TF SavedModel when the model must run in TensorFlow: TensorFlow Serving, a TensorFlow pipeline, or a service that takes SavedModels. It is FP32 only and installs TensorFlow, which the other two formats do not need. [ONNX](onnx.md) runs on more runtimes and supports FP16 and INT8, and [LiteRT](litert.md) is the format for phones, embedded boards, and browsers, with INT8 options and no TensorFlow at export time.

### Why is my Ultralytics YOLO26 SavedModel slow on a server CPU?

An Ultralytics YOLO26 SavedModel can be slow on a server with many cores, where TensorFlow chooses its own thread count and Ultralytics does not limit it: on the 384-CPU export host, YOLO26n took 168.7 ms per image through `YOLO()` on all CPUs and 12.7 ms limited to 8 CPUs with `taskset`, see [Benchmarks](#benchmarks). Limit the CPUs of the process with `taskset` or Docker `--cpuset-cpus` and compare on your machine; setting the TensorFlow thread counts with `tf.config.threading` helped much less in the runs for this page, see [Deploy](#deploy).
