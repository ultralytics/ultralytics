---
comments: true
description: Export Ultralytics YOLO26 models to Google LiteRT (TFLite) with format=litert. Covers w8a32, INT8, and w8a16 quantization, output shapes, Android, browser, and Intel devices.
keywords: YOLO26, LiteRT, LiteRT export, TFLite, TensorFlow Lite, LiteRT.js, INT8 quantization, w8a32, w8a16, Android, on-device inference, edge deployment, WebGPU
---

# Google LiteRT Export

[LiteRT](https://developers.google.com/edge/litert/overview) is Google's runtime for on-device AI and the new name of TensorFlow Lite (TFLite). Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="litert"` creates a single `.tflite` file for Android and iOS phones, embedded Linux boards, and web browsers, in FP32 or with one of three INT8 [quantization](https://www.ultralytics.com/glossary/model-quantization) schemes.

## At a Glance

| Property          | Details                                                                                                                                                                                                                     |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `litert`. The older `tflite` and `tfjs` values still work: they log a deprecation warning and export the same LiteRT file                                                                                                   |
| Output            | `yolo26n.tflite`, one file with the class names, task, and image size stored inside it as [metadata](#output-format); quantized exports are named `yolo26n_w8a32.tflite`, `yolo26n_int8.tflite`, and `yolo26n_w8a16.tflite` |
| Runs on           | Android and iOS devices, embedded Linux boards, desktop CPUs, web browsers through [LiteRT.js](https://developers.google.com/edge/litert/web), and [Intel GPUs and NPUs](#intel-gpu-and-npu-inference)                      |
| Runtime           | LiteRT: the `ai-edge-litert` Python package (2.1.4 or newer), the LiteRT libraries for Android and iOS, or LiteRT.js                                                                                                        |
| Precision         | FP32, `w8a32`, INT8, `w8a16`. YOLO26n is 10.0 MB in FP32, 2.9 MB with `w8a32`, 3.1 MB in INT8, and 3.0 MB with `w8a16`                                                                                                      |
| Export host       | Linux x86-64 or macOS on Apple silicon, with Python 3.10 or newer. The export runs on the CPU and needs no GPU                                                                                                              |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to LiteRT for you, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package with the LiteRT export dependencies:

```bash
pip install "ultralytics[export-litert]"
```

This installs `litert-torch`, `ai-edge-litert`, and `ai-edge-quantizer`. With a plain `pip install ultralytics`, the first export installs the missing packages, and can stop with an error [if that replaces PyTorch](#limitations-and-troubleshooting). The export has these version requirements:

| Package             | Required version                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| Python              | 3.10 or newer                                                                                    |
| `litert-torch`      | 0.10.0 or newer                                                                                  |
| `torch`             | 2.4.0 or newer and below 2.15.0, the range that `litert-torch` 0.10.0 accepts                    |
| `ai-edge-litert`    | 2.3.0 or newer to export; 2.1.4 or newer, also on Python 3.10 or newer, to run an exported model |
| `ai-edge-quantizer` | 0.10.0 or newer, for `quantize="w8a32"`, `quantize=8`, and `"w8a16"`                             |

For other installation methods, see the [installation guide](../quickstart.md). Then export your model and load the exported file to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to LiteRT format
        model.export(format="litert")  # creates 'yolo26n.tflite'

        # Export with INT8 weights and FP32 activations, which needs no calibration data
        model.export(format="litert", quantize="w8a32")  # creates 'yolo26n_w8a32.tflite'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to LiteRT format
        yolo export model=yolo26n.pt format=litert # creates 'yolo26n.tflite'

        # Export with INT8 weights and FP32 activations, which needs no calibration data
        yolo export model=yolo26n.pt format=litert quantize=w8a32 # creates 'yolo26n_w8a32.tflite'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported LiteRT model
        model = YOLO("yolo26n.tflite")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported LiteRT model
        yolo predict model=yolo26n.tflite source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported LiteRT model
        model = YOLO("yolo26n.tflite")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported LiteRT model
        yolo val model=yolo26n.tflite data=coco8.yaml
        ```

## Supported Tasks

LiteRT export supports all seven Ultralytics tasks with [YOLOv8](../models/yolov8.md), [YOLO11](../models/yolo11.md), and [YOLO26](../models/yolo26.md) models: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

## Export Arguments

| Argument   | Type                      | Default    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------- | ------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`                     | `'litert'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                                        |
| `imgsz`    | `int` or `tuple`          | `640`      | Image size of the model input: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect, segment, and pose models and for semantic models pretrained on ADE20K, `1024` for OBB models and for semantic models pretrained on Cityscapes, `768` for depth, and `224` for classify.                                          |
| `quantize` | `int` or `str`            | `None`     | Quantization scheme: `'w8a32'` (INT8 weights, FP32 activations, no calibration), `8` (INT8 weights and activations, calibrated on `data`), or `'w8a16'` (INT8 weights, INT16 activations, calibrated on `data`); `32`/unset is FP32. `16` raises an error, and a checkpoint trained with `quantize=8` cannot be exported to LiteRT, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). Replaces the deprecated `half`/`int8` flags. |
| `data`     | `str`                     | `None`     | Dataset YAML used for calibration with `quantize=8` and `'w8a16'`; classification instead takes a dataset directory or a built-in dataset name. Calibration uses the `val` split unless `split` is set. If omitted, Ultralytics uses the small default dataset for the model task, such as `coco8.yaml` for detection.                                                                                                                                    |
| `fraction` | `float`, `int`, or `list` | `1.0`      | Calibration subset: a ratio, an image count (integers above `1`), or a `[train, val, test]` list, from which the entry of the calibration split is used.                                                                                                                                                                                                                                                                                                  |
| `nms`      | `bool`, optional          | `None`     | Select raw output (`None`, default) or the NMS-free head (`False`, FP32 and `'w8a32'` exports only). Embedded NMS (`True`) is not available for LiteRT. See [output format](#output-format).                                                                                                                                                                                                                                                              |
| `batch`    | `int`                     | `1`        | Batch size of the exported model, fixed in the input shape. Also the batch size used for calibration.                                                                                                                                                                                                                                                                                                                                                     |
| `device`   | `str`                     | `None`     | The export always runs on the CPU, whatever value is passed.                                                                                                                                                                                                                                                                                                                                                                                              |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

!!! note "Image size of the official mobile and web models"

    The official LiteRT, [Core ML](coreml.md), and [QNN](qnn.md) mobile models are exported at `imgsz=640`, and at `imgsz=224` for classification. The OBB, depth, and Cityscapes semantic checkpoints export at a larger size by default, so pass `imgsz=640` to match them.

### Quantization

`quantize` selects one of three INT8 schemes. All of them store the weights as 8-bit integers and differ in how the activations, the values that flow between layers, are stored:

| `quantize`    | Weights | Activations | Calibration data | YOLO26n file           | Size    |
| ------------- | ------- | ----------- | ---------------- | ---------------------- | ------- |
| unset or `32` | FP32    | FP32        | No               | `yolo26n.tflite`       | 10.0 MB |
| `'w8a32'`     | INT8    | FP32        | No               | `yolo26n_w8a32.tflite` | 2.9 MB  |
| `8`           | INT8    | INT8        | Yes              | `yolo26n_int8.tflite`  | 3.1 MB  |
| `'w8a16'`     | INT8    | INT16       | Yes              | `yolo26n_w8a16.tflite` | 3.0 MB  |

- **Inputs and outputs**: the input and the outputs stay `float32` in all four precisions, so inference code written for the FP32 model runs a quantized model exported with the same `nms` layout unchanged.
- **`'w8a32'`**: dynamic-range quantization. It needs no calibration data and supports the [NMS-free head](#output-format) with `nms=False`, and it is the precision of the official mobile models behind the CPU and GPU columns in [Measured Performance](#measured-performance).
- **`8` and `'w8a16'`**: static quantization. The export calibrates on every full batch of the `val` split of `data`, or of the split you set with `split`, and warns below 300 images. Use `fraction` to limit the number of images.
- **Detection head with `quantize=8`**: for detect, segment, pose, and OBB models the output layers of the head and the operations that decode them stay in floating point, because INT8 clips class scores and coarsens box coordinates. `'w8a16'` quantizes these layers too, with 16-bit activations.
- **NMS-free head**: `quantize=8` and `'w8a16'` exports of a YOLO26 or YOLOv10 model use the one-to-many head and return raw predictions that need NMS, also with `nms=False`.
- **FP16**: there is no FP16 export. LiteRT GPU delegates run an FP32 model in [half precision](https://www.ultralytics.com/glossary/half-precision) by default, so `quantize=16` is not needed and raises an error.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        model.export(format="litert", quantize=8, data="coco.yaml", fraction=0.1)  # creates 'yolo26n_int8.tflite'

        # Load the exported INT8 model and run inference
        model = YOLO("yolo26n_int8.tflite")
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        yolo export model=yolo26n.pt format=litert quantize=8 data=coco.yaml fraction=0.1 # creates 'yolo26n_int8.tflite'

        # Run inference with the exported INT8 model
        yolo predict model=yolo26n_int8.tflite source='https://ultralytics.com/images/bus.jpg'
        ```

Quantization changes accuracy, and the effect depends on the model, the task, and the calibration images. Validate a quantized model on your own data before you deploy it.

## Output Format

The export writes one `.tflite` file. The model takes one input with shape `(batch, 3, height, width)`: RGB, channels first (NCHW), `float32`, scaled to 0-1. This differs from the TFLite files of the earlier `format="tflite"` export, which took channels-last (NHWC) images. The `nms` argument and the task set the outputs. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | Output with `nms=None`                                  | Output with `nms=False`                             |
| -------- | --------------- | ------------------------------------------------------- | --------------------------------------------------- |
| Detect   | `640`           | `(1, 84, 8400)`                                         | `(1, 300, 6)`                                       |
| Segment  | `640`           | `(1, 116, 8400)` and `(1, 32, 160, 160)`                | `(1, 300, 38)` and `(1, 32, 160, 160)`              |
| Semantic | `1024`          | `(1, 19, 1024, 1024)`, a score for each class and pixel | Same as `nms=None`                                  |
| Depth    | `768`           | `(1, 1, 768, 768)`, depth map                           | Same as `nms=None`                                  |
| Classify | `224`           | `(1, 1000)`, class probabilities                        | Same as `nms=None`                                  |
| Pose     | `640`           | `(1, 56, 8400)`                                         | `(1, 300, 57)`: 6 box values, then 17 keypoints x 3 |
| OBB      | `1024`          | `(1, 20, 21504)`                                        | `(1, 300, 7)`: 4 box values, score, class, angle    |

- **Raw output (`nms=None`)**: shape `(batch, values, anchors)`. For each anchor: the box center x, center y, width, and height, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). The box values and the keypoint x and y are normalized to 0-1, so multiply them by the input width and height. Apply NMS yourself.
- **NMS-free output (`nms=False`)**: `max_det` (default 300) rows of `x1, y1, x2, y2, score, class` in input pixels, with no NMS step. OBB rows are `center x, center y, width, height, score, class, angle`, with the angle in radians. Filter the rows by score. Available for FP32 and `'w8a32'` exports of models with an NMS-free head.
- **Segmentation**: the 32 mask coefficients are the last 32 values of each anchor or row, and the second output holds the mask prototypes.
- **Semantic segmentation**: the output holds class scores, not the class map of the [ONNX export](onnx.md#output-format), so take the argmax over the class axis in your own code.

`YOLO("yolo26n.tflite")` reads the layout from the file metadata and scales the coordinates, so you only need these details when you run the model in your own code. The metadata is a `metadata.json` entry that the export appends to the file as a ZIP archive, readable with any ZIP library. Its keys include `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, `args`, and, for pose models, `kpt_shape`.

### Run with the LiteRT Interpreter

To run the model outside the Ultralytics package, load it with the interpreter of the `ai-edge-litert` package. Read the input and output shapes from the model instead of relying on tensor names. A segmentation model has two outputs: the 3-D one holds the detections and the 4-D one the mask prototypes.

```python
# Export first: yolo export model=yolo26n.pt format=litert nms=False
import json
import zipfile

import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="yolo26n.tflite")
interpreter.allocate_tensors()

image = np.zeros((1, 3, 640, 640), dtype=np.float32)  # your preprocessed image: RGB, NCHW, 0-1
interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image)
interpreter.invoke()
detections = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]  # shape (300, 6)
assert detections.shape == (300, 6), "export the model with nms=False"
detections = detections[detections[:, 4] > 0.25]  # keep rows above the confidence threshold

with zipfile.ZipFile("yolo26n.tflite") as z:
    names = json.loads(z.read("metadata.json"))["names"]  # {'0': 'person', '1': 'bicycle', ...}
```

## Deploy

`YOLO("yolo26n.tflite")` runs the model with the `ai-edge-litert` interpreter on the CPU, with up to 8 threads. Other `device` values, such as `0` or `mps`, also run on the CPU; `xpu` and `npu` select [Intel hardware](#intel-gpu-and-npu-inference). The [inference devices table](../modes/predict.md#inference-devices) compares these names across formats.

The model always runs at its export `imgsz` and `batch`, which Ultralytics reads from the file metadata. An `imgsz` passed to `predict` or `val` is ignored. Larger batches are split into chunks of the export batch size and smaller ones are padded, so any number of images works.

### Intel GPU and NPU Inference

Exported `.tflite` models run on Intel hardware through the [LiteRT Intel OpenVINO](https://developers.google.com/edge/litert/next/intel) plugin. Pass `device=xpu` for the GPU or `device=npu` for the NPU, and Ultralytics installs `ai-edge-litert>=2.3.0` and `ai-edge-litert-sdk-intel` on first use. The plugin pins its own `openvino` version, so keep it in a separate environment if you also run [OpenVINO](openvino.md) models with a different version. The NPU needs an Intel Core Ultra processor with the [Intel NPU driver](https://github.com/intel/linux-npu-driver) installed; see the [latency measured on a Core Ultra X7 358H](#intel-core-ultra-x7-358h). On a machine without the requested Intel device, the OpenVINO plugin fails to compile the model and LiteRT runs it on its own CPU backend (XNNPACK), not through OpenVINO: on an AMD server CPU, YOLO26n took 29 to 30 ms per image with `device=xpu` and `device=npu`, against 18 ms with `device=cpu`. Ultralytics logs no warning in this case, so compare the latency with `device=cpu` to confirm that the device is in use.

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.tflite")
        results = model("https://ultralytics.com/images/bus.jpg", device="npu")  # or "xpu"
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n.tflite source='https://ultralytics.com/images/bus.jpg' device=npu
        ```

### Deployment Guides

- **[Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app)**: runs LiteRT `.tflite` exports on Android with real-time camera inference, single-image prediction, and GPU acceleration.
- **[@ultralytics/yolo npm package](https://www.npmjs.com/package/@ultralytics/yolo)**: runs LiteRT `.tflite` exports in the browser through [LiteRT.js](https://developers.google.com/edge/litert/web) with WebGPU acceleration and a CPU (WASM) fallback. Install it with `npm i @ultralytics/yolo @litertjs/core`.
- **[MediaTek Dimensity](mediatek.md)**: Android deployment on MediaTek phones, including the NPU.
- **[Qualcomm QNN](qnn.md)** and **[Core ML](coreml.md)**: the formats for Snapdragon NPUs and for Apple devices.
- **[NXP i.MX95](nxp.md)** and **[Coral Edge TPU](edge-tpu.md)**: formats for these accelerators.
- **[Raspberry Pi](../guides/raspberry-pi.md)**: set up Ultralytics on Raspberry Pi boards.
- **LiteRT documentation**: integrate the runtime in [Android](https://developers.google.com/edge/litert/android) and [iOS](https://developers.google.com/edge/litert/ios/quickstart) applications, or [in the browser](https://developers.google.com/edge/litert/web).

## Limitations and Troubleshooting

- **Export host**: on Windows and on Linux ARM64, such as a Raspberry Pi, the export fails with `LiteRT export only supported on Linux x86 and macOS`. Export on a supported host and copy the `.tflite` file to the device.
- **First export after a plain install**: `litert-torch` 0.10.0 requires a PyTorch version from 2.4.0 to below 2.15.0. When the first export installs it into an environment with another PyTorch version, pip replaces PyTorch, Ultralytics logs `Restart runtime or rerun command for updates to take effect`, and that export can fail, with `RuntimeError: duplicate registrations for aten.miopen_batch_norm.default` on PyTorch 2.15.0. Run the export again.
- **FP16**: `quantize=16` fails with `quantize=16 (FP16) is not supported; format='litert' supports quantize=8 (INT8), 'w8a16' (INT8 weights + INT16 activations), 'w8a32' (dynamic INT8), 32 or None (FP32).` Export FP32 and let the GPU delegate run it in half precision.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='litert' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends`. Export the checkpoint you started the fine-tuning from with `quantize=8` instead, or export the fine-tuned checkpoint to [ONNX](onnx.md) or [TensorRT](tensorrt.md).
- **NMS-free head with `quantize=8` or `'w8a16'`**: with `nms=False`, the export logs `LiteRT INT8 export does not support end2end models, disabling end2end branch.` and writes raw predictions that need NMS. Use FP32 or `'w8a32'` to keep the NMS-free head.
- **Embedded NMS**: `nms=True` logs `format=litert does not support embedded NMS; exporting native outputs for external NMS.` and writes the raw output.
- **NMS-free head on other models**: `nms=False` needs a model with an NMS-free head, such as YOLO26 or YOLOv10. For other models, for example YOLO11, the export warns `This model has no one-to-one head; using one-to-many outputs.` and writes the raw `(1, 84, 8400)` output.
- **Fixed input shape**: `dynamic=True` fails with `argument 'dynamic' is not supported for format='litert'`. Export again to change the image size or the batch size.
- **Calibration without `data`**: a `quantize=8` or `'w8a16'` export logs `INT8 export requires a missing 'data' arg for calibration. Using default 'data=coco8.yaml'.` and calibrates on the small default dataset for the task. Pass `data` with images that represent your deployment data.
- **Calibration size**: the export logs `>300 images recommended for INT8 calibration, found 4 images.` when the calibration split has fewer than 300 images. `coco8.yaml` has only 4 validation images, which is enough to test the export but not to calibrate a model for deployment.
- **`'w8a16'` on x86 CPUs**: the LiteRT CPU interpreter ran a YOLO26n `'w8a16'` model at about 1 s per image on an AMD EPYC 9655 server CPU with 8 threads, against about 17 ms for the FP32 model. Measure `'w8a16'` on the device and delegate you deploy to.
- **File name**: every export of one checkpoint with the same `quantize` value writes the same file, such as `yolo26n.tflite` for FP32, whatever its `imgsz`, `batch`, or `nms`, so the later export replaces the earlier one.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Measured Performance

The tables below give the latency of one image for the YOLO26n model of each task on four Android phones and on an Intel Core Ultra processor. Each cell shows the total time in milliseconds with the preprocess / inference / postprocess split beneath it, and the fastest column for each model is in bold. The [YOLO26 model page](../models/yolo26.md) lists the accuracy of the PyTorch checkpoints.

### Xiaomi 17

**Hardware:** [Xiaomi 17](https://www.mi.com/global/product/xiaomi-17/) with 12 GB LPDDR5X memory and Android 17 /
API 37. Its 3 nm [Snapdragon 8 Elite Gen 5](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-gen-5)
(SM8850) has an 8-core Qualcomm Oryon CPU (2 Prime cores up to 4.6 GHz and 6 Performance cores up to 3.62 GHz),
Adreno GPU, and Hexagon NPU.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup>  | NPU<br><sup>LiteRT FP32<br>(ms)</sup>    |
| ------------- | -------- | --------------------------- | -------------------------------------- | --------------------------------------- | ---------------------------------------- |
| YOLO26n       | Detect   | 640                         | 36.9<br><sup>1.9 / 32.6 / 2.3</sup>    | 10.6<br><sup>1.8 / 6.2 / 2.6</sup>      | **10.1**<br><sup>3.5 / 4.0 / 2.5</sup>   |
| YOLO26n-seg   | Segment  | 640                         | 50.5<br><sup>1.9 / 42.9 / 5.8</sup>    | **18.0**<br><sup>1.9 / 9.5 / 6.6</sup>  | 19.2<br><sup>3.6 / 5.9 / 9.7</sup>       |
| YOLO26n-sem   | Semantic | 640                         | 45.7<br><sup>1.9 / 35.7 / 8.2</sup>    | 30.6<br><sup>1.9 / 20.5 / 8.3</sup>     | **25.8**<br><sup>3.6 / 11.0 / 11.3</sup> |
| YOLO26n-depth | Depth    | 640                         | 66.8<br><sup>2.2 / 55.0 / 9.7</sup>    | **21.9**<br><sup>2.1 / 10.1 / 9.8</sup> | 24.8<br><sup>4.3 / 6.0 / 14.5</sup>      |
| YOLO26n-cls   | Classify | 224                         | 3.6<br><sup>0.3 / 3.3 / 0.0</sup>      | 2.0<br><sup>0.4 / 1.4 / 0.2</sup>       | **1.5**<br><sup>0.6 / 0.7 / 0.3</sup>    |
| YOLO26n-pose  | Pose     | 640                         | 39.9<br><sup>2.0 / 36.5 / 1.4</sup>    | **10.0**<br><sup>1.8 / 6.7 / 1.4</sup>  | 11.5<br><sup>3.5 / 4.2 / 3.8</sup>       |
| YOLO26n-obb   | OBB      | 640                         | 36.3<br><sup>2.0 / 33.2 / 1.2</sup>    | 10.1<br><sup>2.6 / 5.9 / 1.6</sup>      | **8.2**<br><sup>3.5 / 3.7 / 0.9</sup>    |

- **Speed** values are **single-image burst latencies** — the mean of 15 runs after 3 warmup runs on `bus.jpg`, measured with the [Ultralytics Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) `0.7.1` and the official `v0.7.0` assets (Ultralytics 8.4.188 exports). CPU/GPU order alternated between tasks in one sequential sweep. Native logs confirmed that every CPU row used LiteRT CPU/XNNPACK and every GPU row delegated the complete graph to LiteRT OpenCL (`LITERT_CL`).
- The LiteRT export traces the PyTorch model directly, producing an **NCHW** `.tflite` with a float input — the GPU delegate compiles the whole graph (all seven tasks run on the Adreno GPU here), and `w8a32` needs no calibration data. Consumers should read tensor shapes and signature names instead of assuming the legacy onnx2tf NHWC layout or `Identity` output names; pack RGB data directly as planar CHW or transpose it before inference. Semantic exports return NCHW logits and require a host-side class argmax. The official Android assets are hosted on the [yolo-flutter-app `v0.7.0` release](https://github.com/ultralytics/yolo-flutter-app/releases/tag/v0.7.0), with the detailed benchmark record in [the Flutter performance doc](https://github.com/ultralytics/yolo-flutter-app/blob/main/doc/performance.md).
- The **NPU** column is LiteRT's own NPU path: an FP32 export (`yolo export format=litert`, no `quantize`) compiled on the
  device by the `litert-npu-runtime-qualcomm` and `-mediatek` 2.3.0 runtimes, measured with a prototype NPU build of the Flutter plugin (not yet released).
  The first load includes that compile (about 0.3 s for `n`, up to 6 s for `x`). The `w8a32` assets do not compile fully on these
  NPUs. LiteRT reports an NPU compile even when it silently runs the graph on the GPU, so every NPU row was checked against
  the GPU time; rows that fell back are shown as —.
- The matching Snapdragon **Hexagon NPU** numbers for the official QNN assets are in the [Qualcomm QNN integration](qnn.md).
- Compare the Apple CPU/accelerator results in the [CoreML integration](coreml.md#measured-performance).

The following device sweeps use the same official `v0.7.0` assets.

### Google Pixel 10

**Hardware:** [Google Pixel 10](https://store.google.com/product/pixel_10_specs) with 12 GB memory and Android 17 /
API 37. Its 3 nm [Google Tensor G5](https://blog.google/products-and-platforms/devices/pixel/tensor-g5-pixel-10/) has
an 8-core CPU (1 Prime core up to 3.78 GHz, 5 Performance cores up to 3.05 GHz, and 2 Efficiency cores up to
2.25 GHz), PowerVR D-Series GPU, and Google TPU. Core clocks and the GPU driver name were read from the benchmark
device because Google does not publish them in the linked specifications.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup>  | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup>   |
| ------------- | -------- | --------------------------- | --------------------------------------- | ---------------------------------------- |
| YOLO26n       | Detect   | 640                         | 36.6<br><sup>1.5 / 33.5 / 1.6</sup>     | **36.5**<br><sup>3.5 / 28.4 / 4.6</sup>  |
| YOLO26n-seg   | Segment  | 640                         | 53.1<br><sup>1.5 / 45.8 / 5.7</sup>     | **51.5**<br><sup>1.6 / 40.1 / 9.9</sup>  |
| YOLO26n-sem   | Semantic | 640                         | **57.2**<br><sup>1.5 / 47.7 / 8.0</sup> | 75.9<br><sup>1.8 / 63.4 / 10.8</sup>     |
| YOLO26n-depth | Depth    | 640                         | 75.0<br><sup>1.6 / 65.1 / 8.3</sup>     | **50.1**<br><sup>1.7 / 33.5 / 14.9</sup> |
| YOLO26n-cls   | Classify | 224                         | **6.1**<br><sup>0.3 / 5.5 / 0.3</sup>   | 12.1<br><sup>0.9 / 11.0 / 0.1</sup>      |
| YOLO26n-pose  | Pose     | 640                         | **35.2**<br><sup>1.5 / 31.9 / 1.8</sup> | 38.2<br><sup>3.5 / 29.3 / 5.4</sup>      |
| YOLO26n-obb   | OBB      | 640                         | **32.7**<br><sup>1.5 / 29.3 / 1.9</sup> | 38.8<br><sup>3.8 / 29.6 / 5.4</sup>      |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.7.1` and the
official `v0.7.0` assets. CPU/GPU order alternates between tasks in one sequential sweep. Native logs confirmed that
every CPU row used LiteRT CPU/XNNPACK and every GPU row delegated the complete graph to LiteRT OpenCL (`LITERT_CL`).
The Tensor NPU has no LiteRT row: Google's EdgeTPU service rejects apps that are not on its allowlist, and LiteRT
then runs the model on the GPU.

### Samsung Galaxy S26

**Hardware:** [Samsung Galaxy S26](https://www.samsung.com/uk/smartphones/galaxy-s26/) (SM-S942B) with 12 GB memory
and Android 17 / API 37. Its 2 nm [Exynos 2600](https://semiconductor.samsung.com/processor/mobile-processor/exynos-2600/)
has a 10-core Armv9.3 CPU (1 C1-Ultra core up to 3.8 GHz, 3 performance C1-Pro cores up to 3.26 GHz, and 6 efficiency
C1-Pro cores up to 2.76 GHz), Xclipse 960 GPU, and Samsung NPU.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup>  | NPU<br><sup>LiteRT w8a32 AOT<br>(ms)</sup> |
| ------------- | -------- | --------------------------- | -------------------------------------- | --------------------------------------- | ------------------------------------------ |
| YOLO26n       | Detect   | 640                         | 30.2<br><sup>1.3 / 27.3 / 1.6</sup>    | 15.9<br><sup>1.4 / 11.5 / 2.9</sup>     | **12.2**<br><sup>1.3 / 8.6 / 2.2</sup>     |
| YOLO26n-seg   | Segment  | 640                         | 45.6<br><sup>1.3 / 39.1 / 5.2</sup>    | 30.5<br><sup>1.3 / 23.6 / 5.6</sup>     | **17.0**<br><sup>1.4 / 9.7 / 5.8</sup>     |
| YOLO26n-sem   | Semantic | 640                         | 38.1<br><sup>1.3 / 28.6 / 8.2</sup>    | **26.1**<br><sup>1.3 / 16.8 / 7.9</sup> | 28.3<br><sup>1.7 / 15.1 / 11.5</sup>       |
| YOLO26n-depth | Depth    | 640                         | 63.5<br><sup>1.4 / 53.6 / 8.4</sup>    | 37.3<br><sup>1.4 / 24.6 / 11.3</sup>    | **28.4**<br><sup>2.1 / 12.9 / 13.4</sup>   |
| YOLO26n-cls   | Classify | 224                         | 3.8<br><sup>0.2 / 3.4 / 0.2</sup>      | **2.8**<br><sup>0.2 / 2.6 / 0.0</sup>   | 6.1<br><sup>0.5 / 5.2 / 0.4</sup>          |
| YOLO26n-pose  | Pose     | 640                         | 31.1<br><sup>1.3 / 28.4 / 1.4</sup>    | 18.4<br><sup>1.6 / 13.0 / 3.8</sup>     | **13.5**<br><sup>1.8 / 9.5 / 2.2</sup>     |
| YOLO26n-obb   | OBB      | 640                         | 27.9<br><sup>1.3 / 25.7 / 0.9</sup>    | 15.3<br><sup>1.9 / 11.4 / 2.1</sup>     | **11.4**<br><sup>1.7 / 8.3 / 1.4</sup>     |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.7.1` and the
official `v0.7.0` assets. CPU/GPU order alternates between tasks in one sequential sweep. Native logs confirmed that
every CPU row used LiteRT CPU/XNNPACK and every GPU row delegated the complete graph to LiteRT OpenCL (`LITERT_CL`).
The NPU column uses the `w8a32` assets compiled ahead of time for the Exynos 2600 (`E9965`) with LiteRT's Samsung
compiler plugin. Samsung has no on-device compiler in LiteRT 2.3, and Google does not yet publish a Samsung NPU runtime,
so these rows used a dispatch library built from the LiteRT 2.3 sources.

### Xiaomi 17T Pro

**Hardware:** [Xiaomi 17T Pro](https://www.mi.com/global/product/xiaomi-17t-pro/specs/) (2602EPTC0G) with 12 GB
LPDDR5X memory and Android 16 / API 36. Its 3 nm
[MediaTek Dimensity 9500](https://www.mediatek.com/products/smartphones/mediatek-dimensity-9500) (MT6993) has an
8-core Armv9.3 CPU (1 C1-Ultra core up to 4.21 GHz, 3 C1-Premium cores up to 3.5 GHz, and 4 C1-Pro cores up to
2.7 GHz), Mali-G1 Ultra MC12 GPU, and MediaTek NPU 990.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | GPU<br><sup>w8a32 LiteRT<br>(ms)</sup>   | NPU<br><sup>LiteRT FP32<br>(ms)</sup>    |
| ------------- | -------- | --------------------------- | -------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| YOLO26n       | Detect   | 640                         | 31.4<br><sup>1.5 / 28.0 / 1.9</sup>    | 25.6<br><sup>1.4 / 22.2 / 2.0</sup>      | **11.0**<br><sup>1.3 / 7.8 / 1.8</sup>   |
| YOLO26n-seg   | Segment  | 640                         | 77.8<br><sup>2.6 / 65.3 / 9.9</sup>    | 47.4<br><sup>2.6 / 33.6 / 11.3</sup>     | **36.5**<br><sup>2.6 / 23.4 / 10.4</sup> |
| YOLO26n-sem   | Semantic | 640                         | 81.3<br><sup>2.6 / 62.2 / 16.5</sup>   | **79.3**<br><sup>2.6 / 59.5 / 17.1</sup> | 87.6<br><sup>2.6 / 68.3 / 16.7</sup>     |
| YOLO26n-depth | Depth    | 640                         | 111.3<br><sup>3.4 / 91.1 / 16.8</sup>  | **55.3**<br><sup>2.8 / 36.3 / 16.1</sup> | —                                        |
| YOLO26n-cls   | Classify | 224                         | 6.3<br><sup>0.5 / 5.4 / 0.4</sup>      | 8.1<br><sup>0.5 / 7.5 / 0.1</sup>        | **3.2**<br><sup>0.5 / 2.2 / 0.5</sup>    |
| YOLO26n-pose  | Pose     | 640                         | 53.9<br><sup>2.6 / 49.5 / 1.8</sup>    | 32.8<br><sup>2.7 / 26.7 / 3.4</sup>      | **16.2**<br><sup>2.9 / 11.1 / 2.3</sup>  |
| YOLO26n-obb   | OBB      | 640                         | 49.5<br><sup>2.7 / 44.4 / 2.4</sup>    | **29.4**<br><sup>2.8 / 22.7 / 3.9</sup>  | —                                        |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics_yolo` `0.7.1` and the
official `v0.7.0` assets. CPU/GPU order alternates between tasks in one sequential sweep. Native logs confirmed that
every CPU row used LiteRT CPU/XNNPACK and every GPU row delegated the complete graph to LiteRT OpenCL (`LITERT_CL`).
Depth and OBB have no NPU result: the MediaTek NPU compiler rejects their exponential and trigonometric ops, and LiteRT
runs those graphs on the GPU.

For the Android deployment workflow and NPU limitations, see the [MediaTek Dimensity guide](mediatek.md).

### Intel Core Ultra X7 358H

**Hardware:** Intel Core Ultra X7 358H (Panther Lake) and Intel AI Boost NPU 5010, Fedora Linux, NPU driver 1.38.0, `ai-edge-litert` 2.3.0.

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>w8a32 LiteRT<br>(ms)</sup> | CPU<br><sup>w8a32 Intel<br>(ms)</sup> | NPU<br><sup>w8a32 Intel<br>(ms)</sup>   |
| ------------- | -------- | --------------------------- | -------------------------------------- | ------------------------------------- | --------------------------------------- |
| YOLO26n       | Detect   | 640                         | 17.5<br><sup>0.8 / 16.1 / 0.6</sup>    | 22.6<br><sup>1.2 / 20.6 / 0.8</sup>   | **4.8**<br><sup>1.0 / 3.3 / 0.4</sup>   |
| YOLO26n-seg   | Segment  | 640                         | 24.2<br><sup>0.9 / 22.1 / 1.3</sup>    | 32.6<br><sup>1.0 / 30.1 / 1.5</sup>   | **6.2**<br><sup>0.8 / 4.2 / 1.2</sup>   |
| YOLO26n-sem   | Semantic | 640                         | 25.2<br><sup>1.0 / 14.2 / 10.0</sup>   | 33.1<br><sup>1.1 / 21.9 / 10.2</sup>  | **16.5**<br><sup>1.0 / 5.5 / 10.0</sup> |
| YOLO26n-depth | Depth    | 640                         | 31.7<br><sup>0.9 / 30.2 / 0.5</sup>    | 76.6<br><sup>0.9 / 75.2 / 0.5</sup>   | **9.1**<br><sup>1.0 / 7.5 / 0.5</sup>   |
| YOLO26n-cls   | Classify | 224                         | 4.6<br><sup>2.8 / 1.7 / 0.0</sup>      | 7.3<br><sup>3.3 / 3.9 / 0.0</sup>     | **4.1**<br><sup>3.2 / 0.9 / 0.0</sup>   |
| YOLO26n-pose  | Pose     | 640                         | 20.3<br><sup>0.8 / 19.0 / 0.5</sup>    | 28.6<br><sup>1.1 / 26.8 / 0.8</sup>   | **6.3**<br><sup>0.8 / 4.9 / 0.5</sup>   |
| YOLO26n-obb   | OBB      | 640                         | 17.2<br><sup>0.8 / 16.1 / 0.3</sup>    | 22.1<br><sup>1.2 / 20.5 / 0.4</sup>   | **4.6**<br><sup>0.9 / 3.4 / 0.3</sup>   |

**Benchmark:** Mean of 15 `predict()` calls after 3 warmups on `bus.jpg`, using `ultralytics` 8.4.174 and `ai-edge-litert` 2.3.0 with `w8a32` exports. Each cell shows total time with the preprocess / inference / postprocess split beneath it.

## About LiteRT

LiteRT, short for Lite Runtime, continues TensorFlow Lite under a new name and runs the same `.tflite` model format. It executes models on the CPU with XNNPACK and on GPUs and NPUs through hardware delegates, and LiteRT.js brings the same file to [browsers and Node.js](https://www.ultralytics.com/blog/export-ultralytics-yolo-with-litert-for-edge-and-browser-deployment). For the full reference, see the [LiteRT documentation](https://developers.google.com/edge/litert/overview).

<p align="center">
  <img width="75%" src="https://cdn.ul.run/i/77f69836dfc1d1e96a7e9c9968fa98fd.avif" alt="LiteRT edge deployment framework">
</p>

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/FmgrfmZlhpY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Export Ultralytics YOLO26 to Google LiteRT | Deploy Vision AI on Android & iOS | Mobile AI 📱🚀
</p>

## FAQ

### What is the difference between LiteRT, TFLite, and TF.js?

LiteRT is the new name of TensorFlow Lite, with the same `.tflite` model format. In Ultralytics, `format="litert"` replaces two earlier export formats: `tflite` for mobile, embedded, and edge devices, and `tfjs` for browsers and Node.js, where LiteRT.js now runs the same `.tflite` file. Both earlier values still export a LiteRT file with a deprecation warning. The [TFLite](tflite.md) and [TF.js](tfjs.md) pages describe the earlier formats.

### Which quantization should I use for a LiteRT model?

Start with `quantize="w8a32"`. It needs no calibration data, makes the file less than a third of the FP32 size, supports the NMS-free head of YOLO26 with `nms=False`, and is the precision of the official Ultralytics mobile models for CPU and GPU. Use `quantize=8` when the target needs integer activations: the [NXP i.MX95](nxp.md) export, for example, compiles a calibrated INT8 LiteRT model. `'w8a16'` keeps more activation precision than `8`. Both need representative [calibration data](#quantization) and return raw predictions that need NMS. Measure latency and accuracy on your target device before you choose, because support for each scheme differs between CPU, GPU, and NPU delegates.

### Can I export a LiteRT model on a Raspberry Pi or on Windows?

No. The export runs only on Linux x86-64 and on macOS with Apple silicon. Export the model on one of these hosts, or with [Ultralytics Platform](../platform/train/models.md#export-model), and copy the `.tflite` file to the device. See the [Raspberry Pi guide](../guides/raspberry-pi.md) for running models on the board, where a [Coral Edge TPU](edge-tpu.md) can accelerate inference.

### Can I run a YOLO26 LiteRT model in a web browser?

Yes. [LiteRT.js](https://developers.google.com/edge/litert/web) runs the exported `.tflite` file in a browser or in Node.js with WebGPU or WASM, so there is no separate browser export. The [@ultralytics/yolo npm package](https://www.npmjs.com/package/@ultralytics/yolo) wraps it with webcam and single-image inference for the detect, segment, semantic, classify, pose, and OBB tasks.

### How do I run a YOLO26 LiteRT model in an Android app?

Use the [Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app), which loads LiteRT `.tflite` exports on Android and covers all seven YOLO26 tasks. Export at `imgsz=640`, or `imgsz=224` for classification, to match the official models. For a native app, follow the [LiteRT Android guide](https://developers.google.com/edge/litert/android) and feed the model as described in [Output Format](#output-format). On iPhone, [Core ML](coreml.md) is the native format.
