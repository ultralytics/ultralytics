---
comments: true
description: Export Ultralytics YOLO26 models to Apple Core ML with format=coreml. Covers FP16 and 8-bit weights, embedded NMS, output shapes, Neural Engine inference on iPhone, iPad, and Mac.
keywords: YOLO26, Core ML, CoreML, Core ML export, mlpackage, ML Program, Apple Neural Engine, coremltools, FP16, INT8, palettization, iOS, macOS, iPhone, Vision framework, on-device inference
---

# Apple Core ML Export

[Core ML](https://developer.apple.com/documentation/coreml) is Apple's framework for running [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models on iPhone, iPad, Mac, Apple Watch, Apple TV, and Apple Vision Pro. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="coreml"` creates a `yolo26n.mlpackage` model package that Core ML runs on the CPU, the GPU, and the Apple Neural Engine, in FP32, in FP16, or with 8-bit weights.

!!! note "Apple's new Core AI format"

    Apple has introduced the new [Core AI framework and `.aimodel` format](coreai.md) for the iOS 27 and macOS 27 generation, and Ultralytics exports it with `format="coreai"`. Core ML remains the default format of the Ultralytics iOS and Flutter SDKs, which load Core AI as an opt-in on iOS 27 devices, and the format for earlier Apple operating systems.

## At a Glance

| Property          | Details                                                                                                                                                                                                                             |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `coreml` (aliases: `mlpackage`, `mlprogram`, `apple`, `ios`). `mlmodel` exports the earlier [neural network format](#limitations-and-troubleshooting)                                                                               |
| Output            | `yolo26n.mlpackage`, a model package (a directory) in the ML Program format, with the class names, task, and image size stored inside it as [metadata](#output-format)                                                              |
| Runs on           | iPhone, iPad, Mac, Apple Watch, Apple TV, and Apple Vision Pro, on the CPU, the GPU, and the Apple Neural Engine                                                                                                                    |
| Runtime           | Core ML, which is built into Apple's operating systems; the exported ML Program declares Core ML specification 6 (iOS 15, macOS 12) or, for INT8, specification 7 (iOS 16, macOS 13). In Python, the `coremltools` package on macOS |
| Precision         | FP32, FP16, INT8 (8-bit weights). YOLO26n is 9.9 MB in FP32, 5.1 MB in FP16, and 2.7 MB in INT8                                                                                                                                     |
| Export host       | macOS, or Linux x86-64, with Python 3.13 or older. Not Windows. The export needs no GPU                                                                                                                                             |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to Core ML for you, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package with the Core ML export dependencies, on macOS or on x86-64 Linux:

```bash
pip install "ultralytics[export-coreml]"
```

This installs [`coremltools`](https://apple.github.io/coremltools/docs-guides/), `scikit-learn`, and a `numpy` version that `coremltools` works with. With a plain `pip install ultralytics`, the first export installs `coremltools`, and when it also has to replace a newer `numpy`, that export [fails](#limitations-and-troubleshooting); the next one passes. The export has these version requirements:

| Package        | Required version                                                                                                                    |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Python         | 3.13 or older, the newest version that `coremltools` 9.0 has prebuilt packages for                                                  |
| `coremltools`  | 9.0 or newer                                                                                                                        |
| `numpy`        | 2.3.5 or older. The export installs 2.3.5 over a newer version                                                                      |
| `torch`        | 1.11 or newer                                                                                                                       |
| `scikit-learn` | For `quantize=8` and `'w8a16'` only. The `export-coreml` extra installs 1.3.2 or newer, and the export installs it if it is missing |

For other installation methods, see the [installation guide](../quickstart.md). Then export your model and load the exported package to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to Core ML format
        model.export(format="coreml")  # creates 'yolo26n.mlpackage'

        # Or export in FP16 instead, which halves the size and runs on the Neural Engine
        model.export(format="coreml", quantize=16)  # creates 'yolo26n.mlpackage'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to Core ML format
        yolo export model=yolo26n.pt format=coreml # creates 'yolo26n.mlpackage'

        # Or export in FP16 instead, which halves the size and runs on the Neural Engine
        yolo export model=yolo26n.pt format=coreml quantize=16 # creates 'yolo26n.mlpackage'
        ```

Both exports write the same `yolo26n.mlpackage`, so the second one replaces the first. Predict and Validate run only on macOS:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported Core ML model on a Mac
        model = YOLO("yolo26n.mlpackage")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported Core ML model on a Mac
        yolo predict model=yolo26n.mlpackage source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported Core ML model on a Mac
        model = YOLO("yolo26n.mlpackage")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported Core ML model on a Mac
        yolo val model=yolo26n.mlpackage data=coco8.yaml
        ```

## Supported Tasks

Core ML export supports all seven Ultralytics tasks with [YOLOv8](../models/yolov8.md), [YOLO11](../models/yolo11.md), and [YOLO26](../models/yolo26.md) models: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads. Detection models of YOLOv5u, YOLOv9, YOLOv10, and YOLO12 export too, as do [RT-DETR](../models/rtdetr.md) and [YOLO-Worldv2](../models/yolo-world.md) models. Each was tested with one checkpoint per task, for example `yolo11n-seg.pt`, `rtdetr-l.pt`, and `yolov8s-worldv2.pt`.

## Export Arguments

| Argument       | Type             | Default    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| -------------- | ---------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`       | `str`            | `'coreml'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                          |
| `imgsz`        | `int` or `tuple` | `640`      | Image size of the model input, fixed in a default export: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect, segment, and pose models and for semantic models pretrained on ADE20K, `1024` for OBB models and for semantic models pretrained on Cityscapes, `768` for depth, and `224` for classify. |
| `quantize`     | `int` or `str`   | `None`     | Precision: `16` (FP16) or `8` (8-bit weights with FP16 computation; `'w8a16'` is the same export); `32`/unset is FP32. With `nms=True`, unset is FP16. A checkpoint trained with `quantize=8` cannot be exported to Core ML, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). Replaces the deprecated `half`/`int8` flags.                                                                                          |
| `nms`          | `bool`, optional | `None`     | Select raw output (`None`, default), embedded NMS (`True`, for detect, segment, and pose models with `dynamic=False`), or the NMS-free head (`False`). See [output format](#output-format).                                                                                                                                                                                                                                                 |
| `conf`         | `float`          | `None`     | Confidence threshold stored in an `nms=True` export. Uses `0.25` when unset.                                                                                                                                                                                                                                                                                                                                                                |
| `iou`          | `float`          | `0.5`      | IoU threshold stored in an `nms=True` export.                                                                                                                                                                                                                                                                                                                                                                                               |
| `agnostic_nms` | `bool`           | `False`    | Class-agnostic selection: in an `nms=True` export a box suppresses overlapping boxes of other classes too. It also changes the selection of the rows of an `nms=False` export.                                                                                                                                                                                                                                                              |
| `max_det`      | `int`            | `300`      | Number of rows in the output of an `nms=False` export and of an `nms=True` segmentation or pose export. An `nms=True` detection export has no such limit.                                                                                                                                                                                                                                                                                   |
| `dynamic`      | `bool`           | `False`    | Makes the batch size, height, and width of the input dynamic, and changes the input from an image to an array. Not available for classification and RT-DETR models or with `nms=True`. See [Deploy](#deploy).                                                                                                                                                                                                                               |
| `batch`        | `int`            | `1`        | Largest batch size of a `dynamic=True` export. A default export has a batch size of 1, and other values raise an error.                                                                                                                                                                                                                                                                                                                     |
| `device`       | `str`            | `None`     | Device that traces the model for the conversion, `cpu` by default. An FP32 export traced with `mps` had the same package size and returned the same outputs as one traced on the CPU.                                                                                                                                                                                                                                                       |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

!!! note "Image size and settings of the official mobile models"

    The official Core ML, [LiteRT](litert.md), and [QNN](qnn.md) mobile models are exported at `imgsz=640`, and at `imgsz=224` for classification. The Cityscapes semantic, depth, and OBB checkpoints export at a larger size by default, so pass `imgsz=640` to match them. The Core ML models of the [Ultralytics iOS SDK](https://github.com/ultralytics/yolo-ios-app) are exported with `quantize=8 nms=False`.

### Precision

`quantize` selects how the weights are stored and the precision Core ML computes in. None of the three precisions needs calibration data. The sizes are for a YOLO26n detection model exported at `imgsz=640`:

| `quantize`       | Weights | Computation | Minimum OS       | YOLO26n size |
| ---------------- | ------- | ----------- | ---------------- | ------------ |
| unset or `32`    | FP32    | FP32        | iOS 15, macOS 12 | 9.9 MB       |
| `16`             | FP16    | FP16        | iOS 15, macOS 12 | 5.1 MB       |
| `8` or `'w8a16'` | 8-bit   | FP16        | iOS 16, macOS 13 | 2.7 MB       |

- **FP32**: the default when `nms=True` is not set. On an Apple M4, the FP32 exports of all seven tasks returned the same outputs with the `CPU_AND_NE` compute units, which Ultralytics uses, as with `CPU_ONLY`, and were not faster with them. Export in FP16 or INT8 to use the Neural Engine. On an iPhone 17 Pro, an FP32 YOLO26n detection model was at most 5% faster with the Neural Engine by the median of three runs, see [Measured Performance](#iphone-17-pro).
- **FP16**: `quantize=16` stores the weights and computes in [half precision](https://www.ultralytics.com/glossary/half-precision). The image input and the output arrays keep the types of the FP32 export (`float32`, or `int32` for semantic segmentation), so inference code written for the FP32 model runs it unchanged.
- **INT8**: `quantize=8` and `quantize="w8a16"` are the same export. It groups the weights of each layer into 256 values with k-means and stores an 8-bit index for every weight ([palettization](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html)), and computes in FP16 like an FP16 export. Weight tensors with 512 values or fewer are not palettized. This is weight compression and not the calibrated activation quantization that `quantize=8` selects for other formats, so it takes no `data` argument. With `scikit-learn` already installed, the export of a YOLO26n detection model took 18 to 26 s on an Apple M4 and 23 to 24 s on an x86-64 server, and the export of YOLO26x took 104 s on that server.
- **Accuracy**: see the [Apple M4 tables](#apple-m4) for COCO results in each precision. Validate an exported model on your own data before you deploy it.

## Output Format

The export writes one package, a directory that Xcode, Core ML, and `coremltools` open as a single model:

| File in `yolo26n.mlpackage/`               | Contents                                                                                                                                                                                         |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Manifest.json`                            | The list of the files in the package                                                                                                                                                             |
| `Data/com.apple.CoreML/model.mlmodel`      | The network graph, the input and output descriptions, and the metadata: class names, task, image size, batch size, stride, and export arguments, which Ultralytics reads when it loads the model |
| `Data/com.apple.CoreML/weights/weight.bin` | The weights                                                                                                                                                                                      |

The model takes one input named `image`. For a default export it is a Core ML image input: an RGB image of the export `imgsz` with pixel values from 0 to 255. The division by 255 is part of the model, so do not scale the image yourself, and in an app Apple's [Vision framework](https://developer.apple.com/documentation/vision) can resize a camera frame to this input. An export with `dynamic=True` takes a `float32` array of shape `(batch, 3, height, width)` instead, RGB and scaled to 0-1.

The names of the output arrays are generated by the converter, such as `var_1401`, and change between models and export settings, so read them from the model description instead of hard-coding them. The `nms` argument and the task set the outputs. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | Output with `nms=None`                        | Output with `nms=False`                             | Output with `nms=True`                                    |
| -------- | --------------- | --------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------- |
| Detect   | `640`           | `(1, 84, 8400)`                               | `(1, 300, 6)`                                       | `confidence` `(boxes, 80)` and `coordinates` `(boxes, 4)` |
| Segment  | `640`           | `(1, 116, 8400)` and `(1, 32, 160, 160)`      | `(1, 300, 38)` and `(1, 32, 160, 160)`              | `(1, 300, 38)` and `(1, 32, 160, 160)`                    |
| Semantic | `1024`          | `(1, 1024, 1024)`, class index for each pixel | Same as `nms=None`                                  | Same as `nms=None`                                        |
| Depth    | `768`           | `(1, 1, 768, 768)`, depth map                 | Same as `nms=None`                                  | Same as `nms=None`                                        |
| Classify | `224`           | `classLabel` and `classLabel_probs`           | Same as `nms=None`                                  | Same as `nms=None`                                        |
| Pose     | `640`           | `(1, 56, 8400)`                               | `(1, 300, 57)`: 6 box values, then 17 keypoints x 3 | `(1, 300, 57)`                                            |
| OBB      | `1024`          | `(1, 20, 21504)`                              | `(1, 300, 7)`: 4 box values, score, class, angle    | Same as `nms=None`                                        |

- **Raw output (`nms=None`)**: shape `(1, values, anchors)`, the same raw predictions as a default [ONNX export](onnx.md#output-format): for each anchor the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS yourself.
- **NMS-free output (`nms=False`)**: `max_det` (default 300) rows of `x1, y1, x2, y2, score, class` in input pixels, with no NMS step. OBB rows are `center x, center y, width, height, score, class, angle`. Filter the rows by score. Available for models with an NMS-free head, such as YOLO26 and YOLOv10.
- **Embedded NMS for detection (`nms=True`)**: the package is a Core ML pipeline of the model and Apple's NMS stage. It returns `confidence`, one row of class scores for each kept box, and `coordinates`, the box center x, center y, width, and height of each kept box relative to the image size (0-1). The number of rows changes from image to image. `conf` (default `0.25`), `iou` (default `0.5`), and `agnostic_nms` are stored in the pipeline at export, and the inputs `confidenceThreshold` and `iouThreshold`, which can be left out, replace the two thresholds for one prediction. `max_det` does not apply: an export with `max_det=2` still returned five boxes for a test image. For a model whose number of classes is not a multiple of 80, the export pads `confidence` with extra columns up to the next multiple of 80.
- **Embedded NMS for segmentation and pose (`nms=True`)**: NMS runs inside the model graph, which returns `max_det` rows of `x1, y1, x2, y2, score, class` followed by the mask coefficients or keypoints. Rows past the last detection have a score of 0, so filter the rows by score. `conf`, `iou`, `agnostic_nms`, and `max_det` are fixed at export, and these exports are always FP16 or INT8.
- **Segmentation**: the 3-D output holds the predictions and the 4-D output the mask prototypes.
- **Semantic segmentation**: the output is the class index of every pixel as `int32`, not a score for each class.
- **Classification**: the package is a Core ML classifier, which returns `classLabel`, the name of the top class, and `classLabel_probs`, a dictionary of class names and probabilities. `coremltools` also returns the probabilities as a `(1, classes)` array.

`YOLO("yolo26n.mlpackage")` reads the layout from the model metadata, so you only need these details when you run the model in your own code. The metadata is stored as strings in the user-defined metadata of the model. Its keys include `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, `args`, and, for pose models, `kpt_shape`.

### Run with coremltools

To run the package outside the Ultralytics package on a Mac, load it with `coremltools`, which also selects the [compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits):

```python
# Export first: yolo export model=yolo26n.pt format=coreml quantize=16 nms=False
import ast

import coremltools as ct
import numpy as np
from PIL import Image

model = ct.models.MLModel("yolo26n.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)

image = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))  # your letterboxed image: RGB, 0-255
outputs = model.predict({"image": image})
detections = next(iter(outputs.values()))[0]  # the only output array, first image: shape (300, 6)
assert detections.shape == (300, 6), "export a detection model at imgsz=640 with nms=False"
detections = detections[detections[:, 4] > 0.25]  # keep rows above the confidence threshold

names = ast.literal_eval(model.user_defined_metadata["names"])  # {0: 'person', 1: 'bicycle', ...}
```

## Deploy

On an iPhone or iPad, add the `.mlpackage` to your Xcode project, or use the [Ultralytics iOS SDK](#ultralytics-ios-sdk-and-flutter-plugin), which loads it and decodes the outputs for you. On a Mac, install the package with `pip install ultralytics` and copy the `.mlpackage` to it: `YOLO("yolo26n.mlpackage")` runs the model with `coremltools`, which Ultralytics installs on first use.

- **Compute units**: Ultralytics loads a model with the Core ML compute units `CPU_AND_NE`, the CPU and the Neural Engine, and RT-DETR models with `ALL`. If the model does not load with them, it loads with `CPU_ONLY`. There is no argument to choose other compute units; load the package with [`coremltools`](#run-with-coremltools) to do that. On an Apple M4 with macOS 27.2, `ALL` and `CPU_AND_GPU` also ran the FP32, FP16, and INT8 models of all seven YOLO26n tasks; other macOS versions were not tested.
- **`device`**: the `device` argument does not select the Core ML hardware. With `device="mps"` Ultralytics preprocesses and postprocesses on the Apple GPU while Core ML still uses the same compute units. On an M4 this did not make inference faster: an FP16 YOLO26n model took about 8 ms per image with `device="mps"` against about 6 ms without it, so leave `device` unset.
- **Precision and the Neural Engine**: on an Apple M4, the inference stage of `YOLO()` took 18 to 20 ms per image in `predict` for an FP32 YOLO26n model (21.0 ms in the COCO validation in [Apple M4](#apple-m4)) and about 6 ms for the FP16 and INT8 models, for default (`nms=None`) exports on one test image with macOS 27.2. With `coremltools` alone, the same FP16 and INT8 models took about 2 ms with `CPU_AND_NE` against 9 to 10 ms with `CPU_ONLY`.
- **Fixed input shape**: a default export always runs at its export `imgsz`, which Ultralytics reads from the model metadata. An `imgsz` passed to `predict` or `val` is ignored. Ultralytics passes each image of a batch to Core ML as a separate single-image request, so any number of images works, and `val` sets `batch=1`.
- **Dynamic export (`dynamic=True`)**: the model declares batch sizes from 1 to the export `batch` and image sizes from 32 pixels to twice the export `imgsz`, and Ultralytics passes image sizes in multiples of the model stride. Ultralytics splits larger batches into chunks of the export `batch`. With the NMS-free head the smallest size is larger, 128 pixels for a YOLO26n model exported at `imgsz=640` with `nms=False`. Tested with a detection model at 320, 640, and 1280 pixels and with a segmentation model at 640.

### Ultralytics iOS SDK and Flutter Plugin

The [Ultralytics iOS SDK](https://github.com/ultralytics/yolo-ios-app) is a Swift package that loads Core ML exports of all seven YOLO26 tasks and returns decoded results. Export with the settings of the official app models:

```bash
# INT8 weights and the NMS-free head; use imgsz=224 for classification models
yolo export model=yolo26n.pt format=coreml quantize=8 nms=False imgsz=640
```

Add `https://github.com/ultralytics/yolo-ios-app.git` to your app with Swift Package Manager, then load the exported package:

```swift
import UltralyticsYOLO

// Load your exported Core ML model, then run inference on a UIImage
let yolo = YOLO("/path/to/yolo26n.mlpackage", task: .detect) { result in
    if case .success(let model) = result {
        let results = model(uiImage)  // boxes, labels, confidences, timing
    }
}
```

The SDK runs Core ML models through Vision's `VNCoreMLRequest`, which scales each image to the model input. For camera apps, the SDK's `YOLOView` runs real-time inference with native overlays, and the [Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) runs the same Core ML exports on iOS from a codebase shared with Android. The SDK asks Core ML for `.cpuAndNeuralEngine` on iOS 16 and later instead of `.all`, because in a camera app the GPU is already busy with the preview and the overlays.

### Deployment Guides

- **[Integrating a Core ML Model into Your App](https://developer.apple.com/documentation/coreml/integrating-a-core-ml-model-into-your-app)**: Apple's guide to adding a model to an Xcode project and calling it.
- **[Recognizing Objects in Live Capture](https://developer.apple.com/documentation/vision/recognizing-objects-in-live-capture)**: Apple's sample for running a Core ML detector on camera frames with Vision and `VNCoreMLRequest`.
- **[MLComputeUnits](https://developer.apple.com/documentation/coreml/mlcomputeunits)**: the `computeUnits` setting of `MLModelConfiguration` that selects the CPU, the GPU, and the Neural Engine.
- **[Optimize your Core ML usage](https://developer.apple.com/videos/play/wwdc2022/10027/)**: Apple's session on Xcode performance reports, which show the compute unit and latency of each layer on a device.
- **[Core ML Tools guide](https://apple.github.io/coremltools/docs-guides/)**: the conversion and optimization reference for the `coremltools` package that this export uses.
- **[Ultralytics Platform](../platform/train/models.md#open-on-iphone-or-android)**: open a Core ML export in the Ultralytics YOLO app on an iPhone by scanning a QR code.
- **[Core AI](coreai.md)**, **[LiteRT](litert.md)**, and **[Qualcomm QNN](qnn.md)**: Apple's format for iOS 27 and the formats for Android devices and Snapdragon NPUs.

## Limitations and Troubleshooting

- **First export after a plain install**: when the first export has to replace `numpy` (versions above 2.3.5 [break the conversion](https://github.com/apple/coremltools/issues/2633)), Ultralytics logs `Restart runtime or rerun command for updates to take effect` and that export fails with `TypeError: only 0-dimensional arrays can be converted to Python scalars`. Run the export again in a new process, which means a kernel restart in a notebook, or install `"ultralytics[export-coreml]"` before the first export as in the [Quickstart](#quickstart).
- **Export host**: the export runs on macOS and on x86-64 Linux. It is not supported on Windows or on ARM64 Linux, such as a Raspberry Pi: `coremltools` 9.0 publishes prebuilt packages for macOS and x86-64 Linux only.
- **Python version**: `coremltools` 9.0 has prebuilt packages for Python 3.13 and older. On Python 3.14, `pip install "ultralytics[export-coreml]"` installs none of the Core ML packages, so the export builds `coremltools` from source. The first export then fails as after a plain install, and the next one with `RuntimeError: BlobWriter not loaded`. Create the environment with Python 3.13 or older.
- **Inference needs macOS**: on Linux, predicting or validating with `YOLO("yolo26n.mlpackage")` fails with `Model prediction is only supported on macOS version 10.13 or later.` Validate the PyTorch checkpoint there, or copy the package to a Mac.
- **Automatic installation turned off**: with `YOLO_AUTOINSTALL=false`, the export fails with `No module named 'coremltools'` in an environment without the package.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='coreml' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends`. Export the checkpoint you started the fine-tuning from with `quantize=8` instead, or export the fine-tuned checkpoint to [ONNX](onnx.md) or [TensorRT](tensorrt.md).
- **`quantize="w8a32"`**: fails with `quantize='w8a32' is not supported; format='coreml' supports quantize=16 (FP16), 8 (INT8), 'w8a16' (INT8 weights + INT16 activations), 32 or None (FP32).` For Core ML, `'w8a16'` is the same export as `8`, with FP16 computation.
- **Embedded NMS on other tasks**: `nms=True` works for detect, segment, and pose models. For OBB and classification models the export logs `CoreML 'nms=True' is only supported for detect, segment and pose models. Forcing 'nms=None'.`, for semantic segmentation and depth estimation models `'nms=True' is not valid for semantic models. Forcing 'nms=None'.` (or `depth models`), and for RT-DETR models `'nms=True' is not available for end2end models or non-YOLO heads. Forcing 'nms=None'.` All of them write the output of a default export.
- **Embedded NMS precision**: an `nms=True` export without `quantize` is FP16. `quantize=32` gives an FP32 pipeline for a detection model, and for segmentation and pose models the export logs `CoreML 'nms=True' requires FP16 for segment models. Forcing 'quantize=16'.` (or `pose models`)
- **Embedded NMS and validation**: an `nms=True` export removes boxes below its `conf` before `val` sees them, so it scores lower than the same model with external NMS: on COCO val2017, an FP16 YOLO26n export with `nms=True` reached 33.8 mAP<sup>val</sup> 50-95 with the default `conf=0.25` and 39.9 when exported with `conf=0.001`, still 1.0 below the 40.9 of a default FP16 export. Export with `conf=0.001` to validate such a model, and choose the deployment threshold separately.
- **NMS-free head on other models**: `nms=False` needs a model with an NMS-free head, such as YOLO26 or YOLOv10. For other models, for example YOLO11, the export warns `This model has no one-to-one head; using one-to-many outputs.` and writes the raw `(1, 84, 8400)` output.
- **Batch size**: a `batch` above 1 without `dynamic=True` fails with `batch sizes > 1 are not supported without 'dynamic=True' for CoreML export. Please retry at 'dynamic=True'.`
- **Dynamic shapes**: `dynamic=True` without `batch` logs `'dynamic=True' export requires a maximum batch size, e.g. 'batch=16'.` and exports a model with a maximum batch size of 1. With `nms=True` it fails with `'nms=True' cannot be used together with 'dynamic=True' for CoreML export. Please disable one of them.`, and for classification and RT-DETR models with `'dynamic=True' is not supported for CoreML classification or RT-DETR models.` An image size outside the range fails at prediction with an error such as `Size (1312) of dimension (2) is not in allowed range (32..1280)`.
- **Semantic segmentation on the Neural Engine**: on an Apple M4, a YOLO26n-sem model took longer with `CPU_AND_NE` than with `CPU_ONLY`, with `coremltools` alone on one test image: 59 to 64 ms against 22 ms for the FP16 export at `imgsz=1024`, and about 15 ms against 9 ms for the INT8 export at `imgsz=640`. The FP16 models of the other six tasks, and their INT8 models at `imgsz=640` (224 for classification), ran faster with `CPU_AND_NE`, and on the [iPhone 17 Pro](#iphone-17-pro) the INT8 semantic model was faster with the Neural Engine too. `YOLO()` always uses `CPU_AND_NE` for these models, so measure a semantic model with both settings in your own code.
- **`.cpuOnly` on iOS 27.0.1**: on an iPhone 17 Pro, `.cpuOnly` inference of the YOLO26l and YOLO26x depth, pose, and OBB models crashed inside Core ML's CPU kernels (`EXC_BAD_ACCESS` in `libBNNS`). They run with `.cpuAndNeuralEngine`, which the iOS SDK requests.
- **Package name**: every export of one checkpoint writes the same `yolo26n.mlpackage`, whatever its `quantize`, `imgsz`, `nms`, or `dynamic`, so the later export replaces the earlier one. Rename the package to keep both, and keep the `.mlpackage` suffix, for example `yolo26n_fp16.mlpackage`, because Ultralytics recognizes the format by it. A package without the suffix fails with `is not a supported model format`.
- **PyTorch version warning**: with a recent PyTorch, `coremltools` logs `Torch version 2.15.0 has not been tested with coremltools. You may run into unexpected errors.` when it loads, and with a recent `scikit-learn` it logs `scikit-learn version 1.9.1 is not supported.` The exports and Mac results on this page were made with these versions.
- **Legacy `.mlmodel` files**: `format="mlmodel"` exports the earlier Core ML neural network format as a single `yolo26n.mlmodel` file for apps that cannot load an ML Program. There, `quantize=8` and `'w8a16'` apply linear 8-bit weight quantization. Use the default `.mlpackage` for new projects.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Measured Performance

The tables below were measured on an iPhone 17 Pro and on an Apple M4 Mac. The [YOLO26 model page](../models/yolo26.md) lists the accuracy of the PyTorch checkpoints for every task.

### iPhone 17 Pro

**Hardware:** [iPhone 17 Pro](https://support.apple.com/en-us/125090) with iOS 26.6. Its A19 Pro has a 6-core CPU (2 Performance and 4 Efficiency cores), a 6-core GPU, and a 16-core Neural Engine.

The first table gives the latency of one image for the official INT8 model of each task, the `v8.3.0` assets of the [Ultralytics iOS SDK](https://github.com/ultralytics/yolo-ios-app), which are exported with `nms=False`. Each cell shows the total time in milliseconds with the preprocess / inference / postprocess split beneath it; the faster compute unit is in bold:

| Model         | Task     | size<br><sup>(pixels)</sup> | CPU<br><sup>Core ML `.cpuOnly`<br>(ms)</sup> | CPU + Neural Engine<br><sup>Core ML `.cpuAndNeuralEngine`<br>(ms)</sup> |
| ------------- | -------- | --------------------------- | -------------------------------------------- | ----------------------------------------------------------------------- |
| YOLO26n       | Detect   | 640                         | 9.7<br><sup>0.0 / 9.7 / 0.0</sup>            | **4.7**<br><sup>0.0 / 4.7 / 0.0</sup>                                   |
| YOLO26n-seg   | Segment  | 640                         | 12.8<br><sup>0.0 / 12.2 / 0.6</sup>          | **5.8**<br><sup>0.0 / 5.0 / 0.7</sup>                                   |
| YOLO26n-sem   | Semantic | 640                         | 9.3<br><sup>0.0 / 8.9 / 0.5</sup>            | **5.3**<br><sup>0.0 / 4.8 / 0.5</sup>                                   |
| YOLO26n-depth | Depth    | 640                         | 25.8<br><sup>0.0 / 24.8 / 1.0</sup>          | **8.0**<br><sup>0.0 / 6.9 / 1.1</sup>                                   |
| YOLO26n-cls   | Classify | 224                         | 2.1<br><sup>0.0 / 2.0 / 0.0</sup>            | **1.8**<br><sup>0.0 / 1.8 / 0.0</sup>                                   |
| YOLO26n-pose  | Pose     | 640                         | 11.9<br><sup>0.0 / 11.9 / 0.0</sup>          | **5.4**<br><sup>0.0 / 5.4 / 0.0</sup>                                   |
| YOLO26n-obb   | OBB      | 640                         | 10.6<br><sup>0.0 / 10.6 / 0.0</sup>          | **4.9**<br><sup>0.0 / 4.9 / 0.0</sup>                                   |

The same measurement for the five detection models, with the time the SDK takes to load each model:

| Model   | CPU<br><sup>`.cpuOnly`<br>(ms)</sup> | CPU + Neural Engine<br><sup>`.cpuAndNeuralEngine`<br>(ms)</sup> | Model load<br><sup>`.cpuAndNeuralEngine`<br>(s)</sup> |
| ------- | ------------------------------------ | --------------------------------------------------------------- | ----------------------------------------------------- |
| YOLO26n | 9.2                                  | **4.5**                                                         | 0.9                                                   |
| YOLO26s | 19.5                                 | **10.5**                                                        | 1.3                                                   |
| YOLO26m | 46.7                                 | **11.8**                                                        | 1.8                                                   |
| YOLO26l | 60.1                                 | **12.0**                                                        | 2.4                                                   |
| YOLO26x | 125.9                                | **20.5**                                                        | 3.3                                                   |

YOLO26n detection in the three precisions and with two `nms` settings, exported with `ultralytics` 8.4.215 at `imgsz=640`, as the total time in milliseconds:

| Precision | `nms=None`, CPU | `nms=None`, CPU + Neural Engine | `nms=False`, CPU | `nms=False`, CPU + Neural Engine |
| --------- | --------------- | ------------------------------- | ---------------- | -------------------------------- |
| FP32      | 15.5            | 14.7                            | 14.5             | 14.4                             |
| FP16      | 9.2             | 4.9                             | 9.2              | 5.0                              |
| INT8      | 9.2             | 4.2                             | 9.1              | 5.0                              |

- **Method**: mean of 15 predictions after 3 warmup predictions on `bus.jpg`, through the Swift SDK 8.9.23 in a release build. `.cpuOnly` and `.cpuAndNeuralEngine` are the Core ML compute units the SDK requests, and Core ML decides where each layer runs. Every table was measured in three separate runs and shows the median. The three runs agreed within 15%, except YOLO26n with `.cpuAndNeuralEngine`, which took 2.6 to 4.8 ms in the size table and 3.8 to 4.8 ms as the INT8 `nms=None` export. The task and size tables use the official `v8.3.0` assets, and the precision table uses exports made with `ultralytics` 8.4.215, so their YOLO26n cells differ, by up to 0.5 ms with the Neural Engine and 0.6 ms on the CPU.
- **Preprocessing**: Vision scales the image inside the inference request, so preprocessing is reported as 0 and its cost is part of inference.
- **FP32**: by these medians the FP32 export was at most 5% faster with `.cpuAndNeuralEngine` than with `.cpuOnly`, while FP16 and INT8 were about twice as fast. Export in FP16 or INT8 for the Neural Engine.
- **`nms`**: for YOLO26n detection, the NMS-free head (`nms=False`) and the raw output with NMS in the SDK (`nms=None`) both took 4 to 5 ms with the Neural Engine in FP16 and INT8.
- **Camera apps**: these are single-image latencies. A real-time camera app adds capture and display time; see the [iOS SDK performance notes](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md).
- Compare Android CPU/GPU results in the [LiteRT integration](litert.md#measured-performance), [MediaTek Dimensity](mediatek.md#measured-performance) results, and Snapdragon NPU results in the [Qualcomm QNN integration](qnn.md#measured-performance).

### Apple M4

The tables below compare PyTorch with FP32, FP16, and INT8 Core ML models of every YOLO26 detection model on an Apple M4 (4 performance and 6 efficiency cores) with macOS 27.2, validated through `YOLO()` on COCO val2017. PyTorch ran on the CPU with 8 threads, `batch=1`, and `rect=False`, so that it sees the same 640x640 input as the Core ML models, which ran with the default compute units of Ultralytics.

=== "Latency"

    Milliseconds per image, lower is better. The fastest format for each model is in bold.

    | Model   | `imgsz` | PyTorch | Core ML FP32 | Core ML FP16 | Core ML INT8 |
    | ------- | ------- | ------- | ------------ | ------------ | ------------ |
    | YOLO26n | 640     | 25.6    | 21.0         | **6.0**      | **6.0**      |
    | YOLO26s | 640     | 60.0    | 50.0         | 7.9          | **7.6**      |
    | YOLO26m | 640     | 137.2   | 127.6        | 14.3         | **11.3**     |
    | YOLO26l | 640     | 191.2   | 187.7        | 17.2         | **13.5**     |
    | YOLO26x | 640     | 341.1   | 334.2        | 28.6         | **24.3**     |

=== "Accuracy"

    mAP<sup>val</sup> 50-95 on COCO val2017 from the COCO evaluator (`save_json=True`), higher is better.

    | Model   | PyTorch | Core ML FP32 | Core ML FP16 | Core ML INT8 |
    | ------- | ------- | ------------ | ------------ | ------------ |
    | YOLO26n | 40.9    | 40.9         | 40.9         | 40.8         |
    | YOLO26s | 48.5    | 48.5         | 48.5         | 48.3         |
    | YOLO26m | 53.2    | 53.2         | 53.2         | 53.1         |
    | YOLO26l | 54.9    | 54.9         | 54.8         | 54.8         |
    | YOLO26x | 57.5    | 57.5         | 57.4         | 57.3         |

=== "Size"

    Size in MB (10^6 bytes), as the sum of all files in the `.mlpackage`. The smallest export for each model is in bold. The export log prints the size in MiB, 9.5 MB, 4.8 MB, and 2.6 MB for YOLO26n.

    | Model   | PyTorch | Core ML FP32 | Core ML FP16 | Core ML INT8 |
    | ------- | ------- | ------------ | ------------ | ------------ |
    | YOLO26n | 5.5     | 9.9          | 5.1          | **2.7**      |
    | YOLO26s | 20.4    | 38.3         | 19.2         | **9.8**      |
    | YOLO26m | 44.3    | 81.9         | 41.1         | **20.8**     |
    | YOLO26l | 53.2    | 99.6         | 50.0         | **25.3**     |
    | YOLO26x | 118.7   | 223.3        | 111.8        | **56.2**     |

With the NMS-free head (`nms=False`), YOLO26n reached 40.3 mAP<sup>val</sup> 50-95 in PyTorch and in FP32, 40.2 in FP16, and 40.1 in INT8.

The latency of YOLO26n with each Core ML compute unit, measured with `coremltools` alone as the median of 20 predictions on one image, in milliseconds:

| Precision | `CPU_ONLY` | `CPU_AND_NE` | `ALL` | `CPU_AND_GPU` |
| --------- | ---------- | ------------ | ----- | ------------- |
| FP32      | 13.7       | 13.6         | 8.3   | 8.0           |
| FP16      | 9.6        | 1.8          | 2.0   | 7.7           |
| INT8      | 9.4        | 1.8          | 1.8   | 8.1           |

- **FP16 and INT8** ran 3.5 to 14 times faster than FP32 through `YOLO()`, at an accuracy within 0.1 mAP of FP32 for FP16 and within 0.2 for INT8. INT8 was as fast as FP16 for YOLO26n and up to 1.3 times faster for the larger models.
- **FP32** gave the same outputs with `CPU_AND_NE` as with `CPU_ONLY`. Only the GPU compute units made it faster.
- **INT8** made the package 3.7 to 4.0 times smaller than FP32.

Measured with `ultralytics` 8.4.215, `coremltools` 9.0, and PyTorch 2.15.0 at `batch=1` on November 17, 2026. Latency is the inference stage that `val` reports and does not include preprocessing or postprocessing. A second run of the three YOLO26n Core ML cases on the same day differed by up to 2.3 ms, 21.0 against 18.8 ms for FP32.

## About Core ML

Core ML is built into iOS, iPadOS, macOS, watchOS, tvOS, and visionOS. It loads models in the ML Program format, the `.mlpackage` this export writes, and runs each layer on the CPU, the GPU, or the Apple Neural Engine, the [NPU](https://www.ultralytics.com/glossary/neural-processing-unit-npu) of Apple silicon, so inference works offline and the data stays on the device. For the full reference, see the [Core ML documentation](https://developer.apple.com/documentation/coreml).

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/5c3edb9e4431e48c58368d218d5a3615.avif" alt="Apple Core ML deployment pipeline">
</p>

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/KcTSdIUYcVE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Export Ultralytics YOLO26 to CoreML with INT8 Quantization | Apple Deployment | iOS/MacOS 🍎
</p>

## FAQ

### Should I export a Core ML model in FP16 or INT8?

Both run on the Neural Engine and compute in FP16, so the choice is about size. FP16 (`quantize=16`) halves the package, 5.1 MB against 9.9 MB for YOLO26n, and INT8 (`quantize=8`) stores 8-bit weights, 2.7 MB for YOLO26n, and is the precision of the official Ultralytics app models. On COCO val2017, a default YOLO26n export reached 40.9 mAP<sup>val</sup> 50-95 in FP16 and 40.8 in INT8 on an Apple M4, and both took about the same time per image there. On an iPhone 17 Pro both took 4 to 5 ms for YOLO26n with the Neural Engine, within the run-to-run variation. INT8 needs iOS 16 or macOS 13, while FP16 also runs on iOS 15 and macOS 12. Neither needs calibration data. Avoid the default FP32 export for deployment: it is the largest and was at most 5% faster with the Neural Engine compute units than on the CPU alone in our tests, see [Precision](#precision). Validate the exported model on your own data and device before you ship it.

### Which `nms` setting should I use for an iOS app?

It depends on who decodes the output. With the [Ultralytics iOS SDK](#ultralytics-ios-sdk-and-flutter-plugin) or the Flutter plugin, export YOLO26 models with `nms=False`, as the official models are: the SDK then reads the final detections of the NMS-free head and needs no NMS step. On COCO val2017, YOLO26n in INT8 reached 40.1 mAP<sup>val</sup> 50-95 with the NMS-free head against 40.8 with the default output and external NMS. On an iPhone 17 Pro both took 4 to 5 ms per image. With Apple's Vision framework and your own code, `nms=True` turns a detection model into a Core ML pipeline with Apple's NMS stage, whose thresholds you can change for each prediction. Vision returns its detections as `VNRecognizedObjectObservation` objects, and the score of a box is the `confidence` of the observation. For an `nms=True` detection model, SDK 8.9.23 and earlier report about 1.0, the confidence of the top label of each box, in place of the box scores; later versions report the `confidence` of the observation. The default `nms=None` returns raw predictions and leaves NMS to your code. See [Output Format](#output-format) for the outputs of each setting.

### Can I export or test a Core ML model without a Mac?

You can export without one: the export runs on x86-64 Linux, and [Ultralytics Platform](../platform/train/models.md#export-model) exports in the cloud. Running the model needs Apple hardware, because Core ML is part of Apple's operating systems. On Linux and Windows, validate the PyTorch checkpoint or an [ONNX](onnx.md) export of it, then validate the Core ML model itself on a Mac, where `YOLO("yolo26n.mlpackage").val()` runs it, or on the device.
