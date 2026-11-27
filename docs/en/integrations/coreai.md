---
comments: true
description: Export Ultralytics YOLO26 models to Apple Core AI with format=coreai. Covers the yolo26n.aimodel asset, FP16 with quantize=16, nms=None and nms=False outputs, Swift and Python inference, Core AI vs Core ML, and Mac benchmarks.
keywords: YOLO26, Core AI, Apple Core AI, CoreAI, aimodel, coreai-torch, coreai-core, Core AI export, torch.export, FP16, Apple Neural Engine, iOS 27, macOS 27, Swift, Core ML comparison, on-device inference
---

# Apple Core AI Export

[Core AI](https://developer.apple.com/core-ai/) is Apple's framework for running [neural networks](https://www.ultralytics.com/glossary/neural-network-nn) on Apple silicon, introduced with iOS 27 and macOS 27, with its own `.aimodel` format and Swift API. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="coreai"` captures it with `torch.export`, converts it with Apple's [`coreai-torch`](https://github.com/apple/coreai-torch) package, and saves a `yolo26n.aimodel` asset that Core AI specializes for the CPU, GPU, and Neural Engine of the device that loads it. Core AI is a separate deployment path, not a new name for [Core ML](coreml.md): the two use different model formats, conversion tools, and runtime APIs.

## At a Glance

| Property          | Details                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `format` argument | `coreai`                                                                                                                                                                                   |
| Output            | `yolo26n.aimodel`, a directory with the model program, a hash, and a `metadata.json` that holds the [Ultralytics metadata](#output-format)                                                 |
| Runs on           | Apple devices with the 27 generation of iOS, iPadOS, macOS, tvOS, visionOS, and watchOS, on the CPU, GPU, and Neural Engine. `YOLO()` runs it on Apple silicon Macs with macOS 26 or newer |
| Runtime           | The Core AI framework of the operating system, through its Swift API; in Python, the `coreai.runtime` module of the `coreai-core` package on macOS                                         |
| Precision         | FP32 (default) and FP16. YOLO26n is 9.9 MB in FP32 and 5.0 MB in FP16. There is no INT8 export                                                                                             |
| Export host       | macOS 26 or newer on Apple silicon, or x86-64 Linux with glibc 2.34 or newer, with Python 3.11 to 3.15 and PyTorch 2.8 or newer. Not Windows or ARM64 Linux. The export needs no GPU       |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to Core AI, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package with the Core AI export dependencies in a virtual environment, on macOS or on x86-64 Linux:

```bash
python3 -m venv coreai-env && source coreai-env/bin/activate
pip install "ultralytics[export-coreai]"
```

The `export-coreai` extra installs Apple's `coreai-torch` converter and the `coreai-core` package it depends on, which holds the Core AI compiler and, on macOS, the Python runtime, with their dependencies, such as `ml-dtypes`, `rich`, `scipy`, and `yuvio`. It replaces none of the packages that `pip install ultralytics` installed. In an environment with only `pip install ultralytics`, the first export installs the same packages, logs `Restart runtime or rerun command for updates to take effect`, and still completes in the same process. With `YOLO_AUTOINSTALL=false`, an environment without `coreai-torch` fails the export with `No module named 'coreai_torch'`, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). The export has these version requirements:

| Requirement      | Version                                                                                                                                                                                                                                                                                            |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python           | 3.11 to 3.14 for the `export-coreai` extra. On Python 3.15 the extra installs nothing, and the first export installs `coreai-torch` with pip, as `coreai-core` has packages for Python 3.15. Python 3.10 and older cannot install `coreai-torch`. Run with 3.11 to 3.15 on Linux and 3.14 on macOS |
| PyTorch          | 2.8 or newer. Run with 2.8.0 and 2.15.0, and with 2.13.0 in one comparison                                                                                                                                                                                                                         |
| `coreai-torch`   | 0.4.3 or newer. Run with 0.4.4, and with 0.4.3 in one comparison                                                                                                                                                                                                                                   |
| `coreai-core`    | Installed by `coreai-torch`, which pins the version: 0.4.4 requires 1.0.0. Run with 1.0.0, and with 1.0.0b3 in one comparison                                                                                                                                                                      |
| Operating system | Export: macOS 26 or newer on Apple silicon, or x86-64 Linux with glibc 2.34 or newer, the platforms of the `coreai-core` packages. Predict and validate: macOS on Apple silicon. Run with macOS 27.2.1 and Ubuntu 24.04 (glibc 2.39)                                                               |

For other installation methods, see the [installation guide](../quickstart.md). Then export an Ultralytics YOLO26 model, and on a Mac load the exported asset to run inference or validate its accuracy:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to Core AI format
        model.export(format="coreai")  # creates 'yolo26n.aimodel'

        # Or export in FP16 instead, which halves the size
        model.export(format="coreai", quantize=16)  # creates 'yolo26n.aimodel'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to Core AI format
        yolo export model=yolo26n.pt format=coreai # creates 'yolo26n.aimodel'

        # Or export in FP16 instead, which halves the size
        yolo export model=yolo26n.pt format=coreai quantize=16 # creates 'yolo26n.aimodel'
        ```

A YOLO26n export took 1.1 to 4.6 s on the x86-64 Linux server of this page and 0.8 to 8.0 s on a MacBook Air with an Apple M4 in all runs for this page, plus the installation on first use: the first export that installed the packages took 9.7 s on the Linux server (10.6 s on Python 3.15) and 25.7 s on the Mac. Both exports above write the same `yolo26n.aimodel`, so the second one replaces the first.

Predict and Validate with an exported Ultralytics YOLO26 Core AI model run only on macOS on Apple silicon, even when the export ran on Linux:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported Core AI model
        model = YOLO("yolo26n.aimodel")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported Core AI model
        yolo predict model=yolo26n.aimodel source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported Core AI model
        model = YOLO("yolo26n.aimodel")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported Core AI model
        yolo val model=yolo26n.aimodel data=coco8.yaml
        ```

`YOLO()` runs an `.aimodel` asset with the Python runtime of `coreai-core`. Assets exported with `coreai-core` 1.0.0 took 1.7 to 1.8 s per image for YOLO26n detection on a MacBook Air with an Apple M4, in that runtime, while the same asset ran in 6.6 ms through the Core AI framework in [Swift](#swift), so use `predict` and `val` to check the predictions and the accuracy of an export, and measure its speed in your app, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

## Supported Tasks

Core AI export supports all seven Ultralytics tasks: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with [YOLO26](../models/yolo26.md), the only family that ships those heads. Every task was exported on Linux and on a Mac and run with `predict` and `val` on the Mac, for YOLO26n in FP32 and FP16 with `nms=None` and `nms=False`. YOLO11n detection, segmentation, and pose, YOLOv8n, and YOLOv10n with `nms=False` were exported and run on the Mac too, and YOLOv5nu and YOLOE-26n-seg exported on Linux. RT-DETR models do not export, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). The Ultralytics CI exports and validates the YOLO26n model of each task on macOS 26 at 160 pixels in FP32.

## Export Arguments

| Argument   | Type             | Default    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------- | ---------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'coreai'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                                                                                              |
| `imgsz`    | `int` or `tuple` | `640`      | Image size of the model input, fixed in the `.aimodel` asset: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect and segment, `640` for semantic models pretrained on ADE20K and `1024` for those pretrained on Cityscapes, `768` for depth, `224` for classify, `640` for pose, and `1024` for OBB.                                                                      |
| `quantize` | `int` or `str`   | `None`     | Precision of the asset: `16` (FP16, also spelled `'fp16'` or `'w16a16'`) or `32`/unset (FP32, also `'fp32'` or `'w32a32'`). INT8 values raise an error, see [Precision](#precision). Replaces the deprecated `half` flag.                                                                                                                                                                                                                                                                                       |
| `batch`    | `int`            | `1`        | Batch size of the exported model, fixed in the input shape.                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `nms`      | `bool`, optional | `None`     | `None` writes the raw output, which needs NMS for YOLO detection, segmentation, pose, and OBB models; `False` keeps the NMS-free head of models that have one, such as YOLO26 and YOLOv10; a model without one, such as YOLO11, logs `This model has no one-to-one head; using one-to-many outputs.` and writes the raw output. `True` is not supported: the export logs `format=coreai does not support embedded NMS; using nms=None.` and writes the raw output. See [Choosing the Head](#choosing-the-head). |
| `device`   | `str`            | `None`     | Has no effect on this format: the export always traces the model on the CPU. `device=0` on a CUDA host wrote a 9.9 MB asset as `device="cpu"` does, and `device="mps"` on a Mac one that gave the same predictions.                                                                                                                                                                                                                                                                                             |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md). `dynamic=True`, `optimize=True`, any `opset`, `workspace`, `data`, or `fraction`, and `simplify=False` are not supported by this format and fail with an error such as `argument 'dynamic' is not supported for format='coreai'`; their default values pass.

### Precision

`quantize` sets the data type of the whole graph, including the input and outputs. No calibration data is needed:

| `quantize`                           | Result                                                                                                                                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| unset, `32`, `'fp32'`, or `'w32a32'` | FP32 asset, 9.9 MB for YOLO26n                                                                                                                                                             |
| `16`, `'fp16'`, or `'w16a16'`        | FP16 asset, 5.0 MB for YOLO26n. The deprecated `half=True` gives an FP16 asset of the same size and logs `'half' is deprecated and will be removed in the future. Use 'quantize' instead.` |
| `8`, `'int8'`, `'w8a8'`, `int8=True` | Fails with `quantize=8 (INT8) is not supported; format='coreai' supports quantize=16 (FP16), 32 or None (FP32).`                                                                           |
| `'w8a16'`, `'w8a32'`                 | Fails with the matching message, for example `quantize='w8a16' is not supported`                                                                                                           |
| `True` or a value not in that list   | Fails with `'quantize=True' is invalid. Valid 'quantize' values are 8, 16, 32, 'int8', 'fp16', 'fp32', 'w8a8', 'w16a16', 'w32a32', 'w8a16', or 'w8a32'.`                                   |

- **FP16**: the export converts the model and the example input to FP16 before `torch.export`, so the asset takes a `float16` input and returns `float16` outputs, except the detections of an `nms=False` export, which stay `float32`. `YOLO()` reads the input type from the asset and converts the image to it.
- **INT8 and palettization**: Ultralytics has no INT8 Core AI export. Apple's [Core AI Optimization](https://apple.github.io/coreai-optimization/) package supports [quantization](https://www.ultralytics.com/glossary/model-quantization), palettization, and pruning for Core AI models; it is not used by this export.

The sizes of the `.aimodel` asset for the official YOLO26n checkpoints at their default export size, in MB (10^6 bytes). The export log prints the size in MiB, as `(9.4 MB)` and `(4.8 MB)` for the detection model:

| Task     | Model              | FP32 (MB) | FP16 (MB) |
| -------- | ------------------ | --------- | --------- |
| Detect   | `yolo26n.pt`       | 9.9       | 5.0       |
| Segment  | `yolo26n-seg.pt`   | 11.1      | 5.6       |
| Semantic | `yolo26n-sem.pt`   | 6.3       | 3.2       |
| Depth    | `yolo26n-depth.pt` | 20.8      | 10.4      |
| Classify | `yolo26n-cls.pt`   | 11.3      | 5.7       |
| Pose     | `yolo26n-pose.pt`  | 11.9      | 6.0       |
| OBB      | `yolo26n-obb.pt`   | 10.2      | 5.2       |

## Output Format

The export writes `yolo26n.aimodel` next to the `.pt` file. The asset is a directory, and Core AI loads the whole directory as one model:

| File in `yolo26n.aimodel/` | Contents                                                                                                                                                              |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.mlirb`               | The model program with its weights, 9.9 MB for YOLO26n in FP32                                                                                                        |
| `main.hash`                | A 32-byte hash of the program                                                                                                                                         |
| `metadata.json`            | The asset metadata and the Ultralytics metadata, 1.0 to 19.3 kB for the official YOLO26n checkpoints (the 1,000 class names make the classification file the largest) |

`metadata.json` holds the asset fields `author` (`Ultralytics`), `license`, `description`, `producer` (the `coreai-core` version that wrote it), `assetVersion`, and `creationDate`, and in `creatorDefinedMetadata` the Ultralytics metadata as strings, such as `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, `args`, and, for pose models, `kpt_shape`. `YOLO()` reads it from there when it loads the asset.

The model has one inference function, `main`, with one input, `images`, of shape `(batch, 3, height, width)`: RGB, channels first (NCHW), scaled to 0-1, `float32` (`float16` for an FP16 export). Ultralytics resizes each image to that size with gray padding (letterbox; classification models are resized and center-cropped instead), and the box coordinates in the output refer to the padded image. The outputs are named `output0` and, for segmentation models, `output1`. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Input `images`       | Output with `nms=None`                                    | Output with `nms=False`                             |
| -------- | -------------------- | --------------------------------------------------------- | --------------------------------------------------- |
| Detect   | `(1, 3, 640, 640)`   | `(1, 84, 8400)`                                           | `(1, 300, 6)`                                       |
| Segment  | `(1, 3, 640, 640)`   | `(1, 116, 8400)` and `(1, 32, 160, 160)`                  | `(1, 300, 38)` and `(1, 32, 160, 160)`              |
| Semantic | `(1, 3, 1024, 1024)` | `(1, 19, 1024, 1024)`, score of each class for each pixel | Same as `nms=None`                                  |
| Depth    | `(1, 3, 768, 768)`   | `(1, 1, 768, 768)`, depth map                             | Same as `nms=None`                                  |
| Classify | `(1, 3, 224, 224)`   | `(1, 1000)`, class probabilities                          | Same as `nms=None`                                  |
| Pose     | `(1, 3, 640, 640)`   | `(1, 56, 8400)`                                           | `(1, 300, 57)`: 6 box values, then 17 keypoints x 3 |
| OBB      | `(1, 3, 1024, 1024)` | `(1, 20, 21504)`                                          | `(1, 300, 7)`: 4 box values, score, class, angle    |

- **Raw output (`nms=None`)**: detection, segmentation, pose, and OBB models write shape `(batch, values, anchors)`. For each anchor: the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS to this output yourself. See the [ONNX output format](onnx.md#output-format) for the layout of each task.
- **NMS-free head (`nms=False`)**: YOLO26 detection, segmentation, pose, and OBB models write 300 rows per image, each with the box corners x1, y1, x2, y2 in input pixels, the score, the class index, and then the values of the task, so no NMS is needed. OBB rows hold the rotated box center x, center y, width, and height, the score, the class index, and the angle in radians. Keep the rows whose score is above your confidence threshold.
- **Segmentation**: `output1` holds the mask prototypes.
- **Semantic segmentation**: the output is the score of every class for every pixel, `(1, 19, 1024, 1024)` for the Cityscapes checkpoint, not a class map as in the [ONNX export](onnx.md#output-format). Ultralytics takes the argmax after inference; an application does the same. The depth output is a depth map. Both are the same for every value of `nms`.

On a Mac, the outputs of `YOLO()` with each export were compared with an ONNX FP32 export of the same checkpoint with the same `nms`, on one preprocessed image per task (`bus.jpg`, OBB on `boats.jpg`). The FP32 cells give the largest absolute difference, the FP16 cells of the raw output the correlation, and the `nms=False` cells the detections above a score of 0.25 that both models returned; for semantic segmentation, the share of pixels where the class with the highest score equals the class of the ONNX class map:

| Task     | FP32, `nms=None`             | FP32, `nms=False`                                | FP16, `nms=None`                                 | FP16, `nms=False`                          |
| -------- | ---------------------------- | ------------------------------------------------ | ------------------------------------------------ | ------------------------------------------ |
| Detect   | 0.0037                       | 5 of 5 detections, largest difference 0.0002     | correlation 1.0000                               | 5 of 5 detections, largest difference 0.14 |
| Segment  | 0.0051                       | 4 of 4 detections, largest difference 0.0003     | correlation 1.0000                               | 4 of 4 detections, largest difference 0.23 |
| Semantic | 100.00% of pixels same class | Same as `nms=None`                               | 99.88% of pixels same class                      | Same as `nms=None`                         |
| Depth    | 0.0001                       | Same as `nms=None`                               | correlation 1.0000                               | Same as `nms=None`                         |
| Classify | 0.0000, same top-1 class     | Same as `nms=None`                               | correlation 0.9996, same top-1 and top-5 classes | Same as `nms=None`                         |
| Pose     | 0.0058                       | 4 of 4 detections, largest difference 0.0001     | correlation 1.0000                               | 4 of 4 detections, largest difference 0.35 |
| OBB      | 0.0068                       | 171 of 171 detections, largest difference 0.0002 | correlation 1.0000                               | 171 detections in both, same classes       |

In the FP32 cells, the largest difference is 0.0068. Direct calls of the Python runtime with the default and the CPU-only specialization gave the same outputs as `YOLO()`.

`YOLO("yolo26n.aimodel")` reads the layout from the metadata, so you only need these details when you run the asset in your own code.

### Run with the Core AI Python Runtime

To run the asset on a Mac without the Ultralytics package, load it with the `coreai.runtime` module of `coreai-core`, which `coreai-torch` installs. Loading the model and running it are asynchronous calls:

```python
# Export first: yolo export model=yolo26n.pt format=coreai nms=False
import asyncio

import numpy as np
from coreai.runtime import AIModel, NDArray


async def main():
    model = await AIModel.load("yolo26n.aimodel")
    function = model.load_function("main")
    image = np.zeros((1, 3, 640, 640), dtype=np.float32)  # your letterboxed image: RGB, 0-1, NCHW
    outputs = await function({"images": NDArray(image)})
    detections = outputs["output0"].numpy()[0]  # shape (300, 6): x1, y1, x2, y2, score, class
    return detections[detections[:, 4] > 0.25]  # keep rows above the confidence threshold


print(asyncio.run(main()))
```

On `bus.jpg`, letterboxed as Ultralytics does it, these calls returned the same 5 detections, classes, and scores as `YOLO()` with the FP32 asset, and so did an FP16 `nms=False` asset with a `float16` array. Run it on a Mac with `coreai-core` installed. Assets exported with `coreai-core` 1.0.0 run slowly in it, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

## Deploy

On an iPhone, iPad, or Mac with the 27 generation of Apple's operating systems, add the `.aimodel` directory to your app and run it with the Core AI framework, or use the [Ultralytics iOS SDK](#ultralytics-ios-sdk-and-flutter-plugin), which loads it and decodes the outputs for you. On a Mac, `YOLO("yolo26n.aimodel")` runs the asset with the `coreai.runtime` module of `coreai-core`, which Ultralytics installs on first use unless `YOLO_AUTOINSTALL=false`:

- **Compute units**: Ultralytics loads the asset with the default specialization options of Core AI, which let the system choose among the CPU, GPU, and Neural Engine, and there is no argument to choose them. With `coreai-core` 1.0.0 in Python on an Apple M4, the default, CPU-only, GPU-preferred, and Neural Engine-preferred options ran at the same speed, and the default and CPU-only options gave identical outputs. An application chooses the options when it loads the model, see [Managing model specialization and caching](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching).
- **First load**: Core AI specializes an asset for the device the first time it loads it and caches the result. In Swift on an Apple M4, loading a YOLO26n asset and running it once took 0.37 to 0.58 s the first time and 74 ms in a new process afterwards. Apple's [ahead-of-time compilation](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time) shortens the first load on a device.
- **Fixed input shape**: the model always runs at its export `imgsz` and `batch`, which Ultralytics reads from `metadata.json`. An `imgsz` passed to `predict` or `val` is ignored: `imgsz=320` and `imgsz=1280` gave the detections of the 640 model on one test image. Smaller batches are padded to the export batch size and larger ones are split into chunks of it, so any number of images works; an asset exported with `batch=2` gave the same detections and COCO8 accuracy as one exported with `batch=1`.
- **`device`**: the `device` argument does not select the Core AI hardware. `device="mps"` gave the same detections and COCO8 accuracy as `device="cpu"`.
- **Directory name**: keep the name of the asset directory ending in `.aimodel`, in lower case. Ultralytics recognizes the format by `.aimodel` anywhere in the name, but the Core AI runtime opens only a directory whose name ends in it: `yolo26n_copy.aimodel` loaded, while `YOLO26N.AIMODEL`, `model.aimodel.bak`, and `yolo26n.aimodel.zip` failed with `RuntimeError: The operation couldn’t be completed. No such file or directory`, and a directory without `.aimodel` in its name with `TypeError: model='detector' is not a supported model format.`
- **Files to copy**: copy the whole `.aimodel` directory. Without `metadata.json`, a YOLO26n detection model loaded with the class names `class0`, `class1`, and so on; without `main.hash`, or with other files added to the directory, it loaded and predicted as before.
- **Assets from Linux**: an asset exported on x86-64 Linux ran on the Mac with the same detections and COCO8 accuracy as one exported on the Mac, for YOLO26n detection in FP32 and FP16 and for segmentation.
- **Several models**: two Core AI models, a detection and a segmentation model, ran alternately in one process.

### Choosing the Head

With `nms=False`, YOLO26 exports its end-to-end head, which selects detections inside the graph. Core AI has no top-k primitive, so that selection lowers to a full sort and is charged a fixed cost at the Apple Neural Engine partition boundary, about 1.7 ms regardless of `max_det` ([apple/coreai-torch#66](https://github.com/apple/coreai-torch/issues/66)). Exporting an Ultralytics YOLO26 model with `nms=None`, the default, writes the raw `(1, 84, 8400)` predictions instead and leaves [non-maximum suppression](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) to the predictor:

```bash
yolo export model=yolo26n.pt format=coreai nms=None quantize=16
```

On an iPhone 17 Pro with iOS 27.0, YOLO26n at 640 took 3.01 ms with the head in the graph and 1.28 ms without it (FP16, ahead-of-time compiled, three interleaved blocks of 50 iterations). In another comparison run on the same phone, the YOLO26n model with `nms=False` took 3.06 ms in Core AI against 1.53 ms in Core ML. These phone measurements were carried over from an earlier version of this page and were not repeated. In Swift on an Apple M4 Mac, the FP16 YOLO26n assets with `nms=False` and `nms=None` took 6.9 and 7.2 ms for the model call, and the raw head still needs NMS after it. Both heads run through `YOLO()`. On the last 500 images of COCO val2017, YOLO26n in FP16 reached 41.1 mAP<sup>val</sup> 50-95 with `nms=False` and 42.0 with `nms=None`, the same gap as between the two heads of the PyTorch model, see [Measured Performance](#measured-performance). Use `nms=False` when a single graph call must return finished detections, or keep the default `nms=None` for external NMS, which the Ultralytics SDKs use.

### Swift

On iOS 27 or macOS 27, an application loads and runs the exported Ultralytics YOLO26 asset through Apple's Core AI Swift API. The asset has the entrypoint `main`, takes a single `images` input of shape `[batch, 3, height, width]`, and returns `output0` (instance segmentation models also return `output1`):

```swift
import CoreAI

let modelURL = Bundle.main.url(forResource: "yolo26n", withExtension: "aimodel")!
let model = try await AIModel(contentsOf: modelURL)
guard let function = try model.loadFunction(named: "main") else {
    throw AppError.missingInferenceFunction
}

let outputs = try await function.run(inputs: ["images": imageTensor])
```

`imageTensor` is the letterboxed image as an `NDArray` of that shape, in `Float` for an FP32 export or `Float16` for an FP16 export, and `AppError` is an error type of your app. The block was compiled with Swift 6.4.2 and run on macOS 27.2.1, with these two definitions added: YOLO26n assets took 6.6 ms per call in FP32 and 7.2 ms in FP16, the median of 20 calls on an Apple M4 after the first one, with the default specialization.

### Ultralytics iOS SDK and Flutter Plugin

The [Ultralytics iOS SDK](https://github.com/ultralytics/yolo-ios-app) (8.9.15 and later) and the [Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) (0.6.15 and later) load `.aimodel` assets as an opt-in on iOS 27 devices; [Core ML](coreml.md#ultralytics-ios-sdk-and-flutter-plugin) remains their default. Unlike the Core ML and Vision workflow, the Core AI path in the SDK does its own letterbox preprocessing and `NDArray` construction, reads the same Ultralytics metadata from the asset's `metadata.json`, and reuses the Core ML output decoders. It loads when an app passes an `.aimodel` path or an `.aimodel.zip` URL, and it is not available in the iOS Simulator. The SDK documentation gives the export for its assets:

```bash
# FP16 with the raw head (nms=None), as the SDK assets
yolo export model=yolo26n.pt format=coreai quantize=16 imgsz=640
```

The [on-device comparison](https://github.com/ultralytics/yolo-ios-app/blob/main/docs/performance.md) of the iOS SDK on an iPhone 17 Pro puts Core AI close to Core ML for the full pipeline: ahead for detection, classification, and OBB, level for segmentation and pose, and behind for semantic segmentation, depth estimation, and CPU-only inference, with FP16 Core AI assets about twice the download size of the INT8 Core ML assets, so the SDK and the Flutter plugin keep Core ML as their default. The SDK was not run for this page.

### Core AI or Core ML

| Capability                       | Core AI                                                                              | Core ML                                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Model artifact                   | `.aimodel`                                                                           | `.mlpackage` or `.mlmodel`                                                                |
| Ultralytics export               | `format=coreai`, FP32 and FP16                                                       | `format=coreml`, FP32, FP16, and INT8, with optional embedded NMS                         |
| Operating systems                | iOS 27, macOS 27, and the rest of that generation                                    | Broad support across current and earlier Apple operating systems                          |
| Apple runtime API                | `AIModel`, `InferenceFunction`, and `NDArray`                                        | `MLModel`, often through `VNCoreMLModel` and `VNCoreMLRequest`                            |
| Conversion workflow              | PyTorch `torch.export` through `coreai-torch`                                        | TorchScript conversion through `coremltools`                                              |
| Image integration                | Applications prepare tensors or use Core AI image descriptors and buffers            | Direct integration with the Vision framework for image scaling, orientation, and requests |
| Hardware                         | CPU, GPU, and Apple Neural Engine                                                    | CPU, GPU, and Apple Neural Engine                                                         |
| Model preparation                | Specialization at installation or first use, with optional ahead-of-time compilation | Xcode or on-device model compilation                                                      |
| Custom operations                | Custom Core AI lowerings and Metal kernels                                           | Core ML custom layers and supported MIL operations                                        |
| Ultralytics iOS and Flutter SDKs | Opt-in on iOS 27 and later devices                                                   | Fully supported, and the default                                                          |

Use Core ML when an application needs earlier Apple operating systems, the default path of the Ultralytics iOS or Flutter SDK, Vision framework image handling, INT8 models, or embedded NMS, and for model types such as decision trees and tabular pipelines, which Apple directs to Core ML. Evaluate Core AI when the application can require the 27 generation of Apple's operating systems and needs explicit specialization and cache control, dynamic or stateful execution, custom Metal kernels, or Core AI's graph debugging and profiling tools; the Ultralytics export itself writes fixed-shape assets with one inference function. The two frameworks are expected to coexist while applications move between them: a `.aimodel` cannot replace an `.mlpackage` in an existing app without a Core AI implementation of loading, preprocessing, inference, and output decoding.

### Deployment Guides

- **[Core AI framework documentation](https://developer.apple.com/documentation/coreai)**: Apple's reference for `AIModel`, inference functions, and `NDArray`.
- **[`AIModel`](https://developer.apple.com/documentation/coreai/aimodel)**, **[Managing model specialization and caching](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching)**, and **[Compiling Core AI models ahead of time](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time)**: how an app loads an asset, controls specialization, and ships a precompiled model to cut the first load.
- **[Core AI PyTorch Extensions](https://apple.github.io/coreai-torch/)**: the `coreai-torch` converter that this export uses, with custom lowerings and Metal kernels for unsupported operations.
- **[Core AI Optimization](https://apple.github.io/coreai-optimization/)**: Apple's compression tools for Core AI models.
- **[Apple Core AI models repository](https://github.com/apple/coreai-models)**: ready-to-run models and Swift utilities for common inference patterns.
- **[Ultralytics Core ML integration](coreml.md)**: the Core ML export for earlier Apple operating systems and for the default path of the Ultralytics SDKs.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare Core AI with the other export formats.

## Limitations and Troubleshooting

- **Inference needs macOS on Apple silicon**: on Linux, `YOLO("yolo26n.aimodel")` succeeds, but the first `predict` or `val` fails with `RuntimeError: The operation could not be completed. No such file or directory` when the Core AI runtime opens the asset. Export on Linux, then predict and validate on a Mac or on the device. `yolo benchmark` reports this format as `inference requires macOS` on Linux.
- **Slow inference through `YOLO()`**: assets exported with `coreai-core` 1.0.0 took 1.7 to 1.8 s per image in `val` for YOLO26n detection on a MacBook Air with an Apple M4 and macOS 27.2.1, and 7.2 to 7.6 s for semantic segmentation at 1024 pixels, for FP32 and FP16 exports. Direct calls of the Python runtime took 1.7 to 2.0 s for the YOLO26n detection model with every specialization option and with the macOS framework selected by `USE_OS_COREAI=1`, and the runtime of `coreai-core` 1.0.0b3 took 1.8 s for such an asset too. The same assets ran in 6.6 ms (FP32) and 7.2 ms (FP16) in Swift on that Mac. An FP32 asset exported with `coreai-torch` 0.4.3, `coreai-core` 1.0.0b3, and PyTorch 2.13.0 took 11 ms through `YOLO()` with that version, and its FP16 asset failed to load through `YOLO()` with `Program load failed — operation not permitted (check entitlements)`. The predictions and accuracy are correct; measure speed in your app.
- **`Dev installation detected. Using local Core AI Framework.`**: `coreai-core` 1.0.0 logs this line when it loads on macOS 27, also when it was installed from PyPI, and then uses the Core AI runtime that it ships instead of the one of macOS. `USE_OS_COREAI=1` selects the one of macOS; in the measurements above it did not make inference faster.
- **Assets from `coreai-torch` 0.4.3**: with `coreai-core` 1.0.0, `YOLO()` fails on an asset exported with `coreai-torch` 0.4.3 and `coreai-core` 1.0.0b3 with `RuntimeError: Kernel MPSGraph invoked with invalid parameters. Unresolved symbol: MPSGraph (prepare)`. With `USE_OS_COREAI=1` the same asset ran in 20.6 ms through `YOLO()` on macOS 27.2.1. Exporting the model again also fixes the error, but the new asset runs at the speed of the bullet above.
- **Export host**: the export runs on macOS 26 or newer on Apple silicon and on x86-64 Linux, the platforms of the `coreai-core` packages. Elsewhere, for example on ARM64 Linux such as a Raspberry Pi and, by the code, on Windows, it fails with `Core AI export requires macOS>=26 on Apple silicon or x86_64 Linux, the platforms coreai-core publishes.` The Linux package needs glibc 2.34 or newer, for example Ubuntu 22.04 or newer.
- **Python version**: `coreai-torch` needs Python 3.11 or newer. On Python 3.10, the first export tries to install it and fails with `No matching distribution found for coreai-torch>=0.4.3` and then `No module named 'coreai_torch'`. On Python 3.15, `pip install "ultralytics[export-coreai]"` installs no Core AI package, because the extra covers Python 3.11 to 3.14; the first export installs it with pip, or install it yourself with `pip install coreai-torch`.
- **PyTorch version**: with PyTorch older than 2.8 the export fails with `Core AI export requires torch>=2.8.0 but torch==2.7.1+cpu is installed`. With PyTorch newer than 2.13, `coreai-torch` 0.4.4 warns `coreai-torch has only been validated with torch<=2.13.0; found torch 2.15.0. Some functionality may not work as expected.` The exports on this page were made with PyTorch 2.15.0 and 2.8.0, and with 2.13.0 in one comparison.
- **Automatic installation turned off**: with `YOLO_AUTOINSTALL=false`, an environment without `coreai-torch` logs `Ultralytics requirement ['coreai-torch>=0.4.3'] not found, AutoUpdate disabled by YOLO_AUTOINSTALL=False` and the export fails with `No module named 'coreai_torch'`. Install `"ultralytics[export-coreai]"` first, as in the [Quickstart](#quickstart).
- **No INT8**: `quantize=8` and the other INT8 values fail with `quantize=8 (INT8) is not supported; format='coreai' supports quantize=16 (FP16), 32 or None (FP32).` Export FP16 for the smallest asset, or use [Core ML](coreml.md) for INT8 models.
- **Fixed input size and batch**: the asset has no dynamic shapes. `dynamic=True` fails with `argument 'dynamic' is not supported for format='coreai'`, and an `imgsz` passed to `predict` or `val` is ignored. Export at the size you will run.
- **Embedded NMS**: `nms=True` logs `format=coreai does not support embedded NMS; using nms=None.` and writes the raw output, so NMS stays on the host. Core ML can embed NMS for detection, segmentation, and pose models.
- **NMS-free head on the Neural Engine**: with `nms=False`, the selection of the end-to-end head costs a fixed time at the Apple Neural Engine boundary on an iPhone 17 Pro ([apple/coreai-torch#66](https://github.com/apple/coreai-torch/issues/66)), and on that phone with iOS 27.0 an FP16 YOLO26n pose model with `nms=False` returned no detections with the default specialization ([apple/coreai-torch#115](https://github.com/apple/coreai-torch/issues/115)). Both issues are open; the raw head (`nms=None`) avoids them, see [Choosing the Head](#choosing-the-head).
- **Semantic segmentation output**: the asset returns the score of every class for every pixel, and the argmax runs after inference. An argmax inside the Core AI graph returned wrong class maps with the default specialization on an iPhone 17 Pro ([apple/coreai-torch#116](https://github.com/apple/coreai-torch/issues/116), open). On the Mac mini in [Measured Performance](#measured-performance), this postprocessing took 6.4 to 7.4 ms against 0.4 ms for the class map of the Core ML export.
- **FP16 assets can abort on load**: Some FP16 `.aimodel` assets fail to load their Apple Neural Engine program and MPSGraph raises a failed assertion, which ends the process rather than falling back ([apple/coreai-torch#67](https://github.com/apple/coreai-torch/issues/67), open). This happens inside Apple's runtime, before any Ultralytics code runs, and the same asset loads with a CPU-only specialization. Fall back to FP32 if an asset does this; the FP16 SDK assets checked on an iPhone 17 Pro (every nano model, detect at every size, and the largest model of each task) loaded without an abort.
- **RT-DETR**: the export of `rtdetr-l.pt` fails with `ValueError: The exported program contains unsupported ATen ops: aten.grid_sampler_2d.default.` Export RT-DETR models to [Core ML](coreml.md) or [ONNX](onnx.md) instead.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='coreai' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends. Export a non-QAT checkpoint to this format instead.`
- **Asset name**: every export of one checkpoint writes the same `yolo26n.aimodel`, whatever its `quantize`, `imgsz`, or `nms`, so the later export replaces the earlier one. Rename the directory to keep both, and keep the `.aimodel` ending, see [Deploy](#deploy).
- **No iOS Simulator**: the iOS Simulator does not include Core AI, so test on a device with iOS 27.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Measured Performance

The first table below was measured for an earlier version of this page on a Mac mini and carried over unchanged; the accuracy and size tables were measured for this page. The [YOLO26 model page](../models/yolo26.md) lists the accuracy of the PyTorch checkpoints of each task.

### Apple M4 Mac mini

End-to-end single-image inference for YOLO26n FP16 (`quantize=16`) Core ML and Core AI exports with the default raw
head (`nms=None`) on a Mac mini with an [Apple M4](https://support.apple.com/en-us/121555) (4 Performance and 6
Efficiency CPU cores, 10-core GPU, 16-core Neural Engine), 16 GB memory and macOS 27.0, using `ultralytics` 8.4.168,
`coremltools` 9.0 for Core ML inference, and `coreai-torch` 0.4.3 with `coreai-core` 1.0.0b3 for Core AI inference
on Python 3.13. Each cell shows the **total time** (preprocessing + inference + postprocessing) with the per-stage
split beneath it.

| Model         | Task     | size<br><sup>(pixels)</sup> | Core ML CPU<br><sup>`CPU_ONLY`<br>(ms)</sup> | Core ML CPU + ANE preferred<br><sup>`CPU_AND_NE`<br>(ms)</sup> | Core AI CPU<br><sup>`cpu_only()`<br>(ms)</sup> | Core AI CPU + ANE preferred<br><sup>`neural_engine()`<br>(ms)</sup> |
| ------------- | -------- | --------------------------- | -------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------- |
| YOLO26n       | Detect   | 640                         | 14.4<br><sup>0.6 / 13.4 / 0.4</sup>          | 7.6<br><sup>0.6 / 6.7 / 0.4</sup>                              | 16.8<br><sup>0.6 / 15.9 / 0.3</sup>            | **2.8**<br><sup>0.6 / 2.0 / 0.2</sup>                               |
| YOLO26n-seg   | Segment  | 640                         | 18.7<br><sup>0.6 / 16.5 / 1.5</sup>          | 9.2<br><sup>0.6 / 7.1 / 1.5</sup>                              | 25.3<br><sup>0.6 / 23.2 / 1.5</sup>            | **5.1**<br><sup>0.6 / 3.1 / 1.4</sup>                               |
| YOLO26n-sem   | Semantic | 640                         | 33.9<br><sup>1.3 / 32.2 / 0.4</sup>          | 73.7<br><sup>1.4 / 71.9 / 0.4</sup>                            | 47.1<br><sup>1.2 / 38.5 / 7.4</sup>            | **18.7**<br><sup>1.2 / 11.1 / 6.4</sup>                             |
| YOLO26n-depth | Depth    | 640                         | 36.8<br><sup>0.8 / 35.5 / 0.5</sup>          | 12.1<br><sup>0.9 / 10.7 / 0.5</sup>                            | 40.2<br><sup>0.8 / 39.0 / 0.5</sup>            | **7.5**<br><sup>0.7 / 6.3 / 0.5</sup>                               |
| YOLO26n-cls   | Classify | 224                         | 3.8<br><sup>1.9 / 1.8 / 0.0</sup>            | 3.3<br><sup>1.9 / 1.4 / 0.0</sup>                              | 3.0<br><sup>1.9 / 1.0 / 0.0</sup>              | **2.4**<br><sup>1.9 / 0.6 / 0.0</sup>                               |
| YOLO26n-pose  | Pose     | 640                         | 15.5<br><sup>0.6 / 14.6 / 0.3</sup>          | 7.0<br><sup>0.6 / 6.2 / 0.3</sup>                              | 17.8<br><sup>0.5 / 17.0 / 0.3</sup>            | **2.7**<br><sup>0.5 / 2.0 / 0.2</sup>                               |
| YOLO26n-obb   | OBB      | 640                         | 32.7<br><sup>1.3 / 31.1 / 0.2</sup>          | 16.9<br><sup>1.5 / 15.2 / 0.2</sup>                            | 37.2<br><sup>1.1 / 35.9 / 0.2</sup>            | **5.7**<br><sup>1.3 / 4.3 / 0.1</sup>                               |

- **Speed** values are **single-image burst latencies**: the mean of 15 `predict` calls after 3 warmup calls on
  `bus.jpg` through the Ultralytics Python API, with each model and compute unit in a fresh process. CPU/accelerator
  order alternated between tasks in one sequential sweep. Core ML rows load with `coremltools.ComputeUnit.CPU_ONLY` or
  `CPU_AND_NE`; Core AI rows specialize with `SpecializationOptions.cpu_only()` or
  `SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())`, with final operation
  placement controlled by each framework.
- Detect, segment, classify, pose and OBB returned the same predictions in both formats on every compute unit. The
  Core ML FP16 semantic model runs slower with the Neural Engine preferred than on CPU only on this Mac, and Core AI
  semantic postprocessing takes 6.4 to 7.4 ms against 0.4 ms for Core ML.
- The semantic segmentation, depth, and OBB models of this table were exported at 640 pixels, not at their default sizes.
- Assets exported with `coreai-core` 1.0.0 run much slower in the Python runtime than in this table, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).
- Compare the on-device iPhone 17 Pro results in the [Core ML integration](coreml.md#measured-performance).

### Accuracy

YOLO26n detection on the last 500 images of COCO val2017 in sorted order, validated through `YOLO()` at 640 pixels and `batch=1` on a MacBook Air with an Apple M4 and macOS 27.2.1. PyTorch ran on the CPU with `rect=False`, at the same 640x640 input as the Core AI models. The accuracy differs from the values for the full validation set on the [YOLO26 model page](../models/yolo26.md):

| Metric                  | PyTorch | Core AI FP32 | Core AI FP32, `nms=False` | Core AI FP16 | Core AI FP16, `nms=False` |
| ----------------------- | ------- | ------------ | ------------------------- | ------------ | ------------------------- |
| mAP<sup>val</sup> 50-95 | 42.0    | 42.0         | 41.2                      | 42.0         | 41.1                      |
| mAP<sup>val</sup> 50    | 56.8    | 56.8         | 55.6                      | 56.9         | 55.5                      |

- **FP16** gave the mAP<sup>val</sup> 50-95 of FP32, 42.0, with the raw output.
- **`nms=False`** lowered mAP<sup>val</sup> 50-95 by 0.8 in FP32 and by 0.9 in FP16 against `nms=None`. The PyTorch model has the same gap between its heads, 40.9 against 40.1 on the full validation set, see the [YOLO26 model page](../models/yolo26.md).
- Larger models were not validated: with assets exported by `coreai-core` 1.0.0, one YOLO26n image took about 1.7 s.

Measured with `ultralytics` 8.4.218, `coreai-torch` 0.4.4, `coreai-core` 1.0.0, and PyTorch 2.15.0 on Python 3.14.

### Size

The size of the `.aimodel` asset of each YOLO26 detection model at 640 pixels, in MB (10^6 bytes), exported on Linux:

| Model   | FP32 (MB) | FP16 (MB) |
| ------- | --------- | --------- |
| YOLO26n | 9.9       | 5.0       |
| YOLO26s | 38.2      | 19.2      |
| YOLO26m | 81.9      | 41.0      |
| YOLO26l | 99.5      | 49.9      |
| YOLO26x | 223.2     | 111.7     |

## About Core AI

Core AI was introduced with iOS 27 and macOS 27 as Apple's framework for running AI models on device across the CPU, GPU, and Neural Engine, from compact vision models to large generative models. An `.aimodel` asset is unspecialized: Core AI specializes it for the device that loads it and caches the result, and applications can trigger specialization early or ship an ahead-of-time compiled model. Next to the Swift framework, Apple publishes the PyTorch converter [`coreai-torch`](https://github.com/apple/coreai-torch), the [Core AI Optimization](https://apple.github.io/coreai-optimization/) compression tools, the Core AI Debugger, which traces tensor values back to the Python source, and Xcode and Instruments profiling. For the full overview, see [Apple's Core AI page](https://developer.apple.com/core-ai/), and follow the [Ultralytics roadmap](https://www.ultralytics.com/roadmap) for changes to the integration.

## FAQ

### Does Core AI replace Core ML for Ultralytics YOLO models?

No, Ultralytics exports both Core AI and Core ML models, and the two formats are expected to coexist while apps move to the 27 generation of Apple's operating systems. Core ML remains the default of the Ultralytics iOS SDK and Flutter plugin and the format for earlier operating systems, INT8 models, and embedded NMS, and Apple continues to support it. Any later change of the default depends on the adoption of those operating systems, stable Core AI tooling, and on-device performance. See [Core AI or Core ML](#core-ai-or-core-ml).

### Can I export or test an Ultralytics Core AI model without a Mac?

You can export without one: Ultralytics exports Core AI models on x86-64 Linux, and [Ultralytics Platform](../platform/train/models.md#export-model) exports in the cloud. Running the model needs Apple hardware: `YOLO("yolo26n.aimodel")` predicts and validates only on macOS on Apple silicon, and an app runs it with the Core AI framework of iOS 27, macOS 27, or the rest of that generation, not in the iOS Simulator. An asset exported on Linux gave the same detections on a Mac as one exported there.

### Should I export Ultralytics YOLO26 to Core AI in FP16 or FP32?

Export FP16 (`quantize=16`) for an app: it halves the asset, 5.0 MB against 9.9 MB for YOLO26n, and it is the precision of the Ultralytics iOS SDK assets. On the last 500 images of COCO val2017, FP16 YOLO26n reached 42.0 mAP<sup>val</sup> 50-95 against 42.0 in FP32 through `YOLO()` on a Mac. There is no INT8 Core AI export; use [Core ML](coreml.md) when you need INT8 weights.

### Why is `YOLO()` inference of an Ultralytics Core AI model slow on my Mac?

Ultralytics YOLO26 assets exported with `coreai-core` 1.0.0 took 1.7 to 1.8 s per image for YOLO26n detection on a MacBook Air with an Apple M4 in the Python runtime that `YOLO()` uses, while the same asset ran in 6.6 ms through the Core AI framework in Swift. The predictions are correct, so use `predict` and `val` in Python to check an export, and measure the speed in your app. See [Limitations and Troubleshooting](#limitations-and-troubleshooting).

### Can I rename an Ultralytics Core ML `.mlpackage` to `.aimodel`?

No, an Ultralytics Core ML `.mlpackage` and a Core AI `.aimodel` contain different model representations and are loaded by different frameworks, so a renamed Core ML package does not load in Core AI. Export the `.pt` checkpoint again with `format="coreai"`.
