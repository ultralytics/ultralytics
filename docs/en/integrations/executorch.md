---
comments: true
description: Export Ultralytics YOLO26 models to the ExecuTorch .pte format with format=executorch for CPU inference through XNNPACK on Android, iOS, and embedded Linux. Covers FP32 and INT8, calibration, output shapes, and benchmarks.
keywords: YOLO26, ExecuTorch, ExecuTorch export, pte, XNNPACK, PyTorch, INT8 quantization, on-device inference, Android, iOS, Raspberry Pi, embedded Linux, edge AI, mobile deployment
---

# PyTorch ExecuTorch Export

[ExecuTorch](https://docs.pytorch.org/executorch/) is the PyTorch runtime for on-device inference on phones, embedded boards, and other [edge](https://www.ultralytics.com/glossary/edge-computing) devices. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="executorch"` creates a `yolo26n_executorch_model/` directory that holds a `.pte` program for XNNPACK, the ExecuTorch backend for ARM and x86 CPUs, in FP32 or INT8.

## At a Glance

| Property          | Details                                                                                                              |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `executorch`                                                                                                         |
| Output            | `yolo26n_executorch_model/`, a directory with `model.pte` and [`metadata.yaml`](#output-format)                      |
| Runs on           | ARM and x86 CPUs through XNNPACK: Android and iOS devices, embedded Linux boards such as the Raspberry Pi, desktops  |
| Runtime           | ExecuTorch: the `executorch` Python package, or the ExecuTorch libraries for Android, iOS, and C++                   |
| Precision         | FP32 (default) and INT8. YOLO26n is 9.9 MB in FP32 and 3.2 MB in INT8                                                |
| Export host       | Linux x86-64 and macOS on Apple silicon, with Python 3.10 or newer and PyTorch 2.9 or newer. The export needs no GPU |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to ExecuTorch, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package in a virtual environment, here on Linux or macOS:

```bash
python3 -m venv executorch-env && source executorch-env/bin/activate
pip install ultralytics
```

The first export installs the [`executorch`](https://pypi.org/project/executorch/) package and the packages it requires, such as `torchao`: 23 to 43 packages in the environments used for this page, none of which replaced an installed package. The export logs `Restart runtime or rerun command for updates to take effect` and still completes in the same run.

To install `executorch` before the first export, run `pip install "ultralytics[export-executorch]"` instead, which also limits `numpy` to 2.3.5 or older; the export then installs nothing. For an editable repository install, use `pip install -e ".[export-base,export-executorch]"`. The export has these version requirements:

| Requirement      | Version                                                                                                                                                                                                                                                              |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python           | 3.10 or newer for FP32 and 3.11 or newer for INT8. FP32 exports ran with every version from 3.10 to 3.15 and INT8 exports with 3.11 to 3.15. On Python 3.10, pip installs PyTorch 2.14.1 and `executorch` 1.5.1, which exports FP32 only                             |
| PyTorch          | 2.9.0 or newer for FP32, the version the export checks for; the oldest version run for this page was 2.12.1. INT8 needs PyTorch 2.15.0, the version that `executorch` 1.6.0 requires                                                                                 |
| `executorch`     | Installed by the first export. INT8 needs 1.6.0 or newer. The first export installed 1.6.0 with PyTorch 2.15.0, 1.5.1 with PyTorch 2.13.0 and 2.14.1, and 1.3.1 with PyTorch 2.12.1; below PyTorch 2.13, Ultralytics installs `executorch` older than 1.5            |
| Operating system | `executorch` 1.6.0 has packages for Linux x86-64 and ARM64 with glibc 2.28 or newer, macOS 14 or newer on Apple silicon, and Windows x86-64. Exports for this page ran on Ubuntu 24.04 x86-64 and on macOS 27 with an Apple M4; ARM64 Linux and Windows were not run |

For other installation methods, see the [installation guide](../quickstart.md). Then export your model and load the exported directory to run inference or validate its accuracy:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to ExecuTorch format
        model.export(format="executorch")  # creates 'yolo26n_executorch_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to ExecuTorch format
        yolo export model=yolo26n.pt format=executorch # creates 'yolo26n_executorch_model/'
        ```

On a shared AMD EPYC 9655 server, FP32 exports of YOLO26n took 9 to 12 s with `executorch` installed, and 19 to 29 s for the first export in a new environment, which also installs it. INT8 exports of YOLO26n took 20 to 25 s with the 4 calibration images of `coco8.yaml` and 55 to 60 s with 500 COCO images. The INT8 export of YOLO26x with 500 images took 267 s. The exported model runs on the export host, so you can test it there:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ExecuTorch model
        model = YOLO("yolo26n_executorch_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported ExecuTorch model
        yolo predict model=yolo26n_executorch_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ExecuTorch model
        model = YOLO("yolo26n_executorch_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported ExecuTorch model
        yolo val model=yolo26n_executorch_model data=coco8.yaml
        ```

## Supported Tasks

ExecuTorch export supports all seven Ultralytics tasks: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with [YOLO26](../models/yolo26.md), the only family that ships those heads. For this page, the YOLO26n checkpoint of every task was exported in FP32 and in INT8 and run on the export host, and so were [YOLO11](../models/yolo11.md) and [YOLOv10](../models/yolov10.md) detection models; [YOLOv8](../models/yolov8.md) and [YOLOv5](../models/yolov5.md) detection models were exported and run in FP32. [RT-DETR](../models/rtdetr.md) exports in FP32 only, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

## Export Arguments

| Argument   | Type                      | Default        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------- | ------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `format`   | `str`                     | `'executorch'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                         |
| `imgsz`    | `int` or `tuple`          | `640`          | Image size of the model input, fixed in the `.pte` file: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect, segment, and pose models and for semantic models pretrained on ADE20K, `1024` for OBB models and for semantic models pretrained on Cityscapes, `768` for depth, and `224` for classify. |
| `quantize` | `int` or `str`            | `None`         | Precision: `8` (static INT8, calibrated on `data`) or `32`/unset (FP32). `16`, `'w8a16'`, and `'w8a32'` raise an error, see [Precision](#precision). Replaces the deprecated `half`/`int8` flags.                                                                                                                                                                                                                                          |
| `data`     | `str`                     | `None`         | Dataset YAML with the calibration images for `quantize=8`; classification instead takes a dataset directory or a built-in dataset name. If omitted, Ultralytics uses the small default dataset of the task, such as `coco8.yaml` for detection. See [Calibration](#calibration).                                                                                                                                                           |
| `split`    | `str`                     | `'val'`        | Dataset split that the export calibrates on.                                                                                                                                                                                                                                                                                                                                                                                               |
| `fraction` | `float`, `int`, or `list` | `1.0`          | Calibration subset: a ratio, an image count (integers above `1`), or a `[train, val, test]` list, from which the entry of the calibration split is used.                                                                                                                                                                                                                                                                                   |
| `batch`    | `int`                     | `1`            | Batch size of the exported model, fixed in the input shape. Also the batch size used for calibration.                                                                                                                                                                                                                                                                                                                                      |
| `nms`      | `bool`, optional          | `None`         | Detection, segmentation, pose, and OBB models always export the raw output that needs NMS: `True` (embedded NMS) and `False` (the NMS-free head) are not available, see [Output Format](#output-format).                                                                                                                                                                                                                                   |
| `device`   | `str`                     | `None`         | The export always runs on the CPU, whatever value is passed.                                                                                                                                                                                                                                                                                                                                                                               |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Precision

`quantize` selects the precision of the weights and of the activations, the values that flow between layers:

| `quantize`    | Result                                                                                                               |
| ------------- | -------------------------------------------------------------------------------------------------------------------- |
| unset or `32` | FP32 `model.pte`. YOLO26n is 9.9 MB                                                                                  |
| `8`           | INT8 `model.pte`, calibrated on `data`. YOLO26n is 3.2 MB                                                            |
| `16`          | Fails with `quantize=16 (FP16) is not supported; format='executorch' supports quantize=8 (INT8), 32 or None (FP32).` |
| `'w8a32'`     | Fails with `quantize='w8a32' is not supported; format='executorch' supports quantize=8 (INT8), 32 or None (FP32).`   |
| `'w8a16'`     | Fails with `quantize='w8a16' is not supported; format='executorch' supports quantize=8 (INT8), 32 or None (FP32).`   |

- **INT8**: static [quantization](https://www.ultralytics.com/glossary/model-quantization) with the XNNPACK quantizer of ExecuTorch: 8-bit weights with one scale per output channel and 8-bit activations, whose ranges the export measures on the calibration images.
- **FP32 head**: in detect, segment, pose, and OBB models the head stays in FP32, because a single 8-bit scale over box coordinates and class scores would round the scores to zero. Semantic segmentation, depth, and classification models are quantized completely.
- **Inputs and outputs**: the input and the outputs stay `float32` in both precisions, so code written for the FP32 model runs the INT8 model unchanged.

The sizes of `model.pte` for the official YOLO26n checkpoints at their default export size, in MB (10^6 bytes). The export log prints the size of the whole directory in MiB, as `(9.4 MB)` and `(3.1 MB)` for the detection model:

| Task     | Default `imgsz` | FP32 | INT8 |
| -------- | --------------- | ---- | ---- |
| Detect   | `640`           | 9.9  | 3.2  |
| Segment  | `640`           | 11.1 | 4.5  |
| Semantic | `1024`          | 6.3  | 1.7  |
| Depth    | `768`           | 20.8 | 5.4  |
| Classify | `224`           | 11.3 | 3.0  |
| Pose     | `640`           | 11.9 | 5.3  |
| OBB      | `1024`          | 10.2 | 3.6  |

### Calibration

A `quantize=8` export runs every full batch of the `val` split of `data`, or of the split you set with `split`, through the model to measure the activation ranges. Pass `data` with images that look like your deployment data, and use `fraction` to limit their number:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        model.export(format="executorch", quantize=8, data="coco.yaml", fraction=0.1)  # creates 'yolo26n_executorch_model/'

        # Load the exported INT8 model and run inference
        model = YOLO("yolo26n_executorch_model")
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        yolo export model=yolo26n.pt format=executorch quantize=8 data=coco.yaml fraction=0.1 # creates 'yolo26n_executorch_model/'

        # Run inference with the exported INT8 model
        yolo predict model=yolo26n_executorch_model source='https://ultralytics.com/images/bus.jpg'
        ```

- **Default dataset**: without `data`, the export logs `INT8 export requires a missing 'data' arg for calibration. Using default 'data=coco8.yaml'.` and calibrates on the small default dataset of the task: `coco8.yaml`, `coco8-seg.yaml`, `cityscapes8.yaml`, `depth8.yaml`, `imagenet10`, `coco8-pose.yaml`, or `dota8.yaml`. These are enough to test the export.
- **Calibration size warning**: below 300 images the export logs a warning such as `>300 images recommended for INT8 calibration, found 4 images.`
- **`fraction`**: a ratio or an image count. `fraction=0.5` and `fraction=2` both selected 2 of the 4 validation images of `coco8.yaml`, and `fraction=0.1` selects the first 500 of the 5000 sorted validation images of `coco.yaml`.
- **`split`**: `split="train"` calibrates on the training images. A split that the dataset does not have fails with `coco8.yaml 'test:' images not found`.
- **`batch`**: calibration runs in batches of the export batch size and drops an incomplete last batch. With fewer images than one batch, the images are repeated to fill it.
- **Effect of the calibration images**: on 500 COCO validation images, YOLO26n reached 42.0 mAP50-95 in FP32, and in INT8 41.2 when calibrated on the 4 images of `coco8.yaml`, 41.6 on the 128 images of `coco128.yaml`, and 41.3 on 500 other COCO validation images, see [Benchmarks](#benchmarks). Other models and tasks can react more strongly, so validate an INT8 model on your own data before you deploy it.

## Output Format

The export writes one directory, and deletes an existing directory of that name first:

| File            | Contents                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `model.pte`     | The ExecuTorch program with one method, `forward`, lowered to the XNNPACK backend                                        |
| `metadata.yaml` | Class names, task, image size, batch size, stride, and export arguments, which Ultralytics reads when it loads the model |

The model takes one input with shape `(batch, 3, height, width)`: RGB, channels first (NCHW), `float32`, scaled to 0-1. Ultralytics resizes each image to that size with gray padding (letterbox; classification models are resized and center-cropped instead), and the box coordinates in the output refer to the padded image. The task sets the outputs, which are `float32`. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | Outputs                                                             |
| -------- | --------------- | ------------------------------------------------------------------- |
| Detect   | `640`           | `(1, 84, 8400)`                                                     |
| Segment  | `640`           | `(1, 116, 8400)` and `(1, 32, 160, 160)`                            |
| Semantic | `1024`          | `(1, 19, 1024, 1024)`, a score for each class and pixel             |
| Depth    | `768`           | `(1, 1, 768, 768)`, depth map                                       |
| Classify | `224`           | `(1, 1000)`, class probabilities                                    |
| Pose     | `640`           | `(1, 56, 8400)`: 4 box values, 1 class score, then 17 keypoints x 3 |
| OBB      | `1024`          | `(1, 20, 21504)`: 4 box values, 15 class scores, then the angle     |

- **Raw output**: detection, segmentation, pose, and OBB models write shape `(batch, values, anchors)`. For each anchor: the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS to this output yourself. See the [ONNX output format](onnx.md#output-format) for the layout of each task.
- **RT-DETR**: the output of RT-DETR-L has shape `(1, 300, 6)` instead.
- **`nms`**: the output is the same for every value of `nms`. `nms=False` logs `This export format does not support end2end models, disabling the end2end branch.` for models with an NMS-free head, such as YOLO26, and `nms=True` logs `format=executorch does not support embedded NMS; exporting native outputs for external NMS.`
- **Segmentation**: the 32 mask coefficients are the last 32 values of each anchor, and the second output holds the mask prototypes.
- **Semantic segmentation**: the output holds a score for each class and pixel, not the class map of the [ONNX export](onnx.md#output-format), so take the argmax over the class axis in your own code.

The table below compares the raw outputs with those of the [ONNX](onnx.md) export of the same checkpoint on one image, `bus.jpg`. The FP32 column is the largest absolute difference of any output value, where box values are in pixels, and the INT8 column is the correlation of the first output with the ONNX output:

| Task     | FP32 against ONNX         | INT8 against ONNX                 |
| -------- | ------------------------- | --------------------------------- |
| Detect   | 0.0099                    | 0.9997                            |
| Segment  | 0.0054                    | 0.9998, prototypes 0.9852         |
| Semantic | Same class on every pixel | Same class on 98.3% of the pixels |
| Depth    | 0.0001                    | 0.5483                            |
| Classify | below 0.0001              | 0.9604, same top-1 class          |
| Pose     | 0.0034                    | 0.9996                            |
| OBB      | 0.0168                    | 0.9998                            |

INT8 changes the outputs, and how much depends on the task and on the calibration images. The depth model above was calibrated on the 4 images of `depth8.yaml` and differs strongly from ONNX on this street photo, while on the 4 validation images of `depth8.yaml`, the images it was calibrated on, it reached a delta1 of 0.85005 against 0.8535 for the FP32 model. Validate an INT8 model on your own data before you deploy it.

`YOLO("yolo26n_executorch_model")` reads the layout from `metadata.yaml`, so you only need these details when you run the `.pte` file in your own code. The keys of `metadata.yaml` include `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, `args`, and, for pose models, `kpt_shape`.

### Run with the ExecuTorch Runtime

To run the model outside the Ultralytics package, load `model.pte` with the runtime of the `executorch` Python package:

```python
# Export first: yolo export model=yolo26n.pt format=executorch
import torch
import yaml
from executorch.runtime import Runtime

program = Runtime.get().load_program("yolo26n_executorch_model/model.pte")
method = program.load_method("forward")

image = torch.zeros(1, 3, 640, 640)  # your image, letterboxed to 640x640: RGB, NCHW, float32, 0-1
output = method.execute([image])[0]  # raw predictions, shape (1, 84, 8400)
assert tuple(output.shape) == (1, 84, 8400)

with open("yolo26n_executorch_model/metadata.yaml") as f:
    names = yaml.safe_load(f)["names"]  # {0: 'person', 1: 'bicycle', ...}
```

## Deploy

`YOLO("yolo26n_executorch_model")` runs `model.pte` with the runtime of the `executorch` Python package, which Ultralytics installs on first use, on the CPU through XNNPACK. This works on the export host too, so you can check a model before you copy it to a device:

- **Fixed input shape**: the model always runs at its export `imgsz` and `batch`, which Ultralytics reads from `metadata.yaml`. An `imgsz` passed to `predict` or `val` is ignored; export again to change it. Smaller batches are padded to the export batch size, and by the Ultralytics code larger ones are split into chunks of it, so any number of images works.
- **`device`**: the model runs on the CPU. Leave `device` unset or pass `cpu`. On Apple silicon, `device="mps"` fails, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). By the Ultralytics code, a CUDA `device` such as `0` is replaced by the CPU for this format; this was not run for this page.
- **CPU threads**: the number of threads follows the CPUs the process may use. On a Linux server with 384 logical CPUs, a process that only ran the ExecuTorch runtime had 126 threads, and one FP32 YOLO26n inference took 20 to 21 ms. Limited to 8 CPUs with `taskset -c 0-7` it had 15 threads and took 8 ms, with 4 CPUs 10 to 11 ms, and with 1 CPU, on a single thread, 31 to 32 ms. On a machine with many cores, limit the CPUs of the process, on Linux with `taskset` or Docker `--cpuset-cpus`.
- **Directory name**: pass the directory, not the `.pte` file, and keep `_executorch_model` in its name, because Ultralytics recognizes the format by it, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).
- **One `.pte` file per directory**: Ultralytics loads whichever `.pte` file it finds first in the directory and its subdirectories, so keep only one there.
- **`metadata.yaml`**: Ultralytics reads the class names, the task, and the input size from it. Without it, a YOLO26n detection model exported at 640 still loaded, with the class names `class0`, `class1`, and so on; other tasks and sizes were not tried. Copy the whole directory to the device.

### Mobile Integration

In an Android or iOS application, add the ExecuTorch runtime library to the project, load `model.pte`, and pass it an image preprocessed as described in [Output Format](#output-format); the application code applies NMS to the raw output of detection, segmentation, pose, and OBB models. The examples in this section and in [Embedded Linux](#embedded-linux) were not run for this page.

#### iOS

The ExecuTorch runtime for iOS is a Swift package that requires iOS 17 or newer. The XNNPACK backend is a separate product of that package, `backend_xnnpack`, which the application links next to the `executorch` runtime with the `-all_load` linker flag, see [Using ExecuTorch on iOS](https://docs.pytorch.org/executorch/stable/using-executorch-ios.html). Example iOS integration (Objective-C/C++):

```objc
// iOS uses C++ APIs for model loading and inference
// See https://docs.pytorch.org/executorch/stable/using-executorch-ios.html for complete examples

#include <executorch/extension/module/module.h>

using namespace ::executorch::extension;

// Load the model
Module module("/path/to/model.pte");

// Create input tensor
float input[1 * 3 * 640 * 640];
auto tensor = from_blob(input, {1, 3, 640, 640});

// Run inference
const auto result = module.forward(tensor);
```

#### Android

Add the [ExecuTorch Android AAR](https://docs.pytorch.org/executorch/stable/using-executorch-android.html) from Maven Central, which ships `arm64-v8a` and `x86_64` libraries and includes the XNNPACK backend:

```kotlin
dependencies {
    implementation("org.pytorch:executorch-android:<version>")
}
```

Example Android integration (Kotlin):

```kotlin
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

// Load the model
val module = Module.load("/path/to/model.pte")

// Prepare input tensor
val inputTensor = Tensor.fromBlob(floatData, longArrayOf(1, 3, 640, 640))
val inputEValue = EValue.from(inputTensor)

// Run inference
val outputs = module.forward(inputEValue)
val predictions = outputs[0].toTensor().dataAsFloatArray
```

### Embedded Linux

On a board that runs Python, such as a Raspberry Pi with a 64-bit operating system, install the package with `pip install ultralytics` and load the directory with `YOLO()` as in the [Quickstart](#quickstart); the `executorch` packages for ARM Linux are 64-bit only, and this was not run on a board for this page. In a C++ application, use the ExecuTorch C++ API:

```cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using namespace ::executorch::extension;

// Load model
Module module("model.pte");

// Prepare input
std::vector<float> input_data = preprocessImage(image);
auto input_tensor = from_blob(input_data.data(), {1, 3, 640, 640});

// Run inference
const auto outputs = module.forward(input_tensor);
```

### Deployment Guides

- **[Using ExecuTorch on Android](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)** and **[on iOS](https://docs.pytorch.org/executorch/stable/using-executorch-ios.html)**: add the runtime to an application and call it from Kotlin, Java, Swift, or Objective-C.
- **[Using ExecuTorch with C++](https://docs.pytorch.org/executorch/stable/using-executorch-cpp.html)**: the `Module` API and the build targets for the runtime, the XNNPACK backend, and the thread pool.
- **[XNNPACK backend](https://docs.pytorch.org/executorch/stable/backends/xnnpack/xnnpack-overview.html)**: the CPU architectures and quantization schemes that the backend supports.
- **[Raspberry Pi](../guides/raspberry-pi.md)**: set up Ultralytics on Raspberry Pi boards.
- **[Ultralytics and ExecuTorch](https://www.ultralytics.com/blog/deploy-ultralytics-yolo-models-using-the-executorch-integration)**: an overview of the integration on the Ultralytics blog.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare ExecuTorch with the other export formats.

## Limitations and Troubleshooting

- **INT8 on Python 3.10 or with PyTorch older than 2.15**: the export fails with `INT8 export requires executorch>=1.6.0`. `executorch` 1.6.0 requires PyTorch 2.15.0 and Python 3.11 or newer, so pip installs an older `executorch` everywhere else: 1.5.1 on Python 3.10 and with PyTorch 2.13.0, and 1.3.1 with PyTorch 2.12.1. FP32 exports work in these environments, and a YOLO26n INT8 detection model exported with `executorch` 1.6.0 gave the same detections on one image with the 1.5.1 (PyTorch 2.13.0) and 1.3.1 (PyTorch 2.12.1) runtimes. To export INT8, use Python 3.11 or newer with PyTorch 2.15.0.
- **PyTorch older than 2.9**: the export fails with `ExecuTorch requires torch>=2.9.0 but torch==2.8.0+cu128 is installed`. This happens after a plain install on Python 3.9, where pip installs PyTorch 2.8.0. Use Python 3.10 or newer.
- **Upgrading `executorch` by hand**: Ultralytics installs the `executorch` version that matches the installed PyTorch, and below PyTorch 2.13 a version older than 1.5, because by the Ultralytics code newer runtimes fail there. Do not upgrade `executorch` alone.
- **Automatic installation turned off**: with `YOLO_AUTOINSTALL=false` and a plain `pip install ultralytics`, the export logs `Ultralytics requirement ['executorch'] not found, AutoUpdate disabled by YOLO_AUTOINSTALL=False` and fails with `No module named 'executorch'`. Install the package first with `pip install "ultralytics[export-executorch]"`; the export then needs no installation.
- **Passing a `.pte` file**: `YOLO("yolo26n_executorch_model/model.pte")` fails with `acceptable suffix is {'.pt'}, not .pte`. Pass the directory.
- **Output directory**: every export of one checkpoint writes the same `yolo26n_executorch_model/` directory, whatever its `imgsz`, `batch`, or `quantize`, and deletes the existing directory with everything in it first. Rename the directory before you export a second variant, and keep `_executorch_model` in its name, for example `yolo26n_int8_executorch_model/`: a copy named `yolo26n_et` failed with `is not a supported model format`.
- **`device="mps"`**: prediction on the Apple GPU fails with `Torch tensor is on device mps, and only CPU and CUDA tensors can be passed to portable bindings.` Leave `device` unset. The export itself ignores `device="mps"` and runs on the CPU.
- **NMS-free head and embedded NMS**: neither is available, and both `nms=False` and `nms=True` write the raw output, see [Output Format](#output-format). To get an NMS-free model for a phone, see [LiteRT](litert.md) or [Core ML](coreml.md).
- **Fixed input shape**: `dynamic=True` fails with `argument 'dynamic' is not supported for format='executorch'`. Export again to change the image size or the batch size.
- **RT-DETR in INT8**: a `quantize=8` export of RT-DETR-L failed with `Missing out variants: {'quantized_decomposed::dequantize_per_channel'}` in both of two runs. The FP32 export works.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='executorch' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends`. Export the checkpoint you started the fine-tuning from with `quantize=8` instead.
- **Windows and ARM64 Linux**: `executorch` has packages for both, but exports there were not run for this page, and the Ultralytics tests skip the ExecuTorch export on Windows.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

The Ultralytics team benchmarked YOLO26 models on a Raspberry Pi 5, comparing speed and accuracy between PyTorch and ExecuTorch.

!!! tip "Performance"

    === "Raspberry Pi 5"

        | Model   | Format     | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
        | ------- | ---------- | ------ | --------- | ------------------- | ---------------------- |
        | YOLO26n | PyTorch    | ✅     | 5.3       | 0.4790              | 314.80                 |
        | YOLO26n | ExecuTorch | ✅     | 9.4       | 0.4800              | 142                    |
        | YOLO26s | PyTorch    | ✅     | 19.5      | 0.5730              | 930.90                 |
        | YOLO26s | ExecuTorch | ✅     | 36.5      | 0.5780              | 376.1                  |

    Benchmarked with Ultralytics 8.4.9

    !!! note

        Inference time does not include pre/post-processing.

This table was recorded with that earlier version and was not measured again for this page; its validation dataset and image size are not recorded. Its Size column is the whole export directory in MiB (2^20 bytes); a current YOLO26n export is also 9.4 in that unit, and 9.9 MB in the Size table below. The [Raspberry Pi guide](../guides/raspberry-pi.md#raspberry-pi-5-yolo26-benchmarks) compares ExecuTorch with seven other formats on the same board.

### FP32 and INT8 on a Server CPU

The tables below compare PyTorch with FP32 and INT8 ExecuTorch models of every YOLO26 detection model on the CPU of the export host, an **AMD EPYC 9655** server shared with other jobs. For the latency table, every process was limited to 8 CPUs with `taskset -c 0-7`, the thread count that Ultralytics gives PyTorch, because the ExecuTorch runtime is slower with all 384 logical CPUs of this server, see [Deploy](#deploy). The results apply to this setup.

=== "Latency"

    Milliseconds per image, lower is better, as the mean of two runs. The fastest format for each model is in bold.

    | Model   | `imgsz` | PyTorch | ExecuTorch FP32 | ExecuTorch INT8 |
    | ------- | ------- | ------- | --------------- | --------------- |
    | YOLO26n | 640     | 9.7     | 10.4            | **8.6**         |
    | YOLO26s | 640     | 22.7    | 24.2            | **16.4**        |
    | YOLO26m | 640     | 53.8    | 61.1            | **33.8**        |
    | YOLO26l | 640     | 63.9    | 73.1            | **38.3**        |
    | YOLO26x | 640     | 117.2   | 150.7           | **68.0**        |

=== "Accuracy"

    mAP<sup>val</sup> 50-95 on 500 images of COCO val2017, higher is better.

    | Model   | PyTorch | ONNX FP32 | ExecuTorch FP32 | ExecuTorch INT8 |
    | ------- | ------- | --------- | --------------- | --------------- |
    | YOLO26n | 42.1    | 42.0      | 42.0            | 41.3            |
    | YOLO26s | 48.9    | 49.4      | 49.4            | 48.6            |
    | YOLO26m | 53.0    | 53.4      | 53.4            | 53.2            |
    | YOLO26l | 54.9    | 55.4      | 55.4            | 54.9            |
    | YOLO26x | 55.0    | 55.6      | 55.6            | 55.4            |

=== "Size"

    Size of the `.pt` checkpoint and of `model.pte` in MB (10^6 bytes). The smallest file for each model is in bold.

    | Model   | PyTorch | ExecuTorch FP32 | ExecuTorch INT8 |
    | ------- | ------- | --------------- | --------------- |
    | YOLO26n | 5.5     | 9.9             | **3.2**         |
    | YOLO26s | 20.4    | 38.2            | **11.4**        |
    | YOLO26m | 44.3    | 81.9            | **25.3**        |
    | YOLO26l | 53.2    | 99.5            | **29.9**        |
    | YOLO26x | 118.7   | 223.2           | **66.2**        |

- **INT8** was the fastest format for every model with 8 CPUs, 1.2 to 2.2 times faster than FP32 ExecuTorch, with a file 3.0 to 3.4 times smaller, and lost 0.2 to 0.8 mAP.
- **FP32** reproduced the accuracy of the ONNX export and was 7% to 29% slower than PyTorch.
- **All CPUs**: without the limit, on all 384 logical CPUs, YOLO26n took 19.2 ms in FP32 and 22.6 ms in INT8, in one run each, and INT8 was slower than FP32 for every model except YOLO26x.

Measured with `ultralytics` 8.4.215, `executorch` 1.6.0, PyTorch 2.15.0, and ONNX Runtime 1.32.0 at `batch=1`. Accuracy is on the last 500 images of COCO val2017 in sorted order, so it differs from the values for the full validation set on the [YOLO26 model page](../models/yolo26.md); the INT8 models were calibrated on the first 500 with `data=coco.yaml fraction=0.1`. Latency is the inference stage that `val` reports and does not include preprocessing or postprocessing; the two runs differed by up to 8%.

## About ExecuTorch

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/11dc46f6702048f937a7b6f39f53f05a.avif" alt="PyTorch ExecuTorch mobile inference framework">
</p>

[ExecuTorch](https://docs.pytorch.org/executorch/) is the on-device inference runtime of the PyTorch project. A model is captured with `torch.export`, lowered to a backend for the target hardware, and saved as a `.pte` program that a small C++ runtime executes. The Ultralytics export lowers models to XNNPACK, the ExecuTorch backend for ARM and x86 CPUs.

## FAQ

### Does an ExecuTorch model from Ultralytics use the GPU or NPU of a phone?

No. The export lowers the model to XNNPACK only, which runs on the CPU. ExecuTorch has other backends, such as Core ML, Vulkan, and Qualcomm AI Engine, but a model has to be lowered to them at export, and Ultralytics does not do that. To use the GPU or NPU with an Ultralytics export, see [Core ML](coreml.md) for Apple devices, [Qualcomm QNN](qnn.md) for Snapdragon NPUs, and [LiteRT](litert.md) for Android GPUs.

### Should I use ExecuTorch or LiteRT for a mobile app?

Both formats run on phone CPUs through XNNPACK. The [LiteRT](litert.md) export also offers a quantization that needs no calibration data (`w8a32`), the NMS-free head of YOLO26, GPU delegates, and the [Ultralytics Flutter plugin](https://github.com/ultralytics/yolo-flutter-app), which runs its models on Android. ExecuTorch fits an application that already ships the ExecuTorch runtime, for example for other PyTorch models; its export writes the raw output, so the application applies NMS itself.

### How does ExecuTorch relate to TorchScript?

ExecuTorch is the current PyTorch route to mobile and embedded devices: it captures the model with `torch.export` and runs a `.pte` file. [TorchScript](torchscript.md) is the earlier format, which PyTorch has deprecated; Ultralytics keeps its export for existing C++ deployments.
