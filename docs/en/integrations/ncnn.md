---
comments: true
description: Export Ultralytics YOLO26 models to Tencent NCNN with format=ncnn. Covers FP16 weights, output files and shapes, Vulkan GPU inference, CPU threads, and mobile and embedded deployment.
keywords: YOLO26, NCNN, Tencent NCNN, NCNN export, PNNX, Vulkan, FP16, ARM, Raspberry Pi, Android, mobile inference, embedded devices, edge AI, model deployment
---

# Tencent NCNN Export

[NCNN](https://github.com/Tencent/ncnn) is Tencent's open-source [neural network](https://www.ultralytics.com/glossary/neural-network-nn) inference framework for phones, embedded boards, and desktop computers. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="ncnn"` creates a `yolo26n_ncnn_model/` directory that holds the network as a `.param` and `.bin` file pair, with FP32 or FP16 weights.

## At a Glance

| Property          | Details                                                                                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `ncnn`                                                                                                                                                      |
| Output            | `yolo26n_ncnn_model/`, a directory with `model.ncnn.param`, `model.ncnn.bin`, `model_ncnn.py`, and `metadata.yaml`                                          |
| Runs on           | ARM and x86-64 CPUs, and GPUs with a [Vulkan](#vulkan-gpu-inference) driver: phones, single-board computers such as the Raspberry Pi, and desktop computers |
| Runtime           | NCNN: the `ncnn` Python package, or the NCNN C++ library in Android, iOS, and embedded applications                                                         |
| Precision         | FP32 or FP16 weights; there is no INT8 export. YOLO26n is 9.8 MB in FP32 and 5.0 MB with FP16 weights                                                       |
| Export host       | Linux x86-64 or ARM64, macOS, or Windows x64, the platforms that `pnnx` has builds for. The export needs no GPU                                             |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to NCNN for you, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package:

```bash
pip install ultralytics
```

The first export installs the `ncnn` and [`pnnx`](https://github.com/pnnx/pnnx) packages if they are missing, and the first prediction with an exported model installs `ncnn` if it is missing. A fresh environment needs nothing else: the first export completes in the same run, after it logs `Restart runtime or rerun command for updates to take effect`. The export has these version requirements:

| Package | Required version                                                                                                                       |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `pnnx`  | 20260526 exactly. The export installs this version and replaces any other one                                                          |
| `ncnn`  | Any version                                                                                                                            |
| `torch` | 2.0 or newer: the export does not check the version, and the Ultralytics tests skip NCNN on older ones because inference crashed there |

For other installation methods, see the [installation guide](../quickstart.md). Then export your model and load the exported directory to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to NCNN format
        model.export(format="ncnn")  # creates 'yolo26n_ncnn_model/'

        # Or export with FP16 weights instead, which halves the file size
        model.export(format="ncnn", quantize=16)  # creates 'yolo26n_ncnn_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to NCNN format
        yolo export model=yolo26n.pt format=ncnn # creates 'yolo26n_ncnn_model/'

        # Or export with FP16 weights instead, which halves the file size
        yolo export model=yolo26n.pt format=ncnn quantize=16 # creates 'yolo26n_ncnn_model/'
        ```

Both exports write the same `yolo26n_ncnn_model/` directory, so the second one replaces the first.

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported NCNN model
        model = YOLO("yolo26n_ncnn_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported NCNN model
        yolo predict model=yolo26n_ncnn_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported NCNN model
        model = YOLO("yolo26n_ncnn_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported NCNN model
        yolo val model=yolo26n_ncnn_model data=coco8.yaml
        ```

## Supported Tasks

NCNN export supports all seven Ultralytics tasks with [YOLOv8](../models/yolov8.md), [YOLO11](../models/yolo11.md), and [YOLO26](../models/yolo26.md) models: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads. Detection models of YOLOv5u, YOLOv9, YOLOv10, and YOLO12 export too. RT-DETR, YOLO-World, and text-prompt YOLOE models do not convert, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

## Export Arguments

| Argument   | Type             | Default  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------- | ---------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'ncnn'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                            |
| `imgsz`    | `int` or `tuple` | `640`    | Image size of the model input, fixed in the exported model: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect, segment, and pose models and for semantic models pretrained on ADE20K, `1024` for OBB models and for semantic models pretrained on Cityscapes, `768` for depth, and `224` for classify. |
| `quantize` | `int` or `str`   | `None`   | Precision of the stored weights: `16` (FP16) halves the file size; `32`/unset is FP32. `8` raises an error, and a checkpoint trained with `quantize=8` cannot be exported to NCNN, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). Replaces the deprecated `half` flag.                                                                                                                                              |
| `nms`      | `bool`, optional | `None`   | NCNN models always return raw predictions for external NMS. The NMS-free head (`False`) and embedded NMS (`True`) are not available, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).                                                                                                                                                                                                                                 |
| `device`   | `str`            | `None`   | Device that traces the model for the conversion: `cpu` (the default, also on a machine with a GPU), `0` for a CUDA GPU, or `mps` for Apple silicon.                                                                                                                                                                                                                                                                                           |

Of the format-specific export arguments, NCNN accepts only these. For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Precision

`quantize` selects how the weights are stored in `model.ncnn.bin`. The sizes are for a YOLO26n detection model exported at `imgsz=640`:

| `quantize`    | Weights | YOLO26n size |
| ------------- | ------- | ------------ |
| unset or `32` | FP32    | 9.8 MB       |
| `16`          | FP16    | 5.0 MB       |

- **FP16 weights**: `quantize=16` stores the weights in [half precision](https://www.ultralytics.com/glossary/half-precision) and needs no calibration data. The input and the outputs stay `float32`, so inference code written for the FP32 model runs it unchanged.
- **Accuracy**: on COCO val2017, on an x86-64 server CPU, the FP32 and FP16 NCNN models of YOLO26n, the ONNX model, and the PyTorch model (`rect=False`) all reached 40.9 mAP<sup>val</sup> 50-95 at the same 640x640 input.
- **FP16 computation**: NCNN can also compute in half precision, whatever the precision of the stored weights. On an Apple M4 it did so for both exports by default: on one test image, the raw box coordinates of the FP32 export differed from the [ONNX](onnx.md) model by about 2.1 pixels at most for anchors above 0.25 confidence, and the whole output matched it within 0.01 with `net.opt.use_fp16_storage = False` set in the [ncnn package](#run-with-the-ncnn-package). On an x86-64 server CPU, the raw outputs of the FP32 export differed from the ONNX model by less than 0.03 on the same image for detect, segment, depth, classify, pose, and OBB models. With FP16 weights, the box coordinates of those anchors differed from ONNX by about 0.3 pixels on the server and 1.6 pixels on the M4. On the M4 these differences changed the accuracy by less than 0.1: on COCO128, the Ultralytics validator gave 47.7 mAP 50-95 for the FP32 export and 47.7 with FP16 weights, against 47.7 for the PyTorch model (`rect=False`) at the same 640x640 input.

## Output Format

The export writes one directory:

| File               | Contents                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `model.ncnn.param` | The network graph as a text file                                                                                         |
| `model.ncnn.bin`   | The weights                                                                                                              |
| `model_ncnn.py`    | A Python script written by PNNX that runs the model on a random input with the `ncnn` package                            |
| `metadata.yaml`    | Class names, task, image size, batch size, stride, and export arguments, which Ultralytics reads when it loads the model |

The export also leaves a `__pycache__` folder in the directory, which you can delete. The model takes one input named `in0` with shape `(3, height, width)`: one RGB image, channels first, `float32`, scaled to 0-1. There is no batch dimension, and the outputs have none either. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | Outputs                                                     |
| -------- | --------------- | ----------------------------------------------------------- |
| Detect   | `640`           | `out0` `(84, 8400)`                                         |
| Segment  | `640`           | `out0` `(116, 8400)` and `out1` `(32, 160, 160)`            |
| Semantic | `1024`          | `out0` `(19, 1024, 1024)`, a score for each class and pixel |
| Depth    | `768`           | `out0` `(1, 768, 768)`, depth map                           |
| Classify | `224`           | `out0` `(1000,)`, class probabilities                       |
| Pose     | `640`           | `out0` `(56, 8400)`                                         |
| OBB      | `1024`          | `out0` `(20, 21504)`                                        |

- **Raw predictions**: detection, segmentation, pose, and OBB models return the same raw predictions as a default [ONNX export](onnx.md#output-format), without the batch dimension: for each anchor the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS yourself.
- **Segmentation**: `out0` holds the predictions and `out1` the mask prototypes.
- **Semantic segmentation**: the output holds a score for each class and pixel, not the class map of the ONNX export, so take the argmax over the class axis in your own code.
- **NMS**: there is no NMS-free or embedded-NMS output, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

`YOLO("yolo26n_ncnn_model")` reads the layout from `metadata.yaml`, adds the batch dimension, and applies NMS, so you only need these details when you run the model in your own code. The keys of `metadata.yaml` include `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, `args`, and, for pose models, `kpt_shape`.

### Run with the ncnn Package

To run the model outside the Ultralytics package, load the two files with the `ncnn` Python package. The same `.param` and `.bin` files load in the NCNN C++ library on Android, iOS, and embedded Linux.

```python
# Export first: yolo export model=yolo26n.pt format=ncnn
import ncnn
import numpy as np

net = ncnn.Net()
net.opt.num_threads = 4  # set options before the model loads
assert net.load_param("yolo26n_ncnn_model/model.ncnn.param") == 0
assert net.load_model("yolo26n_ncnn_model/model.ncnn.bin") == 0

image = np.zeros((3, 640, 640), dtype=np.float32)  # your letterboxed image: RGB, CHW, 0-1
with net.create_extractor() as ex:
    ex.input("in0", ncnn.Mat(image).clone())
    _, out0 = ex.extract("out0")
predictions = np.array(out0)  # shape (84, 8400), raw predictions that need NMS
assert predictions.shape == (84, 8400), "export a detection model at imgsz=640"
```

`ncnn.Mat(image)` reads the memory of the NumPy array without copying it, so pass a C-contiguous `float32` array and keep it alive until the extraction is done, or copy it with `.clone()` as above. An array made with `transpose` is not contiguous, and `astype` or a division after it keeps that layout, so finish the preprocessing with `np.ascontiguousarray(x, dtype=np.float32)`. In a test with this model, a non-contiguous array ran without an error and returned wrong predictions, and a `float64` array crashed Python.

## Deploy

On the target device, install the package with `pip install ultralytics` and copy the `yolo26n_ncnn_model/` directory to it. `YOLO("yolo26n_ncnn_model")` then runs the model with the `ncnn` Python package, which Ultralytics installs on first use. The `device` argument selects where NCNN runs:

| `device`                         | Where the model runs                                                                                                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| unset or `cpu`                   | NCNN on the CPU                                                                                                                                                                                   |
| `vulkan:0`, `vulkan:1`, ...      | NCNN on the Vulkan device with that index, see [Vulkan GPU Inference](#vulkan-gpu-inference)                                                                                                      |
| `0` on a machine with a CUDA GPU | NCNN on the CPU: NCNN has no CUDA backend                                                                                                                                                         |
| `mps` on Apple silicon           | NCNN on the CPU, while Ultralytics preprocesses and postprocesses on the Apple GPU. On an M4 this added about 0.6 s to the first image and did not make inference faster, so leave `device` unset |

- **Fixed input shape**: the model always runs at its export `imgsz`, which Ultralytics reads from `metadata.yaml`. An `imgsz` passed to `predict` or `val` is ignored. Export again to change the image size.
- **Batches**: the model takes one image, and Ultralytics runs the images of a batch one after another, so any number of images works. `val` sets `batch=1` whatever value you pass.
- **CPU threads**: NCNN chooses the thread count itself: 4 on an Apple M4, the number of its performance cores. On Linux, Ultralytics limits that count to 8 and to the CPUs the process may use: its CPU affinity (`taskset`, Docker `--cpuset-cpus`) and a cgroup v2 CPU quota (Docker `--cpus`). See [Limitations and Troubleshooting](#limitations-and-troubleshooting) to use a different count.

### Vulkan GPU Inference

NCNN runs on GPUs through [Vulkan](https://www.vulkan.org/), which works with GPU vendors such as AMD, Intel, ARM, Qualcomm, and Apple, without CUDA. Pass `device="vulkan:0"` for the first Vulkan device, or `vulkan:1` for the second one:

!!! example "Vulkan inference"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported NCNN model
        model = YOLO("yolo26n_ncnn_model")

        # Run inference on the first Vulkan device
        results = model("https://ultralytics.com/images/bus.jpg", device="vulkan:0")
        ```

    === "CLI"

        ```bash
        # Run inference on the first Vulkan device
        yolo predict model=yolo26n_ncnn_model source='https://ultralytics.com/images/bus.jpg' device=vulkan:0
        ```

The GPU needs a Vulkan driver, which most current GPU drivers include; on Linux, `vulkaninfo` lists the installed ones. The index counts the Vulkan devices that NCNN finds, which need not be hardware GPUs. Print them before you rely on the result:

```python
import ncnn

for i in range(ncnn.get_gpu_count()):
    print(i, ncnn.get_gpu_info(i).device_name())
```

Vulkan is not always faster than the CPU. On an Apple M4, YOLO26n took about 31 ms per image on the Vulkan device and about 12 ms on the CPU. Measure both on your device.

### Deployment Guides

- **[Raspberry Pi](../guides/raspberry-pi.md)**: set up Ultralytics on Raspberry Pi boards, where the guide exports to NCNN on the board itself.
- **[NVIDIA Jetson](../guides/nvidia-jetson.md)**: NCNN on the ARM CPU of Jetson devices, next to TensorRT on their GPU.
- **[NCNN build instructions](https://github.com/Tencent/ncnn/wiki/how-to-build)**: build the NCNN C++ library for Android, iOS, Linux, macOS, Windows, and other targets.
- **[NCNN on GitHub](https://github.com/Tencent/ncnn)**: the source code, the Python package, and C++ examples.
- **[Interactive tracking UI example](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLO-Interactive-Tracking-UI)**: an Ultralytics example that runs NCNN models on CPU-only devices.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare NCNN with the other export formats.

## Limitations and Troubleshooting

- **INT8**: `quantize=8` fails with `quantize=8 (INT8) is not supported; format='ncnn' supports quantize=16 (FP16), 32 or None (FP32).` Use FP16 to reduce the file size, or a format with INT8 export such as [LiteRT](litert.md) or [MNN](mnn.md).
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='ncnn' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends`. Export the checkpoint you started the fine-tuning from instead, or export the fine-tuned checkpoint to [ONNX](onnx.md) or [TensorRT](tensorrt.md).
- **NMS-free head**: NCNN models always return raw predictions that need NMS. With `nms=False`, the export of a YOLO26 or YOLOv10 model logs `This export format does not support end2end models, disabling the end2end branch.` and uses the one-to-many head; for other detection, segmentation, pose, and OBB models, for example YOLO11, it logs `This model has no one-to-one head; using one-to-many outputs.`
- **Embedded NMS**: `nms=True` logs `format=ncnn does not support embedded NMS; exporting native outputs for external NMS.` and writes the raw output.
- **Fixed input shape and batch size**: `dynamic=True` fails with `argument 'dynamic' is not supported for format='ncnn'`, and a `batch` other than 1 fails with `argument 'batch' is not supported for format='ncnn'`. Export again to change the image size.
- **Models that do not convert**: the export of [RT-DETR](../models/rtdetr.md) (`rtdetr-l.pt`), [YOLO-Worldv2](../models/yolo-world.md) (`yolov8s-worldv2.pt`), and text-prompt [YOLOE](../models/yoloe.md) (`yoloe-26s-seg.pt`) models fails with `NCNN failed to load the exported model, it may contain layers NCNN does not support.` A text-prompt YOLOE model exports after you fix its classes with `set_classes()`, and the prompt-free `yoloe-26s-seg-pf.pt` model exports as it is.
- **PNNX version**: the export requires `pnnx` 20260526 and replaces a newer version, because models converted with PNNX 20260704 [crash at inference](https://github.com/pnnx/pnnx/issues/293).
- **Automatic installation turned off**: with `YOLO_AUTOINSTALL=false`, the export fails with `No module named 'ncnn'` in an environment without the packages. Install them with `pip install --no-deps ncnn` and `pip install pnnx==20260526`.
- **Thread count**: `predict` and `val` have no argument for the NCNN thread count, and the `OMP_NUM_THREADS` variable does not change it. On Linux, Ultralytics uses at most 8 threads, and larger models can run faster with more on machines with more cores: run with the ncnn package on 16 cores of an AMD EPYC 9655, YOLO26x took about 177 ms per image with 8 threads and about 154 ms with 16, and YOLO26n took about 23 ms with either. To choose the thread count, run the model with the [ncnn package](#run-with-the-ncnn-package) and set `net.opt.num_threads` before it loads. Changing `num_threads` after the model had loaded made inference slower.
- **Incomplete model directory**: when `model.ncnn.bin` is missing or cut short, for example after an interrupted copy to the device, Python stops with a segmentation fault and no traceback. The last lines of the output name the cause, `fopen yolo26n_ncnn_model/model.ncnn.bin failed` or `ModelBin read weight_data failed`. A directory without `model.ncnn.param` fails with `StopIteration`. Copy the directory again.
- **`device="vulkan"` without an index**: fails with `IndexError: list index out of range`. Write the index, as in `device="vulkan:0"`.
- **Vulkan device is not a GPU**: on the Linux server used for this page, the only Vulkan device that NCNN listed was `llvmpipe`, a software renderer that runs on the CPU. `device="vulkan:0"` then runs without an error and without a GPU. An index that does not exist, such as `vulkan:9` on a machine with one device, also runs without an error.
- **Output directory**: every export of one checkpoint writes the same `yolo26n_ncnn_model/` directory, whatever its `quantize` or `imgsz`, so the later export replaces the earlier one. Rename the directory to keep both, and keep `_ncnn_model` in its name, for example `yolo26n_fp16_ncnn_model/`, because Ultralytics recognizes the format by it. A directory without it, or the path of the `.param` file, fails with `is not a supported model format`.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

Ultralytics publishes NCNN benchmarks for two ARM platforms. The [Raspberry Pi 5 benchmarks](../guides/raspberry-pi.md#raspberry-pi-5-yolo26-benchmarks) compare NCNN with seven other formats for YOLO26n and YOLO26s, and the [NVIDIA Jetson benchmarks](../guides/nvidia-jetson.md#nvidia-jetson-yolo26-benchmarks) include NCNN for all five YOLO26 detection models. The [YOLO26 model page](../models/yolo26.md) lists the accuracy of the PyTorch checkpoints.

The sizes of the exported YOLO26 detection models, as the sum of `model.ncnn.param` and `model.ncnn.bin` in MB (10^6 bytes) at `imgsz=640`. The export log prints the size of the whole directory in MiB, 9.4 MB and 4.8 MB for YOLO26n:

| Model   | PyTorch | NCNN FP32 | NCNN FP16 |
| ------- | ------- | --------- | --------- |
| YOLO26n | 5.5     | 9.8       | 5.0       |
| YOLO26s | 20.4    | 38.1      | 19.2      |
| YOLO26m | 44.3    | 81.8      | 41.0      |
| YOLO26l | 53.2    | 99.4      | 49.8      |
| YOLO26x | 118.7   | 223.0     | 111.7     |

Exported with `ultralytics` 8.4.215, NCNN 1.0.20261020, and PNNX 20260526 on November 16, 2026.

## About NCNN

NCNN is developed by [Tencent](https://www.tencent.com/) and used in Tencent applications such as QQ and WeChat. It has no third-party runtime dependencies, runs on CPUs and on GPUs through Vulkan, and converts PyTorch models with its PNNX tool, which the Ultralytics export uses. For the full reference, see the [NCNN documentation](https://ncnn.readthedocs.io/en/latest/index.html).

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/5b5b6ed9bafb015dab714e5b4b3111d7.avif" alt="NCNN high-performance neural network inference framework">
</p>

## FAQ

### Should I export an NCNN model with FP32 or FP16 weights?

Use FP16 (`quantize=16`) when the size of the model matters: YOLO26n is 5.0 MB with FP16 weights against 9.8 MB in FP32, and it needs no calibration data. On COCO val2017, YOLO26n reached the same 40.9 mAP<sup>val</sup> 50-95 with FP16 weights as in FP32. FP16 weights make the file smaller, not the model faster: NCNN chooses the precision it computes in from the hardware, see [Precision](#precision). Validate the exported model on your own data and device before you deploy it.

### Can I export to NCNN on a Raspberry Pi?

The `pnnx` and `ncnn` packages have builds for 64-bit ARM Linux, and the [Raspberry Pi guide](../guides/raspberry-pi.md#use-ncnn-on-raspberry-pi) exports on the board with the same `format="ncnn"` command. You can also export on a PC and copy the `yolo26n_ncnn_model/` directory to the board. On a 32-bit ARM system `pnnx` has no build, so export on another machine; `ncnn` has builds for it, but Ultralytics was not tested there.

### How do I run a YOLO26 NCNN model in an Android or iOS app?

Build or download the NCNN library for your platform as described in the [NCNN build instructions](https://github.com/Tencent/ncnn/wiki/how-to-build), add `model.ncnn.param` and `model.ncnn.bin` to the app, and feed the `in0` input as described in [Output Format](#output-format). Your code has to letterbox the image, decode the raw predictions, and apply NMS. If you prefer a ready-made app component, the [Ultralytics YOLO Flutter plugin](https://github.com/ultralytics/yolo-flutter-app) runs [LiteRT](litert.md) exports on Android and [Core ML](coreml.md) exports on iOS.

### Why is my NCNN model slower than PyTorch on a desktop or server CPU?

NCNN is built for ARM CPUs in phones and embedded boards. On a server with two 96-core AMD EPYC 9655 processors, YOLO26n took about 30 ms per image with NCNN on 8 threads, against about 13 ms for both PyTorch and ONNX Runtime, and on an Apple M4, where NCNN chooses 4 threads, about 12 ms. For x86-64 servers, [OpenVINO](openvino.md) and [ONNX](onnx.md) are the formats Ultralytics benchmarks.
