---
comments: true
description: Export Ultralytics YOLO26 models to NVIDIA TensorRT engines with format=engine. Covers FP16 and INT8 quantization, dynamic shapes, Jetson DLA, and benchmarks.
keywords: YOLO26, TensorRT, TensorRT export, engine file, NVIDIA GPU, FP16, INT8 quantization, INT8 calibration, dynamic batch, NVIDIA Jetson, DLA, TensorRT 11, Triton
---

# TensorRT Export for YOLO26 Models

[TensorRT](https://developer.nvidia.com/tensorrt) is NVIDIA's SDK for high-speed [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) inference. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="engine"` builds a `.engine` file that is optimized for the NVIDIA GPU it is built on, in FP32, FP16, or INT8 precision.

!!! warning "TensorRT engines are hardware- and runtime-specific"

    TensorRT tunes an engine for the GPU it is built on. Build the engine on the deployment GPU, or on one of the same model, with the same TensorRT version; an `.engine` file is not a portable model format. For edge deployment, [Ultralytics Platform](../platform/train/models.md#nvidia-jetson-tensorrt-targets) offers eight Jetson target selections, with physical build and validation status documented for each, or you can export locally on the destination device.

## At a Glance

| Property          | Details                                                                                                                                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `engine` (aliases: `tensorrt`, `trt`)                                                                                                                                                        |
| Output            | `yolo26n.engine`, one file that starts with a [JSON metadata header](#load-the-engine-outside-ultralytics) (class names, task, image size). The intermediate `yolo26n.onnx` stays next to it |
| Runs on           | NVIDIA GPUs: data center, workstation, and [Jetson](../guides/nvidia-jetson.md), including Jetson DLA cores (not on TensorRT 11)                                                             |
| Runtime           | TensorRT 7.0 or newer (10.2.0 is not supported)                                                                                                                                              |
| Precision         | FP32, FP16, INT8                                                                                                                                                                             |
| Export host       | A machine with an NVIDIA GPU and CUDA. Build on the deployment GPU or one of the same model                                                                                                  |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can build a TensorRT engine for you on the GPU type you select, including [NVIDIA Jetson targets](../platform/train/models.md#nvidia-jetson-tensorrt-targets), so you do not need a local export environment.

## Quickstart

Install the Ultralytics package on a machine with an NVIDIA GPU and CUDA. On Linux, the first export installs the TensorRT Python package that matches your CUDA version if it is missing. On NVIDIA Jetson, use the TensorRT version that ships with JetPack. On Windows, install TensorRT yourself before you export.

```bash
pip install ultralytics
```

For other installation methods, see the [installation guide](../quickstart.md). Then [export your model](https://academy.ultralytics.com/courses/yolo-in-production/optimize-with-tensorrt) and load the exported engine to run inference or validate its accuracy. Inference and validation require an NVIDIA GPU. Building a YOLO26n engine takes one to three minutes on an NVIDIA T4.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to TensorRT format
        model.export(format="engine")  # creates 'yolo26n.engine'

        # Or export an FP16 engine instead
        model.export(format="engine", quantize=16)  # creates 'yolo26n.engine'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to TensorRT format
        yolo export model=yolo26n.pt format=engine # creates 'yolo26n.engine'

        # Or export an FP16 engine instead
        yolo export model=yolo26n.pt format=engine quantize=16 # creates 'yolo26n.engine'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported TensorRT model
        model = YOLO("yolo26n.engine")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported TensorRT model
        yolo predict model=yolo26n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported TensorRT model
        model = YOLO("yolo26n.engine")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported TensorRT model
        yolo val model=yolo26n.engine data=coco8.yaml
        ```

## Supported Tasks

TensorRT export supports all seven Ultralytics tasks with [YOLOv8](../models/yolov8.md), [YOLO11](../models/yolo11.md), and [YOLO26](../models/yolo26.md) models: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

## Export Arguments

| Argument    | Type                      | Default    | Description                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------- | ------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`    | `str`                     | `'engine'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                    |
| `imgsz`     | `int` or `tuple`          | `640`      | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                                                                                                                                                                                                     |
| `quantize`  | `int` or `str`            | `None`     | Quantization precision: `16` (FP16) or `8` (INT8/PTQ; needs calibration `data`/`fraction`); `32`/unset is FP32. A checkpoint trained with `quantize=8` exports INT8 from the ranges it carries, without calibration, also when `quantize` is unset; `quantize=16` or `quantize=32` raises an error. Replaces the deprecated `half`/`int8` flags.                                                      |
| `dynamic`   | `bool`                    | `False`    | Builds an engine that accepts batch sizes from 1 to `batch` and image sizes from 32 pixels to twice `imgsz`; the minimum is higher for [dynamic INT8 engines on older GPUs](#limitations-and-troubleshooting). When `False`, the engine accepts only the export `batch` and `imgsz`.                                                                                                                  |
| `simplify`  | `bool`                    | `True`     | Simplifies the model graph with `onnxslim`, potentially improving performance and compatibility.                                                                                                                                                                                                                                                                                                      |
| `opset`     | `int`                     | `None`     | Specifies the ONNX opset version for the intermediate ONNX graph. If not set, uses opset `18` (lower on PyTorch < 2.4), capped at the installed `onnx` version.                                                                                                                                                                                                                                       |
| `workspace` | `float` or `None`         | `None`     | Upper limit in GiB for the memory TensorRT uses while it builds the engine. Leave it at `None` to let TensorRT allocate memory automatically.                                                                                                                                                                                                                                                         |
| `nms`       | `bool`, optional          | `None`     | Select raw output (`None`, default), embedded NMS (`True`), or the NMS-free head (`False`). With `nms=True`, `conf` (default `0.25`), `iou`, `max_det`, and `agnostic_nms` are fixed at export; `nms=True` is ignored for semantic and depth models and raises an error for classify models.                                                                                                          |
| `batch`     | `int`                     | `1`        | Batch size of the engine, or the maximum batch size with `dynamic=True`. Also the batch size used for INT8 calibration.                                                                                                                                                                                                                                                                               |
| `data`      | `str`                     | `None`     | Path to the [dataset](../datasets/index.md) YAML, used for INT8 calibration; classification instead takes a dataset directory or a built-in dataset name. Calibration uses the `val` split unless `split` is set. If omitted with `quantize=8`, Ultralytics uses the small default dataset for the model task, such as `coco8.yaml` for detection; a checkpoint trained with `quantize=8` needs none. |
| `fraction`  | `float`, `int`, or `list` | `1.0`      | Calibration subset as a ratio, image count, or `[train, val, test]` ratios/counts. Two-item lists leave `test` full, while `0` skips it.                                                                                                                                                                                                                                                              |
| `device`    | `str`                     | `None`     | Specifies the device for exporting: GPU (`device=0`, used when unset), DLA for NVIDIA Jetson (`device=dla:0` or `device=dla:1`, with `quantize=16` or `quantize=8`).                                                                                                                                                                                                                                  |

!!! warning "JetPack and DLA compatibility"

    On NVIDIA Jetson, Ultralytics exports with the TensorRT runtime of your JetPack release. On JetPack 7 the export requires TensorRT 10.15 or newer and below 11. `device=dla:0` and `device=dla:1` raise an error on TensorRT 11. NVIDIA lists TensorRT 10.7 as the last 10.x release with [DLA support](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html), and [TensorRT 11.4.1 adds JetPack 7.2.2 support](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-jetson.html), which Ultralytics does not use yet.

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Exporting TensorRT with INT8 Quantization

`quantize=8` builds an INT8 engine with post-training [quantization](https://www.ultralytics.com/glossary/model-quantization) (PTQ): the export runs calibration images through the model to measure the value range of each tensor. A checkpoint fine-tuned with `quantize=8` through [quantization-aware training](../modes/export.md#quantization-aware-training) already carries its INT8 ranges, so its export skips calibration.

The `quantize` and `data` arguments are the same on every TensorRT version, but the export of a checkpoint not trained with `quantize=8` works differently underneath:

|                                  | TensorRT 7-10                                                | TensorRT 11                                                                                                                |
| -------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| INT8 method                      | Implicit quantization with the `IInt8Calibrator` interface   | Explicit quantization: [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer) inserts Q/DQ nodes into the ONNX graph |
| Calibration algorithm            | `MINMAX_CALIBRATION` on GPU, `ENTROPY_CALIBRATION_2` for DLA | ModelOpt `max`, run on the CPU                                                                                             |
| Calibration images used          | Every full batch of the split                                | About 512 images in shuffled order, or every full batch of a smaller split                                                 |
| `dynamic=True` with `quantize=8` | Not covered by tests; export without `dynamic=True`          | Supported                                                                                                                  |
| FP16 (`quantize=16`)             | TensorRT builder flag                                        | ModelOpt AutoCast [mixed-precision](https://www.ultralytics.com/glossary/mixed-precision) conversion                       |

ModelOpt is installed on first use. Ultralytics does not reuse a calibration cache, so every export calibrates again. For details on both methods, see [Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html) in the TensorRT documentation.

For a checkpoint not trained with `quantize=8`, pass calibration images with `data`, for example `data="my_dataset.yaml"`. Calibration uses the `val` split of that dataset unless you set `split`, and `fraction` limits the number of images. NVIDIA recommends at least 500 images that represent your deployment data, and the export logs a warning below 300. Without `data`, the export logs a warning and uses the [small example dataset for the model task](../datasets/index.md), such as `coco8.yaml` for detection. Its validation split has 4 images, which is enough to test the export but not to calibrate a model for deployment.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        model.export(format="engine", quantize=8, data="coco.yaml", fraction=0.1)

        # Load the exported INT8 engine and run inference
        model = YOLO("yolo26n.engine")
        result = model.predict("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        yolo export model=yolo26n.pt format=engine quantize=8 data=coco.yaml fraction=0.1 # creates 'yolo26n.engine'

        # Run inference with the exported INT8 engine
        yolo predict model=yolo26n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

On the benchmark GPU, INT8 engines ran 1.04 to 2.3 times faster than FP32 engines, but faster than FP16 engines for only three of ten models. INT8 lost 1.7 to 2.9 mAP on YOLO26n to YOLO26l and far more on YOLO26x and YOLO26n-pose. The YOLO26n INT8 engine was larger than its FP32 engine (30 MB against 11 MB), while large models shrank (53 MB against 103 MB for YOLO26l). INT8 can also shift confidence values, so choose the confidence threshold from the INT8 engine's own [F1](https://www.ultralytics.com/glossary/f1-score) curve. See the [benchmarks](#fp32-fp16-and-int8-compared) for every model.

## Output Format

The engine takes one input named `images` and returns the same outputs as the ONNX export it is built from, with one exception: on TensorRT versions earlier than 10, semantic models return float class scores with shape `(batch, classes, height, width)` in place of the `uint8` class map. See the [ONNX output format](onnx.md#output-format) for the output shape of each task and each `nms` setting.

### Load the Engine Outside Ultralytics

An Ultralytics `.engine` file is not a plain TensorRT engine. It starts with a 4-byte little-endian length, followed by that many bytes of JSON metadata, followed by the serialized engine. TensorRT's `deserialize_cuda_engine` returns `None` when it is given the whole file. Skip the header to load the engine with the TensorRT API:

```python
import json

import tensorrt as trt

with open("yolo26n.engine", "rb") as f:
    n = int.from_bytes(f.read(4), byteorder="little")  # length of the metadata header
    metadata = json.loads(f.read(n))  # class names, task, imgsz, batch, export arguments
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
```

When another tool expects a plain TensorRT engine file, write the bytes after the header to a new file, or build the engine in that tool from the `yolo26n.onnx` file that the export leaves next to the engine. On TensorRT 11, FP16 and calibrated INT8 exports also leave `yolo26n.fp16.onnx` and `yolo26n.int8.onnx`; build an INT8 engine in another tool from the `.int8.onnx` file. For a checkpoint trained with `quantize=8`, `yolo26n.onnx` already holds the Q/DQ nodes, and no `.int8.onnx` file is written; on TensorRT 11 that export leaves `yolo26n.fp16.onnx`, the same Q/DQ graph with its other layers in FP16.

## Deploy

`YOLO("yolo26n.engine")` loads the engine with the TensorRT Python runtime. The engine runs on GPU 0 by default, also when you pass `device="cpu"`. Pass `device=1` to run it on another GPU of the same model. An engine exported with `device=dla:0` or `device=dla:1` runs on that DLA core.

An engine exported without `dynamic=True` has a fixed input shape. Predict and val use the export `imgsz` and ignore a different `imgsz` argument. Ultralytics splits larger batches into batches of the export size and pads smaller ones, so any number of images works. An engine exported with `dynamic=True` accepts every batch size and image size inside its range and raises `ValueError: input size ... is outside the TensorRT optimization profile` for a shape outside it. With `nms=True`, a batch above the range is split into batches of the export size instead.

### Deployment Guides

- **[NVIDIA Triton Inference Server](../guides/triton-inference-server.md)**: Serve an ONNX export with Triton's TensorRT accelerator, which builds the engine on the server GPU.

- **[NVIDIA Jetson quickstart](../guides/nvidia-jetson.md)**: Set up Ultralytics on Jetson devices and export TensorRT engines with JetPack, including DLA.

- **[NVIDIA DGX Spark quickstart](../guides/nvidia-dgx-spark.md)**: Set up Ultralytics and TensorRT on NVIDIA DGX Spark.

- **[DeepStream on NVIDIA Jetson](../guides/deepstream-nvidia-jetson.md)**: Run exported models in NVIDIA DeepStream video pipelines.

- **[NVIDIA TensorRT on GitHub](https://github.com/NVIDIA/TensorRT)**: Source code, samples, and tools for TensorRT.

## Limitations and Troubleshooting

- **Engines are not portable**: an engine is built for one GPU model and one TensorRT version. Rebuild it when you change the GPU, the TensorRT version, or the JetPack release. When an engine cannot be loaded, Ultralytics logs `TensorRT model exported with a different version than expected`.
- **GPU required**: the export uses `device=0` when `device` is unset and fails with `export running on CPU but must be on GPU` for `device=cpu`.
- **Engine does not load in other tools**: the `.engine` file starts with a metadata header, so the TensorRT API cannot read it directly. See [Load the Engine Outside Ultralytics](#load-the-engine-outside-ultralytics).
- **TensorRT version**: Ultralytics requires TensorRT 7.0.0 or newer. TensorRT 10.2.0 is not supported, and the automatic install skips it. On JetPack 7 and NVIDIA DGX systems, the export requires TensorRT 10.15 or newer (and below 11 on Jetson) and tries to install it when the installed version is outside that range.
- **NMS-free head on old TensorRT**: TensorRT versions earlier than 8.5.0 cannot build the YOLO26 NMS-free head. The export logs a warning and writes raw predictions that need NMS. Upgrade TensorRT to 8.5.0 or newer to export with `nms=False`.
- **INT8 on JetPack 6**: with TensorRT 10.3.0 on JetPack 6, an INT8 export with `nms=True` fails, and an INT8 export with `nms=False` falls back to raw predictions. See [the Jetson guide](../guides/nvidia-jetson.md#why-does-my-tensorrt-int8-export-disable-nms-free-inference-on-jetpack-6) for the fix.
- **INT8 on TensorRT 10.16**: INT8 export with TensorRT 10.16.1 failed on NVIDIA RTX PRO 6000 Blackwell and RTX 6000D GPUs with `Could not find any implementation for node`. On the RTX PRO 6000 Blackwell, TensorRT 10.13, 10.14, 10.15, and 11.4 built the INT8 engine.
- **DLA**: `device=dla:0` and `device=dla:1` work only on NVIDIA Jetson, and only with `quantize=16` or `quantize=8`. The export raises an error on TensorRT 11. Layers that DLA does not support run on the GPU.
- **Dynamic shapes**: `dynamic=True` needs a maximum batch size, so also set `batch`, for example `batch=8`. Without it the export logs a warning and the maximum batch size is 1. The engine accepts batches from 1 to `batch` and image sizes from 32 pixels up to twice the export `imgsz`. For example, an engine exported with `imgsz=320 batch=4` runs at 640 pixels and rejects 704 pixels or a batch of 6 with `input size ... is outside the TensorRT optimization profile`. Combine `dynamic=True` with `quantize=8` only on TensorRT 11. On GPUs with compute capability 7.5 or lower, such as the T4, the minimum size of those dynamic INT8 engines is 64 pixels (128 for P6 models).
- **Fixed input size**: an engine exported without `dynamic=True` always runs at its export `imgsz`. Passing another `imgsz` to predict or val has no effect, so re-export to change the input size.
- **INT8 export without `data`**: unless the checkpoint was trained with `quantize=8`, the export calibrates on the small default dataset for the task, which has 4 validation images for detection, and logs a warning. Pass `data` with representative images; the export warns below 300.
- **Slow first inferences**: the first few calls with a new engine, or with a new `imgsz` on a dynamic engine, take longer than later calls.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

The [YOLO26 model page](../models/yolo26.md) lists the TensorRT 10 speed of every model on an NVIDIA T4 GPU for all tasks except semantic segmentation.

### FP32, FP16, and INT8 Compared

The tables below compare PyTorch with FP32, FP16, and INT8 TensorRT 11.4 engines of every YOLO26 detection model, and of the YOLO26n model for five other tasks. All values were measured on one **NVIDIA RTX PRO 6000 Blackwell Server Edition** GPU.

=== "Latency"

    Milliseconds per image, lower is better. The fastest format for each model is in bold; the YOLO26x INT8 engine is excluded because its accuracy was unstable.

    | Model        | `imgsz` | PyTorch | TensorRT FP32 | TensorRT FP16 | TensorRT INT8 |
    | ------------ | ------- | ------- | ------------- | ------------- | ------------- |
    | YOLO26n      | 640     | 2.85    | 0.67          | **0.48**      | 0.53          |
    | YOLO26s      | 640     | 2.98    | 1.15          | **0.61**      | 0.66          |
    | YOLO26m      | 640     | 3.29    | 2.22          | **1.03**      | 1.08          |
    | YOLO26l      | 640     | 4.70    | 3.03          | 1.49          | **1.43**      |
    | YOLO26x      | 640     | 5.93    | 4.31          | **2.09**      | 1.84          |
    | YOLO26n-seg  | 640     | 3.18    | 0.93          | **0.65**      | 0.72          |
    | YOLO26n-sem  | 1024    | 1.74    | 0.74          | **0.45**      | 0.71          |
    | YOLO26n-cls  | 224     | 1.75    | 0.29          | 0.24          | **0.20**      |
    | YOLO26n-pose | 640     | 3.39    | 0.76          | **0.51**      | 0.59          |
    | YOLO26n-obb  | 1024    | 4.55    | 1.03          | **0.64**      | 0.88          |

=== "Accuracy"

    Higher is better. A range means that separate INT8 builds of the same model gave different accuracy. OBB uses the DOTAv1 val split and semantic segmentation uses `imgsz=1024` instead of 1024 x 2048, so these two rows differ from the model tables.

    | Model        | Metric                             | PyTorch | TensorRT FP32 | TensorRT FP16 | TensorRT INT8 |
    | ------------ | ---------------------------------- | ------- | ------------- | ------------- | ------------- |
    | YOLO26n      | mAP<sup>val</sup> 50-95, COCO      | 40.8    | 40.9          | 40.9          | 38.7          |
    | YOLO26s      | mAP<sup>val</sup> 50-95, COCO      | 48.6    | 48.5          | 48.5          | 46.3          |
    | YOLO26m      | mAP<sup>val</sup> 50-95, COCO      | 53.1    | 53.2          | 53.2          | 51.5          |
    | YOLO26l      | mAP<sup>val</sup> 50-95, COCO      | 55.0    | 54.9          | 54.8          | 52.0          |
    | YOLO26x      | mAP<sup>val</sup> 50-95, COCO      | 57.5    | 57.5          | 57.5          | 25.3 to 38.5  |
    | YOLO26n-seg  | Mask mAP<sup>val</sup> 50-95, COCO | 34.3    | 34.4          | 34.4          | 32.2          |
    | YOLO26n-sem  | mIoU, Cityscapes                   | 72.2    | 71.3          | 71.3          | 67.0          |
    | YOLO26n-cls  | Top-1 accuracy, ImageNet           | 71.4    | 71.4          | 71.4          | 69.5          |
    | YOLO26n-pose | Pose mAP<sup>val</sup> 50-95, COCO | 57.0    | 56.7          | 56.7          | 33.4          |
    | YOLO26n-obb  | mAP<sup>val</sup> 50-95, DOTAv1    | 46.5    | 46.4          | 46.4          | 44.3          |

=== "Size"

    File size in MB. The smallest TensorRT engine for each model is in bold, with the same YOLO26x exception.

    | Model        | PyTorch | TensorRT FP32 | TensorRT FP16 | TensorRT INT8 |
    | ------------ | ------- | ------------- | ------------- | ------------- |
    | YOLO26n      | 5.5     | 10.9          | **6.8**       | 29.6          |
    | YOLO26s      | 20.4    | 41.0          | **21.2**      | 36.0          |
    | YOLO26m      | 44.3    | 84.8          | **43.0**      | 47.3          |
    | YOLO26l      | 53.2    | 102.6         | **52.0**      | 52.7          |
    | YOLO26x      | 118.7   | 227.4         | **114.0**     | 84.8          |
    | YOLO26n-seg  | 6.7     | 14.2          | **7.8**       | 30.4          |
    | YOLO26n-sem  | 3.5     | 11.3          | **6.2**       | 65.6          |
    | YOLO26n-cls  | 5.8     | 12.1          | **6.3**       | 8.2           |
    | YOLO26n-pose | 7.9     | 13.4          | **8.2**       | 30.7          |
    | YOLO26n-obb  | 5.9     | 15.6          | **8.4**       | 67.8          |

- **FP16** kept the FP32 accuracy within 0.1 points on every model. On the detection models it ran 1.4 to 2.2 times faster than the FP32 engine and 2.8 to 5.9 times faster than PyTorch.
- **INT8** ran within 12% of the FP16 latency on the detection models and lost 1.7 to 2.9 mAP on YOLO26n, YOLO26s, YOLO26m, and YOLO26l. On this GPU it was faster than FP16 only for YOLO26l, YOLO26x, and YOLO26n-cls.
- **YOLO26x INT8** accuracy fell to between 25.3 and 38.5 mAP across three builds, against 57.5 in FP32. **YOLO26n-pose INT8** fell from 56.7 to 33.4 pose mAP. Validate every INT8 engine on your own data before you deploy it.
- **TensorRT 10.16** FP32 and FP16 engines of YOLO26n and YOLO26s matched the TensorRT 11.4 accuracy. Their FP16 engines were about 7% faster and their FP32 engines 4% to 9% slower than on TensorRT 11.4.

For NVIDIA Jetson and DGX Spark, see the [Jetson benchmarks](../guides/nvidia-jetson.md#nvidia-jetson-yolo26-benchmarks) and the [DGX Spark guide](../guides/nvidia-dgx-spark.md).

Measured with `ultralytics` 8.4.211, TensorRT 11.4.0, PyTorch 2.15.0, and CUDA 13.2 at `batch=1` on November 13, 2026. Latency is the inference stage that `val` reports and does not include preprocessing or postprocessing.

## About TensorRT

[TensorRT](https://developer.nvidia.com/tensorrt) builds an engine by fusing layers, selecting the fastest kernel for each layer on the build GPU, and optionally lowering precision to FP16 or INT8. For the full reference, see the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/).

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/c446134bc8f887106e37cacc8a72b475.avif" alt="NVIDIA TensorRT connecting deep learning frameworks to NVIDIA data center GPUs, Jetson, and DLA hardware">
</p>

## FAQ

### Can I run a TensorRT engine on a different GPU than the one it was built on?

Only on a GPU of the same model with the same TensorRT version. An engine is built for one GPU model and one TensorRT version. To share one file across different hardware, export to [ONNX](onnx.md) and build the engine on each device. For NVIDIA Jetson, [Ultralytics Platform](../platform/train/models.md#nvidia-jetson-tensorrt-targets) can build engines for eight Jetson targets.

### Should I export a TensorRT engine in FP16 or INT8?

Start with FP16 (`quantize=16`). It needs no calibration data and keeps FP32 accuracy. In the [benchmarks](#fp32-fp16-and-int8-compared) on an NVIDIA RTX PRO 6000 Blackwell with TensorRT 11.4, FP16 reduced YOLO26s latency from 1.15 ms to 0.61 ms per image at the same 48.5 mAP<sup>val</sup> 50-95. INT8 (`quantize=8`) ran at 0.66 ms and 46.3 mAP<sup>val</sup> 50-95, so it was not faster than FP16 on that GPU, and it needs representative [calibration data](#exporting-tensorrt-with-int8-quantization) unless the checkpoint was trained with `quantize=8`. INT8 can pay off on other hardware: the [Jetson benchmarks](../guides/nvidia-jetson.md#nvidia-jetson-yolo26-benchmarks) list FP16 and INT8 results for embedded devices. Measure both precisions on your deployment device before you choose INT8.

### How do I run batches or different image sizes with a TensorRT engine?

Export with `dynamic=True` and set `batch` to the largest batch you plan to run. The engine then accepts batches from 1 to `batch` and image sizes from 32 pixels up to twice the export `imgsz`, with a higher minimum for [dynamic INT8 engines on older GPUs](#limitations-and-troubleshooting). Without `dynamic=True` the engine has one fixed input shape, and Ultralytics splits or pads each batch to fit it.

!!! example "Dynamic engine"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="engine", dynamic=True, batch=8, quantize=16)

        model = YOLO("yolo26n.engine")
        results = model(["https://ultralytics.com/images/bus.jpg"] * 4, batch=4)  # a batch of 4 images
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=engine dynamic=True batch=8 quantize=16
        ```

### How do I export a TensorRT engine for NVIDIA Jetson?

Run the export on the Jetson itself with the TensorRT version from your JetPack release. To use a DLA core, add `device=dla:0` with `quantize=16` or `quantize=8`. See the [NVIDIA Jetson guide](../guides/nvidia-jetson.md) for setup and benchmarks.

!!! example "Export on Jetson"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="engine", quantize=16)  # FP16 engine on the Jetson GPU

        # Or build the FP16 engine for DLA core 0 instead
        model.export(format="engine", quantize=16, device="dla:0")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=engine quantize=16 # FP16 engine on the Jetson GPU

        # Or build the FP16 engine for DLA core 0 instead
        yolo export model=yolo26n.pt format=engine quantize=16 device=dla:0
        ```

### How do I deploy YOLO26 TensorRT models on an NVIDIA Triton Inference Server?

The [Triton Inference Server guide](../guides/triton-inference-server.md) exports the model to ONNX and enables Triton's TensorRT accelerator in `config.pbtxt`. Triton then builds the TensorRT engine on the server GPU at first load and caches it, so the engine always matches the server's GPU and TensorRT version. The guide covers the model repository layout, the server container, and client inference with Ultralytics.
