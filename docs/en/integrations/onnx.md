---
comments: true
description: Export Ultralytics YOLO26 models to ONNX and run them with ONNX Runtime on CPUs and GPUs. Covers output shapes, opset, dynamic axes, FP16, and INT8.
keywords: YOLO26, ONNX, ONNX export, ONNX Runtime, Ultralytics, ONNX opset, ONNX output shape, dynamic axes, INT8 quantization, FP16, OpenCV DNN, model deployment
---

# ONNX Export for YOLO26 Models

[ONNX](https://onnx.ai/) ([Open Neural Network Exchange](https://www.ultralytics.com/glossary/onnx-open-neural-network-exchange)) is an open model format that most inference runtimes can load. Exporting an [Ultralytics YOLO26](https://platform.ultralytics.com/ultralytics/yolo26) model with `format="onnx"` creates a single `.onnx` file that runs with ONNX Runtime on CPUs, NVIDIA and AMD GPUs, Apple silicon, and Windows devices.

## At a Glance

| Property          | Details                                                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `format` argument | `onnx`                                                                                                                                                                         |
| Output            | `yolo26n.onnx`, one file with the class names, task, and image size stored as metadata; `yolo26n_int8.onnx` for a calibrated `quantize=8` export                               |
| Runs on           | CPUs, NVIDIA GPUs (CUDA), [AMD GPUs](#amd-gpu-inference-with-migraphx) (MIGraphX), Apple silicon (Core ML execution provider), and [Windows ML](#windows-ml-inference) devices |
| Runtime           | [ONNX Runtime](https://onnxruntime.ai/), or [OpenCV](https://www.ultralytics.com/glossary/opencv) DNN with `dnn=True`                                                          |
| Precision         | FP32, FP16, INT8. YOLO26n is 9.9 MB in FP32, 5.0 MB in FP16, and 3.1 MB in INT8                                                                                                |
| Export host       | Linux, macOS, or Windows, with or without a GPU                                                                                                                                |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to ONNX for you, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package. The first export installs `onnx`, `onnxslim`, and `onnxruntime` if they are missing (`onnxruntime-gpu` when you export with `device=0` on an NVIDIA GPU).

```bash
pip install ultralytics
```

For other installation methods, see the [installation guide](../quickstart.md). Then [export your model](https://academy.ultralytics.com/courses/yolo-in-production/export-to-onnx) and load the exported file to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to ONNX format
        model.export(format="onnx")  # creates 'yolo26n.onnx'

        # Export an INT8-quantized ONNX model with calibration data
        model.export(format="onnx", quantize=8, data="coco8.yaml")  # creates 'yolo26n_int8.onnx'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to ONNX format
        yolo export model=yolo26n.pt format=onnx # creates 'yolo26n.onnx'

        # Export an INT8-quantized ONNX model with calibration data
        yolo export model=yolo26n.pt format=onnx quantize=8 data=coco8.yaml # creates 'yolo26n_int8.onnx'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ONNX model
        model = YOLO("yolo26n.onnx")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported ONNX model
        yolo predict model=yolo26n.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ONNX model
        model = YOLO("yolo26n.onnx")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported ONNX model
        yolo val model=yolo26n.onnx data=coco8.yaml
        ```

## Supported Tasks

ONNX export supports all seven Ultralytics tasks with [YOLOv8](../models/yolov8.md), [YOLO11](../models/yolo11.md), and [YOLO26](../models/yolo26.md) models: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

## Export Arguments

| Argument   | Type                      | Default  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------- | ------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`                     | `'onnx'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `imgsz`    | `int` or `tuple`          | `640`    | Image size of the model input: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect, segment, pose, and semantic models pretrained on ADE20K, `1024` for OBB and semantic models pretrained on Cityscapes, `768` for depth, and `224` for classify.                                                                                                                                       |
| `quantize` | `int` or `str`            | `None`   | Quantization precision: `16` (FP16; a CPU export keeps FP32 inputs and outputs, a GPU export makes them FP16 too) or `8` (INT8 static quantization with ONNX Runtime using calibration images from `data`, producing an `_int8.onnx` model); `32`/unset is FP32. A checkpoint trained with `quantize=8` exports INT8 with `quantize` unset or `8` (other values raise an error), as Q/DQ nodes from the ranges it carries, without calibration and under the plain `.onnx` name. Replaces the deprecated `half`/`int8` flags. |
| `data`     | `str`                     | `None`   | Dataset YAML used for INT8 calibration; classification instead takes a dataset directory or a built-in dataset name. If omitted with `quantize=8`, Ultralytics selects the default calibration dataset for the model task; a checkpoint trained with `quantize=8` needs none.                                                                                                                                                                                                                                                 |
| `fraction` | `float`, `int`, or `list` | `1.0`    | Calibration subset as a ratio, image count, or `[train, val, test]` ratios/counts. Two-item lists leave `test` full, while `0` skips it.                                                                                                                                                                                                                                                                                                                                                                                      |
| `dynamic`  | `bool`                    | `False`  | Makes the batch, height, and width of the input dynamic axes, so one file accepts different batch sizes (up to the export `batch` with `nms=True`) and image sizes. See [dynamic export](#how-do-i-export-an-onnx-model-with-dynamic-batch-or-image-size).                                                                                                                                                                                                                                                                    |
| `simplify` | `bool`                    | `True`   | Simplifies the model graph with `onnxslim`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `opset`    | `int`                     | `None`   | Specifies the ONNX opset version for compatibility with different ONNX parsers and runtimes. If not set, uses opset `18` (lower on PyTorch < 2.4), capped at the installed `onnx` version. See [minimum opset](#limitations-and-troubleshooting).                                                                                                                                                                                                                                                                             |
| `nms`      | `bool`, optional          | `None`   | Select raw output (`None`, default), embedded NMS (`True`), or the NMS-free head (`False`). See [output format](#output-format).                                                                                                                                                                                                                                                                                                                                                                                              |
| `batch`    | `int`                     | `1`      | Batch size of the exported model. A static export has this batch size fixed in the input shape. With `dynamic=True` and `nms=True`, set it to the largest batch size you plan to run.                                                                                                                                                                                                                                                                                                                                         |
| `device`   | `str`                     | `None`   | Device to export on: `cpu` (the default, also on a machine with a GPU), `0` for a CUDA GPU, or `mps` for Apple silicon. It changes the result only for `quantize=16`, see [FP16 inputs](#limitations-and-troubleshooting).                                                                                                                                                                                                                                                                                                    |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### INT8 Quantization

For checkpoints not trained with [quantization-aware training](../modes/export.md#quantization-aware-training), `quantize=8` applies ONNX Runtime static [quantization](https://www.ultralytics.com/glossary/model-quantization) to `Conv`, `Gemm`, and `MatMul` nodes only. All other nodes stay in float, including the ops that assemble `output0`: box coordinates (pixels) and class scores (0 to 1) share that tensor, so a single INT8 scale for it would round every class score to zero. Weights are quantized per channel at opset 13 and above. Exclude the same nodes and set `per_channel=True` if you run `quantize_static` on an exported FP32 model yourself:

```python
import onnx
from onnxruntime.quantization import quantize_static

model = "yolo26n.onnx"
exclude = [n.name for n in onnx.load(model).graph.node if n.op_type not in {"Conv", "Gemm", "MatMul"}]
quantize_static(model, "yolo26n_int8.onnx", calibration_reader, nodes_to_exclude=exclude, per_channel=True)
```

Here `calibration_reader` is an `onnxruntime.quantization.CalibrationDataReader` that yields `{"images": array}` with the batch size of the model, preprocessed as for inference: letterboxed to the export size (resized and center-cropped for classification), RGB, NCHW, `float32` in 0-1.

An INT8 model keeps the input and output types of the FP32 model, so inference code written for the FP32 model runs it unchanged. INT8 makes the file three to four times smaller than FP32, but it ran slower than FP32 with ONNX Runtime on the GPU and the CPU in the [benchmarks](#fp32-fp16-and-int8-with-onnx-runtime). Measure latency on your target hardware before you choose a precision.

## Output Format

The exported model takes one input named `images` with shape `(batch, 3, height, width)`, RGB and scaled to 0-1. The input is `float32`, except for GPU exports made with `quantize=16`, which take `float16`. The `nms` argument sets the layout of `output0`. The shapes below are for a YOLO26n detection model with 80 classes exported at `imgsz=640`:

| `nms`            | `output0` shape | Contents                                                                                                                       |
| ---------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `None` (default) | `(1, 84, 8400)` | For each of the 8400 anchors: box center x, center y, width, height in input pixels, then 80 class scores. Apply NMS yourself. |
| `True`           | `(1, 300, 6)`   | NMS runs inside the model. Each row is `x1, y1, x2, y2, score, class`; rows past the last detection are zeros.                 |
| `False`          | `(1, 300, 6)`   | NMS-free head. Each row is `x1, y1, x2, y2, score, class`, with no NMS step. Filter rows by score yourself.                    |

The other tasks use the same input and these outputs. The shapes are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | Output with `nms=None`                                          | Output with `nms=False` or `nms=True`                                                       |
| -------- | --------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Detect   | `640`           | `output0` `(1, 84, 8400)`                                       | `output0` `(1, 300, 6)`                                                                     |
| Segment  | `640`           | `output0` `(1, 116, 8400)`, `output1` `(1, 32, 160, 160)`       | `output0` `(1, 300, 38)`, `output1` `(1, 32, 160, 160)`                                     |
| Semantic | `1024`          | `output0` `(1, 1024, 1024)`, `uint8` class index for each pixel | Same as `nms=None`                                                                          |
| Depth    | `768`           | `output0` `(1, 1, 768, 768)`, depth map                         | Same as `nms=None`                                                                          |
| Classify | `224`           | `output0` `(1, 1000)`, class probabilities after softmax        | Same as `nms=None` for `nms=False`; `nms=True` raises an error                              |
| Pose     | `640`           | `output0` `(1, 56, 8400)`                                       | `output0` `(1, 300, 57)`: 6 box values, then 17 keypoints x 3                               |
| OBB      | `1024`          | `output0` `(1, 20, 21504)`                                      | `output0` `(1, 300, 7)`: `center x, center y, width, height, score, class, angle` (radians) |

Segmentation rows carry 32 mask coefficients after the box values, and `output1` holds the mask prototypes. In FP32 and INT8 models all outputs are `float32` except the semantic class map.

With `nms=True`, the confidence threshold `conf` (default `0.25`), the IoU threshold `iou` (default `0.5`), `agnostic_nms`, and `max_det` (default `300`) are fixed in the graph at export. A lower `conf` at prediction time cannot bring back detections the graph already removed, so export again to change these values. `max_det` and `agnostic_nms` also apply to `nms=False`: `max_det=100` gives `output0` shape `(1, 100, 6)`.

`YOLO("yolo26n.onnx")` reads the layout from the file metadata, so you only need these shapes when you run the model in your own code. The metadata is also available to your code as strings: `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, and `args`.

```python
import ast

import onnxruntime as ort

meta = ort.InferenceSession("yolo26n.onnx").get_modelmeta().custom_metadata_map
names = ast.literal_eval(meta["names"])  # {0: 'person', 1: 'bicycle', ...}
imgsz = ast.literal_eval(meta["imgsz"])  # [640, 640]
```

## Deploy

`YOLO("yolo26n.onnx")` runs the model with ONNX Runtime and selects the execution provider from the `device` argument. On a machine with a CUDA GPU, the GPU is used when `device` is not set.

| `device`               | Execution provider                                                                                              |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| `cpu`                  | `CPUExecutionProvider`                                                                                          |
| `0` on an NVIDIA GPU   | `CUDAExecutionProvider`, from the `onnxruntime-gpu` package that is installed on first use                      |
| `0` on an AMD ROCm GPU | `MIGraphXExecutionProvider`, see [AMD GPU Inference with MIGraphX](#amd-gpu-inference-with-migraphx)            |
| `mps` on Apple silicon | `CoreMLExecutionProvider` for nodes with static shapes; nodes with dynamic shapes run on `CPUExecutionProvider` |

The log line `Using ONNX Runtime 1.31.1 with CUDAExecutionProvider` shows which provider is active. The behavior at inference time depends on how the model was exported:

- **Static export (default)**: the model always runs at its export `imgsz` and `batch`, which Ultralytics reads from the file metadata. An `imgsz` passed to `predict` is ignored. Larger batches are split into chunks of the export batch size and smaller ones are padded, so any number of images works.
- **Dynamic export (`dynamic=True`)**: `predict` accepts any `imgsz` that is a multiple of the model stride. The batch size can vary, but with `nms=True` it is limited to the `batch` value used at export, so set that to the largest batch you plan to run.
- **GPU memory**: with `CUDAExecutionProvider`, a static model uses ONNX Runtime IO binding, which keeps input and output tensors on the GPU. A dynamic model copies them through CPU memory on every call.
- **CPU threads**: Ultralytics limits ONNX Runtime to 8 intra-op threads, or to the number of physical CPU cores available to the process if that is lower. Set the `ORT_INTRA_OP_NUM_THREADS` environment variable to use a different number.

Pass `dnn=True` to `predict` or `val` to run the model on the CPU with OpenCV DNN in place of ONNX Runtime. OpenCV DNN supports fewer export options, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

To run the model outside the Ultralytics package, see these resources:

- **[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)**: load and run an ONNX model in Python.
- **[ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/)**: hardware backends such as CUDA, TensorRT, OpenVINO, Core ML, and DirectML.
- **[Ultralytics examples](https://github.com/ultralytics/ultralytics/tree/main/examples)**: ONNX Runtime inference in [Python](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-ONNXRuntime), [C++](https://github.com/ultralytics/ultralytics/tree/main/examples/cpp/ONNXRuntime), and [Rust](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLO-Series-ONNXRuntime-Rust), and OpenCV DNN inference in [Python](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-OpenCV-ONNX-Python).
- **[Triton Inference Server](../guides/triton-inference-server.md)**: serve the ONNX model over HTTP or gRPC with NVIDIA Triton.
- **[Deploying on edge devices](https://onnxruntime.ai/docs/tutorials/iot-edge/)**: ONNX Runtime tutorials for IoT and edge hardware.

### Windows ML Inference

On Windows 11 24H2 (build 26100) or newer with Python 3.10+, you can install `onnxruntime-windowsml` in a separate environment to let [Windows ML select execution providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers) for your hardware. Install it instead of other ONNX Runtime builds, then use the same exported `.onnx` model with `YOLO("model.onnx")`. The backend preserves the PyTorch thread count and supplied session options, including an explicit provider policy. Without a supplied policy, Windows ML selects hardware for maximum performance, even when the PyTorch device is `cpu`. Measure prediction latency on your device before choosing this runtime; it is optional and is not installed by default.

On integrated GPUs that share power with the CPU, idle PyTorch threads in preprocessing and postprocessing can slow Windows ML inference. Call `torch.set_num_threads(1)` before predicting to reduce this. It applies to the whole process, so other PyTorch models in that process also run on one thread.

```python
import torch

from ultralytics import YOLO

torch.set_num_threads(1)  # optional, measure on your device
results = YOLO("yolo26n.onnx")("https://ultralytics.com/images/bus.jpg")
```

### AMD GPU Inference with MIGraphX

Ultralytics runs exported ONNX models on AMD GPUs through the ONNX Runtime [MIGraphX execution provider](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html). On a ROCm (HIP) system with Python 3.11 or newer on x86-64, the ONNX backend installs the `onnxruntime-ep-migraphx` plugin on first use and selects `MIGraphXExecutionProvider` automatically with no code changes, and all YOLO26 tasks are supported. See the [AMD GPU inference guide](amd.md) for prerequisites, installation, usage, [benchmarks](amd.md#benchmarks), and the compiled-program cache.

## Limitations and Troubleshooting

- **Minimum opset**: the export fails with `ONNX export requires opset>=N` when `opset` is too low for the model. The minimum is `7` for classification, `9` for detection, segmentation, semantic segmentation, pose, and OBB, `11` for depth estimation and for detection, segmentation, pose, and OBB models exported with `nms=True`, and `16` for [RT-DETR](../models/rtdetr.md).
- **FP16 inputs**: an FP16 model exported on a GPU (`quantize=16 device=0`) expects FP16 input tensors in your own code. An FP16 model exported on CPU keeps FP32 inputs and outputs and converts inside the graph. `YOLO()` feeds the right input type to both, on GPU and on CPU.
- **INT8 file name**: `quantize=8` writes `yolo26n_int8.onnx` and removes the intermediate FP32 file, so load the `_int8` file for inference. A checkpoint trained with `quantize=8` is the exception: its export writes Q/DQ nodes to `yolo26n.onnx` and skips calibration.
- **INT8 calibration size**: for a checkpoint not trained with `quantize=8`, the export warns when the calibration split has fewer than 300 images. Calibration uses the `val` split by default, and `coco8.yaml` has only 4 validation images, so use your own dataset for a model you plan to deploy.
- **Embedded NMS**: `nms=True` requires PyTorch 1.13 or newer. It raises `'nms=True' is not valid for classification models` for classification, and is ignored with a warning for semantic segmentation, depth estimation, and RT-DETR models.
- **NMS-free head**: `nms=False` needs a model with an NMS-free head, such as YOLO26 or YOLOv10. For other models, for example YOLO11, the export warns `This model has no one-to-one head; using one-to-many outputs` and writes the raw `(1, 84, 8400)` output.
- **GPU not used**: if the log shows `CUDA requested but CUDAExecutionProvider not available. Using CPU...`, the installed ONNX Runtime build has no working CUDA provider. Install `onnxruntime-gpu` in place of `onnxruntime` and check that its CUDA and cuDNN versions match your system.
- **OpenCV DNN**: with OpenCV 4.14.0, `dnn=True` runs default, `nms=False`, INT8, and GPU-exported FP16 models. It fails to load models exported with `nms=True`, with `dynamic=True`, or with `quantize=16` on CPU.
- **Dynamic shapes with NMS**: when you combine `dynamic=True` with `nms=True`, also set `batch` to the largest batch size you plan to run. The export warns if `batch` is left at `1`.
- **Graph simplification**: if `onnxslim` fails, the export logs a warning and saves the model without simplification.
- **IR version**: the exporter caps the ONNX IR version at 10 so that older ONNX Runtime releases can load the file. A default export at opset 18 has IR version 8.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

YOLO26n runs up to 43% faster than YOLO11n with ONNX Runtime on CPU (Intel Xeon CPU @ 2.00 GHz). The [YOLO26 model page](../models/yolo26.md) lists the CPU ONNX speed of every model for all tasks except semantic segmentation.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/cxU5E2SkivU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO26 vs Ultralytics YOLO11 ONNX Inference Test | ~43% Faster Inference with YOLO26 🚀
</p>

### FP32, FP16, and INT8 with ONNX Runtime

The tables below compare PyTorch with FP32, FP16, and INT8 ONNX models of every YOLO26 detection model, run with the ONNX Runtime `CUDAExecutionProvider`. All GPU values were measured on one **NVIDIA RTX PRO 6000 Blackwell Server Edition** GPU.

=== "Latency"

    Milliseconds per image, lower is better. The fastest format for each model is in bold.

    | Model   | `imgsz` | PyTorch | ONNX FP32 | ONNX FP16 | ONNX INT8 |
    | ------- | ------- | ------- | --------- | --------- | --------- |
    | YOLO26n | 640     | 2.85    | 1.96      | **1.88**  | 3.73      |
    | YOLO26s | 640     | 2.98    | 2.14      | **1.73**  | 4.31      |
    | YOLO26m | 640     | 3.29    | 3.01      | **2.14**  | 5.37      |
    | YOLO26l | 640     | 4.70    | 4.09      | **3.08**  | 7.51      |
    | YOLO26x | 640     | 5.93    | 5.56      | **3.90**  | 9.38      |

=== "Accuracy"

    mAP<sup>val</sup> 50-95 on COCO val2017, higher is better.

    | Model   | PyTorch | ONNX FP32 | ONNX FP16 | ONNX INT8 |
    | ------- | ------- | --------- | --------- | --------- |
    | YOLO26n | 40.8    | 40.9      | 40.9      | 38.7      |
    | YOLO26s | 48.6    | 48.5      | 48.4      | 45.6      |
    | YOLO26m | 53.1    | 53.2      | 53.2      | 50.4      |
    | YOLO26l | 55.0    | 54.9      | 54.8      | 52.4      |
    | YOLO26x | 57.5    | 57.5      | 57.4      | 54.5      |

=== "Size"

    File size in MB. The smallest exported file for each model is in bold.

    | Model   | PyTorch | ONNX FP32 | ONNX FP16 | ONNX INT8 |
    | ------- | ------- | --------- | --------- | --------- |
    | YOLO26n | 5.5     | 9.9       | 5.0       | **3.1**   |
    | YOLO26s | 20.4    | 38.3      | 19.2      | **10.3**  |
    | YOLO26m | 44.3    | 82.0      | 41.0      | **21.4**  |
    | YOLO26l | 53.2    | 99.6      | 49.9      | **26.0**  |
    | YOLO26x | 118.7   | 223.3     | 111.7     | **57.2**  |

**YOLO26n on CPU** (AMD EPYC 9655, PyTorch and ONNX Runtime on 8 threads each; a different CPU from the one used for the YOLO26 model tables):

| Metric                  | PyTorch | ONNX FP32 | ONNX FP16 | ONNX INT8 |
| ----------------------- | ------- | --------- | --------- | --------- |
| Latency (ms/image)      | 13.2    | 13.4      | 13.2      | 56.5      |
| mAP<sup>val</sup> 50-95 | 40.8    | 40.9      | 40.9      | 38.7      |

- **FP16** halved the file size within 0.1 mAP of FP32 and ran up to 1.4 times faster than FP32 on the GPU, with no speedup on the CPU.
- **INT8** gave the smallest file but lost 2.2 to 3.0 mAP and ran slower than FP32 on both the GPU and the CPU.

Measured with `ultralytics` 8.4.211, ONNX Runtime 1.30.0, PyTorch 2.15.0, and CUDA 13.2 at `batch=1` on November 13, 2026. Latency is the inference stage that `val` reports and does not include preprocessing or postprocessing.

## About ONNX

ONNX is a community project that Facebook and Microsoft started, now developed with organizations such as IBM, Amazon, and Google. It defines a common set of operators and a file format for [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models, so a model trained in one framework, such as [PyTorch](https://www.ultralytics.com/glossary/pytorch), can run in many runtimes and on many hardware platforms.

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/3d77aaacb115081597b87f6d3a6af615.avif" alt="ONNX model portability across deep learning frameworks">
</p>

[ONNX Runtime](https://onnxruntime.ai/) is a cross-platform inference engine for ONNX models. Its execution providers use hardware-specific libraries to run a model on CPUs, GPUs, and dedicated accelerators.

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/f669dc67507d4abd9928fcda9dee73b1.avif" alt="ONNX Runtime cross-platform inference acceleration">
</p>

ONNX is also the intermediate step for other Ultralytics export formats, including [TensorRT](tensorrt.md), and the input for vendor NPU toolchains such as [Huawei Ascend](ascend.md), [Qualcomm QNN](qnn.md), and [Rockchip RKNN](rockchip-rknn.md). For the specification, see the [ONNX documentation](https://onnx.ai/onnx/intro/).

## FAQ

### Which ONNX opset should I use for YOLO26 models?

Leave `opset` unset unless your runtime needs a specific version. The exporter then uses opset `18` on PyTorch 2.4 and newer, opset `17` on PyTorch 1.13 to 2.3, a lower opset on older PyTorch versions, and never more than the installed `onnx` package supports. The default stops at 18 because the ONNX Runtime CUDA provider runs some opset 19 and newer operators, such as `Resize`, on the CPU. Set a lower value, for example `opset=12`, when an older runtime or converter rejects the default. The value cannot go below the [minimum opset](#limitations-and-troubleshooting) for the model.

### How do I read the output of a YOLO26 ONNX model in my own code?

It depends on the `nms` argument you exported with. A default export returns raw predictions that need confidence filtering and NMS. An export with `nms=False` or `nms=True` returns up to `max_det` (default 300) final detections per image as `x1, y1, x2, y2, score, class` in input-image pixels, so you only filter by score and scale the boxes back to the original image. OBB rows hold `center x, center y, width, height` in place of the corners, plus the angle. See [Output Format](#output-format) for the shapes.

!!! example "Read NMS-free output with ONNX Runtime"

    ```python
    # Export first: yolo export model=yolo26n.pt format=onnx nms=False
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession("yolo26n.onnx")  # exported with nms=False
    image = np.zeros((1, 3, 640, 640), dtype=np.float32)  # your preprocessed image
    detections = session.run(None, {"images": image})[0][0]  # shape (300, 6)
    detections = detections[detections[:, 4] > 0.25]  # keep rows above the confidence threshold
    ```

### How do I export an ONNX model with dynamic batch or image size?

Set `dynamic=True`. The `images` input then has shape `(batch, 3, height, width)` with dynamic batch, height, and width, and a default (`nms=None`) detection `output0` has shape `(batch, 84, anchors)`, so the same file accepts different batch sizes and image sizes at inference time. With `nms=True`, the batch size is limited to the `batch` value used at export. Height and width must be multiples of the model stride, 32 for the official YOLO26 checkpoints other than classification. A dynamic model does not use IO binding on CUDA and does not load in OpenCV DNN, so keep the default static export when the input size is fixed.

!!! example "Dynamic export"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=onnx dynamic=True
        ```

### Should I use ONNX or TensorRT on NVIDIA GPUs?

Use ONNX when one file must run on different hardware, or when the deployment machine is not the machine you export on. Use [TensorRT](tensorrt.md) for the lowest latency on a specific NVIDIA GPU: a TensorRT engine is built for one GPU architecture and TensorRT version and is not portable. For other targets, see the [model deployment options guide](../guides/model-deployment-options.md).

### Can I run a YOLO26 ONNX model in a web browser or on a mobile device?

Yes. ONNX Runtime has [web](https://onnxruntime.ai/docs/tutorials/web/) and [mobile](https://onnxruntime.ai/docs/tutorials/mobile/) builds that load the same `.onnx` file. Export with `nms=False` to keep post-processing out of your application code. For phones, [Core ML](coreml.md) on iOS and [LiteRT](litert.md) on Android are the native formats.
