---
comments: true
description: Export Ultralytics YOLO26 models to Intel OpenVINO with format=openvino. Covers FP16 and INT8 quantization with NNCF, Intel CPU, GPU, and NPU devices, and benchmarks.
keywords: YOLO26, OpenVINO, OpenVINO export, Intel, OpenVINO IR, Intel CPU, Intel GPU, Intel NPU, INT8 quantization, NNCF, FP16, Intel Core Ultra, model deployment
---

# Intel OpenVINO Export

[OpenVINO](https://docs.openvino.ai/) is Intel's open-source toolkit for optimizing and running AI inference. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="openvino"` creates a `yolo26n_openvino_model/` directory that holds an OpenVINO IR model (`.xml` and `.bin` files) for Intel CPUs, integrated and discrete Intel GPUs, and Intel [NPUs](https://www.ultralytics.com/glossary/neural-processing-unit-npu), in FP32, FP16, or INT8 precision.

## At a Glance

| Property          | Details                                                                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `format` argument | `openvino`                                                                                                                                                   |
| Output            | `yolo26n_openvino_model/`, a directory with `yolo26n.xml`, `yolo26n.bin`, and `metadata.yaml`; `yolo26n_int8_openvino_model/` for a `quantize=8` export      |
| Runs on           | Intel CPUs, integrated and discrete Intel GPUs, and Intel NPUs. The CPU device also runs on other x86-64 and ARM64 processors, such as AMD and Apple silicon |
| Runtime           | [OpenVINO Runtime](https://docs.openvino.ai/2025/openvino-workflow/running-inference.html) 2024.0.0 or newer (the `openvino` Python package)                 |
| Precision         | FP32, FP16, INT8. YOLO26n is 10.1 MB in FP32, 5.3 MB in FP16, and 3.3 MB in INT8                                                                             |
| Export host       | Linux, macOS, or Windows. The export runs on the CPU and needs no GPU                                                                                        |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to OpenVINO for you, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package with the OpenVINO export dependencies:

```bash
pip install "ultralytics[export-openvino]"
```

This installs `openvino` and, on Python 3.9 or newer, `nncf` for INT8 quantization. With a plain `pip install ultralytics`, the first export installs the missing packages. The export has these version requirements:

| Package    | Required version                                                           |
| ---------- | -------------------------------------------------------------------------- |
| `torch`    | 2.1 or newer                                                               |
| `openvino` | 2024.0.0 or newer; 2025.2.0 or newer on macOS 15.4 and later               |
| `nncf`     | 2.14.0 or newer, for `quantize=8` only; below 3.0.0 with PyTorch below 2.3 |

For other installation methods, see the [installation guide](../quickstart.md). Then export your model and load the exported directory to run inference or validate its accuracy. The [OpenVINO on CPU course](https://academy.ultralytics.com/courses/yolo-in-production/openvino-on-cpu) walks through the same steps.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to OpenVINO format
        model.export(format="openvino")  # creates 'yolo26n_openvino_model/'

        # Export an INT8-quantized OpenVINO model with calibration data
        model.export(format="openvino", quantize=8, data="coco8.yaml")  # creates 'yolo26n_int8_openvino_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to OpenVINO format
        yolo export model=yolo26n.pt format=openvino # creates 'yolo26n_openvino_model/'

        # Export an INT8-quantized OpenVINO model with calibration data
        yolo export model=yolo26n.pt format=openvino quantize=8 data=coco8.yaml # creates 'yolo26n_int8_openvino_model/'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported OpenVINO model
        model = YOLO("yolo26n_openvino_model/")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")

        # Run inference on the Intel GPU; use "npu" for the Intel NPU or "cpu" for the CPU
        results = model("https://ultralytics.com/images/bus.jpg", device="xpu")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported OpenVINO model
        yolo predict model=yolo26n_openvino_model source='https://ultralytics.com/images/bus.jpg'

        # Run inference on the Intel GPU; use npu for the Intel NPU or cpu for the CPU
        yolo predict model=yolo26n_openvino_model source='https://ultralytics.com/images/bus.jpg' device=xpu
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported OpenVINO model
        model = YOLO("yolo26n_openvino_model/")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported OpenVINO model
        yolo val model=yolo26n_openvino_model data=coco8.yaml
        ```

## Supported Tasks

OpenVINO export supports all seven Ultralytics tasks with [YOLOv8](../models/yolov8.md), [YOLO11](../models/yolo11.md), and [YOLO26](../models/yolo26.md) models: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

## Export Arguments

| Argument   | Type                      | Default      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------- | ------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `format`   | `str`                     | `'openvino'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                                                 |
| `imgsz`    | `int` or `tuple`          | `640`        | Image size of the model input: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect, segment, pose, and semantic models pretrained on ADE20K, `1024` for OBB and semantic models pretrained on Cityscapes, `768` for depth, and `224` for classify.                                                                            |
| `quantize` | `int` or `str`            | `None`       | Quantization precision: `16` (FP16: weights are stored in half precision, inputs and outputs stay FP32) or `8` (INT8 post-training quantization with NNCF using calibration images from `data`, written to an `_int8_openvino_model/` directory); `32`/unset is FP32. A checkpoint trained with `quantize=8` cannot be exported to OpenVINO, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). Replaces the deprecated `half`/`int8` flags. |
| `data`     | `str`                     | `None`       | Dataset YAML used for INT8 calibration; classification instead takes a dataset directory or a built-in dataset name. Calibration uses the `val` split unless `split` is set. If omitted with `quantize=8`, Ultralytics uses the small default dataset for the model task, such as `coco8.yaml` for detection.                                                                                                                                                      |
| `fraction` | `float`, `int`, or `list` | `1.0`        | Calibration subset as a ratio, image count, or `[train, val, test]` ratios/counts. Two-item lists leave `test` full, while `0` skips it.                                                                                                                                                                                                                                                                                                                           |
| `dynamic`  | `bool`                    | `False`      | Makes the batch, height, and width of the input dynamic, so one model accepts different batch sizes and image sizes. On an Intel NPU a dynamic model still runs at the export `imgsz`, see [Deploy](#deploy).                                                                                                                                                                                                                                                      |
| `nms`      | `bool`, optional          | `None`       | Select raw output (`None`, default), embedded NMS (`True`), or the NMS-free head (`False`). See [output format](#output-format).                                                                                                                                                                                                                                                                                                                                   |
| `batch`    | `int`                     | `1`          | Batch size of the exported model. A static export has this batch size fixed in the input shape. Also the batch size used for INT8 calibration.                                                                                                                                                                                                                                                                                                                     |
| `device`   | `str`                     | `None`       | The export always runs on the CPU. With `quantize=8`, `device=npu` selects symmetric INT8 activations for YOLO detect, segment, pose, and OBB models, which run up to 2x faster on Intel NPUs.                                                                                                                                                                                                                                                                     |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### INT8 Quantization

`quantize=8` applies post-training [quantization](https://www.ultralytics.com/glossary/model-quantization) with [NNCF](https://github.com/openvinotoolkit/nncf): the export runs calibration images through the model and stores weights and activations as 8-bit integers where it can.

- **Calibration data**: the export calibrates on every full batch of the `val` split of `data`, or of the split you set with `split`. Use `fraction` to limit the number of images. The export warns below 300 images.
- **Detection head**: for models with a YOLO detection head (detect, segment, pose, and OBB) and for [RT-DETR](../models/rtdetr.md), the operations that decode the head outputs (reshape, concatenation, box decoding, and DFL) and all sigmoid operations stay in floating point, while the convolutions of the head are quantized. Classification, semantic segmentation, and depth models have no excluded operations.
- **Inputs and outputs**: an INT8 model keeps the `float32` input and outputs of the FP32 model, so inference code written for the FP32 model runs it unchanged.
- **Intel NPU**: add `device=npu` to an INT8 export that you plan to run on an Intel NPU. It switches YOLO detect, segment, pose, and OBB models from asymmetric to symmetric activation quantization.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        model.export(format="openvino", quantize=8, data="coco.yaml", fraction=0.1)

        # Load the exported INT8 model and run inference
        model = YOLO("yolo26n_int8_openvino_model/")
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        yolo export model=yolo26n.pt format=openvino quantize=8 data=coco.yaml fraction=0.1 # creates 'yolo26n_int8_openvino_model/'

        # Run inference with the exported INT8 model
        yolo predict model=yolo26n_int8_openvino_model source='https://ultralytics.com/images/bus.jpg'
        ```

In the [benchmarks](#openvino-yolo26-benchmarks), INT8 models ran 1.02 to 1.33 times faster than FP32 on a server CPU and lost 0.5 to 0.9 mAP on COCO val2017. On Intel Core Ultra CPUs INT8 ran 1.2 to 3.8 times faster than FP32. Validate an INT8 model on your own data before you deploy it.

## Output Format

The export writes one directory with three files:

| File            | Contents                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `yolo26n.xml`   | The network graph in OpenVINO IR format                                                                                  |
| `yolo26n.bin`   | The weights                                                                                                              |
| `metadata.yaml` | Class names, task, image size, batch size, stride, and export arguments, which Ultralytics reads when it loads the model |

The model takes one input with shape `(batch, 3, height, width)`, RGB and scaled to 0-1. The input and the outputs are `float32` in FP32, FP16, and INT8 models, except the semantic segmentation output, a `uint8` class map (`int32` for models with more than 256 classes). The tensors do not carry the `images` and `output0` names of the ONNX export, so address the input and the outputs by index in your own code.

The output shapes are the same as for the ONNX export and depend on the task and the `nms` argument: a YOLO26n detection model exported at `imgsz=640` returns `(1, 84, 8400)` with the default `nms=None` and `(1, 300, 6)` with `nms=False` or `nms=True`. See the [ONNX output format](onnx.md#output-format) for the shapes and contents of every task. With `nms=True`, the confidence threshold `conf` (default `0.25`), the IoU threshold `iou`, `agnostic_nms`, and `max_det` are fixed in the graph at export. A lower `conf` at prediction or validation time cannot bring back detections the graph already removed, so export again to change these values.

## Deploy

`YOLO("yolo26n_openvino_model/")` runs the model with the OpenVINO Runtime and maps the `device` argument onto an OpenVINO device:

| `device`         | OpenVINO device                                                                                                                                                                                                                                                                                     |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| unset            | `AUTO`, which lets OpenVINO choose between the CPU and Intel GPUs (its default priority list [excludes the NPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html)), or `CPU` when the CPU is the only device OpenVINO finds |
| `cpu`            | `CPU`                                                                                                                                                                                                                                                                                               |
| `xpu` or `xpu:0` | `GPU` or `GPU.0`, the Intel integrated or discrete GPU. `xpu:1` selects `GPU.1` on a machine with two Intel GPUs                                                                                                                                                                                    |
| `npu`            | `NPU`, the neural processing unit of Intel Core Ultra processors                                                                                                                                                                                                                                    |

Other values, such as `0` or `mps`, are handled like an unset `device`. The older `intel:cpu`, `intel:gpu`, and `intel:npu` names still work and log a deprecation warning. The [inference devices table](../modes/predict.md#inference-devices) compares these names across formats. The log line `Using OpenVINO LATENCY mode for batch=1 inference on CPU...` names the device that OpenVINO reports when the model loads. With `AUTO` this can be the CPU while an Intel GPU is still compiling the model, so set `device` to be sure. For a `.pt` model, `device="xpu"` selects the PyTorch Intel GPU backend instead, which is also the device for [training on Intel GPUs](../modes/train.md#multi-gpu-training), and `device="npu"` selects a Huawei Ascend NPU, not the Intel NPU.

The behavior at inference time depends on how the model was exported:

- **Static export (default)**: the model always runs at its export `imgsz` and `batch`, which Ultralytics reads from `metadata.yaml`. An `imgsz` passed to `predict` is ignored. Larger batches are split into chunks of the export batch size and smaller ones are padded, so any number of images works.
- **Dynamic export (`dynamic=True`)**: `predict` accepts different batch sizes and every `imgsz` that is a multiple of the model stride. Intel NPUs and INT8 models on Intel AMX CPUs are exceptions, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).
- **Latency mode**: Ultralytics compiles the model with the OpenVINO `LATENCY` performance hint and runs inference synchronously. To serve many parallel requests with the `THROUGHPUT` hint, use the OpenVINO Runtime API as shown in the [latency and throughput guide](../guides/optimizing-openvino-latency-vs-throughput-modes.md).
- **Intel NPU**: Ultralytics compiles the model for batch size 1 on an NPU and splits larger batches into single images. A dynamic export is fixed to its export `imgsz` on an NPU.

### Run with the OpenVINO Runtime API

To run the model outside the Ultralytics package, load the `.xml` file with the `openvino` package. The model expects a preprocessed image and returns the raw outputs described in [Output Format](#output-format), so letterboxing and, for a default export, NMS are up to your code:

```python
# Export first: yolo export model=yolo26n.pt format=openvino nms=False
import numpy as np
import openvino as ov
import yaml

core = ov.Core()
model = core.read_model("yolo26n_openvino_model/yolo26n.xml")
compiled = core.compile_model(model, "AUTO")  # or "CPU", "GPU", "NPU"

image = np.zeros((1, 3, 640, 640), dtype=np.float32)  # your preprocessed image: RGB, NCHW, 0-1
detections = compiled(image)[0][0]  # first output, first image: shape (300, 6)
detections = detections[detections[:, 4] > 0.25]  # keep rows above the confidence threshold

with open("yolo26n_openvino_model/metadata.yaml") as f:
    names = yaml.safe_load(f)["names"]  # {0: 'person', 1: 'bicycle', ...}
```

### Deployment Guides

- **[OpenVINO latency and throughput modes](../guides/optimizing-openvino-latency-vs-throughput-modes.md)**: choose between the lowest latency for one stream and the highest throughput for many.
- **[Intel DL Streamer](../guides/dlstreamer-intel.md)**: run exported models in GStreamer video analytics pipelines on Intel Core Ultra CPUs, GPUs, and NPUs.
- **[OpenVINO inference devices](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes.html)**: the [CPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html), [GPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html), and [NPU](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) devices and the `AUTO` mode in the OpenVINO documentation.
- **[OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks)**: Intel's tutorial notebooks, including [YOLO26 optimization](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/yolov26-optimization).

## Limitations and Troubleshooting

- **PyTorch version**: the export fails with `OpenVINO export requires torch>=2.1` on older PyTorch versions.
- **macOS 15.4 and later**: OpenVINO 2025.1.0 and older fail on these macOS versions, so the export requires `openvino>=2025.2.0` there and updates an older package.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='openvino' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends`. Export the checkpoint you started the fine-tuning from with `quantize=8` instead, or export the fine-tuned checkpoint to [ONNX](onnx.md) or [TensorRT](tensorrt.md).
- **INT8 export without `data`**: the export logs `INT8 export requires a missing 'data' arg for calibration. Using default 'data=coco8.yaml'.` and calibrates on the small default dataset for the task. Pass `data` with images that represent your deployment data.
- **INT8 calibration size**: the export logs `>300 images recommended for INT8 calibration, found 4 images.` when the calibration split has fewer than 300 images. `coco8.yaml` has only 4 validation images, which is enough to test the export but not to calibrate a model for deployment.
- **Output directory**: every export of one checkpoint other than INT8 writes the same `yolo26n_openvino_model/` directory, whatever its `quantize`, `imgsz`, `dynamic`, `nms`, or `batch`, so the later export replaces the earlier one. All INT8 exports, with or without `device=npu`, share `yolo26n_int8_openvino_model/`.
- **Device not available**: when the requested device is missing, Ultralytics logs `OpenVINO device 'GPU' not available. Using 'AUTO' instead.` (or `Using 'CPU' instead.` on a machine where OpenVINO finds only the CPU) and continues on that device. Check that your processor is [supported by OpenVINO](https://docs.openvino.ai/2025/about-openvino/release-notes-openvino/system-requirements.html) and install the Intel [GPU](https://docs.openvino.ai/2025/get-started/install-openvino/configurations/configurations-intel-gpu.html) or [NPU](https://docs.openvino.ai/2025/get-started/install-openvino/configurations/configurations-intel-npu.html) driver.
- **Intel NPU**: Ultralytics runs an NPU at batch size 1 with fixed input shapes, see [Deploy](#deploy). Pass `device=npu` to use it: an unset `device` does not select the NPU.
- **Dynamic shapes with NMS**: an export with `dynamic=True` and `nms=True` logs `'dynamic=True' export requires a maximum batch size, e.g. 'batch=16'.` when `batch` is left at `1`. Ultralytics runs larger batches in chunks of the export `batch`.
- **Dynamic INT8 models on Intel AMX CPUs**: on Linux with a CPU that supports AMX (Intel Sapphire Rapids and newer), Ultralytics runs a dynamic INT8 model on the `CPU` or `AUTO` device at its export `imgsz`, as it does a static model: an `imgsz` passed to `predict` or `val` is ignored, and `val` uses the export `batch`. `predict` still accepts other batch sizes and compiles the model again for each new batch size, so the first inference at a new batch size is slower.
- **Windows**: on Windows, Ultralytics sets the CPU inference precision hint to FP32, which stops OpenVINO from using lower-precision floating-point kernels (INT8 models still run in INT8), and limits oneDNN to instruction sets below AMX. Set the `ONEDNN_MAX_CPU_ISA` environment variable before you load the model to use a different limit.
- **Linux on ARM64**: Ultralytics runs the `CPU` device in the OpenVINO accuracy execution mode with FP32 inference precision.
- **Embedded NMS**: `nms=True` raises `'nms=True' is not valid for classification models.` for classification, and is ignored with a warning for semantic segmentation and depth estimation models (`'nms=True' is not valid for semantic models. Forcing 'nms=None'.`) and for RT-DETR models (`'nms=True' is not available for end2end models or non-YOLO heads. Forcing 'nms=None'.`).
- **NMS-free head**: `nms=False` needs a model with an NMS-free head, such as YOLO26 or YOLOv10. For other models, for example YOLO11, the export warns `This model has no one-to-one head; using one-to-many outputs.` and writes the raw `(1, 84, 8400)` output.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## OpenVINO YOLO26 Benchmarks

The tables below compare PyTorch with FP32, FP16, and INT8 OpenVINO models of every YOLO26 detection model. The first set was measured on a server CPU with COCO val2017 accuracy, and the second set on three Intel Core Ultra processors with their CPU and GPU, and the NPU of two of them. The [YOLO26 model page](../models/yolo26.md) lists the accuracy, CPU ONNX speed, and T4 TensorRT speed of every model.

### FP32, FP16, and INT8 on a Server CPU

All values in this section were measured on the CPU of one server with two **AMD EPYC 9655** 96-core processors, not on an Intel CPU. PyTorch ran on 8 threads and OpenVINO on 96 threads, the defaults that Ultralytics and OpenVINO chose on that server, so the PyTorch and OpenVINO columns differ in thread count as well as in runtime. PyTorch also validated with rectangular inference, each image padded to the smallest stride-multiple rectangle within 640 pixels, while the static OpenVINO models always ran at 640x640.

=== "Latency"

    Milliseconds per image, lower is better. The fastest format for each model is in bold.

    | Model   | `imgsz` | PyTorch | OpenVINO FP32 | OpenVINO FP16 | OpenVINO INT8 |
    | ------- | ------- | ------- | ------------- | ------------- | ------------- |
    | YOLO26n | 640     | 13.31   | 3.41          | **3.33**      | 3.35          |
    | YOLO26s | 640     | 24.28   | 5.18          | 5.18          | **4.51**      |
    | YOLO26m | 640     | 49.94   | 8.39          | 8.34          | **6.60**      |
    | YOLO26l | 640     | 63.75   | 10.92         | 11.05         | **8.94**      |
    | YOLO26x | 640     | 115.11  | 17.12         | 17.16         | **12.83**     |

=== "Accuracy"

    mAP<sup>val</sup> 50-95 on COCO val2017, higher is better.

    | Model   | PyTorch | OpenVINO FP32 | OpenVINO FP16 | OpenVINO INT8 |
    | ------- | ------- | ------------- | ------------- | ------------- |
    | YOLO26n | 40.8    | 40.9          | 40.9          | 40.3          |
    | YOLO26s | 48.6    | 48.5          | 48.6          | 47.6          |
    | YOLO26m | 53.1    | 53.2          | 53.2          | 52.3          |
    | YOLO26l | 55.0    | 54.9          | 54.9          | 54.2          |
    | YOLO26x | 57.4    | 57.4          | 57.5          | 56.9          |

=== "Size"

    Size in MB. The smallest exported model for each model is in bold.

    | Model   | PyTorch | OpenVINO FP32 | OpenVINO FP16 | OpenVINO INT8 |
    | ------- | ------- | ------------- | ------------- | ------------- |
    | YOLO26n | 5.5     | 10.1          | 5.3           | **3.3**       |
    | YOLO26s | 20.4    | 38.4          | 19.5          | **10.5**      |
    | YOLO26m | 44.3    | 82.1          | 41.4          | **21.5**      |
    | YOLO26l | 53.2    | 99.9          | 50.4          | **26.4**      |
    | YOLO26x | 118.7   | 223.6         | 112.3         | **57.4**      |

**YOLO26n on the same 8 CPUs** (PyTorch and OpenVINO both pinned to CPUs 0-7 of that server with `taskset`):

| Metric                  | PyTorch | OpenVINO FP32 | OpenVINO FP16 | OpenVINO INT8 |
| ----------------------- | ------- | ------------- | ------------- | ------------- |
| Latency (ms/image)      | 9.28    | 7.13          | 7.14          | 6.71          |
| mAP<sup>val</sup> 50-95 | 40.8    | 40.9          | 40.9          | 40.3          |

- **OpenVINO FP32** ran 3.9 to 6.7 times faster than PyTorch with the default thread counts above, at the same accuracy within 0.1 mAP. With both pinned to the same 8 CPUs, where PyTorch itself ran faster than with the default setup, OpenVINO FP32 ran YOLO26n 1.3 times faster than PyTorch and INT8 1.06 times faster than FP32.
- **FP16** halved the model size within 0.1 mAP of FP32, at the FP32 latency (within 3%).
- **INT8** ran 1.02 to 1.33 times faster than FP32, made the model 3.1 to 3.9 times smaller, and lost 0.5 to 0.9 mAP.

Measured with `ultralytics` 8.4.212, OpenVINO 2026.4.1, NNCF 3.4.0, and PyTorch 2.15.0 at `batch=1` on November 14, 2026. Latency is the inference stage that `val` reports and does not include preprocessing or postprocessing. PyTorch, FP32, and FP16 values are the mean of three runs, which differed by up to 27% for PyTorch and 6% for OpenVINO. INT8 models were calibrated on 591 COCO train2017 images (`split=train fraction=0.005`) and built three times; INT8 values are the mean of the three builds. The 8-CPU table uses one INT8 build run three times.

### Intel Core Ultra CPU, GPU, and NPU

These tables were measured on three Intel Core Ultra processors with `openvino` 2026.2.1 and `model.benchmark(data="coco128.yaml", format="openvino", device=...)`, with `device` set to `cpu`, `xpu`, or `npu`. Accuracy is mAP50-95 on the 128-image COCO128 dataset, so it is not comparable with the COCO val2017 values above. PyTorch ran on the CPU of the same processor. YOLO26 models run on the NPU of Intel Core Ultra 2xxV series, 3xx series, and newer processors only.

#### Intel Core Ultra X7 358H

The [Intel Core Ultra X7 358H](https://www.ultralytics.com/blog/accelerating-ultralytics-yolo26-with-openvino-on-intel-core-ultra-series-3-panther-lake) (Panther Lake) combines CPU cores, an integrated Intel Arc GPU, and an Intel AI Boost NPU.

=== "Latency"

    Milliseconds per image, lower is better. The fastest column for each model is in bold.

    | Model   | PyTorch CPU | CPU FP32 | CPU FP16 | CPU INT8 | GPU FP32 | GPU FP16 | GPU INT8 | NPU FP32 | NPU FP16 | NPU INT8 |
    | ------- | ----------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
    | YOLO26n | 29.28       | 14.35    | 14.23    | 9.99     | 4.09     | **4.03** | 4.24     | 7.68     | 7.71     | 8.60     |
    | YOLO26s | 59.32       | 35.03    | 33.55    | 16.68    | 5.08     | 5.14     | **4.47** | 9.89     | 9.94     | 10.69    |
    | YOLO26m | 143.21      | 90.53    | 88.82    | 33.33    | 7.10     | 7.06     | **5.61** | 13.02    | 13.13    | 14.84    |
    | YOLO26l | 181.16      | 112.83   | 111.31   | 43.27    | 8.45     | 8.49     | **6.25** | 14.53    | 14.55    | 15.64    |
    | YOLO26x | 379.75      | 241.21   | 236.87   | 78.92    | 13.03    | 12.98    | **8.74** | 23.48    | 23.41    | 21.56    |

=== "Accuracy"

    mAP50-95 on COCO128, higher is better.

    | Model   | PyTorch CPU | CPU FP32 | CPU FP16 | CPU INT8 | GPU FP32 | GPU FP16 | GPU INT8 | NPU FP32 | NPU FP16 | NPU INT8 |
    | ------- | ----------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
    | YOLO26n | 0.4765      | 0.4734   | 0.4771   | 0.4720   | 0.4762   | 0.4762   | 0.4634   | 0.4751   | 0.4762   | 0.4731   |
    | YOLO26s | 0.5703      | 0.5632   | 0.5630   | 0.5513   | 0.5616   | 0.5616   | 0.5462   | 0.5615   | 0.5615   | 0.5469   |
    | YOLO26m | 0.6196      | 0.6191   | 0.6180   | 0.6046   | 0.6166   | 0.6166   | 0.6055   | 0.6168   | 0.6166   | 0.6039   |
    | YOLO26l | 0.6215      | 0.6206   | 0.6210   | 0.5974   | 0.6205   | 0.6205   | 0.5938   | 0.6201   | 0.6205   | 0.5976   |
    | YOLO26x | 0.6512      | 0.6552   | 0.6552   | 0.6365   | 0.6568   | 0.6568   | 0.6385   | 0.6563   | 0.6561   | 0.6413   |

=== "Charts"

    <img width="800" src="https://cdn.ul.run/i/0e59bd9390255fd2bca31c34193b1442.avif" alt="YOLO26 OpenVINO benchmarks on the Intel Core Ultra X7 358H Panther Lake CPU">

    <img width="800" src="https://cdn.ul.run/i/77186d2ff07d47c8d926468ebdecf8e1.avif" alt="YOLO26 OpenVINO benchmarks on the Intel Core Ultra X7 358H integrated Arc GPU">

    <img width="800" src="https://cdn.ul.run/i/755a0515d5283c2e1227fe177d2d1981.avif" alt="YOLO26 OpenVINO benchmarks on the Intel Core Ultra X7 358H AI Boost NPU">

#### Earlier Intel Core Ultra Processors

??? abstract "Intel Core Ultra 7 258V (Lunar Lake)"

    === "Latency"

        Milliseconds per image, lower is better. The fastest column for each model is in bold.

        | Model   | PyTorch CPU | CPU FP32 | CPU FP16 | CPU INT8 | GPU FP32 | GPU FP16 | GPU INT8  | NPU FP32 | NPU FP16 | NPU INT8 |
        | ------- | ----------- | -------- | -------- | -------- | -------- | -------- | --------- | -------- | -------- | -------- |
        | YOLO26n | 27.33       | 16.84    | 16.73    | 8.62     | **3.33** | 3.34     | 3.46      | 23.71    | 23.70    | 23.96    |
        | YOLO26s | 60.20       | 54.71    | 54.45    | 20.19    | 4.66     | 4.65     | **3.99**  | 26.62    | 26.62    | 25.89    |
        | YOLO26m | 152.87      | 178.59   | 179.25   | 53.07    | 9.03     | 8.98     | **5.70**  | 32.94    | 33.01    | 30.95    |
        | YOLO26l | 201.57      | 238.15   | 233.70   | 67.73    | 11.32    | 11.36    | **7.45**  | 36.02    | 35.77    | 32.64    |
        | YOLO26x | 431.04      | 567.22   | 561.39   | 151.11   | 19.08    | 19.63    | **13.69** | 49.64    | 49.69    | 41.75    |

    === "Accuracy"

        mAP50-95 on COCO128, higher is better.

        | Model   | PyTorch CPU | CPU FP32 | CPU FP16 | CPU INT8 | GPU FP32 | GPU FP16 | GPU INT8 | NPU FP32 | NPU FP16 | NPU INT8 |
        | ------- | ----------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
        | YOLO26n | 0.4765      | 0.4734   | 0.4771   | 0.4720   | 0.4764   | 0.4764   | 0.4634   | 0.4724   | 0.4760   | 0.4615   |
        | YOLO26s | 0.5703      | 0.5632   | 0.5630   | 0.5513   | 0.5616   | 0.5616   | 0.5462   | 0.5618   | 0.5616   | 0.5497   |
        | YOLO26m | 0.6196      | 0.6191   | 0.6180   | 0.6046   | 0.6168   | 0.6168   | 0.6055   | 0.6165   | 0.6165   | 0.6042   |
        | YOLO26l | 0.6215      | 0.6206   | 0.6210   | 0.5974   | 0.6203   | 0.6203   | 0.5938   | 0.6200   | 0.6198   | 0.6009   |
        | YOLO26x | 0.6512      | 0.6552   | 0.6552   | 0.6365   | 0.6568   | 0.6568   | 0.6385   | 0.6563   | 0.6563   | 0.6399   |

    === "Charts"

        <img width="800" src="https://cdn.ul.run/i/070db9675dfee5a2662aa09f0ba42c33.avif" alt="YOLO26 OpenVINO benchmarks on the Intel Core Ultra 7 258V Lunar Lake CPU">

        <img width="800" src="https://cdn.ul.run/i/49405e2a81ee5cedcb04c0cb1751bed9.avif" alt="YOLO26 OpenVINO benchmarks on the Intel Core Ultra 7 258V integrated Arc GPU">

        <img width="800" src="https://cdn.ul.run/i/ed172cc0d4eff0c333c3cdd584600c3d.avif" alt="YOLO26 OpenVINO benchmarks on the Intel Core Ultra 7 258V AI Boost NPU">

??? abstract "Intel Core Ultra 7 155H (Meteor Lake)"

    === "Latency"

        Milliseconds per image, lower is better. The fastest column for each model is in bold.

        | Model   | PyTorch CPU | CPU FP32 | CPU FP16 | CPU INT8 | GPU FP32 | GPU FP16 | GPU INT8  |
        | ------- | ----------- | -------- | -------- | -------- | -------- | -------- | --------- |
        | YOLO26n | 33.19       | 13.43    | 13.20    | 11.40    | 9.13     | 9.07     | **5.79**  |
        | YOLO26s | 70.92       | 36.16    | 36.07    | 15.54    | 16.37    | 16.65    | **9.27**  |
        | YOLO26m | 192.45      | 102.52   | 101.88   | 35.72    | 34.47    | 35.71    | **15.46** |
        | YOLO26l | 239.19      | 129.24   | 128.50   | 45.38    | 43.24    | 43.40    | **19.72** |
        | YOLO26x | 511.89      | 293.40   | 296.48   | 85.24    | 79.49    | 79.45    | **34.66** |

    === "Accuracy"

        mAP50-95 on COCO128, higher is better.

        | Model   | PyTorch CPU | CPU FP32 | CPU FP16 | CPU INT8 | GPU FP32 | GPU FP16 | GPU INT8 |
        | ------- | ----------- | -------- | -------- | -------- | -------- | -------- | -------- |
        | YOLO26n | 0.4765      | 0.4734   | 0.4771   | 0.4611   | 0.4758   | 0.4758   | 0.4573   |
        | YOLO26s | 0.5703      | 0.5632   | 0.5630   | 0.5471   | 0.5616   | 0.5616   | 0.5450   |
        | YOLO26m | 0.6196      | 0.6191   | 0.6180   | 0.6025   | 0.6167   | 0.6167   | 0.6039   |
        | YOLO26l | 0.6215      | 0.6206   | 0.6210   | 0.5984   | 0.6210   | 0.6210   | 0.5999   |
        | YOLO26x | 0.6512      | 0.6552   | 0.6552   | 0.6418   | 0.6552   | 0.6552   | 0.6420   |

    === "Charts"

        <img width="800" src="https://cdn.ul.run/i/9fc9ef5ec0d44f4405819edff62c1672.avif" alt="YOLO26 OpenVINO benchmarks on the Intel Core Ultra 7 155H Meteor Lake CPU">

        <img width="800" src="https://cdn.ul.run/i/532d3bb375fcadaab3e1d13526f4638e.avif" alt="YOLO26 OpenVINO benchmarks on the Intel Core Ultra 7 155H integrated Arc GPU">

Across the three processors:

- **CPU**: INT8 ran 1.2 to 3.8 times faster than FP32 and 2.9 to 6.0 times faster than PyTorch. FP32 ran 1.6 to 2.5 times faster than PyTorch on the X7 358H and the 155H, but slower than PyTorch for YOLO26m, YOLO26l, and YOLO26x on the 258V.
- **GPU**: the integrated GPU ran FP32 models 3.6 to 29.1 times faster than PyTorch on the CPU. INT8 was faster than FP32 for every model except YOLO26n on the X7 358H and the 258V, by up to 2.3 times.
- **NPU** (X7 358H and 258V): FP32 models ran 1.2 to 16.2 times faster than PyTorch on the CPU, and INT8 ran 0.88 to 1.19 times as fast as FP32. These INT8 models were exported before `device=npu` selected symmetric activations, so the tables do not show the NPU-tuned INT8 export.
- **FP16** latency was within 5% of FP32 on every device, and INT8 lost 0.1 to 2.7 points of mAP50-95.

## About OpenVINO

OpenVINO, short for Open Visual Inference and [Neural Network](https://www.ultralytics.com/glossary/neural-network-nn) Optimization, converts models from frameworks such as [PyTorch](https://www.ultralytics.com/glossary/pytorch) into its IR format and runs them through one API on Intel CPUs, GPUs, and NPUs. For the full reference, see the [OpenVINO documentation](https://docs.openvino.ai/).

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/c3120a6b5d08e902d20cc1447249c1ac.avif" alt="OpenVINO Intel AI inference toolkit">
</p>

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rMllxg8ZLs8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Export Ultralytics YOLO26 to Intel OpenVINO Format for Faster Inference 🚀
</p>

## FAQ

### Does an OpenVINO model run on AMD or ARM CPUs?

Yes, on the CPU. The OpenVINO `CPU` device runs on x86-64 and ARM64 processors: the [server CPU benchmarks](#fp32-fp16-and-int8-on-a-server-cpu) on this page were measured on AMD EPYC processors, and the same export runs on Apple silicon. The `GPU` and `NPU` devices need Intel hardware, so `device=xpu` and `device=npu` fall back with a warning on other machines. For other targets, see the [model deployment options guide](../guides/model-deployment-options.md).

### Should I export an OpenVINO model in FP32, FP16, or INT8?

It depends on the device. FP16 (`quantize=16`) needs no calibration data, halves the model size, and kept FP32 accuracy and latency in the [benchmarks](#openvino-yolo26-benchmarks), so use it when the size of the model matters. INT8 (`quantize=8`) gave the lowest CPU latency for most models: on the server CPU it reduced YOLO26s latency from 5.18 ms to 4.51 ms per image, at 47.6 mAP<sup>val</sup> 50-95 against 48.5 in FP32, and on Intel Core Ultra CPUs it ran 1.2 to 3.8 times faster than FP32. On the integrated Intel GPUs the INT8 gain was smaller, up to 2.3 times, and on the NPUs the INT8 models of these runs, exported without `device=npu`, were about as fast as FP32. INT8 needs representative [calibration data](#int8-quantization), so validate the INT8 model on your own data before you choose it.

### How do I run batches or different image sizes with an OpenVINO model?

Export with `dynamic=True` and pass the batch and `imgsz` you need at prediction time. The image size must be a multiple of the model stride, 32 for the official YOLO26 checkpoints other than classification; see [Deploy](#deploy) for the exceptions.

!!! example "Dynamic export"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="openvino", dynamic=True)

        model = YOLO("yolo26n_openvino_model/")
        results = model(["https://ultralytics.com/images/bus.jpg"] * 4, imgsz=320)  # a batch of 4 images at 320 pixels
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=openvino dynamic=True
        yolo predict model=yolo26n_openvino_model source='https://ultralytics.com/images/bus.jpg' imgsz=320
        ```

### Should I use OpenVINO or ONNX on Intel hardware?

Use OpenVINO when the deployment hardware is Intel: `device=xpu` and `device=npu` select the Intel GPU and NPU, and INT8 OpenVINO models ran 1.2 to 3.8 times faster than FP32 on Intel Core Ultra CPUs in the [benchmarks](#intel-core-ultra-cpu-gpu-and-npu). Use [ONNX](onnx.md) when one file must also run on NVIDIA or AMD GPUs or in runtimes other than OpenVINO; on Windows, [Windows ML](onnx.md#windows-ml-inference) can select Intel hardware for an ONNX model.
