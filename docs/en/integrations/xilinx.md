---
comments: true
description: Export Ultralytics YOLO26 models to AMD Xilinx Versal AI Edge Series Gen 2 NPUs with format=xilinx. Covers AMD Quark INT8 quantization, target devices, output format, Vitis AI compilation, and Zynq, Kria and VEK280 flows.
keywords: YOLO26, AMD Xilinx, Xilinx export, format=xilinx, Vitis AI, AMD Quark, VINT8, Versal AI Edge Gen 2, VEK385, NPU, rai, VitisAIExecutionProvider, ONNX Runtime, INT8 quantization, Zynq UltraScale+, Kria KV260, DPU, VEK280, edge AI
---

# AMD Xilinx Export

AMD Versal AI Edge Series Gen 2 adaptive SoCs run neural networks on an NPU, which AMD's [Vitis AI](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html) toolchain compiles from an ONNX model. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="xilinx"` creates a `yolo26n_xilinx_model/` directory that holds an ONNX model quantized to INT8 with [AMD Quark](https://quark.docs.amd.com/latest/) and a Vitis AI compiler configuration for the device you select with the `name` argument. The Vitis AI compiler turns the two files into a `.rai` model that runs on the NPU.

## At a Glance

| Property          | Details                                                                                                                                                                          |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `xilinx` (aliases: `vitis`, `vitisai`, `versal`)                                                                                                                                 |
| Output            | `yolo26n_xilinx_model/`, a directory with `yolo26n.onnx` and [`vitisai_config.json`](#output-format)                                                                             |
| Runs on           | The NPU of an AMD Versal AI Edge Series Gen 2 device, such as the VEK385 evaluation kit, after [compilation to `.rai`](#deploy). The same directory runs with stock ONNX Runtime |
| Runtime           | ONNX Runtime with AMD's `VitisAIExecutionProvider` on the device, or stock ONNX Runtime elsewhere (CPU or CUDA)                                                                  |
| Precision         | INT8 only, with the model head kept in floating point. YOLO26n is 3.6 MB                                                                                                         |
| Export host       | Linux with Python 3.11 to 3.13 and a C++ compiler, tested on x86-64 Ubuntu 24.04. The export needs no AMD hardware, account, or license                                          |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to AMD Xilinx for the Versal AI Edge Series Gen 2 device you select, so you do not need a local Linux export environment.

## Quickstart

Install the Ultralytics package with the AMD Xilinx export dependencies on a Linux machine:

```bash
pip install "ultralytics[export-xilinx]"
```

This installs AMD Quark. With a plain `pip install ultralytics`, the first export installs AMD Quark, `onnx`, `onnxruntime`, and `onnxslim` and continues in the same run. The export has these requirements:

| Requirement   | Version                                                                                                                                                    |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python        | 3.11 to 3.13, the versions AMD Quark supports                                                                                                              |
| `amd-quark`   | 0.13 (`>=0.13.0,<0.14.0`). It installs `onnx` 1.21 or 1.22 with it                                                                                         |
| `onnxruntime` | Any build, installed by the first export if no ONNX Runtime package is present. Quark uses it to run the calibration images                                |
| C++ compiler  | `g++` or another `c++` on `PATH`. AMD Quark compiles its custom operators with it on the first export, see [Limitations](#limitations-and-troubleshooting) |

For other installation methods, see the [installation guide](../quickstart.md). Then export your model:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to AMD Xilinx format for the VEK385 (INT8, calibrated on the default dataset of the task)
        model.export(format="xilinx")  # creates 'yolo26n_xilinx_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to AMD Xilinx format for the VEK385
        yolo export model=yolo26n.pt format=xilinx # creates 'yolo26n_xilinx_model/'
        ```

Load the exported directory to run inference or validate its INT8 accuracy. Without AMD's ONNX Runtime build, both run the INT8 model with stock ONNX Runtime; on a host without a GPU, that is the CPU:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported AMD Xilinx model
        model = YOLO("yolo26n_xilinx_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported AMD Xilinx model
        yolo predict model=yolo26n_xilinx_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported AMD Xilinx model
        model = YOLO("yolo26n_xilinx_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported AMD Xilinx model
        yolo val model=yolo26n_xilinx_model data=coco8.yaml
        ```

## Supported Tasks

AMD Xilinx export supports all seven Ultralytics tasks: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with [YOLO26](../models/yolo26.md), the only family that ships those heads. The YOLO26n checkpoint of every task exported, predicted, and validated on the CPU of an x86-64 Linux host, and a [YOLO11](../models/yolo11.md) detection model did too. Whether every operator of a model runs on the NPU is decided by the Vitis AI compiler, see [Deploy](#deploy).

## Export Arguments

| Argument   | Type                      | Default    | Description                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------- | ------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`                     | `'xilinx'` | Target format for the exported model, defining compatibility with various deployment environments. The aliases `'vitis'`, `'vitisai'`, and `'versal'` export the same format.                                                                                                                                                                                                                  |
| `name`     | `str`                     | `None`     | Versal AI Edge Series Gen 2 device the model is compiled for, written to `vitisai_config.json`. Uses `'ve2-xc2ve3858'`, the VEK385 evaluation kit, when unset. See [Target Device](#target-device).                                                                                                                                                                                            |
| `imgsz`    | `int` or `tuple`          | `640`      | Image size of the model input, fixed in the exported model: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect and segment, `640` (ADE20K) or `1024` (Cityscapes) for semantic, `768` for depth, `224` for classify, `640` for pose, and `1024` for OBB. |
| `quantize` | `int` or `str`            | `None`     | Precision. INT8 is the only precision: unset, `8`, `'int8'`, and `'w8a8'` build INT8; `16`, `32`, `'w8a16'`, and `'w8a32'` raise an error. Replaces the deprecated `half`/`int8` flags.                                                                                                                                                                                                        |
| `data`     | `str`                     | `None`     | Dataset YAML used for INT8 calibration; classification instead takes a dataset directory or a built-in dataset name. Calibration uses the `val` split unless `split` is set. When unset, the export uses the default dataset of the model task, such as `coco128.yaml` for detection.                                                                                                          |
| `fraction` | `float`, `int`, or `list` | `1.0`      | Calibration subset: a ratio, an image count (integers above `1`), or a `[train, val, test]` list, from which the entry of the calibration split is used.                                                                                                                                                                                                                                       |
| `nms`      | `bool`, optional          | `None`     | Select raw output (`None`, default) or the NMS-free head (`False`). Embedded NMS (`True`) is not supported: the export logs a warning and writes the raw output.                                                                                                                                                                                                                               |
| `opset`    | `int`                     | `None`     | ONNX opset version of the exported model. Uses `17` when unset; other values are passed through.                                                                                                                                                                                                                                                                                               |
| `simplify` | `bool`                    | `True`     | Simplifies the ONNX graph with `onnxslim` before quantization.                                                                                                                                                                                                                                                                                                                                 |
| `device`   | `str`                     | `None`     | Device that traces the ONNX graph, `cpu` by default. Every export on this page ran on the CPU.                                                                                                                                                                                                                                                                                                 |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Target Device

The Vitis AI compiler builds a model for one Versal AI Edge Series Gen 2 device, and `name` selects it. The export writes the device into `vitisai_config.json` and checks it against this list, ignoring upper or lower case:

| `name`                    | Device                |
| ------------------------- | --------------------- |
| `ve2-xc2ve3858` (default) | VEK385 evaluation kit |
| `ve2-xc2ve3504`           | `xc2ve3504`           |
| `ve2-xc2ve3558`           | `xc2ve3558`           |
| `ve2-xc2ve3804`           | `xc2ve3804`           |
| `ve2-xc2ve3304`           | `xc2ve3304`           |
| `ve2-xc2ve3358`           | `xc2ve3358`           |

For another board, look up the device part number in the documentation of the board. The device does not change the quantized model, only `vitisai_config.json`.

### INT8 Quantization with AMD Quark

The export always quantizes to INT8 with AMD Quark's `VINT8` configuration, symmetric INT8 with power-of-two scales, and enables Quark's NPU option for convolutional networks. The nodes of the model head, the last layer of the network (79 nodes for YOLO26n detection), stay in floating point. AMD's [YOLOv8m tutorial](https://github.com/amd/Vitis-AI/tree/release/6.3/versal_2ve/examples/tutorials/yolov8m) similarly keeps its detection tail out of INT8 and runs it in BF16 on the NPU.

- **Calibration data**: the export runs the images of `data` through the model to choose the INT8 scales, preprocessed as for inference. It calibrates on the `val` split unless you set `split`, and `fraction` limits the number of images. Without `data`, it uses the small default dataset of the task, such as `coco128.yaml` with 128 images for detection, and warns below 300 images: `AMD Xilinx: >300 images recommended for INT8 calibration, found 128 images.`
- **No other precision**: `quantize=16` and `quantize=32` fail, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

Calibrate on images that represent your deployment data:

!!! example "INT8 calibration"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        model.export(format="xilinx", data="coco.yaml", fraction=0.1)  # creates 'yolo26n_xilinx_model/'
        ```

    === "CLI"

        ```bash
        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        yolo export model=yolo26n.pt format=xilinx data=coco.yaml fraction=0.1 # creates 'yolo26n_xilinx_model/'
        ```

On first use, `coco.yaml` downloads the full COCO dataset (20.3 GB, including the training images) even though calibration reads only the validation images.

## Output Format

The export writes one directory:

| File                  | Contents                                                                                                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yolo26n.onnx`        | The INT8 model with `QuantizeLinear` and `DequantizeLinear` nodes, opset 17, with the class names, task, image size, and export arguments stored as ONNX metadata |
| `vitisai_config.json` | The Vitis AI compiler configuration: the `VAIML` target and the device from `name`                                                                                |

The intermediate FP32 ONNX file is deleted after quantization. The model takes one input named `images` with shape `(1, 3, height, width)`: RGB, channels first, `float32`, scaled to 0-1. The batch size is always 1. Its outputs have the same names, shapes, and `float32` type as an [ONNX export](onnx.md#output-format) of the same model, except for semantic segmentation. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | Output with `nms=None`                                | Output with `nms=False`                             |
| -------- | --------------- | ----------------------------------------------------- | --------------------------------------------------- |
| Detect   | `640`           | `(1, 84, 8400)`                                       | `(1, 300, 6)`                                       |
| Segment  | `640`           | `(1, 116, 8400)` and `(1, 32, 160, 160)`              | `(1, 300, 38)` and `(1, 32, 160, 160)`              |
| Semantic | `1024`          | `(1, 19, 1024, 1024)`, `float32` score for each class | Same as `nms=None`                                  |
| Depth    | `768`           | `(1, 1, 768, 768)`, depth map                         | Same as `nms=None`                                  |
| Classify | `224`           | `(1, 1000)`, class probabilities                      | Same as `nms=None`                                  |
| Pose     | `640`           | `(1, 56, 8400)`                                       | `(1, 300, 57)`: 6 box values, then 17 keypoints x 3 |
| OBB      | `1024`          | `(1, 20, 21504)`                                      | `(1, 300, 7)`: 4 box values, score, class, angle    |

- **Raw output (`nms=None`)**: for each anchor the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task. Apply NMS yourself. See the [ONNX output format](onnx.md#output-format) for the layout of each task.
- **NMS-free output (`nms=False`)**: `max_det` rows of `x1, y1, x2, y2, score, class` in input pixels, followed by the mask coefficients or keypoints. OBB rows are `center x, center y, width, height, score, class, angle`. Available for models with an NMS-free head, such as YOLO26.
- **Semantic segmentation**: unlike the `uint8` class map of an ONNX export, the output holds a `float32` score for each class and pixel. Take the argmax over the class axis to get the class map.

## Deploy

`YOLO("yolo26n_xilinx_model")` loads the ONNX model from the directory with ONNX Runtime. When the installed ONNX Runtime offers the `VitisAIExecutionProvider`, as AMD's build does, and the directory holds `vitisai_config.json`, Ultralytics runs the model with that provider and passes the export directory as its cache directory and `yolo26n` as its cache key: where the Vitis AI compiler is installed, the first load compiles the model for the device in `vitisai_config.json` into that cache, and later loads reuse the compiled model. Otherwise it runs the model with stock ONNX Runtime like any ONNX model: with CUDA on a GPU when one is available, else on the CPU.

The steps to run the model on the NPU. Step 1 was run when `format="xilinx"` was added in `ultralytics` 8.4.173; steps 2 and 3 were not run, because they need a Versal AI Edge Series Gen 2 board:

1. **Compile on an x86-64 host.** Run AMD's [Vitis AI 6.3 Docker image for Versal AI Edge Gen 2](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/setup_and_installation/docker-setup.html), `amdih/vitis-ai:versal-2ve-release_v6.3_0924` on Docker Hub, with AMD's AIE-MLv2 Compiler license (`AIEMLv2build`) for the AI Engine compiler ([licensing](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/additional_information/license.html)). Copy the export directory into the container and create an ONNX Runtime session with the `VitisAIExecutionProvider`, as in AMD's [compilation guide](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/model_compilation/compiling.html), with the same options Ultralytics passes:

    ```python
    import onnxruntime as ort

    d = "yolo26n_xilinx_model"
    options = {"config_file": f"{d}/vitisai_config.json", "cache_dir": d, "cache_key": "yolo26n", "target": "VAIML"}
    ort.InferenceSession(f"{d}/yolo26n.onnx", providers=["VitisAIExecutionProvider"], provider_options=[options])
    ```

    With these options the compiled model is written into the export directory under the `yolo26n` cache key, as described in AMD's [compiled model output](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/model_compilation/output.html). For a YOLO26n export for `ve2-xc2ve3858`, the compiler wrote `yolo26n/yolo26n.rai` and placed 99.9% of the operators and of the compute on the NPU in a single partition, with only the input `QuantizeLinear` on the CPU. Check `yolo26n/final-vaiml-pass-summary.txt` for the placement of your model.

2. **Set up the board** with AMD's [board setup](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/setup_and_installation/board_setup.html) and [OSPI and SD card boot flow](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/setup_and_installation/board_setup_ospi_sd_eou.html), which flash AMD's Linux image to the board.
3. **Run on the board.** Copy the export directory, with the compiled model from step 1, to the board. AMD's [ONNX Runtime Python inference](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/model_deployment/setup-hw.html) guide runs it with the board image's ONNX Runtime and the same `InferenceSession` call as in step 1. Loading it on the board with `YOLO("yolo26n_xilinx_model")` was not verified: Ultralytics uses the `VitisAIExecutionProvider` when the installed ONNX Runtime offers it, and keeps an installed `onnxruntime-vitisai` package instead of installing stock `onnxruntime` next to it. According to AMD's [guide to running the compiled model](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/model_deployment/model-inference.html), the provider does not deploy the model on the NPU when it cannot find the compiled model, and a new export of the checkpoint deletes it, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

### Deployment Guides

- **[Vitis AI documentation for Versal AI Edge Gen 2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html)**: AMD's reference for compilation, the board image, and the runtime.
- **[Supported operators](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/additional_information/ops_support.html)**: the operators the NPU runs and the CPU fallback, with the [unsupported operators](https://vitisai.docs.amd.com/projects/gen2/en/latest/docs/additional_information/ops_unsupported.html) listed separately.
- **[AMD YOLOv8m tutorial](https://github.com/amd/Vitis-AI/tree/release/6.3/versal_2ve/examples/tutorials/yolov8m)**: AMD's end-to-end example on the VEK385, with accuracy and latency on the NPU.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare AMD Xilinx with the other export formats.

## Deploying on Other AMD Xilinx Targets

`format="xilinx"` builds models only for the NPU of Versal AI Edge Series Gen 2 devices. Other AMD Xilinx devices use other Vitis AI flows, which start from an Ultralytics PyTorch checkpoint or an [ONNX export](onnx.md):

- **Zynq UltraScale+ MPSoC and Kria (ZCU102, ZCU104, KV260, KR260)**: these devices run models on the DPU, an accelerator built in the programmable logic, through [Vitis AI 3.5](https://xilinx.github.io/Vitis-AI/3.5/html/index.html). AMD's flow quantizes the PyTorch model, compiles it into an `.xmodel` file for the DPU configuration of the board, and runs it with the VART runtime, see AMD's [model development workflow](https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-model-development.html). According to AMD's [list of supported operators](https://docs.amd.com/r/en-US/ug1414-vitis-ai/Currently-Supported-Operators), the DPU accelerates ReLU, ReLU6, LeakyReLU, Hard-Swish, and Hard-Sigmoid activations; SiLU, the default activation of YOLO models, is not in the list, and neither are the Softmax and the MatMul between two activations in the attention blocks of YOLO11 and YOLO26, which run on the CPU as separate subgraphs. [YOLOv8](../models/yolov8.md) has no attention blocks.
- **Versal AI Edge Series (VEK280)**: the first-generation Versal AI Edge NPU compiles a model into a snapshot directory with the Vitis AI 6.3 flow for this series, see AMD's [bring your own model](https://vitisai.docs.amd.com/projects/gen1/en/latest/docs/customization_opportunities/byom.html) guide.

To keep the convolutions of a YOLO26 model on the DPU, train it with Hard-Swish activations. Copy [`yolo26.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26.yaml) to `yolo26-hswish.yaml` and add one line under the parameters, which changes the default activation of every `Conv` block ([Model YAML Configuration Guide](../guides/model-yaml-config.md)):

```yaml
# Parameters
nc: 80 # number of classes
activation: nn.Hardswish() # default Conv activation, accelerated by the DPU
end2end: True # whether to use end-to-end mode
```

Then build the model, transfer the pretrained YOLO26 weights, and [fine-tune](../guides/finetuning-guide.md) it on your dataset:

```python
from ultralytics import YOLO

# Build YOLO26n with Hard-Swish activations; the 'n' in the name selects the nano scale
model = YOLO("yolo26n-hswish.yaml").load("yolo26n.pt")  # transfer pretrained weights

# Fine-tune so the network adapts to Hard-Swish
model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

Activations have no weights, so all 708 pretrained weight tensors transfer, and an ONNX export of the model contains 87 HardSwish operators. Replace `coco8.yaml` with your own [dataset](../datasets/index.md), and compare the accuracy with the SiLU model in [Val mode](../modes/val.md) before you deploy.

## Limitations and Troubleshooting

- **Linux only**: on macOS and Windows the export fails with `AMD Xilinx export is only supported on Linux.` before it builds the ONNX graph. Export on Linux or with [Ultralytics Platform](../platform/train/models.md#export-model). The exported directory also runs with stock ONNX Runtime, see [Quickstart](#quickstart).
- **Python version**: with Python 3.10 or 3.14 the export fails before it builds the ONNX graph, for example with `Python>=3.11,<3.14 is required, but Python==3.14.10 is currently installed`. Exports ran with Python 3.11, 3.12, and 3.13.
- **C++ compiler**: AMD Quark compiles its custom operators with the `c++` compiler on `PATH` on the first export. On Ubuntu and Debian, when no compiler is found, the export installs the `g++` package with `apt` and continues, which needs root or `sudo`, and logs `apt requirements: Ultralytics requirement ['g++'] not found, attempting AutoUpdate...`. With automatic installation turned off (`YOLO_AUTOINSTALL=False`), or on other distributions, install `g++` yourself; without it the export fails with `Stable-ABI ONNX custom_ops extension could not be loaded and JIT compilation failed.` and leaves an empty `yolo26n_xilinx_model/` directory.
- **No other precision**: `quantize=16` fails with `quantize=16 (FP16) is not supported; format='xilinx' supports quantize=8 (INT8).`, and `quantize=32` with the same message for FP32. Without `quantize`, the export logs `xilinx export requires INT8 quantization, enabling it.`
- **Fixed input shape**: `dynamic=True` fails with `argument 'dynamic' is not supported for format='xilinx'`, and `batch` above `1` with the same message for `batch`. Export again to change the image size.
- **Embedded NMS**: `nms=True` logs `format=xilinx does not support embedded NMS; exporting native outputs for external NMS.` and writes the raw output, also for classification models. Export YOLO26 models with `nms=False` for final detections from the NMS-free head, or keep the default raw output and apply NMS in your code.
- **NMS-free head on other models**: `nms=False` needs a model with an NMS-free head, such as YOLO26. For other models, for example YOLO11, the export warns `This model has no one-to-one head; using one-to-many outputs.` and writes the raw output, `(1, 84, 8400)` for YOLO11n detection.
- **Invalid `name`**: a device that is not in the [list](#target-device) fails with `Invalid AMD Xilinx device 'vek385'. Valid devices are (...)`, which lists the six device names. Use the device name, not the name of the board.
- **Few calibration images**: the default datasets of the tasks hold 4 to 128 images, and the export warns below 300, for example `AMD Xilinx: >300 images recommended for INT8 calibration, found 128 images.` Pass `data` with images from your deployment, see [INT8 Quantization](#int8-quantization-with-amd-quark).
- **INT8 accuracy**: the accuracy loss depends on the model and task. On the 458 images of DOTAv1 val at `imgsz=1024`, YOLO26n-obb calibrated on the default `dota128.yaml` scored 38.0 mAP<sup>val</sup> 50-95 against 46.4 for the PyTorch model (on the 4 images of DOTA8 val, 33.7 against 89.4). Validate the exported model on your own data before you deploy it.
- **Output directory**: every export of one checkpoint writes the same `yolo26n_xilinx_model/` directory, whatever its `name` or `imgsz`, and deletes the previous contents first, including a compiled model. Rename the directory to keep an export, and keep `_xilinx_model` at the end of its name, because Ultralytics recognizes the format by it: a copy named `yolo26n_vek385` failed with `is not a supported model format`, while `yolo26n_vek385_xilinx_model` loaded.
- **ONNX file next to the checkpoint**: the export writes its intermediate FP32 model as `yolo26n.onnx` next to `yolo26n.pt` and deletes it even if quantization fails. An [ONNX export](onnx.md) of the same checkpoint in that directory is overwritten and deleted, so move it first or export ONNX again afterwards.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='xilinx' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends`. Export a checkpoint trained without `quantize=8`.
- **ONNX opset**: other `opset` values are passed to the ONNX export unchanged, 13 and 20 exported and quantized, but the Vitis AI compiler was not run on them.
- **CPU speed**: on a host without the `VitisAIExecutionProvider`, the INT8 model runs on the CPU, where its speed says nothing about the speed on the NPU. Use the CPU only to check the accuracy.

## Benchmarks

No Versal AI Edge Gen 2 device was available for this page, so it lists no latency measured on the NPU, and CPU latency says nothing about the NPU. The tables compare the accuracy of the PyTorch models and of their AMD Xilinx exports, both run on the CPU of an x86-64 Linux server, which shows the accuracy cost of INT8 quantization before you compile for the NPU. The [YOLO26 model page](../models/yolo26.md) lists the accuracy and speed of the PyTorch models.

=== "Accuracy"

    mAP<sup>val</sup> 50-95 on COCO val2017 (5,000 images) at `imgsz=640` and `batch=1`, higher is better.

    | Model   | PyTorch | AMD Xilinx INT8 |
    | ------- | ------- | --------------- |
    | YOLO26n | 40.3    | 36.7            |
    | YOLO26s | 48.0    | 42.5            |
    | YOLO26m | 52.7    | 47.8            |
    | YOLO26l | 54.4    | 51.0            |
    | YOLO26x | 57.0    | 54.4            |

=== "Size"

    Size in MB (10^6 bytes): the `.pt` checkpoint and the `.onnx` file of the export directory. The smaller file for each model is in bold. The export log prints the size of the whole directory in MiB, `(3.4 MB)` for YOLO26n.

    | Model   | PyTorch | AMD Xilinx INT8 |
    | ------- | ------- | --------------- |
    | YOLO26n | 5.5     | **3.6**         |
    | YOLO26s | 20.4    | **11.7**        |
    | YOLO26m | 44.3    | **25.6**        |
    | YOLO26l | 53.2    | **30.3**        |
    | YOLO26x | 118.7   | **66.5**        |

INT8 lowered mAP<sup>val</sup> 50-95 by 2.6 to 5.5 points. Measured with `ultralytics` 8.4.215, AMD Quark 0.13, ONNX Runtime 1.32.0, and PyTorch 2.15.0 on the CPU of an AMD EPYC 9655 server, with `batch=1` and `rect=False` for both, so PyTorch also ran at 640x640. The INT8 models were calibrated on the default `coco128.yaml`; a second export of YOLO26n produced a file of the same size and the same mAP.

## About AMD Xilinx

AMD Xilinx adaptive SoCs combine Arm processors, programmable logic, and, on Versal AI Edge devices, AI Engines that run neural networks. Vitis AI is AMD's toolchain that quantizes, compiles, and runs models on them. For the device families and evaluation kits, see AMD's [adaptive SoCs and FPGAs](https://www.amd.com/en/products/adaptive-socs-and-fpgas.html).

## FAQ

### Do I need an AMD board or license to export a model?

No. The export runs on a Linux machine with Python 3.11 to 3.13 and needs no AMD hardware, account, or license, and the exported directory runs with stock ONNX Runtime to check its accuracy. Compiling the `.rai` model needs AMD's Vitis AI Docker image and an AI Engine compiler license, and running it on the NPU needs a Versal AI Edge Series Gen 2 board, see [Deploy](#deploy).

### Which `name` should I use for my device?

Use the Vitis AI device name of the Versal AI Edge Series Gen 2 part on your board, such as `ve2-xc2ve3858` for the VEK385 evaluation kit, the default. See [Target Device](#target-device) for the six supported names.

### Why does my exported model run on the CPU?

Ultralytics uses the `VitisAIExecutionProvider` only when the installed ONNX Runtime offers it, as AMD's build on a Versal AI Edge Series Gen 2 board does, and the export directory holds `vitisai_config.json`. With stock ONNX Runtime it runs the INT8 model like any ONNX model, on the CPU of a host without a GPU, which is how you check its accuracy before you deploy.

### Can I export YOLO for a Zynq UltraScale+ or Kria KV260 board?

Not with `format="xilinx"`, which builds models only for Versal AI Edge Series Gen 2 NPUs. Zynq UltraScale+ and Kria devices use the DPU flow of Vitis AI 3.5, see [Deploying on Other AMD Xilinx Targets](#deploying-on-other-amd-xilinx-targets).

### Is Xilinx part of AMD?

Yes. [AMD completed its acquisition of Xilinx in February 2022](https://www.amd.com/en/newsroom/press-releases/2022-2-14-amd-completes-acquisition-of-xilinx.html), and the Zynq, Kria, Versal, and Vitis products are sold under the AMD brand. Device part numbers keep the `XC` prefix, as in the Vitis AI device name `ve2-xc2ve3858`.
