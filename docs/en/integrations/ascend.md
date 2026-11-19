---
comments: true
description: Export Ultralytics YOLO26 models to Huawei Ascend .om offline models with format=ascend and the CANN ATC compiler. Covers CANN setup, SoC targets, FP16, output shapes, and the deployment steps for Atlas and Orange Pi AIpro devices.
keywords: YOLO26, Huawei Ascend, Ascend export, CANN, ATC, om model, offline model, Ascend310B4, Ascend310P3, Atlas 200I A2, Atlas 300I Pro, Orange Pi AIpro, ais_bench, NPU, FP16, edge AI
---

# Huawei Ascend Export

[Huawei Ascend](https://www.hiascend.com/) AI processors run models compiled into the `.om` offline model format by ATC, the model compiler of Huawei's CANN toolkit. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="ascend"` creates a `yolo26n_ascend_model/` directory that holds an FP16 `.om` file built for one Ascend SoC, which you select with the `name` argument.

## At a Glance

| Property          | Details                                                                                                                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `ascend` (aliases: `huawei`, `cann`, `om`)                                                                                                                                                  |
| Output            | `yolo26n_ascend_model/`, a directory with `yolo26n_Ascend310B4.om` and [`metadata.yaml`](#output-format)                                                                                    |
| Runs on           | The Ascend AI processor of one SoC per export, set with `name`: Ascend 310B (Atlas 200I A2, Orange Pi AIpro) or Ascend 310P (Atlas 300I Pro, Atlas 300V Pro), see [Target SoC](#target-soc) |
| Runtime           | The CANN runtime and the [`ais_bench`](#runtime-installation) Python package on the Ascend device                                                                                           |
| Precision         | FP16 only. YOLO26n for the Ascend 310B4 is 6.4 MB                                                                                                                                           |
| Export host       | Linux with the CANN toolkit and the operator package of the target SoC, tested on x86-64 Ubuntu 24.04. The export needs no Ascend hardware                                                  |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to Ascend for the SoC you select, so you do not need a local CANN installation.

## Quickstart

The export runs the CANN ATC compiler, which is not a Python package. Create a virtual environment on an x86-64 Linux machine and install the Ultralytics package together with the Python packages that ATC imports. The environment takes 6.4 to 6.6 GB, because `pip` installs the CUDA build of PyTorch. On Ubuntu, `python3 -m venv` needs the [`python3-venv` package](#limitations-and-troubleshooting):

```bash
python3 -m venv ascend-env && source ascend-env/bin/activate
pip install ultralytics attrs cloudpickle decorator ml-dtypes psutil scipy tornado absl-py
```

Then download the CANN toolkit and the operator package of your SoC family from Huawei's [CANN download page](https://www.hiascend.com/developer/download/community/result?module=cann): `Ascend-cann-310b-ops` for the Ascend 310B or `Ascend-cann-310p-ops` for the Ascend 310P. Install the toolkit and that one operator package into one directory with the environment active, and load the CANN environment in the shell you export from. The installers take 2.3 GB with the 310B package and 3.7 GB with the 310P package, and the installation 6.9 GB and 10.7 GB:

```bash
chmod +x Ascend-cann-*.run
./Ascend-cann-toolkit_9.2.1_linux-x86_64.run --install --install-path="$HOME/cann" --quiet
./Ascend-cann-310b-ops_9.2.1_linux-x86_64.run --install --install-path="$HOME/cann" --quiet
source "$HOME/cann/cann/set_env.sh" # puts atc on PATH
```

The export has these version requirements:

| Requirement                       | Version                                                                                                                                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CANN                              | Tested with 9.2.1 and 8.3.RC1. With 8.3.RC1 the operator packages are named `Ascend-cann-kernels-310b` and `Ascend-cann-kernels-310p`, both can share one directory, and the environment script is `ascend-toolkit/set_env.sh` |
| Python                            | Exports ran with Python 3.11 and 3.12 on both CANN versions. The CANN 8.3.RC1 installer requires Python 3.7 to 3.11                                                                                                            |
| Python packages of ATC            | `attrs`, `cloudpickle`, `decorator`, `ml-dtypes`, `psutil`, `scipy`, `tornado`, and `absl-py`, in the environment you export from                                                                                              |
| `onnx`, `onnxslim`, `onnxruntime` | Installed by the first export if they are missing, unless automatic installation is turned off. The export continues in the same run                                                                                           |

For other installation methods, see the [installation guide](../quickstart.md). Then export your model:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to Ascend format for an Ascend 310B4 SoC (Atlas 200I A2, Orange Pi AIpro 8T)
        model.export(format="ascend", name="Ascend310B4")  # creates 'yolo26n_ascend_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to Ascend format for an Ascend 310B4 SoC
        yolo export model=yolo26n.pt format=ascend name=Ascend310B4 # creates 'yolo26n_ascend_model/'
        ```

Copy the `yolo26n_ascend_model/` directory to the Ascend device. Predict and Validate run only there, with the [CANN runtime](#runtime-installation). They were not run on an Ascend device for this page:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported Ascend model on the Ascend device
        model = YOLO("yolo26n_ascend_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported Ascend model on the Ascend device
        yolo predict model=yolo26n_ascend_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported Ascend model on the Ascend device
        model = YOLO("yolo26n_ascend_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported Ascend model on the Ascend device
        yolo val model=yolo26n_ascend_model data=coco8.yaml
        ```

## Supported Tasks

Ascend export supports all seven Ultralytics tasks: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with [YOLO26](../models/yolo26.md), the only family that ships those heads. The YOLO26n checkpoint of every task compiled for the Ascend 310B4 and the Ascend 310P3, and a [YOLO11](../models/yolo11.md) detection model compiled too. Other model families were not tested.

## Export Arguments

| Argument   | Type             | Default         | Description                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------- | ---------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'ascend'`      | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                        |
| `name`     | `str`            | `'Ascend310B4'` | Ascend SoC the model is built for, passed to ATC as `--soc_version`. See [Target SoC](#target-soc).                                                                                                                                                                                                                                                                                                                                       |
| `imgsz`    | `int` or `tuple` | `640`           | Image size of the model input, fixed in the `.om` file: an integer for square images or a tuple `(height, width)`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect, segment, and pose models and for semantic models pretrained on ADE20K, `1024` for OBB models and for semantic models pretrained on Cityscapes, `768` for depth, and `224` for classify. |
| `quantize` | `int` or `str`   | `None`          | Precision. Unset and `16` build FP16, the only precision; `32` and `8` raise an error, see [Precision](#precision). Replaces the deprecated `half`/`int8` flags.                                                                                                                                                                                                                                                                          |
| `batch`    | `int`            | `1`             | Batch size of the exported model, fixed in the input shape.                                                                                                                                                                                                                                                                                                                                                                               |
| `nms`      | `bool`, optional | `None`          | Select raw output (`None`, default) or the NMS-free head (`False`). Embedded NMS (`True`) fails for detect, segment, and pose models for the Ascend 310B4, and where it compiled, the model was not run on a device, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).                                                                                                                                             |
| `opset`    | `int`            | `None`          | ONNX opset version of the intermediate ONNX graph. Uses `17` when unset, and reduces higher values to `17`.                                                                                                                                                                                                                                                                                                                               |
| `simplify` | `bool`           | `True`          | Simplifies the intermediate ONNX graph with `onnxslim`.                                                                                                                                                                                                                                                                                                                                                                                   |
| `device`   | `str`            | `None`          | Device that traces the intermediate ONNX graph, `cpu` by default. ATC compiles on the CPU.                                                                                                                                                                                                                                                                                                                                                |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Target SoC

ATC builds a `.om` file for one SoC, and `name` selects it, so export once per SoC. Ultralytics checks only that `name` starts with `Ascend`, upper case, and ATC rejects a SoC it does not know. The sizes are for a YOLO26n detection model exported at `imgsz=640` with CANN 9.2.1, in a directory with the toolkit and the one operator package of that SoC family:

| `name`        | Devices                                    | Operator package       | YOLO26n file             | Size   |
| ------------- | ------------------------------------------ | ---------------------- | ------------------------ | ------ |
| `Ascend310B4` | Atlas 200I A2 and Orange Pi AIpro, 8 TOPS  | `Ascend-cann-310b-ops` | `yolo26n_Ascend310B4.om` | 6.4 MB |
| `Ascend310B1` | Atlas 200I A2 and Orange Pi AIpro, 20 TOPS | `Ascend-cann-310b-ops` | `yolo26n_Ascend310B1.om` | 6.4 MB |
| `Ascend310P3` | Atlas 300I Pro, Atlas 300V Pro             | `Ascend-cann-310p-ops` | `yolo26n_Ascend310P3.om` | 6.5 MB |
| `Ascend310P1` | -                                          | `Ascend-cann-310p-ops` | `yolo26n_Ascend310P1.om` | 6.5 MB |

For a device not listed here, look up the SoC name in the documentation of the device. [Ultralytics Platform](../platform/train/models.md#export-model) exports for the same four targets.

### Precision

Ultralytics compiles every Ascend model with the ATC option `--precision_mode=force_fp16`. There is no FP32 or INT8 export, and the export needs no calibration data:

| `quantize`    | Result                                                                                         |
| ------------- | ---------------------------------------------------------------------------------------------- |
| unset or `16` | FP16 `.om` file. YOLO26n for the Ascend 310B4 is 6.4 MB                                        |
| `32`          | Fails with `quantize=32 (FP32) is not supported; format='ascend' supports quantize=16 (FP16).` |
| `8`           | Fails with `quantize=8 (INT8) is not supported; format='ascend' supports quantize=16 (FP16).`  |

The input and the outputs keep the `float32` type of the ONNX model, except the semantic class map, so inference code passes the same arrays as to an [ONNX export](onnx.md#output-format).

## Output Format

The export writes one directory:

| File                     | Contents                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `yolo26n_Ascend310B4.om` | The compiled model, with the SoC set by `name` in the file name                                                          |
| `metadata.yaml`          | Class names, task, image size, batch size, stride, and export arguments, which Ultralytics reads when it loads the model |

The intermediate `yolo26n.onnx` file stays next to the directory. The `.om` model takes one input named `images` with shape `(batch, 3, height, width)`: RGB, channels first, `float32`, scaled to 0-1. Its outputs had the same shapes and types as the outputs of the intermediate ONNX model in every export on this page. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | Output with `nms=None`                                | Output with `nms=False`                             |
| -------- | --------------- | ----------------------------------------------------- | --------------------------------------------------- |
| Detect   | `640`           | `(1, 84, 8400)`                                       | `(1, 300, 6)`                                       |
| Segment  | `640`           | `(1, 116, 8400)` and `(1, 32, 160, 160)`              | `(1, 300, 38)` and `(1, 32, 160, 160)`              |
| Semantic | `1024`          | `(1, 1024, 1024)`, `uint8` class index for each pixel | Same as `nms=None`                                  |
| Depth    | `768`           | `(1, 1, 768, 768)`, depth map                         | Same as `nms=None`                                  |
| Classify | `224`           | `(1, 1000)`, class probabilities                      | Same as `nms=None`                                  |
| Pose     | `640`           | `(1, 56, 8400)`                                       | `(1, 300, 57)`: 6 box values, then 17 keypoints x 3 |
| OBB      | `1024`          | `(1, 20, 21504)`                                      | `(1, 300, 7)`: 4 box values, score, class, angle    |

- **Raw output (`nms=None`)**: for each anchor the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS yourself. See the [ONNX output format](onnx.md#output-format) for the layout of each task.
- **NMS-free output (`nms=False`)**: `max_det` (default 300) rows of `x1, y1, x2, y2, score, class` in input pixels, followed by the mask coefficients or keypoints. OBB rows are `center x, center y, width, height, score, class, angle`. Filter the rows by score. Available for models with an NMS-free head, such as YOLO26 and YOLOv10.
- **Semantic segmentation**: the output is the class index of every pixel as `uint8` (`int32` for models with more than 256 classes), not a score for each class.
- **Segmentation**: the second output holds the mask prototypes.

`YOLO("yolo26n_ascend_model")` reads the layout from `metadata.yaml`, so you only need these details when you run the `.om` file in your own code. The keys of `metadata.yaml` include `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, `args`, and, for pose models, `kpt_shape`.

## Deploy

On the Ascend device, install the package with `pip install ultralytics`, install the [runtime](#runtime-installation), and copy the `yolo26n_ascend_model/` directory to it. `YOLO("yolo26n_ascend_model")` then runs the `.om` file on the NPU through `ais_bench`. Without `ais_bench`, loading fails with ``Ascend inference requires the CANN runtime and `ais_bench` Python package.``

- **Matching SoC**: export with the `name` of the SoC on your device.
- **NPU index**: Ultralytics runs the model on NPU 0. `device="npu:1"` selects NPU 1 and needs the [`torch_npu`](https://github.com/Ascend/pytorch) package, which Ultralytics uses to parse NPU device names.
- **Fixed input shape**: the model always runs at its export `imgsz` and `batch`, which Ultralytics reads from `metadata.yaml`. An `imgsz` passed to `predict` or `val` is ignored. Larger batches are split into chunks of the export batch size and smaller ones are padded, so any number of images works.
- **One `.om` file per directory**: Ultralytics loads whichever `.om` file it finds first in the directory and its subdirectories, so keep only one there.

### Runtime Installation

Inference needs the CANN runtime on the device and the `ais_bench` Python package, which wraps CANN's Python API. Build and install `ais_bench` from Huawei's [tools repository](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench). These steps were not run on an Ascend device for this page:

```bash
# On the Ascend device, with CANN installed and its environment loaded
git clone https://gitee.com/ascend/tools.git
cd tools/ais-bench_workload/tool/ais_bench
pip3 wheel ./backend -v
pip3 install ./aclruntime-*.whl
pip3 wheel ./ -v
pip3 install ./ais_bench-*.whl
```

### Deployment Guides

- **[CANN documentation](https://www.hiascend.com/cann)**: Huawei's reference for the CANN toolkit, ATC, and the runtime.
- **[ais_bench](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)**: the inference tool and Python package that Ultralytics uses to run `.om` models.
- **[`ascendai/cann` container images](https://hub.docker.com/r/ascendai/cann)**: CANN toolkit images with the 310P, 910, and 950 operators, but none with the 310B operators.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare Ascend with the other export formats.

## Train on Ascend NPUs

Ultralytics also trains and validates PyTorch models on Ascend NPUs through [`torch_npu`](https://github.com/Ascend/pytorch): pass `device=npu:0` for one NPU or `device=npu:0,1` for distributed training over HCCL. This uses the PyTorch model and not an exported `.om` file. See [Huawei Ascend NPU Training](../modes/train.md#huawei-ascend-npu-training) for the versions to install and the supported features.

## Limitations and Troubleshooting

- **No `venv` module**: on Ubuntu 24.04, `python3 -m venv` fails with `The virtual environment was not created successfully because ensurepip is not available.` until you install the package it names, `python3.12-venv` for Python 3.12.
- **Installer not executable**: a downloaded `.run` installer fails with `Permission denied` until you make it executable with `chmod +x`, as in the [Quickstart](#quickstart).
- **`atc` not found**: the export fails with `Ascend export requires the CANN toolkit 'atc' compiler, which was not found on PATH.` before it builds the ONNX graph. Run `source "$HOME/cann/cann/set_env.sh"` with your installation directory in the shell you export from.
- **Python packages of ATC**: without them the export fails in ATC with `No module named 'decorator'`. Install the packages from the [Quickstart](#quickstart) into the environment you export from. A plain `pip install ultralytics` does not include them.
- **CANN installer and pip**: the CANN 9.2.1 installer installs Python wheels with pip and fails with `python3 -m pip or pip3 is not installed` when the `python3` on `PATH` has no pip. Run it with your virtual environment active.
- **Several SoC families in one directory**: with CANN 9.2.1, after the 310P operator package was installed into the directory of the 310B operator package, every 310B export failed, and in a directory with only one of the two packages, exports for the other family failed: `No supported Ops kernel and engine are found for [/model.1/conv/Conv], optype [Conv2D].` The layer name in the message varies. Install the toolkit and one operator package per directory, and load the `set_env.sh` of the directory that matches `name`.
- **CANN 8.3.RC1 without an Ascend driver**: on a machine without the Ascend driver, ATC of CANN 8.3.RC1 fails with `error while loading shared libraries: libascend_hal.so`, and then with `Invalid_Environment_Variable(E40001)` for `python3-config --prefix` when `python3-config` is missing from `PATH` or belongs to a different Python than the one that runs the export. Add `$ASCEND_TOOLKIT_HOME/x86_64-linux/devlib` to `LD_LIBRARY_PATH` and the `bin` directory of the Python that has `python3-config` to `PATH`. CANN 9.2.1 needed neither: its `set_env.sh` adds the `devlib` directory itself, and its exports ran without `python3-config`. The 8.3.RC1 installer also requires Python 3.7 to 3.11: `There is no python3.7,python3.8,python3.9,python3.10,python3.11 in the current environment !`
- **Invalid `name`**: a `name` that does not start with `Ascend` fails with `Invalid Ascend SoC name='310B4'. Expected a CANN --soc_version such as 'Ascend310P3' or 'Ascend310B4'.`, also for `ascend310b4` in lower case. A SoC that ATC does not know fails in ATC, with `The input SoC version is not supported.` on CANN 9.2.1 and `Soc version [Ascend999] is invalid` on CANN 8.3.RC1. Without `name`, the export logs `Huawei Ascend export requires a missing 'name' arg for the target SoC. Using default name='Ascend310B4'.`
- **Embedded NMS**: with `nms=True`, ATC fails for the Ascend 310B4 in the `NonMaxSuppressionV6Fusion` pass with `The shape of score cannot be negative.`, for YOLO26n detect, segment, and pose models on CANN 8.3.RC1 and for a detect model on CANN 9.2.1. For YOLO26n detect, segment, pose, and OBB models for the Ascend 310P3, and an OBB model for the Ascend 310B4 on CANN 8.3.RC1, ATC wrote a file whose name ends in `_linux_x86_64.om` (such as `yolo26n_Ascend310P3_linux_x86_64.om`) of 95 to 149 MB, against about 7 MB without NMS. These files were not run on a device. Export with `nms=False` for final detections, or with the default raw output and NMS in your code. Classification models raise `'nms=True' is not valid for classification models.`, and semantic segmentation and depth models log a warning and export without NMS.
- **NMS-free head on other models**: `nms=False` needs a model with an NMS-free head, such as YOLO26 or YOLOv10. For other models, for example YOLO11, the export warns `This model has no one-to-one head; using one-to-many outputs.` and writes the raw output, `(1, 84, 8400)` for YOLO11n detection.
- **Fixed input shape**: `dynamic=True` fails with `argument 'dynamic' is not supported for format='ascend'`. Export again to change the image size or the batch size.
- **ONNX opset**: an `opset` above 17 logs `Ascend: the CANN ONNX parser requires opset<=17, setting opset=17.`
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='ascend' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends`.
- **Output directory**: every export of one checkpoint writes the same `yolo26n_ascend_model/` directory, whatever its `name`, `imgsz`, `batch`, or `nms`, and deletes the previous contents first. Rename the directory before you export for a second SoC, and keep `_ascend_model` in its name, for example `yolo26n_310p3_ascend_model/`, because Ultralytics recognizes the format by it: a copy named `yolo26n_310b4` failed with `is not a supported model format`.
- **Building `ais_bench` with GCC 13**: on an x86-64 Ubuntu 24.04 machine with GCC 13.3 and CANN 9.2.1, `pip3 wheel ./backend -v` from the [Runtime Installation](#runtime-installation) steps failed to compile `aclruntime` with `‘uint64_t’ has not been declared`. The build was not tried on an Ascend device.
- **Inference on the export host**: without `ais_bench`, `YOLO("yolo26n_ascend_model")` fails with ``Ascend inference requires the CANN runtime and `ais_bench` Python package.`` On the export host you can validate the PyTorch checkpoint or the intermediate `yolo26n.onnx` file with [ONNX Runtime](onnx.md), which does not check the FP16 computation of the `.om` file.

## Benchmarks

No Ascend device was available for this page, so it lists no latency or accuracy measured on the NPU. The [YOLO26 model page](../models/yolo26.md) lists the accuracy of the PyTorch checkpoints.

The sizes of the exported YOLO26 detection models, as the `.om` file in MB (10^6 bytes) at `imgsz=640`. The export log prints the size of the whole directory in MiB as `(6.1 MB)` for YOLO26n for the Ascend 310B4:

| Model   | PyTorch | Ascend 310B4 | Ascend 310P3 |
| ------- | ------- | ------------ | ------------ |
| YOLO26n | 5.5     | 6.4          | 6.5          |
| YOLO26s | 20.4    | 20.8         | 20.9         |
| YOLO26m | 44.3    | 42.9         | 43.1         |
| YOLO26l | 53.2    | 52.2         | 52.3         |
| YOLO26x | 118.7   | 114.5        | 114.6        |

Exported with `ultralytics` 8.4.215 and CANN 9.2.1 on an x86-64 Linux server without Ascend hardware. CANN 8.3.RC1 produced sizes within 1% of these.

## About Huawei Ascend

Ascend is Huawei's family of AI processors, and CANN (Compute Architecture for Neural Networks) is the software stack around them: the ATC compiler that converts ONNX models into `.om` offline models, and the runtime that executes them. For the full reference, see the [CANN documentation](https://www.hiascend.com/cann).

## FAQ

### Do I need an Ascend device to export a model?

No. ATC compiles on the CPU, and every export on this page ran on an x86-64 Linux server without Ascend hardware. You need the device, with the CANN runtime and `ais_bench`, only to run or validate the `.om` file.

### Which `name` should I use for my device?

Use the name of the SoC, not of the board: `Ascend310B4` for an Atlas 200I A2 or Orange Pi AIpro with 8 TOPS, `Ascend310B1` for the 20 TOPS versions, and `Ascend310P3` for an Atlas 300I Pro or Atlas 300V Pro card. See [Target SoC](#target-soc). ATC builds the model for that SoC only, so export once per SoC.

### Which CANN version should I install?

CANN 9.2.1 needed the fewest steps on an export host without an Ascend driver: install the toolkit and one operator package, load its `set_env.sh`, and export. CANN 8.3.RC1 also worked, with the extra environment settings in [Limitations and Troubleshooting](#limitations-and-troubleshooting).

### Can I embed NMS in an Ascend model?

Not with the results on this page: for the Ascend 310B4, an export with `nms=True` of a detect, segment, or pose model fails in ATC, and for the Ascend 310P3 it compiled files of 95 to 149 MB, against about 7 MB without NMS, that were not run on a device. Export YOLO26 models with `nms=False` to get final detections from the NMS-free head without an NMS step, or keep the default raw output and apply NMS in your code. See [Limitations and Troubleshooting](#limitations-and-troubleshooting).
