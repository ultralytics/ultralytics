---
comments: true
description: Export Ultralytics YOLO models to Axelera AI Metis hardware with format=axelera. Covers the Voyager SDK compiler, INT8 calibration, output files, driver and runtime setup, and Metis FPS benchmarks.
keywords: Axelera AI, Metis AIPU, Voyager SDK, Edge AI, YOLOv8, YOLO11, YOLO26, Model Export, Computer Vision, PCIe, M.2, Object Detection, quantization, axelera export, axm, INT8 calibration, axelera-devkit, axelera-rt
---

# Axelera AI Export and Deployment

[Axelera AI](https://axelera.ai/) Metis® AIPUs run models compiled into the `.axm` format by the compiler of the Voyager SDK. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="axelera"` creates a `yolo26n_axelera_model/` directory that holds an INT8 `.axm` file. The export runs on a Linux computer and needs no Metis device; prediction and validation run on Metis hardware.

## At a Glance

| Property          | Details                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `format` argument | `axelera`                                                                                                                                  |
| Output            | `yolo26n_axelera_model/`, a directory with `yolo26n.axm`, `compiler_config_final.toml`, and [`metadata.yaml`](#output-format)              |
| Runs on           | Axelera Metis AIPUs: Metis PCIe and M.2 cards and the systems built on them, see [Metis Hardware](#metis-hardware)                         |
| Runtime           | Voyager SDK 1.9.0 on the device: the `axelera-rt` Python package and the [`metis-dkms`](#driver-installation) kernel driver 1.7.0 or newer |
| Precision         | INT8 only, calibrated on images from `data`. YOLO26n is 21.7 MB                                                                            |
| Export host       | Linux x86-64 with Python 3.10 to 3.13 and PyTorch 2.8 to 2.14. Not macOS or Windows. The export needs no GPU and no Metis device           |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to Axelera, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package and the Voyager SDK compiler in a virtual environment on an x86-64 Linux machine:

```bash
python3 -m venv axelera-env && source axelera-env/bin/activate
pip install ultralytics axelera-devkit==1.9.0 --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple
```

The `pip` command installs Ultralytics from PyPI and `axelera-devkit` with its compiler packages from the package index of Axelera, and with them `onnx`, `onnxruntime`, and `onnxslim`. The compiler supports PyTorch up to 2.14, so pip installed PyTorch 2.14.1; the export itself then installs nothing. The Voyager SDK packages are governed by the [Axelera AI End User License Agreement](https://www.axelera.ai/eula).

After a plain `pip install ultralytics`, the first export installs `axelera-devkit` itself, see [Limitations and Troubleshooting](#limitations-and-troubleshooting) for what happens when that replaces PyTorch. The export has these requirements:

| Requirement      | Version                                                                                                                                                                                                                                               |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python           | 3.10 to 3.13. Exports ran with every version from 3.10 to 3.13. On Python 3.9 and 3.14, pip could not install `axelera-devkit` 1.9.0 and its compiler packages                                                                                        |
| Operating system | Linux. Exports for this page ran on x86-64 Ubuntu 24.04; the check in Ultralytics also lets ARM64 Linux outside Docker through, which was not run. The driver repository in [Driver Installation](#driver-installation) covers Ubuntu 22.04 and 24.04 |
| PyTorch          | 2.8 or newer for the Axelera export of Ultralytics, and 2.14 or older for the compiler                                                                                                                                                                |
| `axelera-devkit` | 1.9.0, the version Ultralytics pins                                                                                                                                                                                                                   |
| System library   | `libgl1` for OpenCV on a headless machine: `sudo apt install libgl1`                                                                                                                                                                                  |

To build the Python 3.13 environment that the Ultralytics CI uses for its Axelera export, run the environment builder from the root of a repository checkout. It requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and an installed Ultralytics checkout:

```bash
ULTRALYTICS_ISOLATED_VENVS="$PWD/.venvs" python .github/scripts/create-export-env.py --env isolated-axelera
```

For other installation methods, see the [installation guide](../quickstart.md). Then export your model:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to Axelera format with INT8 quantization
        model.export(format="axelera")  # creates 'yolo26n_axelera_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to Axelera format with INT8 quantization
        yolo export model=yolo26n.pt format=axelera # creates 'yolo26n_axelera_model/'
        ```

Without `data`, the export calibrates on the default dataset of the task, `coco128.yaml` for detection, which it downloads on first use, see [Calibration](#calibration). On a shared AMD EPYC 9655 server, with no GPU visible to the export, exports of YOLO26n took 151 to 172 s with the compiler installed, as reported in the export log, and the YOLO26x export took 1331 s.

Copy the `yolo26n_axelera_model/` directory to the machine with the Metis device. Predict and Validate run only there, with the [driver and runtime](#deploy) installed. They were not run on a Metis device for this page:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported Axelera model on the machine with the Metis device
        model = YOLO("yolo26n_axelera_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported Axelera model on the machine with the Metis device
        yolo predict model=yolo26n_axelera_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported Axelera model on the machine with the Metis device
        model = YOLO("yolo26n_axelera_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported Axelera model on the machine with the Metis device
        yolo val model=yolo26n_axelera_model data=coco8.yaml
        ```

## Supported Tasks

Axelera export supports all seven Ultralytics tasks: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with [YOLO26](../models/yolo26.md), the only family that ships those heads. The YOLO26n checkpoint of every task compiled for this page, and so did the [YOLO11n](../models/yolo11.md) and [YOLOv8n](../models/yolov8.md) detection models.

YOLO26 detection, instance segmentation, pose and OBB models export their [NMS-free](../guides/end2end-detection.md) one-to-one head by default, and the Axelera runtime decodes it on the host. Passing `nms=True` exports the one-to-many head instead, which scores lower on Metis: YOLO26n detection reaches mAP<sub>50-95</sub> 0.212 on [COCO128](../datasets/detect/coco128.md) that way, against 0.453 with the NMS-free head. See [Output Format](#output-format) for what each value of `nms` writes.

## Export Arguments

| Argument   | Type                      | Default     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------- | ------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`                     | `'axelera'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                                             |
| `imgsz`    | `int` or `tuple`          | `640`       | Image size of the model input, fixed in the `.axm` file: an integer for square images or a tuple `(height, width)`, each at least `64`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect, segment, and pose models and for semantic models pretrained on ADE20K, `1024` for OBB models and for semantic models pretrained on Cityscapes, `768` for depth, and `224` for classify. |
| `quantize` | `int` or `str`            | `None`      | Precision. Unset and `8` build INT8, the only precision; any other value, such as `16`, `32`, or `'w8a16'`, raises an error, see [Precision](#precision). Replaces the deprecated `half`/`int8` flags.                                                                                                                                                                                                                                                         |
| `data`     | `str`                     | `None`      | Dataset YAML with the calibration images; classification instead takes a dataset directory or a built-in dataset name. If omitted, Ultralytics uses the calibration dataset of the task, such as `coco128.yaml` for detection. See [Calibration](#calibration).                                                                                                                                                                                                |
| `fraction` | `float`, `int`, or `list` | `1.0`       | Calibration subset: a ratio, an image count (integers above `1`), or a `[train, val, test]` list, from which the entry of the calibration split is used.                                                                                                                                                                                                                                                                                                       |
| `split`    | `str`                     | `'val'`     | Dataset split that the calibration images come from.                                                                                                                                                                                                                                                                                                                                                                                                           |
| `batch`    | `int`                     | `1`         | Keep the default. The compiler builds models for one image, and an export with `batch=2` fails, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). To run several images at once, pass `batch` to `predict` instead, see [Maximum Performance](#maximum-performance).                                                                                                                                                                    |
| `device`   | `str`                     | `None`      | Device that Ultralytics prepares the model on, `cpu` by default. Every export on this page ran on the CPU.                                                                                                                                                                                                                                                                                                                                                     |

`nms` has no row because it is not among the Axelera export arguments of Ultralytics: it selects the head that is exported, as described in [Output Format](#output-format). For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Precision

The Voyager SDK compiler quantizes every model to INT8 for the Metis AIPU. There is no FP32 or FP16 export:

| `quantize`   | Result                                                                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| unset or `8` | INT8 `.axm` file. YOLO26n is 21.7 MB. Without `quantize`, the export logs `axelera export requires INT8 quantization, enabling it.` |
| `16`         | Fails with `quantize=16 (FP16) is not supported; format='axelera' supports quantize=8 (INT8).`                                      |
| `32`         | Fails with `quantize=32 (FP32) is not supported; format='axelera' supports quantize=8 (INT8).`                                      |
| `'w8a16'`    | Fails with `quantize='w8a16' is not supported; format='axelera' supports quantize=8 (INT8).`                                        |

### Calibration

The compiler calibrates the activation ranges on the images of the `val` split of `data`, or of the split you set with `split`, which Ultralytics passes to it one image at a time. Without `data`, each task uses its own default dataset:

| Task     | Default `data`     | Images in the calibration split |
| -------- | ------------------ | ------------------------------- |
| Detect   | `coco128.yaml`     | 100 or more, no warning         |
| Segment  | `coco128-seg.yaml` | 100 or more, no warning         |
| Semantic | `cityscapes8.yaml` | 4, with the calibration warning |
| Depth    | `depth8.yaml`      | 4, with the calibration warning |
| Classify | `imagenet100`      | 100 or more, no warning         |
| Pose     | `coco8-pose.yaml`  | 4, with the calibration warning |
| OBB      | `dota128.yaml`     | 100 or more, no warning         |

Below 100 images the export warns `>100 images required for Axelera calibration, found 4 images.` and continues. Pass `data` with images that look like your deployment data, and use `fraction` to limit how many of them are read:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        model.export(format="axelera", data="coco.yaml", fraction=0.1)  # creates 'yolo26n_axelera_model/'
        ```

    === "CLI"

        ```bash
        # Calibrate on 10% of the COCO validation images (500 images); replace coco.yaml with your own dataset
        yolo export model=yolo26n.pt format=axelera data=coco.yaml fraction=0.1 # creates 'yolo26n_axelera_model/'
        ```

- **How many images**: use 100 to 400 calibration images. The compiler read every image it was given in the exports for this page: all 128 of `coco128.yaml`, and 500 with `data="coco.yaml"` and `fraction=0.1`.
- **`fraction`**: `fraction=0.5` with `coco128.yaml` calibrated on 64 images and triggered the warning above, and `fraction=100` on 100 images.
- **Accuracy**: the Metis accuracy figures on this page, in [Supported Tasks](#supported-tasks) and in the tip below, were measured on Metis hardware. The exports run for this page were not validated, because no Metis device was available. Validate the exported model on your device and your own data before you deploy it.

!!! tip "Calibrate depth models on images from your own scenes"

    Quantization ranges come from the calibration images, and the default for depth, `depth8.yaml`, holds four indoor frames. On the 654-image [NYU Depth V2](../datasets/depth/nyu-depth-v2.md) val split, YOLO26n depth calibrated on that default scores δ<sub>1</sub> 0.742 on Metis against 0.783 for the PyTorch model, while 100 NYU training frames give 0.780. Pass `data=` with a dataset YAML whose `val` split holds at least 100 images from the scenes the model will see; calibration reads the `val` split.

## Output Format

The export writes one directory:

| File                         | Contents                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `yolo26n.axm`                | The compiled INT8 model that the Axelera runtime loads                                                                   |
| `compiler_config_final.toml` | The configuration the compiler used for the build                                                                        |
| `metadata.yaml`              | Class names, task, image size, batch size, stride, and export arguments, which Ultralytics reads when it loads the model |

The compiler also leaves an empty `.pkl_memoize_py3` directory in the working directory. The `axmodeltool` command of `axelera-rt` reports the input of the YOLO26n `.axm` file as one quantized tensor of shape `1,640,640,3`, channels last, with scale 0.003922 and zero point -128, and its outputs as the six feature maps of the detection head, `1,80,80,4` to `1,20,20,4` for the boxes and `1,80,80,80` to `1,20,20,80` for the class scores. The Axelera runtime quantizes the input and decodes these outputs on the host.

`nms` selects the head that the export hands to the compiler. The shapes below are the outputs of the PyTorch model that the export logs, for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | Output by default and with `nms=False` | Output with `nms=True`                   |
| -------- | --------------- | -------------------------------------- | ---------------------------------------- |
| Detect   | `640`           | `(1, 300, 6)`                          | `(1, 84, 8400)`                          |
| Segment  | `640`           | `(1, 300, 38)` and `(1, 32, 160, 160)` | `(1, 116, 8400)` and `(1, 32, 160, 160)` |
| Semantic | `1024`          | `(1, 19, 1024, 1024)`                  | Same as the default                      |
| Depth    | `768`           | `(1, 1, 768, 768)`                     | Same as the default                      |
| Classify | `224`           | `(1, 1000)`                            | Same as the default                      |
| Pose     | `640`           | `(1, 300, 57)`                         | `(1, 56, 8400)`                          |
| OBB      | `1024`          | `(1, 300, 7)`                          | `(1, 20, 21504)`                         |

- **Default and `nms=False`**: YOLO26 models keep their NMS-free head, with `max_det` (default 300) rows per image, and `metadata.yaml` records `end2end: true`. Both values wrote the same outputs.
- **`nms=True`**: the export logs `format=axelera does not support embedded NMS; exporting native outputs for external NMS.` and exports the one-to-many head, which needs NMS, and `metadata.yaml` records `end2end: false`. There is no NMS embedded in the model. The metadata that the compiler stores inside the YOLO26n detection `.axm` file still says `end2end = true` for this export, as it does for the default one. On Metis, prefer the default export: `nms=True` scored lower, see [Supported Tasks](#supported-tasks).
- **Models without an NMS-free head**: YOLO11 and YOLOv8 models export the one-to-many head with every value of `nms`, `(1, 84, 8400)` for YOLO11n and YOLOv8n detection. With `nms=False` the export warns `This model has no one-to-one head; using one-to-many outputs.`
- **Segmentation**: the second output holds the mask prototypes.

`YOLO("yolo26n_axelera_model")` reads the layout from `metadata.yaml`. Its keys include `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, `args`, and, for pose models, `kpt_shape`.

## Deploy

On the machine with the Metis device, install the [kernel driver](#driver-installation), install the package with `pip install ultralytics`, and copy the `yolo26n_axelera_model/` directory to it. `YOLO("yolo26n_axelera_model")` then runs the `.axm` file through the `axelera.runtime` module of the `axelera-rt` package. The points below follow from the Ultralytics code and from a computer without a Metis device, not from a run on Metis hardware:

- **Runtime installation**: the first `predict` or `val` installs `axelera-rt==1.9.0` from the package index of Axelera, which can take several minutes. On the computer used for this page that install also replaced `opencv-python` 5.1.0.95 with 4.14.0.94 and `pillow` 13.0.0 with 12.3.0. To install it beforehand, run `pip install axelera-rt==1.9.0 --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple`.
- **Fixed input shape**: the model always runs at its export `imgsz`, which Ultralytics reads from `metadata.yaml`.
- **Batches**: a compiled model takes one image. `predict` hands a larger batch to the Axelera scheduler, see [Maximum Performance](#maximum-performance), and `val` runs one image at a time.
- **One `.axm` file per directory**: Ultralytics loads the first `.axm` file it finds in the directory and its subdirectories, so keep only one there.
- **Directory name**: keep `_axelera_model` in the directory name, because Ultralytics recognizes the format by it, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

### Driver Installation

The Metis kernel driver `metis-dkms` is installed with `apt`, separately from the Python packages.

1. Add the Axelera repository key:

    ```bash
    sudo sh -c "curl -fsSL https://software.axelera.ai/artifactory/api/security/keypair/axelera/public | gpg --dearmor -o /etc/apt/keyrings/axelera.gpg"
    ```

2. Add the repository to apt:

    Choose the appropriate snippet from below to match the OS being used.

    ```bash
    # Ubuntu 22.04
    sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source ubuntu22 main' > /etc/apt/sources.list.d/axelera.list"
    ```

    ```bash
    # Ubuntu 24.04
    sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source ubuntu24 main' > /etc/apt/sources.list.d/axelera.list"
    ```

3. Install the driver and load it. SDK 1.9 needs `metis-dkms` 1.7.0 or newer, so the newest is installed here:

    ```bash
    sudo apt update
    sudo apt install -y metis-dkms
    sudo modprobe metis
    ```

!!! note "Keep the kernel driver in sync with the SDK"

    The kernel driver is installed separately from the Python SDK packages, so after upgrading the SDK you must also bring the driver up to that SDK's minimum — 1.7.0 for SDK 1.9. Nothing checks this at install time, and a driver below the minimum leaves the device unavailable: `axdevice` reports the version required and fails to open the device.

    ```text
    [libaxldev.c:285] Found kernel driver version 1.6.2, but at least version 1.7.0 is required. Please update the kernel driver
    ERROR: AXR_ERROR_CONNECTION_ERROR: Failed to open device metis-0:4:0
    ```

    See also [Step 3: Install the Metis kernel driver](https://docs.axelera.ai/sdk/user-guides/sdk-install#step-3-install-the-metis-kernel-driver). If you are already running an earlier SDK on Ubuntu, update the driver and reload the kernel module (no reboot required):

    ```bash
    sudo apt update
    sudo apt install -y metis-dkms
    sudo rmmod metis
    sudo modprobe metis
    ```

    Then re-run `axdevice` to confirm the device is detected.

!!! note "Models exported with SDK 1.8 keep working"

    SDK 1.9 does not raise the compiled `.axm` model format, so an `_axelera_model` directory exported with 1.8 loads and runs unchanged. Re-export only to pick up compiler improvements.

    An SDK that does raise the format rejects older directories at load time, which is what an upgrade from 1.7 or earlier reports:

    ```text
    Error: Unsupported model version: 4.0, expected at least 5.0
    Please re-deploy or re-download the model.
    ```

### Device Health Check

Verify your Axelera device is functioning properly:

```bash
# if axdevice cannot be found, please run at least one inference (see above) to ensure the required packages are installed
axdevice
```

For detailed diagnostics, see the [AxDevice documentation](https://docs.axelera.ai/sdk/reference/tools/axdevice/).

On a computer without a Metis device, `axdevice` reports `ERROR: AXR_ERROR_CONNECTION_ERROR: No target device found in lspci output`.

### Maximum Performance

Models are compiled for a single image, and a single image runs on one AIPU core. Passing several images to [Predict](../modes/predict.md) hands them to the Axelera scheduler together, which spreads them over the cores of the device:

```bash
yolo predict model=yolo26n_axelera_model source=path/to/images batch=32
```

`yolo26n` at `imgsz=640` over the 128 [COCO128](https://platform.ultralytics.com/ultralytics/datasets/coco128) images in a directory, Metis PCIe, median of three runs:

| Voyager SDK | Devices | `batch=1` | `batch=8` | `batch=32` |
| :---------- | :------ | :-------- | :-------- | :--------- |
| 1.8.0       | 1       | 36 img/s  | 45 img/s  | 40 img/s   |
| 1.8.0       | 2       | 39 img/s  | 41 img/s  | 41 img/s   |
| 1.9.0       | 1       | 41 img/s  | 85 img/s  | 109 img/s  |
| 1.9.0       | 2       | 35 img/s  | 83 img/s  | 113 img/s  |

On SDK 1.9 a batch is what raises throughput, 2.7 times at `batch=32`, where 1.8 stays near 40 img/s at any batch size. Use a large batch for offline throughput and `batch=1` when per-image latency matters. A second device adds little to `predict`: at `batch=32` each image already spends about 4 ms in host-side preprocessing, roughly as long as its time on the device.

### Beyond `predict`

For production requiring maximum throughput, the [Axelera Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk) offers:

- Streaming inference pipelines that keep several frames in flight
- Compiling one model across several cores
- Tiled inferencing for higher-resolution cameras

Those paths skip the PyTorch pre- and post-processing that `predict` runs on the host. See the [model-zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo) for FPS benchmarks or [contact Axelera](https://axelera.ai/contact-us) for production support.

### Deployment Guides

- **[YOLO on Voyager SDK examples](https://github.com/ultralytics/ultralytics/tree/main/examples/python/Axelera)**: composable Python pipelines with `axelera-rt` that run an exported `.axm` file without PyTorch.
- **[Voyager SDK on GitHub](https://github.com/axelera-ai-hub/voyager-sdk)**: the SDK, drivers, and compiler tools.
- **[Voyager SDK installation guide](https://docs.axelera.ai/sdk/user-guides/sdk-install#step-3-install-the-metis-kernel-driver)** and **[AxDevice documentation](https://docs.axelera.ai/sdk/reference/tools/axdevice/)**: the driver and device tools.
- **[Axelera Community](https://community.axelera.ai/)**: support from Axelera.
- **[Ultralytics YOLO on Axelera hardware](https://www.ultralytics.com/blog/bringing-ultralytics-yolo-models-to-axelera-ai-hardware-for-edge-ai)**: an overview on the Ultralytics blog.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare Axelera with the other export formats.

## Limitations and Troubleshooting

- **Export host**: on macOS the export stops with `export is only supported on Linux and is not supported on ARM64 Docker.` before it installs anything, and the same check stops it on Windows and in a Docker container on ARM64 Linux, where it was not run for this page. Export on a Linux computer or with [Ultralytics Platform](../platform/train/models.md#export-model).
- **First export after a plain install**: with only `pip install ultralytics`, which installed PyTorch 2.15.0, the first export installs `axelera-devkit`, which replaces it with PyTorch 2.14.1, logs `Restart runtime or rerun command for updates to take effect`, and fails with `Could not find a qtoolsv2 C++ extension compatible with torch 2.15.0.` Run the export again, or install the compiler first as in the [Quickstart](#quickstart).
- **Automatic installation turned off**: with `YOLO_AUTOINSTALL=false` and no compiler installed, the export logs `Ultralytics requirement ['axelera-devkit==1.9.0'] not found, AutoUpdate disabled by YOLO_AUTOINSTALL=False` and fails with `Axelera export requires axelera-devkit==1.9.0.` After the install command of the [Quickstart](#quickstart), the export runs with `YOLO_AUTOINSTALL=false`.
- **`batch` above 1**: the export fails with `Axelera: export failure` and `Lowering failed.`, for example with `AssertionError: Only batch sizes of 1 supported` for YOLOv8n. YOLO26n, YOLO11n, and YOLO26n classification models failed in the same step. Export with the default `batch=1`.
- **Image size below 64**: `imgsz=32` fails with `Axelera export requires imgsz>=64, but got imgsz=[32, 32].`
- **`dynamic`, `opset`, and `simplify`**: `dynamic=True`, any `opset`, and `simplify=False` fail with `argument 'dynamic' is not supported for format='axelera'`, and the same message for `opset` and `simplify`; `simplify=True` is the default and passes. The model has a fixed input size; export again to change it.
- **Operator warnings**: for the YOLO26n and YOLO11n models the compiler lists `Reshape`, `Transpose`, `MatMul`, and `Softmax` nodes of the attention blocks and logs `The operator compatibility warnings above suggest this model may not be supported by the Axelera Compiler.` These exports completed.
- **Small calibration sets**: the default datasets of the semantic segmentation, depth, and pose tasks hold 4 images, so their exports log the calibration warning. See [Calibration](#calibration).
- **Output directory**: every export of one checkpoint writes the same `yolo26n_axelera_model/` directory, whatever its `imgsz`, `data`, or `nms`, and replaces it. Rename the directory before you export a second variant, and keep `_axelera_model` in its name: a copy named `yolo26n_metis` failed with `is not a supported model format`.
- **Missing model file**: a directory without an `.axm` file fails with `No .axm file found in: empty_axelera_model`.
- **Inference without a Metis device**: on the x86-64 export host, the first `predict` with the exported model failed with `Axelera inference requires axelera-rt==1.9.0.` when `axelera-rt` was missing and automatic installation was off, and with `this host has no dmabuf heap; pass input_dmabuf=False, output_dmabuf=False to AxRuntimeModel if you want host-allocated buffers` once it was installed. Validate the PyTorch checkpoint on the export host instead; this does not check the INT8 computation of the `.axm` file.
- **M.2 power limitations**: large or extra-large models may encounter runtime errors on M.2 accelerators due to power supply constraints.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

The Metis AIPU is designed for high throughput at low power.

| Model   | Metis PCIe FPS (frames per second) | Metis M.2 FPS (frames per second) |
| ------- | ---------------------------------- | --------------------------------- |
| YOLOv8n | 847                                | 771                               |
| YOLO11n | 746                                | 574                               |
| YOLO26n | 648.6                              | 484.9                             |

_Benchmarks based on Axelera AI data. Actual FPS depends on model size, batching, and input resolution._

These are host throughput figures, measured by the [model zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo) with the Voyager SDK's streaming pipeline over a 720p video file. A single `predict` call reports per-image latency instead, which is a different quantity.

The table below lists the sizes of current exports of the YOLO26 detection models, as the `.axm` file in MB (10^6 bytes) at `imgsz=640`. The export log prints the size of the whole directory in MiB, as `(20.7 MB)` for YOLO26n:

| Model   | PyTorch | Axelera |
| ------- | ------- | ------- |
| YOLO26n | 5.5     | 21.7    |
| YOLO26s | 20.4    | 26.6    |
| YOLO26m | 44.3    | 40.2    |
| YOLO26l | 53.2    | 58.9    |
| YOLO26x | 118.7   | 81.8    |

Exported with `ultralytics` 8.4.215 and `axelera-devkit` 1.9.0 on an x86-64 Linux server without Metis hardware. The [YOLO26 model page](../models/yolo26.md) lists the accuracy of the PyTorch checkpoints on the full COCO validation set.

## About Axelera AI

![Axelera AI edge deployment ecosystem for YOLO](https://github.com/user-attachments/assets/c97a0297-390d-47df-bb13-ff1aa499f34a)

[Axelera AI](https://axelera.ai/) is an [Ultralytics partner](https://www.ultralytics.com/partners/axelera) that builds the Metis AIPU, an accelerator for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) inference on [Edge AI](https://www.ultralytics.com/glossary/edge-ai) devices, with up to 856 TOPS on a four-AIPU card. Its Voyager SDK includes the compiler, which quantizes and compiles models into `.axm` files on a PC, and the runtime, which runs them on the AIPU.

### Metis Hardware

Axelera sells the Metis AIPU as accelerator cards for existing hosts and in complete systems, through the [Axelera store](https://store.axelera.ai/):

| Product           | Form Factor    | Compute            | Performance (INT8) | Target Application                                                                                                                         |
| :---------------- | :------------- | :----------------- | :----------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| **Metis PCIe x4** | PCIe Gen3 x16  | **4x** Metis AIPUs | **856 TOPS**       | High-density [video analytics](../guides/analytics.md), smart cities                                                                       |
| **Metis PCIe x1** | PCIe Gen3 x1   | **1x** Metis AIPU  | **214 TOPS**       | Industrial PCs, retail [queue management](../guides/queue-management.md)                                                                   |
| **Metis M.2**     | M.2 2280 M-Key | **1x** Metis AIPU  | **214 TOPS**       | [Drones](https://www.ultralytics.com/blog/build-ai-powered-drone-applications-with-ultralytics-yolo11), robotics, portable medical devices |
| **Metis M.2 MAX** | M.2 2280       | **1x** Metis AIPU  | **214 TOPS**       | Environments requiring advanced thermal management                                                                                         |

Choose Metis PCIe x4 for 30 or more video streams, Metis PCIe x1 for a PC host, the Metis Compute Board for a standalone device, and Metis M.2 otherwise, or Metis M.2 MAX where the device needs advanced thermal management.

For turnkey solutions, Axelera partners with manufacturers to provide systems pre-validated for the Metis AIPU.

- **Metis Compute Board**: A standalone edge device pairing the Metis AIPU with a Rockchip RK3588 ARM CPU.
- **Workstations**: Enterprise towers from **Dell** (Precision 3460XE) and **Lenovo** (ThinkStation P360 Ultra).
- **Industrial PCs**: Ruggedized systems from **Advantech** and **Aetina** designed for [manufacturing automation](https://www.ultralytics.com/solutions/computer-vision-in-manufacturing).

## FAQ

### Do I need a Metis device to export a model?

No. The Voyager SDK compiler runs on a Linux computer, and every export on this page ran on a server without Axelera hardware and with no GPU visible to the export. You need the device, with the `metis-dkms` driver and `axelera-rt`, only to run or validate the `.axm` file.

### Can I test an Axelera model on my PC before I copy it to the device?

Not with Ultralytics: without a Metis device, `predict` and `val` fail, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). On the export host you can validate the PyTorch checkpoint, and `axmodeltool` from `axelera-rt` prints the input and output tensors of the `.axm` file. This does not replace a validation on the device, because INT8 quantization changes the outputs.

### How many calibration images should I use?

Use 100 to 400 images that represent your deployment scene; the export warns below 100. The default datasets of some tasks hold only 4 images, which is enough to test the export but not to calibrate a model for deployment. See [Calibration](#calibration).

### Which YOLO models can I export to Axelera?

[YOLOv8](../models/yolov8.md), [YOLO11](../models/yolo11.md), and [YOLO26](../models/yolo26.md) models, including models you trained yourself with [Train mode](../modes/train.md). See [Supported Tasks](#supported-tasks) for the tasks of each family.

### Where can I find the Voyager SDK?

The SDK, drivers, and compiler tools are available from the [Voyager SDK GitHub repository](https://github.com/axelera-ai-hub/voyager-sdk). Ultralytics installs the two Python packages it needs, `axelera-devkit` for export and `axelera-rt` for inference, from the package index of Axelera.
