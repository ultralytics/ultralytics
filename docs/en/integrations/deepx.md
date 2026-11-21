---
comments: true
description: Export Ultralytics YOLO26 models to the DEEPX .dxnn format with format=deepx for INT8 inference on DEEPX NPUs. Covers the DX-COM compiler, calibration, output files, DX-RT runtime setup, and Raspberry Pi 5 benchmarks.
keywords: YOLO26, DEEPX, DEEPX export, dxnn, DX-COM, dx_com, DX-RT, dx_engine, DX-M1, NPU, INT8 quantization, Raspberry Pi 5, M.2 AI accelerator, edge AI, embedded inference, model deployment
---

# DEEPX Export

[DEEPX](https://deepx.ai/) [NPUs](https://www.ultralytics.com/glossary/neural-processing-unit-npu), such as the DX-M1 M.2 module, run models compiled into the `.dxnn` format by DX-COM, the DEEPX compiler. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="deepx"` creates a `yolo26n_deepx_model/` directory that holds an INT8 `.dxnn` file. The export runs on an x86-64 Linux computer and needs no DEEPX hardware.

## At a Glance

| Property          | Details                                                                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `deepx`                                                                                                                                 |
| Output            | `yolo26n_deepx_model/`, a directory with `yolo26n.dxnn`, `config.json`, and [`metadata.yaml`](#output-format)                           |
| Runs on           | DEEPX NPUs, such as the DX-M1 and DX-M1M M.2 modules, in an x86-64 or ARM64 Linux host such as the Raspberry Pi 5                       |
| Runtime           | DX-RT 3.5.0 or newer with the NPU driver and the [`dx_engine`](#runtime-installation) Python package on the device                      |
| Precision         | INT8 only, calibrated on images from `data`. YOLO26n is 3.5 MB                                                                          |
| Export host       | Linux x86-64 with glibc 2.31 or newer and Python 3.8 to 3.14. Not ARM64, macOS, or Windows. The export needs no GPU and no DEEPX device |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to DEEPX, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package in a virtual environment on an x86-64 Linux machine:

```bash
python3 -m venv deepx-env && source deepx-env/bin/activate
pip install ultralytics
```

The first export installs `onnx`, `onnxslim`, `onnxruntime`, the DX-COM compiler [`dx_com`](https://pypi.org/project/dx-com/), and the packages that `dx_com` requires except `opencv-python`: `dx_com` asks for the GUI build of OpenCV, and Ultralytics keeps the OpenCV package that is already installed. The export logs `Restart runtime or rerun command for updates to take effect` and still completes in the same run. It did not replace `numpy` or OpenCV in any environment used for this page, and in an environment with `opencv-python-headless`, pip only reported `dx-com 2.5.0 requires opencv-python>=4.10.0.84, which is not installed.` `dx_com` 2.5.0 also requires PyTorch 2.3 or newer, `protobuf` 5.26 or newer, and `Pillow` 10.4 or newer, and the first export installs any of these that are missing: it installed `protobuf` after a plain `pip install ultralytics`, while PyTorch and Pillow were already new enough.

To install the compiler before the first export, run `pip install "ultralytics[export-deepx]"` instead, which on Python 3.12 or older requires `numpy` below 2; the first export then installs only `onnxslim` with its dependency `colorama`. For an editable repository install, use `pip install -e ".[export-base,export-deepx]"`. The export has these version requirements:

| Requirement      | Version                                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python           | 3.8 to 3.14, the versions that `dx_com` 2.5.0 has packages for. Exports ran with every version from 3.8 to 3.14                                         |
| Operating system | Linux x86-64 with glibc 2.31 or newer. `dx_com` has no packages for ARM64 Linux, macOS, or Windows, see [Limitations](#limitations-and-troubleshooting) |
| `dx_com`         | 2.5.0 or newer. Every export on this page used 2.5.0                                                                                                    |
| Memory           | At least 16 GB of RAM for DX-COM 2.5.0. Not measured for this page; the environment builder below skips DEEPX on machines with less than 15 GiB         |

To build the Python 3.14 environment that the Ultralytics CI uses for its DEEPX export, run the environment builder from the root of a repository checkout. It requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and an installed Ultralytics checkout:

```bash
ULTRALYTICS_ISOLATED_VENVS="$PWD/.venvs" python .github/scripts/create-export-env.py --env isolated-deepx
```

!!! warning "Runtime compatibility"

    DX-COM 2.5.0 produces `.dxnn` file format v9, which only loads on DX-RT 3.5.0 or newer. Install the matching DEEPX runtime described in [Runtime Installation](#runtime-installation), and check the [DEEPX version compatibility matrix](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/04_Version_Compatibility.md) before upgrading individual components.

For other installation methods, see the [installation guide](../quickstart.md). Then export your model:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to DEEPX format with INT8 quantization
        model.export(format="deepx")  # creates 'yolo26n_deepx_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to DEEPX format with INT8 quantization
        yolo export model=yolo26n.pt format=deepx # creates 'yolo26n_deepx_model/'
        ```

Without `data`, the export calibrates on the small default dataset of the task, such as the 4 images of `coco8.yaml`, which is enough to test the export; calibrate on images from your deployment data before you ship the model, see [Calibration](#calibration). On a shared AMD EPYC 9655 server, without a GPU, exports of YOLO26n took 23 to 28 s with the compiler installed, and 31 to 42 s for the first export in a new environment, which also installs it. The YOLO26n semantic model at 1024 pixels took 228 to 230 s, and RT-DETR-L took 150 s.

Copy the `yolo26n_deepx_model/` directory to the device with the DEEPX NPU. Predict and Validate run only there, with the [DEEPX runtime](#runtime-installation). They were not run on a DEEPX device for this page:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported DEEPX model on the device with the DEEPX NPU
        model = YOLO("yolo26n_deepx_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported DEEPX model on the device with the DEEPX NPU
        yolo predict model=yolo26n_deepx_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported DEEPX model on the device with the DEEPX NPU
        model = YOLO("yolo26n_deepx_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported DEEPX model on the device with the DEEPX NPU
        yolo val model=yolo26n_deepx_model data=coco8.yaml
        ```

## Supported Tasks

DEEPX export supports all seven Ultralytics tasks: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with [YOLO26](../models/yolo26.md), the only family that ships those heads. The YOLO26n checkpoint of every task compiled, and so did [YOLO11](../models/yolo11.md), [YOLOv8](../models/yolov8.md), [YOLOv10](../models/yolov10.md), and [RT-DETR](../models/rtdetr.md) detection models. None of these exports was run on a DEEPX device for this page.

## Export Arguments

| Argument   | Type             | Default   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------- | ---------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'deepx'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `imgsz`    | `int` or `tuple` | `640`     | Image size of the model input, fixed in the `.dxnn` file. The export needs a square input, because Ultralytics passes DX-COM a single resize size: pass an integer, or a tuple with equal height and width. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect, segment, and pose models and for semantic models pretrained on ADE20K, `1024` for OBB models and for semantic models pretrained on Cityscapes, `768` for depth, and `224` for classify. |
| `quantize` | `int` or `str`   | `None`    | Precision. Unset and `8` build INT8, the only precision; any other value, such as `16`, `32`, or `'w8a16'`, raises an error, see [Precision](#precision). Replaces the deprecated `half`/`int8` flags.                                                                                                                                                                                                                                                                                                                             |
| `data`     | `str`            | `None`    | Dataset YAML with the calibration images; classification instead takes a dataset directory or a built-in dataset name. Calibration reads the image directory of the `val` split unless `split` is set. If omitted, Ultralytics uses the small default dataset of the task, such as `coco8.yaml`. See [Calibration](#calibration).                                                                                                                                                                                                  |
| `split`    | `str`            | `'val'`   | Dataset split whose image directory DX-COM calibrates on.                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `optimize` | `bool`           | `False`   | `True` compiles with `opt_level=1`, the default of DX-COM itself, which DEEPX describes as lower latency and higher accuracy for a longer compilation. See [Compiler Optimization](#compiler-optimization).                                                                                                                                                                                                                                                                                                                        |
| `nms`      | `bool`, optional | `None`    | Select raw output (`None`, default) or the NMS-free head (`False`). Embedded NMS (`True`) is not available, see [Output Format](#output-format).                                                                                                                                                                                                                                                                                                                                                                                   |
| `opset`    | `int`            | `None`    | ONNX opset version of the intermediate ONNX graph. When unset, `18` with PyTorch 2.4 or newer and `17` with PyTorch 2.3 (the oldest PyTorch that `dx_com` accepts). Exports with `opset=13` and `opset=21` compiled too.                                                                                                                                                                                                                                                                                                           |
| `simplify` | `bool`           | `True`    | Simplifies the intermediate ONNX graph with `onnxslim`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `device`   | `str`            | `None`    | Device that traces the intermediate ONNX graph, `cpu` by default. DX-COM chooses its own device: by its command-line help it quantizes on a CUDA GPU when one is available, whatever `device` is, and you select or hide the GPU with `CUDA_VISIBLE_DEVICES`. Without a GPU it logs `No supported GPU/CUDA environment was detected. Quantization will run on CPU, so compilation may be slower.`                                                                                                                                  |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Precision

DX-COM quantizes every model to 8-bit weights and 8-bit activations with post-training quantization. There is no FP32 or FP16 export:

| `quantize`   | Result                                                                                                                            |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| unset or `8` | INT8 `.dxnn` file. YOLO26n is 3.5 MB. Without `quantize`, the export logs `deepx export requires INT8 quantization, enabling it.` |
| `16`         | Fails with `quantize=16 (FP16) is not supported; format='deepx' supports quantize=8 (INT8).`                                      |
| `32`         | Fails with `quantize=32 (FP32) is not supported; format='deepx' supports quantize=8 (INT8).`                                      |
| `'w8a16'`    | Fails with `quantize='w8a16' is not supported; format='deepx' supports quantize=8 (INT8).`                                        |

The division of the pixel values by 255 is compiled into the model, and the `.dxnn` file takes `uint8` images, see [Output Format](#output-format).

### Calibration

DX-COM calibrates the activation ranges itself. Ultralytics passes it the image directory of the `val` split of `data`, or of the split you set with `split`, and a configuration with at most 100 calibration images (`calibration_num`) and the `ema` method, which the export saves as `config.json`. DX-COM reads `.jpg`, `.jpeg`, and `.png` files from that directory and its subdirectories and, as that configuration tells it, resizes them with gray padding to the export size, scales them to 0-1, and converts them to RGB. Pass `data` with images that look like your deployment data:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Calibrate on COCO128, of which DX-COM reads 100 images; replace coco128.yaml with your own dataset
        model.export(format="deepx", data="coco128.yaml")  # creates 'yolo26n_deepx_model/'
        ```

    === "CLI"

        ```bash
        # Calibrate on COCO128, of which DX-COM reads 100 images; replace coco128.yaml with your own dataset
        yolo export model=yolo26n.pt format=deepx data=coco128.yaml # creates 'yolo26n_deepx_model/'
        ```

- **Image directory required**: the split must be a directory of images. A dataset whose split is a text file with image paths, such as `coco.yaml`, fails in DX-COM, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).
- **No `fraction`**: the argument that limits the calibration images of other formats is not available for DEEPX. To calibrate on a subset, point `data` at a dataset whose split directory holds only those images.
- **Calibration size warning**: the export warns `>300 images recommended for INT8 calibration, found 4 images.` below 300 images, but DX-COM reads at most 100: it read 100 of the 128 images of `coco128.yaml`, and all 4 images of `coco8.yaml`.
- **Accuracy**: the accuracy of the exports on this page was not measured, because no DEEPX device was available. Validate the exported model on your device and your own data before you deploy it.

### Compiler Optimization

`optimize` sets the optimization level of DX-COM: `False` (default) compiles with `opt_level=0` and `True` with `opt_level=1`, the default of the `dxcom` command line. Its help describes the levels as `Higher levels improve model performance (lower latency) but increase compilation time`, and the description of `dx_com` on PyPI as `0` (fast) or `1` (higher accuracy). On the same server, the DEEPX export step that the log times took 111 to 114 s with `optimize=True` in 3 runs, against 23 to 27 s for the same step without it, and the `.dxnn` file was 3.7 MB against 3.5 MB. Neither the latency nor the accuracy of the two levels was compared on a DEEPX device for this page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Compile with the higher DX-COM optimization level
        model.export(format="deepx", optimize=True)  # creates 'yolo26n_deepx_model/'
        ```

    === "CLI"

        ```bash
        # Compile with the higher DX-COM optimization level
        yolo export model=yolo26n.pt format=deepx optimize=True # creates 'yolo26n_deepx_model/'
        ```

## Output Format

The export writes one directory:

| File            | Contents                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `yolo26n.dxnn`  | The compiled INT8 model that DX-RT loads                                                                                 |
| `config.json`   | The DX-COM configuration of the export: input shape, calibration settings, and the path of the calibration images        |
| `metadata.yaml` | Class names, task, image size, batch size, stride, and export arguments, which Ultralytics reads when it loads the model |

The intermediate `yolo26n.onnx` file stays next to the directory. DX-COM compiles the transpose to channels first and the division by 255 into the model, and reports the input of the `.dxnn` file as `uint8 NHWC [1, 640, 640, 3]`: one RGB image, channels last, with values from 0 to 255. This differs from the [ONNX export](onnx.md#output-format), which takes channels-first `float32` images scaled to 0-1. DX-COM did not compile the resize and the color conversion into the model (`Skipped nodes (cannot be inserted in NPU graph): resize, convertColor, expandDim`), so in your own code letterbox the image and pass it as one RGB `uint8` array of shape `(640, 640, 3)`, as Ultralytics does. The batch size is always 1.

The outputs below are those of the intermediate ONNX graph that DX-COM compiles, for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation. The outputs of the `.dxnn` files were not read on a DEEPX device for this page:

| Task     | Default `imgsz` | Output with `nms=None`                                  | Output with `nms=False`                             |
| -------- | --------------- | ------------------------------------------------------- | --------------------------------------------------- |
| Detect   | `640`           | `(1, 84, 8400)`                                         | `(1, 300, 6)`                                       |
| Segment  | `640`           | `(1, 116, 8400)` and `(1, 32, 160, 160)`                | `(1, 300, 38)` and `(1, 32, 160, 160)`              |
| Semantic | `1024`          | `(1, 19, 1024, 1024)`, a score for each class and pixel | Same as `nms=None`                                  |
| Depth    | `768`           | `(1, 1, 768, 768)`, depth map                           | Same as `nms=None`                                  |
| Classify | `224`           | `(1, 1000)`, class probabilities                        | Same as `nms=None`                                  |
| Pose     | `640`           | `(1, 56, 8400)`                                         | `(1, 300, 57)`: 6 box values, then 17 keypoints x 3 |
| OBB      | `1024`          | `(1, 20, 21504)`                                        | `(1, 300, 7)`: 4 box values, score, class, angle    |

- **Raw output (`nms=None`)**: for each anchor the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS yourself. See the [ONNX output format](onnx.md#output-format) for the layout of each task.
- **NMS-free output (`nms=False`)**: `max_det` (default 300) rows of `x1, y1, x2, y2, score, class` in input pixels, followed by the mask coefficients or keypoints. OBB rows are `center x, center y, width, height, score, class, angle`. Filter the rows by score. Available for models with an NMS-free head, such as YOLO26. For the detection, segmentation, and pose exports with `nms=False`, DX-COM warned that the output of the graph it had modified differed from the original graph, with a cosine similarity of 0.84 to 0.94, and named resize approximation or TopK/NMS as the possible cause: `After surgeon, cosine similarity for output index 0 is below 0.99: 0.9314. This may be due to NPU constraints requiring resize approximation, which can cause slight variations in outputs.` for detection. It did not warn in the exports with the raw output. Compare the detections of such a model with the PyTorch model on your device.
- **Embedded NMS**: `nms=True` logs `format=deepx does not support embedded NMS; using nms=None.` and writes the raw output.
- **Semantic segmentation**: the output holds a score for each class and pixel, not the class map of the [ONNX export](onnx.md#output-format), so take the argmax over the class axis in your own code.
- **Segmentation**: the second output holds the mask prototypes.
- **NPU and CPU parts**: DX-COM split the detection, segmentation, depth, pose, and OBB models into one part for the NPU and one for the CPU of the host, and compiled the classification and semantic segmentation models for the NPU alone. For the detection model it logged `Marked 11 MemoryOp(s) from NPU to CPU`.

`YOLO("yolo26n_deepx_model")` reads the layout from `metadata.yaml`, so you only need these details when you run the `.dxnn` file in your own code. The keys of `metadata.yaml` include `names`, `task`, `imgsz`, `stride`, `batch`, `end2end`, `args`, and, for pose models, `kpt_shape`.

### Inspecting the Compiled Model

DX-COM 2.5.0 replaces the DX-TRON graph visualizer with an interactive HTML Compilation Summary Report that shows the compiled model structure and the CPU/NPU workload split. If needed, generate it by recompiling the intermediate ONNX model written by the export with the `dxcom` CLI on the x86-64 export host:

```bash
dxcom -m yolo26n.onnx -c yolo26n_deepx_model/config.json -o yolo26n_deepx_report --export_html
```

Open `yolo26n_deepx_report/yolo26n_summary.html` in a browser to view the report. The `dxcom` command compiles with its own default, `--opt_level 1`, which matches an export with `optimize=True`, and writes a second `yolo26n.dxnn` file into the report directory. Add `--opt_level 0` to get the report of a model exported with the default `optimize=False`: that command wrote a `.dxnn` file of the same size as the export, though not byte-identical, so the report describes a new compilation and not the exported file.

## Deploy

On the device with the DEEPX NPU, create and activate a virtual environment as in the [Quickstart](#quickstart), install the package with `pip install ultralytics`, install the [runtime](#runtime-installation), and copy the `yolo26n_deepx_model/` directory to it. `YOLO("yolo26n_deepx_model")` then runs the `.dxnn` file through DX-RT with the `dx_engine` Python package. Without `dx_engine`, loading fails with ``DEEPX inference requires the DEEPX DX-Runtime and `dx_engine` Python package.`` The points below follow from the Ultralytics code and were not run on a DEEPX device for this page:

- **Input conversion**: Ultralytics converts each preprocessed image to the `uint8` channels-last array that the `.dxnn` file takes, so `predict` and `val` need no extra arguments.
- **Fixed input shape**: the model always runs at its export `imgsz`, which Ultralytics reads from `metadata.yaml`. An `imgsz` passed to `predict` or `val` is ignored; export again to change it.
- **Batches**: Ultralytics runs the images of a batch through the NPU one at a time, so any number of images works.
- **One `.dxnn` file per directory**: Ultralytics loads whichever `.dxnn` file it finds first in the directory and its subdirectories, so keep only one there.
- **Directory name**: keep `_deepx_model` in the directory name, because Ultralytics recognizes the format by it, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).

### Runtime Installation

Inference requires the DEEPX NPU driver (v2.7.0 or newer), the DX-RT runtime (v3.5.0 or newer), NPU firmware v2.7.6 or newer, and the `dx_engine` Python package. These are the versions required to run models compiled with DX-COM 2.5.0. The DEEPX runtime supports both x86-64 Linux and ARM64 Linux, such as the Raspberry Pi 5. The steps below were not run on a DEEPX device for this page.

Install the NPU driver and the `libdxrt-bin` runtime package with either of the following options.

#### Option A: Install from GitHub Release Packages

Download the driver and runtime `.deb` packages from the DEEPX-AI GitHub repositories and install them. The runtime package is provided per CPU architecture (`amd64` or `arm64`):

```bash
# Install the NPU driver and DX-RT runtime
sudo apt update
wget https://github.com/DEEPX-AI/dx_rt_npu_linux_driver/raw/main/release/2.7.0/dxrt-driver-dkms_2.7.0-2_all.deb
sudo apt install ./dxrt-driver-dkms_2.7.0-2_all.deb
wget https://github.com/DEEPX-AI/dx_rt/raw/main/release/3.5.0/libdxrt-bin_3.5.0_$(dpkg --print-architecture).deb
sudo apt install ./libdxrt-bin_3.5.0_$(dpkg --print-architecture).deb
```

#### Option B: Install from the DEEPX apt Repository

Register the DEEPX apt repository once, then install and upgrade the driver and runtime with `apt`:

```bash
# Add the DEEPX apt repository
wget -O - https://apt.releases.deepx.ai/gpg | sudo gpg --dearmor -o /usr/share/keyrings/deepx-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/deepx-archive-keyring.gpg] https://apt.releases.deepx.ai $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") main" | sudo tee /etc/apt/sources.list.d/deepx.list

# Install the NPU driver and DX-RT runtime
sudo apt update
sudo apt install dxrt-driver-dkms libdxrt-bin
```

Reboot the host after installing the driver so the new kernel module is loaded before running `dxcli`.

#### Python Package and Firmware

`libdxrt-bin` bundles `dx_engine` wheels for Python 3.8–3.14 in `/usr/share/libdxrt-bin/python/` but does not install them. In the environment where Ultralytics and NumPy are installed, install the wheel that matches your Python version:

```bash
python -m pip install --upgrade --no-index --find-links /usr/share/libdxrt-bin/python dx_engine
```

DX-RT 3.5.0 requires NPU firmware v2.7.6 or newer. Check the firmware version and device ID with `dxcli -s` and, if it is older, update it with the firmware image for your module. DX-M1 and DX-M1M M.2 modules use different firmware images, so download only the one that matches your module; `dxcli -s` reports the module as `M1` or `M1M` in its device information:

```bash
# Download the firmware image matching your module (DX-M1 shown)
wget -O dx-m1-2.7.6.bin https://github.com/DEEPX-AI/dx_fw/raw/main/m1/2.7.6/mdot2/fw.bin
# wget -O dx-m1m-2.7.6.bin https://github.com/DEEPX-AI/dx_fw/raw/main/m1m/2.7.6/mdot2/fw.bin # DX-M1M

# Update the selected device; replace DEVICE_ID with its ID from dxcli -s
sudo systemctl stop dxrt.service
dxcli -d DEVICE_ID -u dx-m1-2.7.6.bin # for DX-M1M use dx-m1m-2.7.6.bin instead
```

After flashing, power the host off completely and start it again before checking the firmware version. If physical power disconnection is not possible, reboot the host as described in the [DEEPX troubleshooting guide](https://developer.deepx.ai/tech-docs/2.4.0/DX-AllSuite/FAQ_Troubleshooting_Guide/).

Verify the installation with `dxcli -s`, which reports the runtime, driver, and firmware versions of each device. `import dx_engine` raises a `RuntimeError` if the major.minor versions of the `dx_engine` wheel and the installed `libdxrt` runtime differ, so reinstall the wheel from `/usr/share/libdxrt-bin/python/` after upgrading the runtime.

Once the runtime is installed, run inference and validation on your DEEPX device as shown in the [Quickstart](#quickstart): the exported `_deepx_model` directory loads directly with `YOLO(...)`.

### PCIe Gen 3 on Raspberry Pi 5

To get the best inference throughput from the DX-M1 NPU connected to a Raspberry Pi 5, open the boot configuration file and enable PCIe Gen 3 support.

```sh
sudo nano /boot/firmware/config.txt
```

Add the following lines at the end of the file:

```text
dtparam=pciex1
dtparam=pciex1_gen=3
```

Save and exit (Ctrl+X, then Y, then Enter), then reboot:

```sh
sudo reboot
```

Check the PCIe generation. Expected speed is 8GT/s for PCIe Gen3.

```sh
sudo lspci -vvv | grep -iA 33 accelerators | grep -E "LnkCap|LnkSta"
```

### Deployment Guides

- **[DEEPX version compatibility matrix](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/04_Version_Compatibility.md)**: the DX-COM, DX-RT, driver, and firmware versions that DEEPX releases together.
- **[DX-RT on GitHub](https://github.com/DEEPX-AI/dx_rt)**: the runtime, its release packages and release notes, and the `dxcli` tool.
- **[DEEPX troubleshooting guide](https://developer.deepx.ai/tech-docs/2.4.0/DX-AllSuite/FAQ_Troubleshooting_Guide/)**: DEEPX's answers to installation and firmware problems.
- **[Ultralytics and DEEPX](https://www.ultralytics.com/blog/ultralytics-yolo-partners-with-deepx-edge-ai-inference-for-physical-ai)**: an overview of the partnership on the Ultralytics blog.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare DEEPX with the other export formats.

## Limitations and Troubleshooting

- **Export host**: `dx_com` has packages only for x86-64 Linux. On macOS the export stops with `DEEPX export only supported on non-aarch64 Linux` before it installs anything, and the same check stops it on Windows and on ARM64 Linux, where it was not run for this page. Export on an x86-64 Linux computer or with [Ultralytics Platform](../platform/train/models.md#export-model).
- **Automatic installation turned off**: with `YOLO_AUTOINSTALL=false` and a plain `pip install ultralytics`, the export logs `Ultralytics requirements ['onnx>=1.12.0,<2.0.0', 'onnxruntime', 'onnxslim>=0.1.82'] not found, AutoUpdate disabled by YOLO_AUTOINSTALL=False` and fails with `No module named 'onnx'`. Install the packages first with `pip install "ultralytics[export-deepx]" onnxslim`; the export then needs no installation.
- **Datasets listed in a text file**: DX-COM reads the calibration images from a directory. With `data="coco.yaml"`, whose `val` split is the file `val2017.txt`, the export fails with `Given extensions ['jpeg', 'JPEG', 'jpg', 'JPG', 'png', 'PNG'] do not exist in .../coco/val2017.txt.` Use a dataset whose split is a directory of `.jpg`, `.jpeg`, or `.png` images.
- **`fraction`, `batch`, and `dynamic`**: setting these arguments fails with `argument 'fraction' is not supported for format='deepx'`, and the same message for `batch` and `dynamic`. The model has a batch size of 1 and a fixed input size; export again to change the image size.
- **Rectangular `imgsz`**: `imgsz=(480, 640)` fails in DX-COM with `Invalid JSON Configuration for input 'images': The config specifies (1, 3, 480, 480), but model requires (1, 3, 480, 640).` Export with a square size.
- **Calibration without `data`**: the export logs `INT8 export requires a missing 'data' arg for calibration. Using default 'data=coco8.yaml'.` and calibrates on the small default dataset of the task. Pass `data` with images that represent your deployment data.
- **NMS-free head on other models**: `nms=False` needs a model with an NMS-free head, such as YOLO26. For other models, for example YOLO11, the export warns `This model has no one-to-one head; using one-to-many outputs.` and writes the raw output, `(1, 84, 8400)` for YOLO11n detection.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='deepx' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends`. Export the checkpoint you started the fine-tuning from instead.
- **Output directory**: every export of one checkpoint writes the same `yolo26n_deepx_model/` directory, whatever its `imgsz`, `data`, `optimize`, or `nms`, and overwrites the files in it. Rename the directory before you export a second variant, and keep `_deepx_model` in its name, for example `yolo26n_640_deepx_model/`, because Ultralytics recognizes the format by it: a copy named `yolo26n_npu` failed with `is not a supported model format`.
- **Runtime version**: a `.dxnn` file compiled with DX-COM 2.5.0 needs DX-RT 3.5.0 or newer, with the driver and firmware versions listed in [Runtime Installation](#runtime-installation).
- **Inference on the export host**: without the DEEPX runtime, `YOLO("yolo26n_deepx_model")` fails with ``DEEPX inference requires the DEEPX DX-Runtime and `dx_engine` Python package.`` On the export host you can validate the PyTorch checkpoint or the intermediate `yolo26n.onnx` file with [ONNX Runtime](onnx.md), which does not check the INT8 computation of the `.dxnn` file.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

The Ultralytics team benchmarked YOLO26n models on a Raspberry Pi 5 with a DX-M1 M.2 module, comparing speed and accuracy between PyTorch on the CPU and DEEPX on the NPU.

!!! tip "Performance"

    <div align="center">
    <img width="800" src="https://cdn.ul.run/i/857bf50e3c613f76c2b26b28311106ce.avif" alt="Raspberry Pi 5 DEEPX M1 NPU vs PyTorch benchmarks">
    </div>

    | Model        | Format  | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
    | ------------ | ------- | ------ | --------- | ------------------- | ---------------------- |
    | YOLO26n      | PyTorch | ✅     | 5.3       | 0.4760              | 315.2                  |
    | YOLO26n      | DEEPX   | ✅     | 6.6       | 0.4660              | 34.6                   |
    | YOLO26n-seg  | PyTorch | ✅     | 6.5       | 0.4080              | 485.4                  |
    | YOLO26n-seg  | DEEPX   | ✅     | 7.9       | 0.3920              | 53.8                   |
    | YOLO26n-pose | PyTorch | ✅     | 7.6       | 0.4230              | 506.3                  |
    | YOLO26n-pose | DEEPX   | ✅     | 8.8       | 0.4590              | 37.6                   |
    | YOLO26n-obb  | PyTorch | ✅     | 5.7       | 0.817               | 1094.4                 |
    | YOLO26n-obb  | DEEPX   | ✅     | 7.3       | 0.783               | 56.4                   |

    | Model       | Format  | Status | Size (MB) | acc (top1) | acc (top5) | Inference time (ms/im) |
    | ----------- | ------- | ------ | --------- | ---------- | ---------- | ---------------------- |
    | YOLO26n-cls | PyTorch | ✅     | 5.6       | 0.431      | 0.716      | 23.8                   |
    | YOLO26n-cls | DEEPX   | ✅     | 5.9       | 0.333      | 0.686      | 2.7                    |

    Benchmarked with Ultralytics 8.4.48

    !!! note

        Validation for the above benchmarks was done using COCO128 for detection, COCO128-seg for segmentation, ImageNet100 for classification, COCO8-pose for pose estimation and DOTA128 for OBB models. Inference time does not include pre/post-processing.

The sizes in the tables above were recorded with that earlier version and are not comparable with the table below, which lists the sizes of current exports of the YOLO26 detection models, as the `.dxnn` file in MB (10^6 bytes) at `imgsz=640`. The export log prints the size of the whole directory in MiB, as `(3.3 MB)` for YOLO26n:

| Model   | PyTorch | DEEPX |
| ------- | ------- | ----- |
| YOLO26n | 5.5     | 3.5   |
| YOLO26s | 20.4    | 10.9  |
| YOLO26m | 44.3    | 22.0  |
| YOLO26l | 53.2    | 26.9  |
| YOLO26x | 118.7   | 59.5  |

Exported with `ultralytics` 8.4.215 and `dx_com` 2.5.0 on an x86-64 Linux server without DEEPX hardware. The [YOLO26 model page](../models/yolo26.md) lists the accuracy of the PyTorch checkpoints on the full COCO validation set.

## About DEEPX

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/e9365c5acf3cdcbfc6526467c3e58fcf.avif" alt="DEEPX NPU Inference">
</p>

[DEEPX](https://deepx.ai/) is an AI semiconductor company and [Ultralytics partner](https://www.ultralytics.com/partners/deepx) that designs NPUs for power-efficient inference in embedded and industrial devices such as robots and smart cameras. Its software consists of the DX-COM compiler, which converts ONNX models into `.dxnn` files on a PC, and the DX-RT runtime, which runs them on the NPU.

## FAQ

### Do I need a DEEPX device to export a model?

No. DX-COM compiles on an x86-64 Linux computer and needs no GPU, and every export on this page ran on a server without DEEPX hardware. You need the device, with the DX-RT runtime and `dx_engine`, only to run or validate the `.dxnn` file.

### Can I test a DEEPX model on my PC before I copy it to the device?

Not with `YOLO()` on a computer without the DX-RT runtime and `dx_engine`: loading the model fails there. Installing DX-RT on a computer without a DEEPX NPU was not tried for this page. On the export host you can validate the PyTorch checkpoint or the intermediate `yolo26n.onnx` file that the export leaves next to the directory, with [ONNX Runtime](onnx.md): the `.dxnn` file is compiled from that graph. This does not replace a validation on the device, because INT8 quantization changes the outputs.

### How many calibration images does a DEEPX export need?

DX-COM reads at most 100 calibration images, the limit that Ultralytics sets in its configuration, so put about 100 images that represent your deployment scene into the calibration split; further images are not read. The 4 images of the default `coco8.yaml` are only enough to test the export. See [Calibration](#calibration) for the dataset layout that DX-COM accepts.

### Can I export a DEEPX model on a Raspberry Pi 5?

No. The `dx_com` compiler has packages only for x86-64 Linux, and the export stops on an ARM64 machine. Export on an x86-64 Linux computer or with [Ultralytics Platform](../platform/train/models.md#export-model), then copy the `yolo26n_deepx_model/` directory to the Raspberry Pi 5.
