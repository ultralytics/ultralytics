---
comments: true
description: Export Ultralytics YOLO models to the DEEPX .dxnn format for INT8 inference on DEEPX NPUs, with runtime setup and Raspberry Pi 5 benchmarks.
keywords: DEEPX, NPU, model export, Ultralytics, YOLO, edge AI, INT8 quantization, embedded inference, dx_com, dx_engine, dxnn, deep learning, hardware acceleration
---

# DEEPX Export for Ultralytics YOLO Models

Deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models on specialized NPU hardware requires a compatible and optimized model format. Exporting [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models to DEEPX format enables efficient, INT8-quantized inference on DEEPX NPU accelerators. This guide walks you through converting your YOLO models to DEEPX format and deploying them on DEEPX-powered hardware.

## What is DEEPX?

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/e9365c5acf3cdcbfc6526467c3e58fcf.avif" alt="DEEPX NPU Inference">
</p>

[DEEPX](https://deepx.ai/) is an AI semiconductor company and [Ultralytics partner](https://www.ultralytics.com/partners/deepx) specializing in [Neural Processing Units](https://www.ultralytics.com/glossary/neural-processing-unit-npu) (NPUs) designed for power-efficient [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) inference at the edge. DEEPX NPUs are engineered for demanding embedded and industrial AI applications, delivering high throughput with minimal power consumption. Their hardware is well suited for deployment scenarios where cloud connectivity is unreliable or undesirable, such as robotics, smart cameras, and industrial automation systems.

## DEEPX Export Format

The DEEPX export produces a compiled `.dxnn` model binary that is optimized for execution on DEEPX NPU hardware. The compilation pipeline uses the `dx_com` toolkit to perform INT8 quantization and hardware-specific optimization, generating a self-contained model directory ready for deployment.

## Key Features of DEEPX Models

DEEPX models offer several advantages for edge deployment:

- **INT8 Quantization**: Models are quantized to INT8 precision during export, significantly reducing model size and maximizing NPU throughput. Learn more about [model quantization](https://www.ultralytics.com/glossary/model-quantization).
- **NPU-Optimized**: The `.dxnn` format is specifically compiled for DEEPX NPU hardware, leveraging dedicated acceleration units for fast, efficient inference.
- **Low Power Consumption**: By offloading inference to the NPU, DEEPX models consume far less power than equivalent CPU or GPU inference.
- **Calibration-Based Accuracy**: The export uses EMA-based calibration with real dataset images to minimize accuracy loss during quantization.
- **Self-Contained Output**: The exported model directory bundles the compiled binary, calibration config, and metadata for straightforward deployment.

## Supported Tasks

DEEPX export supports all seven Ultralytics tasks. Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

{% include "macros/supported-tasks.md" %}

## Export to DEEPX: Converting Your YOLO Model

Export an Ultralytics YOLO model to DEEPX format and run inference with the exported model.

!!! note

    DEEPX export is only supported on x86-64 Linux machines. ARM64 (aarch64) is not supported for the export step. However, the exported dxnn models are fully compatible and executable on ARM64 platforms.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO
        pip install ultralytics
        ```

The `dx_com` compiler is automatically installed from [PyPI](https://pypi.org/project/dx-com/) on first export. DX-COM 2.5.0 provides wheels for Python 3.8–3.14 on x86-64 Linux with glibc 2.31 or newer and requires at least 16 GB of RAM. To preinstall the export dependencies:

```bash
pip install "ultralytics[export-deepx]"
```

For an editable repository install, replace `"ultralytics[export-deepx]"` with `-e ".[export-base,export-deepx]"`. To reproduce the Python 3.14 environment and smoke export used by CI, run the existing environment builder from the repository root:

```bash
ULTRALYTICS_ISOLATED_VENVS="$PWD/.venvs" python .github/scripts/create-export-env.py --env isolated-deepx
```

The environment builder requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and an installed Ultralytics checkout. It applies the tested dependency constraints automatically.

!!! warning "Runtime compatibility"

    DX-COM 2.5.0 produces `.dxnn` file format v9, which only loads on DX-RT 3.5.0 or newer. Install the matching DEEPX runtime described in [Runtime Installation](#runtime-installation), and check the [DEEPX version compatibility matrix](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/04_Version_Compatibility.md) before upgrading individual components.

### Usage

The DEEPX format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Inference and validation run on DEEPX NPU hardware. Export your model, then load the exported model to run inference or validate its accuracy.

!!! note

    Predict and Validate require a DEEPX NPU device with the DEEPX runtime installed (see [Runtime Installation](#runtime-installation)). Export runs on an x86-64 Linux host without an NPU; copy the exported `_deepx_model/` directory to the NPU device to run inference.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to DEEPX format (quantize=8 is enforced automatically)
        model.export(format="deepx")  # creates 'yolo26n_deepx_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to DEEPX format
        yolo export model=yolo26n.pt format=deepx # creates 'yolo26n_deepx_model/'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported DEEPX model
        model = YOLO("yolo26n_deepx_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported DEEPX model
        yolo predict model=yolo26n_deepx_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported DEEPX model
        model = YOLO("yolo26n_deepx_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported DEEPX model
        yolo val model=yolo26n_deepx_model data=coco8.yaml
        ```

### Export Arguments

| Argument   | Type             | Default   | Description                                                                                                                                                                                                                |
| :--------- | :--------------- | :-------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'deepx'` | Target format for the exported model, defining compatibility with DEEPX NPU hardware.                                                                                                                                      |
| `imgsz`    | `int` or `tuple` | `640`     | Desired image size for the model input. DEEPX export requires a square input — pass an integer (e.g., `640`) or a tuple where height equals width.                                                                         |
| `quantize` | `int` or `str`   | `8`/auto  | Quantization precision. `8` (INT8) is required for DEEPX export and auto-enabled if not specified. Replaces the deprecated `half`/`int8` flags.                                                                            |
| `simplify` | `bool`           | `True`    | Simplifies the intermediate ONNX graph with `onnxslim`.                                                                                                                                                                    |
| `opset`    | `int`            | `None`    | Specifies the ONNX opset version for the intermediate ONNX graph. If not set, uses opset `18` (lower on PyTorch < 2.4), capped at the installed `onnx` version.                                                            |
| `data`     | `str`            | `None`    | Dataset YAML used for INT8 calibration; classification instead takes a dataset directory or a built-in dataset name. If omitted with `quantize=8`, Ultralytics selects the default calibration dataset for the model task. |
| `device`   | `str`            | `None`    | Specifies the device for exporting: GPU (`device=0`) or CPU (`device=cpu`).                                                                                                                                                |
| `optimize` | `bool`           | `False`   | Enables higher compiler optimization which reduces inference latency and increases compilation time.                                                                                                                       |

!!! tip

    Always run DEEPX export on an **x86-64 Linux** host. The `dx_com` compiler does not support ARM64.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Output Structure

After a successful export, a model directory is created with the following layout:

```text
yolo26n_deepx_model/
├── yolo26n.dxnn          # Compiled DEEPX model binary (NPU executable)
├── yolo26n_summary.html  # DX-COM Compilation Summary Report
├── config.json           # Calibration and preprocessing configuration
└── metadata.yaml         # Model metadata (classes, image size, task, etc.)
```

The `.dxnn` file is the compiled model binary that the `dx_engine` runtime loads directly on the NPU. The `metadata.yaml` contains class names, image size, and other information used by the Ultralytics inference pipeline. The `yolo26n_summary.html` report is described in [Inspecting the Compiled Model](#inspecting-the-compiled-model).

## Deploying Exported YOLO DEEPX Models

Once you've successfully exported your Ultralytics YOLO model to DEEPX format, the next step is deploying these models on DEEPX NPU hardware.

### Runtime Installation

Inference requires the DEEPX NPU driver (v2.7.0 or newer), the DX-RT runtime (v3.5.0 or newer), NPU firmware v2.7.6 or newer, and the `dx_engine` Python package. These are the versions required to run models compiled with DX-COM 2.5.0.

!!! note

    DEEPX runtime supports both x86-64 Linux and ARM64 (e.g., Raspberry Pi 5).

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

#### Python Package and Firmware

`libdxrt-bin` installs `dx_engine` into the system Python and bundles wheels for Python 3.8–3.14 in `/usr/share/libdxrt-bin/python/`. To use `dx_engine` inside a virtual environment, install the wheel that matches your Python version:

```bash
pip install --no-index --find-links /usr/share/libdxrt-bin/python dx_engine
```

DX-RT 3.5.0 requires NPU firmware v2.7.6 or newer. Check the firmware version with `dxcli -s` and, if it is older, update it with the firmware image for your module. DX-M1 and DX-M1M M.2 modules use different firmware images, so download only the one that matches your module; `dxcli -s` reports the module as `M1` or `M1M` in its device information:

```bash
# Download the firmware image for your module (DX-M1 by default)
wget https://github.com/DEEPX-AI/dx_fw/raw/main/m1/2.7.6/mdot2/fw.bin
# wget https://github.com/DEEPX-AI/dx_fw/raw/main/m1m/2.7.6/mdot2/fw.bin # use this line instead for DX-M1M

# Update the firmware
sudo systemctl stop dxrt.service
dxcli -u fw.bin
sudo systemctl start dxrt.service
```

Verify the installation with `dxcli -s`, which reports the runtime, driver, and firmware versions of each device. `import dx_engine` raises a `RuntimeError` if the `dx_engine` wheel and the installed `libdxrt` runtime versions differ, so reinstall the wheel from `/usr/share/libdxrt-bin/python/` after upgrading the runtime.

Once the runtime is installed, run inference and validation on your DEEPX device exactly as shown in the [Usage](#usage) section above — the exported `_deepx_model` loads directly with `YOLO(...)`.

### Inspecting the Compiled Model

DX-COM 2.5.0 replaces the DX-TRON graph visualizer with an interactive HTML Compilation Summary Report that shows the compiled model structure and the CPU/NPU workload split. To generate it, recompile the intermediate ONNX model written by the export with the `dxcom` CLI on the x86-64 export host:

```bash
dxcom -m yolo26n.onnx -c yolo26n_deepx_model/config.json -o yolo26n_deepx_report --export_html
```

Open `yolo26n_deepx_report/yolo26n_summary.html` in a browser to view the report.

## Benchmarks

The Ultralytics team benchmarked YOLO26 models, comparing speed and accuracy between PyTorch and DEEPX.

!!! tip "Performance"

    === "Raspberry Pi 5 + DX-M1 M.2 Module"

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

    === "More devices coming soon!"

    Benchmarked with Ultralytics 8.4.48

    !!! note

        Validation for the above benchmarks was done using COCO128 for detection, COCO128-seg for segmentation, ImageNet100 for classification, COCO8-pose for pose estimation and DOTA128 for OBB models. Inference time does not include pre/post-processing.

!!! tip "Performance Optimization Tips"

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

## Recommended Workflow

1. **Train** your model using Ultralytics [Train Mode](../modes/train.md)
2. **Export** to DEEPX format using `model.export(format="deepx")`
3. **Validate** accuracy with `yolo val` to verify minimal quantization loss
4. **Predict** using `yolo predict` for qualitative validation
5. **Deploy** the exported `_deepx_model/` directory to DEEPX NPU hardware using the `dx_engine` runtime

## Real-World Applications

YOLO models deployed on DEEPX NPU hardware are well suited for a wide range of [edge AI](https://www.ultralytics.com/glossary/edge-ai) applications:

- **Smart Surveillance**: Real-time [object detection](https://www.ultralytics.com/glossary/object-detection) for security and monitoring systems with low power consumption and no cloud dependency.
- **Industrial Automation**: On-device quality control, defect detection, and process monitoring in factory environments.
- **Robotics**: Vision-based navigation, obstacle avoidance, and object recognition on autonomous robots and drones.
- **Smart Agriculture**: Crop health monitoring, pest detection, and yield estimation using [computer vision in agriculture](https://www.ultralytics.com/solutions/computer-vision-in-agriculture).
- **Retail Analytics**: Customer flow analysis, shelf monitoring, and inventory tracking with real-time edge inference.

## Summary

In this guide, you've learned how to export Ultralytics YOLO models to DEEPX format and deploy them on DEEPX NPU hardware. The export pipeline uses INT8 calibration and the `dx_com` compiler to produce a hardware-optimized `.dxnn` binary, while the `dx_engine` runtime handles inference on the device.

Together, [Ultralytics YOLO](https://www.ultralytics.com/yolo) and [DEEPX NPUs](https://www.ultralytics.com/blog/ultralytics-yolo-partners-with-deepx-edge-ai-inference-for-physical-ai) run computer vision workloads on embedded and edge devices with high throughput and low power consumption for real-time applications.

For further details on usage, visit the [DEEPX official website](https://deepx.ai/).

## FAQ

### How do I export my Ultralytics YOLO model to DEEPX format?

You can export your model using the `export()` method in Python or via the CLI. The export automatically enables INT8 quantization and uses a calibration dataset to minimize accuracy loss. The `dx_com` compiler package is installed automatically if not already present.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="deepx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=deepx
        ```

### Why does DEEPX export require INT8 quantization?

DEEPX NPUs are designed to execute INT8 computations at maximum efficiency. The `dx_com` compiler quantizes the model during export using EMA-based calibration with real dataset images, enabling the NPU to deliver its full performance. INT8 is always enforced for DEEPX exports — if you request a different precision, it will be overridden with a warning.

### What platforms are supported for DEEPX export?

DEEPX model export (compilation) requires an **x86-64 Linux** host. The export step is not supported on ARM64 (aarch64) and Windows machines. Inference using the exported `.dxnn` model can be run on any Linux platform (x86-64 and ARM64) supported by the `dx_engine` runtime.

### What is the output of a DEEPX export?

The export creates a directory (e.g., `yolo26n_deepx_model/`) containing:

- `yolo26n.dxnn` — the compiled NPU binary
- `yolo26n_summary.html` — the DX-COM Compilation Summary Report
- `config.json` — calibration and preprocessing settings
- `metadata.yaml` — model metadata including class names and image size

### Can I deploy custom-trained models on DEEPX hardware?

Yes. Any model trained using [Ultralytics Train Mode](../modes/train.md) and exported with `format="deepx"` can be deployed on DEEPX NPU hardware, provided it uses supported layer operations. Export supports all seven Ultralytics tasks: detection, instance segmentation, semantic segmentation, depth estimation, classification, pose estimation, and oriented bounding box (OBB).

### How many calibration images should I use for DEEPX export?

The DEEPX export pipeline configures the `dx_com` compiler for 100 EMA calibration steps over images from the calibration dataset. A few hundred images is usually sufficient for good quantization accuracy. Point `data` at a smaller dataset if compilation time becomes a concern on large datasets.

### How do I install the DEEPX runtime for inference?

The DEEPX runtime is not bundled with `ultralytics` and must be installed separately before running inference. On x86-64 Linux machines and ARM64 Linux machines (e.g., Raspberry Pi 5), install the NPU driver (`dxrt-driver-dkms`) and runtime (`libdxrt-bin`) from the DEEPX-AI GitHub release packages or the DEEPX apt repository, update the NPU firmware to v2.7.6 or newer if needed, and install the bundled `dx_engine` Python wheel into your environment. Models exported with DX-COM 2.5.0 require DX-RT 3.5.0 or newer. See the [Runtime Installation](#runtime-installation) section above for step-by-step commands.
