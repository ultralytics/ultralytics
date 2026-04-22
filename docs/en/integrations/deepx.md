---
comments: true
description: Learn how to export Ultralytics YOLO models to DeepX format for efficient deployment on DeepX NPU hardware with INT8 quantization and high-performance edge inference.
keywords: DeepX, NPU, model export, Ultralytics, YOLO, edge AI, INT8 quantization, embedded inference, dx_com, dx_engine, dxnn, deep learning, hardware acceleration
---

# DeepX Export for Ultralytics YOLO Models

Deploying computer vision models on specialized NPU hardware requires a compatible and optimized model format. Exporting [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models to DeepX format enables efficient, INT8-quantized inference on DeepX NPU accelerators. This guide walks you through converting your YOLO models to DeepX format and deploying them on DeepX-powered hardware.

## What is DeepX?

[DeepX](https://www.deepx.ai/) is an AI semiconductor company specializing in Neural Processing Units (NPUs) designed for power-efficient deep learning inference at the edge. DeepX NPUs are engineered for demanding embedded and industrial AI applications, delivering high throughput with minimal power consumption. Their hardware is well suited for deployment scenarios where cloud connectivity is unreliable or undesirable, such as robotics, smart cameras, and industrial automation systems.

## DeepX Export Format

The DeepX export produces a compiled `.dxnn` model binary that is optimized for execution on DeepX NPU hardware. The compilation pipeline uses the `dx_com` toolkit to perform INT8 quantization and hardware-specific optimization, generating a self-contained model directory ready for deployment.

## Key Features of DeepX Models

DeepX models offer several advantages for edge deployment:

- **INT8 Quantization**: Models are quantized to INT8 precision during export, significantly reducing model size and maximizing NPU throughput.
- **NPU-Optimized**: The `.dxnn` format is specifically compiled for DeepX NPU hardware, leveraging dedicated acceleration units for fast, efficient inference.
- **Low Power Consumption**: By offloading inference to the NPU, DeepX models consume far less power than equivalent CPU or GPU inference.
- **Calibration-Based Accuracy**: The export uses EMA-based calibration with real dataset images to minimize accuracy loss during quantization.
- **Self-Contained Output**: The exported model directory bundles the compiled binary, calibration config, and metadata for straightforward deployment.

## Export to DeepX: Converting Your YOLO Model

Export an Ultralytics YOLO model to DeepX format and run inference with the exported model.

!!! note

    DeepX export is only supported on **x86-64 Linux** machines. ARM64 (aarch64) is not supported for the export step.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO
        pip install ultralytics
        ```

The `dx_com` compiler package will be automatically installed from the DeepX SDK repository on first export. For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to DeepX format (int8=True is enforced automatically)
        model.export(format="deepx")  # creates 'yolo26n_deepx_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to DeepX format
        yolo export model=yolo26n.pt format=deepx # creates 'yolo26n_deepx_model/'
        ```

### Export Arguments

| Argument    | Type             | Default          | Description                                                                                                                             |
| :---------- | :--------------- | :--------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| `format`    | `str`            | `'deepx'`        | Target format for the exported model, defining compatibility with DeepX NPU hardware.                                                   |
| `imgsz`     | `int` or `tuple` | `640`            | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.       |
| `batch`     | `int`            | `1`              | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `int8`      | `bool`           | `True`           | Enables INT8 quantization. Required for DeepX export — automatically set to `True` if not specified.                                    |
| `data`      | `str`            | `'coco128.yaml'` | Dataset configuration file used for INT8 calibration. Specifies the calibration image source.                                           |
| `fraction`  | `float`          | `1.0`            | Fraction of the calibration dataset to use. Reduce to speed up export; 100–400 images are typically sufficient for good accuracy.       |
| `device`    | `str`            | `None`           | Specifies the device for exporting: GPU (`device=0`) or CPU (`device=cpu`).                                                             |
| `opt_level` | `int`            | `0`              | Optimization level for the DeepX compiler (`0` or `1`). Higher levels reduce inference latency but increase compilation time.           |

!!! tip

    Always run DeepX export on an **x86-64 Linux** host. The `dx_com` compiler does not support ARM64.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Output Structure

After a successful export, a model directory is created with the following layout:

```
yolo26n_deepx_model/
├── yolo26n.dxnn     # Compiled DeepX model binary (NPU executable)
├── config.json      # Calibration and preprocessing configuration
└── metadata.yaml    # Model metadata (classes, image size, task, etc.)
```

The `.dxnn` file is the compiled model binary that the `dx_engine` runtime loads directly on the NPU. The `metadata.yaml` contains class names, image size, and other information used by the Ultralytics inference pipeline.

### Visualizing with dxtron

During export, the [dxtron](https://sdk.deepx.ai/) visualizer is automatically installed on x86-64 Linux if not already present. You can use it to inspect your exported model:

```bash
dxtron yolo26n_deepx_model/yolo26n.dxnn
```

## Deploying Exported YOLO DeepX Models

Once you've successfully exported your Ultralytics YOLO model to DeepX format, the next step is deploying these models on DeepX NPU hardware.

### Installation

The `dx_engine` runtime package is required for inference on DeepX NPU hardware. The backend will attempt to install it automatically on first use:

1. **Debian Trixie (ARM64)**: Installs automatically via the [Sixfab APT repository](https://github.com/sixfab/sixfab_dx/).
2. **Other platforms**: The backend automatically downloads and installs the NPU driver and runtime from GitHub, then installs the `dx_engine` wheel from `/usr/share/libdxrt/src/python_package/`.

!!! tip "Controlling install verbosity"

    Set `YOLO_VERBOSE=false` to suppress detailed installation output (dpkg, pip):

    ```bash
    YOLO_VERBOSE=false python predict_dxnn_deepx.py
    ```

If automatic installation fails, install the runtime manually:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO
        pip install ultralytics
        ```

!!! warning "Manual Runtime Installation (non-Trixie platforms)"

    If the DeepX runtime is not available on your system, install it manually:

    1. **Install the NPU driver:**
        ```bash
        # Download from https://github.com/DEEPX-AI/dx_rt_npu_linux_driver/blob/main/release/2.4.0/
        sudo dpkg -i dxrt-driver-dkms_2.4.0-2_all.deb
        ```

    2. **Install the runtime library:**
        ```bash
        # Download from https://github.com/DEEPX-AI/dx_rt/blob/main/release/3.3.0/
        sudo dpkg -i libdxrt_3.3.0_all.deb
        ```

    3. **Install the `dx_engine` Python package:**
        ```bash
        pip install /usr/share/libdxrt/src/python_package/dx_engine-*.whl
        ```

    !!! note "PEP 668 (Python 3.12+)"

        On systems with Python 3.12+, `pip install` outside a virtual environment may fail due to [PEP 668](https://peps.python.org/pep-0668/). Create and activate a virtual environment first:

        ```bash
        python3 -m venv /path/to/venv
        source /path/to/venv/bin/activate
        pip install /usr/share/libdxrt/src/python_package/dx_engine-*.whl
        ```

### Usage

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported DeepX model
        model = YOLO("yolo26n_deepx_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")

        # Process results
        for r in results:
            print(f"Detected {len(r.boxes)} objects")
            r.show()
        ```

    === "CLI"

        ```bash
        # Run inference with the exported DeepX model
        yolo predict model='yolo26n_deepx_model' source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    The DeepX backend converts each input image from normalized float `[0, 1]` in BCHW format to uint8 `[0, 255]` in HWC format before passing it to the NPU runtime, as required by the `dx_engine` inference contract.

## Real-World Applications

YOLO models deployed on DeepX NPU hardware are well suited for a wide range of edge AI applications:

- **Smart Surveillance**: Real-time object detection for security and monitoring systems with low power consumption and no cloud dependency.
- **Industrial Automation**: On-device quality control, defect detection, and process monitoring in factory environments.
- **Robotics**: Vision-based navigation, obstacle avoidance, and object recognition on autonomous robots and drones.
- **Smart Agriculture**: Crop health monitoring, pest detection, and yield estimation using [computer vision in agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture).
- **Retail Analytics**: Customer flow analysis, shelf monitoring, and inventory tracking with real-time edge inference.

## Summary

In this guide, you've learned how to export Ultralytics YOLO models to DeepX format and deploy them on DeepX NPU hardware. The export pipeline uses INT8 calibration and the `dx_com` compiler to produce a hardware-optimized `.dxnn` binary, while the `dx_engine` runtime handles inference on the device.

The combination of [Ultralytics YOLO](https://www.ultralytics.com/yolo) and DeepX's NPU technology provides an effective solution for running advanced computer vision workloads on embedded and edge devices — delivering high throughput with low power consumption for real-time applications.

For further details on usage, visit the [DeepX official website](https://www.deepx.ai/).

Also, if you'd like to know more about other Ultralytics YOLO integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of useful resources and insights there.

## FAQ

### How do I export my Ultralytics YOLO model to DeepX format?

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

### Why does DeepX export require INT8 quantization?

DeepX NPUs are designed to execute INT8 computations at maximum efficiency. The `dx_com` compiler quantizes the model during export using EMA-based calibration with real dataset images, enabling the NPU to deliver its full performance. INT8 is always enforced for DeepX exports — if you pass `int8=False`, it will be overridden with a warning.

### What platforms are supported for DeepX export?

DeepX model export (compilation) requires an **x86-64 Linux** host. The export step is not supported on ARM64/aarch64 or Windows machines. Inference using the exported `.dxnn` model can be run on any Linux platform supported by the `dx_engine` runtime (x86-64 and ARM64).

### What is the output of a DeepX export?

The export creates a directory (e.g., `yolo26n_deepx_model/`) containing:

- `yolo26n.dxnn` — the compiled NPU binary
- `config.json` — calibration and preprocessing settings
- `metadata.yaml` — model metadata including class names and image size

### Can I deploy custom-trained models on DeepX hardware?

Yes. Any model trained using [Ultralytics Train Mode](../modes/train.md) and exported with `format="deepx"` can be deployed on DeepX NPU hardware, provided it uses supported layer operations. Export supports detection, segmentation, pose estimation, oriented bounding box (OBB), and classification tasks.

### How many calibration images should I use for DeepX export?

The DeepX export pipeline defaults to 100 calibration images using the EMA calibration method. This is generally sufficient for good quantization accuracy. You can adjust the calibration dataset using the `data` and `fraction` arguments, but using more than a few hundred images rarely improves results significantly.

### How do I install the DeepX runtime on non-Trixie platforms?

On platforms other than Debian Trixie (ARM64), the runtime is not auto-installed via APT. You need to manually install the NPU driver (`dxrt-driver-dkms`), runtime library (`libdxrt`), and the `dx_engine` Python wheel. The backend will attempt to find and install the wheel from `/usr/share/libdxrt/src/python_package/` automatically, but if that fails, see the [Manual Runtime Installation](#installation_1) section above for step-by-step instructions. If you encounter PEP 668 errors on Python 3.12+, use a virtual environment.
