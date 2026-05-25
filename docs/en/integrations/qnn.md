---
comments: true
description: Learn how to export Ultralytics YOLO models locally to the Qualcomm QNN context-binary format with the ONNX Runtime QNN Execution Provider for accelerated inference on Snapdragon CPU, GPU, and Hexagon NPU hardware.
keywords: Qualcomm QNN, AI Engine Direct, QAIRT, ONNX Runtime, onnxruntime-qnn, Snapdragon, Hexagon NPU, model export, Ultralytics, YOLO, edge AI, context binary, on-device inference
---

# Qualcomm QNN Export for Ultralytics YOLO Models

Deploying computer vision models on Qualcomm Snapdragon devices requires a model format tuned for the Qualcomm AI Engine Direct (QNN) runtime. Exporting [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models to the QNN format lets you run accelerated inference across Snapdragon CPU, GPU (Adreno), and NPU (Hexagon) hardware found in billions of mobile phones, laptops, automotive systems, and IoT devices.

## What is Qualcomm QNN?

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/qnn_cover.avif" alt="Qualcomm QNN on-device inference">
</p>

[Qualcomm AI Engine Direct](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) — commonly referred to as **QNN** and distributed as part of the Qualcomm AI Runtime (QAIRT) SDK — is Qualcomm's low-level inference stack for Snapdragon processors. It provides a unified API with backend-specific libraries that target the CPU, the Adreno GPU, and the Hexagon Tensor Processor (HTP/NPU), giving developers full-stack access to the Snapdragon AI accelerators. QNN is the modern successor to the older Snapdragon Neural Processing Engine (SNPE) SDK.

## QNN Export Format

Ultralytics compiles YOLO models to QNN **locally** using the [ONNX Runtime](https://onnxruntime.ai/) QNN Execution Provider (the pip-installable `onnxruntime-qnn` package, which bundles the QAIRT libraries). The exporter converts your model to [ONNX](onnx.md), **INT8-quantizes it** with calibration data (the Hexagon NPU is an int8 accelerator), then initializes an ONNX Runtime session with context-binary caching enabled — this compiles the quantized graph into a **QNN context binary** embedded in `<model>_qnn.onnx`. No Qualcomm account, cloud upload, or separate SDK download is required.

The exported `_qnn_model/` directory bundles the context-binary ONNX and a `metadata.yaml` describing class names, image size, and task.

## Key Features of QNN Models

- **INT8 Quantization**: The model is quantized to INT8 with the ONNX Runtime QNN QDQ flow and a calibration dataset, matching the Hexagon NPU's native precision for maximum throughput and minimal size. Learn more about [model quantization](https://www.ultralytics.com/glossary/model-quantization).
- **Fully Local Compilation**: The context binary is generated entirely on your host machine — no Qualcomm account, API token, or cloud upload.
- **Full Snapdragon Acceleration**: Run inference on the Hexagon NPU (HTP), Adreno GPU, or CPU through a single unified runtime.
- **Broad Device Reach**: Target the wide range of Snapdragon platforms shipping in phones, PCs (Windows on Snapdragon), automotive, XR, and embedded products.
- **Precompiled Context Binary**: Shipping a context binary minimizes on-device graph compilation, reducing model load latency on the target.
- **Self-Contained Output**: The exported directory includes the context-binary ONNX and metadata for straightforward deployment.

## Supported Tasks

All standard Ultralytics tasks are supported for QNN export across YOLO26, YOLO11, and YOLOv8 model families.

| Task                                                           | Supported |
| :------------------------------------------------------------- | :-------- |
| [Object Detection](https://docs.ultralytics.com/tasks/detect/) | ✅        |
| [Segmentation](https://docs.ultralytics.com/tasks/segment/)    | ✅        |
| [Pose Estimation](https://docs.ultralytics.com/tasks/pose/)    | ✅        |
| [OBB Detection](https://docs.ultralytics.com/tasks/obb/)       | ✅        |
| [Classification](https://docs.ultralytics.com/tasks/classify/) | ✅        |

## Export to QNN: Converting Your YOLO Model

Export an Ultralytics YOLO model to QNN format for deployment on Snapdragon hardware. The context binary is finalized for a target Hexagon Tensor Processor (HTP) architecture, which you select with the `name` argument.

### Supported HTP Architectures

Pass the target architecture via `name` (e.g. `name="73"`). Valid values:

| `name` | Hexagon HTP | Snapdragon platform          |
| :----- | :---------- | :--------------------------- |
| `68`   | v68         | Snapdragon 865               |
| `69`   | v69         | Snapdragon 888 / 8 Gen 1     |
| `73`   | v73         | Snapdragon 8 Gen 2 (default) |
| `75`   | v75         | Snapdragon 8 Gen 3           |
| `79`   | v79         | Snapdragon 8 Elite           |

!!! note "Platform support"

    QNN export uses the `onnxruntime-qnn` package. Stable wheels are published for **Windows (x64 and ARM64)** and **Linux ARM64 (aarch64)**; a **Linux x86-64** wheel is available on the [ONNX Runtime nightly feed](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages). There is no macOS wheel — on macOS build ONNX Runtime from source with `--use_qnn`, or run the export on a supported platform. QNN context-binary generation works on an x64 host (no Snapdragon device required for the export step).

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO
        pip install ultralytics
        ```

The `onnxruntime-qnn` package (which provides the ONNX Runtime QNN Execution Provider and bundles the QAIRT libraries) is installed automatically on first export. For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

The QNN format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Inference and validation run on Qualcomm Snapdragon hardware through ONNX Runtime's QNN Execution Provider (the same `onnxruntime-qnn` package used for export). Export your model, then load the exported model on a Snapdragon device to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export to Qualcomm QNN format (INT8, enforced automatically), targeting an HTP architecture via 'name'
        # 'name' can be one of 68, 69, 73, 75, 79 (Snapdragon 865, 888/8 Gen 1, 8 Gen 2, 8 Gen 3, 8 Elite)
        model.export(format="qnn", name="73")  # creates 'yolo26n_qnn_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to Qualcomm QNN format for the target HTP architecture
        # 'name' can be one of 68, 69, 73, 75, 79 (Snapdragon 865, 888/8 Gen 1, 8 Gen 2, 8 Gen 3, 8 Elite)
        yolo export model=yolo26n.pt format=qnn name=73 # creates 'yolo26n_qnn_model/'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported QNN model (on a Snapdragon device with onnxruntime-qnn)
        model = YOLO("yolo26n_qnn_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported QNN model
        yolo predict model=yolo26n_qnn_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported QNN model (on a Snapdragon device with onnxruntime-qnn)
        model = YOLO("yolo26n_qnn_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported QNN model
        yolo val model=yolo26n_qnn_model data=coco8.yaml
        ```

### Export Arguments

| Argument   | Type             | Default        | Description                                                                                                                                                                               |
| :--------- | :--------------- | :------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'qnn'`        | Target format for the exported model, defining compatibility with the Qualcomm QNN runtime.                                                                                               |
| `imgsz`    | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)`.                                                                                 |
| `batch`    | `int`            | `1`            | Specifies the export model batch size, which is baked into the generated QNN context binary.                                                                                              |
| `name`     | `str`            | `'73'`         | Target Hexagon HTP architecture version: `68`, `69`, `73`, `75`, or `79` (Snapdragon 865, 888/8 Gen 1, 8 Gen 2, 8 Gen 3, 8 Elite). The context binary is finalized for this architecture. |
| `int8`     | `bool`           | `True`         | Enables INT8 quantization. Required for QNN HTP export — automatically set to `True` if not specified.                                                                                    |
| `data`     | `str`            | `'coco8.yaml'` | Dataset configuration file used for INT8 calibration. Specifies the calibration image source.                                                                                             |
| `fraction` | `float`          | `1.0`          | Fraction of the calibration dataset to use for INT8 quantization.                                                                                                                         |
| `device`   | `str`            | `None`         | Specifies the device for the ONNX export step: GPU (`device=0`) or CPU (`device=cpu`).                                                                                                    |

!!! note "Precision"

    The Hexagon NPU (HTP) is an int8 accelerator, so QNN export quantizes the model to **INT8** using the ONNX Runtime QNN QDQ flow with calibration images from `data`. `int8=True` is enforced automatically.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Output Structure

After a successful export, a model directory is created with the following layout:

    yolo26n_qnn_model/
    ├── yolo26n_qnn.onnx   # ONNX wrapping the precompiled QNN context binary
    └── metadata.yaml      # Model metadata (classes, image size, task, etc.)

The `yolo26n_qnn.onnx` file embeds the QNN context binary and is loaded by ONNX Runtime with the QNN Execution Provider on the Snapdragon device. The `metadata.yaml` contains class names, image size, and other information used by the Ultralytics pipeline.

## Deploying Exported YOLO QNN Models

QNN models run on Qualcomm Snapdragon hardware. On a Snapdragon device with `onnxruntime-qnn` installed, run the exported model directly with the Ultralytics API (`yolo predict`/`yolo val`, see [Usage](#usage) above) — Ultralytics loads the context binary through the [ONNX Runtime QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html) and selects the HTP (NPU), GPU, or CPU backend.

For custom pipelines, you can also load the context-binary ONNX directly with ONNX Runtime. `onnxruntime-qnn` is a plugin Execution Provider, so register it at runtime:

```python
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

# On the Snapdragon device, register the QNN plugin EP and select its device(s)
ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]

options = ort.SessionOptions()
options.add_provider_for_devices(devices, {"backend_path": qnn_ep.get_qnn_htp_path()})
session = ort.InferenceSession("yolo26n_qnn_model/yolo26n_qnn.onnx", sess_options=options)
outputs = session.run(None, {"images": input_tensor})  # input_tensor: float32 NCHW
```

Because the QNN context binary is precompiled, the session loads quickly without recompiling the graph on-device.

## Recommended Workflow

1. **Train** your model using Ultralytics [Train Mode](../modes/train.md)
2. **Export** to QNN format using `model.export(format="qnn")` on a supported platform (Windows or Linux ARM64)
3. **Deploy** the exported `_qnn_model/` directory to your Snapdragon device
4. **Run** inference with ONNX Runtime and the QNN Execution Provider, selecting the HTP, GPU, or CPU backend

## Real-World Applications

YOLO models running on Qualcomm Snapdragon hardware are well suited for a wide range of [edge AI](https://www.ultralytics.com/glossary/edge-ai) applications:

- **Smartphones**: Real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and scene understanding in camera and photo apps with NPU acceleration.
- **Windows on Snapdragon**: On-device computer vision in Copilot+ PCs without offloading to the cloud.
- **Automotive**: Driver monitoring, occupant detection, and ADAS features on Snapdragon Digital Chassis platforms.
- **XR and Wearables**: Low-power, low-latency perception for AR/VR headsets and smart glasses.
- **IoT and Robotics**: Efficient vision inference on Snapdragon-powered cameras, drones, and embedded systems.

## Summary

In this guide, you've learned how to export Ultralytics YOLO models to the Qualcomm QNN format **locally** with the ONNX Runtime QNN Execution Provider. The export pipeline converts your model to ONNX, then compiles it into a QNN context binary on your host machine — no Qualcomm account or cloud required — producing a `_qnn.onnx` optimized for Snapdragon CPU, Adreno GPU, and Hexagon NPU hardware via the QNN/QAIRT runtime.

The combination of [Ultralytics YOLO](https://www.ultralytics.com/yolo) and Qualcomm's on-device AI stack provides an effective solution for running advanced [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) workloads across the broad Snapdragon ecosystem.

Also, if you'd like to know more about other Ultralytics YOLO integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of useful resources and insights there.

## FAQ

### How do I export my Ultralytics YOLO model to QNN format?

You can export your model using the `export()` method in Python or via the CLI with `format="qnn"`. The export first creates an ONNX model, then compiles it locally into a QNN context binary using the ONNX Runtime QNN Execution Provider. The `onnxruntime-qnn` package is installed automatically on first export.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="qnn")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=qnn
        ```

### Do I need a Qualcomm account or cloud access?

No. QNN export runs entirely on your local machine using the `onnxruntime-qnn` package, which bundles the QAIRT libraries. No Qualcomm account, API token, or network access is required.

### Which platforms can I export on?

`onnxruntime-qnn` ships prebuilt wheels for **Windows (x64 and ARM64)** and **Linux ARM64 (aarch64)**. There is currently no Linux x86-64 or macOS wheel; on those hosts you can build ONNX Runtime from source with `--use_qnn`, or run the export on a supported platform. The context-binary generation step itself works on an x64 host and does not require a physical Snapdragon device.

### What is the difference between QNN and SNPE?

QNN (Qualcomm AI Engine Direct, part of the QAIRT SDK) is Qualcomm's current inference stack and the recommended replacement for the older Snapdragon Neural Processing Engine (SNPE) SDK. New deployments should target QNN.

### Can I run a QNN model with `yolo predict` and `yolo val`?

Yes, on a Qualcomm Snapdragon device with `onnxruntime-qnn` installed — `YOLO("yolo26n_qnn_model")` loads the context binary through the QNN Execution Provider and runs `predict`/`val` like any other format. On an x86 host without QNN hardware the model cannot execute, since the context binary targets the Snapdragon NPU.

### What is the output of a QNN export?

The export creates a directory (e.g., `yolo26n_qnn_model/`) containing the context-binary ONNX (`yolo26n_qnn.onnx`) and a `metadata.yaml` with class names, image size, and task information.
