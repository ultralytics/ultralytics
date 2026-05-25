---
comments: true
description: Learn how to export Ultralytics YOLO models to Qualcomm QNN format with Qualcomm AI Hub for accelerated inference on Snapdragon CPU, GPU, and Hexagon NPU hardware.
keywords: Qualcomm QNN, AI Engine Direct, QAIRT, Qualcomm AI Hub, Snapdragon, Hexagon NPU, model export, Ultralytics, YOLO, edge AI, .dlc, on-device inference
---

# Qualcomm QNN Export for Ultralytics YOLO Models

Deploying computer vision models on Qualcomm Snapdragon devices requires a model format tuned for the Qualcomm AI Engine Direct (QNN) runtime. Exporting [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models to the QNN format lets you run accelerated inference across Snapdragon CPU, GPU (Adreno), and NPU (Hexagon) hardware found in billions of mobile phones, laptops, automotive systems, and IoT devices.

## What is Qualcomm QNN?

[Qualcomm AI Engine Direct](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) — commonly referred to as **QNN** and distributed as part of the Qualcomm AI Runtime (QAIRT) SDK — is Qualcomm's low-level inference stack for Snapdragon processors. It provides a unified API with backend-specific libraries that target the CPU, the Adreno GPU, and the Hexagon Tensor Processor (HTP/NPU), giving developers full-stack access to the Snapdragon AI accelerators. QNN is the modern successor to the older Snapdragon Neural Processing Engine (SNPE) SDK.

## QNN Export Format

Ultralytics compiles YOLO models to QNN using [Qualcomm AI Hub](https://aihub.qualcomm.com/), a cloud service that converts an [ONNX](onnx.md) graph into a Qualcomm artifact for a specific Snapdragon target. Two runtimes are available:

- **QNN DLC** (`.dlc`, default): a portable QNN Deep Learning Container that can be deployed across compatible Snapdragon devices.
- **QNN context binary** (`.bin`): a device-specific, precompiled context binary for the fastest possible load time on a single target.

The exported `_qnn_model/` directory bundles the compiled model and a `metadata.yaml` describing class names, image size, and task.

## Key Features of QNN Models

- **Full Snapdragon Acceleration**: Run inference on the Hexagon NPU, Adreno GPU, or CPU through a single unified runtime.
- **Broad Device Reach**: Target the wide range of Snapdragon platforms shipping in phones, PCs (Windows on Snapdragon), automotive, XR, and embedded products.
- **Cloud Compilation**: Qualcomm AI Hub performs the conversion in the cloud, so no large local SDK installation is required for the export step.
- **Portable or Device-Specific**: Choose a portable `.dlc` for flexibility or a context binary for minimum on-device load latency.
- **Self-Contained Output**: The exported directory includes the compiled model and metadata for straightforward deployment.

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

Export an Ultralytics YOLO model to QNN format for deployment on Snapdragon hardware.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO
        pip install ultralytics
        ```

The `qai-hub` client is installed automatically on first export. Qualcomm AI Hub requires a free API token: create an account at [app.aihub.qualcomm.com](https://app.aihub.qualcomm.com/), copy your token, and configure it once:

```bash
qai-hub configure --api_token <YOUR_TOKEN>
```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before exporting, browse the available Qualcomm AI Hub devices to pick a target, then pass it via the `name` argument:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to Qualcomm QNN format for a specific Snapdragon target
        model.export(format="qnn", name="Snapdragon 8 Elite QRD")  # creates 'yolo26n_qnn_model/'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to Qualcomm QNN format
        yolo export model=yolo26n.pt format=qnn name="Snapdragon 8 Elite QRD" # creates 'yolo26n_qnn_model/'
        ```

To list every device available on your account:

```python
import qai_hub as hub

print(hub.get_devices())  # e.g. "Snapdragon 8 Elite QRD", "Samsung Galaxy S24 (Family)", ...
```

### Export Arguments

| Argument | Type             | Default                    | Description                                                                                                                       |
| :------- | :--------------- | :------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'qnn'`                    | Target format for the exported model, defining compatibility with the Qualcomm QNN runtime.                                       |
| `imgsz`  | `int` or `tuple` | `640`                      | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)`.                         |
| `name`   | `str`            | `'Snapdragon 8 Elite QRD'` | Qualcomm AI Hub target device. Run `qai_hub.get_devices()` to list every available device.                                        |
| `batch`  | `int`            | `1`                        | Specifies export model batch inference size.                                                                                      |
| `device` | `str`            | `None`                     | Specifies the device for the ONNX export step: GPU (`device=0`) or CPU (`device=cpu`). Cloud compilation runs on Qualcomm AI Hub. |

!!! tip

    QNN export requires a configured Qualcomm AI Hub API token and network access, since the model is compiled in the cloud. Compilation time depends on the AI Hub queue and the selected device.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

### Output Structure

After a successful export, a model directory is created with the following layout:

    yolo26n_qnn_model/
    ├── yolo26n.dlc      # Compiled QNN model (Deep Learning Container)
    └── metadata.yaml    # Model metadata (classes, image size, task, etc.)

The `.dlc` file is the compiled QNN model that the QNN/QAIRT runtime loads on the Snapdragon device. The `metadata.yaml` contains class names, image size, and other information used by the Ultralytics pipeline.

## Deploying Exported YOLO QNN Models

QNN models run on Qualcomm Snapdragon hardware, so local desktop inference through `yolo predict` is not supported. There are two common paths to run the exported model:

- **On-device with the QNN/QAIRT runtime**: Deploy the `.dlc` to a Snapdragon device and execute it with the [Qualcomm AI Runtime SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk), selecting the CPU, GPU, or HTP (NPU) backend.
- **Through Qualcomm AI Hub**: Submit inference and profiling jobs directly to provisioned cloud devices to validate accuracy and measure on-device latency before shipping.

### Profiling on AI Hub

You can measure real on-device latency for your compiled model on Qualcomm AI Hub:

```python
import qai_hub as hub

# Submit a profiling job for the compiled model on a Snapdragon device
profile_job = hub.submit_profile_job(
    model="yolo26n_qnn_model/yolo26n.dlc",
    device=hub.Device("Snapdragon 8 Elite QRD"),
)
print(profile_job.url)  # view detailed latency and layer-level metrics in the browser
```

## Recommended Workflow

1. **Train** your model using Ultralytics [Train Mode](../modes/train.md)
2. **Configure** your Qualcomm AI Hub token with `qai-hub configure --api_token <TOKEN>`
3. **Export** to QNN format using `model.export(format="qnn", name="<device>")`
4. **Profile** on-device latency using Qualcomm AI Hub
5. **Deploy** the exported `_qnn_model/` directory to Snapdragon hardware using the QNN/QAIRT runtime

## Real-World Applications

YOLO models running on Qualcomm Snapdragon hardware are well suited for a wide range of [edge AI](https://www.ultralytics.com/glossary/edge-ai) applications:

- **Smartphones**: Real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and scene understanding in camera and photo apps with NPU acceleration.
- **Windows on Snapdragon**: On-device computer vision in Copilot+ PCs without offloading to the cloud.
- **Automotive**: Driver monitoring, occupant detection, and ADAS features on Snapdragon Digital Chassis platforms.
- **XR and Wearables**: Low-power, low-latency perception for AR/VR headsets and smart glasses.
- **IoT and Robotics**: Efficient vision inference on Snapdragon-powered cameras, drones, and embedded systems.

## Summary

In this guide, you've learned how to export Ultralytics YOLO models to the Qualcomm QNN format using Qualcomm AI Hub. The export pipeline converts your model to ONNX, then compiles it in the cloud into a `.dlc` (or context binary) optimized for Snapdragon CPU, GPU, and Hexagon NPU hardware via the QNN/QAIRT runtime.

The combination of [Ultralytics YOLO](https://www.ultralytics.com/yolo) and Qualcomm's on-device AI stack provides an effective solution for running advanced [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) workloads across the broad Snapdragon ecosystem.

For further details, visit the [Qualcomm AI Hub documentation](https://aihub.qualcomm.com/).

Also, if you'd like to know more about other Ultralytics YOLO integrations, visit our [integration guide page](../integrations/index.md). You'll find plenty of useful resources and insights there.

## FAQ

### How do I export my Ultralytics YOLO model to QNN format?

You can export your model using the `export()` method in Python or via the CLI with `format="qnn"`. The export first creates an ONNX model, then compiles it to a QNN `.dlc` through Qualcomm AI Hub. A free AI Hub API token must be configured once with `qai-hub configure --api_token <TOKEN>`.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.export(format="qnn", name="Snapdragon 8 Elite QRD")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=qnn name="Snapdragon 8 Elite QRD"
        ```

### Do I need a Qualcomm AI Hub account?

Yes. QNN export uses Qualcomm AI Hub to compile the model in the cloud. Create a free account at [app.aihub.qualcomm.com](https://app.aihub.qualcomm.com/), then configure your API token once with `qai-hub configure --api_token <TOKEN>`. Network access is required during export.

### How do I choose the target device for export?

Pass the AI Hub device name with the `name` argument, e.g. `name="Snapdragon 8 Elite QRD"`. Run `qai_hub.get_devices()` to list all devices available to your account. A portable `.dlc` runs on compatible Snapdragon targets, so you do not need a separate export per device.

### What is the difference between QNN and SNPE?

QNN (Qualcomm AI Engine Direct, part of the QAIRT SDK) is Qualcomm's current inference stack and the recommended replacement for the older Snapdragon Neural Processing Engine (SNPE) SDK. New deployments should target QNN.

### Can I run a QNN model locally with `yolo predict`?

No. QNN models are compiled for Snapdragon hardware and are not loadable in the local Ultralytics inference pipeline. Deploy the exported `.dlc` to a Snapdragon device with the QNN/QAIRT runtime, or run inference and profiling jobs through Qualcomm AI Hub.

### What is the output of a QNN export?

The export creates a directory (e.g., `yolo26n_qnn_model/`) containing the compiled `.dlc` model (or `.bin` context binary) and a `metadata.yaml` with class names, image size, and task information.
