---
comments: true
description: Learn to export Ultralytics YOLO11 models to Sony's IMX500 format for efficient edge AI deployment on Raspberry Pi AI Camera with on-chip processing.
keywords: Sony, IMX500, IMX 500, Atrios, MCT, model export, quantization, pruning, deep learning optimization, Raspberry Pi AI Camera, edge AI, PyTorch, IMX
---

# Sony IMX500 Export for Ultralytics YOLO11

This guide covers exporting and deploying Ultralytics YOLO11 models to Raspberry Pi AI Cameras that feature the Sony IMX500 sensor.

Deploying computer vision models on devices with limited computational power, such as [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/), can be tricky. Using a model format optimized for faster performance makes a huge difference.

The IMX500 model format is designed to use minimal power while delivering fast performance for neural networks. It allows you to optimize your [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) models for high-speed and low-power inferencing. In this guide, we'll walk you through exporting and deploying your models to the IMX500 format while making it easier for your models to perform well on the [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/).

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/assets/releases/download/v8.3.0/ai-camera.avif" alt="Raspberry Pi AI Camera">
</p>

## Why Should You Export to IMX500

Sony's [IMX500 Intelligent Vision Sensor](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera) is a game-changing piece of hardware in edge AI processing. It's the world's first intelligent vision sensor with on-chip AI capabilities. This sensor helps overcome many challenges in edge AI, including data processing bottlenecks, privacy concerns, and performance limitations.
While other sensors merely pass along images and frames, the IMX500 tells a whole story. It processes data directly on the sensor, allowing devices to generate insights in real-time.

## Sony's IMX500 Export for YOLO11 Models

The IMX500 is designed to transform how devices handle data directly on the sensor, without needing to send it off to the cloud for processing.

The IMX500 works with quantized models. Quantization makes models smaller and faster without losing much [accuracy](https://www.ultralytics.com/glossary/accuracy). It is ideal for the limited resources of edge computing, allowing applications to respond quickly by reducing latency and allowing for quick data processing locally, without cloud dependency. Local processing also keeps user data private and secure since it's not sent to a remote server.

**IMX500 Key Features:**

- **Metadata Output:** Instead of transmitting images only, the IMX500 can output both image and metadata (inference result), and can output metadata only for minimizing data size, reducing bandwidth, and lowering costs.
- **Addresses Privacy Concerns:** By processing data on the device, the IMX500 addresses privacy concerns, ideal for human-centric applications like person counting and occupancy tracking.
- **Real-time Processing:** Fast, on-sensor processing supports real-time decisions, perfect for edge AI applications such as autonomous systems.

**Before You Begin:** For best results, ensure your YOLO11 model is well-prepared for export by following our [Model Training Guide](https://docs.ultralytics.com/modes/train/), [Data Preparation Guide](https://docs.ultralytics.com/datasets/), and [Hyperparameter Tuning Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/).

## Supported Tasks

Currently, you can only export models that include the following tasks to IMX500 format.

- [Object detection](https://docs.ultralytics.com/tasks/detect/)
- [Pose estimation](https://docs.ultralytics.com/tasks/pose/)

## Usage Examples

Export an Ultralytics YOLO11 model to IMX500 format and run inference with the exported model.

!!! note

    Here we perform inference just to make sure the model works as expected. However, for deployment and inference on the Raspberry Pi AI Camera, please jump to [Using IMX500 Export in Deployment](#using-imx500-export-in-deployment) section.

!!! example "Object Detection"

    === "Python"

         ```python
         from ultralytics import YOLO

         # Load a YOLO11n PyTorch model
         model = YOLO("yolo11n.pt")

         # Export the model
         model.export(format="imx", data="coco8.yaml")  # exports with PTQ quantization by default

         # Load the exported model
         imx_model = YOLO("yolo11n_imx_model")

         # Run inference
         results = imx_model("https://ultralytics.com/images/bus.jpg")
         ```

    === "CLI"

         ```bash
         # Export a YOLO11n PyTorch model to imx format with Post-Training Quantization (PTQ)
         yolo export model=yolo11n.pt format=imx data=coco8.yaml

         # Run inference with the exported model
         yolo predict model=yolo11n_imx_model source='https://ultralytics.com/images/bus.jpg'
         ```

!!! example "Pose Estimation"

    === "Python"

         ```python
         from ultralytics import YOLO

         # Load a YOLO11n-pose PyTorch model
         model = YOLO("yolo11n-pose.pt")

         # Export the model
         model.export(format="imx", data="coco8-pose.yaml")  # exports with PTQ quantization by default

         # Load the exported model
         imx_model = YOLO("yolo11n-pose_imx_model")

         # Run inference
         results = imx_model("https://ultralytics.com/images/bus.jpg")
         ```

    === "CLI"

         ```bash
         # Export a YOLO11n-pose PyTorch model to imx format with Post-Training Quantization (PTQ)
         yolo export model=yolo11n-pose.pt format=imx data=coco8-pose.yaml

         # Run inference with the exported model
         yolo predict model=yolo11n-pose_imx_model source='https://ultralytics.com/images/bus.jpg'
         ```

!!! warning

    The Ultralytics package installs additional export dependencies at runtime. The first time you run the export command, you may need to restart your console to ensure it works correctly.

## Export Arguments

| Argument   | Type             | Default        | Description                                                                                                                                                                                                                                                      |
| ---------- | ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'imx'`        | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                               |
| `imgsz`    | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                                                                                                |
| `int8`     | `bool`           | `True`         | Activates INT8 quantization, further compressing the model and speeding up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for edge devices.                                                                    |
| `data`     | `str`            | `'coco8.yaml'` | Path to the [dataset](https://docs.ultralytics.com/datasets/) configuration file (default: `coco8.yaml`), essential for quantization.                                                                                                                            |
| `fraction` | `float`          | `1.0`          | Specifies the fraction of the dataset to use for INT8 quantization calibration. Allows for calibrating on a subset of the full dataset, useful for experiments or when resources are limited. If not specified with INT8 enabled, the full dataset will be used. |
| `device`   | `str`            | `None`         | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`).                                                                                                                                                                                        |

!!! tip

    If you are exporting on a GPU with CUDA support, please pass the argument `device=0` for faster export.

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

The export process will create an ONNX model for quantization validation, along with a directory named `<model-name>_imx_model`. This directory will include the `packerOut.zip` file, which is essential for packaging the model for deployment on the IMX500 hardware. Additionally, the `<model-name>_imx_model` folder will contain a text file (`labels.txt`) listing all the labels associated with the model.

!!! example "Folder Structure"

    === "Object Detection"

        ```bash
        yolo11n_imx_model
        ├── dnnParams.xml
        ├── labels.txt
        ├── packerOut.zip
        ├── yolo11n_imx.onnx
        ├── yolo11n_imx_MemoryReport.json
        └── yolo11n_imx.pbtxt
        ```

    === "Pose Estimation"

        ```bash
        yolo11n-pose_imx_model
        ├── dnnParams.xml
        ├── labels.txt
        ├── packerOut.zip
        ├── yolo11n-pose_imx.onnx
        ├── yolo11n-pose_imx_MemoryReport.json
        └── yolo11n-pose_imx.pbtxt
        ```

## Using IMX500 Export in Deployment

After exporting Ultralytics YOLO11n model to IMX500 format, it can be deployed to Raspberry Pi AI Camera for inference.

### Hardware Prerequisites

Make sure you have the below hardware:

1. Raspberry Pi 5 or Raspberry Pi 4 Model B
2. Raspberry Pi AI Camera

Connect the Raspberry Pi AI camera to the 15-pin MIPI CSI connector on the Raspberry Pi and power on the Raspberry Pi

### Software Prerequisites

!!! note

    This guide has been tested with Raspberry Pi OS Bookworm running on a Raspberry Pi 5

Step 1: Open a terminal window and execute the following commands to update the Raspberry Pi software to the latest version.

```bash
sudo apt update && sudo apt full-upgrade
```

Step 2: Install IMX500 firmware which is required to operate the IMX500 sensor.

```bash
sudo apt install imx500-all
```

Step 3: Reboot Raspberry Pi for the changes to take into effect

```bash
sudo reboot
```

Step 4: Install [Aitrios Raspberry Pi application module library](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library)

```bash
pip install git+https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git
```

Step 5: Run YOLO11 object detection and pose estimation by using the below scripts which are available in [aitrios-rpi-application-module-library examples](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library/tree/main/examples/aicam).

!!! note

    Make sure to replace `model_file` and `labels.txt` directories according to your environment before running these scripts.

!!! example "Python Scripts"

    === "Object Detection"

        ```python
        import numpy as np
        from modlib.apps import Annotator
        from modlib.devices import AiCamera
        from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
        from modlib.models.post_processors import pp_od_yolo_ultralytics


        class YOLO(Model):
            """YOLO model for IMX500 deployment."""

            def __init__(self):
                """Initialize the YOLO model for IMX500 deployment."""
                super().__init__(
                    model_file="yolo11n_imx_model/packerOut.zip",  # replace with proper directory
                    model_type=MODEL_TYPE.CONVERTED,
                    color_format=COLOR_FORMAT.RGB,
                    preserve_aspect_ratio=False,
                )

                self.labels = np.genfromtxt(
                    "yolo11n_imx_model/labels.txt",  # replace with proper directory
                    dtype=str,
                    delimiter="\n",
                )

            def post_process(self, output_tensors):
                """Post-process the output tensors for object detection."""
                return pp_od_yolo_ultralytics(output_tensors)


        device = AiCamera(frame_rate=16)  # Optimal frame rate for maximum DPS of the YOLO model running on the AI Camera
        model = YOLO()
        device.deploy(model)

        annotator = Annotator()

        with device as stream:
            for frame in stream:
                detections = frame.detections[frame.detections.confidence > 0.55]
                labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]

                annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)
                frame.display()
        ```

    === "Pose Estimation"

        ```python
        from modlib.apps import Annotator
        from modlib.devices import AiCamera
        from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
        from modlib.models.post_processors import pp_yolo_pose_ultralytics


        class YOLOPose(Model):
            """YOLO pose estimation model for IMX500 deployment."""

            def __init__(self):
                """Initialize the YOLO pose estimation model for IMX500 deployment."""
                super().__init__(
                    model_file="yolo11n-pose_imx_model/packerOut.zip",  # replace with proper directory
                    model_type=MODEL_TYPE.CONVERTED,
                    color_format=COLOR_FORMAT.RGB,
                    preserve_aspect_ratio=False,
                )

            def post_process(self, output_tensors):
                """Post-process the output tensors for pose estimation."""
                return pp_yolo_pose_ultralytics(output_tensors)


        device = AiCamera(frame_rate=17)  # Optimal frame rate for maximum DPS of the YOLO-pose model running on the AI Camera
        model = YOLOPose()
        device.deploy(model)

        annotator = Annotator()

        with device as stream:
            for frame in stream:
                detections = frame.detections[frame.detections.confidence > 0.4]

                annotator.annotate_keypoints(frame, detections)
                annotator.annotate_boxes(frame, detections, corner_length=20)
                frame.display()
        ```

## Benchmarks

YOLOv8n, YOLO11n, YOLOv8n-pose and YOLO11n-pose benchmarks below were run by the Ultralytics team on Raspberry Pi AI Camera with `imx` model format measuring speed and accuracy.

| Model        | Format | Status | Size of `packerOut.zip` (MB) | mAP50-95(B) | Inference time (ms/im) |
| ------------ | ------ | ------ | ---------------------------- | ----------- | ---------------------- |
| YOLOv8n      | imx    | ✅     | 2.1                          | 0.470       | 58.79                  |
| YOLO11n      | imx    | ✅     | 2.2                          | 0.517       | 58.82                  |
| YOLOv8n-pose | imx    | ✅     | 2.0                          | 0.687       | 58.79                  |
| YOLO11n-pose | imx    | ✅     | 2.1                          | 0.788       | 62.50                  |

!!! note

    Validation for the above benchmarks were done using COCO128 dataset for detection models and COCO8-Pose dataset for pose estimation models

## What's Under the Hood?

<p align="center">
  <img width="640" src="https://github.com/ultralytics/assets/releases/download/v8.3.0/imx500-deploy.avif" alt="IMX500 deployment">
</p>

### Sony Model Compression Toolkit (MCT)

[Sony's Model Compression Toolkit (MCT)](https://github.com/SonySemiconductorSolutions/mct-model-optimization) is a powerful tool for optimizing deep learning models through quantization and pruning. It supports various quantization methods and provides advanced algorithms to reduce model size and computational complexity without significantly sacrificing accuracy. MCT is particularly useful for deploying models on resource-constrained devices, ensuring efficient inference and reduced latency.

### Supported Features of MCT

Sony's MCT offers a range of features designed to optimize neural network models:

1. **Graph Optimizations**: Transforms models into more efficient versions by folding layers like batch normalization into preceding layers.
2. **Quantization Parameter Search**: Minimizes quantization noise using metrics like Mean-Square-Error, No-Clipping, and Mean-Average-Error.
3. **Advanced Quantization Algorithms**:
    - **Shift Negative Correction**: Addresses performance issues from symmetric activation quantization.
    - **Outliers Filtering**: Uses z-score to detect and remove outliers.
    - **Clustering**: Utilizes non-uniform quantization grids for better distribution matching.
    - **Mixed-Precision Search**: Assigns different quantization bit-widths per layer based on sensitivity.
4. **Visualization**: Use TensorBoard to observe model performance insights, quantization phases, and bit-width configurations.

#### Quantization

MCT supports several quantization methods to reduce model size and improve inference speed:

1. **Post-Training Quantization (PTQ)**:
    - Available via Keras and PyTorch APIs.
    - Complexity: Low
    - Computational Cost: Low (CPU minutes)
2. **Gradient-based Post-Training Quantization (GPTQ)**:
    - Available via Keras and PyTorch APIs.
    - Complexity: Medium
    - Computational Cost: Moderate (2-3 GPU hours)
3. **Quantization-Aware Training (QAT)**:
    - Complexity: High
    - Computational Cost: High (12-36 GPU hours)

MCT also supports various quantization schemes for weights and activations:

1. Power-of-Two (hardware-friendly)
2. Symmetric
3. Uniform

#### Structured Pruning

MCT introduces structured, hardware-aware model pruning designed for specific hardware architectures. This technique leverages the target platform's Single Instruction, Multiple Data (SIMD) capabilities by pruning SIMD groups. This reduces model size and complexity while optimizing channel utilization, aligned with the SIMD architecture for targeted resource utilization of weights memory footprint. Available via Keras and PyTorch APIs.

### IMX500 Converter Tool (Compiler)

The IMX500 Converter Tool is integral to the IMX500 toolset, allowing the compilation of models for deployment on Sony's IMX500 sensor (for instance, Raspberry Pi AI Cameras). This tool facilitates the transition of Ultralytics YOLO11 models processed through Ultralytics software, ensuring they are compatible and perform efficiently on the specified hardware. The export procedure following model quantization involves the generation of binary files that encapsulate essential data and device-specific configurations, streamlining the deployment process on the Raspberry Pi AI Camera.

## Real-World Use Cases

Export to IMX500 format has wide applicability across industries. Here are some examples:

- **Edge AI and IoT**: Enable object detection on drones or security cameras, where real-time processing on low-power devices is essential.
- **Wearable Devices**: Deploy models optimized for small-scale AI processing on health-monitoring wearables.
- **Smart Cities**: Use IMX500-exported YOLO11 models for traffic monitoring and safety analysis with faster processing and minimal latency.
- **Retail Analytics**: Enhance in-store monitoring by deploying optimized models in point-of-sale systems or smart shelves.

## Conclusion

Exporting Ultralytics YOLO11 models to Sony's IMX500 format allows you to deploy your models for efficient inference on IMX500-based cameras. By leveraging advanced quantization techniques, you can reduce model size and improve inference speed without significantly compromising accuracy.

For more information and detailed guidelines, refer to Sony's [IMX500 website](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera).

## FAQ

### How do I export a YOLO11 model to IMX500 format for Raspberry Pi AI Camera?

To export a YOLO11 model to IMX500 format, use either the Python API or CLI command:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="imx")  # Exports with PTQ quantization by default
```

The export process will create a directory containing the necessary files for deployment, including `packerOut.zip`.

### What are the key benefits of using the IMX500 format for edge AI deployment?

The IMX500 format offers several important advantages for edge deployment:

- On-chip AI processing reduces latency and power consumption
- Outputs both image and metadata (inference result) instead of images only
- Enhanced privacy by processing data locally without cloud dependency
- Real-time processing capabilities ideal for time-sensitive applications
- Optimized quantization for efficient model deployment on resource-constrained devices

### What hardware and software prerequisites are needed for IMX500 deployment?

For deploying IMX500 models, you'll need:

Hardware:

- Raspberry Pi 5 or Raspberry Pi 4 Model B
- Raspberry Pi AI Camera with IMX500 sensor

Software:

- Raspberry Pi OS Bookworm
- IMX500 firmware and tools (`sudo apt install imx500-all`)

### What performance can I expect from YOLO11 models on the IMX500?

Based on Ultralytics benchmarks on Raspberry Pi AI Camera:

- YOLO11n achieves 62.50ms inference time per image
- mAP50-95 of 0.492 on COCO128 dataset
- Model size of only 3.2MB after quantization

This demonstrates that IMX500 format provides efficient real-time inference while maintaining good accuracy for edge AI applications.
