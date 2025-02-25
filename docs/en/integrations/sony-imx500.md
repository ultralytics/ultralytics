---
comments: true
description: Learn to export Ultralytics YOLOv8 models to Sony's IMX500 format to optimize your models for efficient deployment.
keywords: Sony, IMX500, IMX 500, Atrios, MCT, model export, quantization, pruning, deep learning optimization, Raspberry Pi AI Camera, edge AI, PyTorch, IMX
---

# Sony IMX500 Export for Ultralytics YOLOv8

This guide covers exporting and deploying Ultralytics YOLOv8 models to Raspberry Pi AI Cameras that feature the Sony IMX500 sensor.

Deploying computer vision models on devices with limited computational power, such as [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/), can be tricky. Using a model format optimized for faster performance makes a huge difference.

The IMX500 model format is designed to use minimal power while delivering fast performance for neural networks. It allows you to optimize your [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) models for high-speed and low-power inferencing. In this guide, we'll walk you through exporting and deploying your models to the IMX500 format while making it easier for your models to perform well on the [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/).

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/assets/releases/download/v8.3.0/ai-camera.avif" alt="Raspberry Pi AI Camera">
</p>

## Why Should You Export to IMX500

Sony's [IMX500 Intelligent Vision Sensor](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera) is a game-changing piece of hardware in edge AI processing. It's the world's first intelligent vision sensor with on-chip AI capabilities. This sensor helps overcome many challenges in edge AI, including data processing bottlenecks, privacy concerns, and performance limitations.  
While other sensors merely pass along images and frames, the IMX500 tells a whole story. It processes data directly on the sensor, allowing devices to generate insights in real-time.

## Sony's IMX500 Export for YOLOv8 Models

The IMX500 is designed to transform how devices handle data directly on the sensor, without needing to send it off to the cloud for processing.

The IMX500 works with quantized models. Quantization makes models smaller and faster without losing much [accuracy](https://www.ultralytics.com/glossary/accuracy). It is ideal for the limited resources of edge computing, allowing applications to respond quickly by reducing latency and allowing for quick data processing locally, without cloud dependency. Local processing also keeps user data private and secure since it's not sent to a remote server.

**IMX500 Key Features:**

- **Metadata Output:** Instead of transmitting images only, the IMX500 can output both image and metadata (inference result), and can output metadata only for minimizing data size, reducing bandwidth, and lowering costs.
- **Addresses Privacy Concerns:** By processing data on the device, the IMX500 addresses privacy concerns, ideal for human-centric applications like person counting and occupancy tracking.
- **Real-time Processing:** Fast, on-sensor processing supports real-time decisions, perfect for edge AI applications such as autonomous systems.

**Before You Begin:** For best results, ensure your YOLOv8 model is well-prepared for export by following our [Model Training Guide](https://docs.ultralytics.com/modes/train/), [Data Preparation Guide](https://docs.ultralytics.com/datasets/), and [Hyperparameter Tuning Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/).

## Usage Examples

Export an Ultralytics YOLOv8 model to IMX500 format and run inference with the exported model.

!!! note

    IMX export is currently only supported for the YOLOv8n model. Here we perform inference just to make sure the model works as expected. However, for deployment and inference on the Raspberry Pi AI Camera, please jump to [Using IMX500 Export in Deployment](#using-imx500-export-in-deployment) section.

!!! example

    === "Python"

         ```python
         from ultralytics import YOLO

         # Load a YOLOv8n PyTorch model
         model = YOLO("yolov8n.pt")

         # Export the model
         model.export(format="imx")  # exports with PTQ quantization by default

         # Load the exported model
         imx_model = YOLO("yolov8n_imx_model")

         # Run inference
         results = imx_model("https://ultralytics.com/images/bus.jpg")
         ```

    === "CLI"

         ```bash
         # Export a YOLOv8n PyTorch model to imx format with Post-Training Quantization (PTQ)
         yolo export model=yolov8n.pt format=imx

         # Run inference with the exported model
         yolo predict model=yolov8n_imx_model source='https://ultralytics.com/images/bus.jpg'
         ```

The export process will create an ONNX model for quantization validation, along with a directory named `<model-name>_imx_model`. This directory will include the `packerOut.zip` file, which is essential for packaging the model for deployment on the IMX500 hardware. Additionally, the `<model-name>_imx_model` folder will contain a text file (`labels.txt`) listing all the labels associated with the model.

```bash
yolov8n_imx_model
├── dnnParams.xml
├── labels.txt
├── packerOut.zip
├── yolov8n_imx.onnx
├── yolov8n_imx500_model_MemoryReport.json
└── yolov8n_imx500_model.pbtxt
```

## Export Arguments

| Argument | Type             | Default        | Description                                                                                                                                                                                   |
| -------- | ---------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` | `str`            | `'imx'`        | Target format for the exported model, defining compatibility with various deployment environments.                                                                                            |
| `imgsz`  | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.                                                             |
| `int8`   | `bool`           | `True`         | Activates INT8 quantization, further compressing the model and speeding up inference with minimal [accuracy](https://www.ultralytics.com/glossary/accuracy) loss, primarily for edge devices. |
| `data`   | `str`            | `'coco8.yaml'` | Path to the [dataset](https://docs.ultralytics.com/datasets) configuration file (default: `coco8.yaml`), essential for quantization.                                                          |

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).

## Using IMX500 Export in Deployment

After exporting Ultralytics YOLOv8n model to IMX500 format, it can be deployed to Raspberry Pi AI Camera for inference.

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

Step 2: Install IMX500 firmware which is required to operate the IMX500 sensor along with a packager tool.

```bash
sudo apt install imx500-all imx500-tools
```

Step 3: Install prerequisites to run `picamera2` application. We will use this application later for the deployment process.

```bash
sudo apt install python3-opencv python3-munkres
```

Step 4: Reboot Raspberry Pi for the changes to take into effect

```bash
sudo reboot
```

### Package Model and Deploy to AI Camera

After obtaining `packerOut.zip` from the IMX500 conversion process, you can pass this file into the packager tool to obtain an RPK file. This file can then be deployed directly to the AI Camera using `picamera2`.

Step 1: Package the model into RPK file

```bash
imx500-package -i <path to packerOut.zip> -o <output folder>
```

The above will generate a `network.rpk` file inside the specified output folder.

Step 2: Clone `picamera2` repository, install it and navigate to the imx500 examples

```bash
git clone https://github.com/raspberrypi/picamera2
cd picamera2
pip install -e .  --break-system-packages
cd examples/imx500
```

Step 3: Run YOLOv8 object detection, using the labels.txt file that has been generated during the IMX500 export.

```bash
python imx500_object_detection_demo.py --model <path to network.rpk> --fps 25 --bbox-normalization --ignore-dash-labels --bbox-order xy --labels <path to labels.txt>
```

Then you will be able to see live inference output as follows

<p align="center">
  <img width="100%" src="https://github.com/ultralytics/assets/releases/download/v8.3.0/imx500-inference-rpi.avif" alt="Inference on Raspberry Pi AI Camera">
</p>

## Benchmarks

YOLOv8 benchmarks below were run by the Ultralytics team on Raspberry Pi AI Camera with `imx` model format measuring speed and accuracy.

| Model   | Format | Status | Size (MB) | mAP50-95(B) | Inference time (ms/im) |
| ------- | ------ | ------ | --------- | ----------- | ---------------------- |
| YOLOv8n | imx    | ✅     | 2.9       | 0.522       | 66.66                  |

!!! note

    Validation for the above benchmark was done using coco8 dataset

## What's Under the Hood?

<p align="center">
  <img width="640" src="https://github.com/ultralytics/assets/releases/download/v8.3.0/imx500-deploy.avif" alt="IMX500 deployment">
</p>

### Sony Model Compression Toolkit (MCT)

[Sony's Model Compression Toolkit (MCT)](https://github.com/sony/model_optimization) is a powerful tool for optimizing deep learning models through quantization and pruning. It supports various quantization methods and provides advanced algorithms to reduce model size and computational complexity without significantly sacrificing accuracy. MCT is particularly useful for deploying models on resource-constrained devices, ensuring efficient inference and reduced latency.

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

The IMX500 Converter Tool is integral to the IMX500 toolset, allowing the compilation of models for deployment on Sony's IMX500 sensor (for instance, Raspberry Pi AI Cameras). This tool facilitates the transition of Ultralytics YOLOv8 models processed through Ultralytics software, ensuring they are compatible and perform efficiently on the specified hardware. The export procedure following model quantization involves the generation of binary files that encapsulate essential data and device-specific configurations, streamlining the deployment process on the Raspberry Pi AI Camera.

## Real-World Use Cases

Export to IMX500 format has wide applicability across industries. Here are some examples:

- **Edge AI and IoT**: Enable object detection on drones or security cameras, where real-time processing on low-power devices is essential.
- **Wearable Devices**: Deploy models optimized for small-scale AI processing on health-monitoring wearables.
- **Smart Cities**: Use IMX500-exported YOLOv8 models for traffic monitoring and safety analysis with faster processing and minimal latency.
- **Retail Analytics**: Enhance in-store monitoring by deploying optimized models in point-of-sale systems or smart shelves.

## Conclusion

Exporting Ultralytics YOLOv8 models to Sony's IMX500 format allows you to deploy your models for efficient inference on IMX500-based cameras. By leveraging advanced quantization techniques, you can reduce model size and improve inference speed without significantly compromising accuracy.

For more information and detailed guidelines, refer to Sony's [IMX500 website](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera).

## FAQ

### How do I export a YOLOv8 model to IMX500 format for Raspberry Pi AI Camera?

To export a YOLOv8 model to IMX500 format, use either the Python API or CLI command:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="imx")  # Exports with PTQ quantization by default
```

The export process will create a directory containing the necessary files for deployment, including `packerOut.zip` which can be used with the IMX500 packager tool on Raspberry Pi.

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
- IMX500 firmware and tools (`sudo apt install imx500-all imx500-tools`)
- Python packages for `picamera2` (`sudo apt install python3-opencv python3-munkres`)

### What performance can I expect from YOLOv8 models on the IMX500?

Based on Ultralytics benchmarks on Raspberry Pi AI Camera:

- YOLOv8n achieves 66.66ms inference time per image
- mAP50-95 of 0.522 on COCO8 dataset
- Model size of only 2.9MB after quantization

This demonstrates that IMX500 format provides efficient real-time inference while maintaining good accuracy for edge AI applications.

### How do I package and deploy my exported model to the Raspberry Pi AI Camera?

After exporting to IMX500 format:

1. Use the packager tool to create an RPK file:

    ```bash
    imx500-package -i <path to packerOut.zip> -o <output folder>
    ```

2. Clone and install picamera2:

    ```bash
    git clone https://github.com/raspberrypi/picamera2
    cd picamera2 && pip install -e . --break-system-packages
    ```

3. Run inference using the generated RPK file:

    ```bash
    python imx500_object_detection_demo.py --model <path to network.rpk> --fps 25 --bbox-normalization --labels <path to labels.txt>
    ```
