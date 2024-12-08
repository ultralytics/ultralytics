# YOLOv8 - Classification with ONNX Runtime

This project implements classification using YOLOv8 and ONNX Runtime.

## Installation

To run this project, you'll need to install the necessary dependencies. Follow the instructions below to set up your environment.

### Installing Required Dependencies

Install the required dependencies with this command:

```bash
pip install -r requirements.txt
```

### Installing `onnxruntime-gpu`

If you have an NVIDIA GPU and want to leverage GPU acceleration, you can install the onnxruntime-gpu package using the following command:

```bash
pip install onnxruntime-gpu
```

Note: Make sure you have the appropriate GPU drivers installed on your system.

### Installing `onnxruntime` (CPU version)

If you don't have an NVIDIA GPU or prefer to use the CPU version of onnxruntime, you can install the onnxruntime package using the following command:

```bash
pip install onnxruntime
```

### Usage

After installing the required packages, you can run YOLOv8 classification with the following command:

```bash
python yolov8_classifier.py --model yolov8n.onnx --img image.jpg
```

Replace yolov8n.onnx with the path to your YOLOv8 ONNX model file and image.jpg with the path to your input image.
