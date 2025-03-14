# RTDETR - ONNX Runtime

This project implements RTDETR using ONNX Runtime.

## Installation

To run this project, you need to install the required dependencies. The following instructions will guide you through the installation process.

### Installing Required Dependencies

You can install the required dependencies by running the following command:

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

After successfully installing the required packages, you can run the RTDETR implementation using the following command:

```bash
python main.py --model rtdetr-l.onnx --img image.jpg --conf-thres 0.5 --iou-thres 0.5
```

Make sure to replace rtdetr-l.onnx with the path to your RTDETR ONNX model file, image.jpg with the path to your input image, and adjust the confidence threshold (conf-thres) and IoU threshold (iou-thres) values as needed.
