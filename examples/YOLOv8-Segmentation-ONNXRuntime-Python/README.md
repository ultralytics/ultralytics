# YOLOv8-ONNXRuntime-Segmentation-Python Demo

This project implements YOLOv8 segmentation using ONNX Runtime. **No PyTorch code, just Numpy and OpenCV.**

## Installing Required Dependencies

```bash
pip install untralytics
pip install onnxruntime-gpu
# pip install onnxruntime  # No NVIDIA GPU
pip install numpy
pip install opencv
```

## Usage

```bash
python main.py --model yolov8s-seg.onnx --img IMAGE_PATH
```
