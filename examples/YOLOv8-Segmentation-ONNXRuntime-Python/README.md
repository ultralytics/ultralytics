# YOLOv8-Segmentation-ONNXRuntime-Python Demo

This project implements YOLOv8 segmentation using ONNX Runtime.

- No PyTorch imported! Just Numpy and OpenCV.
- Support both FP32 and FP16 ONNX model.

## Installation

```bash
pip install untralytics  # for exporting YOLOv8-seg ONNX model and using some basic functions
pip install onnxruntime-gpu
# pip install onnxruntime  # If has no NVIDIA GPU -> uncomment this line
pip install numpy
pip install opencv
```

## 1. Get ONNX model first

```bash
yolo export model=yolov8s-seg.pt imgsz=640 format=onnx opset=12 simplify=true

```

## 2. Run

```python
python main.py --model <MODEL_PATH> --source <IMAGE_PATH>
```

Then you will get the results like this:
![demo](https://github.com/jamjamjon/ultralytics/assets/51357717/44759936-43d7-430b-89ea-c3af770c21c0)
