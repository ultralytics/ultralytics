# YOLOv8 LibTorch Inference C++

This example demonstrates how to perform inference using YOLOv8 models in C++ with LibTorch API.

## Dependencies

| Dependency   | Version  |
| ------------ | -------- |
| OpenCV       | >=4.0.0  |
| C++ Standard | >=17     |
| Cmake        | >=3.18   |
| Libtorch     | >=1.12.1 |

## Usage

```bash
git clone ultralytics
cd ultralytics
pip install .
cd examples/YOLOv8-LibTorch-CPP-Inference

mkdir build
cd build
cmake ..
make
./yolov8_libtorch_inference
```

## Exporting YOLOv8

To export YOLOv8 models:

```bash
yolo export model=yolov8s.pt imgsz=640 format=torchscript
```
