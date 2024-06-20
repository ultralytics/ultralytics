# YOLOv8 OpenVINO Inference C++

This example demonstrates how to perform inference using YOLOv8 in C++ with OpenVINO and OpenCV API.

## Features

- [x] Support for `ONNX` and `OpenVINO IR` model formats
- [x] Support for `FP32`, `FP16` and `INT8` precisions
- [x] Support load model with dynamic shape

## Dependencies

| Dependency | Version  |
| ---------- | -------- |
| OpenVINO   | >=2023.3 |
| OpenCV     | >=4.5.0  |
| C++        | >=14     |
| CMake      | >=3.12.0 |

## Build

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics/YOLOv8-OpenVINO-CPP-Inference

mkdir build
cd build
cmake ..
make

```

## Usage

```bash
./detect <model_path.{onnx, xml}> <image_path.jpg>
```

## Exporting YOLOv8

```commandline
yolo export model=yolov8s.pt imgsz=640 format=openvino
```
