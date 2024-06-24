# YOLOv8 OpenVINO Inference in C++ 🦾

Welcome to the YOLOv8 OpenVINO Inference example in C++! This guide will help you get started with leveraging the powerful YOLOv8 models using OpenVINO and OpenCV API in your C++ projects. Whether you're looking to enhance performance or add flexibility to your applications, this example has got you covered.

## 🌟 Features

- 🚀 **Model Format Support**: Compatible with `ONNX` and `OpenVINO IR` formats.
- ⚡ **Precision Options**: Run models in `FP32`, `FP16`, and `INT8` precisions.
- 🔄 **Dynamic Shape Loading**: Easily handle models with dynamic input shapes.

## 📋 Dependencies

To ensure smooth execution, please make sure you have the following dependencies installed:

| Dependency | Version  |
| ---------- | -------- |
| OpenVINO   | >=2023.3 |
| OpenCV     | >=4.5.0  |
| C++        | >=14     |
| CMake      | >=3.12.0 |

## ⚙️ Build Instructions

Follow these steps to build the project:

1. Clone the repository:

   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics/YOLOv8-OpenVINO-CPP-Inference
   ```

2. Create a build directory and compile the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## 🛠️ Usage

Once built, you can run inference on an image using the following command:

```bash
./detect <model_path.{onnx, xml}> <image_path.jpg>
```

## 🔄 Exporting YOLOv8 Models

To use your YOLOv8 model with OpenVINO, you need to export it first. Use the command below to export the model:

```commandline
yolo export model=yolov8s.pt imgsz=640 format=openvino
```

## 📸 Screenshots

### Running Using OpenVINO Model

![Running OpenVINO Model](https://github.com/ultralytics/ultralytics/assets/76827698/2d7cf201-3def-4357-824c-12446ccf85a9)

### Running Using ONNX Model

![Running ONNX Model](https://github.com/ultralytics/ultralytics/assets/76827698/9b90031c-cc81-4cfb-8b34-c619e09035a7)

## ❤️ Contributions

We hope this example helps you integrate YOLOv8 with OpenVINO and OpenCV into your C++ projects effortlessly. Happy coding! 🚀
