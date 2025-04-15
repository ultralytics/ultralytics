# YOLOv8 MNN Inference in C++

Welcome to the Ultralytics YOLOv8 MNN Inference example in C++! This guide will help you get started with leveraging the powerful [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) models using the [Alibaba MNN](https://mnn-docs.readthedocs.io/en/latest/) inference engine in your C++ projects. Whether you're looking to enhance performance on CPU hardware or add flexibility to your applications, this example provides a solid foundation. Learn more about optimizing models and deployment strategies on the [Ultralytics blog](https://www.ultralytics.com/blog).

## 🌟 Features

- 🚀 **Model Format Support**: Native support for the MNN format.
- ⚡ **Precision Options**: Run models in **FP32**, **FP16** ([half-precision](https://www.ultralytics.com/glossary/half-precision)), and **INT8** ([model quantization](https://www.ultralytics.com/glossary/model-quantization)) precisions for optimized performance and reduced resource consumption.
- 🔄 **Dynamic Shape Loading**: Easily handle models with dynamic input shapes, a common requirement in many [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.
- 📦 **Flexible API Usage**: Choose between MNN's high-level [Express API](https://github.com/alibaba/MNN) for a user-friendly interface or the lower-level [Interpreter API](https://mnn-docs.readthedocs.io/en/latest/cpp/Interpreter.html) for fine-grained control.

## 📋 Dependencies

To ensure smooth execution, please make sure you have the following dependencies installed:

| Dependency                                        | Version  | Description                                                                      |
| :------------------------------------------------ | :------- | :------------------------------------------------------------------------------- |
| [MNN](https://mnn-docs.readthedocs.io/en/latest/) | >=2.0.0  | The core inference engine from Alibaba.                                          |
| [C++](https://en.cppreference.com/w/)             | >=14     | A modern C++ compiler supporting C++14 features.                                 |
| [CMake](https://cmake.org/documentation/)         | >=3.12.0 | Cross-platform build system generator required for building MNN and the example. |
| [OpenCV](https://opencv.org/)                     | Optional | Used for image loading and preprocessing within the example (built with MNN).    |

## ⚙️ Build Instructions

Follow these steps to build the project:

1.  Clone the Ultralytics repository:

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    cd ultralytics/examples/YOLOv8-MNN-CPP
    ```

2.  Clone the [Alibaba MNN repository](https://github.com/alibaba/MNN):

    ```bash
    git clone https://github.com/alibaba/MNN.git
    cd MNN
    ```

3.  Build the MNN library:

    ```bash
    # Create build directory
    mkdir build && cd build

    # Configure CMake (enable OpenCV integration, disable shared libs, enable image codecs)
    cmake -DMNN_BUILD_OPENCV=ON -DBUILD_SHARED_LIBS=OFF -DMNN_IMGCODECS=ON ..

    # Build the library (use -j flag for parallel compilation)
    make -j$(nproc) # Use nproc for Linux, sysctl -n hw.ncpu for macOS
    ```

    **Note:** If you encounter issues during the build process, consult the official [MNN documentation](https://mnn-docs.readthedocs.io/en/latest/) for detailed build instructions and troubleshooting tips.

4.  Copy the required MNN libraries and headers to the example project directory:

    ```bash
    # Navigate back to the example directory
    cd ../..

    # Create directories for libraries and headers if they don't exist
    mkdir -p libs include

    # Copy static libraries
    cp MNN/build/libMNN.a libs/                 # Main MNN library
    cp MNN/build/express/libMNN_Express.a libs/ # MNN Express API library
    cp MNN/build/tools/cv/libMNNOpenCV.a libs/  # MNN OpenCV wrapper library

    # Copy header files
    cp -r MNN/include .
    cp -r MNN/tools/cv/include . # MNN OpenCV wrapper headers
    ```

    **Note:**

    - The library file extensions (`.a` for static) and paths might vary based on your operating system (e.g., use `.lib` on Windows) and build configuration. Adjust the commands accordingly.
    - This example uses static linking (`.a` files). If you built shared libraries (`.so`, `.dylib`, `.dll`), ensure they are correctly placed or accessible in your system's library path.

5.  Create a build directory for the example project and compile using CMake:
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```

## 🔄 Exporting YOLOv8 Models

To use your Ultralytics YOLOv8 model with this C++ example, you first need to export it to the MNN format. This can be done easily using the `yolo export` command provided by the Ultralytics Python package.

Refer to the [Ultralytics Export documentation](https://docs.ultralytics.com/modes/export/) for detailed instructions and options.

```bash
# Export a YOLOv8n model to MNN format with input size 640x640
yolo export model=yolov8n.pt imgsz=640 format=mnn
```

Alternatively, you can use the `MNNConvert` tool provided by MNN:

```bash
# Assuming MNNConvert is built and in your PATH or MNN build directory
# Convert an ONNX model (first export YOLOv8 to ONNX)
yolo export model=yolov8n.pt format=onnx
./MNN/build/MNNConvert -f ONNX --modelFile yolov8n.onnx --MNNModel yolov8n.mnn --bizCode biz
```

For more details on model conversion using MNN tools, see the [MNN Convert documentation](https://mnn-docs.readthedocs.io/en/latest/tools/convert.html).

## 🛠️ Usage

### Ultralytics CLI in Python (for comparison)

You can verify the exported MNN model using the Ultralytics Python package for a quick check.

Download an example image:

```bash
wget https://ultralytics.com/images/bus.jpg
```

Run prediction using the MNN model:

```bash
yolo predict model=yolov8n.mnn source=bus.jpg
```

Expected Python Output:

```
ultralytics/examples/YOLOv8-MNN-CPP/assets/bus.jpg: 640x640 4 persons, 1 bus, 84.6ms
Speed: 9.7ms preprocess, 128.7ms inference, 12.4ms postprocess per image at shape (1, 3, 640, 640)
Results saved to runs/detect/predict
```

_(Note: Speed and specific detections might vary based on hardware and model version)_

### MNN Express API in C++

This example uses the higher-level Express API for simpler inference code.

```bash
./build/main yolov8n.mnn bus.jpg
```

Expected C++ Express API Output:

```
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
Detection: box = {48.63, 399.30, 243.65, 902.90}, class = person, score = 0.86
Detection: box = {22.14, 228.36, 796.07, 749.74}, class = bus, score = 0.86
Detection: box = {669.92, 375.82, 809.86, 874.41}, class = person, score = 0.86
Detection: box = {216.01, 405.24, 346.36, 858.19}, class = person, score = 0.82
Detection: box = {-0.11, 549.41, 62.05, 874.88}, class = person, score = 0.33
Result image write to `mnn_yolov8_cpp.jpg`.
Speed: 35.6ms preprocess, 386.0ms inference, 68.3ms postprocess
```

_(Note: Speed and specific detections might vary based on hardware and MNN configuration)_

### MNN Interpreter API in C++

This example uses the lower-level Interpreter API, offering more control over the inference process.

```bash
./build/main_interpreter yolov8n.mnn bus.jpg
```

Expected C++ Interpreter API Output:

```
The device supports: i8sdot:0, fp16:0, i8mm: 0, sve2: 0, sme2: 0
Detection: box = {48.63, 399.30, 243.65, 902.90}, class = person, score = 0.86
Detection: box = {22.14, 228.36, 796.07, 749.74}, class = bus, score = 0.86
Detection: box = {669.92, 375.82, 809.86, 874.41}, class = person, score = 0.86
Detection: box = {216.01, 405.24, 346.36, 858.19}, class = person, score = 0.82
Result image written to `mnn_yolov8_cpp.jpg`.
Speed: 26.0ms preprocess, 190.9ms inference, 58.9ms postprocess
```

_(Note: Speed and specific detections might vary based on hardware and MNN configuration)_

## ❤️ Contributions

We hope this example helps you integrate Ultralytics YOLOv8 with MNN into your C++ projects effortlessly! Contributions to improve this example or add new features are highly welcome. Please see the [Ultralytics contribution guidelines](https://docs.ultralytics.com/help/contributing/) for more information on how to get involved.

For further guides, tutorials, and documentation on Ultralytics YOLO models and tools, visit the main [Ultralytics documentation](https://docs.ultralytics.com/). Happy coding! 🚀
