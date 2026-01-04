# YOLOv8 OpenVINO Inference in C++

Welcome to the [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) OpenVINO Inference example in C++! This guide will help you get started with leveraging the powerful YOLOv8 models using the [Intel OpenVINO™ toolkit](https://docs.openvino.ai/) and [OpenCV API](https://docs.opencv.org/) in your C++ projects. Whether you're looking to enhance performance on Intel hardware or add flexibility to your applications, this example provides a solid foundation. Learn more about optimizing models on the [Ultralytics blog](https://www.ultralytics.com/blog).

## 🌟 Features

- 🚀 **Model Format Support**: Compatible with [ONNX](https://onnx.ai/) and [OpenVINO Intermediate Representation (IR)](https://docs.openvino.ai/2023.3/openvino_docs_MO_DG_IR_and_opsets.html) formats. Check the [Ultralytics ONNX integration](https://docs.ultralytics.com/integrations/onnx/) for more details.
- ⚡ **Precision Options**: Run models in **FP32**, **FP16** ([half-precision](https://www.ultralytics.com/glossary/half-precision)), and **INT8** ([quantization](https://www.ultralytics.com/glossary/model-quantization)) precisions for optimized performance.
- 🔄 **Dynamic Shape Loading**: Easily handle models with dynamic input shapes, common in many [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.

## 📋 Dependencies

To ensure smooth execution, please make sure you have the following dependencies installed:

| Dependency                                            | Version  |
| ----------------------------------------------------- | -------- |
| [OpenVINO](https://docs.openvino.ai/latest/home.html) | >=2023.3 |
| [OpenCV](https://opencv.org/)                         | >=4.5.0  |
| [C++](https://en.cppreference.com/w/)                 | >=14     |
| [CMake](https://cmake.org/documentation/)             | >=3.12.0 |

## ⚙️ Build Instructions

Follow these steps to build the project:

1.  Clone the Ultralytics repository:

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    cd ultralytics/examples/YOLOv8-OpenVINO-CPP-Inference
    ```

2.  Create a build directory and compile the project using CMake:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

## 🛠️ Usage

Once built, you can run [inference](https://www.ultralytics.com/glossary/real-time-inference) on an image using the compiled executable. Provide the path to your model file (either `.xml` for OpenVINO IR or `.onnx`) and the path to your image:

```bash
# Example using an OpenVINO IR model
./detect path/to/your/model.xml path/to/your/image.jpg

# Example using an ONNX model
./detect path/to/your/model.onnx path/to/your/image.jpg
```

This command will process the image using the specified YOLOv8 model and display the [object detection](https://www.ultralytics.com/glossary/object-detection) results. Explore various [Ultralytics Solutions](https://docs.ultralytics.com/solutions/) for real-world applications.

## 🔄 Exporting YOLOv8 Models

To use your Ultralytics YOLOv8 model with this C++ example, you first need to export it to the OpenVINO IR or ONNX format. Use the `yolo export` command available in the Ultralytics Python package. Find detailed instructions in the [Export mode documentation](https://docs.ultralytics.com/modes/export/).

```bash
# Export to OpenVINO format (generates .xml and .bin files)
yolo export model=yolov8s.pt imgsz=640 format=openvino

# Export to ONNX format
yolo export model=yolov8s.pt imgsz=640 format=onnx
```

For more details on exporting and optimizing models for OpenVINO, refer to the [Ultralytics OpenVINO integration guide](https://docs.ultralytics.com/integrations/openvino/).

## 📸 Screenshots

### Running Using OpenVINO Model

![Running OpenVINO Model](https://github.com/ultralytics/ultralytics/assets/76827698/2d7cf201-3def-4357-824c-12446ccf85a9)

### Running Using ONNX Model

![Running ONNX Model](https://github.com/ultralytics/ultralytics/assets/76827698/9b90031c-cc81-4cfb-8b34-c619e09035a7)

## ❤️ Contributions

We hope this example helps you integrate YOLOv8 with OpenVINO and OpenCV into your C++ projects effortlessly. Contributions to improve this example or add new features are welcome! Please see the [Ultralytics contribution guidelines](https://docs.ultralytics.com/help/contributing/) for more information. Visit the main [Ultralytics documentation](https://docs.ultralytics.com/) for further guides and resources. Happy coding! 🚀
