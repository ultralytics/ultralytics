# YOLO11 Triton Inference Server C++ Client

[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-orange)](https://github.com/ultralytics/ultralytics)
[![Triton](https://img.shields.io/badge/NVIDIA-Triton-green)](https://github.com/triton-inference-server/server)

This example demonstrates how to perform object detection using Ultralytics YOLO11 models deployed on the NVIDIA Triton Inference Server. The implementation highlights efficient image preprocessing, FP16 (half-precision) data conversion, seamless communication with the Triton server via gRPC, and visualization of detection results with bounding boxes and confidence scores.

## ⚡ Features

- **High-Performance Inference**: Utilizes FP16 (half-precision) data format for optimized memory usage and accelerated inference.
- **Non-Maximum Suppression (NMS)**: Removes duplicate detections to ensure precise object detection results.
- **Seamless Triton Integration**: Communicates with the NVIDIA Triton Inference Server via gRPC for efficient and scalable model serving.
- **Detection Visualization**: Annotates images with bounding boxes, class labels, and confidence scores for intuitive result interpretation.

## 🛠️ Dependencies

Ensure you have the following dependencies installed before proceeding:

| Dependency              | Version | Description                                   |
| ----------------------- | ------- | --------------------------------------------- |
| Triton Inference Server | 22.06   | Running with a deployed FP16 YOLO11 model     |
| Triton Client libraries | 2.23    | Required for communication with Triton Server |
| C++ compiler            | C++ 17+ | For compiling the C++ client application      |
| OpenCV library          | 3.4.15  | For image processing and visualization        |
| CMake                   | 3.5+    | For building the project                      |

For more information on Triton, see the [NVIDIA Triton Inference Server documentation](https://github.com/triton-inference-server/server) and explore [model deployment options with Ultralytics](https://docs.ultralytics.com/guides/model-deployment-options/).

## 🏗️ Building the Project

1. **Install the Triton Client libraries:**

   ```bash
   wget https://github.com/triton-inference-server/server/releases/download/v2.23.0/v2.23.0_ubuntu2004.clients.tar.gz
   mkdir tritonclient
   tar -xvf v2.23.0_ubuntu2004.clients.tar.gz -C tritonclient
   rm -rf v2.23.0_ubuntu2004.clients.tar.gz
   ```

2. **Clone the Ultralytics repository:**

   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics/examples/YOLO11-Triton-CPP
   ```

3. **Configure and build the project using CMake:**

   ```bash
   mkdir build
   cd build
   cmake .. -DTRITON_CLIENT_DIR=/path/to/tritonclient
   make
   ```

For additional guidance on integrating Ultralytics YOLO models with various platforms, check out the [Ultralytics integrations documentation](https://docs.ultralytics.com/integrations/).

## 🚀 Usage

1. **Deploy your FP16 (half-precision) YOLO11 model on a Triton Inference Server.**  
   Learn more about deploying models with [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/).

2. **Run the YOLO11-Triton-CPP application:**

   ```bash
   ./YOLO11TritonCPP
   ```

By default, the application will:

- Connect to the Triton server at `localhost:8001`
- Use the model named `yolov11` with version `1`
- Process the image file `test.jpg`
- Save detection results to `output.jpg`

For more on object detection workflows, see [Ultralytics object detection tasks](https://docs.ultralytics.com/tasks/detect/).

## ⚙️ Configuration

You can modify the following parameters in [main.cpp](main.cpp):

```cpp
std::string triton_address = "localhost:8001";
std::string model_name = "yolov11";
std::string model_version = "1";
std::string image_path = "test.jpg";
std::string output_path = "output.jpg";
std::vector<std::string> object_class_list = {"class1", "class2"};
```

To learn more about configuring and customizing YOLO models, visit the [Ultralytics configuration guide](https://docs.ultralytics.com/usage/cfg/).

## 🌟 Contributors

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [main Ultralytics repository](https://github.com/ultralytics/ultralytics).

- Ahmet Selim Demirel
- Doğan Mehmet Başoğlu
- Enes Uzun
- Elif Cansu Ada
- Mevlüt Ardıç
- Serhat Karaca

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

---

For more resources, explore the [Ultralytics documentation](https://docs.ultralytics.com/), [Ultralytics blog](https://www.ultralytics.com/blog), and [Ultralytics HUB](https://docs.ultralytics.com/hub/).

**We encourage your contributions to make this project even better! 🚀**
