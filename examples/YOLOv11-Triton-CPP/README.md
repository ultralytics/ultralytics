# YOLOv11 Triton Inference Server C++ Client

[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11-orange)](https://github.com/ultralytics/ultralytics)
[![Triton](https://img.shields.io/badge/NVIDIA-Triton-green)](https://github.com/triton-inference-server/server)

This example demonstrates how to perform object detection using Ultralytics YOLOv11 models deployed on the NVIDIA Triton Inference Server. The implementation showcases efficient image preprocessing, FP16 data conversion, communication with the Triton server via gRPC, and visualization of detection results with bounding boxes and confidence scores.

## Features

- **High-Performance Inference**: FP16 (half-precision) data format for optimized memory usage and faster inference.
- **Non-Maximum Suppression (NMS)**: Eliminates duplicate detections to ensure accurate object detection results.
- **Seamless Triton Integration**: Communicates with NVIDIA Triton Inference Server via gRPC for efficient model serving.
- **Detection Visualization**: Annotates images with bounding boxes, class labels, and confidence scores for easy interpretation.
## Dependencies

Ensure you have the following dependencies installed before proceeding:

| Dependency                     | Version       | Description
|--------------------------------|---------------|--------------------------------------------------|
| Triton Inference Server        | 22.06         | Running with a deployed FP16 YOLOv11 model  
| Triton Client libraries        | 2.23          | Required for communication with Triton Server 
| C++ compiler                   | C++ 17+       | For compiling the C++ client application 
| OpenCV library                 | 3.4.15        | For image processing and visualization 
| CMake                          | 3.5+          | For building the project   


## Building the Project


1. Ensure you have the Triton Client libraries installed
```bash
    wget https://github.com/triton-inference-server/server/releases/download/v2.23.0/v2.23.0_ubuntu2004.clients.tar.gz
    mkdir tritonclient
    tar -xvf v2.23.0_ubuntu2004.clients.tar.gz -C tritonclient
    rm -rf v2.23.0_ubuntu2004.clients.tar.gz
```

2. Clone the Repository:
```bash
    git clone https://github.com/ultralytics/ultralytics.git
    cd ultralytics/examples/YOLOV11-Triton-CPP
```
3. Configure and build the project using CMake:

```bash
    mkdir build
    cd build
    cmake .. -DTRITON_CLIENT_DIR=/path/to/tritonclient
    make
```

## Usage

1. Deploy your Fp16(half precision) YOLOv11 model on a Triton Inference Server
2. Run the YOLO-v11-Triton-CPP application:

    ```bash
    ./YOLOv11TritonCPP
    ```

By default, the application will:
- Connect to Triton server at `localhost:8001`
- Use the model named `yolov11` with version `1`
- Process the image file `test.jpg`
- Save detection results to `output.jpg`

## Configuration

You can modify the following parameters in [main.cpp](main.cpp):

```cpp
std::string triton_address= "localhost:8001"; 
std::string model_name= "yolov11"; 
std::string model_version= "1";
std::string image_path = "test.jpg";
std::string output_path = "output.jpg";
std::vector<std::string> object_class_list = {"class1", "class2"};
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the main Ultralytics repository.
