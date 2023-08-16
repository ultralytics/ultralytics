<h1 align="center">YOLOv8 OnnxRuntime C++</h1>

<p align="center">
  <img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B">
  <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white"></img>
</p>

This example demonstrates how to perform inference using YOLOv8 in C++ with ONNX Runtime and OpenCV's API.

## Benefits ✨

- Friendly for deployment in the industrial sector.
- Faster than OpenCV's DNN inference on both CPU and GPU.
- Supports FP32 and FP16 CUDA acceleration.

## Exporting YOLOv8 Models 📦

To export YOLOv8 models, use the following Python script:

```python
from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
```

Alternatively, you can use the following command for exporting the model in the terminal

```bash
yolo export model=yolov8n.pt opset=12 simplify=True dynamic=False format=onnx imgsz=640,640
```

## Download COCO.yaml file 📂

In order to run example, you also need to download coco.yaml. You can download the file manually from [here](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml)

## Dependencies ⚙️

| Dependency                       | Version        |
| -------------------------------- | -------------- |
| Onnxruntime(linux,windows,macos) | >=1.14.1       |
| OpenCV                           | >=4.0.0        |
| C++ Standard                     | >=17           |
| Cmake                            | >=3.5          |
| Cuda (Optional)                  | >=11.4  \<12.0 |
| cuDNN (Cuda required)            | =8             |

Note: The dependency on C++17 is due to the usage of the C++17 filesystem feature.

Note (2): Due to ONNX Runtime, we need to use CUDA 11 and cuDNN 8. Keep in mind that this requirement might change in the future.

## Build 🛠️

1. Clone the repository to your local machine.
1. Navigate to the root directory of the repository.
1. Create a build directory and navigate to it:

```console
mkdir build && cd build
```

4. Run CMake to generate the build files:

```console
cmake ..
```

5. Build the project:

```console
make
```

6. The built executable should now be located in the `build` directory.

## Usage 🚀

```c++
// CPU inference
DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {imgsz_w, imgsz_h}, 0.1, 0.5, false};
// GPU inference
DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {imgsz_w, imgsz_h}, 0.1, 0.5, true};
// Load your image
cv::Mat img = cv::imread(img_path);
// Init Inference Session
char* ret = yoloDetector->CreateSession(params);

ret = yoloDetector->RunSession(img, res);
```

This repository should also work for YOLOv5, which needs a permute operator for the output of the YOLOv5 model, but this has not been implemented yet.
