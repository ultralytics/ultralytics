# YOLOv8 OnnxRuntime C++

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white">

This example demonstrates how to perform inference using YOLOv8 in C++ with ONNX Runtime and OpenCV's API.

## Benefits ✨

- Friendly for deployment in the industrial sector.
- Faster than OpenCV's DNN inference on both CPU and GPU.
- Supports FP32 and FP16 CUDA acceleration.

## Note ☕

1. Benefit for Ultralytics' latest release, a `Transpose` op is added to the YOLOv8 model, while make v8 and v5 has the same output shape. Therefore, you can run inference with YOLOv5/v7/v8 via this project.

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

## Exporting YOLOv8 FP16 Models 📦

```python
import onnx
from onnxconverter_common import float16

model = onnx.load(R"YOUR_ONNX_PATH")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, R"YOUR_FP16_ONNX_PATH")
```

## Download COCO.yaml file 📂

In order to run example, you also need to download coco.yaml. You can download the file manually from [here](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml)

## Dependencies ⚙️

| Dependency                       | Version       |
| -------------------------------- | ------------- |
| Onnxruntime(linux,windows,macos) | >=1.14.1      |
| OpenCV                           | >=4.0.0       |
| C++ Standard                     | >=17          |
| Cmake                            | >=3.5         |
| Cuda (Optional)                  | >=11.4 \<12.0 |
| cuDNN (Cuda required)            | =8            |

Note: The dependency on C++17 is due to the usage of the C++17 filesystem feature.

Note (2): Due to ONNX Runtime, we need to use CUDA 11 and cuDNN 8. Keep in mind that this requirement might change in the future.

## Build 🛠️

1. Clone the repository to your local machine.

2. Navigate to the root directory of the repository.

3. Create a build directory and navigate to it:

   ```console
   mkdir build && cd build
   ```

4. Run CMake to generate the build files:

   ```console
   cmake ..
   ```

   **Notice**:

   If you encounter an error indicating that the `ONNXRUNTIME_ROOT` variable is not set correctly, you can resolve this by building the project using the appropriate command tailored to your system.

   ```console
   # compiled in a win32 system
   cmake -D WIN32=TRUE ..
   # compiled in a linux system
   cmake -D LINUX=TRUE ..
   # compiled in an apple system
   cmake -D APPLE=TRUE ..
   ```

5. Build the project:

   ```console
   make
   ```

6. The built executable should now be located in the `build` directory.

## Usage 🚀

```c++
//change your param as you like
//Pay attention to your device and the onnx model type(fp32 or fp16)
DL_INIT_PARAM params;
params.rectConfidenceThreshold = 0.1;
params.iouThreshold = 0.5;
params.modelPath = "yolov8n.onnx";
params.imgSize = { 640, 640 };
params.cudaEnable = true;
params.modelType = YOLO_DETECT_V8;
yoloDetector->CreateSession(params);
Detector(yoloDetector);
```
