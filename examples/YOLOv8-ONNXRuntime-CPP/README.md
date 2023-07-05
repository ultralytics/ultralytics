# YOLOv8 OnnxRuntime C++

This example demonstrates how to perform inference using YOLOv8 in C++ with ONNX Runtime and OpenCV's API.

We recommend using Visual Studio to build the project.

## Benefits

- Friendly for deployment in the industrial sector.
- Faster than OpenCV's DNN inference on both CPU and GPU.
- Supports CUDA acceleration.
- Easy to add FP16 inference (using template functions).

## Exporting YOLOv8 Models

To export YOLOv8 models, use the following Python script:

```python
from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
```

## Dependencies

| Dependency              | Version  |
| ----------------------- | -------- |
| Onnxruntime-win-x64-gpu | >=1.14.1 |
| OpenCV                  | >=4.0.0  |
| C++                     | >=17     |

Note: The dependency on C++17 is due to the usage of the C++17 filesystem feature.

## Usage

```c++
// CPU inference
DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {imgsz_w, imgsz_h}, class_num, 0.1, 0.5, false};
// GPU inference
DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {imgsz_w, imgsz_h}, class_num, 0.1, 0.5, true};

// Load your image
cv::Mat img = cv::imread(img_path);

char* ret = p1->CreateSession(params);

ret = p->RunSession(img, res);
```

This repository should also work for YOLOv5, which needs a permute operator for the output of the YOLOv5 model, but this has not been implemented yet.
