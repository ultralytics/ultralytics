# YOLOv8 OnnxRuntime C++

This example demonstrates how to perform inference using YOLOv8 in C++ with OnnxRuntime and OpenCV's API.


It is recommended to use VisualStudio to build the project.

## Specific
1. Frendly for deployment in the industrial sector
2. More fatser than OpenCV's DNN inference
3. Support CUDA accelerate
4. Frendly for adding FP16 inference(using template function)



## Exporting YOLOv8 Models

To export YOLOv8 models:

```python
from ultralytics import YOLO



def main():
    model = YOLO(R'your_model.pt')
    model.export(format='onnx', opset=12, simplify=True, dynamic=False, imgsz=640)


if __name__ == '__main__':
    main()
```

## Usage
```c++
//cpu inference
DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8 , 80, {imgsz_w, imgsz_h}, 0.1, 0.5, false};
//gpu inference
DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8 , 80, {imgsz_w, imgsz_h}, 0.1, 0.5, true};

```


This repository should work for YOLOv5 as well which need a permute operator to the output of yolov5 model, but they have not been tested.
