# YOLOv8 OnnxRuntime C++

This example demonstrates how to perform inference using YOLOv8 in C++ with OnnxRuntime and OpenCV's API.

It is recommended to use VisualStudio to build the project.

## Specific

1. Frendly for deployment in the industrial sector
1. More faster than OpenCV's DNN inference both cpu and gpu
1. Support CUDA accelerate
1. Frendly for adding FP16 inference(using template function)

## Exporting YOLOv8 Models

To export YOLOv8 models:

```python
from ultralytics import YOLO


def main():
    model = YOLO(R"your_model.pt")
    model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)


if __name__ == "__main__":
    main()
```

## Dependency

| Dependency              | version  |
| ----------------------- | -------- |
| Onnxruntime-win-x64-gpu | >=1.14.1 |
| Opencv                  | >=4.0.0  |
| C++                     | >=17     |

Version dependency of c++ is not necessary,the version requirement here is due to the use of c++17's advanced feature filesystem.

## Usage

```c++
//cpu inference
DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {imgsz_w, imgsz_h}, class_num, 0.1, 0.5, false};
//gpu inference
DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {imgsz_w, imgsz_h} , class_num, 0.1, 0.5, true};

//load your image
cv::Mat img = cv::imread(img_path);

char* ret = p1->CreateSession(params);

ret = p->RunSession(img, res);
```

This repository should work for YOLOv5 as well which need a permute operator to the output of yolov5 model, but they have not been realized.
