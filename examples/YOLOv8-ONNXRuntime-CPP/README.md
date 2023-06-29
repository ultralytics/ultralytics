# YOLOv8 OnnxRuntime C++

This example demonstrates how to perform inference using YOLOv8 in C++ with OnnxRuntime and OpenCV's API.

It is recommended to use VisualStudio to build the project.

## Exporting YOLOv8 Models

To export YOLOv8 models:

```python
from ultralytics import YOLO


def main():
    model = YOLO(R"E:\project\Project_Python\yolov8\runs\detect\train2\weights\best.pt")
    model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)


if __name__ == "__main__":
    main()
```

yolov8n.onnx:

This repository should work for YOLOv5 as well which need a permute operator to the output of yolov5 model, but they have not been tested.
