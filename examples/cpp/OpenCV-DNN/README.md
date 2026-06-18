# Ultralytics YOLO OpenCV DNN Inference in C++

<img alt="C++" src="https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus&logoColor=white"> <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?logo=opencv&logoColor=white"> <img alt="ONNX" src="https://img.shields.io/badge/ONNX-005CED.svg?logo=onnx&logoColor=white">

A C++ application that runs Ultralytics YOLO ONNX models with the [OpenCV DNN module](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html). It supports [detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [pose](https://docs.ultralytics.com/tasks/pose), [OBB](https://docs.ultralytics.com/tasks/obb), [classify](https://docs.ultralytics.com/tasks/classify), and semantic segmentation, sharing its post-processing with the other examples in [`../common`](../common).

> [!IMPORTANT]
> The OpenCV DNN module cannot run the NMS-in-graph operators used by **YOLO26 end-to-end** exports, and it cannot read class names or the task from the ONNX metadata. So this example targets **grid models** ([YOLOv8](https://docs.ultralytics.com/models/yolov8) / [YOLO11](https://docs.ultralytics.com/models/yolo11), or [YOLO26](https://docs.ultralytics.com/models/yolo26) with its end-to-end head disabled), class names fall back to the 80 COCO names, and the task is inferred from the output shapes (use `--task` for grid pose/obb).

## ✨ Features

- **All tasks:** [detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [pose](https://docs.ultralytics.com/tasks/pose), [OBB](https://docs.ultralytics.com/tasks/obb), [classify](https://docs.ultralytics.com/tasks/classify), and semantic segmentation — on grid models.
- **All generations (grid):** [YOLOv8](https://docs.ultralytics.com/models/yolov8), [YOLO11](https://docs.ultralytics.com/models/yolo11), and [YOLO26](https://docs.ultralytics.com/models/yolo26) with its end-to-end head disabled. The OpenCV DNN module cannot run the YOLO26 end-to-end (NMS-in-graph) operators.
- **Zero configuration:** OpenCV exposes no model metadata, so the task is inferred from the output shapes and class names fall back to COCO (pass `--task` for grid pose/obb).
- **CPU or CUDA:** runs on the OpenCV DNN CPU backend, or the CUDA backend with `--cuda` (requires a CUDA-enabled OpenCV).

## 📋 Dependencies

| Dependency                                        | Version  | Description                                                 |
| :------------------------------------------------ | :------- | :---------------------------------------------------------- |
| [OpenCV](https://opencv.org/)                     | >=4.7.0  | DNN module for inference, plus image I/O, drawing, and NMS. |
| [C++](https://en.cppreference.com/w/)             | >=17     | Modern C++ compiler.                                        |
| [CMake](https://cmake.org/documentation/)         | >=3.5    | Build system.                                               |
| [CUDA](https://developer.nvidia.com/cuda/toolkit) | optional | Only for the OpenCV CUDA DNN backend (`--cuda`).            |

## 📦 Exporting a Model

The OpenCV DNN module runs grid models only. [YOLOv8](https://docs.ultralytics.com/models/yolov8) and [YOLO11](https://docs.ultralytics.com/models/yolo11) export to a grid by default:

```bash
yolo export model=yolo11n.pt format=onnx opset=12 imgsz=640 # detect (also -seg / -pose / -obb / -cls / -sem)
```

[YOLO26](https://docs.ultralytics.com/models/yolo26) is end-to-end (NMS-free) by architecture, so a normal export emits a `[1, 300, 6]` tensor that the OpenCV DNN module cannot run. Disable the end-to-end head first to get a grid `[1, 84, 8400]` output:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.model.model[-1].end2end = False  # grid output the OpenCV DNN module can run
model.export(format="onnx", opset=12, imgsz=640)
```

## 🛠️ Build

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics/examples/cpp/OpenCV-DNN
mkdir build && cd build
cmake .. && cmake --build . --config Release
```

OpenCV is found with `find_package(OpenCV)` and the shared helpers in [`../common`](../common) are added automatically. For GPU inference build OpenCV with the CUDA DNN backend and pass `--cuda`.

## 🚀 Usage

```bash
# Defaults: --model yolo26n.onnx --source bus.jpg --conf 0.25 --iou 0.45 --imgsz 640 --out result.jpg
./yolo_opencv_dnn --model yolo26n.onnx --source bus.jpg
./yolo_opencv_dnn --model yolo26n-seg.onnx --source bus.jpg --out seg.jpg
./yolo_opencv_dnn --model yolo26n-pose.onnx --source bus.jpg --task pose --show
```

| Argument   | Default        | Description                                                                                          |
| :--------- | :------------- | :--------------------------------------------------------------------------------------------------- |
| `--model`  | `yolo26n.onnx` | Path to the exported ONNX model (grid output).                                                       |
| `--source` | `bus.jpg`      | Input image.                                                                                         |
| `--conf`   | `0.25`         | Confidence threshold.                                                                                |
| `--iou`    | `0.45`         | NMS IoU threshold.                                                                                   |
| `--imgsz`  | `640`          | Square input size of the exported model.                                                             |
| `--task`   | _auto_         | Override the task (`detect`/`segment`/`pose`/`obb`/`classify`/`semantic`); needed for grid pose/obb. |
| `--cuda`   | _off_          | Use the OpenCV CUDA DNN backend (requires a CUDA-enabled OpenCV).                                    |
| `--out`    | `result.jpg`   | Output image path.                                                                                   |
| `--show`   | _off_          | Also open a display window.                                                                          |

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
