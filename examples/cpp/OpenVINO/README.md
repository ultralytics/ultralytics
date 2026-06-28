# Ultralytics YOLO OpenVINO Inference in C++

<img alt="C++" src="https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus&logoColor=white"> <img alt="OpenVINO" src="https://img.shields.io/badge/OpenVINO-00C7FD.svg?logo=intel&logoColor=white"> <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?logo=opencv&logoColor=white">

A single C++ application that runs **every [Ultralytics YOLO](https://docs.ultralytics.com/) task and model generation** with the [Intel OpenVINO™ toolkit](https://docs.openvino.ai/) and [OpenCV](https://opencv.org/). Point it at an OpenVINO IR (`.xml`) or an `.onnx` file — the program reads the class names from the model and picks the right post-processing automatically.

## ✨ Features

- **All tasks:** [detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [pose](https://docs.ultralytics.com/tasks/pose), [OBB](https://docs.ultralytics.com/tasks/obb), [classify](https://docs.ultralytics.com/tasks/classify), and YOLO26 semantic segmentation.
- **All generations:** [YOLOv8](https://docs.ultralytics.com/models/yolov8), [YOLO11](https://docs.ultralytics.com/models/yolo11), and [YOLO26](https://docs.ultralytics.com/models/yolo26). Grid (YOLOv8/11) and end-to-end (YOLO26) outputs are detected automatically from the tensor shape.
- **Two formats:** OpenVINO IR (`.xml`/`.bin`) and [ONNX](https://onnx.ai/) — the OpenVINO runtime reads both.
- **Automatic task detection:** the IR carries no `task` field, so the task is inferred from the output shapes and class-label count. Class names come from the IR `rt_info` (`labels`); models without that metadata fall back to the 80 COCO names in [`../common`](../common).
- **Shared post-processing:** the parsing/NMS/mask/keypoint/semantic logic is the same `common/yolo_postprocess.hpp` used by the other examples.

## 📋 Dependencies

| Dependency                                | Version  |
| ----------------------------------------- | -------- |
| [OpenVINO](https://docs.openvino.ai/)     | >=2023.3 |
| [OpenCV](https://opencv.org/)             | >=4.5.0  |
| [C++](https://en.cppreference.com/w/)     | >=17     |
| [CMake](https://cmake.org/documentation/) | >=3.12.0 |

## 📦 Exporting a Model

```bash
yolo export model=yolo26n.pt imgsz=640 format=openvino # detect IR  (also -seg / -pose / -obb / -cls / -sem)
yolo export model=yolo26n.pt imgsz=640 format=onnx     # or ONNX, read directly by OpenVINO
```

[YOLOv8](https://docs.ultralytics.com/models/yolov8) and [YOLO11](https://docs.ultralytics.com/models/yolo11) grid models work too — the output layout is detected automatically.

See the [Export documentation](https://docs.ultralytics.com/modes/export). The OpenVINO export produces a `*_openvino_model/` directory containing the `.xml`, `.bin`, and metadata.

## 🛠️ Build

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics/examples/cpp/OpenVINO
mkdir build && cd build
cmake .. && cmake --build . --config Release
```

OpenVINO is found via `find_package(OpenVINO)` and the shared helpers in [`../common`](../common) are added to the include path automatically.

## 🚀 Usage

```bash
# Defaults: --model yolo26n.onnx --source bus.jpg --conf 0.25 --iou 0.45 --device AUTO --out result.jpg
./yolo_openvino --model yolo26n_openvino_model/yolo26n.xml --source bus.jpg
./yolo_openvino --model yolo26n-seg.onnx --source bus.jpg --out seg.jpg
./yolo_openvino --model yolo26n-pose.onnx --source bus.jpg --show
```

| Argument   | Default        | Description                                   |
| :--------- | :------------- | :-------------------------------------------- |
| `--model`  | `yolo26n.onnx` | OpenVINO IR (`.xml`) or ONNX (`.onnx`) model. |
| `--source` | `bus.jpg`      | Input image.                                  |
| `--conf`   | `0.25`         | Confidence threshold.                         |
| `--iou`    | `0.45`         | NMS IoU threshold (grid models only).         |
| `--device` | `AUTO`         | OpenVINO device: `AUTO`, `CPU`, or `GPU`.     |
| `--out`    | `result.jpg`   | Output image path.                            |
| `--show`   | _off_          | Also open a display window.                   |

The annotated result is always written to `--out` and the detections are printed to the console. The detected task is shown at startup, e.g. `Model: yolo26n.xml | task: detect | classes: 80`.

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
