# Ultralytics YOLO ONNX Runtime C++ Example

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white">

A single C++ application that runs **every [Ultralytics YOLO](https://docs.ultralytics.com/) task and model generation** with [ONNX Runtime](https://onnxruntime.ai/) and [OpenCV](https://opencv.org/). Point it at any exported `.onnx` model — the program reads the task, class names, and input size from the model metadata and picks the right post-processing automatically.

## ✨ Features

- **All tasks:** [detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [pose](https://docs.ultralytics.com/tasks/pose), [OBB](https://docs.ultralytics.com/tasks/obb), [classify](https://docs.ultralytics.com/tasks/classify), and YOLO26 semantic segmentation.
- **All generations:** [YOLOv8](https://docs.ultralytics.com/models/yolov8), [YOLO11](https://docs.ultralytics.com/models/yolo11), and [YOLO26](https://docs.ultralytics.com/models/yolo26). The grid output of YOLOv8/11 and the end-to-end (NMS-free) output of YOLO26 are detected automatically from the tensor shape — no flags needed.
- **Zero configuration:** task, class names, and `imgsz` come from the model metadata that Ultralytics bakes into every export. No `coco.yaml` or hard-coded class lists.
- **Simple CLI:** choose the model, source image, and thresholds at runtime — no recompiling.

## 📦 Exporting a Model

Export any model and task to ONNX with the Ultralytics `export` mode. `opset=12` is recommended for broad compatibility.

```bash
yolo export model=yolo26n.pt       format=onnx opset=12   # detect   (end2end)
yolo export model=yolo26n-seg.pt   format=onnx opset=12   # segment
yolo export model=yolo26n-pose.pt  format=onnx opset=12   # pose
yolo export model=yolo26n-obb.pt   format=onnx opset=12   # obb
yolo export model=yolo26n-cls.pt   format=onnx opset=12   # classify
yolo export model=yolo26n-sem.pt   format=onnx opset=12   # semantic (YOLO26)
yolo export model=yolo11n.pt       format=onnx opset=12   # YOLOv8/YOLO11 also work
```

See the [Export documentation](https://docs.ultralytics.com/modes/export) for more options.

## ⚙️ Dependencies

| Dependency                                           | Version       | Notes                                                          |
| :--------------------------------------------------- | :------------ | :------------------------------------------------------------- |
| [ONNX Runtime](https://onnxruntime.ai/docs/install/) | >=1.14        | Download the pre-built binaries (CPU or GPU).                  |
| [OpenCV](https://opencv.org/releases/)               | >=4.0         | Image I/O, drawing, and NMS.                                   |
| C++ Compiler                                         | C++17         | For `<filesystem>`.                                            |
| [CMake](https://cmake.org/download/)                 | >=3.5         | Build system.                                                  |
| [CUDA](https://developer.nvidia.com/cuda/toolkit)    | optional      | Only for the ONNX Runtime CUDA execution provider (`--cuda`).  |

## 🛠️ Build

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics/examples/cpp/ONNXRuntime
mkdir build && cd build

# Point ONNXRUNTIME_ROOT at the extracted ONNX Runtime. Use -DUSE_CUDA=OFF for a CPU-only build.
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime -DUSE_CUDA=OFF
cmake --build . --config Release
```

The shared helpers in [`../common`](../common) (color palette, COCO names, annotator) are header-only and added to the include path automatically.

## 🚀 Usage

```bash
# If the ONNX Runtime libraries are not installed system-wide, add them to the loader path:
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH

# Defaults: --model yolo26n.onnx --source bus.jpg --conf 0.25 --iou 0.45 --out result.jpg
./yolo_onnxruntime --model yolo26n.onnx        --source bus.jpg
./yolo_onnxruntime --model yolo26n-seg.onnx    --source bus.jpg --out seg.jpg
./yolo_onnxruntime --model yolo11n-pose.onnx   --source bus.jpg --show
./yolo_onnxruntime --model yolo26n-sem.onnx    --source street.jpg
```

| Argument   | Default         | Description                                                       |
| :--------- | :-------------- | :--------------------------------------------------------------- |
| `--model`  | `yolo26n.onnx`  | Path to the exported `.onnx` model (any task/generation).        |
| `--source` | `bus.jpg`       | Input image.                                                     |
| `--conf`   | `0.25`          | Confidence threshold.                                            |
| `--iou`    | `0.45`          | NMS IoU threshold (grid models only; end2end models skip NMS).   |
| `--out`    | `result.jpg`    | Output image path.                                               |
| `--cuda`   | _off_           | Use the CUDA execution provider (requires a GPU ONNX Runtime build compiled with `-DUSE_CUDA=ON`). |
| `--show`   | _off_           | Also open a display window.                                      |

The annotated result is always written to `--out` and the detections are printed to the console. The task is shown at startup, e.g. `Model: yolo26n.onnx | task: detect | classes: 80`.

## 🏷️ Class Names & Task

The task type, class `names`, and input `imgsz` are read directly from the model metadata that Ultralytics bakes into every export — so the same binary handles a COCO detector, a 1000-class ImageNet classifier, and a 19-class Cityscapes semantic model with no changes. If a model somehow lacks `names`, the example falls back to the 80 [COCO](https://docs.ultralytics.com/datasets/detect/coco) names from [`../common/coco_names.hpp`](../common/coco_names.hpp).

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
