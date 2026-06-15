# Ultralytics YOLO LibTorch Inference C++

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="LibTorch" src="https://img.shields.io/badge/LibTorch-EE4C2C.svg?logo=pytorch&logoColor=white"> <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?logo=opencv&logoColor=white">

A C++ application that runs every [Ultralytics YOLO](https://docs.ultralytics.com/) task and model generation as [TorchScript](https://docs.pytorch.org/docs/stable/jit.html) with the [LibTorch (PyTorch C++ API)](https://docs.pytorch.org/cppdocs/) and [OpenCV](https://opencv.org/). Point it at any exported `.torchscript` model; the task, class names, and input size are read from the model metadata, and the right post-processing is selected automatically.

## ✨ Features

- **All tasks:** [detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [pose](https://docs.ultralytics.com/tasks/pose), [OBB](https://docs.ultralytics.com/tasks/obb), [classify](https://docs.ultralytics.com/tasks/classify), and YOLO26 semantic segmentation.
- **All generations:** [YOLOv8](https://docs.ultralytics.com/models/yolov8), [YOLO11](https://docs.ultralytics.com/models/yolo11), and [YOLO26](https://docs.ultralytics.com/models/yolo26). Grid (YOLOv8/11) and end-to-end (YOLO26) outputs are detected automatically.
- **Zero configuration:** task, class names, and `imgsz` come from the TorchScript `config.txt` metadata that Ultralytics embeds on export.

## ⚙️ Dependencies

Ensure you have the following dependencies installed before proceeding:

| Dependency   | Version  | Resource                                     |
| :----------- | :------- | :------------------------------------------- |
| OpenCV       | >=4.0.0  | [https://opencv.org/](https://opencv.org/)   |
| C++ Standard | >=17     | [https://isocpp.org/](https://isocpp.org/)   |
| CMake        | >=3.18   | [https://cmake.org/](https://cmake.org/)     |
| Libtorch     | >=1.12.1 | [https://pytorch.org/](https://pytorch.org/) |

You can download the required version of LibTorch from the official [PyTorch](https://pytorch.org/) website. Make sure to select the correct version corresponding to your system and CUDA version (if using GPU).

## 📦 Exporting a Model

Export any model and task to TorchScript with the Ultralytics `export` mode:

```bash
yolo export model=yolo26n.pt      imgsz=640 format=torchscript   # detect (also -seg / -pose / -obb / -cls / -sem)
yolo export model=yolo11n.pt      imgsz=640 format=torchscript   # YOLOv8/YOLO11 work too
```

See the [Export documentation](https://docs.ultralytics.com/modes/export) for more options.

## 🛠️ Build

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics/examples/cpp/LibTorch
mkdir build && cd build

# Add -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/opencv" if they are not auto-detected.
cmake .. && cmake --build . --config Release
```

The shared helpers in [`../common`](../common) are header-only and added to the include path automatically.

## 🚀 Usage

```bash
# If LibTorch is not installed system-wide, add its libraries to the loader path:
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH

# Defaults: --model yolo26n.torchscript --source bus.jpg --conf 0.25 --iou 0.45 --out result.jpg
./yolo_libtorch --model yolo26n.torchscript      --source bus.jpg
./yolo_libtorch --model yolo26n-seg.torchscript  --source bus.jpg --out seg.jpg
./yolo_libtorch --model yolo11n-pose.torchscript --source bus.jpg --show
```

| Argument   | Default               | Description                                              |
| :--------- | :-------------------- | :------------------------------------------------------- |
| `--model`  | `yolo26n.torchscript` | Path to the exported TorchScript model (any task).      |
| `--source` | `bus.jpg`             | Input image.                                            |
| `--conf`   | `0.25`                | Confidence threshold.                                   |
| `--iou`    | `0.45`                | NMS IoU threshold (grid models only).                  |
| `--out`    | `result.jpg`          | Output image path.                                     |
| `--cuda`   | _off_                 | Use CUDA if the LibTorch build and a device support it. |
| `--show`   | _off_                 | Also open a display window.                             |

The annotated result is always written to `--out` and the detections are printed to the console. The detected task is shown at startup, e.g. `Model: yolo26n.torchscript | task: detect | classes: 80`.

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
