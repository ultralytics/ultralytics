# Ultralytics YOLO MNN Inference in C++

<img alt="C++" src="https://img.shields.io/badge/C%2B%2B-11-00599C.svg?logo=cplusplus&logoColor=white"> <img alt="MNN" src="https://img.shields.io/badge/MNN-FF6A00.svg?logo=alibabacloud&logoColor=white"> <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?logo=opencv&logoColor=white">

A C++ application that runs every [Ultralytics YOLO](https://docs.ultralytics.com/) task and model generation with the [Alibaba MNN](https://mnn-docs.readthedocs.io/en/latest/) inference engine and [OpenCV](https://opencv.org/). Point it at any `.mnn` model; the task, class names, and input size are read from the model `bizCode` metadata, and the right post-processing is selected automatically.

## 🌟 Features

- **All tasks:** [detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [pose](https://docs.ultralytics.com/tasks/pose), [OBB](https://docs.ultralytics.com/tasks/obb), [classify](https://docs.ultralytics.com/tasks/classify), and YOLO26 semantic segmentation.
- **All generations:** [YOLOv8](https://docs.ultralytics.com/models/yolov8), [YOLO11](https://docs.ultralytics.com/models/yolo11), and [YOLO26](https://docs.ultralytics.com/models/yolo26). Grid and end-to-end outputs are detected automatically.
- **Zero configuration:** the task, class names, and `imgsz` come from the MNN `bizCode` metadata. When a model has none (for example after a plain `MNNConvert`), the task is inferred from the output shapes and names fall back to COCO.

## 📋 Dependencies

| Dependency                                        | Version  | Description                              |
| :------------------------------------------------ | :------- | :--------------------------------------- |
| [MNN](https://mnn-docs.readthedocs.io/en/latest/) | >=2.0.0  | The core inference engine from Alibaba.  |
| [OpenCV](https://opencv.org/)                     | >=4.0.0  | Image I/O, drawing, and NMS.             |
| [C++](https://en.cppreference.com/w/)             | >=17     | Modern C++ compiler.                     |
| [CMake](https://cmake.org/documentation/)         | >=3.12.0 | Build system.                            |

## ⚙️ Build

First build the MNN library and converter from source:

```bash
git clone https://github.com/alibaba/MNN.git && cd MNN
mkdir build && cd build
cmake -DMNN_BUILD_CONVERTER=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Then build the example, pointing it at the MNN headers and library:

```bash
cd ultralytics/examples/cpp/MNN
mkdir build && cd build
cmake .. -DMNN_INCLUDE_DIR=/path/to/MNN/include -DMNN_LIB_DIR=/path/to/MNN/build
cmake --build . --config Release
```

The shared helpers in [`../common`](../common) are header-only and added to the include path automatically.

## 📦 Exporting a Model

Export directly to MNN with the Ultralytics `export` mode. This is the **recommended** path: it keeps the model metadata in `bizCode`, so the task and class names are read automatically.

```bash
yolo export model=yolo26n.pt imgsz=640 format=mnn     # detect (also -seg / -pose / -obb / -cls / -sem)
```

> [!NOTE]
> Prefer `format=mnn` over the standalone `MNNConvert` tool. `MNNConvert --bizCode <code>` overwrites the model metadata, so the task is then inferred from the output shapes and class names fall back to COCO (wrong for OBB, classify, and semantic models). `MNNConvert` can also fail to convert some YOLO26 segment graphs.

If you still want to convert an existing ONNX model:

```bash
yolo export model=yolo11n.pt format=onnx opset=12
/path/to/MNN/build/MNNConvert -f ONNX --modelFile yolo11n.onnx --MNNModel yolo11n.mnn --bizCode biz
```

## 🚀 Usage

```bash
# If MNN is built as a shared library, add it to the loader path:
export LD_LIBRARY_PATH=/path/to/MNN/build:$LD_LIBRARY_PATH

# Defaults: --model yolo26n.mnn --source bus.jpg --conf 0.25 --iou 0.45 --out result.jpg
./yolo_mnn --model yolo26n.mnn      --source bus.jpg
./yolo_mnn --model yolo26n-pose.mnn --source bus.jpg --out pose.jpg --show
```

| Argument    | Default       | Description                          |
| :---------- | :------------ | :----------------------------------- |
| `--model`   | `yolo26n.mnn` | Path to the exported MNN model.      |
| `--source`  | `bus.jpg`     | Input image.                         |
| `--conf`    | `0.25`        | Confidence threshold.                |
| `--iou`     | `0.45`        | NMS IoU threshold (grid models only).|
| `--threads` | `4`           | CPU threads.                         |
| `--out`     | `result.jpg`  | Output image path.                   |
| `--show`    | _off_         | Also open a display window.          |

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
