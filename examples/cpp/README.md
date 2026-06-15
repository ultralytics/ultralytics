<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO C++ Examples

This directory groups all of the C++ inference examples for [Ultralytics YOLO](https://docs.ultralytics.com/models/) models in one place. Each subfolder is a self-contained project showing how to run [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26), [YOLO11](https://docs.ultralytics.com/models/yolo11), and [YOLOv8](https://docs.ultralytics.com/models/yolov8) models against a different inference backend.

The **ONNXRuntime** and **OpenVINO** examples support **every task** — [detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [pose](https://docs.ultralytics.com/tasks/pose), [OBB](https://docs.ultralytics.com/tasks/obb), [classify](https://docs.ultralytics.com/tasks/classify), and YOLO26 semantic segmentation — selecting the task automatically from the model metadata, and handle both grid (YOLOv8/11) and end-to-end (YOLO26) outputs. The other backends focus on object detection.

## 📂 Examples

| Example                      | Backend                                                              | Build target(s)                   | Notes                                                       |
| ---------------------------- | ------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------- |
| [OpenCV-DNN](./OpenCV-DNN)   | [OpenCV DNN](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)     | `yolo_opencv_dnn`                 | ONNX models via the OpenCV DNN module (CPU/CUDA).           |
| [ONNXRuntime](./ONNXRuntime) | [ONNX Runtime](https://onnxruntime.ai/)                             | `yolo_onnxruntime`                | ONNX models with FP32/FP16, CPU or CUDA execution provider. |
| [LibTorch](./LibTorch)       | [LibTorch](https://docs.pytorch.org/cppdocs/)                       | `yolo_libtorch`                   | TorchScript models via the PyTorch C++ API.                 |
| [MNN](./MNN)                 | [Alibaba MNN](https://mnn-docs.readthedocs.io/en/latest/)           | `yolo_mnn`, `yolo_mnn_interpreter`| MNN models (FP32/FP16/INT8), Express and Interpreter APIs.  |
| [OpenVINO](./OpenVINO)       | [Intel OpenVINO](https://docs.openvino.ai/)                         | `yolo_openvino`                   | OpenVINO IR or ONNX models on Intel hardware.               |
| [Triton](./Triton)           | [NVIDIA Triton](https://github.com/triton-inference-server/server)  | `yolo_triton`                     | gRPC client for a model served by Triton Inference Server.  |

## ✅ How to Test

All examples follow the same flow: **export a model → build the C++ project → run the executable**. Install the [Ultralytics package](https://docs.ultralytics.com/quickstart/) first (`pip install ultralytics`) so the `yolo export` command is available, then pick an example below.

> [!NOTE]
> The **ONNXRuntime** and **OpenVINO** examples detect the output layout automatically, so YOLOv8/11 (grid) and YOLO26 (end-to-end, NMS-free) models both work out of the box. The other backends run their own [NMS](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) on grid outputs, so for those export YOLO26 with NMS disabled (`nms=False`).

### OpenCV-DNN

```bash
# 1. Export an ONNX model into the example directory
yolo export model=yolo11s.pt imgsz=640,480 format=onnx opset=12

# 2. Build
cd examples/cpp/OpenCV-DNN && mkdir build && cd build && cmake .. && make

# 3. Edit projectBasePath / model path in main.cpp, then run
./yolo_opencv_dnn
```

### ONNXRuntime (all tasks)

```bash
# 1. Export any model and task — the task and class names are read from the metadata
yolo export model=yolo26n.pt format=onnx opset=12         # or -seg / -pose / -obb / -cls / -sem

# 2. Build (point ONNXRUNTIME_ROOT at your ONNX Runtime install; -DUSE_CUDA=OFF for CPU)
cd examples/cpp/ONNXRuntime && mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime -DUSE_CUDA=OFF && make

# 3. Run — task auto-detected, result written to the --out path
LD_LIBRARY_PATH=/path/to/onnxruntime/lib ./yolo_onnxruntime --model yolo26n.onnx --source bus.jpg
```

### LibTorch

```bash
# 1. Export a TorchScript model
yolo export model=yolo11s.pt imgsz=640 format=torchscript

# 2. Build (add CMAKE_PREFIX_PATH if LibTorch/OpenCV are not auto-detected)
cd examples/cpp/LibTorch && mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/opencv" && make

# 3. Set the model and image paths in main.cc, then run
./yolo_libtorch
```

### MNN

```bash
# 1. Export an MNN model
yolo export model=yolo11n.pt imgsz=640 format=mnn

# 2. Build MNN, then the example (see MNN/README.md for the full MNN build)
cd examples/cpp/MNN && mkdir build && cd build && cmake .. && make

# 3. Run (Express API or Interpreter API)
./yolo_mnn yolo11n.mnn bus.jpg
./yolo_mnn_interpreter yolo11n.mnn bus.jpg
```

### OpenVINO (all tasks)

```bash
# 1. Export an OpenVINO IR (or ONNX) model — any task
yolo export model=yolo26n.pt imgsz=640 format=openvino

# 2. Build
cd examples/cpp/OpenVINO && mkdir build && cd build && cmake .. && make

# 3. Run — task auto-detected from the output shapes
./yolo_openvino --model yolo26n_openvino_model/yolo26n.xml --source bus.jpg
```

### Triton

```bash
# 1. Deploy an FP16 YOLO model on an NVIDIA Triton Inference Server (model name "yolo11")

# 2. Build the client (point TRITON_CLIENT_DIR at the Triton client libraries)
cd examples/cpp/Triton && mkdir build && cd build
cmake .. -DTRITON_CLIENT_DIR=/path/to/tritonclient && make

# 3. Run the client (connects to localhost:8001 by default)
./yolo_triton
```

See each example's own `README.md` for full dependency lists, platform notes, and configuration options.

## 🤝 Contributing

Contributions are welcome! See the [Ultralytics Contributing Guide](https://docs.ultralytics.com/help/contributing) for details. If you find an issue with one of these examples, please open an issue or pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
