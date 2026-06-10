<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO C++ Examples

This directory groups all of the C++ inference examples for [Ultralytics YOLO](https://docs.ultralytics.com/models/) models in one place. Each subfolder is a self-contained project showing how to run [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26), [YOLO11](https://docs.ultralytics.com/models/yolo11), and [YOLOv8](https://docs.ultralytics.com/models/yolov8) detection models against a different inference backend.

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
> These examples run their own [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) on the raw model output. For **YOLO26**, export with NMS disabled (add `nms=False` to the export command) so the output tensor shape matches what the C++ code expects. Native support for YOLO26 end-to-end (NMS-included) outputs is a planned follow-up.

### OpenCV-DNN

```bash
# 1. Export an ONNX model into the example directory
yolo export model=yolo11s.pt imgsz=640,480 format=onnx opset=12

# 2. Build
cd examples/cpp/OpenCV-DNN && mkdir build && cd build && cmake .. && make

# 3. Edit projectBasePath / model path in main.cpp, then run
./yolo_opencv_dnn
```

### ONNXRuntime

```bash
# 1. Export an ONNX model
yolo export model=yolo11n.pt format=onnx opset=12 simplify=True dynamic=False imgsz=640

# 2. Build (point ONNXRUNTIME_ROOT at your ONNX Runtime install)
cd examples/cpp/ONNXRuntime && mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime && make

# 3. Place the .onnx model and coco.yaml next to the binary, then run
./yolo_onnxruntime
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

### OpenVINO

```bash
# 1. Export an OpenVINO IR (or ONNX) model
yolo export model=yolo11s.pt imgsz=640 format=openvino

# 2. Build
cd examples/cpp/OpenVINO && mkdir build && cd build && cmake .. && make

# 3. Run with the model and image paths as arguments
./yolo_openvino path/to/model.xml path/to/image.jpg
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
