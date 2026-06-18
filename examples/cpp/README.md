<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO C++ Examples

This directory groups all of the C++ inference examples for [Ultralytics YOLO](https://docs.ultralytics.com/models/) models in one place. Each subfolder is a self-contained project showing how to run [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26), [YOLO11](https://docs.ultralytics.com/models/yolo11), and [YOLOv8](https://docs.ultralytics.com/models/yolov8) models against a different inference backend.

Every backend supports **every task** — [detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [pose](https://docs.ultralytics.com/tasks/pose), [OBB](https://docs.ultralytics.com/tasks/obb), [classify](https://docs.ultralytics.com/tasks/classify), and YOLO26 semantic segmentation — selecting the task automatically from the model metadata or output shapes. Most also handle both grid (YOLOv8/11) and end-to-end (YOLO26) outputs; **OpenCV-DNN** supports the same tasks but only on **grid** models, because the OpenCV DNN module cannot run the YOLO26 end-to-end (NMS-in-graph) operators.

## 📂 Examples

| Example                      | Backend                                                            | Build target       | Notes                                                                       |
| ---------------------------- | ------------------------------------------------------------------ | ------------------ | --------------------------------------------------------------------------- |
| [OpenCV-DNN](./OpenCV-DNN)   | [OpenCV DNN](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)   | `yolo_opencv_dnn`  | All tasks on grid models (YOLOv8/11, or YOLO26 with `nms=False`); CPU/CUDA. |
| [ONNXRuntime](./ONNXRuntime) | [ONNX Runtime](https://onnxruntime.ai/)                            | `yolo_onnxruntime` | All tasks; ONNX FP32/FP16; CPU or CUDA execution provider.                  |
| [LibTorch](./LibTorch)       | [LibTorch](https://docs.pytorch.org/cppdocs/)                      | `yolo_libtorch`    | All tasks; TorchScript via the PyTorch C++ API.                             |
| [MNN](./MNN)                 | [Alibaba MNN](https://mnn-docs.readthedocs.io/en/latest/)          | `yolo_mnn`         | All tasks; MNN models on CPU.                                               |
| [OpenVINO](./OpenVINO)       | [Intel OpenVINO](https://docs.openvino.ai/)                        | `yolo_openvino`    | All tasks; OpenVINO IR or ONNX on Intel hardware.                           |
| [Triton](./Triton)           | [NVIDIA Triton](https://github.com/triton-inference-server/server) | `yolo_triton`      | All tasks; gRPC client for a model served by Triton.                        |

## ✅ How to Test

All examples follow the same flow: **export a model → build the C++ project → run the executable**. Install the [Ultralytics package](https://docs.ultralytics.com/quickstart/) first (`pip install ultralytics`) so the `yolo export` command is available, then pick an example below.

The **ONNXRuntime**, **OpenVINO**, **LibTorch**, **MNN**, and **Triton** examples support every task (detect, segment, pose, OBB, classify, semantic) and read the task and class names from the model metadata (or, for Triton, infer the task from the output shapes), so the same binary handles any model. They take the model and image as `--model` / `--source` arguments.

> [!NOTE]
> These five examples detect the output layout automatically, so YOLOv8/11 (grid) and YOLO26 (end-to-end, NMS-free) models both work out of the box. **OpenCV-DNN** supports the same tasks but only on grid models, since the OpenCV DNN module cannot run YOLO26 end-to-end operators — for it, use grid models (YOLOv8/11, or YOLO26 exported with `nms=False`).

### OpenCV-DNN

```bash
# 1. Export a grid ONNX model (YOLO26 needs nms=False; cv::dnn cannot run end2end ops)
yolo export model=yolo26n.pt format=onnx opset=12 imgsz=640 nms=False

# 2. Build
cd examples/cpp/OpenCV-DNN && mkdir build && cd build && cmake .. && make

# 3. Run (use --task for grid pose/obb; class names fall back to COCO)
./yolo_opencv_dnn --model yolo26n.onnx --source bus.jpg
```

### ONNXRuntime (all tasks)

```bash
# 1. Export any model and task — the task and class names are read from the metadata
yolo export model=yolo26n.pt format=onnx opset=12 # or -seg / -pose / -obb / -cls / -sem

# 2. Build (point ONNXRUNTIME_ROOT at your ONNX Runtime install; -DUSE_CUDA=OFF for CPU)
cd examples/cpp/ONNXRuntime && mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime -DUSE_CUDA=OFF && make

# 3. Run — task auto-detected, result written to the --out path
LD_LIBRARY_PATH=/path/to/onnxruntime/lib ./yolo_onnxruntime --model yolo26n.onnx --source bus.jpg
```

### LibTorch (all tasks)

```bash
# 1. Export a TorchScript model (any task)
yolo export model=yolo26n.pt imgsz=640 format=torchscript

# 2. Build (add CMAKE_PREFIX_PATH if LibTorch/OpenCV are not auto-detected)
cd examples/cpp/LibTorch && mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch;/path/to/opencv" && make

# 3. Run — task auto-detected from the model metadata
LD_LIBRARY_PATH=/path/to/libtorch/lib ./yolo_libtorch --model yolo26n.torchscript --source bus.jpg
```

### MNN (all tasks)

```bash
# 1. Export an MNN model (any task). Prefer format=mnn over MNNConvert so metadata is kept.
yolo export model=yolo26n.pt imgsz=640 format=mnn

# 2. Build MNN from source, then the example pointing at the MNN headers and library
#    (see MNN/README.md for the full MNN build)
cd examples/cpp/MNN && mkdir build && cd build
cmake .. -DMNN_INCLUDE_DIR=/path/to/MNN/include -DMNN_LIB_DIR=/path/to/MNN/build && make

# 3. Run — task auto-detected from the bizCode metadata
LD_LIBRARY_PATH=/path/to/MNN/build ./yolo_mnn --model yolo26n.mnn --source bus.jpg
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
# 1. Deploy any YOLO model on an NVIDIA Triton Inference Server (e.g. model name "yolo26_det")

# 2. Build the client (point TRITON_CLIENT_DIR at the Triton client libraries)
cd examples/cpp/Triton && mkdir build && cd build
cmake .. -DTRITON_CLIENT_DIR=/path/to/tritonclient && make

# 3. Run the client - task auto-detected from the model outputs (default URL localhost:8001).
#    YOLO26 auto-detects every task; only legacy grid YOLOv8/11 pose/obb need --task.
./yolo_triton --model yolo26_det --source bus.jpg
./yolo_triton --model yolo26_pose --source bus.jpg
```

See each example's own `README.md` for full dependency lists, platform notes, and configuration options.

## 🤝 Contributing

Contributions are welcome! See the [Ultralytics Contributing Guide](https://docs.ultralytics.com/help/contributing) for details. If you find an issue with one of these examples, please open an issue or pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
