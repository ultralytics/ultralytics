# Ultralytics YOLO Triton Inference in C++

<img alt="C++" src="https://img.shields.io/badge/C%2B%2B-17-00599C.svg?logo=cplusplus&logoColor=white"> <img alt="NVIDIA Triton" src="https://img.shields.io/badge/NVIDIA%20Triton-76B900.svg?logo=nvidia&logoColor=white"> <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?logo=opencv&logoColor=white"> <img alt="gRPC" src="https://img.shields.io/badge/gRPC-244c5a.svg?logo=google&logoColor=white">

A C++ gRPC client that runs **every [Ultralytics YOLO](https://docs.ultralytics.com/) task and model generation** against a model served by the [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server). The client reads the input/output layout from the model metadata, infers the task from the output shapes, and shares its post-processing with the other C++ examples — so the same binary handles detection, segmentation, pose, OBB, classification, and Ultralytics YOLO26 semantic segmentation.

## ✨ Features

- **All tasks:** [detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [pose](https://docs.ultralytics.com/tasks/pose), [OBB](https://docs.ultralytics.com/tasks/obb), [classify](https://docs.ultralytics.com/tasks/classify), and Ultralytics YOLO26 semantic segmentation.
- **All generations:** [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8), [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11), and [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26). The grid output of YOLOv8/11 and the end-to-end (NMS-free) output of YOLO26 are detected automatically from the tensor shape.
- **FP16 and FP32:** the input and output datatypes are read from the model metadata, so half-precision ([FP16](https://www.ultralytics.com/glossary/half-precision)) and full-precision models both work with no flags.
- **Seamless Triton integration:** communicates with the server over gRPC for efficient, scalable model serving.
- **Simple CLI:** choose the server URL, model name, source image, and thresholds at runtime — no recompiling.

## 📋 Dependencies

Ensure you have the following dependencies installed before proceeding:

| Dependency              | Version | Description                                   |
| ----------------------- | ------- | --------------------------------------------- |
| Triton Inference Server | 22.06+  | Running with a deployed YOLO model            |
| Triton Client libraries | 2.23+   | Required for communication with Triton Server |
| C++ compiler            | C++ 17+ | For compiling the C++ client application      |
| OpenCV library          | >=3.4   | For image processing and visualization        |
| CMake                   | 3.5+    | For building the project                      |

For more information on Triton, see the [NVIDIA Triton Inference Server documentation](https://github.com/triton-inference-server/server) and explore [model deployment options with Ultralytics](https://docs.ultralytics.com/guides/model-deployment-options).

## 📦 Deploying a Model

Export any model and task, then add it to a Triton [model repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md). ONNX is a convenient serving format and keeps the `output0` (and `output1` for segmentation) tensor names this client expects.

```bash
yolo export model=yolo26n.pt format=onnx opset=12              # detect   (end2end)
yolo export model=yolo26n-seg.pt format=onnx opset=12          # segment
yolo export model=yolo26n-pose.pt format=onnx opset=12         # pose
yolo export model=yolo26n-obb.pt format=onnx opset=12          # obb
yolo export model=yolo26n-cls.pt format=onnx opset=12          # classify
yolo export model=yolo26n-sem.pt format=onnx opset=12          # semantic
yolo export model=yolo11n.pt format=onnx opset=12 dynamic=True # YOLOv8/YOLO11 (grid) work too
```

Add `quantize=16 device=0` to export an FP16 model on a GPU; the client reads the input/output datatype from the metadata and handles FP16 or FP32 automatically.

Place the exported model under `<repository>/<model_name>/1/model.onnx`. Triton's ONNX backend auto-completes the configuration, so a `config.pbtxt` is optional. A minimal repository looks like:

```text
models/
└── yolo26n/
    └── 1/
        └── model.onnx
```

Then start Triton pointing at the repository:

```bash
tritonserver --model-repository=/models
```

See the [Ultralytics Triton guide](https://docs.ultralytics.com/guides/triton-inference-server) for a full walkthrough.

## 🛠️ Building the Project

1. **Install the Triton Client libraries:**

   ```bash
   wget https://github.com/triton-inference-server/server/releases/download/v2.23.0/v2.23.0_ubuntu2004.clients.tar.gz
   mkdir tritonclient
   tar -xvf v2.23.0_ubuntu2004.clients.tar.gz -C tritonclient
   rm -f v2.23.0_ubuntu2004.clients.tar.gz
   ```

2. **Clone the Ultralytics repository:**

   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics/examples/cpp/Triton
   ```

3. **Configure and build the project using CMake:**

   ```bash
   mkdir build
   cd build
   cmake .. -DTRITON_CLIENT_DIR=/path/to/tritonclient
   make
   ```

The shared helpers in [`../common`](../common) are header-only and added to the include path automatically.

## 🚀 Usage

Start your Triton server with a deployed Ultralytics YOLO model, then run the client. Use the model name as deployed in the repository as `--model`.

```bash
# Defaults: --url localhost:8001 --model yolo26n --source bus.jpg --conf 0.25 --iou 0.45 --out result.jpg
./yolo_triton --model yolo26n --source bus.jpg                     # detect   (auto)
./yolo_triton --model yolo26n-seg --source bus.jpg --out seg.jpg   # segment  (auto)
./yolo_triton --model yolo26n-pose --source bus.jpg --out pose.jpg # pose     (auto, end2end)
./yolo_triton --model yolo26n-obb --source boats.jpg --out obb.jpg # obb      (auto, end2end)
./yolo_triton --model yolo26n-cls --source bus.jpg --out cls.jpg   # classify (auto)
./yolo_triton --model yolo26n-sem --source bus.jpg --out sem.jpg   # semantic (auto)
./yolo_triton --model yolo11n-pose --source bus.jpg --task pose    # legacy grid pose: needs --task
./yolo_triton --url 192.168.1.10:8001 --model yolo26n --source street.jpg --show
```

> [!NOTE]
> Triton exposes no task or class-name metadata, so the task is inferred from the output shapes. With **YOLO26** (end-to-end) models every task — including pose and OBB is detected automatically. Only the legacy **grid** YOLOv8/11 **pose** `[1, 56, 8400]` and **obb** `[1, 20, 8400]` outputs are ambiguous with detection (they differ only by the class count, which Triton does not expose), so for those pass `--task pose` or `--task obb`. Class names fall back to COCO, so a non-COCO model (1000-class classify, DOTA obb) prints class indices rather than names.

| Argument    | Default          | Description                                                    |
| :---------- | :--------------- | :------------------------------------------------------------- |
| `--url`     | `localhost:8001` | Triton server gRPC endpoint.                                   |
| `--model`   | `yolo26n`        | Model name as deployed in the Triton repository.               |
| `--version` | _latest_         | Model version (empty selects the latest).                      |
| `--source`  | `bus.jpg`        | Input image.                                                   |
| `--conf`    | `0.25`           | Confidence threshold.                                          |
| `--iou`     | `0.45`           | NMS IoU threshold (grid models only; end2end models skip NMS). |
| `--imgsz`   | `640`            | Input size used when the model shape is dynamic.               |
| `--task`    | _auto_           | Override the inferred task (`detect`, `segment`, `pose`, ...). |
| `--out`     | `result.jpg`     | Output image path.                                             |
| `--show`    | _off_            | Also open a display window.                                    |

The annotated result is always written to `--out` and the detections are printed to the console.

## 🏷️ Class Names & Task

Triton exposes no class names, so the example falls back to the 80 [COCO](https://docs.ultralytics.com/datasets/detect/coco) names from [`../common/coco_names.hpp`](../common/coco_names.hpp). The task is inferred from the output shapes; pass `--task` to override it (required for grid pose/obb, as noted above).

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
