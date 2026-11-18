# Ultralytics YOLO ONNX Runtime Inference in Python

<img alt="Python" src="https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white"> <img alt="ONNX Runtime" src="https://img.shields.io/badge/ONNX%20Runtime-717272.svg?logo=onnx&logoColor=white"> <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?logo=opencv&logoColor=white">

A single Python script that runs **every [Ultralytics YOLO](https://docs.ultralytics.com) task and model generation** with [ONNX Runtime](https://onnxruntime.ai/). Point it at any exported `.onnx` model; the task, class names, and input size are read from the model metadata and the right post-processing is chosen automatically.

## ✨ Features

- **All tasks:** [detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [semantic](https://docs.ultralytics.com/tasks/semantic), [depth](https://docs.ultralytics.com/tasks/depth), [classify](https://docs.ultralytics.com/tasks/classify), [pose](https://docs.ultralytics.com/tasks/pose), and [OBB](https://docs.ultralytics.com/tasks/obb).
- **All generations:** [YOLOv8](https://docs.ultralytics.com/models/yolov8), [YOLO11](https://docs.ultralytics.com/models/yolo11), [YOLO26](https://docs.ultralytics.com/models/yolo26), and [RT-DETR](https://docs.ultralytics.com/models/rtdetr). Grid and end-to-end (NMS-free) outputs are detected from the tensor shape.
- **Zero configuration:** task, class names, and `imgsz` come from the model metadata.
- **FP32 and FP16:** half-precision models are detected from the input type.
- **CPU or GPU:** uses the CUDA execution provider when `onnxruntime-gpu` is installed.

## ⚙️ Installation

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics/examples/python/ONNXRuntime
pip install -r requirements.txt
```

## 📦 Exporting a Model

```bash
yolo export model=yolo26n.pt format=onnx opset=12       # detect
yolo export model=yolo26n-seg.pt format=onnx opset=12   # segment
yolo export model=yolo26n-sem.pt format=onnx opset=12   # semantic
yolo export model=yolo26n-depth.pt format=onnx opset=12 # depth
yolo export model=yolo26n-cls.pt format=onnx opset=12   # classify
yolo export model=yolo26n-pose.pt format=onnx opset=12  # pose
yolo export model=yolo26n-obb.pt format=onnx opset=12   # obb
yolo export model=rtdetr-l.pt format=onnx opset=17      # RT-DETR
```

See the [Export documentation](https://docs.ultralytics.com/modes/export) for more options.

## 🚀 Usage

```bash
# Defaults: --model yolo26n.onnx --source bus.jpg --conf 0.25 --iou 0.5 --out result.jpg
python main.py --model yolo26n.onnx --source bus.jpg
python main.py --model yolo26n-seg.onnx --source bus.jpg --out seg.jpg
python main.py --model yolo26n-pose.onnx --source bus.jpg --show
```

| Argument   | Default        | Description                                                  |
| :--------- | :------------- | :----------------------------------------------------------- |
| `--model`  | `yolo26n.onnx` | Path to the exported `.onnx` model (any task or generation). |
| `--source` | `bus.jpg`      | Input image.                                                 |
| `--conf`   | `0.25`         | Confidence threshold.                                        |
| `--iou`    | `0.5`          | NMS IoU threshold (grid models only).                        |
| `--out`    | `result.jpg`   | Output image path.                                           |
| `--task`   | _metadata_     | Override the task read from the model metadata.              |
| `--show`   | _off_          | Also open a display window.                                  |

The annotated result is written to `--out` and the detections are printed to the console. The shared pre and post-processing lives in [`../common`](../common).

## 🤝 Contributing

Contributions are welcome! If you find an issue or have a suggestion, please open an issue or pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
