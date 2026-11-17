# Ultralytics YOLO OpenCV DNN Inference in Python

<img alt="Python" src="https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white"> <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV--DNN-5C3EE8.svg?logo=opencv&logoColor=white">

A single Python script that runs [Ultralytics YOLO](https://docs.ultralytics.com) ONNX models with the [OpenCV DNN module](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html) on every task ([detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [semantic](https://docs.ultralytics.com/tasks/semantic), [depth](https://docs.ultralytics.com/tasks/depth), [classify](https://docs.ultralytics.com/tasks/classify), [pose](https://docs.ultralytics.com/tasks/pose), and [OBB](https://docs.ultralytics.com/tasks/obb)). It supports [YOLOv8](https://docs.ultralytics.com/models/yolov8), [YOLO11](https://docs.ultralytics.com/models/yolo11), and [YOLO26](https://docs.ultralytics.com/models/yolo26) **grid** exports. OpenCV DNN cannot run the YOLO26 end-to-end (NMS in graph) operators in every version, so export with the default `nms=None`.

OpenCV reads no ONNX metadata, so the task is inferred from the output shapes and class names fall back to [COCO](https://docs.ultralytics.com/datasets/detect/coco).

## ⚙️ Installation

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics/examples/python/OpenCV-DNN
pip install -r requirements.txt
```

## 🚀 Usage

```bash
yolo export model=yolo26n.pt format=onnx opset=12
python main.py --model yolo26n.onnx --source bus.jpg

# Semantic, depth, classification, and OBB models need their task and input size
yolo export model=yolo26n-cls.pt format=onnx opset=12
python main.py --model yolo26n-cls.onnx --task classify --imgsz 224
```

| Argument   | Default        | Description                                                              |
| :--------- | :------------- | :----------------------------------------------------------------------- |
| `--model`  | `yolo26n.onnx` | Path to the exported grid `.onnx` model.                                 |
| `--source` | `bus.jpg`      | Input image.                                                             |
| `--conf`   | `0.25`         | Confidence threshold.                                                    |
| `--iou`    | `0.5`          | NMS IoU threshold.                                                       |
| `--imgsz`  | `640`          | Model input size (`768` depth, `224` classify, `1024` OBB and semantic). |
| `--task`   | _inferred_     | Set for semantic, depth, classify, OBB, and `rtdetr` models.             |
| `--out`    | `result.jpg`   | Output image path.                                                       |
| `--show`   | _off_          | Also open a display window.                                              |

The shared pre and post-processing lives in [`../common`](../common).

## 🤝 Contributing

Contributions are welcome! If you find an issue or have a suggestion, please open an issue or pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
