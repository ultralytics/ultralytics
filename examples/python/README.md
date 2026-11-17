<a href="https://www.ultralytics.com" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO Python Examples

<img alt="Python" src="https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white"> <a href="https://docs.ultralytics.com" target="_blank"><img alt="Ultralytics Docs" src="https://img.shields.io/badge/Ultralytics-Docs-042AFF.svg?logo=ultralytics&logoColor=white"></a>

This directory groups all of the Python examples for [Ultralytics YOLO](https://docs.ultralytics.com/models) models in one place. Each subfolder is a self-contained project with its own `README.md` and `requirements.txt`, showing how to run [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26), [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11), and [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8) models.

The inference examples support **every task** ([detect](https://docs.ultralytics.com/tasks/detect), [segment](https://docs.ultralytics.com/tasks/segment), [semantic](https://docs.ultralytics.com/tasks/semantic), [depth](https://docs.ultralytics.com/tasks/depth), [classify](https://docs.ultralytics.com/tasks/classify), [pose](https://docs.ultralytics.com/tasks/pose), and [OBB](https://docs.ultralytics.com/tasks/obb)) plus [RT-DETR](https://docs.ultralytics.com/models/rtdetr), and share their pre and post-processing in [`common`](./common), so each `main.py` only has to run the model.

## 📂 Examples

| Example                                                                                                                                                                                                                       | Backend                                                                   | Notes                                                                |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [![ONNXRuntime](https://img.shields.io/badge/ONNX%20Runtime-717272.svg?logo=onnx&logoColor=white)](https://github.com/ultralytics/ultralytics/tree/main/examples/python/ONNXRuntime)                                          | [ONNX Runtime](https://onnxruntime.ai/)                                   | All tasks; grid and end-to-end outputs; ONNX FP32/FP16; CPU or CUDA. |
| [![OpenCV-DNN](https://img.shields.io/badge/OpenCV--DNN-5C3EE8.svg?logo=opencv&logoColor=white)](https://github.com/ultralytics/ultralytics/tree/main/examples/python/OpenCV-DNN)                                             | [OpenCV DNN](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)          | All tasks; grid and end-to-end outputs; no model metadata needed.    |
| [![Axelera](https://img.shields.io/badge/Axelera-Voyager%20SDK-0A2540.svg)](https://github.com/ultralytics/ultralytics/tree/main/examples/python/Axelera)                                                                     | [Axelera Voyager SDK](https://docs.ultralytics.com/integrations/axelera)  | YOLO pipelines on Axelera Metis devices.                             |
| [![Action-Recognition](https://img.shields.io/badge/Action%20Recognition-FFD21E.svg?logo=huggingface&logoColor=black)](https://github.com/ultralytics/ultralytics/tree/main/examples/python/Action-Recognition)               | [Hugging Face Transformers](https://huggingface.co/docs/transformers)     | Zero-shot action recognition on tracked people.                      |
| [![Interactive-Tracking](https://img.shields.io/badge/Interactive%20Tracking-042AFF.svg?logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics/tree/main/examples/python/Interactive-Tracking)         | [Ultralytics](https://docs.ultralytics.com/modes/track)                   | Click to pick a tracked object in a live window.                     |
| [![High-Resolution-Tiling](https://img.shields.io/badge/High%20Resolution%20Tiling-042AFF.svg?logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics/tree/main/examples/python/High-Resolution-Tiling) | [Ultralytics](https://docs.ultralytics.com/guides/high-resolution-tiling) | Run YOLO on large images and videos with the `tile` argument.        |
| [![Region-Counter](https://img.shields.io/badge/Region%20Counter-042AFF.svg?logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics/tree/main/examples/python/Region-Counter)                           | [Ultralytics](https://docs.ultralytics.com/guides/region-counting)        | Count objects in interactive, draggable video regions.               |

## ✅ How to Test

The inference examples follow the same flow: **export a model → install the requirements → run the script**. Install the [Ultralytics package](https://docs.ultralytics.com/quickstart) first (`pip install ultralytics`) so the `yolo export` command is available.

> [!NOTE]
> ONNXRuntime reads the task, class names, and input size from the model metadata, so the same script handles any model. **OpenCV-DNN** reads no metadata and infers the task from the output shapes, so semantic, depth, classify, OBB, and RT-DETR models need `--task` (and `--imgsz`).

### ONNXRuntime

```bash
# 1. Export any model and task: the task and class names are read from the metadata
yolo export model=yolo26n.pt format=onnx opset=12 # or -seg / -sem / -depth / -cls / -pose / -obb

# 2. Install
cd examples/python/ONNXRuntime && pip install -r requirements.txt

# 3. Run: task auto-detected, result written to the --out path
python main.py --model yolo26n.onnx --source bus.jpg
```

### OpenCV-DNN

```bash
# 1. Export yolo26n.onnx (grid by default, end-to-end with nms=False)
yolo export model=yolo26n.pt format=onnx opset=12

# 2. Install
cd examples/python/OpenCV-DNN && pip install -r requirements.txt

# 3. Run (use --task and --imgsz for semantic, depth, classify, OBB, and RT-DETR)
python main.py --model yolo26n.onnx --source bus.jpg
```

See each example's own `README.md` for full details and configuration options.

## 🤝 Contributing

Contributions are welcome! See the [Ultralytics Contributing Guide](https://docs.ultralytics.com/help/contributing) for details. If you find an issue with one of these examples, please open an issue or pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
