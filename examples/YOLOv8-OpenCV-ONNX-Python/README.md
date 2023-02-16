# YOLOv8 - OpenCV

Implementation YOLOv8 on OpenCV using ONNX Format.

Just simply clone and run

```bash
pip install -r requirements.txt
python main.py
```

If you start from scratch:

```bash
pip install ultralytics
yolo export model=yolov8n.pt imgsz=640 format=onnx opset=12
```

_\*Make sure to include "opset=12"_
