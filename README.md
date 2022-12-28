[![Ultralytics CI](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg)](https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml)

### Install

```bash
pip install ultralytics
```
Development
```
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e .
```

## Usage
### 1. CLI
To simply use the latest Ultralytics YOLO models
```bash
yolo task=detect    mode=train     model=yolov8n.yaml ...
          classify       predict         yolov8n-cls.yaml
          segment        val             yolov8n-seg.yaml
```
### 2. Python SDK
To use pythonic interface of Ultralytics YOLO model
```python
from ultralytics import YOLO

model = YOLO.new('yolov8n.yaml')  # create a new model from scratch
model = YOLO.load('yolov8n.pt')  # load a pretrained model (recommended for best training results)

results = model.train(data='coco128.yaml', epochs=100, imgsz=640, ...)
results = model.val()
results = model.predict(source='bus.jpg')
success = model.export(format='onnx')
```
If you're looking to modify YOLO for R&D or to build on top of it, refer to [Using Trainer]() Guide on our docs.
