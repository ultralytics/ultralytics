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
yolo task=detect    mode=train  model=s.yaml ...
          classify       infer        s-cls.yaml
          segment        val          s-seg.yaml
```
### 2. Python SDK
To use pythonic interface of Ultralytics YOLO model
```python
import ultralytics
from ultralytics import YOLO

model = YOLO()
model.new("s-seg.yaml") # automatically detects task type
model.load("s-seg.pt") # load checkpoint
model.train(data="coco128-segments", epochs=1, lr0=0.01, ...)
```
If you're looking to modify YOLO for R&D or to build on top of it, refer to [Using Trainer]() Guide on our docs.
