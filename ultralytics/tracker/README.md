## Tracker

### Trackers

- [x] ByteTracker
- [x] BoT-SORT
- [ ] SMILEtrack

### Usage
python interface:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or a segmentation model .i.e yolov8n-seg.pt
model.track(
    source="video/streams",
    stream=True,
    tracker="botsort/bytetrack",
    tracker_cfg="your modified config file",
    ...,
)
```

cli:

```bash
yolo detect track source=... tracker=... tracker_cfg=...
yolo segment track source=... tracker=... tracker_cfg=...
```
