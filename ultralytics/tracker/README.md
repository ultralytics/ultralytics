## Tracker

### Trackers

- [x] ByteTracker
- [x] BoT-SORT

### Usage

python interface:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or a segmentation model .i.e yolov8n-seg.pt
model.track(
    source="video/streams",
    stream=True,
    tracker="botsort.yaml",  # or 'bytetrack.yaml'
    ...,
)
```

cli:

```bash
yolo detect track source=... tracker=...
yolo segment track source=... tracker=...
```

By default, trackers will use the configuration in `ultralytics/tracker/cfg`.
We also support using a modified tracker config file. Please refer to the tracker config files
in `ultralytics/tracker/cfg`.
