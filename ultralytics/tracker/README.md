## Tracker

### Trackers
- [X] ByteTracker
- [X] BoT-SORT
- [ ] SMILEtrack

### Usage
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.track(source="test.mp4", stream=True)
```

### Coming soon
- [ ] cli support
- [X] save/show videos and write results like `model.predict()`
- [ ] add tracker related config .i.e `match_thres`, `track_type`

