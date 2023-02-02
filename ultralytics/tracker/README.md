## Tracker

### Trackers
- [X] ByteTracker
- [X] BoT-SORT
- [ ] SMILEtrack

### Coming soon
- [ ] cli support
- [ ] save/show videos and write results like `model.predict()`
- [ ] add tracker related config .i.e `match_thres`, `track_type`

### Usage
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.track(source="test.mp4", stream=True)
```
