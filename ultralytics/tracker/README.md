## Tracker

### Trackers
- [X] ByteTracker
- [X] BoT-SORT
- [ ] SMILEtrack

### Usage
settings:
```yaml
# Tracker settings ------------------------------------------------------------------------------------------------------
tracker: "botsort"  # tracker type, ['botsort', 'bytetrack']
track_high_thresh: 0.5  # threshold for the first association
track_low_thresh: 0.1   # threshold for the second association
new_track_thresh: 0.6   # threshold for init new track if the detection does not match any tracks
track_buffer: 30        # buffer to calculate the time when to remove tracks
match_thresh: 0.8       # threshold for matching tracks
# min_box_area: 10      # threshold for min box areas(for tracker evaluation, not used for now)
# mot20: False          # for tracker evaluation(not used for now)

# botsort only settings
cmc_method: "sparseOptFlow"  # method of global motion compensation
# ReID model related thresh(have not suppported yet)
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
```

python interface:
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt") # or a segmentation model .i.e yolov8n-seg.pt
model.track(source="video/streams", stream=True, tracker="botsort/bytetrack", track_high_thresh=0.5, ...)
```

cli:
```bash
yolo detect track source=... tracker=... track_high_thresh=...
yolo segment track source=... tracker=... track_high_thresh=...
```

