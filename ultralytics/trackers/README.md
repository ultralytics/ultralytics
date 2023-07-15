# Tracker

## Supported Trackers

- [x] ByteTracker
- [x] BoT-SORT

## Usage

### python interface:

You can use the Python interface to track objects using the YOLO model.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or a segmentation model .i.e yolov8n-seg.pt
model.track(
    source="video/streams",
    stream=True,
    tracker="botsort.yaml",  # or 'bytetrack.yaml'
    show=True,
)
```

You can get the IDs of the tracked objects using the following code:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

for result in model.track(source="video.mp4"):
    print(
        result.boxes.id.cpu().numpy().astype(int)
    )  # this will print the IDs of the tracked objects in the frame
```

If you want to use the tracker with a folder of images or when you loop on the video frames, you should use the `persist` parameter to tell the model that these frames are related to each other so the IDs will be fixed for the same objects. Otherwise, the IDs will be different in each frame because in each loop, the model creates a new object for tracking, but the `persist` parameter makes it use the same object for tracking.

```python
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("video.mp4")
model = YOLO("yolov8n.pt")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = results[0].boxes.id.cpu().numpy().astype(int)
    for box, id in zip(boxes, ids):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Id {id}",
            (box[0], box[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```

## Change tracker parameters

You can change the tracker parameters by eding the `tracker.yaml` file which is located in the ultralytics/cfg/trackers folder.

## Command Line Interface (CLI)

You can also use the command line interface to track objects using the YOLO model.

```bash
yolo detect track source=... tracker=...
yolo segment track source=... tracker=...
yolo pose track source=... tracker=...
```

By default, trackers will use the configuration in `ultralytics/cfg/trackers`.
We also support using a modified tracker config file. Please refer to the tracker config files
in `ultralytics/cfg/trackers`.<br>
