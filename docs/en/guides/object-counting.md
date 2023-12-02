---
comments: true
description: Object Counting Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Object Counting, Object Tracking, Notebook, IPython Kernel, CLI, Python SDK
---

# Object Counting using Ultralytics YOLOv8 🚀

## What is Object Counting?

Object counting with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) involves accurate identification and counting of specific objects in videos and camera streams. YOLOv8 excels in real-time applications, providing efficient and precise object counting for various scenarios like crowd analysis and surveillance, thanks to its state-of-the-art algorithms and deep learning capabilities.

## Advantages of Object Counting?

- **Resource Optimization:** Object counting facilitates efficient resource management by providing accurate counts, and optimizing resource allocation in applications like inventory management.
- **Enhanced Security:** Object counting enhances security and surveillance by accurately tracking and counting entities, aiding in proactive threat detection.
- **Informed Decision-Making:** Object counting offers valuable insights for decision-making, optimizing processes in retail, traffic management, and various other domains.

## Real World Applications

|                                                                           Logistics                                                                           |                                                                     Aquaculture                                                                     |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![Conveyor Belt Packets Counting Using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/70e2d106-510c-4c6c-a57a-d34a765aa757) | ![Fish Counting in Sea using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/c60d047b-3837-435f-8d29-bb9fc95d2191) |
|                                                    Conveyor Belt Packets Counting Using Ultralytics YOLOv8                                                    |                                                    Fish Counting in Sea using Ultralytics YOLOv8                                                    |

## Example

```python
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("path/to/video/file.mp4")

counter = object_counter.ObjectCounter()  # Init Object Counter
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
counter.set_args(view_img=True, reg_pts=region_points,
                 classes_names=model.names, draw_tracks=True)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        exit(0)
    tracks = model.track(frame, persist=True, show=False)
    counter.start_counting(frame, tracks)
```

???+ tip "Region is Movable"

    You can move the region anywhere in the frame by clicking on its edges

### Optional Arguments `set_args`

| Name            | Type    | Default                                          | Description                           |
|-----------------|---------|--------------------------------------------------|---------------------------------------|
| view_img        | `bool`  | `False`                                          | Display the frame with counts         |
| line_thickness  | `int`   | `2`                                              | Increase the thickness of count value |
| reg_pts         | `list`  | `(20, 400), (1080, 404), (1080, 360), (20, 360)` | Region Area Points                    |
| classes_names   | `dict`  | `model.model.names`                              | Classes Names Dict                    |
| region_color    | `tuple` | `(0, 255, 0)`                                    | Region Area Color                     |
| track_thickness | `int`   | `2`                                              | Tracking line thickness               |
