---
comments: true
description: Workouts Monitoring Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Pose Estimation, Pushups, Pull ups, Ab workouts, Notebook, IPython Kernel, CLI, Python SDK
---

# Workouts Monitoring using Ultralytics YOLOv8 🚀 
Monitoring workouts through pose estimation with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) enhances exercise assessment by accurately tracking key body landmarks and joints in real-time. This technology provides instant feedback on exercise form, tracks workout routines, and measures performance metrics, optimizing training sessions for users and trainers alike.

## Advantages of Workouts Monitoring?

- **Optimized Performance:** Tailoring workouts based on monitoring data for better results.
- **Goal Achievement:** Track and adjust fitness goals for measurable progress.
- **Personalization:** Customized workout plans based on individual data for effectiveness.
- **Health Awareness:** Early detection of patterns indicating health issues or overtraining.
- **Informed Decisions:** Data-driven decisions for adjusting routines and setting realistic goals.

## Real World Applications


## Example
```python
from ultralytics import YOLO
from ultralytics.solutions import ai_gym
import cv2

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture("path/to/video.mp4")

gym_object = ai_gym.Aigym()  # init AI GYM module
gym_object.set_args(line_thickness=2, view_img=True, pose_type="pushup")

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success: exit(0)
    frame_count += 1
    results = model.predict(frame, verbose=False)
    gym_object.start_counting(frame, results, frame_count)
```

???+ tip "Support"

    "pushup", "pullup" and "abworkout" supported


### Optional Arguments `set_args` 
| Name            | Type   | Default  | Description                                                             |
|-----------------|--------|----------|-------------------------------------------------------------------------|
| view_img        | `bool` | `False`  | Display the frame with counts                                           |
| line_thickness  | `int`  | `2`      | Increase the thickness of count value                                   |
| pose_type       | `str`  | `pushup` | Pose that need to be monitored, "pullup" and "abworkout" also supported |
| pose_up_angle   | `int`  | `145`    | Pose Up Angle value                                                     |
| pose_down_angle | `int`  | `90`     | Pose Down Angle value                                                   |
