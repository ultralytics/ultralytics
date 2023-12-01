---
comments: true
description: Workouts Monitoring Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Pose Estimation, PushUps, PullUps, Ab workouts, Notebook, IPython Kernel, CLI, Python SDK
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

|                                                  Workouts Monitoring                                                   |                                                  Workouts Monitoring                                                   |
|:----------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------:|
| ![PushUps Counting](https://github.com/RizwanMunawar/ultralytics/assets/62513924/cf016a41-589f-420f-8a8c-2cc8174a16de) | ![PullUps Counting](https://github.com/RizwanMunawar/ultralytics/assets/62513924/cb20f316-fac2-4330-8445-dcf5ffebe329) |
|                                                    PushUps Counting                                                    |                                                    PullUps Counting                                                    |

## Example

```python
from ultralytics import YOLO
from ultralytics.solutions import ai_gym
import cv2

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture("path/to/video.mp4")

gym_object = ai_gym.AIGym()  # init AI GYM module
gym_object.set_args(line_thickness=2, view_img=True, pose_type="pushup", kpts_to_check=[6, 8, 10])

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

### KeyPoints Map

![keyPoints Order Ultralytics YOLOv8 Pose](https://github.com/RizwanMunawar/ultralytics/assets/62513924/520059af-f961-433b-b2fb-7fe8c4336ee5)

### Arguments `set_args`

| Name            | Type   | Default  | Description                                                                            |
|-----------------|--------|----------|----------------------------------------------------------------------------------------|
| kpts_to_check   | `list` | `None`   | List of three keypoints index, for counting specific workout, followed by keypoint Map |
| view_img        | `bool` | `False`  | Display the frame with counts                                                          |
| line_thickness  | `int`  | `2`      | Increase the thickness of count value                                                  |
| pose_type       | `str`  | `pushup` | Pose that need to be monitored, "pullup" and "abworkout" also supported                |
| pose_up_angle   | `int`  | `145`    | Pose Up Angle value                                                                    |
| pose_down_angle | `int`  | `90`     | Pose Down Angle value                                                                  |
