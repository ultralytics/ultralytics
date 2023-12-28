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

!!! Example "Workouts Monitoring Example"

    === "Workouts Monitoring"
        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import ai_gym
        import cv2

        model = YOLO("yolov8n-pose.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        gym_object = ai_gym.AIGym()  # init AI GYM module
        gym_object.set_args(line_thickness=2,
                            view_img=True,
                            pose_type="pushup",
                            kpts_to_check=[6, 8, 10])

        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
              print("Video frame is empty or video processing has been successfully completed.")
              break
            frame_count += 1
            results = model.predict(im0, verbose=False)
            im0 = gym_object.start_counting(im0, results, frame_count)

        cv2.destroyAllWindows()
        ```

    === "Workouts Monitoring with Save Output"
        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import ai_gym
        import cv2

        model = YOLO("yolov8n-pose.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        video_writer = cv2.VideoWriter("workouts.avi",
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        int(cap.get(5)),
                                        (int(cap.get(3)), int(cap.get(4))))

        gym_object = ai_gym.AIGym()  # init AI GYM module
        gym_object.set_args(line_thickness=2,
                            view_img=True,
                            pose_type="pushup",
                            kpts_to_check=[6, 8, 10])

        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
              print("Video frame is empty or video processing has been successfully completed.")
              break
            frame_count += 1
            results = model.predict(im0, verbose=False)
            im0 = gym_object.start_counting(im0, results, frame_count)
            video_writer.write(im0)

        cv2.destroyAllWindows()
        video_writer.release()
        ```

???+ tip "Support"

    "pushup", "pullup" and "abworkout" supported

### KeyPoints Map

![keyPoints Order Ultralytics YOLOv8 Pose](https://github.com/ultralytics/ultralytics/assets/62513924/f45d8315-b59f-47b7-b9c8-c61af1ce865b)

### Arguments `set_args`

| Name            | Type   | Default  | Description                                                                            |
|-----------------|--------|----------|----------------------------------------------------------------------------------------|
| kpts_to_check   | `list` | `None`   | List of three keypoints index, for counting specific workout, followed by keypoint Map |
| view_img        | `bool` | `False`  | Display the frame with counts                                                          |
| line_thickness  | `int`  | `2`      | Increase the thickness of count value                                                  |
| pose_type       | `str`  | `pushup` | Pose that need to be monitored, "pullup" and "abworkout" also supported                |
| pose_up_angle   | `int`  | `145`    | Pose Up Angle value                                                                    |
| pose_down_angle | `int`  | `90`     | Pose Down Angle value                                                                  |

### Arguments `model.predict`

| Name            | Type           | Default                | Description                                                                |
|-----------------|----------------|------------------------|----------------------------------------------------------------------------|
| `source`        | `str`          | `'ultralytics/assets'` | source directory for images or videos                                      |
| `conf`          | `float`        | `0.25`                 | object confidence threshold for detection                                  |
| `iou`           | `float`        | `0.7`                  | intersection over union (IoU) threshold for NMS                            |
| `imgsz`         | `int or tuple` | `640`                  | image size as scalar or (h, w) list, i.e. (640, 480)                       |
| `half`          | `bool`         | `False`                | use half precision (FP16)                                                  |
| `device`        | `None or str`  | `None`                 | device to run on, i.e. cuda device=0/1/2/3 or device=cpu                   |
| `max_det`       | `int`          | `300`                  | maximum number of detections per image                                     |
| `vid_stride`    | `bool`         | `False`                | video frame-rate stride                                                    |
| `stream_buffer` | `bool`         | `False`                | buffer all streaming frames (True) or return the most recent frame (False) |
| `visualize`     | `bool`         | `False`                | visualize model features                                                   |
| `augment`       | `bool`         | `False`                | apply image augmentation to prediction sources                             |
| `agnostic_nms`  | `bool`         | `False`                | class-agnostic NMS                                                         |
| `classes`       | `list[int]`    | `None`                 | filter results by class, i.e. classes=0, or classes=[0,2,3]                |
| `retina_masks`  | `bool`         | `False`                | use high-resolution segmentation masks                                     |
| `embed`         | `list[int]`    | `None`                 | return feature vectors/embeddings from given layers                        |
