---
comments: true
description: Workouts Monitoring Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Pose Estimation, PushUps, PullUps, Ab workouts, Notebook, IPython Kernel, CLI, Python SDK
---

# Workouts Monitoring using Ultralytics YOLOv8 🚀

Monitoring workouts through pose estimation with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) enhances exercise assessment by accurately tracking key body landmarks and joints in real-time. This technology provides instant feedback on exercise form, tracks workout routines, and measures performance metrics, optimizing training sessions for users and trainers alike.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LGGxqLZtvuw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Workouts Monitoring using Ultralytics YOLOv8 | Pushups, Pullups, Ab Workouts
</p>

## Advantages of Workouts Monitoring?

- **Optimized Performance:** Tailoring workouts based on monitoring data for better results.
- **Goal Achievement:** Track and adjust fitness goals for measurable progress.
- **Personalization:** Customized workout plans based on individual data for effectiveness.
- **Health Awareness:** Early detection of patterns indicating health issues or over-training.
- **Informed Decisions:** Data-driven decisions for adjusting routines and setting realistic goals.

## Real World Applications

|                                                  Workouts Monitoring                                                   |                                                  Workouts Monitoring                                                   |
|:----------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------:|
| ![PushUps Counting](https://github.com/RizwanMunawar/ultralytics/assets/62513924/cf016a41-589f-420f-8a8c-2cc8174a16de) | ![PullUps Counting](https://github.com/RizwanMunawar/ultralytics/assets/62513924/cb20f316-fac2-4330-8445-dcf5ffebe329) |
|                                                    PushUps Counting                                                    |                                                    PullUps Counting                                                    |

!!! Example "Workouts Monitoring Example"

    === "Workouts Monitoring"

        ```python
        import cv2
        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n-pose.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        gym_object = solutions.AIGym(
            line_thickness=2,
            view_img=True,
            pose_type="pushup",
            kpts_to_check=[6, 8, 10],
        )

        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            frame_count += 1
            results = model.track(im0, verbose=False)  # Tracking recommended
            # results = model.predict(im0)  # Prediction also supported
            im0 = gym_object.start_counting(im0, results, frame_count)

        cv2.destroyAllWindows()
        ```

    === "Workouts Monitoring with Save Output"

        ```python
        import cv2
        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n-pose.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        video_writer = cv2.VideoWriter("workouts.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        gym_object = solutions.AIGym(
            line_thickness=2,
            view_img=True,
            pose_type="pushup",
            kpts_to_check=[6, 8, 10],
        )

        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            frame_count += 1
            results = model.track(im0, verbose=False)  # Tracking recommended
            # results = model.predict(im0)  # Prediction also supported
            im0 = gym_object.start_counting(im0, results, frame_count)
            video_writer.write(im0)

        cv2.destroyAllWindows()
        video_writer.release()
        ```

???+ tip "Support"

    "pushup", "pullup" and "abworkout" supported

### KeyPoints Map

![keyPoints Order Ultralytics YOLOv8 Pose](https://github.com/ultralytics/ultralytics/assets/62513924/f45d8315-b59f-47b7-b9c8-c61af1ce865b)

### Arguments `AIGym`

| Name              | Type    | Default  | Description                                                                            |
|-------------------|---------|----------|----------------------------------------------------------------------------------------|
| `kpts_to_check`   | `list`  | `None`   | List of three keypoints index, for counting specific workout, followed by keypoint Map |
| `line_thickness`  | `int`   | `2`      | Thickness of the lines drawn.                                                          |
| `view_img`        | `bool`  | `False`  | Flag to display the image.                                                             |
| `pose_up_angle`   | `float` | `145.0`  | Angle threshold for the 'up' pose.                                                     |
| `pose_down_angle` | `float` | `90.0`   | Angle threshold for the 'down' pose.                                                   |
| `pose_type`       | `str`   | `pullup` | Type of pose to detect (`'pullup`', `pushup`, `abworkout`, `squat`).                   |

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

### Arguments `model.track`

| Name      | Type    | Default        | Description                                                 |
|-----------|---------|----------------|-------------------------------------------------------------|
| `source`  | `im0`   | `None`         | source directory for images or videos                       |
| `persist` | `bool`  | `False`        | persisting tracks between frames                            |
| `tracker` | `str`   | `botsort.yaml` | Tracking method 'bytetrack' or 'botsort'                    |
| `conf`    | `float` | `0.3`          | Confidence Threshold                                        |
| `iou`     | `float` | `0.5`          | IOU Threshold                                               |
| `classes` | `list`  | `None`         | filter results by class, i.e. classes=0, or classes=[0,2,3] |
| `verbose` | `bool`  | `True`         | Display the object tracking results                         |
