---
comments: true
description: Speed Estimation Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Speed Estimation, Object Tracking, Notebook, IPython Kernel, CLI, Python SDK
---

# Speed Estimation using Ultralytics YOLOv8 🚀

## What is Speed Estimation?

Speed estimation is the process of calculating the rate of movement of an object within a given context, often employed in computer vision applications. Using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) you can now calculate the speed of object using [object tracking](https://docs.ultralytics.com/modes/track/) alongside distance and time data, crucial for tasks like traffic and surveillance. The accuracy of speed estimation directly influences the efficiency and reliability of various applications, making it a key component in the advancement of intelligent systems and real-time decision-making processes.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rCggzXRRSRo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Speed Estimation using Ultralytics YOLOv8
</p>

## Advantages of Speed Estimation?

- **Efficient Traffic Control:** Accurate speed estimation aids in managing traffic flow, enhancing safety, and reducing congestion on roadways.
- **Precise Autonomous Navigation:** In autonomous systems like self-driving cars, reliable speed estimation ensures safe and accurate vehicle navigation.
- **Enhanced Surveillance Security:** Speed estimation in surveillance analytics helps identify unusual behaviors or potential threats, improving the effectiveness of security measures.

## Real World Applications

|                                                                     Transportation                                                                      |                                                                      Transportation                                                                       |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![Speed Estimation on Road using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/c8a0fd4a-d394-436d-8de3-d5b754755fc7) | ![Speed Estimation on Bridge using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/cee10e02-b268-4304-b73a-5b9cb42da669) |
|                                                    Speed Estimation on Road using Ultralytics YOLOv8                                                    |                                                    Speed Estimation on Bridge using Ultralytics YOLOv8                                                    |

!!! Example "Speed Estimation using YOLOv8 Example"

    === "Speed Estimation"

        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import speed_estimation
        import cv2

        model = YOLO("yolov8n.pt")
        names = model.model.names

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("speed_estimation.avi",
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps,
                                       (w, h))

        line_pts = [(0, 360), (1280, 360)]

        # Init speed-estimation obj
        speed_obj = speed_estimation.SpeedEstimator()
        speed_obj.set_args(reg_pts=line_pts,
                           names=names,
                           view_img=True)

        while cap.isOpened():

            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            tracks = model.track(im0, persist=True, show=False)

            im0 = speed_obj.estimate_speed(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        ```

???+ warning "Speed is Estimate"

    Speed will be an estimate and may not be completely accurate. Additionally, the estimation can vary depending on GPU speed.

### Optional Arguments `set_args`

| Name               | Type   | Default                    | Description                                       |
|--------------------|--------|----------------------------|---------------------------------------------------|
| `reg_pts`          | `list` | `[(20, 400), (1260, 400)]` | Points defining the Region Area                   |
| `names`            | `dict` | `None`                     | Classes names                                     |
| `view_img`         | `bool` | `False`                    | Display frames with counts                        |
| `line_thickness`   | `int`  | `2`                        | Increase bounding boxes thickness                 |
| `region_thickness` | `int`  | `5`                        | Thickness for object counter region or line       |
| `spdl_dist_thresh` | `int`  | `10`                       | Euclidean Distance threshold for speed check line |

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
