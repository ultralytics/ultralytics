---
comments: true
description: Learn how to estimate object speed using Ultralytics YOLOv8 for applications in traffic control, autonomous navigation, and surveillance.
keywords: Ultralytics YOLOv8, speed estimation, object tracking, computer vision, traffic control, autonomous navigation, surveillance, security
---

# Speed Estimation using Ultralytics YOLOv8 🚀

## What is Speed Estimation?

[Speed estimation](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects) is the process of calculating the rate of movement of an object within a given context, often employed in computer vision applications. Using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) you can now calculate the speed of object using [object tracking](../modes/track.md) alongside distance and time data, crucial for tasks like traffic and surveillance. The accuracy of speed estimation directly influences the efficiency and reliability of various applications, making it a key component in the advancement of intelligent systems and real-time decision-making processes.

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

!!! tip "Check Out Our Blog"

    For deeper insights into speed estimation, check out our blog post: [Ultralytics YOLOv8 for Speed Estimation in Computer Vision Projects](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects)

## Advantages of Speed Estimation?

- **Efficient Traffic Control:** Accurate speed estimation aids in managing traffic flow, enhancing safety, and reducing congestion on roadways.
- **Precise Autonomous Navigation:** In autonomous systems like self-driving cars, reliable speed estimation ensures safe and accurate vehicle navigation.
- **Enhanced Surveillance Security:** Speed estimation in surveillance analytics helps identify unusual behaviors or potential threats, improving the effectiveness of security measures.

## Real World Applications

|                                                                     Transportation                                                                      |                                                                      Transportation                                                                       |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Speed Estimation on Road using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/c8a0fd4a-d394-436d-8de3-d5b754755fc7) | ![Speed Estimation on Bridge using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/cee10e02-b268-4304-b73a-5b9cb42da669) |
|                                                    Speed Estimation on Road using Ultralytics YOLOv8                                                    |                                                    Speed Estimation on Bridge using Ultralytics YOLOv8                                                    |

!!! Example "Speed Estimation using YOLOv8 Example"

    === "Speed Estimation"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        names = model.model.names

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        line_pts = [(0, 360), (1280, 360)]

        # Init speed-estimation obj
        speed_obj = solutions.SpeedEstimator(
            reg_pts=line_pts,
            names=names,
            view_img=True,
        )

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

### Arguments `SpeedEstimator`

| Name               | Type   | Default                    | Description                                          |
| ------------------ | ------ | -------------------------- | ---------------------------------------------------- |
| `names`            | `dict` | `None`                     | Dictionary of class names.                           |
| `reg_pts`          | `list` | `[(20, 400), (1260, 400)]` | List of region points for speed estimation.          |
| `view_img`         | `bool` | `False`                    | Whether to display the image with annotations.       |
| `line_thickness`   | `int`  | `2`                        | Thickness of the lines for drawing boxes and tracks. |
| `region_thickness` | `int`  | `5`                        | Thickness of the region lines.                       |
| `spdl_dist_thresh` | `int`  | `10`                       | Distance threshold for speed calculation.            |

### Arguments `model.track`

| Name      | Type    | Default        | Description                                                 |
| --------- | ------- | -------------- | ----------------------------------------------------------- |
| `source`  | `im0`   | `None`         | source directory for images or videos                       |
| `persist` | `bool`  | `False`        | persisting tracks between frames                            |
| `tracker` | `str`   | `botsort.yaml` | Tracking method 'bytetrack' or 'botsort'                    |
| `conf`    | `float` | `0.3`          | Confidence Threshold                                        |
| `iou`     | `float` | `0.5`          | IOU Threshold                                               |
| `classes` | `list`  | `None`         | filter results by class, i.e. classes=0, or classes=[0,2,3] |
| `verbose` | `bool`  | `True`         | Display the object tracking results                         |

## FAQ

### How do I estimate object speed using Ultralytics YOLOv8?

Estimating object speed with Ultralytics YOLOv8 involves combining object detection and tracking techniques. First, you need to detect objects in each frame using the YOLOv8 model. Then, track these objects across frames to calculate their movement over time. Finally, use the distance traveled by the object between frames and the frame rate to estimate its speed.

**Example**:

```python
import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture("path/to/video/file.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize SpeedEstimator
speed_obj = solutions.SpeedEstimator(
    reg_pts=[(0, 360), (1280, 360)],
    names=names,
    view_img=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    tracks = model.track(im0, persist=True, show=False)
    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

For more details, refer to our [official blog post](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects).

### What are the benefits of using Ultralytics YOLOv8 for speed estimation in traffic management?

Using Ultralytics YOLOv8 for speed estimation offers significant advantages in traffic management:

- **Enhanced Safety**: Accurately estimate vehicle speeds to detect over-speeding and improve road safety.
- **Real-Time Monitoring**: Benefit from YOLOv8's real-time object detection capability to monitor traffic flow and congestion effectively.
- **Scalability**: Deploy the model on various hardware setups, from edge devices to servers, ensuring flexible and scalable solutions for large-scale implementations.

For more applications, see [advantages of speed estimation](#advantages-of-speed-estimation).

### Can YOLOv8 be integrated with other AI frameworks like TensorFlow or PyTorch?

Yes, YOLOv8 can be integrated with other AI frameworks like TensorFlow and PyTorch. Ultralytics provides support for exporting YOLOv8 models to various formats like ONNX, TensorRT, and CoreML, ensuring smooth interoperability with other ML frameworks.

To export a YOLOv8 model to ONNX format:

```bash
yolo export --weights yolov8n.pt --include onnx
```

Learn more about exporting models in our [guide on export](../modes/export.md).

### How accurate is the speed estimation using Ultralytics YOLOv8?

The accuracy of speed estimation using Ultralytics YOLOv8 depends on several factors, including the quality of the object tracking, the resolution and frame rate of the video, and environmental variables. While the speed estimator provides reliable estimates, it may not be 100% accurate due to variances in frame processing speed and object occlusion.

**Note**: Always consider margin of error and validate the estimates with ground truth data when possible.

For further accuracy improvement tips, check the [Arguments `SpeedEstimator` section](#arguments-speedestimator).

### Why choose Ultralytics YOLOv8 over other object detection models like TensorFlow Object Detection API?

Ultralytics YOLOv8 offers several advantages over other object detection models, such as the TensorFlow Object Detection API:

- **Real-Time Performance**: YOLOv8 is optimized for real-time detection, providing high speed and accuracy.
- **Ease of Use**: Designed with a user-friendly interface, YOLOv8 simplifies model training and deployment.
- **Versatility**: Supports multiple tasks, including object detection, segmentation, and pose estimation.
- **Community and Support**: YOLOv8 is backed by an active community and extensive documentation, ensuring developers have the resources they need.

For more information on the benefits of YOLOv8, explore our detailed [model page](../models/yolov8.md).
