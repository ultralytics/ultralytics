---
comments: true
description: Learn how to use Ultralytics YOLOv8 to proactively detect illegal parking situations for applications such as traffic control, autonomous navigation, and surveillance.
keywords: Ultralytics YOLOv8, illegal parking, object tracking, computer vision, traffic control, autonomous navigation, surveillance, security
---

# Illegal Parking using Ultralytics YOLOv8 🚀

## What is Illegal Parking?

Illegal parking is the monitoring of abnormal parking of vehicles, which is usually used in computer applications. With [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/), you can now use the result data of [object tracking](../modes/track.md) to determine whether the vehicle is in a stopped state and how long it has been stopped, which is an important contribution to traffic safety management and urban vehicle management, especially for the rapid and active detection of vehicle breakdowns on highways, which can significantly reduce the probability of chain accidents.

<!-- TODO: Introduction Video
<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.bilibili.com/video/BV1V1HbeHERo/?vd_source=c0ebd7c851c28801019d136cdbbb653b"
    title="bilibili video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Illegal Parking using Ultralytics YOLOv8
</p>
-->

## Advantages of Illegal Parking?

- **Efficient Traffic Control:** Illegal parking monitoring is actively discovered by the system, which can respond to sudden traffic incidents more quickly and greatly reduce the incidence of chain accidents, enhancing safety, and reducing congestion on roadways.
- **Enhanced Surveillance Security:** Illegal estimation in surveillance analytics helps identify unusual behaviors or potential threats, improving the effectiveness of security measures.

## Real World Applications

<!-- TODO: Pictures of application scenarios
|                                                                             Transportation                                                                             |                                                                               Urban Management                                                                               |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![Illegal Parking on Highway using Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/speed-estimation-on-road-using-ultralytics-yolov8.avif) | ![Illegal Parking on urban road using Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/speed-estimation-on-bridge-using-ultralytics-yolov8.avif) |
|                                                           Illegal Parking on Highway using Ultralytics YOLOv8                                                            |                                                             Illegal Parking on urban road using Ultralytics YOLOv8                                                            |
-->

!!! Example "Illegal Parking using YOLOv8 Example"

    === "Illegal Parking"

        ```python
        import cv2

        from ultralytics import YOLO
        from ultralytics.solutions.illegal_parking import IllegalParking

        model = YOLO("yolov8n.pt")
        names = model.model.names

        cap = cv2.VideoCapture("path/to/video/video.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("illegal_parking.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Illegal parking monitoring areas
        polygon_pts = [
            [
                (10, 240),
                (1550, 10),
                (1800, 10),
                (1800, 960),
                (10, 960),
            ]
        ]

        # Init illegal-parking obj
        obj = IllegalParking(
            polygon_coords=polygon_pts,
            names=names,
            view_img=False,
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            tracks = model.track(im0, persist=True, show=False, verbose=False)

            im0 = obj.found_illegal_parking(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

???+ warning "Parking Timer depends on the processing speed "

    The parking time is correct only when the video processing speed is greater than or equal to the video generation speed.

### Arguments `IllegalParking`

| Name                   | Type    | Default    | Description                                                                  |
| ---------------------- | ------- | ---------- | ---------------------------------------------------------------------------- |
| `names`                | `dict`  | `required` | Dictionary of class names.                                                   |
| `polygon_coords`       | `list`  | `None`     | List of polygon coordinates for judging illegal parking.                     |
| `view_img`             | `bool`  | `False`    | Whether to display the image with annotations.                               |
| `line_thickness`       | `int`   | `2`        | Thickness of the lines for drawing boxes and tracks.                         |
| `polygon_thickness`    | `int`   | `5`        | Thickness of the polygon lines.                                              |
| `parking_threshold`    | `int`   | `50`       | Threshold for the number of frames to park.                                  |
| `object_iou_threshold` | `float` | `0.80`     | The intersection over union(iou) threshold of the object at different times. |

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

### How do I monitor object illegal parking using Ultralytics YOLOv8?

Monitoring illegally parked objects with Ultralytics YOLOv8 requires a combination of object detection and tracking techniques. First, you need to detect objects in each frame using the YOLOv8 model. Then, track these objects across frames to calculate their motion over time. Finally, use the intersection over union (IoU) of the objects between frames to determine whether the vehicle is stopped and start calculating how long it has been stopped.

**Example**:

```python
import cv2

from ultralytics import YOLO
from ultralytics.solutions.illegal_parking import IllegalParking

model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture("video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("illegal_parking.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init illegal-parking obj
obj = IllegalParking(names=names, view_img=False, parking_threshold=25)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False, verbose=False)

    im0 = obj.found_illegal_parking(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

<!-- TODO:
For more details, refer to our [official blog post](https...).
-->

### What are the benefits of using Ultralytics YOLOv8 for illegal parking in traffic management?

Using Ultralytics YOLOv8 for monitoring illegal parking offers significant advantages in traffic management:

- **Enhanced Safety**: Can quickly and accurately detect traffic anomalies, buy time for traffic management and reduce the risk of accidents.
- **Real-Time Monitoring**: Benefit from YOLOv8's real-time object detection capability to monitor traffic flow and congestion effectively.
- **Scalability**: Deploy the model on various hardware setups, from edge devices to servers, ensuring flexible and scalable solutions for large-scale implementations.

For more applications, see [advantages of illegal parking](#advantages-of-illegal-parking).

### Can YOLOv8 be integrated with other AI frameworks like TensorFlow or PyTorch?

Yes, YOLOv8 can be integrated with other AI frameworks like TensorFlow and PyTorch. Ultralytics provides support for exporting YOLOv8 models to various formats like ONNX, TensorRT, and CoreML, ensuring smooth interoperability with other ML frameworks.

To export a YOLOv8 model to ONNX format:

```bash
yolo export --weights yolov8n.pt --include onnx
```

Learn more about exporting models in our [guide on export](../modes/export.md).

### How accurate is the illegal parking using Ultralytics YOLOv8?

The accuracy of using Ultralytics YOLOv8 to monitor illegal parking depends on several factors, including the quality of object tracking, the resolution and frame rate of the video, and environmental variables. It is almost 100% accurate in detecting parking, but the calculation of parking duration depends on the speed of video processing, that is, the performance of the GPU.

### Why choose Ultralytics YOLOv8 over other object detection models like TensorFlow Object Detection API?

Ultralytics YOLOv8 offers several advantages over other object detection models, such as the TensorFlow Object Detection API:

- **Real-Time Performance**: YOLOv8 is optimized for real-time detection, providing high speed and accuracy.
- **Ease of Use**: Designed with a user-friendly interface, YOLOv8 simplifies model training and deployment.
- **Versatility**: Supports multiple tasks, including object detection, segmentation, and pose estimation.
- **Community and Support**: YOLOv8 is backed by an active community and extensive documentation, ensuring developers have the resources they need.

For more information on the benefits of YOLOv8, explore our detailed [model page](../models/yolov8.md).
