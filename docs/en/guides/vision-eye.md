---
comments: true
description: Discover VisionEye's object mapping and tracking powered by Ultralytics YOLOv8. Simulate human eye precision, track objects, and calculate distances effortlessly.
keywords: VisionEye, YOLOv8, Ultralytics, object mapping, object tracking, distance calculation, computer vision, AI, machine learning, Python, tutorial
---

# VisionEye View Object Mapping using Ultralytics YOLOv8 🚀

## What is VisionEye Object Mapping?

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) VisionEye offers the capability for computers to identify and pinpoint objects, simulating the observational [precision](https://www.ultralytics.com/glossary/precision) of the human eye. This functionality enables computers to discern and focus on specific objects, much like the way the human eye observes details from a particular viewpoint.

## Samples

|                                                                        VisionEye View                                                                        |                                                                         VisionEye View With Object Tracking                                                                         |                                                                 VisionEye View With Distance Calculation                                                                 |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![VisionEye View Object Mapping using Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/visioneye-view-object-mapping-yolov8.avif) | ![VisionEye View Object Mapping with Object Tracking using Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/visioneye-object-mapping-with-tracking.avif) | ![VisionEye View with Distance Calculation using Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/visioneye-distance-calculation-yolov8.avif) |
|                                                    VisionEye View Object Mapping using Ultralytics YOLOv8                                                    |                                                     VisionEye View Object Mapping with Object Tracking using Ultralytics YOLOv8                                                     |                                                    VisionEye View with Distance Calculation using Ultralytics YOLOv8                                                     |

!!! example "VisionEye Object Mapping using YOLOv8"

    === "VisionEye Object Mapping"

        ```python
        import cv2

        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator, colors

        model = YOLO("yolov8n.pt")
        names = model.model.names
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("visioneye-pinpoint.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        center_point = (-10, h)

        while True:
            ret, im0 = cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = model.predict(im0)
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(im0, line_width=2)

            for box, cls in zip(boxes, clss):
                annotator.box_label(box, label=names[int(cls)], color=colors(int(cls)))
                annotator.visioneye(box, center_point)

            out.write(im0)
            cv2.imshow("visioneye-pinpoint", im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        ```

    === "VisionEye Object Mapping with Object Tracking"

        ```python
        import cv2

        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator, colors

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("visioneye-pinpoint.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        center_point = (-10, h)

        while True:
            ret, im0 = cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            annotator = Annotator(im0, line_width=2)

            results = model.track(im0, persist=True)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    annotator.box_label(box, label=str(track_id), color=colors(int(track_id)))
                    annotator.visioneye(box, center_point)

            out.write(im0)
            cv2.imshow("visioneye-pinpoint", im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        ```

    === "VisionEye with Distance Calculation"

        ```python
        import math

        import cv2

        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator

        model = YOLO("yolov8s.pt")
        cap = cv2.VideoCapture("Path/to/video/file.mp4")

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("visioneye-distance-calculation.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        center_point = (0, h)
        pixel_per_meter = 10

        txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

        while True:
            ret, im0 = cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            annotator = Annotator(im0, line_width=2)

            results = model.track(im0, persist=True)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    annotator.box_label(box, label=str(track_id), color=bbox_clr)
                    annotator.visioneye(box, center_point)

                    x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)  # Bounding box centroid

                    distance = (math.sqrt((x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2)) / pixel_per_meter

                    text_size, _ = cv2.getTextSize(f"Distance: {distance:.2f} m", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    cv2.rectangle(im0, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), txt_background, -1)
                    cv2.putText(im0, f"Distance: {distance:.2f} m", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, txt_color, 3)

            out.write(im0)
            cv2.imshow("visioneye-distance-calculation", im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        ```

### `visioneye` Arguments

| Name        | Type    | Default          | Description                    |
| ----------- | ------- | ---------------- | ------------------------------ |
| `color`     | `tuple` | `(235, 219, 11)` | Line and object centroid color |
| `pin_color` | `tuple` | `(255, 0, 255)`  | VisionEye pinpoint color       |

## Note

For any inquiries, feel free to post your questions in the [Ultralytics Issue Section](https://github.com/ultralytics/ultralytics/issues/new/choose) or the discussion section mentioned below.

## FAQ

### How do I start using VisionEye Object Mapping with Ultralytics YOLOv8?

To start using VisionEye Object Mapping with Ultralytics YOLOv8, first, you'll need to install the Ultralytics YOLO package via pip. Then, you can use the sample code provided in the documentation to set up [object detection](https://www.ultralytics.com/glossary/object-detection) with VisionEye. Here's a simple example to get you started:

```python
import cv2

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("path/to/video/file.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    for result in results:
        # Perform custom logic with result
        pass

    cv2.imshow("visioneye", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

### What are the key features of VisionEye's object tracking capability using Ultralytics YOLOv8?

VisionEye's object tracking with Ultralytics YOLOv8 allows users to follow the movement of objects within a video frame. Key features include:

1. **Real-Time Object Tracking**: Keeps up with objects as they move.
2. **Object Identification**: Utilizes YOLOv8's powerful detection algorithms.
3. **Distance Calculation**: Calculates distances between objects and specified points.
4. **Annotation and Visualization**: Provides visual markers for tracked objects.

Here's a brief code snippet demonstrating tracking with VisionEye:

```python
import cv2

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("path/to/video/file.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)
    for result in results:
        # Annotate and visualize tracking
        pass

    cv2.imshow("visioneye-tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

For a comprehensive guide, visit the [VisionEye Object Mapping with Object Tracking](#samples).

### How can I calculate distances with VisionEye's YOLOv8 model?

Distance calculation with VisionEye and Ultralytics YOLOv8 involves determining the distance of detected objects from a specified point in the frame. It enhances spatial analysis capabilities, useful in applications such as autonomous driving and surveillance.

Here's a simplified example:

```python
import math

import cv2

from ultralytics import YOLO

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("path/to/video/file.mp4")
center_point = (0, 480)  # Example center point
pixel_per_meter = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)
    for result in results:
        # Calculate distance logic
        distances = [
            (math.sqrt((box[0] - center_point[0]) ** 2 + (box[1] - center_point[1]) ** 2)) / pixel_per_meter
            for box in results
        ]

    cv2.imshow("visioneye-distance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

For detailed instructions, refer to the [VisionEye with Distance Calculation](#samples).

### Why should I use Ultralytics YOLOv8 for object mapping and tracking?

Ultralytics YOLOv8 is renowned for its speed, [accuracy](https://www.ultralytics.com/glossary/accuracy), and ease of integration, making it a top choice for object mapping and tracking. Key advantages include:

1. **State-of-the-art Performance**: Delivers high accuracy in real-time object detection.
2. **Flexibility**: Supports various tasks such as detection, tracking, and distance calculation.
3. **Community and Support**: Extensive documentation and active GitHub community for troubleshooting and enhancements.
4. **Ease of Use**: Intuitive API simplifies complex tasks, allowing for rapid deployment and iteration.

For more information on applications and benefits, check out the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

### How can I integrate VisionEye with other [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) tools like Comet or ClearML?

Ultralytics YOLOv8 can integrate seamlessly with various machine learning tools like Comet and ClearML, enhancing experiment tracking, collaboration, and reproducibility. Follow the detailed guides on [how to use YOLOv5 with Comet](https://www.ultralytics.com/blog/how-to-use-yolov5-with-comet) and [integrate YOLOv8 with ClearML](https://docs.ultralytics.com/integrations/clearml/) to get started.

For further exploration and integration examples, check our [Ultralytics Integrations Guide](https://docs.ultralytics.com/integrations/).
