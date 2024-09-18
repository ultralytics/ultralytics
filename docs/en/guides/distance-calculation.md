---
comments: true
description: Learn how to calculate distances between objects using Ultralytics YOLOv8 for accurate spatial positioning and scene understanding.
keywords: Ultralytics, YOLOv8, distance calculation, computer vision, object tracking, spatial positioning
---

# Distance Calculation using Ultralytics YOLOv8

## What is Distance Calculation?

Measuring the gap between two objects is known as distance calculation within a specified space. In the case of [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), the bounding box centroid is employed to calculate the distance for bounding boxes highlighted by the user.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LE8am1QoVn4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Distance Calculation using Ultralytics YOLOv8
</p>

## Visuals

|                                         Distance Calculation using Ultralytics YOLOv8                                         |
| :---------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics YOLOv8 Distance Calculation](https://github.com/ultralytics/docs/releases/download/0/distance-calculation.avif) |

## Advantages of Distance Calculation?

- **Localization Precision:** Enhances accurate spatial positioning in computer vision tasks.
- **Size Estimation:** Allows estimation of object size for better contextual understanding.

???+ tip "Distance Calculation"

    - Click on any two bounding boxes with Left Mouse click for distance calculation

!!! example "Distance Calculation using YOLOv8 Example"

    === "Video Stream"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        names = model.model.names

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init distance-calculation obj
        dist_obj = solutions.DistanceCalculation(names=names, view_img=True)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            tracks = model.track(im0, persist=True, show=False)
            im0 = dist_obj.start_process(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

???+ note

    - Mouse Right Click will delete all drawn points
    - Mouse Left Click can be used to draw points

???+ warning "Distance is Estimate"

        Distance will be an estimate and may not be fully accurate, as it is calculated using 2-dimensional data, which lacks information about the object's depth.

### Arguments `DistanceCalculation()`

| `Name`           | `Type`  | `Default`       | Description                                               |
| ---------------- | ------- | --------------- | --------------------------------------------------------- |
| `names`          | `dict`  | `None`          | Dictionary of classes names.                              |
| `view_img`       | `bool`  | `False`         | Flag to indicate if the video stream should be displayed. |
| `line_thickness` | `int`   | `2`             | Thickness of the lines drawn on the image.                |
| `line_color`     | `tuple` | `(255, 255, 0)` | Color of the lines drawn on the image (BGR format).       |
| `centroid_color` | `tuple` | `(255, 0, 255)` | Color of the centroids drawn (BGR format).                |

### Arguments `model.track`

{% include "macros/track-args.md" %}

## FAQ

### How do I calculate distances between objects using Ultralytics YOLOv8?

To calculate distances between objects using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), you need to identify the bounding box centroids of the detected objects. This process involves initializing the `DistanceCalculation` class from Ultralytics' `solutions` module and using the model's tracking outputs to calculate the distances. You can refer to the implementation in the [distance calculation example](#distance-calculation-using-ultralytics-yolov8).

### What are the advantages of using distance calculation with Ultralytics YOLOv8?

Using distance calculation with Ultralytics YOLOv8 offers several advantages:

- **Localization Precision:** Provides accurate spatial positioning for objects.
- **Size Estimation:** Helps estimate physical sizes, contributing to better contextual understanding.
- **Scene Understanding:** Enhances 3D scene comprehension, aiding improved decision-making in applications like autonomous driving and surveillance.

### Can I perform distance calculation in real-time video streams with Ultralytics YOLOv8?

Yes, you can perform distance calculation in real-time video streams with Ultralytics YOLOv8. The process involves capturing video frames using OpenCV, running YOLOv8 object detection, and using the `DistanceCalculation` class to calculate distances between objects in successive frames. For a detailed implementation, see the [video stream example](#distance-calculation-using-ultralytics-yolov8).

### How do I delete points drawn during distance calculation using Ultralytics YOLOv8?

To delete points drawn during distance calculation with Ultralytics YOLOv8, you can use a right mouse click. This action will clear all the points you have drawn. For more details, refer to the note section under the [distance calculation example](#distance-calculation-using-ultralytics-yolov8).

### What are the key arguments for initializing the DistanceCalculation class in Ultralytics YOLOv8?

The key arguments for initializing the `DistanceCalculation` class in Ultralytics YOLOv8 include:

- `names`: Dictionary mapping class indices to class names.
- `view_img`: Flag to indicate if the video stream should be displayed.
- `line_thickness`: Thickness of the lines drawn on the image.
- `line_color`: Color of the lines drawn on the image (BGR format).
- `centroid_color`: Color of the centroids (BGR format).

For an exhaustive list and default values, see the [arguments of DistanceCalculation](#arguments-distancecalculation).
