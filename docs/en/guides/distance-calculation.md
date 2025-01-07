---
comments: true
description: Learn how to calculate distances between objects using Ultralytics YOLO11 for accurate spatial positioning and scene understanding.
keywords: Ultralytics, YOLO11, distance calculation, computer vision, object tracking, spatial positioning
---

# Distance Calculation using Ultralytics YOLO11

## What is Distance Calculation?

Measuring the gap between two objects is known as distance calculation within a specified space. In the case of [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), the [bounding box](https://www.ultralytics.com/glossary/bounding-box) centroid is employed to calculate the distance for bounding boxes highlighted by the user.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LE8am1QoVn4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Distance Calculation using Ultralytics YOLO11
</p>

## Visuals

|                                         Distance Calculation using Ultralytics YOLO11                                         |
| :---------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics YOLO11 Distance Calculation](https://github.com/ultralytics/docs/releases/download/0/distance-calculation.avif) |

## Advantages of Distance Calculation?

- **Localization [Precision](https://www.ultralytics.com/glossary/precision):** Enhances accurate spatial positioning in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.
- **Size Estimation:** Allows estimation of object size for better contextual understanding.

???+ tip "Distance Calculation"

    - Click on any two bounding boxes with Left Mouse click for distance calculation

!!! example "Distance Calculation using YOLO11 Example"

    === "Video Stream"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init distance-calculation obj
        distance = solutions.DistanceCalculation(model="yolo11n.pt", show=True)

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            im0 = distance.calculate(im0)
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

| `Name`       | `Type` | `Default` | Description                                          |
| ------------ | ------ | --------- | ---------------------------------------------------- |
| `model`      | `str`  | `None`    | Path to Ultralytics YOLO Model File                  |
| `line_width` | `int`  | `2`       | Line thickness for bounding boxes.                   |
| `show`       | `bool` | `False`   | Flag to control whether to display the video stream. |

### Arguments `model.track`

{% include "macros/track-args.md" %}

## FAQ

### How do I calculate distances between objects using Ultralytics YOLO11?

To calculate distances between objects using [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), you need to identify the bounding box centroids of the detected objects. This process involves initializing the `DistanceCalculation` class from Ultralytics' `solutions` module and using the model's tracking outputs to calculate the distances. You can refer to the implementation in the [distance calculation example](#distance-calculation-using-ultralytics-yolo11).

### What are the advantages of using distance calculation with Ultralytics YOLO11?

Using distance calculation with Ultralytics YOLO11 offers several advantages:

- **Localization Precision:** Provides accurate spatial positioning for objects.
- **Size Estimation:** Helps estimate physical sizes, contributing to better contextual understanding.
- **Scene Understanding:** Enhances 3D scene comprehension, aiding improved decision-making in applications like autonomous driving and surveillance.

### Can I perform distance calculation in real-time video streams with Ultralytics YOLO11?

Yes, you can perform distance calculation in real-time video streams with Ultralytics YOLO11. The process involves capturing video frames using [OpenCV](https://www.ultralytics.com/glossary/opencv), running YOLO11 [object detection](https://www.ultralytics.com/glossary/object-detection), and using the `DistanceCalculation` class to calculate distances between objects in successive frames. For a detailed implementation, see the [video stream example](#distance-calculation-using-ultralytics-yolo11).

### How do I delete points drawn during distance calculation using Ultralytics YOLO11?

To delete points drawn during distance calculation with Ultralytics YOLO11, you can use a right mouse click. This action will clear all the points you have drawn. For more details, refer to the note section under the [distance calculation example](#distance-calculation-using-ultralytics-yolo11).

### What are the key arguments for initializing the DistanceCalculation class in Ultralytics YOLO11?

The key arguments for initializing the `DistanceCalculation` class in Ultralytics YOLO11 include:

- `model`: Model file path.
- `show`: Flag to indicate if the video stream should be displayed.
- `line_width`: Thickness of bounding box and the lines drawn on the image.

For an exhaustive list and default values, see the [arguments of DistanceCalculation](#arguments-distancecalculation).
