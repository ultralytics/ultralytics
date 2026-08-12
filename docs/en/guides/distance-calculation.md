---
title: Distance Calculation with YOLO26
comments: true
description: Learn how to calculate distances between objects using Ultralytics YOLO26 for accurate spatial positioning and scene understanding.
keywords: Ultralytics, YOLO26, distance calculation, computer vision, object tracking, spatial positioning
---

# Distance Calculation using Ultralytics YOLO26

## What is Distance Calculation?

Distance calculation is the process of measuring the space between two detected objects within an image or video frame. In the case of [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics), the [bounding box](https://www.ultralytics.com/glossary/bounding-box) centroid is employed to calculate the distance for bounding boxes highlighted by the user.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Oe0vmsvnY74"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to estimate distance between detected objects with Ultralytics YOLO in Pixels 🚀
</p>

## Visuals

|                                         Distance Calculation using Ultralytics YOLO26                                          |
| :----------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics YOLO26 Distance Calculation](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/distance-calculation.avif) |

## Advantages of Distance Calculation

- **Localization [Precision](https://www.ultralytics.com/glossary/precision):** Enhances accurate spatial positioning in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks.
- **Size Estimation:** Allows estimation of object size for better contextual understanding.
- **Scene Understanding:** Improves 3D scene comprehension for better decision-making in applications like [autonomous vehicles](https://www.ultralytics.com/glossary/autonomous-vehicles) and surveillance systems.
- **Collision Avoidance:** Enables systems to detect potential collisions by monitoring distances between moving objects.
- **Spatial Analysis:** Facilitates analysis of object relationships and interactions within the monitored environment.

???+ tip "Distance Calculation"

    - Click any two bounding boxes with the left mouse button to calculate distance.
    - Use the right mouse button to delete all drawn points.
    - Left-click anywhere in the frame to add new points.

???+ warning "Distance is an estimate"

    Distance is an estimate and may not be fully accurate because it is calculated using 2D data,
    which lacks depth information.

## Calculate Distances with YOLO26

The `DistanceCalculation` solution tracks objects across frames and measures the Euclidean distance, in pixels, between the centroids of any two bounding boxes you select with the mouse. Run the example below, then left-click two objects to draw the connecting line and read the distance; right-click to clear your selection.

!!! example "Distance Calculation using Ultralytics YOLO"

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("distance_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize distance calculation object
        distancecalculator = solutions.DistanceCalculation(
            model="yolo26n.pt",  # path to the YOLO26 model file.
            show=True,  # display the output
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = distancecalculator(im0)

            print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `DistanceCalculation()` Arguments

Here's a table with the `DistanceCalculation` arguments:

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `model` | `str` | `None` | Path to an Ultralytics YOLO model file. |


You can also make use of various `track` arguments in the `DistanceCalculation` solution.

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `tracker` | `str` | `'tracktrack.yaml'` | Specifies the tracking algorithm to use. Built-in options: `botsort.yaml`, `bytetrack.yaml`, `ocsort.yaml`, `deepocsort.yaml`, `fasttrack.yaml`, `tracktrack.yaml`. |
| `conf` | `float` | `0.1` | Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives. |
| `iou` | `float` | `0.7` | Sets the [Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for filtering overlapping detections. |
| `classes` | `list` | `None` | Filters results by class index. For example, `classes=[0, 2, 3]` only tracks the specified classes. |
| `verbose` | `bool` | `True` | Controls the display of tracking results, providing a visual output of tracked objects. |
| `device` | `str` | `None` | Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution. |


Moreover, the following visualization arguments are available:

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `show` | `bool` | `False` | If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing. |
| `line_width` | `int or None` | `None` | Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity. |
| `show_conf` | `bool` | `True` | Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection. |
| `show_labels` | `bool` | `True` | Displays labels for each detection in the visual output. Provides immediate understanding of detected objects. |


## Implementation Details

The `DistanceCalculation` class works by tracking objects across video frames and calculating the Euclidean distance between the centroids of selected bounding boxes. When you click on two objects, the solution:

1. Extracts the centroids (center points) of the selected bounding boxes
2. Calculates the Euclidean distance between these centroids in pixels
3. Displays the distance on the frame with a connecting line between the objects

The implementation uses the `mouse_event_for_distance` method to handle mouse interactions, allowing users to select objects and clear selections as needed. The `process` method handles the frame-by-frame processing, tracking objects, and calculating distances.

## Applications

Distance calculation with YOLO26 has numerous practical applications:

- **Retail Analytics:** Measure customer proximity to products and analyze store layout effectiveness
- **Industrial Safety:** Monitor safe distances between workers and machinery
- **Traffic Management:** Analyze vehicle spacing and detect tailgating
- **Sports Analysis:** Calculate distances between players, the ball, and key field positions
- **Healthcare:** Ensure proper distancing in waiting areas and monitor patient movement
- **Robotics:** Enable robots to maintain appropriate distances from obstacles and people

## FAQ

### How do I calculate distances between objects using Ultralytics YOLO26?

To calculate distances between objects using [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics), you need to identify the bounding box centroids of the detected objects. This process involves initializing the `DistanceCalculation` class from Ultralytics' `solutions` module and using the model's tracking outputs to calculate the distances.

### What are the advantages of using distance calculation with Ultralytics YOLO26?

Using distance calculation with Ultralytics YOLO26 offers several advantages:

- **Localization Precision:** Provides accurate spatial positioning for objects.
- **Size Estimation:** Helps estimate physical sizes, contributing to better contextual understanding.
- **Scene Understanding:** Enhances 3D scene comprehension, aiding improved decision-making in applications like autonomous driving and surveillance.
- **Real-time Processing:** Performs calculations on-the-fly, making it suitable for live video analysis.
- **Integration Capabilities:** Works seamlessly with other YOLO26 solutions like [object tracking](../modes/track.md) and [speed estimation](speed-estimation.md).

### Can I perform distance calculation in real-time video streams with Ultralytics YOLO26?

Yes, you can perform distance calculation in real-time video streams with Ultralytics YOLO26. The process involves capturing video frames using [OpenCV](https://www.ultralytics.com/glossary/opencv), running YOLO26 [object detection](https://www.ultralytics.com/glossary/object-detection), and using the `DistanceCalculation` class to calculate distances between objects in successive frames. For a detailed implementation, see the [video stream example](#calculate-distances-with-yolo26).

### How do I delete points drawn during distance calculation using Ultralytics YOLO26?

To delete points drawn during distance calculation with Ultralytics YOLO26, you can use a right mouse click. This action will clear all the points you have drawn. For more details, refer to the note section under the [distance calculation example](#calculate-distances-with-yolo26).

### What are the key arguments for initializing the DistanceCalculation class in Ultralytics YOLO26?

The key arguments for initializing the `DistanceCalculation` class in Ultralytics YOLO26 include:

- `model`: Path to the YOLO26 model file.
- `tracker`: Tracking algorithm to use (default is 'botsort.yaml').
- `conf`: Confidence threshold for detections.
- `show`: Flag to display the output.

For an exhaustive list and default values, see the [arguments of DistanceCalculation](#distancecalculation-arguments).
