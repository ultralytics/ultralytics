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
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Oe0vmsvnY74"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to estimate distance between detected objects with Ultralytics YOLO in Pixels 🚀
</p>

## Visuals

|                                         Distance Calculation using Ultralytics YOLO11                                         |
| :---------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics YOLO11 Distance Calculation](https://github.com/ultralytics/docs/releases/download/0/distance-calculation.avif) |

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
            model="yolo11n.pt",  # path to the YOLO11 model file.
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

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model"]) }}

You can also make use of various `track` arguments in the `DistanceCalculation` solution.

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Moreover, the following visualization arguments are available:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## Implementation Details

The `DistanceCalculation` class works by tracking objects across video frames and calculating the Euclidean distance between the centroids of selected bounding boxes. When you click on two objects, the solution:

1. Extracts the centroids (center points) of the selected bounding boxes
2. Calculates the Euclidean distance between these centroids in pixels
3. Displays the distance on the frame with a connecting line between the objects

The implementation uses the `mouse_event_for_distance` method to handle mouse interactions, allowing users to select objects and clear selections as needed. The `process` method handles the frame-by-frame processing, tracking objects, and calculating distances.

## Applications

Distance calculation with YOLO11 has numerous practical applications:

- **Retail Analytics:** Measure customer proximity to products and analyze store layout effectiveness
- **Industrial Safety:** Monitor safe distances between workers and machinery
- **Traffic Management:** Analyze vehicle spacing and detect tailgating
- **Sports Analysis:** Calculate distances between players, the ball, and key field positions
- **Healthcare:** Ensure proper distancing in waiting areas and monitor patient movement
- **Robotics:** Enable robots to maintain appropriate distances from obstacles and people

## FAQ

### How do I calculate distances between objects using Ultralytics YOLO11?

To calculate distances between objects using [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), you need to identify the bounding box centroids of the detected objects. This process involves initializing the `DistanceCalculation` class from Ultralytics' `solutions` module and using the model's tracking outputs to calculate the distances.

### What are the advantages of using distance calculation with Ultralytics YOLO11?

Using distance calculation with Ultralytics YOLO11 offers several advantages:

- **Localization Precision:** Provides accurate spatial positioning for objects.
- **Size Estimation:** Helps estimate physical sizes, contributing to better contextual understanding.
- **Scene Understanding:** Enhances 3D scene comprehension, aiding improved decision-making in applications like autonomous driving and surveillance.
- **Real-time Processing:** Performs calculations on-the-fly, making it suitable for live video analysis.
- **Integration Capabilities:** Works seamlessly with other YOLO11 solutions like [object tracking](../modes/track.md) and [speed estimation](speed-estimation.md).

### Can I perform distance calculation in real-time video streams with Ultralytics YOLO11?

Yes, you can perform distance calculation in real-time video streams with Ultralytics YOLO11. The process involves capturing video frames using [OpenCV](https://www.ultralytics.com/glossary/opencv), running YOLO11 [object detection](https://www.ultralytics.com/glossary/object-detection), and using the `DistanceCalculation` class to calculate distances between objects in successive frames. For a detailed implementation, see the [video stream example](#distance-calculation-using-ultralytics-yolo11).

### How do I delete points drawn during distance calculation using Ultralytics YOLO11?

To delete points drawn during distance calculation with Ultralytics YOLO11, you can use a right mouse click. This action will clear all the points you have drawn. For more details, refer to the note section under the [distance calculation example](#distance-calculation-using-ultralytics-yolo11).

### What are the key arguments for initializing the DistanceCalculation class in Ultralytics YOLO11?

The key arguments for initializing the `DistanceCalculation` class in Ultralytics YOLO11 include:

- `model`: Path to the YOLO11 model file.
- `tracker`: Tracking algorithm to use (default is 'botsort.yaml').
- `conf`: Confidence threshold for detections.
- `show`: Flag to display the output.

For an exhaustive list and default values, see the [arguments of DistanceCalculation](#distancecalculation-arguments).
