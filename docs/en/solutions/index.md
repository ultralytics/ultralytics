---
title: YOLO26 Computer Vision Solutions
comments: true
description: Ultralytics Solutions use YOLO26 for object counting, heatmaps, speed estimation, queue and parking management, object blurring, security alarms, and more.
keywords: Ultralytics, YOLO26, object counting, object blurring, security systems, AI solutions, real-time analysis, computer vision applications
---

# Ultralytics Solutions: Harness YOLO26 to Solve Real-World Problems

Ultralytics Solutions are ready-to-use applications of Ultralytics YOLO models for common [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) problems such as [object counting, heatmaps, speed estimation](https://academy.ultralytics.com/courses/yolo-in-production/counting-heatmaps-and-speed), queue management, and object blurring. Most solutions wrap detection, [tracking](../modes/track.md), or pose estimation in a configurable class that you can run from Python or the `yolo solutions` CLI.

![Ultralytics Solutions Thumbnail](https://cdn.ul.run/i/af2a06dd04b546ae345e818580d16e7e.avif)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/bjkt5OE_ANA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Ultralytics Solutions from the Command Line (CLI) | Ultralytics YOLO26 🚀
</p>

## Solutions

Each solution below has a dedicated guide:

- [Analytics](../guides/analytics.md): Generate line, bar, pie, and area charts from YOLO26 detections to analyze trends over time.
- [Distance Calculation](../guides/distance-calculation.md): Calculate distances between objects using [bounding box](https://www.ultralytics.com/glossary/bounding-box) centroids in YOLO26 for spatial analysis.
- [Heatmaps](../guides/heatmaps.md): Accumulate YOLO26 detections into a heatmap overlay that shows where objects appear or move most often.
- [Instance Segmentation with Object Tracking](../guides/instance-segmentation-and-tracking.md): Combine [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) and object tracking with YOLO26 to follow object outlines across video frames.
- [Live Inference with Streamlit](../guides/streamlit-live-inference.md): Run real-time [object detection](https://www.ultralytics.com/glossary/object-detection) with YOLO26 in your web browser through a Streamlit interface.
- [Object Blurring](../guides/object-blurring.md): Apply object blurring using YOLO26 to protect privacy in image and video processing.
- [Object Counting](../guides/object-counting.md): Count objects that cross a line or enter a region in videos and live streams with YOLO26.
- [Object Counting in Regions](../guides/region-counting.md): Count objects inside multiple user-defined regions with YOLO26.
- [Object Cropping](../guides/object-cropping.md): Crop detected objects from images and videos with YOLO26 and save each one as a separate image.
- [Parking Management](../guides/parking-management.md): Monitor occupied and available parking spaces with YOLO26.
- [Queue Management](../guides/queue-management.md): Count people waiting in a queue region with YOLO26 to monitor queue lengths.
- [Security Alarm System](../guides/security-alarm-system.md): Build a security alarm system with YOLO26 that sends an email alert when the number of detected objects reaches a threshold.
- [Similarity Search](../guides/similarity-search.md): Retrieve images with natural language queries such as "person holding a bag" or "vehicles in motion" by combining [OpenAI CLIP](https://www.ultralytics.com/glossary/clip-contrastive-language-image-pre-training) embeddings with cosine-similarity search.
- [Speed Estimation](../guides/speed-estimation.md): Estimate object speed with YOLO26 and object tracking for applications such as traffic monitoring.
- [Track Objects in Zone](../guides/trackzone.md): Track objects only inside a defined zone of the video frame with YOLO26.
- [VisionEye View Objects Mapping](../guides/vision-eye.md): Connect detected objects to a fixed observer point, mimicking how the human eye focuses on specific objects.
- [Workouts Monitoring](../guides/workouts-monitoring.md): Count exercise repetitions and track workout stages in real time with YOLO26 [pose estimation](https://www.ultralytics.com/glossary/pose-estimation).

### Solutions Arguments

{% from "macros/solutions-args.md" import param_table %}
{{ param_table() }}

!!! note "Track args"

    Solutions forward the inference and tracking arguments below to `model.track()`.

{% include "macros/solutions-track-args.md" %}

!!! note "Visualization args"

    You can use `show_conf`, `show_labels`, and other mentioned arguments to customize the visualization.

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "show_conf", "show_labels"]) }}

### Usage of SolutionAnnotator

All Ultralytics Solutions use the [`SolutionAnnotator`](../reference/solutions/solutions.md#ultralytics.solutions.solutions.SolutionAnnotator) class, which extends the main [`Annotator`](../reference/utils/plotting.md#ultralytics.utils.plotting.Annotator) class and provides the following methods:

| Method                             | Return Type  | Description                                                                       |
| ---------------------------------- | ------------ | --------------------------------------------------------------------------------- |
| `draw_region()`                    | `None`       | Draws a region using specified points, colors, and thickness.                     |
| `queue_counts_display()`           | `None`       | Displays queue counts in the specified region.                                    |
| `display_analytics()`              | `None`       | Displays overall statistics for parking lot management.                           |
| `estimate_pose_angle()`            | `float`      | Calculates the angle between three points in an object pose.                      |
| `draw_specific_kpts()`             | `np.ndarray` | Draws specific keypoints on the image.                                            |
| `plot_workout_information()`       | `int`        | Draws a labeled text box on the image.                                            |
| `plot_angle_and_count_and_stage()` | `None`       | Visualizes angle, step count, and stage for workout monitoring.                   |
| `plot_distance_and_line()`         | `None`       | Displays the distance between centroids and connects them with a line.            |
| `display_objects_labels()`         | `None`       | Annotates bounding boxes with object class labels.                                |
| `sweep_annotator()`                | `None`       | Visualizes a vertical sweep line and optional label.                              |
| `visioneye()`                      | `None`       | Maps and connects object centroids to a visual "eye" point.                       |
| `adaptive_label()`                 | `None`       | Draws a circular or rectangular background label in the center of a bounding box. |

### Working with SolutionResults

Every solution except Similarity Search returns a `SolutionResults` object when called.

- For object counting, the results include `in_count`, `out_count`, and `classwise_count`.

!!! example "SolutionResults"

    ```python
    import cv2

    from ultralytics import solutions

    im0 = cv2.imread("path/to/img")

    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

    counter = solutions.ObjectCounter(
        show=True,  # display the output
        region=region_points,  # pass region points
        model="yolo26n.pt",  # model="yolo26n-obb.pt" for object counting with OBB model.
        # classes=[0, 2],  # count specific classes i.e. person and car with COCO pretrained model.
        # tracker="botsort.yaml"  # Choose trackers i.e "bytetrack.yaml"
    )
    results = counter(im0)
    print(results.in_count)  # display in_counts
    print(results.out_count)  # display out_counts
    print(results.classwise_count)  # display classwise_count
    ```

`SolutionResults` objects have the following attributes:

| Attribute            | Type                               | Description                                                                                                     |
| -------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `plot_im`            | `np.ndarray`                       | Image with visual overlays such as counts, blur effects, or solution-specific enhancements.                     |
| `in_count`           | `int`                              | Total number of objects detected entering the defined zone in the video stream.                                 |
| `out_count`          | `int`                              | Total number of objects detected exiting the defined zone in the video stream.                                  |
| `classwise_count`    | `dict[str, dict[str, int] \| int]` | Per-class counts: `{"IN": int, "OUT": int}` for ObjectCounter and Heatmap, or an `int` for Analytics.           |
| `queue_count`        | `int`                              | Number of objects currently within a predefined queue or waiting area (suitable for queue management).          |
| `workout_count`      | `list[int]`                        | Per-track workout repetition counts from AI Gym (one entry per currently tracked individual).                   |
| `workout_angle`      | `list[float]`                      | Per-track exercise angles from AI Gym (one entry per currently tracked individual).                             |
| `workout_stage`      | `list[str]`                        | Per-track current workout stage from AI Gym (one entry per currently tracked individual).                       |
| `pixels_distance`    | `float`                            | Pixel-based distance between two objects or points, such as bounding boxes (suitable for distance calculation). |
| `available_slots`    | `int`                              | Number of unoccupied slots in a monitored area (suitable for parking management).                               |
| `filled_slots`       | `int`                              | Number of occupied slots in a monitored area (suitable for parking management).                                 |
| `email_sent`         | `bool`                             | Indicates whether a notification or alert email has been successfully sent (suitable for security alarm).       |
| `total_tracks`       | `int`                              | Total number of unique object tracks observed during video analysis.                                            |
| `region_counts`      | `dict[str, int]`                   | Object counts within user-defined regions or zones.                                                             |
| `speed_dict`         | `dict[int, int]`                   | Track-wise dictionary of estimated object speeds in km/h from SpeedEstimator, useful for velocity analysis.     |
| `total_crop_objects` | `int`                              | Total number of cropped object images generated by the ObjectCropper solution.                                  |
| `speed`              | `dict[str, float]`                 | Dictionary containing performance metrics for tracking and solution processing.                                 |

For more details, refer to the [`SolutionResults` class documentation](../reference/solutions/solutions.md#ultralytics.solutions.solutions.SolutionResults).

### Solutions Usage via CLI

!!! tip "Command Info"

    Most solutions can be run directly from the command-line interface.

    **Syntax**

        yolo SOLUTIONS SOLUTION_NAME ARGS

    - **SOLUTIONS** is a required keyword.
    - **SOLUTION_NAME** is one of: `['count', 'crop', 'blur', 'workout', 'heatmap', 'isegment', 'queue', 'speed', 'analytics', 'trackzone', 'inference', 'visioneye', 'region', 'security', 'parking']`.
    - **ARGS** (optional) are custom `arg=value` pairs, such as `show_in=True`, to override default settings.

```bash
yolo solutions count show=True # for object counting

yolo solutions count source="path/to/video.mp4" # specify video file path
```

### Contribute to Our Solutions

Community contributions are welcome. If you have built a solution with Ultralytics YOLO that is not covered here, consider [sharing it as a guide](../help/contributing.md) so other users can benefit.

## FAQ

### How can I use Ultralytics YOLO for real-time object counting?

Ultralytics YOLO26 [counts objects](../guides/object-counting.md) in real time by combining detection with tracking, so each object is counted once as it crosses a line or enters a region. To set up YOLO26 for live video stream analysis, install `ultralytics`, load a model, and pass video frames to `solutions.ObjectCounter`.

### What are the benefits of using Ultralytics YOLO for security systems?

Ultralytics YOLO26 adds real-time object detection and alerting to [camera-based security systems](https://www.ultralytics.com/blog/security-alarm-system-projects-with-ultralytics-yolov8). With the [Security Alarm System](../guides/security-alarm-system.md) solution, YOLO26 sends an email alert when the number of detected objects in the surveillance area reaches a threshold.

### How can Ultralytics YOLO improve queue management systems?

Ultralytics YOLO26 [counts and tracks people inside a queue region](../guides/queue-management.md), giving you live queue lengths that help reduce wait times and plan staffing.

### Can Ultralytics YOLO be used for workout monitoring?

Yes. Ultralytics YOLO26 pose models track body keypoints in real time, so the [workout monitoring solution](../guides/workouts-monitoring.md) can measure joint angles, count repetitions, and report the current exercise stage.

### How does Ultralytics YOLO help in creating heatmaps for data visualization?

Ultralytics YOLO26 can generate [heatmaps](../guides/heatmaps.md) that show where objects appear or move most often in a scene, highlighting regions of high activity. These [data visualizations](https://www.ultralytics.com/glossary/data-visualization) help you understand traffic patterns and trends over time.
