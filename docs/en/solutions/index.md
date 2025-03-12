---
comments: true
description: Explore Ultralytics Solutions using YOLO11 for object counting, blurring, security, and more. Enhance efficiency and solve real-world problems with cutting-edge AI.
keywords: Ultralytics, YOLO11, object counting, object blurring, security systems, AI solutions, real-time analysis, computer vision applications
---

# Ultralytics Solutions: Harness YOLO11 to Solve Real-World Problems

Ultralytics Solutions provide cutting-edge applications of YOLO models, offering real-world solutions like object counting, blurring, and security systems, enhancing efficiency and [accuracy](https://www.ultralytics.com/glossary/accuracy) in diverse industries. Discover the power of YOLO11 for practical, impactful implementations.

![Ultralytics Solutions Thumbnail](https://github.com/ultralytics/docs/releases/download/0/ultralytics-solutions-thumbnail.avif)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/bjkt5OE_ANA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Ultralytics Solutions from the Command Line (CLI) | Ultralytics YOLO11 🚀
</p>

## Solutions

Here's our curated list of Ultralytics solutions that can be used to create awesome [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) projects.

- [Object Counting](../guides/object-counting.md): Learn to perform real-time object counting with YOLO11. Gain the expertise to accurately count objects in live video streams.
- [Object Cropping](../guides/object-cropping.md): Master object cropping with YOLO11 for precise extraction of objects from images and videos.
- [Object Blurring](../guides/object-blurring.md): Apply object blurring using YOLO11 to protect privacy in image and video processing.
- [Workouts Monitoring](../guides/workouts-monitoring.md): Discover how to monitor workouts using YOLO11. Learn to track and analyze various fitness routines in real time.
- [Objects Counting in Regions](../guides/region-counting.md): Count objects in specific regions using YOLO11 for accurate detection in varied areas.
- [Security Alarm System](../guides/security-alarm-system.md): Create a security alarm system with YOLO11 that triggers alerts upon detecting new objects. Customize the system to fit your specific needs.
- [Heatmaps](../guides/heatmaps.md): Utilize detection heatmaps to visualize data intensity across a matrix, providing clear insights in computer vision tasks.
- [Instance Segmentation with Object Tracking](../guides/instance-segmentation-and-tracking.md): Implement [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) and object tracking with YOLO11 to achieve precise object boundaries and continuous monitoring.
- [VisionEye View Objects Mapping](../guides/vision-eye.md): Develop systems that mimic human eye focus on specific objects, enhancing the computer's ability to discern and prioritize details.
- [Speed Estimation](../guides/speed-estimation.md): Estimate object speed using YOLO11 and object tracking techniques, crucial for applications like autonomous vehicles and traffic monitoring.
- [Distance Calculation](../guides/distance-calculation.md): Calculate distances between objects using [bounding box](https://www.ultralytics.com/glossary/bounding-box) centroids in YOLO11, essential for spatial analysis.
- [Queue Management](../guides/queue-management.md): Implement efficient queue management systems to minimize wait times and improve productivity using YOLO11.
- [Parking Management](../guides/parking-management.md): Organize and direct vehicle flow in parking areas with YOLO11, optimizing space utilization and user experience.
- [Analytics](../guides/analytics.md): Conduct comprehensive data analysis to discover patterns and make informed decisions, leveraging YOLO11 for descriptive, predictive, and prescriptive analytics.
- [Live Inference with Streamlit](../guides/streamlit-live-inference.md): Leverage the power of YOLO11 for real-time [object detection](https://www.ultralytics.com/glossary/object-detection) directly through your web browser with a user-friendly Streamlit interface.
- [Track Objects in Zone](../guides/trackzone.md) 🚀 NEW: Learn how to track objects within specific zones of video frames using YOLO11 for precise and efficient monitoring.

### Solutions Arguments

{% from "macros/solutions-args.md" import param_table %}
{{ param_table() }}

!!! note "Track args"

     Solutions also support some of the arguments from `track`, including parameters such as `conf`, `line_width`, `tracker`, `model`, `show`, `verbose` and `classes`.

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

### Usage of SolutionAnnotator

All Ultralytics Solutions use the separate class [`SolutionAnnotator`](https://docs.ultralytics.com/reference/solutions/solutions/#ultralytics.solutions.solutions.SolutionAnnotator), that extends the main [`Annotator`](https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Annotator) class, and have the following methods:

| Method                             | Return Type | Description                                                            |
| ---------------------------------- | ----------- | ---------------------------------------------------------------------- |
| `draw_region()`                    | `None`      | Draws a region using specified points, colors, and thickness.          |
| `queue_counts_display()`           | `None`      | Displays queue counts in the specified region.                         |
| `display_analytics()`              | `None`      | Displays overall statistics for parking lot management.                |
| `estimate_pose_angle()`            | `float`     | Calculates the angle between three points in an object pose.           |
| `draw_specific_points()`           | `None`      | Draws specific keypoints on the image.                                 |
| `plot_workout_information()`       | `None`      | Draws a labeled text box on the image.                                 |
| `plot_angle_and_count_and_stage()` | `None`      | Visualizes angle, step count, and stage for workout monitoring.        |
| `plot_distance_and_line()`         | `None`      | Displays the distance between centroids and connects them with a line. |
| `display_objects_labels()`         | `None`      | Annotates bounding boxes with object class labels.                     |
| `seg_bbox()`                       | `None`      | Draws contours for segmented objects and optionally labels them.       |
| `sweep_annotator()`                | `None`      | Visualizes a vertical sweep line and optional label.                   |
| `visioneye()`                      | `None`      | Maps and connects object centroids to a visual "eye" point.            |
| `circle_label()`                   | `None`      | Draws a circular label in the place of a bounding box.                 |
| `text_label()`                     | `None`      | Draws a rectangular label in the place of a bounding box.              |

### Working with SolutionResults

All Solutions calls return a list of `SolutionResults` objects, containing comprehensive information about the solutions.

- For object counting, the results include `incounts`, `outcounts`, and `classwise_counts`.

!!! example "SolutionResults"

    ```python
    counter = solutions.ObjectCounter(
        show=True,  # display the output
        region=region_points,  # pass region points
        model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
        # classes=[0, 2],           # count specific classes i.e. person and car with COCO pretrained model.
        # tracker="botsort.yaml"    # Choose trackers i.e "bytetrack.yaml"
    )
    results = counter.count(im0)
    print(results.in_counts)  # display in_counts
    print(results.out_counts)  # display out_counts
    ```

For more details, refer to the [`SolutionResults` class documentation](https://docs.ultralytics.com/reference/solutions/solutions/#ultralytics.solutions.solutions.SolutionAnnotator).

### Solutions Usage via CLI

!!! tip "Command Info"

    Most of the Solutions can be used directly through the command-line interface, including:

    `Count`, `Crop`, `Blur`, `Workout`, `Heatmap`, `Isegment`, `Visioneye`, `Speed`, `Queue`, `Analytics`, `Inference`

    **Syntax**

        yolo SOLUTIONS SOLUTION_NAME ARGS

    - **SOLUTIONS** is a required keyword.
    - **SOLUTION_NAME** is one of: `['count', 'crop', 'blur', 'workout', 'heatmap', 'isegment', 'queue', 'speed', 'analytics', 'trackzone', 'inference', 'visioneye']`.
    - **ARGS** (optional) are custom `arg=value` pairs, such as `show_in=True`, to override default settings.

```bash
yolo solutions count show=True  # for object counting

yolo solutions source="path/to/video/file.mp4"  # specify video file path
```

### Contribute to Our Solutions

We welcome contributions from the community! If you've mastered a particular aspect of Ultralytics YOLO that's not yet covered in our solutions, we encourage you to share your expertise. Writing a guide is a great way to give back to the community and help us make our documentation more comprehensive and user-friendly.

To get started, please read our [Contributing Guide](../help/contributing.md) for guidelines on how to open up a Pull Request (PR) 🛠️. We look forward to your contributions!

Let's work together to make the Ultralytics YOLO ecosystem more robust and versatile 🙏!

## FAQ

### How can I use Ultralytics YOLO for real-time object counting?

Ultralytics YOLO11 can be used for real-time object counting by leveraging its advanced object detection capabilities. You can follow our detailed guide on [Object Counting](../guides/object-counting.md) to set up YOLO11 for live video stream analysis. Simply install YOLO11, load your model, and process video frames to count objects dynamically.

### What are the benefits of using Ultralytics YOLO for security systems?

Ultralytics YOLO11 enhances security systems by offering real-time object detection and alert mechanisms. By employing YOLO11, you can create a security alarm system that triggers alerts when new objects are detected in the surveillance area. Learn how to set up a [Security Alarm System](../guides/security-alarm-system.md) with YOLO11 for robust security monitoring.

### How can Ultralytics YOLO improve queue management systems?

Ultralytics YOLO11 can significantly improve queue management systems by accurately counting and tracking people in queues, thus helping to reduce wait times and optimize service efficiency. Follow our detailed guide on [Queue Management](../guides/queue-management.md) to learn how to implement YOLO11 for effective queue monitoring and analysis.

### Can Ultralytics YOLO be used for workout monitoring?

Yes, Ultralytics YOLO11 can be effectively used for monitoring workouts by tracking and analyzing fitness routines in real-time. This allows for precise evaluation of exercise form and performance. Explore our guide on [Workouts Monitoring](../guides/workouts-monitoring.md) to learn how to set up an AI-powered workout monitoring system using YOLO11.

### How does Ultralytics YOLO help in creating heatmaps for [data visualization](https://www.ultralytics.com/glossary/data-visualization)?

Ultralytics YOLO11 can generate heatmaps to visualize data intensity across a given area, highlighting regions of high activity or interest. This feature is particularly useful in understanding patterns and trends in various computer vision tasks. Learn more about creating and using [Heatmaps](../guides/heatmaps.md) with YOLO11 for comprehensive data analysis and visualization.
