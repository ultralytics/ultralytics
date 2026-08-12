---
title: YOLO26 Computer Vision Solutions
comments: true
description: Explore Ultralytics Solutions using YOLO26 for object counting, blurring, security, and more. Enhance efficiency and solve real-world problems with cutting-edge AI.
keywords: Ultralytics, YOLO26, object counting, object blurring, security systems, AI solutions, real-time analysis, computer vision applications
---

# Ultralytics Solutions: Harness YOLO26 to Solve Real-World Problems

Ultralytics Solutions provide cutting-edge applications of YOLO models, offering real-world solutions like object counting, blurring, and security systems, enhancing efficiency and [accuracy](https://www.ultralytics.com/glossary/accuracy) in diverse industries. Discover the power of YOLO26 for practical, impactful implementations.

![Ultralytics Solutions Thumbnail](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-solutions-thumbnail.avif)

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

Here's our curated list of Ultralytics solutions that can be used to create awesome [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) projects.

- [Analytics](../guides/analytics.md): Conduct comprehensive data analysis to discover patterns and make informed decisions, leveraging YOLO26 for descriptive, predictive, and prescriptive analytics.
- [Distance Calculation](../guides/distance-calculation.md): Calculate distances between objects using [bounding box](https://www.ultralytics.com/glossary/bounding-box) centroids in YOLO26, essential for spatial analysis.
- [Heatmaps](../guides/heatmaps.md): Utilize detection heatmaps to visualize data intensity across a matrix, providing clear insights in computer vision tasks.
- [Instance Segmentation with Object Tracking](../guides/instance-segmentation-and-tracking.md): Implement [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) and object tracking with YOLO26 to achieve precise object boundaries and continuous monitoring.
- [Live Inference with Streamlit](../guides/streamlit-live-inference.md): Leverage the power of YOLO26 for real-time [object detection](https://www.ultralytics.com/glossary/object-detection) directly through your web browser with a user-friendly Streamlit interface.
- [Object Blurring](../guides/object-blurring.md): Apply object blurring using YOLO26 to protect privacy in image and video processing.
- [Object Counting](../guides/object-counting.md): Learn to perform real-time object counting with YOLO26. Gain the expertise to accurately count objects in live video streams.
- [Object Counting in Regions](../guides/region-counting.md): Count objects in specific regions using YOLO26 for accurate detection in varied areas.
- [Object Cropping](../guides/object-cropping.md): Master object cropping with YOLO26 for precise extraction of objects from images and videos.
- [Parking Management](../guides/parking-management.md): Organize and direct vehicle flow in parking areas with YOLO26, optimizing space utilization and user experience.
- [Queue Management](../guides/queue-management.md): Implement efficient queue management systems to minimize wait times and improve productivity using YOLO26.
- [Security Alarm System](../guides/security-alarm-system.md): Create a security alarm system with YOLO26 that triggers alerts upon detecting new objects. Customize the system to fit your specific needs.
- [Similarity Search](../guides/similarity-search.md): Enable intelligent image retrieval by combining [OpenAI CLIP](https://cookbook.openai.com/examples/custom_image_embedding_search) embeddings with cosine-similarity search, allowing natural language queries like "person holding a bag" or "vehicles in motion."
- [Speed Estimation](../guides/speed-estimation.md): Estimate object speed using YOLO26 and object tracking techniques, crucial for applications like autonomous vehicles and traffic monitoring.
- [Track Objects in Zone](../guides/trackzone.md): Learn how to track objects within specific zones of video frames using YOLO26 for precise and efficient monitoring.
- [VisionEye View Objects Mapping](../guides/vision-eye.md): Develop systems that mimic human eye focus on specific objects, enhancing the computer's ability to discern and prioritize details.
- [Workouts Monitoring](../guides/workouts-monitoring.md): Discover how to monitor workouts using YOLO26. Learn to track and analyze various fitness routines in real time.

### Solutions Arguments

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `model` | `str` | `None` | Path to an Ultralytics YOLO model file. |
| `region` | `list` or `dict` | `None` | Points defining the region of interest, either a list of `(x, y)` tuples or a dictionary mapping region names to point lists for multiple regions (`RegionCounter` only). When `None`, solutions that require a region fall back to a predefined default. |
| `show_in` | `bool` | `True` | Flag to control whether to display the in counts on the video stream. |
| `show_out` | `bool` | `True` | Flag to control whether to display the out counts on the video stream. |
| `analytics_type` | `str` | `'line'` | Type of graph, i.e., `line`, `bar`, `area`, or `pie`. |
| `colormap` | `int` | `cv2.COLORMAP_DEEPGREEN` | Colormap to use for the heatmap. |
| `json_file` | `str` | `None` | Path to the JSON file that contains all parking coordinates data. |
| `up_angle` | `float` | `145.0` | Angle threshold for the 'up' pose. |
| `kpts` | `list[int]` | `'[6, 8, 10]'` | List of three keypoint indices used for monitoring workouts. These keypoints correspond to body joints or parts, such as shoulders, elbows, and wrists, for exercises like push-ups, pull-ups, squats, and ab-workouts. |
| `down_angle` | `int` | `90` | Angle threshold for the 'down' pose. |
| `blur_ratio` | `float` | `0.5` | Adjusts percentage of blur intensity, with values in range `0.1 - 1.0`. |
| `crop_dir` | `str` | `'cropped-detections'` | Directory name for storing cropped detections. |
| `records` | `int` | `5` | Total detections count to trigger an email with security alarm system. |
| `vision_point` | `tuple[int, int]` | `(20, 20)` | The point where vision will track objects and draw paths using VisionEye Solution. |
| `source` | `str` | `None` | Path to the input source (video, RTSP, etc.). Only usable with Solutions command line interface (CLI). |
| `figsize` | `tuple[float, float]` | `(12.8, 7.2)` | Figure size for analytics charts such as heatmaps or graphs. |
| `fps` | `float` | `30.0` | Frames per second used for speed calculations. |
| `max_hist` | `int` | `5` | Maximum historical points to track per object for speed/direction calculations. |
| `meter_per_pixel` | `float` | `0.05` | Scaling factor used for converting pixel distance to real-world units. |
| `max_speed` | `int` | `120` | Maximum speed limit in visual overlays (used in alerts). |
| `data` | `str` | `'images'` | Path to image directory used for similarity search. |
| `imgsz` | `int` | `640` | Input image size for model inference. |


!!! note "Track args"

     Solutions also support some of the arguments from `track`, including parameters such as `conf`, `line_width`, `tracker`, `model`, `show`, `verbose` and `classes`.

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `tracker` | `str` | `'tracktrack.yaml'` | Specifies the tracking algorithm to use. Built-in options: `botsort.yaml`, `bytetrack.yaml`, `ocsort.yaml`, `deepocsort.yaml`, `fasttrack.yaml`, `tracktrack.yaml`. |
| `conf` | `float` | `0.1` | Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives. |
| `iou` | `float` | `0.7` | Sets the [Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for filtering overlapping detections. |
| `classes` | `list` | `None` | Filters results by class index. For example, `classes=[0, 2, 3]` only tracks the specified classes. |
| `verbose` | `bool` | `True` | Controls the display of tracking results, providing a visual output of tracked objects. |
| `device` | `str` | `None` | Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution. |


!!! note "Visualization args"

    You can use `show_conf`, `show_labels`, and other mentioned arguments to customize the visualization.

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `show` | `bool` | `False` | If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing. |
| `line_width` | `int or None` | `None` | Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity. |
| `show_conf` | `bool` | `True` | Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection. |
| `show_labels` | `bool` | `True` | Displays labels for each detection in the visual output. Provides immediate understanding of detected objects. |


### Usage of SolutionAnnotator

All Ultralytics Solutions use the separate class [`SolutionAnnotator`](../reference/solutions/solutions.md#ultralytics.solutions.solutions.SolutionAnnotator), that extends the main [`Annotator`](../reference/utils/plotting.md#ultralytics.utils.plotting.Annotator) class, and have the following methods:

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

Except [`Similarity Search`](../guides/similarity-search.md), each Solution call returns a `SolutionResults` object.

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

`SolutionResults` object have the following attributes:

| Attribute            | Type               | Description                                                                                                   |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `plot_im`            | `np.ndarray`       | Image with visual overlays such as counts, blur effects, or solution-specific enhancements.                   |
| `in_count`           | `int`              | Total number of objects detected entering the defined zone in the video stream.                               |
| `out_count`          | `int`              | Total number of objects detected exiting the defined zone in the video stream.                                |
| `classwise_count`    | `Dict[str, int]`   | Dictionary recording class-wise in/out object counts for advanced analytics.                                  |
| `queue_count`        | `int`              | Number of objects currently within a predefined queue or waiting area (suitable for queue management).        |
| `workout_count`      | `list[int]`        | Per-track workout repetition counts from AI Gym (one entry per currently tracked individual).                 |
| `workout_angle`      | `list[float]`      | Per-track exercise angles from AI Gym (one entry per currently tracked individual).                           |
| `workout_stage`      | `list[str]`        | Per-track current workout stage from AI Gym (one entry per currently tracked individual).                     |
| `pixels_distance`    | `float`            | Pixel-based distance between two objects or points e.g., bounding boxes. (Suitable for distance calculation). |
| `available_slots`    | `int`              | Number of unoccupied slots in a monitored area (suitable for parking management).                             |
| `filled_slots`       | `int`              | Number of occupied slots in a monitored area. (suitable for parking management)                               |
| `email_sent`         | `bool`             | Indicates whether a notification or alert email has been successfully sent (suitable for security alarm).     |
| `total_tracks`       | `int`              | Total number of unique object tracks observed during video analysis.                                          |
| `region_counts`      | `Dict[str, int]`   | Object counts within user-defined regions or zones.                                                           |
| `speed_dict`         | `Dict[str, float]` | Track-wise dictionary of calculated object speeds, useful for velocity analysis.                              |
| `total_crop_objects` | `int`              | Total number of cropped object images generated by the ObjectCropper solution.                                |
| `speed`              | `Dict[str, float]` | Dictionary containing performance metrics for tracking and solution processing.                               |

For more details, refer to the [`SolutionResults` class documentation](../reference/solutions/solutions.md#ultralytics.solutions.solutions.SolutionResults).

### Solutions Usage via CLI

!!! tip "Command Info"

    Most of the Solutions can be used directly through the command-line interface, including:

    `Count`, `Crop`, `Blur`, `Workout`, `Heatmap`, `Isegment`, `Visioneye`, `Speed`, `Queue`, `Analytics`, `Inference`, `Trackzone`

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

We welcome contributions from the community! If you've mastered a particular aspect of Ultralytics YOLO that's not yet covered in our solutions, we encourage you to share your expertise. Writing a guide is a great way to give back to the community and help us make our documentation more comprehensive and user-friendly.

To get started, please read our [Contributing Guide](../help/contributing.md) for guidelines on how to open up a Pull Request (PR) 🛠️. We look forward to your contributions!

Let's work together to make the Ultralytics YOLO ecosystem more robust and versatile 🙏!

## FAQ

### How can I use Ultralytics YOLO for real-time object counting?

Ultralytics YOLO26 can be used for real-time object counting by leveraging its advanced object detection capabilities. You can follow our detailed guide on [Object Counting](../guides/object-counting.md) to set up YOLO26 for live video stream analysis. Simply install YOLO26, load your model, and process video frames to count objects dynamically.

### What are the benefits of using Ultralytics YOLO for security systems?

Ultralytics YOLO26 enhances security systems by offering real-time object detection and alert mechanisms. By employing YOLO26, you can create a security alarm system that triggers alerts when new objects are detected in the surveillance area. Learn how to set up a [Security Alarm System](../guides/security-alarm-system.md) with YOLO26 for robust security monitoring.

### How can Ultralytics YOLO improve queue management systems?

Ultralytics YOLO26 can significantly improve queue management systems by accurately counting and tracking people in queues, thus helping to reduce wait times and optimize service efficiency. Follow our detailed guide on [Queue Management](../guides/queue-management.md) to learn how to implement YOLO26 for effective queue monitoring and analysis.

### Can Ultralytics YOLO be used for workout monitoring?

Yes, Ultralytics YOLO26 can be effectively used for monitoring workouts by tracking and analyzing fitness routines in real-time. This allows for precise evaluation of exercise form and performance. Explore our guide on [Workouts Monitoring](../guides/workouts-monitoring.md) to learn how to set up an AI-powered workout monitoring system using YOLO26.

### How does Ultralytics YOLO help in creating heatmaps for [data visualization](https://www.ultralytics.com/glossary/data-visualization)?

Ultralytics YOLO26 can generate heatmaps to visualize data intensity across a given area, highlighting regions of high activity or interest. This feature is particularly useful in understanding patterns and trends in various computer vision tasks. Learn more about creating and using [Heatmaps](../guides/heatmaps.md) with YOLO26 for comprehensive data analysis and visualization.
