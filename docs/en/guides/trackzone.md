---
comments: true
description: Discover how TrackZone leverages Ultralytics YOLO26 to precisely track objects within specific zones, enabling real-time insights for crowd analysis, surveillance, and targeted monitoring.
keywords: TrackZone, object tracking, YOLO26, Ultralytics, real-time object detection, AI, deep learning, crowd analysis, surveillance, zone-based tracking, resource optimization
---

# TrackZone using Ultralytics YOLO26

## What is TrackZone?

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-track-the-objects-in-zone-using-ultralytics-yolo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open TrackZone In Colab"></a>

TrackZone specializes in monitoring objects within designated areas of a frame instead of the whole frame. Built on [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/), it integrates object detection and [tracking](../modes/track.md) specifically within zones for videos and live camera feeds. YOLO26's advanced algorithms and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) technologies make it a perfect choice for real-time use cases, offering precise and efficient object tracking in applications like crowd monitoring and surveillance.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/SMSJvjUG1ko"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Track Objects in Region using Ultralytics YOLO26 | TrackZone 🚀
</p>

## Advantages of Object Tracking in Zones (TrackZone)

- **Targeted Analysis:** Tracking objects within specific zones allows for more focused insights, enabling precise monitoring and analysis of areas of interest, such as entry points or restricted zones.
- **Reduced Downstream Workload:** By ignoring objects outside the zone, TrackZone removes irrelevant detections so there are fewer objects to count, log, or alert on in the logic you build on top of it. Detection still runs on the full frame, so the benefit is cleaner, more focused output rather than faster model inference.
- **Enhanced Security:** Zonal tracking improves surveillance by monitoring critical areas, aiding in the early detection of unusual activity or security breaches.
- **Scalable Solutions:** The ability to focus on specific zones makes TrackZone adaptable to various scenarios, from retail spaces to industrial settings, ensuring seamless integration and scalability.

## Real World Applications

|                                                                             Agriculture                                                                              |                                                                             Transportation                                                                             |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Plants Tracking in Field Using Ultralytics YOLO26](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/plants-tracking-in-zone-using-ultralytics-yolo11.avif) | ![Vehicles Tracking on Road using Ultralytics YOLO26](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/vehicle-tracking-in-zone-using-ultralytics-yolo11.avif) |
|                                                          Plants Tracking in Field Using Ultralytics YOLO26                                                           |                                                           Vehicles Tracking on Road using Ultralytics YOLO26                                                           |

!!! example "TrackZone using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Run a trackzone example
        yolo solutions trackzone show=True

        # Pass a source video
        yolo solutions trackzone source="path/to/video.mp4" show=True

        # Pass region coordinates
        yolo solutions trackzone show=True region="[(150, 150), (1130, 150), (1130, 570), (150, 570)]"
        ```

        TrackZone relies on the `region` list to know which part of the frame to monitor. Define the polygon to match the physical zone you care about (doors, gates, etc.), and keep `show=True` enabled while configuring so you can verify the overlay aligns with the video feed.

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Define region points
        region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("trackzone_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init trackzone (object tracking in zones, not complete frame)
        trackzone = solutions.TrackZone(
            show=True,  # display the output
            region=region_points,  # pass region points
            model="yolo26n.pt",  # use any model that Ultralytics supports, e.g., YOLOv9, YOLOv10
            # line_width=2,  # adjust the line width for bounding boxes and text display
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = trackzone(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the video file

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

!!! tip "Defining the tracking zone"

    - Each entry in `region` is an `(x, y)` pixel coordinate in the video frame. List the points in the order they should be connected around the perimeter of the area you want to monitor.
    - Coordinates are tied to the frame resolution, so a region sized for a 1280×720 feed will not line up with a 640×480 one. Keep `show=True` while configuring so you can confirm the overlay matches your feed.
    - `TrackZone` reduces the points to their [convex hull](https://en.wikipedia.org/wiki/Convex_hull), so a concave shape is simplified to the smallest convex polygon that contains all of its points. For non-convex shapes or several separate areas, use the [RegionCounter](region-counting.md) solution instead.
    - If you omit `region` entirely, a default zone of `[(75, 75), (565, 75), (565, 285), (75, 285)]` is used.

### `TrackZone` Arguments

Here's a table with the `TrackZone` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "region"]) }}

The TrackZone solution includes support for `track` parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Moreover, the following visualization options are available:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## Count Objects Inside the Zone

Every call to the tracker returns a `SolutionResults` object whose `total_tracks` attribute holds the number of objects currently tracked inside the zone. Read it on each frame to monitor live occupancy, for example to log how busy an entry point or restricted area is:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]
trackzone = solutions.TrackZone(show=False, region=region_points, model="yolo26n.pt")

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = trackzone(im0)
    print(f"Objects currently in zone: {results.total_tracks}")  # live zone occupancy

cap.release()
```

## FAQ

### How do I track objects in a specific area or zone of a video frame using Ultralytics YOLO26?

Tracking objects in a defined area or zone of a video frame is straightforward with Ultralytics YOLO26. Simply use the command provided below to initiate tracking. This approach ensures efficient analysis and accurate results, making it ideal for applications like surveillance, crowd management, or any scenario requiring zonal tracking.

```bash
yolo solutions trackzone source="path/to/video.mp4" show=True
```

### How can I use TrackZone in Python with Ultralytics YOLO26?

With just a few lines of code, you can set up object tracking in specific zones, making it easy to integrate into your projects.

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("trackzone_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

trackzone = solutions.TrackZone(
    show=True, region=[(150, 150), (1130, 150), (1130, 570), (150, 570)], model="yolo26n.pt"
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = trackzone(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

### How do I configure the zone points for video processing using Ultralytics TrackZone?

Configuring zone points for video processing with Ultralytics TrackZone is simple and customizable. You can directly define and adjust the zones through a Python script, allowing precise control over the areas you want to monitor.

```python
# Define region points
region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

# Initialize trackzone
trackzone = solutions.TrackZone(
    show=True,  # display the output
    region=region_points,  # pass region points
)
```

Remember that `TrackZone` reduces the points to their convex hull, so list them in order around the perimeter of the area you want to monitor.

### When should I use TrackZone instead of ObjectCounter or RegionCounter?

All three solutions work with regions, but they answer different questions:

| Solution | Use it to | Typical output |
| --- | --- | --- |
| **TrackZone** | Track objects and monitor live occupancy inside a single convex zone | Tracked IDs and `total_tracks` for the zone |
| [ObjectCounter](object-counting.md) | Count objects that cross a line or enter and leave a region | Cumulative in and out counts |
| [RegionCounter](region-counting.md) | Count objects inside one or more arbitrary (including non-convex) regions | Per-region object counts |

Choose TrackZone when you want continuous tracking inside one area, and [RegionCounter](region-counting.md) when you need multiple zones or a non-convex shape.
