---
comments: true
description: Discover how TrackZone leverages Ultralytics YOLO11 to precisely track objects within specific zones, enabling real-time insights for crowd analysis, surveillance, and targeted monitoring.
keywords: TrackZone, object tracking, YOLO11, Ultralytics, real-time object detection, AI, deep learning, crowd analysis, surveillance, zone-based tracking, resource optimization
---

# TrackZone using Ultralytics YOLO11

## What is TrackZone?

TrackZone specializes in monitoring objects within designated areas of a frame instead of the whole frame. Built on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/), it integrates object detection and tracking specifically within zones for videos and live camera feeds. YOLO11's advanced algorithms and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) technologies make it a perfect choice for real-time use cases, offering precise and efficient object tracking in applications like crowd monitoring and surveillance.

## Advantages of Object Tracking in Zones (TrackZone)

- **Targeted Analysis:** Tracking objects within specific zones allows for more focused insights, enabling precise monitoring and analysis of areas of interest, such as entry points or restricted zones.
- **Improved Efficiency:** By narrowing the tracking scope to defined zones, TrackZone reduces computational overhead, ensuring faster processing and optimal performance.
- **Enhanced Security:** Zonal tracking improves surveillance by monitoring critical areas, aiding in the early detection of unusual activity or security breaches.
- **Scalable Solutions:** The ability to focus on specific zones makes TrackZone adaptable to various scenarios, from retail spaces to industrial settings, ensuring seamless integration and scalability.

## Real World Applications

|                                                                             Agriculture                                                                             |                                                                            Transportation                                                                             |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Plants Tracking in Field Using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/plants-tracking-in-zone-using-ultralytics-yolo11.avif) | ![Vehicles Tracking on Road using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/vehicle-tracking-in-zone-using-ultralytics-yolo11.avif) |
|                                                          Plants Tracking in Field Using Ultralytics YOLO11                                                          |                                                          Vehicles Tracking on Road using Ultralytics YOLO11                                                           |

!!! example "TrackZone using YOLO11 Example"

    === "CLI"

        ```bash
        # Run a trackzone example
        yolo solutions trackzone show=True

        # Pass a source video
        yolo solutions trackzone show=True source="path/to/video/file.mp4"

        # Pass region coordinates
        yolo solutions trackzone show=True region=[(150, 150), (1130, 150), (1130, 570), (150, 570)]
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define region points
        region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init TrackZone (Object Tracking in Zones, not complete frame)
        trackzone = solutions.TrackZone(
            show=True,  # Display the output
            region=region_points,  # Pass region points
            model="yolo11n.pt",  # You can use any model that Ultralytics support, i.e. YOLOv9, YOLOv10
            # line_width=2,  # Adjust the line width for bounding boxes and text display
            # classes=[0, 2],  # If you want to count specific classes i.e. person and car with COCO pretrained model.
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            im0 = trackzone.trackzone(im0)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

### Argument `TrackZone`

Here's a table with the `TrackZone` arguments:

| Name         | Type   | Default                                              | Description                                          |
| ------------ | ------ | ---------------------------------------------------- | ---------------------------------------------------- |
| `model`      | `str`  | `None`                                               | Path to Ultralytics YOLO Model File                  |
| `region`     | `list` | `[(150, 150), (1130, 150), (1130, 570), (150, 570)]` | List of points defining the object tracking region.  |
| `line_width` | `int`  | `2`                                                  | Line thickness for bounding boxes.                   |
| `show`       | `bool` | `False`                                              | Flag to control whether to display the video stream. |

### Arguments `model.track`

{% include "macros/track-args.md" %}

## FAQ

### How do I track objects in a specific area or zone of a video frame using Ultralytics YOLO11?

Tracking objects in a defined area or zone of a video frame is straightforward with Ultralytics YOLO11. Simply use the command provided below to initiate tracking. This approach ensures efficient analysis and accurate results, making it ideal for applications like surveillance, crowd management, or any scenario requiring zonal tracking.

```bash
yolo solutions trackzone source="path/to/video/file.mp4" show=True
```

### How can I use TrackZone in Python with Ultralytics YOLO11?

With just a few lines of code, you can set up object tracking in specific zones, making it easy to integrate into your projects.

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init TrackZone (Object Tracking in Zones, not complete frame)
trackzone = solutions.TrackZone(
    show=True,  # Display the output
    region=region_points,  # Pass region points
    model="yolo11n.pt",
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = trackzone.trackzone(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

### How do I configure the zone points for video processing using Ultralytics TrackZone?

Configuring zone points for video processing with Ultralytics TrackZone is simple and customizable. You can directly define and adjust the zones through a Python script, allowing precise control over the areas you want to monitor.

```python
# Define region points
region_points = [(150, 150), (1130, 150), (1130, 570), (150, 570)]

# Init TrackZone (Object Tracking in Zones, not complete frame)
trackzone = solutions.TrackZone(
    show=True,  # Display the output
    region=region_points,  # Pass region points
)
```

### What models are supported by TrackZone?

TrackZone supports any model compatible with Ultralytics framework, including [Ultralytics YOLOv8](../models/yolov8.md), [YOLOv9](../models/yolov9.md), [YOLOv10](../models/yolov10.md), and [Ultralytics YOLO11](../models/yolo11.md) variants. You can load custom-trained models as well by specifying the model path in the model argument.

### Can I adjust the line width and appearance of bounding boxes?

Yes, you can customize the line width of bounding boxes using the line_width argument. This flexibility allows you to adjust the visuals for better clarity and presentation based on your application's needs.

### How do I filter specific object classes using TrackZone?

You can filter objects by specifying the classes argument in the TrackZone initialization. For example, setting `classes=[0, 2]` will track only persons and cars if using a COCO-pretrained model.

```python
trackzone = solutions.TrackZone(
    show=True,
    region=region_points,
    model="yolo11n.pt",
    classes=[0, 2],  # Track only persons and cars
)
```

### Is real-time streaming supported by TrackZone?

Yes, TrackZone supports real-time object tracking from live video feeds, including IP cameras, webcams, and RTSP streams. Replace the source path with the video stream URL or camera ID when initializing the video source.

```python
cap = cv2.VideoCapture(0)  # Use a webcam
```
