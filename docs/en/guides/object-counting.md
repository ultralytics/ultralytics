---
comments: true
description: Learn to accurately identify and count objects in real-time using Ultralytics YOLO11 for applications like crowd analysis and surveillance.
keywords: object counting, YOLO11, Ultralytics, real-time object detection, AI, deep learning, object tracking, crowd analysis, surveillance, resource optimization
---

# Object Counting using Ultralytics YOLO11

## What is Object Counting?

Object counting with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) involves accurate identification and counting of specific objects in videos and camera streams. YOLO11 excels in real-time applications, providing efficient and precise object counting for various scenarios like crowd analysis and surveillance, thanks to its state-of-the-art algorithms and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) capabilities.

<table>
  <tr>
    <td align="center">
      <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Ag2e-5_NpS0"
        title="YouTube video player" frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen>
      </iframe>
      <br>
      <strong>Watch:</strong> Object Counting using Ultralytics YOLO11
    </td>
    <td align="center">
      <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Fj9TStNBVoY"
        title="YouTube video player" frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen>
      </iframe>
      <br>
      <strong>Watch:</strong> Class-wise Object Counting using Ultralytics YOLO11
    </td>
  </tr>
</table>

## Advantages of Object Counting?

- **Resource Optimization:** Object counting facilitates efficient resource management by providing accurate counts, and optimizing resource allocation in applications like inventory management.
- **Enhanced Security:** Object counting enhances security and surveillance by accurately tracking and counting entities, aiding in proactive threat detection.
- **Informed Decision-Making:** Object counting offers valuable insights for decision-making, optimizing processes in retail, traffic management, and various other domains.

## Real World Applications

|                                                                        Logistics                                                                        |                                                                         Aquaculture                                                                          |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Conveyor Belt Packets Counting Using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/conveyor-belt-packets-counting.avif) | ![Fish Counting in Sea using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/fish-counting-in-sea-using-ultralytics-yolov8.avif) |
|                                                 Conveyor Belt Packets Counting Using Ultralytics YOLO11                                                 |                                                        Fish Counting in Sea using Ultralytics YOLO11                                                         |

!!! example "Object Counting using YOLO11 Example"

    === "Count in Region"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define region points
        region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init Object Counter
        counter = solutions.ObjectCounter(
            show=True,
            region=region_points,
            model="yolo11n.pt",
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            im0 = counter.count(im0)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "OBB Object Counting"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # line or region points
        line_points = [(20, 400), (1080, 400)]

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init Object Counter
        counter = solutions.ObjectCounter(
            show=True,
            region=line_points,
            model="yolo11n-obb.pt",
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            im0 = counter.count(im0)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "Count in Polygon"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define region points
        region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init Object Counter
        counter = solutions.ObjectCounter(
            show=True,
            region=region_points,
            model="yolo11n.pt",
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            im0 = counter.count(im0)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "Count in Line"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define region points
        line_points = [(20, 400), (1080, 400)]

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init Object Counter
        counter = solutions.ObjectCounter(
            show=True,
            region=line_points,
            model="yolo11n.pt",
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            im0 = counter.count(im0)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "Specific Classes"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init Object Counter
        counter = solutions.ObjectCounter(
            show=True,
            model="yolo11n.pt",
            classes=[0, 1],
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            im0 = counter.count(im0)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

### Argument `ObjectCounter`

Here's a table with the `ObjectCounter` arguments:

| Name         | Type   | Default                    | Description                                                            |
| ------------ | ------ | -------------------------- | ---------------------------------------------------------------------- |
| `model`      | `str`  | `None`                     | Path to Ultralytics YOLO Model File                                    |
| `region`     | `list` | `[(20, 400), (1260, 400)]` | List of points defining the counting region.                           |
| `line_width` | `int`  | `2`                        | Line thickness for bounding boxes.                                     |
| `show`       | `bool` | `False`                    | Flag to control whether to display the video stream.                   |
| `show_in`    | `bool` | `True`                     | Flag to control whether to display the in counts on the video stream.  |
| `show_out`   | `bool` | `True`                     | Flag to control whether to display the out counts on the video stream. |

### Arguments `model.track`

{% include "macros/track-args.md" %}

## FAQ

### How do I count objects in a video using Ultralytics YOLO11?

To count objects in a video using Ultralytics YOLO11, you can follow these steps:

1. Import the necessary libraries (`cv2`, `ultralytics`).
2. Define the counting region (e.g., a polygon, line, etc.).
3. Set up the video capture and initialize the object counter.
4. Process each frame to track objects and count them within the defined region.

Here's a simple example for counting in a region:

```python
import cv2

from ultralytics import solutions


def count_objects_in_region(video_path, output_video_path, model_path):
    """Count objects in a specific region within a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = counter.count(im0)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_objects_in_region("path/to/video.mp4", "output_video.avi", "yolo11n.pt")
```

Explore more configurations and options in the [Object Counting](#object-counting-using-ultralytics-yolo11) section.

### What are the advantages of using Ultralytics YOLO11 for object counting?

Using Ultralytics YOLO11 for object counting offers several advantages:

1. **Resource Optimization:** It facilitates efficient resource management by providing accurate counts, helping optimize resource allocation in industries like inventory management.
2. **Enhanced Security:** It enhances security and surveillance by accurately tracking and counting entities, aiding in proactive threat detection.
3. **Informed Decision-Making:** It offers valuable insights for decision-making, optimizing processes in domains like retail, traffic management, and more.

For real-world applications and code examples, visit the [Advantages of Object Counting](#advantages-of-object-counting) section.

### How can I count specific classes of objects using Ultralytics YOLO11?

To count specific classes of objects using Ultralytics YOLO11, you need to specify the classes you are interested in during the tracking phase. Below is a Python example:

```python
import cv2

from ultralytics import solutions


def count_specific_classes(video_path, output_video_path, model_path, classes_to_count):
    """Count specific classes of objects in a video."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    line_points = [(20, 400), (1080, 400)]
    counter = solutions.ObjectCounter(show=True, region=line_points, model=model_path, classes=classes_to_count)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        im0 = counter.count(im0)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_specific_classes("path/to/video.mp4", "output_specific_classes.avi", "yolo11n.pt", [0, 2])
```

In this example, `classes_to_count=[0, 2]`, which means it counts objects of class `0` and `2` (e.g., person and car).

### Why should I use YOLO11 over other [object detection](https://www.ultralytics.com/glossary/object-detection) models for real-time applications?

Ultralytics YOLO11 provides several advantages over other object detection models like Faster R-CNN, SSD, and previous YOLO versions:

1. **Speed and Efficiency:** YOLO11 offers real-time processing capabilities, making it ideal for applications requiring high-speed inference, such as surveillance and autonomous driving.
2. **[Accuracy](https://www.ultralytics.com/glossary/accuracy):** It provides state-of-the-art accuracy for object detection and tracking tasks, reducing the number of false positives and improving overall system reliability.
3. **Ease of Integration:** YOLO11 offers seamless integration with various platforms and devices, including mobile and edge devices, which is crucial for modern AI applications.
4. **Flexibility:** Supports various tasks like object detection, segmentation, and tracking with configurable models to meet specific use-case requirements.

Check out Ultralytics [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/) for a deeper dive into its features and performance comparisons.

### Can I use YOLO11 for advanced applications like crowd analysis and traffic management?

Yes, Ultralytics YOLO11 is perfectly suited for advanced applications like crowd analysis and traffic management due to its real-time detection capabilities, scalability, and integration flexibility. Its advanced features allow for high-accuracy object tracking, counting, and classification in dynamic environments. Example use cases include:

- **Crowd Analysis:** Monitor and manage large gatherings, ensuring safety and optimizing crowd flow.
- **Traffic Management:** Track and count vehicles, analyze traffic patterns, and manage congestion in real-time.

For more information and implementation details, refer to the guide on [Real World Applications](#real-world-applications) of object counting with YOLO11.
