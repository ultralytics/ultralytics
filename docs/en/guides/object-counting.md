---
comments: true
description: Learn to accurately identify and count objects in real-time using Ultralytics YOLO11 for applications like crowd analysis and surveillance.
keywords: object counting, YOLO11, Ultralytics, real-time object detection, AI, deep learning, object tracking, crowd analysis, surveillance, resource optimization
---

# Object Counting using Ultralytics YOLO11

## What is Object Counting?

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-count-the-objects-using-ultralytics-yolo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Object Counting In Colab"></a>

Object counting with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) involves accurate identification and counting of specific objects in videos and camera streams. YOLO11 excels in real-time applications, providing efficient and precise object counting for various scenarios like crowd analysis and surveillance, thanks to its state-of-the-art algorithms and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) capabilities.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/vKcD44GkSF8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Perform Real-Time Object Counting with Ultralytics YOLO11 🍏
</p>

## Advantages of Object Counting

- **Resource Optimization:** Object counting facilitates efficient resource management by providing accurate counts, optimizing resource allocation in applications like [inventory management](https://docs.ultralytics.com/guides/analytics/).
- **Enhanced Security:** Object counting enhances security and surveillance by accurately tracking and counting entities, aiding in proactive [threat detection](https://docs.ultralytics.com/guides/security-alarm-system/).
- **Informed Decision-Making:** Object counting offers valuable insights for decision-making, optimizing processes in retail, [traffic management](https://www.ultralytics.com/blog/ai-in-traffic-management-from-congestion-to-coordination), and various other domains.

## Real World Applications

|                                                                        Logistics                                                                        |                                                                         Aquaculture                                                                          |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Conveyor Belt Packets Counting Using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/conveyor-belt-packets-counting.avif) | ![Fish Counting in Sea using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/fish-counting-in-sea-using-ultralytics-yolov8.avif) |
|                                                 Conveyor Belt Packets Counting Using Ultralytics YOLO11                                                 |                                                        Fish Counting in Sea using Ultralytics YOLO11                                                         |

!!! example "Object Counting using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Run a counting example
        yolo solutions count show=True

        # Pass a source video
        yolo solutions count source="path/to/video.mp4"

        # Pass region coordinates
        yolo solutions count region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # region_points = [(20, 400), (1080, 400)]                                      # line counting
        region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangle region
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon region

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize object counter object
        counter = solutions.ObjectCounter(
            show=True,  # display the output
            region=region_points,  # pass region points
            model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
            # classes=[0, 2],  # count specific classes i.e. person and car with COCO pretrained model.
            # tracker="botsort.yaml",  # choose trackers i.e "bytetrack.yaml"
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = counter(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `ObjectCounter` Arguments

Here's a table with the `ObjectCounter` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "show_in", "show_out", "region"]) }}

The `ObjectCounter` solution allows the use of several `track` arguments:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the visualization arguments listed below are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

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

    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
    counter = solutions.ObjectCounter(show=True, region=region_points, model=model_path)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = counter(im0)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_objects_in_region("path/to/video.mp4", "output_video.avi", "yolo11n.pt")
```

For more advanced configurations and options, check out the [RegionCounter solution](https://docs.ultralytics.com/guides/region-counting/) for counting objects in multiple regions simultaneously.

### What are the advantages of using Ultralytics YOLO11 for object counting?

Using Ultralytics YOLO11 for object counting offers several advantages:

1. **Resource Optimization:** It facilitates efficient resource management by providing accurate counts, helping optimize resource allocation in industries like [inventory management](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management).
2. **Enhanced Security:** It enhances security and surveillance by accurately tracking and counting entities, aiding in proactive threat detection and [security systems](https://docs.ultralytics.com/guides/security-alarm-system/).
3. **Informed Decision-Making:** It offers valuable insights for decision-making, optimizing processes in domains like retail, traffic management, and more.
4. **Real-time Processing:** YOLO11's architecture enables [real-time inference](https://www.ultralytics.com/glossary/real-time-inference), making it suitable for live video streams and time-sensitive applications.

For implementation examples and practical applications, explore the [TrackZone solution](https://docs.ultralytics.com/guides/trackzone/) for tracking objects in specific zones.

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
            print("Video frame is empty or processing is complete.")
            break
        results = counter(im0)
        video_writer.write(results.plot_im)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


count_specific_classes("path/to/video.mp4", "output_specific_classes.avi", "yolo11n.pt", [0, 2])
```

In this example, `classes_to_count=[0, 2]` means it counts objects of class `0` and `2` (e.g., person and car in the COCO dataset). You can find more information about class indices in the [COCO dataset documentation](https://docs.ultralytics.com/datasets/detect/coco/).

### Why should I use YOLO11 over other [object detection](https://www.ultralytics.com/glossary/object-detection) models for real-time applications?

Ultralytics YOLO11 provides several advantages over other object detection models like [Faster R-CNN](https://docs.ultralytics.com/compare/yolo11-vs-efficientdet/), SSD, and previous YOLO versions:

1. **Speed and Efficiency:** YOLO11 offers real-time processing capabilities, making it ideal for applications requiring high-speed inference, such as surveillance and [autonomous driving](https://www.ultralytics.com/blog/ai-in-self-driving-cars).
2. **[Accuracy](https://www.ultralytics.com/glossary/accuracy):** It provides state-of-the-art accuracy for object detection and tracking tasks, reducing the number of false positives and improving overall system reliability.
3. **Ease of Integration:** YOLO11 offers seamless integration with various platforms and devices, including mobile and [edge devices](https://docs.ultralytics.com/guides/nvidia-jetson/), which is crucial for modern AI applications.
4. **Flexibility:** Supports various tasks like object detection, [segmentation](https://docs.ultralytics.com/tasks/segment/), and tracking with configurable models to meet specific use-case requirements.

Check out Ultralytics [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/) for a deeper dive into its features and performance comparisons.

### Can I use YOLO11 for advanced applications like crowd analysis and traffic management?

Yes, Ultralytics YOLO11 is perfectly suited for advanced applications like crowd analysis and traffic management due to its real-time detection capabilities, scalability, and integration flexibility. Its advanced features allow for high-accuracy object tracking, counting, and classification in dynamic environments. Example use cases include:

- **Crowd Analysis:** Monitor and manage large gatherings, ensuring safety and optimizing crowd flow with [region-based counting](https://docs.ultralytics.com/guides/region-counting/).
- **Traffic Management:** Track and count vehicles, analyze traffic patterns, and manage congestion in real-time with [speed estimation](https://docs.ultralytics.com/guides/speed-estimation/) capabilities.
- **Retail Analytics:** Analyze customer movement patterns and product interactions to optimize store layouts and improve customer experience.
- **Industrial Automation:** Count products on conveyor belts and monitor production lines for quality control and efficiency improvements.

For more specialized applications, explore [Ultralytics Solutions](https://docs.ultralytics.com/solutions/) for a comprehensive set of tools designed for real-world computer vision challenges.
