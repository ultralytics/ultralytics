---
comments: true
description: Learn to create line graphs, bar plots, and pie charts using Python with guided instructions and code snippets. Maximize your data visualization skills!.
keywords: Ultralytics, YOLOv8, data visualization, line graphs, bar plots, pie charts, Python, analytics, tutorial, guide
---

# Analytics using Ultralytics YOLOv8

## Introduction

This guide provides a comprehensive overview of three fundamental types of [data visualizations](https://www.ultralytics.com/glossary/data-visualization): line graphs, bar plots, and pie charts. Each section includes step-by-step instructions and code snippets on how to create these visualizations using Python.

### Visual Samples

|                                       Line Graph                                       |                                      Bar Plot                                      |                                      Pie Chart                                       |
| :------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------: |
| ![Line Graph](https://github.com/ultralytics/docs/releases/download/0/line-graph.avif) | ![Bar Plot](https://github.com/ultralytics/docs/releases/download/0/bar-plot.avif) | ![Pie Chart](https://github.com/ultralytics/docs/releases/download/0/pie-chart.avif) |

### Why Graphs are Important

- Line graphs are ideal for tracking changes over short and long periods and for comparing changes for multiple groups over the same period.
- Bar plots, on the other hand, are suitable for comparing quantities across different categories and showing relationships between a category and its numerical value.
- Lastly, pie charts are effective for illustrating proportions among categories and showing parts of a whole.

!!! analytics "Analytics Examples"

    === "Line Graph"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8s.pt")

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("line_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        analytics = solutions.Analytics(
            type="line",
            writer=out,
            im0_shape=(w, h),
            view_img=True,
        )
        total_counts = 0
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                frame_count += 1
                results = model.track(frame, persist=True, verbose=True)

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    for box in boxes:
                        total_counts += 1

                analytics.update_line(frame_count, total_counts)

                total_counts = 0
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        ```

    === "Multiple Lines"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8s.pt")

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter("multiple_line_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        analytics = solutions.Analytics(
            type="line",
            writer=out,
            im0_shape=(w, h),
            view_img=True,
            max_points=200,
        )

        frame_count = 0
        data = {}
        labels = []

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                frame_count += 1

                results = model.track(frame, persist=True)

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    clss = results[0].boxes.cls.cpu().tolist()

                    for box, track_id, cls in zip(boxes, track_ids, clss):
                        # Store each class label
                        if model.names[int(cls)] not in labels:
                            labels.append(model.names[int(cls)])

                        # Store each class count
                        if model.names[int(cls)] in data:
                            data[model.names[int(cls)]] += 1
                        else:
                            data[model.names[int(cls)]] = 0

                # update lines every frame
                analytics.update_multiple_lines(data, labels, frame_count)
                data = {}  # clear the data list for next frame
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        ```

    === "Pie Chart"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8s.pt")

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("pie_chart.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        analytics = solutions.Analytics(
            type="pie",
            writer=out,
            im0_shape=(w, h),
            view_img=True,
        )

        clswise_count = {}

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model.track(frame, persist=True, verbose=True)
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    clss = results[0].boxes.cls.cpu().tolist()
                    for box, cls in zip(boxes, clss):
                        if model.names[int(cls)] in clswise_count:
                            clswise_count[model.names[int(cls)]] += 1
                        else:
                            clswise_count[model.names[int(cls)]] = 1

                    analytics.update_pie(clswise_count)
                    clswise_count = {}

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        ```

    === "Bar Plot"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8s.pt")

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("bar_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        analytics = solutions.Analytics(
            type="bar",
            writer=out,
            im0_shape=(w, h),
            view_img=True,
        )

        clswise_count = {}

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model.track(frame, persist=True, verbose=True)
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    clss = results[0].boxes.cls.cpu().tolist()
                    for box, cls in zip(boxes, clss):
                        if model.names[int(cls)] in clswise_count:
                            clswise_count[model.names[int(cls)]] += 1
                        else:
                            clswise_count[model.names[int(cls)]] = 1

                    analytics.update_bar(clswise_count)
                    clswise_count = {}

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        ```

    === "Area chart"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8s.pt")

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("area_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        analytics = solutions.Analytics(
            type="area",
            writer=out,
            im0_shape=(w, h),
            view_img=True,
        )

        clswise_count = {}
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame_count += 1
                results = model.track(frame, persist=True, verbose=True)

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    clss = results[0].boxes.cls.cpu().tolist()

                    for box, cls in zip(boxes, clss):
                        if model.names[int(cls)] in clswise_count:
                            clswise_count[model.names[int(cls)]] += 1
                        else:
                            clswise_count[model.names[int(cls)]] = 1

                analytics.update_area(frame_count, clswise_count)
                clswise_count = {}
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        ```

### Argument `Analytics`

Here's a table with the `Analytics` arguments:

| Name           | Type              | Default       | Description                                                                      |
| -------------- | ----------------- | ------------- | -------------------------------------------------------------------------------- |
| `type`         | `str`             | `None`        | Type of data or object.                                                          |
| `im0_shape`    | `tuple`           | `None`        | Shape of the initial image.                                                      |
| `writer`       | `cv2.VideoWriter` | `None`        | Object for writing video files.                                                  |
| `title`        | `str`             | `ultralytics` | Title for the visualization.                                                     |
| `x_label`      | `str`             | `x`           | Label for the x-axis.                                                            |
| `y_label`      | `str`             | `y`           | Label for the y-axis.                                                            |
| `bg_color`     | `str`             | `white`       | Background color.                                                                |
| `fg_color`     | `str`             | `black`       | Foreground color.                                                                |
| `line_color`   | `str`             | `yellow`      | Color of the lines.                                                              |
| `line_width`   | `int`             | `2`           | Width of the lines.                                                              |
| `fontsize`     | `int`             | `13`          | Font size for text.                                                              |
| `view_img`     | `bool`            | `False`       | Flag to display the image or video.                                              |
| `save_img`     | `bool`            | `True`        | Flag to save the image or video.                                                 |
| `max_points`   | `int`             | `50`          | For multiple lines, total points drawn on frame, before deleting initial points. |
| `points_width` | `int`             | `15`          | Width of line points highlighter.                                                |

### Arguments `model.track`

{% include "macros/track-args.md" %}

## Conclusion

Understanding when and how to use different types of visualizations is crucial for effective data analysis. Line graphs, bar plots, and pie charts are fundamental tools that can help you convey your data's story more clearly and effectively.

## FAQ

### How do I create a line graph using Ultralytics YOLOv8 Analytics?

To create a line graph using Ultralytics YOLOv8 Analytics, follow these steps:

1. Load a YOLOv8 model and open your video file.
2. Initialize the `Analytics` class with the type set to "line."
3. Iterate through video frames, updating the line graph with relevant data, such as object counts per frame.
4. Save the output video displaying the line graph.

Example:

```python
import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("Path/to/video/file.mp4")
out = cv2.VideoWriter("line_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

analytics = solutions.Analytics(type="line", writer=out, im0_shape=(w, h), view_img=True)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)
        total_counts = sum([1 for box in results[0].boxes.xyxy])
        analytics.update_line(frame_count, total_counts)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

For further details on configuring the `Analytics` class, visit the [Analytics using Ultralytics YOLOv8 📊](#analytics-using-ultralytics-yolov8) section.

### What are the benefits of using Ultralytics YOLOv8 for creating bar plots?

Using Ultralytics YOLOv8 for creating bar plots offers several benefits:

1. **Real-time Data Visualization**: Seamlessly integrate [object detection](https://www.ultralytics.com/glossary/object-detection) results into bar plots for dynamic updates.
2. **Ease of Use**: Simple API and functions make it straightforward to implement and visualize data.
3. **Customization**: Customize titles, labels, colors, and more to fit your specific requirements.
4. **Efficiency**: Efficiently handle large amounts of data and update plots in real-time during video processing.

Use the following example to generate a bar plot:

```python
import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("Path/to/video/file.mp4")
out = cv2.VideoWriter("bar_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

analytics = solutions.Analytics(type="bar", writer=out, im0_shape=(w, h), view_img=True)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)
        clswise_count = {
            model.names[int(cls)]: boxes.size(0)
            for cls, boxes in zip(results[0].boxes.cls.tolist(), results[0].boxes.xyxy)
        }
        analytics.update_bar(clswise_count)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

To learn more, visit the [Bar Plot](#visual-samples) section in the guide.

### Why should I use Ultralytics YOLOv8 for creating pie charts in my data visualization projects?

Ultralytics YOLOv8 is an excellent choice for creating pie charts because:

1. **Integration with Object Detection**: Directly integrate object detection results into pie charts for immediate insights.
2. **User-Friendly API**: Simple to set up and use with minimal code.
3. **Customizable**: Various customization options for colors, labels, and more.
4. **Real-time Updates**: Handle and visualize data in real-time, which is ideal for video analytics projects.

Here's a quick example:

```python
import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("Path/to/video/file.mp4")
out = cv2.VideoWriter("pie_chart.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

analytics = solutions.Analytics(type="pie", writer=out, im0_shape=(w, h), view_img=True)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)
        clswise_count = {
            model.names[int(cls)]: boxes.size(0)
            for cls, boxes in zip(results[0].boxes.cls.tolist(), results[0].boxes.xyxy)
        }
        analytics.update_pie(clswise_count)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

For more information, refer to the [Pie Chart](#visual-samples) section in the guide.

### Can Ultralytics YOLOv8 be used to track objects and dynamically update visualizations?

Yes, Ultralytics YOLOv8 can be used to track objects and dynamically update visualizations. It supports tracking multiple objects in real-time and can update various visualizations like line graphs, bar plots, and pie charts based on the tracked objects' data.

Example for tracking and updating a line graph:

```python
import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("Path/to/video/file.mp4")
out = cv2.VideoWriter("line_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

analytics = solutions.Analytics(type="line", writer=out, im0_shape=(w, h), view_img=True)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)
        total_counts = sum([1 for box in results[0].boxes.xyxy])
        analytics.update_line(frame_count, total_counts)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

To learn about the complete functionality, see the [Tracking](../modes/track.md) section.

### What makes Ultralytics YOLOv8 different from other object detection solutions like [OpenCV](https://www.ultralytics.com/glossary/opencv) and [TensorFlow](https://www.ultralytics.com/glossary/tensorflow)?

Ultralytics YOLOv8 stands out from other object detection solutions like OpenCV and TensorFlow for multiple reasons:

1. **State-of-the-art [Accuracy](https://www.ultralytics.com/glossary/accuracy)**: YOLOv8 provides superior accuracy in object detection, segmentation, and classification tasks.
2. **Ease of Use**: User-friendly API allows for quick implementation and integration without extensive coding.
3. **Real-time Performance**: Optimized for high-speed inference, suitable for real-time applications.
4. **Diverse Applications**: Supports various tasks including multi-object tracking, custom model training, and exporting to different formats like ONNX, TensorRT, and CoreML.
5. **Comprehensive Documentation**: Extensive [documentation](https://docs.ultralytics.com/) and [blog resources](https://www.ultralytics.com/blog) to guide users through every step.

For more detailed comparisons and use cases, explore our [Ultralytics Blog](https://www.ultralytics.com/blog/ai-use-cases-transforming-your-future).
