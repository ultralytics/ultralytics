---
comments: true
description: Learn to create line graphs, bar plots, and pie charts using Python with guided instructions and code snippets. Maximize your data visualization skills!.
keywords: Ultralytics, YOLO11, data visualization, line graphs, bar plots, pie charts, Python, analytics, tutorial, guide
---

# Analytics using Ultralytics YOLO11

## Introduction

This guide provides a comprehensive overview of three fundamental types of [data visualizations](https://www.ultralytics.com/glossary/data-visualization): line graphs, bar plots, and pie charts. Each section includes step-by-step instructions and code snippets on how to create these visualizations using Python.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/tVuLIMt4DMY"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to generate Analytical Graphs using Ultralytics | Line Graphs, Bar Plots, Area and Pie Charts
</p>

### Visual Samples

|                                       Line Graph                                       |                                      Bar Plot                                      |                                      Pie Chart                                       |
| :------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------: |
| ![Line Graph](https://github.com/ultralytics/docs/releases/download/0/line-graph.avif) | ![Bar Plot](https://github.com/ultralytics/docs/releases/download/0/bar-plot.avif) | ![Pie Chart](https://github.com/ultralytics/docs/releases/download/0/pie-chart.avif) |

### Why Graphs are Important

- Line graphs are ideal for tracking changes over short and long periods and for comparing changes for multiple groups over the same period.
- Bar plots, on the other hand, are suitable for comparing quantities across different categories and showing relationships between a category and its numerical value.
- Lastly, pie charts are effective for illustrating proportions among categories and showing parts of a whole.

!!! example "Analytics Examples"

    === "CLI"

        ```bash
         yolo solutions analytics show=True

        # pass the source
        yolo solutions analytics source="path/to/video/file.mp4"

        # generate the pie chart
        yolo solutions analytics analytics_type="pie" show=True

        # generate the bar plots
        yolo solutions analytics analytics_type="bar" show=True

        # generate the area plots
        yolo solutions analytics analytics_type="area" show=True
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        out = cv2.VideoWriter(
            "ultralytics_analytics.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (1920, 1080),  # This is fixed
        )

        # Init analytics
        analytics = solutions.Analytics(
            show=True,  # Display the output
            analytics_type="line",  # Pass the analytics type, could be "pie", "bar" or "area".
            model="yolo11n.pt",  # Path to the YOLO11 model file
            # classes=[0, 2],  # If you want to count specific classes i.e person and car with COCO pretrained model.
        )

        # Process video
        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if success:
                frame_count += 1
                im0 = analytics.process_data(im0, frame_count)  # update analytics graph every frame
                out.write(im0)  # write the video file
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        ```

### Argument `Analytics`

Here's a table with the `Analytics` arguments:

| Name             | Type   | Default  | Description                                          |
| ---------------- | ------ | -------- | ---------------------------------------------------- |
| `analytics_type` | `str`  | `'line'` | Type of graph i.e "line", "bar", "area", "pie"       |
| `model`          | `str`  | `None`   | Path to Ultralytics YOLO Model File                  |
| `line_width`     | `int`  | `2`      | Line thickness for bounding boxes.                   |
| `show`           | `bool` | `False`  | Flag to control whether to display the video stream. |

### Arguments `model.track`

{% include "macros/track-args.md" %}

## Conclusion

Understanding when and how to use different types of visualizations is crucial for effective data analysis. Line graphs, bar plots, and pie charts are fundamental tools that can help you convey your data's story more clearly and effectively.

## FAQ

### How do I create a line graph using Ultralytics YOLO11 Analytics?

To create a line graph using Ultralytics YOLO11 Analytics, follow these steps:

1. Load a YOLO11 model and open your video file.
2. Initialize the `Analytics` class with the type set to "line."
3. Iterate through video frames, updating the line graph with relevant data, such as object counts per frame.
4. Save the output video displaying the line graph.

Example:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("Path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1920, 1080),  # This is fixed
)

analytics = solutions.Analytics(
    analytics_type="line",
    show=True,
)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        im0 = analytics.process_data(im0, frame_count)  # update analytics graph every frame
        out.write(im0)  # write the video file
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

For further details on configuring the `Analytics` class, visit the [Analytics using Ultralytics YOLO11 📊](#analytics-using-ultralytics-yolo11) section.

### What are the benefits of using Ultralytics YOLO11 for creating bar plots?

Using Ultralytics YOLO11 for creating bar plots offers several benefits:

1. **Real-time Data Visualization**: Seamlessly integrate [object detection](https://www.ultralytics.com/glossary/object-detection) results into bar plots for dynamic updates.
2. **Ease of Use**: Simple API and functions make it straightforward to implement and visualize data.
3. **Customization**: Customize titles, labels, colors, and more to fit your specific requirements.
4. **Efficiency**: Efficiently handle large amounts of data and update plots in real-time during video processing.

Use the following example to generate a bar plot:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("Path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1920, 1080),  # This is fixed
)

analytics = solutions.Analytics(
    analytics_type="bar",
    show=True,
)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        im0 = analytics.process_data(im0, frame_count)  # update analytics graph every frame
        out.write(im0)  # write the video file
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

To learn more, visit the [Bar Plot](#visual-samples) section in the guide.

### Why should I use Ultralytics YOLO11 for creating pie charts in my data visualization projects?

Ultralytics YOLO11 is an excellent choice for creating pie charts because:

1. **Integration with Object Detection**: Directly integrate object detection results into pie charts for immediate insights.
2. **User-Friendly API**: Simple to set up and use with minimal code.
3. **Customizable**: Various customization options for colors, labels, and more.
4. **Real-time Updates**: Handle and visualize data in real-time, which is ideal for video analytics projects.

Here's a quick example:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("Path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1920, 1080),  # This is fixed
)

analytics = solutions.Analytics(
    analytics_type="pie",
    show=True,
)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        im0 = analytics.process_data(im0, frame_count)  # update analytics graph every frame
        out.write(im0)  # write the video file
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

For more information, refer to the [Pie Chart](#visual-samples) section in the guide.

### Can Ultralytics YOLO11 be used to track objects and dynamically update visualizations?

Yes, Ultralytics YOLO11 can be used to track objects and dynamically update visualizations. It supports tracking multiple objects in real-time and can update various visualizations like line graphs, bar plots, and pie charts based on the tracked objects' data.

Example for tracking and updating a line graph:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("Path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1920, 1080),  # This is fixed
)

analytics = solutions.Analytics(
    analytics_type="line",
    show=True,
)

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        im0 = analytics.process_data(im0, frame_count)  # update analytics graph every frame
        out.write(im0)  # write the video file
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

To learn about the complete functionality, see the [Tracking](../modes/track.md) section.

### What makes Ultralytics YOLO11 different from other object detection solutions like [OpenCV](https://www.ultralytics.com/glossary/opencv) and [TensorFlow](https://www.ultralytics.com/glossary/tensorflow)?

Ultralytics YOLO11 stands out from other object detection solutions like OpenCV and TensorFlow for multiple reasons:

1. **State-of-the-art [Accuracy](https://www.ultralytics.com/glossary/accuracy)**: YOLO11 provides superior accuracy in object detection, segmentation, and classification tasks.
2. **Ease of Use**: User-friendly API allows for quick implementation and integration without extensive coding.
3. **Real-time Performance**: Optimized for high-speed inference, suitable for real-time applications.
4. **Diverse Applications**: Supports various tasks including multi-object tracking, custom model training, and exporting to different formats like ONNX, TensorRT, and CoreML.
5. **Comprehensive Documentation**: Extensive [documentation](https://docs.ultralytics.com/) and [blog resources](https://www.ultralytics.com/blog) to guide users through every step.

For more detailed comparisons and use cases, explore our [Ultralytics Blog](https://www.ultralytics.com/blog/ai-use-cases-transforming-your-future).
