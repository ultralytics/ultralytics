---
comments: true
description: Learn to create line graphs, bar plots, and pie charts using Python with guided instructions and code snippets. Maximize your data visualization skills!
keywords: Ultralytics, YOLO26, data visualization, line graphs, bar plots, pie charts, Python, analytics, tutorial, guide
---

# Analytics using Ultralytics YOLO26

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

|                                                              Line Graph                                                              |                                                             Bar Plot                                                              |                                                               Pie Chart                                                               |
| :----------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------: |
| ![YOLO analytics line graph for object tracking](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/analytics-line-graph.avif) | ![YOLO analytics bar plot for detection counts](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/analytics-bar-plot.avif) | ![YOLO analytics pie chart for class distribution](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/analytics-pie-chart.avif) |

### Why Graphs are Important

- Line graphs are ideal for tracking changes over short and long periods and for comparing changes for multiple groups over the same period.
- Bar plots, on the other hand, are suitable for comparing quantities across different categories and showing relationships between a category and its numerical value.
- Lastly, pie charts are effective for illustrating proportions among categories and showing parts of a whole.

!!! example "Analytics using Ultralytics YOLO"

    === "CLI"

        ```bash
        yolo solutions analytics show=True

        # Pass the source
        yolo solutions analytics source="path/to/video.mp4"

        # Generate the pie chart
        yolo solutions analytics analytics_type="pie" show=True

        # Generate the bar plots
        yolo solutions analytics analytics_type="bar" show=True

        # Generate the area plots
        yolo solutions analytics analytics_type="area" show=True
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(
            "analytics_output.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (1280, 720),  # this is fixed
        )

        # Initialize analytics object
        analytics = solutions.Analytics(
            show=True,  # display the output
            analytics_type="line",  # pass the analytics type, could be "pie", "bar" or "area".
            model="yolo26n.pt",  # path to the YOLO26 model file
            # classes=[0, 2],  # display analytics for specific detection classes
        )

        # Process video
        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if success:
                frame_count += 1
                results = analytics(im0, frame_count)  # update analytics graph every frame

                # print(results)  # access the output

                out.write(results.plot_im)  # write the video file
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `Analytics` Arguments

Here's a table outlining the Analytics arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "analytics_type"]) }}

You can also leverage different [`track`](../modes/track.md) arguments in the `Analytics` solution.

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the following visualization arguments are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## Conclusion

Understanding when and how to use different types of visualizations is crucial for effective data analysis. Line graphs, bar plots, and pie charts are fundamental tools that can help you convey your data's story more clearly and effectively. The Ultralytics YOLO26 Analytics solution provides a streamlined way to generate these visualizations from your [object detection](https://www.ultralytics.com/glossary/object-detection) and tracking results, making it easier to extract meaningful insights from your visual data.

## FAQ

### How do I create a line graph using Ultralytics YOLO26 Analytics?

To create a line graph using Ultralytics YOLO26 Analytics, follow these steps:

1. Load a YOLO26 model and open your video file.
2. Initialize the `Analytics` class with the type set to "line."
3. Iterate through video frames, updating the line graph with relevant data, such as object counts per frame.
4. Save the output video displaying the line graph.

Example:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),  # this is fixed
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
        results = analytics(im0, frame_count)  # update analytics graph every frame
        out.write(results.plot_im)  # write the video file
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

For further details on configuring the `Analytics` class, visit the [Analytics using Ultralytics YOLO26](#analytics-using-ultralytics-yolo26) section.

### What are the benefits of using Ultralytics YOLO26 for creating bar plots?

Using Ultralytics YOLO26 for creating bar plots offers several benefits:

1. **Real-time Data Visualization**: Seamlessly integrate [object detection](https://www.ultralytics.com/glossary/object-detection) results into bar plots for dynamic updates.
2. **Ease of Use**: Simple API and functions make it straightforward to implement and visualize data.
3. **Customization**: Customize titles, labels, colors, and more to fit your specific requirements.
4. **Efficiency**: Efficiently handle large amounts of data and update plots in real-time during video processing.

Use the following example to generate a bar plot:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),  # this is fixed
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
        results = analytics(im0, frame_count)  # update analytics graph every frame
        out.write(results.plot_im)  # write the video file
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

To learn more, visit the [Bar Plot](#visual-samples) section in the guide.

### Why should I use Ultralytics YOLO26 for creating pie charts in my data visualization projects?

Ultralytics YOLO26 is an excellent choice for creating pie charts because:

1. **Integration with Object Detection**: Directly integrate object detection results into pie charts for immediate insights.
2. **User-Friendly API**: Simple to set up and use with minimal code.
3. **Customizable**: Various customization options for colors, labels, and more.
4. **Real-time Updates**: Handle and visualize data in real-time, which is ideal for video analytics projects.

Here's a quick example:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),  # this is fixed
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
        results = analytics(im0, frame_count)  # update analytics graph every frame
        out.write(results.plot_im)  # write the video file
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

For more information, refer to the [Pie Chart](#visual-samples) section in the guide.

### Can Ultralytics YOLO26 be used to track objects and dynamically update visualizations?

Yes, Ultralytics YOLO26 can be used to track objects and dynamically update visualizations. It supports tracking multiple objects in real-time and can update various visualizations like line graphs, bar plots, and pie charts based on the tracked objects' data.

Example for tracking and updating a line graph:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "ultralytics_analytics.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),  # this is fixed
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
        results = analytics(im0, frame_count)  # update analytics graph every frame
        out.write(results.plot_im)  # write the video file
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

To learn about the complete functionality, see the [Tracking](../modes/track.md) section.

### What makes Ultralytics YOLO26 different from other object detection solutions like [OpenCV](https://www.ultralytics.com/glossary/opencv) and [TensorFlow](https://www.ultralytics.com/glossary/tensorflow)?

Ultralytics YOLO26 stands out from other object detection solutions like OpenCV and TensorFlow for multiple reasons:

1. **State-of-the-art [Accuracy](https://www.ultralytics.com/glossary/accuracy)**: YOLO26 provides superior accuracy in object detection, segmentation, and classification tasks.
2. **Ease of Use**: User-friendly API allows for quick implementation and integration without extensive coding.
3. **Real-time Performance**: Optimized for high-speed inference, suitable for real-time applications.
4. **Diverse Applications**: Supports various tasks including multi-object tracking, custom model training, and exporting to different formats like ONNX, TensorRT, and CoreML.
5. **Comprehensive Documentation**: Extensive [documentation](https://docs.ultralytics.com/) and [blog resources](https://www.ultralytics.com/blog) to guide users through every step.

For more detailed comparisons and use cases, explore our [Ultralytics Blog](https://www.ultralytics.com/blog/ai-use-cases-transforming-your-future).
