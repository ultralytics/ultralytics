---
title: Data Analytics with Ultralytics YOLO26
comments: true
description: Build real-time line graphs, bar plots, pie charts, and area plots from YOLO26 object detection and tracking data in Python to visualize counts frame by frame.
keywords: Ultralytics, YOLO26, data visualization, line graphs, bar plots, pie charts, area plots, object tracking analytics, real-time analytics, Python, computer vision
---

# Analytics using Ultralytics YOLO26

Analytics with [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) turns [object detection](https://www.ultralytics.com/glossary/object-detection) and tracking results into real-time charts, so you can watch how object counts change across a video frame by frame. This guide covers four [data visualization](https://www.ultralytics.com/glossary/data-visualization) types — line graphs, bar plots, pie charts, and area plots — and shows how to switch between them with shared Python and CLI examples.

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

## Visual Samples

|                                                              Line Graph                                                              |                                                             Bar Plot                                                              |                                                               Pie Chart                                                               |
| :----------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------: |
| ![YOLO analytics line graph for object tracking](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/analytics-line-graph.avif) | ![YOLO analytics bar plot for detection counts](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/analytics-bar-plot.avif) | ![YOLO analytics pie chart for class distribution](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/analytics-pie-chart.avif) |

## Why Visualize Detection Data?

- **Line graphs** are ideal for tracking changes over short and long periods and for comparing changes for multiple groups over the same period.
- **Bar plots** are suitable for comparing quantities across different categories and showing relationships between a category and its numerical value.
- **Pie charts** are effective for illustrating proportions among categories and showing parts of a whole.
- **Area plots** fill the line graph so per-class object counts over time are easier to read at a glance.

## Generate Analytics Graphs

Pass your video to the `Analytics` solution and select a chart with `analytics_type`. The solution runs detection and tracking on every frame and renders a 1280×720 chart (by default) you can write straight to an output video. Switch between `"line"`, `"bar"`, `"pie"`, and `"area"` with a single argument.

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
2. Initialize the `Analytics` class with `analytics_type="line"`.
3. Iterate through video frames, calling the solution each frame to update the line graph with data such as object counts.
4. Write `results.plot_im` to an output video to save the chart.

Use the [Python example above](#generate-analytics-graphs) as a starting point — it already runs the full frame loop, and a line graph is the default `analytics_type`.

### What are the benefits of using Ultralytics YOLO26 for creating bar plots?

Using Ultralytics YOLO26 for creating bar plots offers several benefits:

1. **Real-time Data Visualization**: Seamlessly integrate [object detection](https://www.ultralytics.com/glossary/object-detection) results into bar plots for dynamic updates.
2. **Ease of Use**: Simple API and functions make it straightforward to implement and visualize data.
3. **Customization**: Customize titles, labels, colors, and more to fit your specific requirements.
4. **Efficiency**: Efficiently handle large amounts of data and update plots in real-time during video processing.

To generate a bar plot, set `analytics_type="bar"` in the [Python example above](#generate-analytics-graphs) — the rest of the frame loop is identical. See the [Visual Samples](#visual-samples) section for a preview.

### Why should I use Ultralytics YOLO26 for creating pie charts in my data visualization projects?

Ultralytics YOLO26 is an excellent choice for creating pie charts because:

1. **Integration with Object Detection**: Directly integrate object detection results into pie charts for immediate insights.
2. **User-Friendly API**: Simple to set up and use with minimal code.
3. **Customizable**: Various customization options for colors, labels, and more.
4. **Real-time Updates**: Handle and visualize data in real-time, which is ideal for video analytics projects.

To generate a pie chart, set `analytics_type="pie"` in the [Python example above](#generate-analytics-graphs). For more information, refer to the [Visual Samples](#visual-samples) section in the guide.

### Can Ultralytics YOLO26 be used to track objects and dynamically update visualizations?

Yes. Tracking is built into the `Analytics` solution: it tracks multiple objects in real time and updates the chart from the tracked objects' data every frame, so line graphs, bar plots, pie charts, and area plots all reflect live counts. This is exactly what the frame loop in the [Python example above](#generate-analytics-graphs) does. To learn about the underlying tracking functionality, see the [Tracking](../modes/track.md) section.

### What makes Ultralytics YOLO26 different from other object detection solutions like [OpenCV](https://www.ultralytics.com/glossary/opencv) and [TensorFlow](https://www.ultralytics.com/glossary/tensorflow)?

Ultralytics YOLO26 stands out from other object detection solutions like OpenCV and TensorFlow for multiple reasons:

1. **State-of-the-art [Accuracy](https://www.ultralytics.com/glossary/accuracy)**: YOLO26 provides superior accuracy in [object detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), and [classification](../tasks/classify.md) tasks.
2. **Ease of Use**: User-friendly API allows for quick implementation and integration without extensive coding.
3. **Real-time Performance**: Optimized for high-speed inference, suitable for real-time applications.
4. **Diverse Applications**: Supports various tasks including multi-object tracking, custom model training, and exporting to different formats like ONNX, TensorRT, and CoreML.
5. **Comprehensive Documentation**: Extensive [documentation](../index.md) and [blog resources](https://www.ultralytics.com/blog) to guide users through every step.

For more detailed comparisons and use cases, explore our [Ultralytics Blog](https://www.ultralytics.com/blog/ai-use-cases-transforming-your-future).
