---
comments: true
description: Transform complex data into insightful heatmaps using Ultralytics YOLO11. Discover patterns, trends, and anomalies with vibrant visualizations.
keywords: Ultralytics, YOLO11, heatmaps, data visualization, data analysis, complex data, patterns, trends, anomalies
---

# Advanced [Data Visualization](https://www.ultralytics.com/glossary/data-visualization): Heatmaps using Ultralytics YOLO11 🚀

## Introduction to Heatmaps

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-generate-heatmaps-using-ultralytics-yolo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Heatmaps In Colab"></a>

A heatmap generated with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) transforms complex data into a vibrant, color-coded matrix. This visual tool employs a spectrum of colors to represent varying data values, where warmer hues indicate higher intensities and cooler tones signify lower values. Heatmaps excel in visualizing intricate data patterns, correlations, and anomalies, offering an accessible and engaging approach to data interpretation across diverse domains.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/4ezde5-nZZw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Heatmaps using Ultralytics YOLO11
</p>

## Why Choose Heatmaps for Data Analysis?

- **Intuitive Data Distribution Visualization:** Heatmaps simplify the comprehension of data concentration and distribution, converting complex datasets into easy-to-understand visual formats.
- **Efficient Pattern Detection:** By visualizing data in heatmap format, it becomes easier to spot trends, clusters, and outliers, facilitating quicker analysis and insights.
- **Enhanced Spatial Analysis and Decision-Making:** Heatmaps are instrumental in illustrating spatial relationships, aiding in decision-making processes in sectors such as business intelligence, environmental studies, and urban planning.

## Real World Applications

|                                                                    Transportation                                                                    |                                                                Retail                                                                |
| :--------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics YOLO11 Transportation Heatmap](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-transportation-heatmap.avif) | ![Ultralytics YOLO11 Retail Heatmap](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-retail-heatmap.avif) |
|                                                      Ultralytics YOLO11 Transportation Heatmap                                                       |                                                  Ultralytics YOLO11 Retail Heatmap                                                   |

!!! example "Heatmaps using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Run a heatmap example
        yolo solutions heatmap show=True

        # Pass a source video
        yolo solutions heatmap source="path/to/video.mp4"

        # Pass a custom colormap
        yolo solutions heatmap colormap=cv2.COLORMAP_INFERNO

        # Heatmaps + object counting
        yolo solutions heatmap region="[(20, 400), (1080, 400), (1080, 360), (20, 360)]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # For object counting with heatmap, you can pass region points.
        # region_points = [(20, 400), (1080, 400)]                                      # line points
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]              # rectangle region
        # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]   # polygon points

        # Initialize heatmap object
        heatmap = solutions.Heatmap(
            show=True,  # display the output
            model="yolo11n.pt",  # path to the YOLO11 model file
            colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap
            # region=region_points,  # object counting with heatmaps, you can pass region_points
            # classes=[0, 2],  # generate heatmap for specific classes, e.g., person and car.
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = heatmap(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `Heatmap()` Arguments

Here's a table with the `Heatmap` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "colormap", "show_in", "show_out", "region"]) }}

You can also apply different `track` arguments in the `Heatmap` solution.

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the supported visualization arguments are listed below:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

#### Heatmap COLORMAPs

| Colormap Name                   | Description                            |
| ------------------------------- | -------------------------------------- |
| `cv::COLORMAP_AUTUMN`           | Autumn color map                       |
| `cv::COLORMAP_BONE`             | Bone color map                         |
| `cv::COLORMAP_JET`              | Jet color map                          |
| `cv::COLORMAP_WINTER`           | Winter color map                       |
| `cv::COLORMAP_RAINBOW`          | Rainbow color map                      |
| `cv::COLORMAP_OCEAN`            | Ocean color map                        |
| `cv::COLORMAP_SUMMER`           | Summer color map                       |
| `cv::COLORMAP_SPRING`           | Spring color map                       |
| `cv::COLORMAP_COOL`             | Cool color map                         |
| `cv::COLORMAP_HSV`              | HSV (Hue, Saturation, Value) color map |
| `cv::COLORMAP_PINK`             | Pink color map                         |
| `cv::COLORMAP_HOT`              | Hot color map                          |
| `cv::COLORMAP_PARULA`           | Parula color map                       |
| `cv::COLORMAP_MAGMA`            | Magma color map                        |
| `cv::COLORMAP_INFERNO`          | Inferno color map                      |
| `cv::COLORMAP_PLASMA`           | Plasma color map                       |
| `cv::COLORMAP_VIRIDIS`          | Viridis color map                      |
| `cv::COLORMAP_CIVIDIS`          | Cividis color map                      |
| `cv::COLORMAP_TWILIGHT`         | Twilight color map                     |
| `cv::COLORMAP_TWILIGHT_SHIFTED` | Shifted Twilight color map             |
| `cv::COLORMAP_TURBO`            | Turbo color map                        |
| `cv::COLORMAP_DEEPGREEN`        | Deep Green color map                   |

These colormaps are commonly used for visualizing data with different color representations.

## How Heatmaps Work in Ultralytics YOLO11

The [Heatmap solution](../reference/solutions/heatmap.md) in Ultralytics YOLO11 extends the [ObjectCounter](../reference/solutions/object_counter.md) class to generate and visualize movement patterns in video streams. When initialized, the solution creates a blank heatmap layer that gets updated as objects move through the frame.

For each detected object, the solution:

1. Tracks the object across frames using YOLO11's tracking capabilities
2. Updates the heatmap intensity at the object's location
3. Applies a selected colormap to visualize the intensity values
4. Overlays the colored heatmap on the original frame

The result is a dynamic visualization that builds up over time, revealing traffic patterns, crowd movements, or other spatial behaviors in your video data.

## FAQ

### How does Ultralytics YOLO11 generate heatmaps and what are their benefits?

Ultralytics YOLO11 generates heatmaps by transforming complex data into a color-coded matrix where different hues represent data intensities. Heatmaps make it easier to visualize patterns, correlations, and anomalies in the data. Warmer hues indicate higher values, while cooler tones represent lower values. The primary benefits include intuitive visualization of data distribution, efficient pattern detection, and enhanced spatial analysis for decision-making. For more details and configuration options, refer to the [Heatmap Configuration](#heatmap-arguments) section.

### Can I use Ultralytics YOLO11 to perform object tracking and generate a heatmap simultaneously?

Yes, Ultralytics YOLO11 supports object tracking and heatmap generation concurrently. This can be achieved through its `Heatmap` solution integrated with object tracking models. To do so, you need to initialize the heatmap object and use YOLO11's tracking capabilities. Here's a simple example:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, show=True, model="yolo11n.pt")

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = heatmap(im0)
cap.release()
cv2.destroyAllWindows()
```

For further guidance, check the [Tracking Mode](../modes/track.md) page.

### What makes Ultralytics YOLO11 heatmaps different from other data visualization tools like those from [OpenCV](https://www.ultralytics.com/glossary/opencv) or Matplotlib?

Ultralytics YOLO11 heatmaps are specifically designed for integration with its [object detection](https://www.ultralytics.com/glossary/object-detection) and tracking models, providing an end-to-end solution for real-time data analysis. Unlike generic visualization tools like OpenCV or Matplotlib, YOLO11 heatmaps are optimized for performance and automated processing, supporting features like persistent tracking, decay factor adjustment, and real-time video overlay. For more information on YOLO11's unique features, visit the [Ultralytics YOLO11 Introduction](https://www.ultralytics.com/blog/introducing-ultralytics-yolov8).

### How can I visualize only specific object classes in heatmaps using Ultralytics YOLO11?

You can visualize specific object classes by specifying the desired classes in the `track()` method of the YOLO model. For instance, if you only want to visualize cars and persons (assuming their class indices are 0 and 2), you can set the `classes` parameter accordingly.

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
heatmap = solutions.Heatmap(show=True, model="yolo11n.pt", classes=[0, 2])

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = heatmap(im0)
cap.release()
cv2.destroyAllWindows()
```

### Why should businesses choose Ultralytics YOLO11 for heatmap generation in data analysis?

Ultralytics YOLO11 offers seamless integration of advanced object detection and real-time heatmap generation, making it an ideal choice for businesses looking to visualize data more effectively. The key advantages include intuitive data distribution visualization, efficient pattern detection, and enhanced spatial analysis for better decision-making. Additionally, YOLO11's cutting-edge features such as persistent tracking, customizable colormaps, and support for various export formats make it superior to other tools like [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) and OpenCV for comprehensive data analysis. Learn more about business applications at [Ultralytics Plans](https://www.ultralytics.com/plans).
