---
comments: true
description: Transform complex data into insightful heatmaps using Ultralytics YOLOv8. Discover patterns, trends, and anomalies with vibrant visualizations.
keywords: Ultralytics, YOLOv8, heatmaps, data visualization, data analysis, complex data, patterns, trends, anomalies
---

# Advanced [Data Visualization](https://www.ultralytics.com/glossary/data-visualization): Heatmaps using Ultralytics YOLOv8 🚀

## Introduction to Heatmaps

A heatmap generated with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) transforms complex data into a vibrant, color-coded matrix. This visual tool employs a spectrum of colors to represent varying data values, where warmer hues indicate higher intensities and cooler tones signify lower values. Heatmaps excel in visualizing intricate data patterns, correlations, and anomalies, offering an accessible and engaging approach to data interpretation across diverse domains.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/4ezde5-nZZw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Heatmaps using Ultralytics YOLOv8
</p>

## Why Choose Heatmaps for Data Analysis?

- **Intuitive Data Distribution Visualization:** Heatmaps simplify the comprehension of data concentration and distribution, converting complex datasets into easy-to-understand visual formats.
- **Efficient Pattern Detection:** By visualizing data in heatmap format, it becomes easier to spot trends, clusters, and outliers, facilitating quicker analysis and insights.
- **Enhanced Spatial Analysis and Decision-Making:** Heatmaps are instrumental in illustrating spatial relationships, aiding in decision-making processes in sectors such as business intelligence, environmental studies, and urban planning.

## Real World Applications

|                                                                    Transportation                                                                    |                                                                Retail                                                                |
| :--------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| ![Ultralytics YOLOv8 Transportation Heatmap](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-transportation-heatmap.avif) | ![Ultralytics YOLOv8 Retail Heatmap](https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-retail-heatmap.avif) |
|                                                      Ultralytics YOLOv8 Transportation Heatmap                                                       |                                                  Ultralytics YOLOv8 Retail Heatmap                                                   |

!!! tip "Heatmap Configuration"

    - `heatmap_alpha`: Ensure this value is within the range (0.0 - 1.0).
    - `decay_factor`: Used for removing heatmap after an object is no longer in the frame, its value should also be in the range (0.0 - 1.0).

!!! example "Heatmaps using Ultralytics YOLOv8 Example"

    === "Heatmap"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init heatmap
        heatmap_obj = solutions.Heatmap(
            colormap=cv2.COLORMAP_PARULA,
            view_img=True,
            shape="circle",
            names=model.names,
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            tracks = model.track(im0, persist=True, show=False)

            im0 = heatmap_obj.generate_heatmap(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "Line Counting"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        line_points = [(20, 400), (1080, 404)]  # line for object counting

        # Init heatmap
        heatmap_obj = solutions.Heatmap(
            colormap=cv2.COLORMAP_PARULA,
            view_img=True,
            shape="circle",
            count_reg_pts=line_points,
            names=model.names,
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            tracks = model.track(im0, persist=True, show=False)
            im0 = heatmap_obj.generate_heatmap(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "Polygon Counting"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Define polygon points
        region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]

        # Init heatmap
        heatmap_obj = solutions.Heatmap(
            colormap=cv2.COLORMAP_PARULA,
            view_img=True,
            shape="circle",
            count_reg_pts=region_points,
            names=model.names,
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            tracks = model.track(im0, persist=True, show=False)
            im0 = heatmap_obj.generate_heatmap(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "Region Counting"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Define region points
        region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

        # Init heatmap
        heatmap_obj = solutions.Heatmap(
            colormap=cv2.COLORMAP_PARULA,
            view_img=True,
            shape="circle",
            count_reg_pts=region_points,
            names=model.names,
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            tracks = model.track(im0, persist=True, show=False)
            im0 = heatmap_obj.generate_heatmap(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "Im0"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8s.pt")  # YOLOv8 custom/pretrained model

        im0 = cv2.imread("path/to/image.png")  # path to image file
        h, w = im0.shape[:2]  # image height and width

        # Heatmap Init
        heatmap_obj = solutions.Heatmap(
            colormap=cv2.COLORMAP_PARULA,
            view_img=True,
            shape="circle",
            names=model.names,
        )

        results = model.track(im0, persist=True)
        im0 = heatmap_obj.generate_heatmap(im0, tracks=results)
        cv2.imwrite("ultralytics_output.png", im0)
        ```

    === "Specific Classes"

        ```python
        import cv2

        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        classes_for_heatmap = [0, 2]  # classes for heatmap

        # Init heatmap
        heatmap_obj = solutions.Heatmap(
            colormap=cv2.COLORMAP_PARULA,
            view_img=True,
            shape="circle",
            names=model.names,
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            tracks = model.track(im0, persist=True, show=False, classes=classes_for_heatmap)

            im0 = heatmap_obj.generate_heatmap(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

### Arguments `Heatmap()`

| Name               | Type             | Default            | Description                                                       |
| ------------------ | ---------------- | ------------------ | ----------------------------------------------------------------- |
| `names`            | `list`           | `None`             | Dictionary of class names.                                        |
| `imw`              | `int`            | `0`                | Image width.                                                      |
| `imh`              | `int`            | `0`                | Image height.                                                     |
| `colormap`         | `int`            | `cv2.COLORMAP_JET` | Colormap to use for the heatmap.                                  |
| `heatmap_alpha`    | `float`          | `0.5`              | Alpha blending value for heatmap overlay.                         |
| `view_img`         | `bool`           | `False`            | Whether to display the image with the heatmap overlay.            |
| `view_in_counts`   | `bool`           | `True`             | Whether to display the count of objects entering the region.      |
| `view_out_counts`  | `bool`           | `True`             | Whether to display the count of objects exiting the region.       |
| `count_reg_pts`    | `list` or `None` | `None`             | Points defining the counting region (either a line or a polygon). |
| `count_txt_color`  | `tuple`          | `(0, 0, 0)`        | Text color for displaying counts.                                 |
| `count_bg_color`   | `tuple`          | `(255, 255, 255)`  | Background color for displaying counts.                           |
| `count_reg_color`  | `tuple`          | `(255, 0, 255)`    | Color for the counting region.                                    |
| `region_thickness` | `int`            | `5`                | Thickness of the region line.                                     |
| `line_dist_thresh` | `int`            | `15`               | Distance threshold for line-based counting.                       |
| `line_thickness`   | `int`            | `2`                | Thickness of the lines used in drawing.                           |
| `decay_factor`     | `float`          | `0.99`             | Decay factor for the heatmap to reduce intensity over time.       |
| `shape`            | `str`            | `"circle"`         | Shape of the heatmap blobs ('circle' or 'rect').                  |

### Arguments `model.track`

{% include "macros/track-args.md" %}

### Heatmap COLORMAPs

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

## FAQ

### How does Ultralytics YOLOv8 generate heatmaps and what are their benefits?

Ultralytics YOLOv8 generates heatmaps by transforming complex data into a color-coded matrix where different hues represent data intensities. Heatmaps make it easier to visualize patterns, correlations, and anomalies in the data. Warmer hues indicate higher values, while cooler tones represent lower values. The primary benefits include intuitive visualization of data distribution, efficient pattern detection, and enhanced spatial analysis for decision-making. For more details and configuration options, refer to the [Heatmap Configuration](#arguments-heatmap) section.

### Can I use Ultralytics YOLOv8 to perform object tracking and generate a heatmap simultaneously?

Yes, Ultralytics YOLOv8 supports object tracking and heatmap generation concurrently. This can be achieved through its `Heatmap` solution integrated with object tracking models. To do so, you need to initialize the heatmap object and use YOLOv8's tracking capabilities. Here's a simple example:

```python
import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("path/to/video/file.mp4")
heatmap_obj = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, view_img=True, shape="circle", names=model.names)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    tracks = model.track(im0, persist=True, show=False)
    im0 = heatmap_obj.generate_heatmap(im0, tracks)
    cv2.imshow("Heatmap", im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

For further guidance, check the [Tracking Mode](../modes/track.md) page.

### What makes Ultralytics YOLOv8 heatmaps different from other data visualization tools like those from [OpenCV](https://www.ultralytics.com/glossary/opencv) or Matplotlib?

Ultralytics YOLOv8 heatmaps are specifically designed for integration with its [object detection](https://www.ultralytics.com/glossary/object-detection) and tracking models, providing an end-to-end solution for real-time data analysis. Unlike generic visualization tools like OpenCV or Matplotlib, YOLOv8 heatmaps are optimized for performance and automated processing, supporting features like persistent tracking, decay factor adjustment, and real-time video overlay. For more information on YOLOv8's unique features, visit the [Ultralytics YOLOv8 Introduction](https://www.ultralytics.com/blog/introducing-ultralytics-yolov8).

### How can I visualize only specific object classes in heatmaps using Ultralytics YOLOv8?

You can visualize specific object classes by specifying the desired classes in the `track()` method of the YOLO model. For instance, if you only want to visualize cars and persons (assuming their class indices are 0 and 2), you can set the `classes` parameter accordingly.

```python
import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("path/to/video/file.mp4")
heatmap_obj = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, view_img=True, shape="circle", names=model.names)

classes_for_heatmap = [0, 2]  # Classes to visualize
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    tracks = model.track(im0, persist=True, show=False, classes=classes_for_heatmap)
    im0 = heatmap_obj.generate_heatmap(im0, tracks)
    cv2.imshow("Heatmap", im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

### Why should businesses choose Ultralytics YOLOv8 for heatmap generation in data analysis?

Ultralytics YOLOv8 offers seamless integration of advanced object detection and real-time heatmap generation, making it an ideal choice for businesses looking to visualize data more effectively. The key advantages include intuitive data distribution visualization, efficient pattern detection, and enhanced spatial analysis for better decision-making. Additionally, YOLOv8's cutting-edge features such as persistent tracking, customizable colormaps, and support for various export formats make it superior to other tools like [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) and OpenCV for comprehensive data analysis. Learn more about business applications at [Ultralytics Plans](https://www.ultralytics.com/plans).
