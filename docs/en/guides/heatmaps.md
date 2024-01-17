---
comments: true
description: Advanced Data Visualization with Ultralytics YOLOv8 Heatmaps
keywords: Ultralytics, YOLOv8, Advanced Data Visualization, Heatmap Technology, Object Detection and Tracking, Jupyter Notebook, Python SDK, Command Line Interface
---

# Advanced Data Visualization: Heatmaps using Ultralytics YOLOv8 🚀

## Introduction to Heatmaps

A heatmap generated with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) transforms complex data into a vibrant, color-coded matrix. This visual tool employs a spectrum of colors to represent varying data values, where warmer hues indicate higher intensities and cooler tones signify lower values. Heatmaps excel in visualizing intricate data patterns, correlations, and anomalies, offering an accessible and engaging approach to data interpretation across diverse domains.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/4ezde5-nZZw"
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

|                                                                 Transportation                                                                  |                                                                 Retail                                                                  |
|:-----------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|
| ![Ultralytics YOLOv8 Transportation Heatmap](https://github.com/RizwanMunawar/ultralytics/assets/62513924/288d7053-622b-4452-b4e4-1f41aeb764aa) | ![Ultralytics YOLOv8 Retail Heatmap](https://github.com/RizwanMunawar/ultralytics/assets/62513924/edef75ad-50a7-4c0a-be4a-a66cdfc12802) |
|                                                    Ultralytics YOLOv8 Transportation Heatmap                                                    |                                                    Ultralytics YOLOv8 Retail Heatmap                                                    |

!!! tip "Heatmap Configuration"

    - `heatmap_alpha`: Ensure this value is within the range (0.0 - 1.0).
    - `decay_factor`: Used for removing heatmap after an object is no longer in the frame, its value should also be in the range (0.0 - 1.0).

!!! Example "Heatmaps using Ultralytics YOLOv8 Example"

    === "Heatmap"

        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import heatmap
        import cv2

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("heatmap_output.avi",
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps,
                                       (w, h))

        # Init heatmap
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                             imw=w,
                             imh=h,
                             view_img=True,
                             shape="circle")

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
        from ultralytics import YOLO
        from ultralytics.solutions import heatmap
        import cv2

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("heatmap_output.avi",
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps,
                                       (w, h))

        line_points = [(20, 400), (1080, 404)]  # line for object counting

        # Init heatmap
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                             imw=w,
                             imh=h,
                             view_img=True,
                             shape="circle",
                             count_reg_pts=line_points)

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
        from ultralytics import YOLO
        from ultralytics.solutions import heatmap
        import cv2

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("heatmap_output.avi",
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps,
                                       (w, h))

        # Define region points
        region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

        # Init heatmap
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                             imw=w,
                             imh=h,
                             view_img=True,
                             shape="circle",
                             count_reg_pts=region_points)

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
        from ultralytics import YOLO
        from ultralytics.solutions import heatmap
        import cv2

        model = YOLO("yolov8s.pt")   # YOLOv8 custom/pretrained model

        im0 = cv2.imread("path/to/image.png")  # path to image file

        # Heatmap Init
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                                     imw=im0.shape[0],  # should same as im0 width
                                     imh=im0.shape[1],  # should same as im0 height
                                     view_img=True,
                                     shape="circle")


        results = model.track(im0, persist=True)
        im0 = heatmap_obj.generate_heatmap(im0, tracks=results)
        cv2.imwrite("ultralytics_output.png", im0)
        ```

    === "Specific Classes"

        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import heatmap
        import cv2

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Video writer
        video_writer = cv2.VideoWriter("heatmap_output.avi",
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps,
                                       (w, h))

        classes_for_heatmap = [0, 2]  # classes for heatmap

        # Init heatmap
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                             imw=w,
                             imh=h,
                             view_img=True,
                             shape="circle")

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            tracks = model.track(im0, persist=True, show=False,
                                 classes=classes_for_heatmap)

            im0 = heatmap_obj.generate_heatmap(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

### Arguments `set_args`

| Name                  | Type           | Default           | Description                                               |
|-----------------------|----------------|-------------------|-----------------------------------------------------------|
| `view_img`            | `bool`         | `False`           | Display the frame with heatmap                            |
| `colormap`            | `cv2.COLORMAP` | `None`            | cv2.COLORMAP for heatmap                                  |
| `imw`                 | `int`          | `None`            | Width of Heatmap                                          |
| `imh`                 | `int`          | `None`            | Height of Heatmap                                         |
| `heatmap_alpha`       | `float`        | `0.5`             | Heatmap alpha value                                       |
| `count_reg_pts`       | `list`         | `None`            | Object counting region points                             |
| `count_txt_thickness` | `int`          | `2`               | Count values text size                                    |
| `count_txt_color`     | `RGB Color`    | `(0, 0, 0)`       | Foreground color for Object counts text                   |
| `count_color`         | `RGB Color`    | `(255, 255, 255)` | Background color for Object counts text                   |
| `count_reg_color`     | `RGB Color`    | `(255, 0, 255)`   | Counting region color                                     |
| `region_thickness`    | `int`          | `5`               | Counting region thickness value                           |
| `decay_factor`        | `float`        | `0.99`            | Decay factor for heatmap area removal after specific time |
| `shape`               | `str`          | `circle`          | Heatmap shape for display "rect" or "circle" supported    |
| `line_dist_thresh`    | `int`          | `15`              | Euclidean Distance threshold for line counter             |

### Arguments `model.track`

| Name      | Type    | Default        | Description                                                 |
|-----------|---------|----------------|-------------------------------------------------------------|
| `source`  | `im0`   | `None`         | source directory for images or videos                       |
| `persist` | `bool`  | `False`        | persisting tracks between frames                            |
| `tracker` | `str`   | `botsort.yaml` | Tracking method 'bytetrack' or 'botsort'                    |
| `conf`    | `float` | `0.3`          | Confidence Threshold                                        |
| `iou`     | `float` | `0.5`          | IOU Threshold                                               |
| `classes` | `list`  | `None`         | filter results by class, i.e. classes=0, or classes=[0,2,3] |

### Heatmap COLORMAPs

| Colormap Name                   | Description                            |
|---------------------------------|----------------------------------------|
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
