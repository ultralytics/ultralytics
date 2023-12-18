---
comments: true
description: Advanced Data Visualization with Ultralytics YOLOv8 Heatmaps
keywords: Ultralytics, YOLOv8, Advanced Data Visualization, Heatmap Technology, Object Detection and Tracking, Jupyter Notebook, Python SDK, Command Line Interface
---

# Advanced Data Visualization: Heatmaps using Ultralytics YOLOv8 🚀

## Introduction to Heatmaps

A heatmap generated with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) transforms complex data into a vibrant, color-coded matrix. This visual tool employs a spectrum of colors to represent varying data values, where warmer hues indicate higher intensities and cooler tones signify lower values. Heatmaps excel in visualizing intricate data patterns, correlations, and anomalies, offering an accessible and engaging approach to data interpretation across diverse domains.

## Why Choose Heatmaps for Data Analysis?

- **Intuitive Data Distribution Visualization:** Heatmaps simplify the comprehension of data concentration and distribution, converting complex datasets into easy-to-understand visual formats.
- **Efficient Pattern Detection:** By visualizing data in heatmap format, it becomes easier to spot trends, clusters, and outliers, facilitating quicker analysis and insights.
- **Enhanced Spatial Analysis and Decision Making:** Heatmaps are instrumental in illustrating spatial relationships, aiding in decision-making processes in sectors such as business intelligence, environmental studies, and urban planning.

## Real World Applications

|                                                                 Transportation                                                                  |                                                                 Retail                                                                  |
|:-----------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------:|
| ![Ultralytics YOLOv8 Transportation Heatmap](https://github.com/RizwanMunawar/ultralytics/assets/62513924/50d197b8-c7f6-4ecf-a664-3d4363b073de) | ![Ultralytics YOLOv8 Retail Heatmap](https://github.com/RizwanMunawar/ultralytics/assets/62513924/ffd0649f-5ff5-48d2-876d-6bdffeff5c54) |
|                                                    Ultralytics YOLOv8 Transportation Heatmap                                                    |                                                    Ultralytics YOLOv8 Retail Heatmap                                                    |

???+ tip "heatmap_alpha"

    heatmap_alpha value should be in range (0.0 - 1.0)

!!! Example "Heatmap Example"

    === "Heatmap"
        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import heatmap
        import cv2

        model = YOLO("yolov8s.pt")   # YOLOv8 custom/pretrained model
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Heatmap Init
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_CIVIDIS,
                              imw=cap.get(4),  # should same as cap width
                              imh=cap.get(3),  # should same as cap height
                              view_img=True)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
              print("Video frame is empty or video processing has been successfully completed.")
              break

            results = model.track(im0, persist=True)
            im0 = heatmap_obj.generate_heatmap(im0, tracks=results)

        cv2.destroyAllWindows()
        ```

    === "Heatmap with im0"
        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import heatmap
        import cv2

        model = YOLO("yolov8s.pt")   # YOLOv8 custom/pretrained model

        im0 = cv2.imread("path/to/image.png")  # path to image file

        # Heatmap Init
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_JET,
                             imw=im0.shape[0],  # should same as im0 width
                             imh=im0.shape[1],  # should same as im0 height
                             view_img=True)


        results = model.track(im0, persist=True)
        im0 = heatmap_obj.generate_heatmap(im0, tracks=results)
        cv2.imwrite("ultralytics_output.png", im0)
        ```

    === "Heatmap with Specific Classes"
        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import heatmap
        import cv2

        model = YOLO("yolov8s.pt")   # YOLOv8 custom/pretrained model
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        classes_for_heatmap = [0, 2]

        # Heatmap init
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_CIVIDIS,
                              imw=cap.get(4),  # should same as cap width
                              imh=cap.get(3),  # should same as cap height
                              view_img=True)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
              print("Video frame is empty or video processing has been successfully completed.")
              break
            results = model.track(im0, persist=True, classes=classes_for_heatmap)
            im0 = heatmap_obj.generate_heatmap(im0, tracks=results)

        cv2.destroyAllWindows()
        ```

    === "Heatmap with Save Output"
        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import heatmap
        import cv2

        model = YOLO("yolov8s.pt")   # YOLOv8 custom/pretrained model
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        video_writer = cv2.VideoWriter("heatmap_output.avi",
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       int(cap.get(5)),
                                       (int(cap.get(3)), int(cap.get(4))))

        # Heatmap init
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_CIVIDIS,
                             imw=cap.get(4),  # should same as cap width
                             imh=cap.get(3),  # should same as cap height
                             view_img=True)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
              print("Video frame is empty or video processing has been successfully completed.")
              break
            results = model.track(im0, persist=True)
            im0 = heatmap_obj.generate_heatmap(im0, tracks=results)
            video_writer.write(im0)

        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "Heatmap with Object Counting"
        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import heatmap
        import cv2

        model = YOLO("yolov8s.pt")   # YOLOv8 custom/pretrained model

        cap = cv2.VideoCapture("path/to/video/file.mp4")  # Video file Path, webcam 0
        assert cap.isOpened(), "Error reading video file"

        # Region for object counting
        count_reg_pts = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

        # Heatmap Init
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_JET,
                              imw=cap.get(4),  # should same as cap width
                              imh=cap.get(3),  # should same as cap height
                              view_img=True,
                              count_reg_pts=count_reg_pts)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
              print("Video frame is empty or video processing has been successfully completed.")
              break
            results = model.track(im0, persist=True)
            im0 = heatmap_obj.generate_heatmap(im0, tracks=results)

        cv2.destroyAllWindows()

        ```

### Arguments `set_args`

| Name                | Type           | Default         | Description                     |
|---------------------|----------------|-----------------|---------------------------------|
| view_img            | `bool`         | `False`         | Display the frame with heatmap  |
| colormap            | `cv2.COLORMAP` | `None`          | cv2.COLORMAP for heatmap        |
| imw                 | `int`          | `None`          | Width of Heatmap                |
| imh                 | `int`          | `None`          | Height of Heatmap               |
| heatmap_alpha       | `float`        | `0.5`           | Heatmap alpha value             |
| count_reg_pts       | `list`         | `None`          | Object counting region points   |
| count_txt_thickness | `int`          | `2`             | Count values text size          |
| count_reg_color     | `tuple`        | `(255, 0, 255)` | Counting region color           |
| region_thickness    | `int`          | `5`             | Counting region thickness value |

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
