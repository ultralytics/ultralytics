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
        import ultralytics.solutions as sol
        import cv2
        
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # Configure heatmap
        sol.configure(view_img=True, names=model.names, heat_shape="circle", color_map=cv2.COLORMAP_PINK,
                      draw_tracks=False, counts_type="classwise",enable_counting=True)
        
        # Init heatmap
        heatmap_obj = sol.heatmap.Heatmap()
        
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
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
        import ultralytics.solutions as sol
        import cv2
        
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # Configure heatmap
        line_points = [(20, 400), (1080, 404)]
        sol.configure(view_img=True, region_pts=line_points, names=model.names,
                      draw_tracks=False, counts_type="line",enable_counting=True)
        
        # Init heatmap
        heatmap_obj = sol.heatmap.Heatmap()
        
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
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
        from ultralytics import YOLO
        import ultralytics.solutions as sol
        import cv2
        
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # Configure heatmap
        polygon_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]
        sol.configure(view_img=True, region_pts=polygon_points, names=model.names,
                      draw_tracks=False, counts_type="line",enable_counting=True)
        
        # Init heatmap
        heatmap_obj = sol.heatmap.Heatmap()
        
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
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
        import ultralytics.solutions as sol
        import cv2
        
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # Configure heatmap
        region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
        sol.configure(view_img=True, region_pts=region_points, names=model.names,
                      draw_tracks=True, counts_type="classwise", enable_counting=True)
        
        # Init heatmap
        heatmap_obj = sol.heatmap.Heatmap()
        
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
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
        import ultralytics.solutions as sol
        import cv2
        
        model = YOLO("yolov8n.pt")
        im0 = cv2.imread("path/to/image.png")  # path to image file

        # Configure heatmap
        sol.configure(view_img=True, names=model.names, draw_tracks=False, 
                      counts_type="classwise", enable_counting=False)

        # Init heatmap
        heatmap_obj = sol.heatmap.Heatmap()

        results = model.track(im0, persist=True, show=False)
        im0 = heatmap_obj.generate_heatmap(im0, tracks=results)
        cv2.imwrite("ultralytics_output.png", im0)
        ```

    === "Specific Classes"

        ```python
        from ultralytics import YOLO
        import ultralytics.solutions as sol
        import cv2
        
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # Configure heatmap
        classes_for_heatmap = [0, 2]  # classes for heatmap
        sol.configure(view_img=True, names=model.names, draw_tracks=False,
                      counts_type="classwise", enable_counting=True)
        
        # Init heatmap
        heatmap_obj = sol.heatmap.Heatmap()
        
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break
            tracks = model.track(im0, persist=True, show=False)
            im0 = heatmap_obj.generate_heatmap(im0, tracks, classes=classes_for_heatmap)
            video_writer.write(im0)
        
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

### Arguments `configure`

| Name               | Type           | Default                    | Description                                                       |
|--------------------|----------------|----------------------------|-------------------------------------------------------------------|
| `view_img`         | `bool`         | `False`                    | Display frames with counts                                        |
| `view_in_counts`   | `bool`         | `True`                     | Display in-counts only on video frame                             |
| `view_out_counts`  | `bool`         | `True`                     | Display out-counts only on video frame                            |
| `line_thickness`   | `int`          | `2`                        | Increase bounding boxes and count text thickness                  |
| `region_pts`       | `list`         | `[(20, 400), (1260, 400)]` | Points defining the region area                                   |
| `names`            | `dict`         | `model.model.names`        | Dictionary of classes names                                       |
| `draw_tracks`      | `bool`         | `False`                    | Enable drawing track lines                                        |
| `line_dist_thresh` | `int`          | `15`                       | Euclidean distance threshold for line counter                     |
| `txt_color`        | `tuple`        | `(255, 255, 255)`          | Foreground color for object counts text                           |
| `bg_color`         | `tuple`        | `(255, 255, 255)`          | Count highlighter color                                           |
| `counts_type`      | `str`          | `line`                     | counter type, "line" or "classwise"                               |
| `color_map`        | `cv2.COLORMAP` | `cv2.COLORMAP_JET`         | Heatmap colormap                                                  |
| `heat_shape`       | `str`          | `circle`                   | Heatmap shape, i.e "circle" or "rect"                             |
| `heat_decay`       | `float`        | `0.99`                     | Heatmap decay factor for heatmap area removal after specific time |
| `heat_alpha`       | `float`        | `0.55`                     | Heatmap alpha value                                               |
| `enable_counting`  | `bool`         | `True`                     | Enable object counting                                            |

### Arguments `model.track`

| Name      | Type    | Default        | Description                                                 |
|-----------|---------|----------------|-------------------------------------------------------------|
| `source`  | `im0`   | `None`         | source directory for images or videos                       |
| `persist` | `bool`  | `False`        | persisting tracks between frames                            |
| `tracker` | `str`   | `botsort.yaml` | Tracking method 'bytetrack' or 'botsort'                    |
| `conf`    | `float` | `0.3`          | Confidence threshold                                        |
| `iou`     | `float` | `0.5`          | IOU threshold                                               |
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
