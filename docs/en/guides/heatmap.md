---
comments: true
description: Heatmaps Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Heatmap, Object Tracking, Notebook, IPython Kernel, CLI, Python SDK
---

# Heatmaps using Ultralytics YOLOv8 🚀

## What is Heatmap?
A heatmap with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) is a visual representation of data where values are depicted using colors. It presents a two-dimensional matrix of data, with colors indicating the intensity of values. Warm colors represent higher values, while cool colors represent lower ones. Heatmaps are widely used to visualize patterns, trends, and relationships in datasets across various fields. They offer a user-friendly and customizable way to quickly interpret complex information.

## Advantages of Heatmaps?

- **Visualizing Data Distribution:** Heatmaps offer an intuitive way to represent the concentration and distribution of data, making complex information easy to grasp.
- **Pattern Recognition:** Heatmaps help identify patterns, clusters, and anomalies in data, aiding in the discovery of trends and irregularities.
- **Spatial Relationships and Decision Support:** They effectively visualize spatial relationships and provide decision support by highlighting key areas of interest, making them valuable in various fields, including business intelligence and analytics.
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

        model = YOLO("yolov8s.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        if not cap.isOpened():
            print("Error reading video file")
            exit(0)

        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_CIVIDIS,
                             imw=cap.get(4),  # should same as im0 width
                             imh=cap.get(3),  # should same as im0 height
                             view_img=True)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                exit(0)
            results = model.track(im0, persist=True)
            frame = heatmap_obj.generate_heatmap(im0, tracks=results)

        ```

    === "Heatmap with Specific Classes"
        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import heatmap
        import cv2

        model = YOLO("yolov8s.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        if not cap.isOpened():
            print("Error reading video file")
            exit(0)

        classes_for_heatmap = [0, 2]

        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_CIVIDIS,
                             imw=cap.get(4),  # should same as im0 width
                             imh=cap.get(3),  # should same as im0 height
                             view_img=True)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                exit(0)
            results = model.track(im0, persist=True,
                                  classes=classes_for_heatmap)
            frame = heatmap_obj.generate_heatmap(im0, tracks=results)

        ```

    === "Heatmap with Save Output"
        ```python
        from ultralytics import YOLO
        import heatmap
        import cv2

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        if not cap.isOpened():
            print("Error reading video file")
            exit(0)

        video_writer = cv2.VideoWriter("heatmap_output.avi",
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       int(cap.get(5)),
                                       (int(cap.get(3)), int(cap.get(4))))

        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_CIVIDIS,
                             imw=cap.get(4),  # should same as im0 width
                             imh=cap.get(3),  # should same as im0 height
                             view_img=True)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                exit(0)
            results = model.track(im0, persist=True)
            frame = heatmap_obj.generate_heatmap(im0, tracks=results)
            video_writer.write(im0)

        video_writer.release()
        ```

### Arguments `set_args`

| Name          | Type           | Default | Description                    |
|---------------|----------------|---------|--------------------------------|
| view_img      | `bool`         | `False` | Display the frame with heatmap |
| colormap      | `cv2.COLORMAP` | `None`  | cv2.COLORMAP for heatmap       |
| imw           | `int`          | `None`  | Width of Heatmap               |
| imh           | `int`          | `None`  | Height of Heatmap              |
| heatmap_alpha | `float`        | `0.5`   | Heatmap alpha value            |


### Arguments `model.track`

| Name      | Type    | Default        | Description                                                 |
|-----------|---------|----------------|-------------------------------------------------------------|
| `source`  | `im0`   | `None`         | source directory for images or videos                       |
| `persist` | `bool`  | `False`        | persisting tracks between frames                            |
| `tracker` | `str`   | `botsort.yaml` | Tracking method 'bytetrack' or 'botsort'                    |
| `conf`    | `float` | `0.3`          | Confidence Threshold                                        |
| `iou`     | `float` | `0.5`          | IOU Threshold                                               |
| `classes` | `list`  | `None`         | filter results by class, i.e. classes=0, or classes=[0,2,3] |
