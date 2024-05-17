---
comments: true
description: Distance Calculation Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Distance Calculation, Object Tracking, Notebook, IPython Kernel, CLI, Python SDK
---

# Distance Calculation using Ultralytics YOLOv8 🚀

## What is Distance Calculation?

Measuring the gap between two objects is known as distance calculation within a specified space. In the case of [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), the bounding box centroid is employed to calculate the distance for bounding boxes highlighted by the user.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/LE8am1QoVn4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Distance Calculation using Ultralytics YOLOv8
</p>

## Visuals

|                                                  Distance Calculation using Ultralytics YOLOv8                                                  |                                                                
|:-----------------------------------------------------------------------------------------------------------------------------------------------:|
| ![Ultralytics YOLOv8 Distance Calculation](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/6b6b735d-3c49-4b84-a022-2bf6e3c72f8b) |

## Advantages of Distance Calculation?

- **Localization Precision:** Enhances accurate spatial positioning in computer vision tasks.
- **Size Estimation:** Allows estimation of physical sizes for better contextual understanding.
- **Scene Understanding:** Contributes to a 3D understanding of the environment for improved decision-making.

???+ tip "Distance Calculation"

    - Click on any two bounding boxes with Left Mouse click for distance calculation

!!! Example "Distance Calculation using YOLOv8 Example"

    === "Video Stream"

        ```python
        from ultralytics import YOLO
        import ultralytics.solutions as sol
        import cv2
        
        model = YOLO("yolov8n.pt")
        names = model.model.names
        
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        sol.configure(names=model.names, pixels_per_meter=10, view_img=True)
        
        # Init distance-calculation obj
        dist_obj = sol.distance_calculation.DistanceCalculation()
        
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break
            tracks = model.track(im0, persist=True, verbose=False)
            im0 = dist_obj.calculate_distance(im0, tracks)
            video_writer.write(im0)
        
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

???+ tip "Note"

    - Mouse Right Click will delete all drawn points
    - Mouse Left Click can be used to draw points

### Optional Arguments `configure`

| Name               | Type    | Default             | Description                       |
|--------------------|---------|---------------------|-----------------------------------|
| `view_img`         | `bool`  | `False`             | Display frames with counts        |
| `line_thickness`   | `int`   | `2`                 | Increase bounding boxes thickness |
| `names`            | `dict`  | `model.model.names` | Dictionary of classes names       |
| `txt_color`        | `tuple` | `(255, 255, 255)`   | Distance display foreground color |
| `bg_color`         | `tuple` | `(255, 255, 255)`   | Distance display background color |
| `pixels_per_meter` | `int`   | `10`                | Pixel per meter                   |

### Arguments `model.track`

| Name      | Type    | Default        | Description                                                 |
|-----------|---------|----------------|-------------------------------------------------------------|
| `source`  | `im0`   | `None`         | source directory for images or videos                       |
| `persist` | `bool`  | `False`        | persisting tracks between frames                            |
| `tracker` | `str`   | `botsort.yaml` | Tracking method 'bytetrack' or 'botsort'                    |
| `conf`    | `float` | `0.3`          | Confidence Threshold                                        |
| `iou`     | `float` | `0.5`          | IOU Threshold                                               |
| `classes` | `list`  | `None`         | filter results by class, i.e. classes=0, or classes=[0,2,3] |
| `verbose` | `bool`  | `True`         | Display the object tracking results                         |
