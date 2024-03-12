---
comments: true
description: Object Counting Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Object Counting, Object Tracking, Notebook, IPython Kernel, CLI, Python SDK
---

# Object Counting using Ultralytics YOLOv8 🚀

## What is Object Counting?

Object counting with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) involves accurate identification and counting of specific objects in videos and camera streams. YOLOv8 excels in real-time applications, providing efficient and precise object counting for various scenarios like crowd analysis and surveillance, thanks to its state-of-the-art algorithms and deep learning capabilities.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Ag2e-5_NpS0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Counting using Ultralytics YOLOv8
</p>

## Advantages of Object Counting?

- **Resource Optimization:** Object counting facilitates efficient resource management by providing accurate counts, and optimizing resource allocation in applications like inventory management.
- **Enhanced Security:** Object counting enhances security and surveillance by accurately tracking and counting entities, aiding in proactive threat detection.
- **Informed Decision-Making:** Object counting offers valuable insights for decision-making, optimizing processes in retail, traffic management, and various other domains.

## Real World Applications

|                                                                           Logistics                                                                           |                                                                     Aquaculture                                                                     |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![Conveyor Belt Packets Counting Using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/70e2d106-510c-4c6c-a57a-d34a765aa757) | ![Fish Counting in Sea using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/c60d047b-3837-435f-8d29-bb9fc95d2191) |
|                                                    Conveyor Belt Packets Counting Using Ultralytics YOLOv8                                                    |                                                    Fish Counting in Sea using Ultralytics YOLOv8                                                    |

!!! Example "Object Counting using YOLOv8 Example"

    === "Count in Region"

        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import object_counter
        import cv2

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define region points
        region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

        # Init Object Counter
        counter = object_counter.ObjectCounter()
        counter.set_args(view_img=True,
                         reg_pts=region_points,
                         classes_names=model.names,
                         draw_tracks=True)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            tracks = model.track(im0, persist=True, show=False)

            im0 = counter.start_counting(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```
    
    === "Count in Polygon"

        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import object_counter
        import cv2
        
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        # Define region points as a polygon with 5 points
        region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]
        
        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))
        
        # Init Object Counter
        counter = object_counter.ObjectCounter()
        counter.set_args(view_img=True,
                         reg_pts=region_points,
                         classes_names=model.names,
                         draw_tracks=True)
        
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            tracks = model.track(im0, persist=True, show=False)
        
            im0 = counter.start_counting(im0, tracks)
            video_writer.write(im0)
        
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```
    
    === "Count in Line"

        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import object_counter
        import cv2

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define line points
        line_points = [(20, 400), (1080, 400)]

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

        # Init Object Counter
        counter = object_counter.ObjectCounter()
        counter.set_args(view_img=True,
                         reg_pts=line_points,
                         classes_names=model.names,
                         draw_tracks=True)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            tracks = model.track(im0, persist=True, show=False)

            im0 = counter.start_counting(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "Specific Classes"

        ```python
        from ultralytics import YOLO
        from ultralytics.solutions import object_counter
        import cv2

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        line_points = [(20, 400), (1080, 400)]  # line or region points
        classes_to_count = [0, 2]  # person and car classes for count

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

        # Init Object Counter
        counter = object_counter.ObjectCounter()
        counter.set_args(view_img=True,
                         reg_pts=line_points,
                         classes_names=model.names,
                         draw_tracks=True)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            tracks = model.track(im0, persist=True, show=False,
                                 classes=classes_to_count)

            im0 = counter.start_counting(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

???+ tip "Region is Movable"

    You can move the region anywhere in the frame by clicking on its edges

### Optional Arguments `set_args`

| Name                  | Type        | Default                    | Description                                   |
|-----------------------|-------------|----------------------------|-----------------------------------------------|
| `view_img`            | `bool`      | `False`                    | Display frames with counts                    |
| `view_in_counts`      | `bool`      | `True`                     | Display in-counts only on video frame         |
| `view_out_counts`     | `bool`      | `True`                     | Display out-counts only on video frame        |
| `line_thickness`      | `int`       | `2`                        | Increase bounding boxes thickness             |
| `reg_pts`             | `list`      | `[(20, 400), (1260, 400)]` | Points defining the Region Area               |
| `classes_names`       | `dict`      | `model.model.names`        | Dictionary of Class Names                     |
| `region_color`        | `RGB Color` | `(255, 0, 255)`            | Color of the Object counting Region or Line   |
| `track_thickness`     | `int`       | `2`                        | Thickness of Tracking Lines                   |
| `draw_tracks`         | `bool`      | `False`                    | Enable drawing Track lines                    |
| `track_color`         | `RGB Color` | `(0, 255, 0)`              | Color for each track line                     |
| `line_dist_thresh`    | `int`       | `15`                       | Euclidean Distance threshold for line counter |
| `count_txt_thickness` | `int`       | `2`                        | Thickness of Object counts text               |
| `count_txt_color`     | `RGB Color` | `(0, 0, 0)`                | Foreground color for Object counts text       |
| `count_color`         | `RGB Color` | `(255, 255, 255)`          | Background color for Object counts text       |
| `region_thickness`    | `int`       | `5`                        | Thickness for object counter region or line   |

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
