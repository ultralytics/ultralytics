---
comments: true
description: Object Counting Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Object Counting, Object Tracking, Notebook, IPython Kernel, CLI, Python SDK
---

# Object Counting using Ultralytics YOLOv8 🚀

## What is Object Counting?

Object counting with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) involves accurate identification and counting of specific objects in videos and camera streams. YOLOv8 excels in real-time applications, providing efficient and precise object counting for various scenarios like crowd analysis and surveillance, thanks to its state-of-the-art algorithms and deep learning capabilities.

<table>
  <tr>
    <td align="center">
      <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Ag2e-5_NpS0"
        title="YouTube video player" frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen>
      </iframe>
      <br>
      <strong>Watch:</strong> Object Counting using Ultralytics YOLOv8
    </td>
    <td align="center">
      <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Fj9TStNBVoY"
        title="YouTube video player" frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen>
      </iframe>
      <br>
      <strong>Watch:</strong> Classwise Object Counting using Ultralytics YOLOv8
    </td>
  </tr>
</table>

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
        import cv2
        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define region points
        region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init Object Counter
        counter = solutions.ObjectCounter(
            view_img=True,
            reg_pts=region_points,
            classes_names=model.names,
            draw_tracks=True,
            line_thickness=2,
        )

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
        import cv2
        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define region points as a polygon with 5 points
        region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init Object Counter
        counter = solutions.ObjectCounter(
            view_img=True,
            reg_pts=region_points,
            classes_names=model.names,
            draw_tracks=True,
            line_thickness=2,
        )

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
        import cv2
        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define line points
        line_points = [(20, 400), (1080, 400)]

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init Object Counter
        counter = solutions.ObjectCounter(
            view_img=True,
            reg_pts=line_points,
            classes_names=model.names,
            draw_tracks=True,
            line_thickness=2,
        )

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
        import cv2
        from ultralytics import YOLO, solutions

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        line_points = [(20, 400), (1080, 400)]  # line or region points
        classes_to_count = [0, 2]  # person and car classes for count

        # Video writer
        video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Init Object Counter
        counter = solutions.ObjectCounter(
            view_img=True,
            reg_pts=line_points,
            classes_names=model.names,
            draw_tracks=True,
            line_thickness=2,
        )

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

            im0 = counter.start_counting(im0, tracks)
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

???+ tip "Region is Movable"

    You can move the region anywhere in the frame by clicking on its edges

### Argument `ObjectCounter`

Here's a table with the `ObjectCounter` arguments:

| Name                 | Type    | Default                    | Description                                                            |
|----------------------|---------|----------------------------|------------------------------------------------------------------------|
| `classes_names`      | `dict`  | `None`                     | Dictionary of class names.                                             |
| `reg_pts`            | `list`  | `[(20, 400), (1260, 400)]` | List of points defining the counting region.                           |
| `count_reg_color`    | `tuple` | `(255, 0, 255)`            | RGB color of the counting region.                                      |
| `count_txt_color`    | `tuple` | `(0, 0, 0)`                | RGB color of the count text.                                           |
| `count_bg_color`     | `tuple` | `(255, 255, 255)`          | RGB color of the count text background.                                |
| `line_thickness`     | `int`   | `2`                        | Line thickness for bounding boxes.                                     |
| `track_thickness`    | `int`   | `2`                        | Thickness of the track lines.                                          |
| `view_img`           | `bool`  | `False`                    | Flag to control whether to display the video stream.                   |
| `view_in_counts`     | `bool`  | `True`                     | Flag to control whether to display the in counts on the video stream.  |
| `view_out_counts`    | `bool`  | `True`                     | Flag to control whether to display the out counts on the video stream. |
| `draw_tracks`        | `bool`  | `False`                    | Flag to control whether to draw the object tracks.                     |
| `track_color`        | `tuple` | `None`                     | RGB color of the tracks.                                               |
| `region_thickness`   | `int`   | `5`                        | Thickness of the object counting region.                               |
| `line_dist_thresh`   | `int`   | `15`                       | Euclidean distance threshold for line counter.                         |
| `cls_txtdisplay_gap` | `int`   | `50`                       | Display gap between each class count.                                  |

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
