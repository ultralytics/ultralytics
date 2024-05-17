---
comments: true
description: Queue Management Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Queue Management, Object Counting, Object Tracking, Object Detection, Notebook, IPython Kernel, CLI, Python SDK
---

# Queue Management using Ultralytics YOLOv8 🚀

## What is Queue Management?

Queue management using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) involves organizing and controlling lines of people or vehicles to reduce wait times and enhance efficiency. It's about optimizing queues to improve customer satisfaction and system performance in various settings like retail, banks, airports, and healthcare facilities.

## Advantages of Queue Management?

- **Reduced Waiting Times:** Queue management systems efficiently organize queues, minimizing wait times for customers. This leads to improved satisfaction levels as customers spend less time waiting and more time engaging with products or services.
- **Increased Efficiency:** Implementing queue management allows businesses to allocate resources more effectively. By analyzing queue data and optimizing staff deployment, businesses can streamline operations, reduce costs, and improve overall productivity.

## Real World Applications

|                                                                                  Logistics                                                                                  |                                                                           Retail                                                                           |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![Queue management at airport ticket counter using Ultralytics YOLOv8](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/10487e76-bf60-4a9c-a0f3-5a75a05fa7a3) | ![Queue monitoring in crowd using Ultralytics YOLOv8](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/dcc6d2ca-5576-434d-83c6-e57fe07bc693) |
|                                                     Queue management at airport ticket counter Using Ultralytics YOLOv8                                                     |                                                        Queue monitoring in crowd Ultralytics YOLOv8                                                        |

!!! Example "Queue Management using YOLOv8 Example"

    === "Queue Manager"

        ```python
        from ultralytics import YOLO
        import ultralytics.solutions as sol
        import cv2
        
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        video_writer = cv2.VideoWriter("queue_output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        queue_region = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
        
        # Configure queue
        sol.configure(view_img=True, names=model.names, 
                      draw_tracks=True, region_pts=queue_region)
        
        # Init queue
        queue = sol.queue_management.QueueManager()
        
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break
            tracks = model.track(im0, show=False, persist=True)
            out = queue.process_queue(im0, tracks)
            video_writer.write(im0)
        
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

    === "Queue Manager Specific Classes"

        ```python
        from ultralytics import YOLO
        import ultralytics.solutions as sol
        import cv2
        
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        
        video_writer = cv2.VideoWriter("queue_output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        queue_region = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
        
        # Configure queue
        classes_to_queue = [0, 2]
        sol.configure(view_img=True, names=model.names,
                      draw_tracks=True, region_pts=queue_region)
        
        # Init queue
        queue = sol.queue_management.QueueManager()
        
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break
            tracks = model.track(im0, show=False, persist=True, classes=classes_to_queue)
            out = queue.process_queue(im0, tracks)
            video_writer.write(im0)
        
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

### Optional Arguments `set_args`

| Name              | Type        | Default                    | Description                                 |
|-------------------|-------------|----------------------------|---------------------------------------------|
| `view_img`        | `bool`      | `False`                    | Display frames with counts                  |
| `line_thickness`  | `int`       | `2`                        | Increase bounding boxes thickness           |
| `region_pts`      | `list`      | `[(20, 400), (1260, 400)]` | Points defining the Region Area             |
| `names`           | `dict`      | `model.names`              | Dictionary of Class Names                   |
| `region_color`    | `RGB Color` | `(255, 0, 255)`            | Color of the Object counting Region or Line |
| `draw_tracks`     | `bool`      | `False`                    | Enable drawing Track lines                  |
| `count_txt_color` | `RGB Color` | `(255, 255, 255)`          | Foreground color for Object counts text     |
| `txt_color`       | `RGB Color` | `(255, 255, 255)`          | Foreground color for object counts text     |
| `bg_color`        | `RGB Color` | `(255, 255, 255)`          | Count highlighter color                     |

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
