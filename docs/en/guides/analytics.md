---
comments: true
description: Comprehensive Guide to Understanding and Creating Line Graphs, Bar Plots, and Pie Charts
keywords: Analytics, Data Visualization, Line Graphs, Bar Plots, Pie Charts, Quickstart Guide, Data Analysis, Python, Visualization Tools
---

# Analytics using Ultralytics YOLOv8 📊

## Introduction

This guide provides a comprehensive overview of three fundamental types of data visualizations: line graphs, bar plots, and pie charts. Each section includes step-by-step instructions and code snippets on how to create these visualizations using Python.

### Visual Samples

|                                                     Line Graph                                                     |                                                     Bar Plot                                                     |                                                     Pie Chart                                                     |
|:------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
| ![Line Graph](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/eeabd90c-04fd-4e5b-aac9-c7777f892200) | ![Bar Plot](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/c1da2d6a-99ff-43a8-b5dc-ca93127917f8) | ![Pie Chart](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/9d8acce6-d9e4-4685-949d-cd4851483187) |

###  Why Graphs are Important

- Line graphs are ideal for tracking changes over short and long periods and for comparing changes for multiple groups over the same period. 
- Bar plots, on the other hand, are suitable for comparing quantities across different categories and showing relationships between a category and its numerical value. 
- Lastly, pie charts are effective for illustrating proportions among categories and showing parts of a whole.

!!! Analytics "Analytics Examples"
    
    === "Line Graph"

        ```python
        import cv2
        from ultralytics import YOLO, solutions

        model = YOLO("yolov8s.pt")

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("line_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        analytics = solutions.Analytics(
            type="line",
            writer=out,
            im0_shape=(w, h),
            view_img=True,
        )
        total_counts = 0
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()

            if success:
                frame_count += 1
                results = model.track(frame, persist=True, verbose=True)

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    for box in boxes:
                        total_counts += 1

                analytics.update_line(frame_count, total_counts)

                total_counts = 0
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        ```

    === "Pie Chart"

        ```python
        import cv2
        from ultralytics import YOLO, solutions

        model = YOLO("yolov8s.pt")

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("pie_chart.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        analytics = solutions.Analytics(
            type="pie",
            writer=out,
            im0_shape=(w, h),
            view_img=True,
        )

        clswise_count = {}

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model.track(frame, persist=True, verbose=True)
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    clss = results[0].boxes.cls.cpu().tolist()
                    for box, cls in zip(boxes, clss):
                        if model.names[int(cls)] in clswise_count:
                            clswise_count[model.names[int(cls)]] += 1
                        else:
                            clswise_count[model.names[int(cls)]] = 1

                    analytics.update_pie(clswise_count)
                    clswise_count = {}

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        ```
    
    === "Bar Plot"

        ```python
        import cv2
        from ultralytics import YOLO, solutions

        model = YOLO("yolov8s.pt")

        cap = cv2.VideoCapture("Path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("bar_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        analytics = solutions.Analytics(
            type="bar",
            writer=out,
            im0_shape=(w, h),
            view_img=True,
        )

        clswise_count = {}

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model.track(frame, persist=True, verbose=True)
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu()
                    clss = results[0].boxes.cls.cpu().tolist()
                    for box, cls in zip(boxes, clss):
                        if model.names[int(cls)] in clswise_count:
                            clswise_count[model.names[int(cls)]] += 1
                        else:
                            clswise_count[model.names[int(cls)]] = 1

                    analytics.update_bar(clswise_count)
                    clswise_count = {}

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        ```

### Argument `Analytics`

Here's a table with the `Analytics` arguments:

| Name         | Type              | Default       | Description                                 |
|--------------|-------------------|---------------|---------------------------------------------|
| `type`       | `str`             | `None`        | Type of data or object.                     |
| `im0_shape`  | `tuple`           | `None`        | Shape of the initial image.                 |
| `writer`     | `cv2.VideoWriter` | `None`        | Object for writing video files.             |
| `title`      | `str`             | `ultralytics` | Title for the visualization.                |
| `x_label`    | `str`             | `x`           | Label for the x-axis.                       |
| `y_label`    | `str`             | `y`           | Label for the y-axis.                       |
| `bg_color`   | `str`             | `white`       | Background color.                           |
| `fg_color`   | `str`             | `black`       | Foreground color.                           |
| `line_color` | `str`             | `yellow`      | Color of the lines.                         |
| `line_width` | `int`             | `2`           | Width of the lines.                         |
| `fontsize`   | `int`             | `13`          | Font size for text.                         |
| `view_img`   | `bool`            | `False`       | Flag to display the image or video.         |
| `save_img`   | `bool`            | `True`        | Flag to save the image or video.            |

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

## Conclusion

Understanding when and how to use different types of visualizations is crucial for effective data analysis. Line graphs, bar plots, and pie charts are fundamental tools that can help you convey your data's story more clearly and effectively.
