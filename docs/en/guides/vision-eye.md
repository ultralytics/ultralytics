---
comments: true
description: Discover VisionEye's object mapping and tracking powered by Ultralytics YOLOv8. Simulate human eye precision, track objects, and calculate distances effortlessly.
keywords: VisionEye, YOLOv8, Ultralytics, object mapping, object tracking, distance calculation, computer vision, AI, machine learning, Python, tutorial
---

# VisionEye View Object Mapping using Ultralytics YOLOv8 🚀

## What is VisionEye Object Mapping?

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) VisionEye offers the capability for computers to identify and pinpoint objects, simulating the observational precision of the human eye. This functionality enables computers to discern and focus on specific objects, much like the way the human eye observes details from a particular viewpoint.

## Samples

|                                                                        VisionEye View                                                                        |                                                                        VisionEye View With Object Tracking                                                                        |                                                                 VisionEye View With Distance Calculation                                                                  |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![VisionEye View Object Mapping using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d593acc-2e37-41b0-ad0e-92b4ffae6647) | ![VisionEye View Object Mapping with Object Tracking using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/fcd85952-390f-451e-8fb0-b82e943af89c) | ![VisionEye View with Distance Calculation using Ultralytics YOLOv8](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/18c4dafe-a22e-4fa9-a7d4-2bb293562a95) |
|                                                    VisionEye View Object Mapping using Ultralytics YOLOv8                                                    |                                                    VisionEye View Object Mapping with Object Tracking using Ultralytics YOLOv8                                                    |                                                     VisionEye View with Distance Calculation using Ultralytics YOLOv8                                                     |

!!! Example "VisionEye Object Mapping using YOLOv8"

    === "VisionEye Object Mapping"

        ```python
        import cv2
        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator, colors

        model = YOLO("yolov8n.pt")
        names = model.model.names
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("visioneye-pinpoint.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        center_point = (-10, h)

        while True:
            ret, im0 = cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = model.predict(im0)
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(im0, line_width=2)

            for box, cls in zip(boxes, clss):
                annotator.box_label(box, label=names[int(cls)], color=colors(int(cls)))
                annotator.visioneye(box, center_point)

            out.write(im0)
            cv2.imshow("visioneye-pinpoint", im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        ```

    === "VisionEye Object Mapping with Object Tracking"

        ```python
        import cv2
        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator, colors

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("visioneye-pinpoint.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        center_point = (-10, h)

        while True:
            ret, im0 = cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            annotator = Annotator(im0, line_width=2)

            results = model.track(im0, persist=True)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    annotator.box_label(box, label=str(track_id), color=colors(int(track_id)))
                    annotator.visioneye(box, center_point)

            out.write(im0)
            cv2.imshow("visioneye-pinpoint", im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        ```
    
    === "VisionEye with Distance Calculation"
    
        ```python
        import math

        import cv2
        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator, colors

        model = YOLO("yolov8s.pt")
        cap = cv2.VideoCapture("Path/to/video/file.mp4")

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter("visioneye-distance-calculation.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        center_point = (0, h)
        pixel_per_meter = 10

        txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))

        while True:
            ret, im0 = cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            annotator = Annotator(im0, line_width=2)

            results = model.track(im0, persist=True)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    annotator.box_label(box, label=str(track_id), color=bbox_clr)
                    annotator.visioneye(box, center_point)

                    x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)  # Bounding box centroid

                    distance = (math.sqrt((x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2)) / pixel_per_meter

                    text_size, _ = cv2.getTextSize(f"Distance: {distance:.2f} m", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    cv2.rectangle(im0, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), txt_background, -1)
                    cv2.putText(im0, f"Distance: {distance:.2f} m", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, txt_color, 3)

            out.write(im0)
            cv2.imshow("visioneye-distance-calculation", im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        ```

### `visioneye` Arguments

| Name        | Type    | Default          | Description                    |
|-------------|---------|------------------|--------------------------------|
| `color`     | `tuple` | `(235, 219, 11)` | Line and object centroid color |
| `pin_color` | `tuple` | `(255, 0, 255)`  | VisionEye pinpoint color       |

## Note

For any inquiries, feel free to post your questions in the [Ultralytics Issue Section](https://github.com/ultralytics/ultralytics/issues/new/choose) or the discussion section mentioned below.
