---
comments: true
description: VisionEye View Object Mapping using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Object Tracking, IDetection, VisionEye, Computer Vision, Notebook, IPython Kernel, CLI, Python SDK
---

# VisionEye View Object Mapping using Ultralytics YOLOv8 🚀

## What is VisionEye Object Mapping?

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) VisionEye offers the capability for computers to identify and pinpoint objects, simulating the observational precision of the human eye. This functionality enables computers to discern and focus on specific objects, much like the way the human eye observes details from a particular viewpoint.

## Samples

|                                                                        VisionEye View                                                                        |                                                                        VisionEye View With Object Tracking                                                                        |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![VisionEye View Object Mapping using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d593acc-2e37-41b0-ad0e-92b4ffae6647) | ![VisionEye View Object Mapping with Object Tracking using Ultralytics YOLOv8](https://github.com/RizwanMunawar/ultralytics/assets/62513924/fcd85952-390f-451e-8fb0-b82e943af89c) |
|                                                    VisionEye View Object Mapping using Ultralytics YOLOv8                                                    |                                                    VisionEye View Object Mapping with Object Tracking using Ultralytics YOLOv8                                                    |

!!! Example "VisionEye Object Mapping using YOLOv8"

    === "VisionEye Object Mapping"

        ```python
        import cv2
        from ultralytics import YOLO
        from ultralytics.utils.plotting import colors, Annotator

        model = YOLO("yolov8n.pt")
        names = model.model.names
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter('visioneye-pinpoint.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        ```

    === "VisionEye Object Mapping with Object Tracking"

        ```python
        import cv2
        from ultralytics import YOLO
        from ultralytics.utils.plotting import colors, Annotator

        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter('visioneye-pinpoint.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))

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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        ```

### `visioneye` Arguments

| Name          | Type    | Default          | Description                                      |
|---------------|---------|------------------|--------------------------------------------------|
| `color`       | `tuple` | `(235, 219, 11)` | Line and object centroid color                   |
| `pin_color`   | `tuple` | `(255, 0, 255)`  | VisionEye pinpoint color                         |
| `thickness`   | `int`   | `2`              | pinpoint to object line thickness                |
| `pins_radius` | `int`   | `10`             | Pinpoint and object centroid point circle radius |

## Note

For any inquiries, feel free to post your questions in the [Ultralytics Issue Section](https://github.com/ultralytics/ultralytics/issues/new/choose) or the discussion section mentioned below.
