---
comments: true
description: Object Segmentation in Bounding Boxes Shape Using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Object Detection, Object Tracking, Object Segmentation, Segbbox, Computer Vision, Notebook, IPython Kernel, CLI, Python SDK
---

# Object Segmentation in Bounding Boxes Shape using Ultralytics YOLOv8 🚀

## What is Object Segmentation?

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) Object segmentation involves identifying and delineating specific objects within an image. It aims to partition an image into meaningful segments corresponding to individual objects, enabling precise localization and analysis.

## Samples

|                                                    Object Segmentation                                                     |                                                      Object Segmentation + Object Tracking                                                      |
|:--------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------:|
| ![Ultralytics BBox-Seg](https://github.com/RizwanMunawar/ultralytics/assets/62513924/6b4ebca2-a1c0-41fd-918c-f3a233bf8c9f) | ![Ultralytics BBox-Seg with Object Tracking](https://github.com/RizwanMunawar/ultralytics/assets/62513924/2a6eb64e-67eb-49b1-b44d-e2467f0825a3) |
|                                                  Ultralytics BBox-Seg 😍                                                   |                                                  Ultralytics BBox-Seg with Object Tracking 🔥                                                   |


!!! Object Segmentation in Bounding Box Shape using YOLOv8

    === "SegBbox Mapping"
        ```python
        import cv2
        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator

        model = YOLO("yolov8n-seg.pt")
        names = model.model.names
        cap = cv2.VideoCapture("path/to/video/file.mp4")

        out = cv2.VideoWriter('segbbox.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                              30, (int(cap.get(3)), int(cap.get(4))))

        while True:
            ret, im0 = cap.read()
            if not ret:
                break

            results = model.predict(im0)
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy

            annotator = Annotator(im0, line_width=2)

            for mask, cls in zip(masks, clss):
                annotator.seg_bbox(mask, det_label=names[int(cls)])

            out.write(im0)
            cv2.imshow("segbbox", im0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        ```

    === "SegBbox Mapping with Object Tracking"
        ```python
        import cv2
        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator

        from collections import defaultdict
        track_history = defaultdict(lambda: [])

        model = YOLO("yolov8n-seg.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")

        out = cv2.VideoWriter('segbbox.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                              30, (int(cap.get(3)), int(cap.get(4))))

        while True:
            ret, im0 = cap.read()
            if not ret:
                break

            results = model.track(im0, persist=True)
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()

            annotator = Annotator(im0, line_width=2)

            for mask, track_id in zip(masks, track_ids):
                annotator.seg_bbox(mask, track_label=str(track_id))

            out.write(im0)
            cv2.imshow("segbbox", im0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        ```

## Note

For any inquiries, feel free to post your questions in the [Ultralytics Issue Section](https://github.com/ultralytics/ultralytics/issues/new/choose) or the discussion section mentioned below.
