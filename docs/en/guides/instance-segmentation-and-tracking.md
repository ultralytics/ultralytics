---
comments: true
description: Instance Segmentation with Object Tracking using Ultralytics YOLOv8
keywords: Ultralytics, YOLOv8, Instance Segmentation, Object Detection, Object Tracking, Segbbox, Computer Vision, Notebook, IPython Kernel, CLI, Python SDK
---

# Instance Segmentation and Tracking using Ultralytics YOLOv8 🚀

## What is Instance Segmentation?

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) instance segmentation involves identifying and outlining individual objects in an image, providing a detailed understanding of spatial distribution. Unlike semantic segmentation, it uniquely labels and precisely delineates each object, crucial for tasks like object detection and medical imaging.

There are two types of instance segmentation tracking available in the Ultralytics package:

- **Instance Segmentation with Class Objects:** Each class object is assigned a unique color for clear visual separation.

- **Instance Segmentation with Object Tracks:** Every track is represented by a distinct color, facilitating easy identification and tracking.

## Samples

|                                                          Instance Segmentation                                                          |                                                           Instance Segmentation + Object Tracking                                                            |
|:---------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![Ultralytics Instance Segmentation](https://github.com/RizwanMunawar/ultralytics/assets/62513924/d4ad3499-1f33-4871-8fbc-1be0b2643aa2) | ![Ultralytics Instance Segmentation with Object Tracking](https://github.com/RizwanMunawar/ultralytics/assets/62513924/2e5c38cc-fd5c-4145-9682-fa94ae2010a0) |
|                                                  Ultralytics Instance Segmentation 😍                                                   |                                                  Ultralytics Instance Segmentation with Object Tracking 🔥                                                   |

!!! Example "Instance Segmentation and Tracking"

    === "Instance Segmentation"
        ```python
        import cv2
        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator, colors

        model = YOLO("yolov8n-seg.pt")
        names = model.model.names
        cap = cv2.VideoCapture("path/to/video/file.mp4")

        out = cv2.VideoWriter('instance-segmentation.avi',
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              30, (int(cap.get(3)), int(cap.get(4))))

        while True:
            ret, im0 = cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = model.predict(im0)
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy

            annotator = Annotator(im0, line_width=2)

            for mask, cls in zip(masks, clss):
                annotator.seg_bbox(mask=mask,
                                   mask_color=colors(int(cls), True),
                                   det_label=names[int(cls)])

            out.write(im0)
            cv2.imshow("instance-segmentation", im0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()

        ```

    === "Instance Segmentation with Object Tracking"
        ```python
        import cv2
        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator, colors

        from collections import defaultdict

        track_history = defaultdict(lambda: [])

        model = YOLO("yolov8n-seg.pt")
        cap = cv2.VideoCapture("path/to/video/file.mp4")

        out = cv2.VideoWriter('instance-segmentation-object-tracking.avi',
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              30, (int(cap.get(3)), int(cap.get(4))))

        while True:
            ret, im0 = cap.read()
            if not ret:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = model.track(im0, persist=True)
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()

            annotator = Annotator(im0, line_width=2)

            for mask, track_id in zip(masks, track_ids):
                annotator.seg_bbox(mask=mask,
                                   mask_color=colors(track_id, True),
                                   track_label=str(track_id))

            out.write(im0)
            cv2.imshow("instance-segmentation-object-tracking", im0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        ```

### `seg_bbox` Arguments

| Name          | Type    | Default         | Description                            |
|---------------|---------|-----------------|----------------------------------------|
| `mask`        | `array` | `None`          | Segmentation mask coordinates          |
| `mask_color`  | `tuple` | `(255, 0, 255)` | Mask color for every segmented box     |
| `det_label`   | `str`   | `None`          | Label for segmented object             |
| `track_label` | `str`   | `None`          | Label for segmented and tracked object |

## Note

For any inquiries, feel free to post your questions in the [Ultralytics Issue Section](https://github.com/ultralytics/ultralytics/issues/new/choose) or the discussion section mentioned below.
