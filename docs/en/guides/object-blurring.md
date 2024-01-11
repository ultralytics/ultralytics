---
comments: true
description: Learn to blur objects using Ultralytics YOLOv8 for privacy in images and videos.
keywords: Ultralytics, YOLOv8, Object Detection, Object Blurring, Privacy Protection, Image Processing, Video Analysis, AI, Machine Learning
---

# Object Blurring using Ultralytics YOLOv8 🚀

## What is Object Blurring?

Object blurring with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/)  involves applying a blurring effect to specific detected objects in an image or video. This can be achieved using the YOLOv8 model capabilities to identify and manipulate objects within a given scene.

## Advantages of Object Blurring?

- **Privacy Protection**: Object blurring is an effective tool for safeguarding privacy by concealing sensitive or personally identifiable information in images or videos.
- **Selective Focus**: YOLOv8 allows for selective blurring, enabling users to target specific objects, ensuring a balance between privacy and retaining relevant visual information.
- **Real-time Processing**: YOLOv8's efficiency enables object blurring in real-time, making it suitable for applications requiring on-the-fly privacy enhancements in dynamic environments.

!!! Example "Object Blurring using YOLOv8 Example"

    === "Object Blurring"
        ```python
        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator, colors
        import cv2

        model = YOLO("yolov8n.pt")
        names = model.names

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Blur ratio
        blur_ratio = 50

        # Video writer
        video_writer = cv2.VideoWriter("object_blurring_output.avi",
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps, (w, h))

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = model.predict(im0, show=False)
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            annotator = Annotator(im0, line_width=2, example=names)

            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                    obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))

                    im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = blur_obj

            cv2.imshow("ultralytics", im0)
            video_writer.write(im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

### Arguments `model.predict`

| Name            | Type           | Default                | Description                                                                |
|-----------------|----------------|------------------------|----------------------------------------------------------------------------|
| `source`        | `str`          | `'ultralytics/assets'` | source directory for images or videos                                      |
| `conf`          | `float`        | `0.25`                 | object confidence threshold for detection                                  |
| `iou`           | `float`        | `0.7`                  | intersection over union (IoU) threshold for NMS                            |
| `imgsz`         | `int or tuple` | `640`                  | image size as scalar or (h, w) list, i.e. (640, 480)                       |
| `half`          | `bool`         | `False`                | use half precision (FP16)                                                  |
| `device`        | `None or str`  | `None`                 | device to run on, i.e. cuda device=0/1/2/3 or device=cpu                   |
| `max_det`       | `int`          | `300`                  | maximum number of detections per image                                     |
| `vid_stride`    | `bool`         | `False`                | video frame-rate stride                                                    |
| `stream_buffer` | `bool`         | `False`                | buffer all streaming frames (True) or return the most recent frame (False) |
| `visualize`     | `bool`         | `False`                | visualize model features                                                   |
| `augment`       | `bool`         | `False`                | apply image augmentation to prediction sources                             |
| `agnostic_nms`  | `bool`         | `False`                | class-agnostic NMS                                                         |
| `classes`       | `list[int]`    | `None`                 | filter results by class, i.e. classes=0, or classes=[0,2,3]                |
| `retina_masks`  | `bool`         | `False`                | use high-resolution segmentation masks                                     |
| `embed`         | `list[int]`    | `None`                 | return feature vectors/embeddings from given layers                        |
