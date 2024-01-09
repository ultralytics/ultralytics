---
comments: true
description: Learn how to isolate and extract specific objects from images and videos using YOLOv8 object cropping.
keywords: Ultralytics, YOLOv8, Object Detection, Object Cropping, Image Analysis, Video Processing, Data Extraction, Python
---

# Object Cropping using Ultralytics YOLOv8 🚀

## What is Object Cropping?

Object cropping with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) involves isolating and extracting specific detected objects from an image or video. The YOLOv8 model capabilities are utilized to accurately identify and delineate objects, enabling precise cropping for further analysis or manipulation.

## Advantages of Object Cropping?

- **Focused Analysis**: YOLOv8 facilitates targeted object cropping, allowing for in-depth examination or processing of individual items within a scene.
- **Reduced Data Volume**: By extracting only relevant objects, object cropping helps in minimizing data size, making it efficient for storage, transmission, or subsequent computational tasks.
- **Enhanced Precision**: YOLOv8's object detection accuracy ensures that the cropped objects maintain their spatial relationships, preserving the integrity of the visual information for detailed analysis.

!!! Example "Object Cropping using YOLOv8 Example"

    === "Object Cropping"
        ```python
        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator, colors
        import cv2
        import os

        model = YOLO("yolov8n.pt")
        names = model.names

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        crop_dir_name = "ultralytics_crop"
        if not os.path.exists(crop_dir_name):
            os.mkdir(crop_dir_name)

        # Video writer
        video_writer = cv2.VideoWriter("object_cropping_output.avi",
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps, (w, h))

        idx = 0
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
                    idx += 1
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                    crop_obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

                    cv2.imwrite(os.path.join(crop_dir_name, str(idx)+".png"), crop_obj)

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
