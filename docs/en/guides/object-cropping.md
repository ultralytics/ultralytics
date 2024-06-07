---
comments: true
description: Learn how to crop and extract objects using Ultralytics YOLOv8 for focused analysis, reduced data volume, and enhanced precision.
keywords: Ultralytics, YOLOv8, object cropping, object detection, image processing, video analysis, AI, machine learning
---

# Object Cropping using Ultralytics YOLOv8 🚀

## What is Object Cropping?

Object cropping with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) involves isolating and extracting specific detected objects from an image or video. The YOLOv8 model capabilities are utilized to accurately identify and delineate objects, enabling precise cropping for further analysis or manipulation.

## Advantages of Object Cropping?

- **Focused Analysis**: YOLOv8 facilitates targeted object cropping, allowing for in-depth examination or processing of individual items within a scene.
- **Reduced Data Volume**: By extracting only relevant objects, object cropping helps in minimizing data size, making it efficient for storage, transmission, or subsequent computational tasks.
- **Enhanced Precision**: YOLOv8's object detection accuracy ensures that the cropped objects maintain their spatial relationships, preserving the integrity of the visual information for detailed analysis.

## Visuals

|                                                                               Airport Luggage                                                                                |                                                                                                                         
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![Conveyor Belt at Airport Suitcases Cropping using Ultralytics YOLOv8](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/648f46be-f233-4307-a8e5-046eea38d2e4) |
|                                                     Suitcases Cropping at airport conveyor belt using Ultralytics YOLOv8                                                     |                                                                                                       

!!! Example "Object Cropping using YOLOv8 Example"

    === "Object Cropping"

        ```python
        import os

        import cv2
        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator, colors

        model = YOLO("yolov8n.pt")
        names = model.names

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        crop_dir_name = "ultralytics_crop"
        if not os.path.exists(crop_dir_name):
            os.mkdir(crop_dir_name)

        # Video writer
        video_writer = cv2.VideoWriter("object_cropping_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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

                    crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

                    cv2.imwrite(os.path.join(crop_dir_name, str(idx) + ".png"), crop_obj)

            cv2.imshow("ultralytics", im0)
            video_writer.write(im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        ```

### Arguments `model.predict`

| Argument        | Type           | Default                | Description                                                                                                                                                                                                                          |
|-----------------|----------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `source`        | `str`          | `'ultralytics/assets'` | Specifies the data source for inference. Can be an image path, video file, directory, URL, or device ID for live feeds. Supports a wide range of formats and sources, enabling flexible application across different types of input. |
| `conf`          | `float`        | `0.25`                 | Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.                                               |
| `iou`           | `float`        | `0.7`                  | Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.                                                 |
| `imgsz`         | `int or tuple` | `640`                  | Defines the image size for inference. Can be a single integer `640` for square resizing or a (height, width) tuple. Proper sizing can improve detection accuracy and processing speed.                                               |
| `half`          | `bool`         | `False`                | Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.                                                                                                       |
| `device`        | `str`          | `None`                 | Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.                                                                 |
| `max_det`       | `int`          | `300`                  | Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes.                                                         |
| `vid_stride`    | `int`          | `1`                    | Frame stride for video inputs. Allows skipping frames in videos to speed up processing at the cost of temporal resolution. A value of 1 processes every frame, higher values skip frames.                                            |
| `stream_buffer` | `bool`         | `False`                | Determines if all frames should be buffered when processing video streams (`True`), or if the model should return the most recent frame (`False`). Useful for real-time applications.                                                |
| `visualize`     | `bool`         | `False`                | Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation.                                                                       |
| `augment`       | `bool`         | `False`                | Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.                                                                                                     |
| `agnostic_nms`  | `bool`         | `False`                | Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.                                                  |
| `classes`       | `list[int]`    | `None`                 | Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.                                              |
| `retina_masks`  | `bool`         | `False`                | Uses high-resolution segmentation masks if available in the model. This can enhance mask quality for segmentation tasks, providing finer detail.                                                                                     |
| `embed`         | `list[int]`    | `None`                 | Specifies the layers from which to extract feature vectors or embeddings. Useful for downstream tasks like clustering or similarity search.                                                                                          |
