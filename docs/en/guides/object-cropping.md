---
comments: true
description: Learn how to crop and extract objects using Ultralytics YOLO11 for focused analysis, reduced data volume, and enhanced precision.
keywords: Ultralytics, YOLO11, object cropping, object detection, image processing, video analysis, AI, machine learning
---

# Object Cropping using Ultralytics YOLO11

## What is Object Cropping?

Object cropping with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) involves isolating and extracting specific detected objects from an image or video. The YOLO11 model capabilities are utilized to accurately identify and delineate objects, enabling precise cropping for further analysis or manipulation.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ydGdibB5Mds"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Cropping using Ultralytics YOLO
</p>

## Advantages of Object Cropping?

- **Focused Analysis**: YOLO11 facilitates targeted object cropping, allowing for in-depth examination or processing of individual items within a scene.
- **Reduced Data Volume**: By extracting only relevant objects, object cropping helps in minimizing data size, making it efficient for storage, transmission, or subsequent computational tasks.
- **Enhanced Precision**: YOLO11's [object detection](https://www.ultralytics.com/glossary/object-detection) [accuracy](https://www.ultralytics.com/glossary/accuracy) ensures that the cropped objects maintain their spatial relationships, preserving the integrity of the visual information for detailed analysis.

## Visuals

|                                                                                Airport Luggage                                                                                 |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Conveyor Belt at Airport Suitcases Cropping using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/suitcases-cropping-airport-conveyor-belt.avif) |
|                                                      Suitcases Cropping at airport conveyor belt using Ultralytics YOLO11                                                      |

!!! example "Object Cropping using YOLO11 Example"

    === "Object Cropping"

        ```python
        import os

        import cv2

        from ultralytics import YOLO
        from ultralytics.utils.plotting import Annotator, colors

        model = YOLO("yolo11n.pt")
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

{% include "macros/predict-args.md" %}

## FAQ

### What is object cropping in Ultralytics YOLO11 and how does it work?

Object cropping using [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) involves isolating and extracting specific objects from an image or video based on YOLO11's detection capabilities. This process allows for focused analysis, reduced data volume, and enhanced [precision](https://www.ultralytics.com/glossary/precision) by leveraging YOLO11 to identify objects with high accuracy and crop them accordingly. For an in-depth tutorial, refer to the [object cropping example](#object-cropping-using-ultralytics-yolo11).

### Why should I use Ultralytics YOLO11 for object cropping over other solutions?

Ultralytics YOLO11 stands out due to its precision, speed, and ease of use. It allows detailed and accurate object detection and cropping, essential for [focused analysis](#advantages-of-object-cropping) and applications needing high data integrity. Moreover, YOLO11 integrates seamlessly with tools like OpenVINO and TensorRT for deployments requiring real-time capabilities and optimization on diverse hardware. Explore the benefits in the [guide on model export](../modes/export.md).

### How can I reduce the data volume of my dataset using object cropping?

By using Ultralytics YOLO11 to crop only relevant objects from your images or videos, you can significantly reduce the data size, making it more efficient for storage and processing. This process involves training the model to detect specific objects and then using the results to crop and save these portions only. For more information on exploiting Ultralytics YOLO11's capabilities, visit our [quickstart guide](../quickstart.md).

### Can I use Ultralytics YOLO11 for real-time video analysis and object cropping?

Yes, Ultralytics YOLO11 can process real-time video feeds to detect and crop objects dynamically. The model's high-speed inference capabilities make it ideal for real-time applications such as surveillance, sports analysis, and automated inspection systems. Check out the [tracking and prediction modes](../modes/predict.md) to understand how to implement real-time processing.

### What are the hardware requirements for efficiently running YOLO11 for object cropping?

Ultralytics YOLO11 is optimized for both CPU and GPU environments, but to achieve optimal performance, especially for real-time or high-volume inference, a dedicated GPU (e.g., NVIDIA Tesla, RTX series) is recommended. For deployment on lightweight devices, consider using CoreML for iOS or TFLite for Android. More details on supported devices and formats can be found in our [model deployment options](../guides/model-deployment-options.md).
