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

## Advantages of Object Cropping

- **Focused Analysis**: YOLO11 facilitates targeted object cropping, allowing for in-depth examination or processing of individual items within a scene.
- **Reduced Data Volume**: By extracting only relevant objects, object cropping helps in minimizing data size, making it efficient for storage, transmission, or subsequent computational tasks.
- **Enhanced Precision**: YOLO11's [object detection](https://www.ultralytics.com/glossary/object-detection) [accuracy](https://www.ultralytics.com/glossary/accuracy) ensures that the cropped objects maintain their spatial relationships, preserving the integrity of the visual information for detailed analysis.

## Visuals

|                                                                                Airport Luggage                                                                                 |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Conveyor Belt at Airport Suitcases Cropping using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/suitcases-cropping-airport-conveyor-belt.avif) |
|                                                      Suitcases Cropping at airport conveyor belt using Ultralytics YOLO11                                                      |

!!! example "Object Cropping using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Crop the objects
        yolo solutions crop show=True

        # Pass a source video
        yolo solutions crop source="path/to/video.mp4"

        # Crop specific classes
        yolo solutions crop classes="[0, 2]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Initialize object cropper object
        cropper = solutions.ObjectCropper(
            show=True,  # display the output
            model="yolo11n.pt",  # model for object cropping i.e yolo11x.pt.
            classes=[0, 2],  # crop specific classes i.e. person and car with COCO pretrained model.
            # conf=0.5,  # adjust confidence threshold for the objects.
            # crop_dir="cropped-detections",  # set the directory name for cropped detections
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = cropper(im0)

            # print(results)  # access the output

        cap.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `ObjectCropper` Arguments

Here's a table with the `ObjectCropper` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "crop_dir"]) }}

Moreover, the following visualization arguments are available for use:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## FAQ

### What is object cropping in Ultralytics YOLO11 and how does it work?

Object cropping using [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) involves isolating and extracting specific objects from an image or video based on YOLO11's detection capabilities. This process allows for focused analysis, reduced data volume, and enhanced [precision](https://www.ultralytics.com/glossary/precision) by leveraging YOLO11 to identify objects with high accuracy and crop them accordingly. For an in-depth tutorial, refer to the [object cropping example](#object-cropping-using-ultralytics-yolo11).

### Why should I use Ultralytics YOLO11 for object cropping over other solutions?

Ultralytics YOLO11 stands out due to its precision, speed, and ease of use. It allows detailed and accurate object detection and cropping, essential for [focused analysis](#advantages-of-object-cropping) and applications needing high data integrity. Moreover, YOLO11 integrates seamlessly with tools like [OpenVINO](../integrations/openvino.md) and [TensorRT](../integrations/tensorrt.md) for deployments requiring real-time capabilities and optimization on diverse hardware. Explore the benefits in the [guide on model export](../modes/export.md).

### How can I reduce the data volume of my dataset using object cropping?

By using Ultralytics YOLO11 to crop only relevant objects from your images or videos, you can significantly reduce the data size, making it more efficient for storage and processing. This process involves training the model to detect specific objects and then using the results to crop and save these portions only. For more information on exploiting Ultralytics YOLO11's capabilities, visit our [quickstart guide](../quickstart.md).

### Can I use Ultralytics YOLO11 for real-time video analysis and object cropping?

Yes, Ultralytics YOLO11 can process real-time video feeds to detect and crop objects dynamically. The model's high-speed inference capabilities make it ideal for real-time applications such as [surveillance](security-alarm-system.md), sports analysis, and automated inspection systems. Check out the [tracking](../modes/track.md) and [prediction modes](../modes/predict.md) to understand how to implement real-time processing.

### What are the hardware requirements for efficiently running YOLO11 for object cropping?

Ultralytics YOLO11 is optimized for both CPU and GPU environments, but to achieve optimal performance, especially for real-time or high-volume inference, a dedicated GPU (e.g., NVIDIA Tesla, RTX series) is recommended. For deployment on lightweight devices, consider using [CoreML](../integrations/coreml.md) for iOS or [TFLite](../integrations/tflite.md) for Android. More details on supported devices and formats can be found in our [model deployment options](../guides/model-deployment-options.md).
