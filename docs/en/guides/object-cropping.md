---
title: Extract and Crop Objects with YOLO26
comments: true
description: Detect and crop objects from images and videos with Ultralytics YOLO26, saving each detection to disk for focused analysis and dataset building.
keywords: Ultralytics, YOLO26, object cropping, object extraction, object detection, dataset creation, image processing, video analysis, AI, machine learning
---

# Object Cropping using Ultralytics YOLO26

## What is Object Cropping?

Object cropping with [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) involves isolating and extracting specific detected objects from an image or video. YOLO26 locates each object with a [bounding box](https://www.ultralytics.com/glossary/bounding-box), and the cropper saves that region as a separate image for further analysis or processing.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/J1BaCqytBmA"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Cropping using Ultralytics YOLO
</p>

## Advantages of Object Cropping

- **Focused Analysis**: YOLO26 enables targeted object cropping, allowing in-depth examination or processing of individual items within a scene.
- **Reduced Data Volume**: By extracting only relevant objects, object cropping minimizes data size, making storage, transmission, and subsequent computational tasks more efficient.
- **Enhanced Precision**: YOLO26's [object detection](https://www.ultralytics.com/glossary/object-detection) [accuracy](https://www.ultralytics.com/glossary/accuracy) produces tight crops that preserve the visual detail needed for further analysis.

## Visuals

|                                                           Airport Luggage                                                           |
| :---------------------------------------------------------------------------------------------------------------------------------: |
| ![Conveyor Belt at Airport Suitcases Cropping using Ultralytics YOLO26](https://cdn.ul.run/i/64b719a995705988d204b205eb0d3a9e.avif) |
|                                Suitcases Cropping at airport conveyor belt using Ultralytics YOLO26                                 |

## Crop Objects with YOLO26

Pass your video to the `ObjectCropper` solution and it detects objects each frame and saves every detection as a separate image. Restrict cropping to specific `classes`, raise `conf` to keep only confident detections, and set `crop_dir` to choose where the crops are written.

!!! example "Object Cropping using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Crop the objects
        yolo solutions crop

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

        # Initialize object cropper
        cropper = solutions.ObjectCropper(
            model="yolo26n.pt",  # model for object cropping, e.g., yolo26x.pt.
            classes=[0, 2],  # crop specific classes such as person and car with the COCO pretrained model.
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

        Every cropped object is written to the `crop_dir` folder (`cropped-detections` by default) with sequentially numbered filenames (e.g. `crop_1.jpg`, `crop_2.jpg`). This makes it easy to inspect detections or build downstream datasets without writing extra code.

### `ObjectCropper` Arguments

Here's a table with the `ObjectCropper` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "crop_dir", "verbose"]) }}

The `ObjectCropper` runs `predict` rather than `track`, so it supports the following inference parameters except `tracker`:

{% include "macros/solutions-track-args.md" %}

## FAQ

### What is object cropping in Ultralytics YOLO26 and how does it work?

Object cropping using [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) involves isolating and extracting specific objects from an image or video based on YOLO26's detection capabilities. This process allows for focused analysis and reduced data volume by using YOLO26 to identify objects accurately and crop them accordingly. For an in-depth tutorial, refer to the [object cropping example](#crop-objects-with-yolo26).

### Why should I use Ultralytics YOLO26 for object cropping over other solutions?

Ultralytics YOLO26 combines accurate detection, fast inference, and a simple API, so cropping takes only a few lines of code and supports [focused analysis](#advantages-of-object-cropping). YOLO26 also exports to formats such as [OpenVINO](../integrations/openvino.md) and [TensorRT](../integrations/tensorrt.md) for real-time deployments on diverse hardware. Explore the benefits in the [guide on model export](../modes/export.md).

### How can I reduce the data volume of my dataset using object cropping?

By using Ultralytics YOLO26 to crop only relevant objects from your images or videos, you can significantly reduce the data size, making it more efficient for storage and processing. Use a pretrained model, or [train](../modes/train.md) one to detect the objects you need, then crop and save only those regions. To get started with Ultralytics YOLO26, visit our [quickstart guide](../quickstart.md).

### Can I use Ultralytics YOLO26 for real-time video analysis and object cropping?

Yes, Ultralytics YOLO26 can process real-time video feeds to detect and crop objects dynamically. Its fast inference suits real-time applications such as [surveillance](security-alarm-system.md), sports analysis, and automated inspection systems. Check out the [tracking](../modes/track.md) and [prediction modes](../modes/predict.md) to understand how to implement real-time processing.

### What are the hardware requirements for efficiently running YOLO26 for object cropping?

Ultralytics YOLO26 is optimized for both CPU and GPU environments, but to achieve optimal performance, especially for real-time or high-volume inference, a dedicated GPU (e.g., an NVIDIA RTX or data center GPU) is recommended. For deployment on lightweight devices, consider using [CoreML](../integrations/coreml.md) for iOS or [LiteRT](../integrations/litert.md) for Android. More details on supported devices and formats can be found in our [model deployment options](model-deployment-options.md).
