---
title: Real-Time Object Blurring with YOLO26
comments: true
description: Automatically detect and blur objects in real time with Ultralytics YOLO26. Anonymize faces, license plates, and sensitive content with adjustable intensity.
keywords: YOLO26, object blurring, real-time processing, privacy protection, face blurring, data anonymization, GDPR, image manipulation, video editing, Ultralytics
---

# Object Blurring using Ultralytics YOLO26 🚀

## What is Object Blurring?

Object blurring with [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) involves applying a blurring effect to specific detected objects in an image or video. This can be achieved using the YOLO26 model capabilities to identify and manipulate objects within a given scene.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/m-Lc5MXbydg"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Face Detection Model using Ultralytics Platform & Blur Faces | Ultralytics YOLO26 🚀
</p>

## Advantages of Object Blurring

- **Privacy Protection**: Object blurring is an effective tool for safeguarding privacy by concealing sensitive or personally identifiable information in images or videos.
- **Selective Focus**: YOLO26 allows for selective blurring, enabling users to target specific objects, ensuring a balance between privacy and retaining relevant visual information.
- **Real-time Processing**: YOLO26's efficiency enables object blurring in real-time, making it suitable for applications requiring on-the-fly privacy enhancements in dynamic environments.
- **Regulatory Compliance**: Helps organizations comply with data protection regulations like GDPR by anonymizing identifiable information in visual content.
- **Content Moderation**: Useful for blurring inappropriate or sensitive content in media platforms while preserving the overall context.

## Blur Objects with YOLO26

Pass your video to the `ObjectBlurrer` solution and it detects objects each frame and applies an averaging (box) blur to every detection. Control the strength with `blur_ratio` (0.1–1.0) and restrict blurring to specific `classes` — for example, only people and cars.

!!! example "Object Blurring using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Blur the objects
        yolo solutions blur show=True

        # Pass a source video
        yolo solutions blur source="path/to/video.mp4"

        # Blur the specific classes
        yolo solutions blur classes="[0, 5]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("object_blurring_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize object blurrer
        blurrer = solutions.ObjectBlurrer(
            show=True,  # display the output
            model="yolo26n.pt",  # model for object blurring, e.g., yolo26m.pt
            # line_width=2,  # width of bounding box.
            # classes=[0, 2],  # blur specific classes, e.g., person and car with the COCO pretrained model.
            # blur_ratio=0.5,  # adjust percentage of blur intensity, value in range 0.1 - 1.0
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = blurrer(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `ObjectBlurrer` Arguments

Here's a table with the `ObjectBlurrer` arguments:

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `model` | `str` | `None` | Path to an Ultralytics YOLO model file. |
| `blur_ratio` | `float` | `0.5` | Adjusts percentage of blur intensity, with values in range `0.1 - 1.0`. |


The `ObjectBlurrer` solution also supports a range of `track` arguments:

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `tracker` | `str` | `'tracktrack.yaml'` | Specifies the tracking algorithm to use. Built-in options: `botsort.yaml`, `bytetrack.yaml`, `ocsort.yaml`, `deepocsort.yaml`, `fasttrack.yaml`, `tracktrack.yaml`. |
| `conf` | `float` | `0.1` | Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives. |
| `iou` | `float` | `0.7` | Sets the [Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for filtering overlapping detections. |
| `classes` | `list` | `None` | Filters results by class index. For example, `classes=[0, 2, 3]` only tracks the specified classes. |
| `verbose` | `bool` | `True` | Controls the display of tracking results, providing a visual output of tracked objects. |
| `device` | `str` | `None` | Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution. |


Moreover, the following visualization arguments can be used:

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `show` | `bool` | `False` | If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing. |
| `line_width` | `int or None` | `None` | Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity. |
| `show_conf` | `bool` | `True` | Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection. |
| `show_labels` | `bool` | `True` | Displays labels for each detection in the visual output. Provides immediate understanding of detected objects. |


## Real-World Applications

### Privacy Protection in Surveillance

[Security cameras](https://www.ultralytics.com/blog/the-cutting-edge-world-of-ai-security-cameras) and surveillance systems can use YOLO26 to automatically blur faces, license plates, or other identifying information while still capturing important activity. This helps maintain security while respecting privacy rights in public spaces.

### Healthcare Data Anonymization

In [medical imaging](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency), patient information often appears in scans or photos. YOLO26 can detect and blur this information to comply with regulations like HIPAA when sharing medical data for research or educational purposes.

### Document Redaction

When sharing documents that contain sensitive information, YOLO26 can automatically detect and blur specific elements like signatures, account numbers, or personal details, streamlining the redaction process while maintaining document integrity.

### Media and Content Creation

Content creators can use YOLO26 to blur brand logos, copyrighted material, or inappropriate content in videos and images, helping avoid legal issues while preserving the overall content quality.

## FAQ

### What is object blurring with Ultralytics YOLO26?

Object blurring with [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) involves automatically detecting and applying a blurring effect to specific objects in images or videos. This technique enhances privacy by concealing sensitive information while retaining relevant visual data. YOLO26's real-time processing capabilities make it suitable for applications requiring immediate privacy protection and selective focus adjustments.

### How can I implement real-time object blurring using YOLO26?

Use the [Python example above](#blur-objects-with-yolo26), which runs YOLO26 [object detection](https://www.ultralytics.com/glossary/object-detection) and blurs each detection every frame. Set `blur_ratio` (for example `0.5` for 50% blur intensity) to control the effect, and pass `classes` to blur only specific object types.

### What are the benefits of using Ultralytics YOLO26 for object blurring?

Ultralytics YOLO26 offers several advantages for object blurring:

- **Privacy Protection**: Effectively obscure sensitive or identifiable information.
- **Selective Focus**: Target specific objects for blurring, maintaining essential visual content.
- **Real-time Processing**: Execute object blurring efficiently in dynamic environments, suitable for instant privacy enhancements.
- **Customizable Intensity**: Adjust the blur ratio to balance privacy needs with visual context.
- **Class-Specific Blurring**: Selectively blur only certain types of objects while leaving others visible.

For more detailed applications, check the [advantages of object blurring section](#advantages-of-object-blurring).

### Can I use Ultralytics YOLO26 to blur faces in a video for privacy reasons?

Yes, Ultralytics YOLO26 can be configured to detect and blur faces in videos to protect privacy. By training or using a pretrained model to specifically recognize faces, the detection results can be processed with [OpenCV](https://www.ultralytics.com/glossary/opencv) to apply a blur effect. Refer to our guide on [object detection with YOLO26](../models/yolo26.md) and modify the code to target face detection.

### How does YOLO26 compare to other object detection models like Faster R-CNN for object blurring?

Ultralytics YOLO26 typically outperforms models like Faster R-CNN in terms of speed, making it more suitable for real-time applications. While both models offer accurate detection, YOLO26's architecture is optimized for rapid inference, which is critical for tasks like real-time object blurring. Learn more about the technical differences and performance metrics in our [YOLO26 documentation](../models/yolo26.md).
