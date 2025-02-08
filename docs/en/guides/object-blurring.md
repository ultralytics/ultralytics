---
comments: true
description: Learn how to use Ultralytics YOLO11 for real-time object blurring to enhance privacy and focus in your images and videos.
keywords: YOLO11, object blurring, real-time processing, privacy protection, image manipulation, video editing, Ultralytics
---

# Object Blurring using Ultralytics YOLO11 🚀

## What is Object Blurring?

Object blurring with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) involves applying a blurring effect to specific detected objects in an image or video. This can be achieved using the YOLO11 model capabilities to identify and manipulate objects within a given scene.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ydGdibB5Mds"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Object Blurring using Ultralytics YOLO11
</p>

## Advantages of Object Blurring?

- **Privacy Protection**: Object blurring is an effective tool for safeguarding privacy by concealing sensitive or personally identifiable information in images or videos.
- **Selective Focus**: YOLO11 allows for selective blurring, enabling users to target specific objects, ensuring a balance between privacy and retaining relevant visual information.
- **Real-time Processing**: YOLO11's efficiency enables object blurring in real-time, making it suitable for applications requiring on-the-fly privacy enhancements in dynamic environments.

## Code

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("Path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_blurring_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init ObjectBlurrer
blurrer = solutions.ObjectBlurrer(
    show=True,  # display the output
    model="yolo11n.pt",  # model for object blurring i.e. yolo11m.pt
    # line_width=2,   # width of bounding box.
    # classes=[0, 2], # count specific classes i.e, person and car with COCO pretrained model.
)

# Adjust percentage of blur intensity
blurrer.set_blur_ratio(0.6)  # the value in range 0.1 - 1.0

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = blurrer.blur(im0)

    # Access the output
    # print(f"Total tracks: , {results['total_tracks']}")

    video_writer.write(results["plot_im"])  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
```

### Argument `ObjectBlurrer`

Here's a table with the `ObjectBlurrer` arguments:

| Name         | Type    | Default        | Description                                                                                                                                                                  |
| ------------ | ------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`      | `str`   | `None`         | Path to Ultralytics YOLO Model File                                                                                                                                          |
| `line_width` | `int`   | `2`            | Line thickness for bounding boxes.                                                                                                                                           |
| `show`       | `bool`  | `False`        | Flag to control whether to display the video stream.                                                                                                                         |
| `conf`       | `float` | `0.3`          | Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives.                                                 |
| `iou`        | `float` | `0.5`          | Sets the [Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for filtering overlapping detections.                   |
| `classes`    | `list`  | `None`         | Filters results by class index. For example, `classes=[0, 2, 3]` only tracks the specified classes.                                                                          |
| `max_det`    | `int`   | `300`          | Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes. |
| `verbose`    | `bool`  | `True`         | Controls the display of solutions results, providing a visual output of tracked objects.                                                                                     |
| `tracker`    | `str`   | `botsort.yaml` | Specifies the tracking algorithm to use, e.g., `bytetrack.yaml` or `botsort.yaml`.                                                                                           |

## FAQ

### What is object blurring with Ultralytics YOLO11?

Object blurring with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) involves automatically detecting and applying a blurring effect to specific objects in images or videos. This technique enhances privacy by concealing sensitive information while retaining relevant visual data. YOLO11's real-time processing capabilities make it suitable for applications requiring immediate privacy protection and selective focus adjustments.

### How can I implement real-time object blurring using YOLO11?

To implement real-time object blurring with YOLO11, follow the provided Python example. This involves using YOLO11 for [object detection](https://www.ultralytics.com/glossary/object-detection) and OpenCV for applying the blur effect. Here's a simplified version:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("Path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("object_blurring_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init ObjectBlurrer
blurrer = solutions.ObjectBlurrer(
    show=True,  # display the output
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object blurring using YOLO11 OBB model.
    blur_ratio=0.5,  # set blur percentage i.e 0.7 for 70% blurred detected objects
    # line_width=2,         # Width of bounding box.
    # classes=[0, 2],       # count specific classes i.e, person and car with COCO pretrained model.
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break
    results = blurrer.blur(im0)
    video_writer.write(results["im0"])

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

### What are the benefits of using Ultralytics YOLO11 for object blurring?

Ultralytics YOLO11 offers several advantages for object blurring:

- **Privacy Protection**: Effectively obscure sensitive or identifiable information.
- **Selective Focus**: Target specific objects for blurring, maintaining essential visual content.
- **Real-time Processing**: Execute object blurring efficiently in dynamic environments, suitable for instant privacy enhancements.

For more detailed applications, check the [advantages of object blurring section](#advantages-of-object-blurring).

### Can I use Ultralytics YOLO11 to blur faces in a video for privacy reasons?

Yes, Ultralytics YOLO11 can be configured to detect and blur faces in videos to protect privacy. By training or using a pre-trained model to specifically recognize faces, the detection results can be processed with [OpenCV](https://www.ultralytics.com/glossary/opencv) to apply a blur effect. Refer to our guide on [object detection with YOLO11](https://docs.ultralytics.com/models/yolov8/) and modify the code to target face detection.

### How does YOLO11 compare to other object detection models like Faster R-CNN for object blurring?

Ultralytics YOLO11 typically outperforms models like Faster R-CNN in terms of speed, making it more suitable for real-time applications. While both models offer accurate detection, YOLO11's architecture is optimized for rapid inference, which is critical for tasks like real-time object blurring. Learn more about the technical differences and performance metrics in our [YOLO11 documentation](https://docs.ultralytics.com/models/yolov8/).
