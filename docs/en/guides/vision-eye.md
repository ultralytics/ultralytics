---
title: VisionEye Object Mapping with YOLO26
comments: true
description: Discover VisionEye's object mapping and tracking powered by Ultralytics YOLO26. Simulate human eye precision, track objects, and calculate distances effortlessly.
keywords: VisionEye, YOLO26, Ultralytics, object mapping, object tracking, distance calculation, computer vision, AI, machine learning, Python, tutorial
---

# VisionEye View Object Mapping using Ultralytics YOLO26 🚀

## What is VisionEye Object Mapping?

[Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) VisionEye offers the capability for computers to identify and pinpoint objects, simulating the observational [precision](https://www.ultralytics.com/glossary/precision) of the human eye. This functionality enables computers to discern and focus on specific objects, much like the way the human eye observes details from a particular viewpoint.

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/visioneye-object-mapping-with-tracking.avif" alt="VisionEye object mapping with YOLO tracking">
</p>

## Map Objects with YOLO26

VisionEye fixes a single observation point in the frame and draws a ray from it to every tracked object, so you can visualize how a scene looks from one viewpoint. Set `vision_point` to the observer's pixel coordinates, then run the solution over your video with the Python API or the CLI.

!!! example "VisionEye Mapping using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Monitor objects position with visioneye
        yolo solutions visioneye show=True

        # Pass a source video
        yolo solutions visioneye source="path/to/video.mp4"

        # Monitor the specific classes
        yolo solutions visioneye classes="[0, 5]"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("visioneye_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize vision eye object
        visioneye = solutions.VisionEye(
            show=True,  # display the output
            model="yolo26n.pt",  # use any model that Ultralytics supports, e.g., YOLOv10
            classes=[0, 2],  # generate visioneye view for specific classes
            vision_point=(50, 50),  # the point where VisionEye will view objects and draw tracks
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            results = visioneye(im0)

            print(results)  # access the output

            video_writer.write(results.plot_im)  # write the video file

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

        The `vision_point` tuple represents the observer's position in pixel coordinates. Adjust it to match the camera perspective so the rendered rays correctly illustrate how objects relate to the chosen viewpoint.

### `VisionEye` Arguments

Here's a table with the `VisionEye` arguments:

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `model` | `str` | `None` | Path to an Ultralytics YOLO model file. |
| `vision_point` | `tuple[int, int]` | `(20, 20)` | The point where vision will track objects and draw paths using VisionEye Solution. |


You can also utilize various `track` arguments within the `VisionEye` solution:

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `tracker` | `str` | `'tracktrack.yaml'` | Specifies the tracking algorithm to use. Built-in options: `botsort.yaml`, `bytetrack.yaml`, `ocsort.yaml`, `deepocsort.yaml`, `fasttrack.yaml`, `tracktrack.yaml`. |
| `conf` | `float` | `0.1` | Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives. |
| `iou` | `float` | `0.7` | Sets the [Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for filtering overlapping detections. |
| `classes` | `list` | `None` | Filters results by class index. For example, `classes=[0, 2, 3]` only tracks the specified classes. |
| `verbose` | `bool` | `True` | Controls the display of tracking results, providing a visual output of tracked objects. |
| `device` | `str` | `None` | Specifies the device for inference (e.g., `cpu`, `cuda:0` or `0`). Allows users to select between CPU, a specific GPU, or other compute devices for model execution. |


Furthermore, some visualization arguments are supported, as listed below:

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `show` | `bool` | `False` | If `True`, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing. |
| `line_width` | `int or None` | `None` | Specifies the line width of bounding boxes. If `None`, the line width is automatically adjusted based on the image size. Provides visual customization for clarity. |
| `show_conf` | `bool` | `True` | Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection. |
| `show_labels` | `bool` | `True` | Displays labels for each detection in the visual output. Provides immediate understanding of detected objects. |


## How VisionEye Works

VisionEye works by establishing a fixed vision point in the frame and drawing lines from this point to detected objects. This simulates how human vision focuses on multiple objects from a single viewpoint. The solution uses [object tracking](../modes/track.md) to maintain consistent identification of objects across frames, creating a visual representation of the spatial relationship between the observer (vision point) and the objects in the scene.

The `process` method in the VisionEye class performs several key operations:

1. Extracts tracks (bounding boxes, classes, and masks) from the input image
2. Creates an annotator to draw bounding boxes and labels
3. For each detected object, draws a box label and creates a vision line from the vision point
4. Returns the annotated image with tracking statistics

This approach is particularly useful for applications requiring spatial awareness and object relationship visualization, such as surveillance systems, autonomous navigation, and interactive installations.

## Applications of VisionEye

VisionEye object mapping has numerous practical applications across various industries:

- **Security and Surveillance**: Monitor multiple objects of interest from a fixed camera position
- **Retail Analytics**: Track customer movement patterns in relation to store displays
- **Sports Analysis**: Analyze player positioning and movement from a coach's perspective
- **Autonomous Vehicles**: Visualize how a vehicle "sees" and prioritizes objects in its environment
- **Human-Computer Interaction**: Create more intuitive interfaces that respond to spatial relationships

By combining VisionEye with other Ultralytics solutions like [distance calculation](distance-calculation.md) or [speed estimation](speed-estimation.md), you can build comprehensive systems that not only track objects but also understand their spatial relationships and behaviors.

## FAQ

### How do I start using VisionEye Object Mapping with Ultralytics YOLO26?

To start using VisionEye Object Mapping with Ultralytics YOLO26, first, you'll need to install the Ultralytics YOLO package via pip. Then, you can use the sample code provided in the documentation to set up [object detection](https://www.ultralytics.com/glossary/object-detection) with VisionEye. Here's a simple example to get you started:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("vision-eye-mapping.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init vision eye object
visioneye = solutions.VisionEye(
    show=True,  # display the output
    model="yolo26n.pt",  # use any model that Ultralytics supports, e.g., YOLOv10
    classes=[0, 2],  # generate visioneye view for specific classes
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = visioneye(im0)

    print(results)  # access the output

    video_writer.write(results.plot_im)  # write the video file

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
```

### Why should I use Ultralytics YOLO26 for object mapping and tracking?

Ultralytics YOLO26 is renowned for its speed, [accuracy](https://www.ultralytics.com/glossary/accuracy), and ease of integration, making it a top choice for object mapping and tracking. Key advantages include:

1. **State-of-the-art Performance**: Delivers high accuracy in real-time object detection.
2. **Flexibility**: Supports various tasks such as detection, tracking, and distance calculation.
3. **Community and Support**: Extensive documentation and active GitHub community for troubleshooting and enhancements.
4. **Ease of Use**: Intuitive API simplifies complex tasks, allowing for rapid deployment and iteration.

For more information on applications and benefits, check out the [Ultralytics YOLO26 documentation](../models/yolo26.md).

### How can I integrate VisionEye with other [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) tools like Comet or ClearML?

Ultralytics YOLO26 can integrate seamlessly with various machine learning tools like Comet and ClearML, enhancing experiment tracking, collaboration, and reproducibility. Follow the detailed guides on [how to use YOLOv5 with Comet](https://www.ultralytics.com/blog/how-to-use-yolov5-with-comet) and [integrate YOLO26 with ClearML](../integrations/clearml.md) to get started.

For further exploration and integration examples, check our [Ultralytics Integrations Guide](../integrations/index.md).
