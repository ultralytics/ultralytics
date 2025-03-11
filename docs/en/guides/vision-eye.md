---
comments: true
description: Discover VisionEye's object mapping and tracking powered by Ultralytics YOLO11. Simulate human eye precision, track objects, and calculate distances effortlessly.
keywords: VisionEye, YOLO11, Ultralytics, object mapping, object tracking, distance calculation, computer vision, AI, machine learning, Python, tutorial
---

# VisionEye View Object Mapping using Ultralytics YOLO11 🚀

## What is VisionEye Object Mapping?

[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) VisionEye offers the capability for computers to identify and pinpoint objects, simulating the observational [precision](https://www.ultralytics.com/glossary/precision) of the human eye. This functionality enables computers to discern and focus on specific objects, much like the way the human eye observes details from a particular viewpoint.

<p align="center">
  <img width="800" src="https://github.com/ultralytics/docs/releases/download/0/visioneye-object-mapping-with-tracking.avif" alt="VisionEye View Object Mapping with Object Tracking using Ultralytics YOLO11">
</p>

!!! example "VisionEye Mapping using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Monitor objects position with visioneye
        yolo solutions visioneye show=True

        # Pass a source video
        yolo solutions visioneye source="path/to/video/file.mp4"

        # Monitor the specific classes
        yolo solutions visioneye classes=[0, 5]
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video/file.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("visioneye_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize vision eye object
        visioneye = solutions.VisionEye(
            show=True,  # display the output
            model="yolo11n.pt",  # use any model that Ultralytics support, i.e, YOLOv10
            classes=[0, 2],  # generate visioneye view for specific classes
            vision_point=(50, 50),  # the point, where vision will view objects and draw tracks
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

### `VisionEye` Arguments

Here's a table with the `VisionEye` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "vision_point"]) }}

You can also utilize various `track` arguments within the `VisionEye` solution:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Furthermore, some visualization arguments are supported, as listed below:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width"]) }}

## Note

For any inquiries, feel free to post your questions in the [Ultralytics Issue Section](https://github.com/ultralytics/ultralytics/issues/new/choose) or the discussion section mentioned below.

## FAQ

### How do I start using VisionEye Object Mapping with Ultralytics YOLO11?

To start using VisionEye Object Mapping with Ultralytics YOLO11, first, you'll need to install the Ultralytics YOLO package via pip. Then, you can use the sample code provided in the documentation to set up [object detection](https://www.ultralytics.com/glossary/object-detection) with VisionEye. Here's a simple example to get you started:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video/file.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("vision-eye-mapping.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init vision eye object
visioneye = solutions.VisionEye(
    show=True,  # display the output
    model="yolo11n.pt",  # use any model that Ultralytics support, i.e, YOLOv10
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

### Why should I use Ultralytics YOLO11 for object mapping and tracking?

Ultralytics YOLO11 is renowned for its speed, [accuracy](https://www.ultralytics.com/glossary/accuracy), and ease of integration, making it a top choice for object mapping and tracking. Key advantages include:

1. **State-of-the-art Performance**: Delivers high accuracy in real-time object detection.
2. **Flexibility**: Supports various tasks such as detection, tracking, and distance calculation.
3. **Community and Support**: Extensive documentation and active GitHub community for troubleshooting and enhancements.
4. **Ease of Use**: Intuitive API simplifies complex tasks, allowing for rapid deployment and iteration.

For more information on applications and benefits, check out the [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolov8/).

### How can I integrate VisionEye with other [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) tools like Comet or ClearML?

Ultralytics YOLO11 can integrate seamlessly with various machine learning tools like Comet and ClearML, enhancing experiment tracking, collaboration, and reproducibility. Follow the detailed guides on [how to use YOLOv5 with Comet](https://www.ultralytics.com/blog/how-to-use-yolov5-with-comet) and [integrate YOLO11 with ClearML](https://docs.ultralytics.com/integrations/clearml/) to get started.

For further exploration and integration examples, check our [Ultralytics Integrations Guide](https://docs.ultralytics.com/integrations/).
