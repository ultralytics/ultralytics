---
comments: true
description: Learn how to estimate object speed using Ultralytics YOLO26 for applications in traffic control, autonomous navigation, and surveillance.
keywords: Ultralytics YOLO26, speed estimation, object tracking, computer vision, traffic control, autonomous navigation, surveillance, security
---

# Speed Estimation using Ultralytics YOLO26 🚀

## What is Speed Estimation?

[Speed estimation](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects) is the process of calculating the rate of movement of an object within a given context, often employed in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications. Using [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics/) you can now calculate the speed of objects using [object tracking](../modes/track.md) alongside distance and time data, crucial for tasks like traffic monitoring and surveillance. The accuracy of speed estimation directly influences the efficiency and reliability of various applications, making it a key component in the advancement of intelligent systems and real-time decision-making processes.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rCggzXRRSRo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Speed Estimation using Ultralytics YOLO26
</p>

!!! tip "Check Out Our Blog"

    For deeper insights into speed estimation, check out our blog post: [Ultralytics YOLO for Speed Estimation in Computer Vision Projects](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects)

## Advantages of Speed Estimation

- **Efficient Traffic Control:** Accurate speed estimation aids in managing traffic flow, enhancing safety, and reducing congestion on roadways.
- **Precise Autonomous Navigation:** In autonomous systems like [self-driving cars](https://www.ultralytics.com/solutions/ai-in-automotive), reliable speed estimation ensures safe and accurate vehicle navigation.
- **Enhanced Surveillance Security:** Speed estimation in surveillance analytics helps identify unusual behaviors or potential threats, improving the effectiveness of security measures.

## Real World Applications

|                                                                            Transportation                                                                             |                                                                              Transportation                                                                               |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Speed Estimation on Road using Ultralytics YOLO26](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/speed-estimation-on-road-using-ultralytics-yolov8.avif) | ![Speed Estimation on Bridge using Ultralytics YOLO26](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/speed-estimation-on-bridge-using-ultralytics-yolov8.avif) |
|                                                           Speed Estimation on Road using Ultralytics YOLO26                                                           |                                                            Speed Estimation on Bridge using Ultralytics YOLO26                                                            |

???+ warning "Speed is an Estimate"

    Speed will be an estimate and may not be completely accurate. Additionally, the estimation can vary on camera specifications and related factors.

!!! example "Speed Estimation using Ultralytics YOLO"

    === "CLI"

        ```bash
        # Run a speed example
        yolo solutions speed show=True

        # Pass a source video
        yolo solutions speed source="path/to/video.mp4"

        # Adjust meter per pixel value based on camera configuration
        yolo solutions speed meter_per_pixel=0.05
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("path/to/video.mp4")
        assert cap.isOpened(), "Error reading video file"

        # Video writer
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        video_writer = cv2.VideoWriter("speed_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        # Initialize speed estimation object
        speedestimator = solutions.SpeedEstimator(
            show=True,  # display the output
            model="yolo26n.pt",  # path to the YOLO26 model file.
            fps=fps,  # adjust speed based on frame per second
            # max_speed=120,  # cap speed to a max value (km/h) to avoid outliers
            # max_hist=5,  # minimum frames object tracked before computing speed
            # meter_per_pixel=0.05,  # highly depends on the camera configuration
            # classes=[0, 2],  # estimate speed of specific classes.
            # line_width=2,  # adjust the line width for bounding boxes
        )

        # Process video
        while cap.isOpened():
            success, im0 = cap.read()

            if not success:
                print("Video frame is empty or processing is complete.")
                break

            results = speedestimator(im0)

            # print(results)  # access the output

            video_writer.write(results.plot_im)  # write the processed frame.

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()  # destroy all opened windows
        ```

### `SpeedEstimator` Arguments

Here's a table with the `SpeedEstimator` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "fps", "max_hist", "meter_per_pixel", "max_speed"]) }}

The `SpeedEstimator` solution allows the use of `track` parameters:

{% from "macros/track-args.md" import param_table %}
{{ param_table(["tracker", "conf", "iou", "classes", "verbose", "device"]) }}

Additionally, the following visualization options are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "line_width", "show_conf", "show_labels"]) }}

## FAQ

### How do I estimate object speed using Ultralytics YOLO26?

Estimating object speed with Ultralytics YOLO26 involves combining [object detection](https://www.ultralytics.com/glossary/object-detection) and tracking techniques. First, you need to detect objects in each frame using the YOLO26 model. Then, track these objects across frames to calculate their movement over time. Finally, use the distance traveled by the object between frames and the frame rate to estimate its speed.

**Example**:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize SpeedEstimator
speedestimator = solutions.SpeedEstimator(
    model="yolo26n.pt",
    show=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = speedestimator(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

For more details, refer to our [official blog post](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects).

### What are the benefits of using Ultralytics YOLO26 for speed estimation in traffic management?

Using Ultralytics YOLO26 for speed estimation offers significant advantages in traffic management:

- **Enhanced Safety**: Accurately estimate vehicle speeds to detect over-speeding and improve road safety.
- **Real-Time Monitoring**: Benefit from YOLO26's real-time object detection capability to monitor traffic flow and congestion effectively.
- **Scalability**: Deploy the model on various hardware setups, from [edge devices](https://docs.ultralytics.com/guides/nvidia-jetson/) to servers, ensuring flexible and scalable solutions for large-scale implementations.

For more applications, see [advantages of speed estimation](#advantages-of-speed-estimation).

### Can YOLO26 be integrated with other AI frameworks like [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) or [PyTorch](https://www.ultralytics.com/glossary/pytorch)?

Yes, YOLO26 can be integrated with other AI frameworks like TensorFlow and PyTorch. Ultralytics provides support for exporting YOLO26 models to various formats like [ONNX](../integrations/onnx.md), [TensorRT](../integrations/tensorrt.md), and [CoreML](../integrations/coreml.md), ensuring smooth interoperability with other ML frameworks.

To export a YOLO26 model to ONNX format:

```bash
yolo export model=yolo26n.pt format=onnx
```

Learn more about exporting models in our [guide on export](../modes/export.md).

### How accurate is the speed estimation using Ultralytics YOLO26?

The [accuracy](https://www.ultralytics.com/glossary/accuracy) of speed estimation using Ultralytics YOLO26 depends on several factors, including the quality of the object tracking, the resolution and frame rate of the video, and environmental variables. While the speed estimator provides reliable estimates, it may not be 100% accurate due to variances in frame processing speed and object occlusion.

**Note**: Always consider margin of error and validate the estimates with ground truth data when possible.

For further accuracy improvement tips, check the [Arguments `SpeedEstimator` section](#speedestimator-arguments).
