---
comments: true
description: Learn how to estimate object speed using Ultralytics YOLO11 for applications in traffic control, autonomous navigation, and surveillance.
keywords: Ultralytics YOLO11, speed estimation, object tracking, computer vision, traffic control, autonomous navigation, surveillance, security
---

# Speed Estimation using Ultralytics YOLO11 🚀

## What is Speed Estimation?

[Speed estimation](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects) is the process of calculating the rate of movement of an object within a given context, often employed in [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) applications. Using [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics/) you can now calculate the speed of object using [object tracking](../modes/track.md) alongside distance and time data, crucial for tasks like traffic and surveillance. The accuracy of speed estimation directly influences the efficiency and reliability of various applications, making it a key component in the advancement of intelligent systems and real-time decision-making processes.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rCggzXRRSRo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Speed Estimation using Ultralytics YOLO11
</p>

!!! tip "Check Out Our Blog"

    For deeper insights into speed estimation, check out our blog post: [Ultralytics YOLO11 for Speed Estimation in Computer Vision Projects](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects)

## Advantages of Speed Estimation?

- **Efficient Traffic Control:** Accurate speed estimation aids in managing traffic flow, enhancing safety, and reducing congestion on roadways.
- **Precise Autonomous Navigation:** In autonomous systems like self-driving cars, reliable speed estimation ensures safe and accurate vehicle navigation.
- **Enhanced Surveillance Security:** Speed estimation in surveillance analytics helps identify unusual behaviors or potential threats, improving the effectiveness of security measures.

## Real World Applications

|                                                                            Transportation                                                                            |                                                                              Transportation                                                                              |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Speed Estimation on Road using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/speed-estimation-on-road-using-ultralytics-yolov8.avif) | ![Speed Estimation on Bridge using Ultralytics YOLO11](https://github.com/ultralytics/docs/releases/download/0/speed-estimation-on-bridge-using-ultralytics-yolov8.avif) |
|                                                          Speed Estimation on Road using Ultralytics YOLO11                                                           |                                                           Speed Estimation on Bridge using Ultralytics YOLO11                                                            |

!!! example "Speed Estimation using YOLO11 Example"

    === "Speed Estimation"

        ```python
        import cv2

        from ultralytics import solutions

        cap = cv2.VideoCapture("Path/to/video/file.mp4")

        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        video_writer = cv2.VideoWriter("speed_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        speed_region = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

        speed = solutions.SpeedEstimator(model="yolo11n.pt", region=speed_region, show=True)

        while cap.isOpened():
            success, im0 = cap.read()

            if success:
                out = speed.estimate_speed(im0)
                video_writer.write(im0)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            print("Video frame is empty or video processing has been successfully completed.")
            break

        cap.release()
        cv2.destroyAllWindows()
        ```

???+ warning "Speed is Estimate"

    Speed will be an estimate and may not be completely accurate. Additionally, the estimation can vary depending on GPU speed.

### Arguments `SpeedEstimator`

| Name         | Type   | Default                    | Description                                          |
| ------------ | ------ | -------------------------- | ---------------------------------------------------- |
| `model`      | `str`  | `None`                     | Path to Ultralytics YOLO Model File                  |
| `region`     | `list` | `[(20, 400), (1260, 400)]` | List of points defining the counting region.         |
| `line_width` | `int`  | `2`                        | Line thickness for bounding boxes.                   |
| `show`       | `bool` | `False`                    | Flag to control whether to display the video stream. |

### Arguments `model.track`

{% include "macros/track-args.md" %}

## FAQ

### How do I estimate object speed using Ultralytics YOLO11?

Estimating object speed with Ultralytics YOLO11 involves combining [object detection](https://www.ultralytics.com/glossary/object-detection) and tracking techniques. First, you need to detect objects in each frame using the YOLO11 model. Then, track these objects across frames to calculate their movement over time. Finally, use the distance traveled by the object between frames and the frame rate to estimate its speed.

**Example**:

```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("path/to/video/file.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize SpeedEstimator
speed_obj = solutions.SpeedEstimator(
    region=[(0, 360), (1280, 360)],
    model="yolo11n.pt",
    show=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    im0 = speed_obj.estimate_speed(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
```

For more details, refer to our [official blog post](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects).

### What are the benefits of using Ultralytics YOLO11 for speed estimation in traffic management?

Using Ultralytics YOLO11 for speed estimation offers significant advantages in traffic management:

- **Enhanced Safety**: Accurately estimate vehicle speeds to detect over-speeding and improve road safety.
- **Real-Time Monitoring**: Benefit from YOLO11's real-time object detection capability to monitor traffic flow and congestion effectively.
- **Scalability**: Deploy the model on various hardware setups, from edge devices to servers, ensuring flexible and scalable solutions for large-scale implementations.

For more applications, see [advantages of speed estimation](#advantages-of-speed-estimation).

### Can YOLO11 be integrated with other AI frameworks like [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) or [PyTorch](https://www.ultralytics.com/glossary/pytorch)?

Yes, YOLO11 can be integrated with other AI frameworks like TensorFlow and PyTorch. Ultralytics provides support for exporting YOLO11 models to various formats like ONNX, TensorRT, and CoreML, ensuring smooth interoperability with other ML frameworks.

To export a YOLO11 model to ONNX format:

```bash
yolo export --weights yolo11n.pt --include onnx
```

Learn more about exporting models in our [guide on export](../modes/export.md).

### How accurate is the speed estimation using Ultralytics YOLO11?

The [accuracy](https://www.ultralytics.com/glossary/accuracy) of speed estimation using Ultralytics YOLO11 depends on several factors, including the quality of the object tracking, the resolution and frame rate of the video, and environmental variables. While the speed estimator provides reliable estimates, it may not be 100% accurate due to variances in frame processing speed and object occlusion.

**Note**: Always consider margin of error and validate the estimates with ground truth data when possible.

For further accuracy improvement tips, check the [Arguments `SpeedEstimator` section](#arguments-speedestimator).

### Why choose Ultralytics YOLO11 over other object detection models like TensorFlow Object Detection API?

Ultralytics YOLO11 offers several advantages over other object detection models, such as the TensorFlow Object Detection API:

- **Real-Time Performance**: YOLO11 is optimized for real-time detection, providing high speed and accuracy.
- **Ease of Use**: Designed with a user-friendly interface, YOLO11 simplifies model training and deployment.
- **Versatility**: Supports multiple tasks, including object detection, segmentation, and pose estimation.
- **Community and Support**: YOLO11 is backed by an active community and extensive documentation, ensuring developers have the resources they need.

For more information on the benefits of YOLO11, explore our detailed [model page](../models/yolov8.md).
