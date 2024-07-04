---
comments: true
description: Explore Ultralytics YOLOv8 for detection, segmentation, classification, OBB, and pose estimation with high accuracy and speed. Learn how to apply each task.
keywords: Ultralytics YOLOv8, detection, segmentation, classification, oriented object detection, pose estimation, computer vision, AI framework
---

# Ultralytics YOLOv8 Tasks

<br>
<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Ultralytics YOLO supported tasks">

YOLOv8 is an AI framework that supports multiple computer vision **tasks**. The framework can be used to perform [detection](detect.md), [segmentation](segment.md), [obb](obb.md), [classification](classify.md), and [pose](pose.md) estimation. Each of these tasks has a different objective and use case.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/NAs-cfq9BDw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Explore Ultralytics YOLO Tasks: Object Detection, Segmentation, OBB, Tracking, and Pose Estimation.
</p>

## [Detection](detect.md)

Detection is the primary task supported by YOLOv8. It involves detecting objects in an image or video frame and drawing bounding boxes around them. The detected objects are classified into different categories based on their features. YOLOv8 can detect multiple objects in a single image or video frame with high accuracy and speed.

[Detection Examples](detect.md){ .md-button }

## [Segmentation](segment.md)

Segmentation is a task that involves segmenting an image into different regions based on the content of the image. Each region is assigned a label based on its content. This task is useful in applications such as image segmentation and medical imaging. YOLOv8 uses a variant of the U-Net architecture to perform segmentation.

[Segmentation Examples](segment.md){ .md-button }

## [Classification](classify.md)

Classification is a task that involves classifying an image into different categories. YOLOv8 can be used to classify images based on their content. It uses a variant of the EfficientNet architecture to perform classification.

[Classification Examples](classify.md){ .md-button }

## [Pose](pose.md)

Pose/keypoint detection is a task that involves detecting specific points in an image or video frame. These points are referred to as keypoints and are used to track movement or pose estimation. YOLOv8 can detect keypoints in an image or video frame with high accuracy and speed.

[Pose Examples](pose.md){ .md-button }

## [OBB](obb.md)

Oriented object detection goes a step further than regular object detection with introducing an extra angle to locate objects more accurate in an image. YOLOv8 can detect rotated objects in an image or video frame with high accuracy and speed.

[Oriented Detection](obb.md){ .md-button }

## Conclusion

YOLOv8 supports multiple tasks, including detection, segmentation, classification, oriented object detection and keypoints detection. Each of these tasks has different objectives and use cases. By understanding the differences between these tasks, you can choose the appropriate task for your computer vision application.

## FAQ

### What tasks does Ultralytics YOLOv8 support and how can I apply them?

Ultralytics YOLOv8 supports several computer vision tasks: [detection](detect.md), [segmentation](segment.md), [classification](classify.md), [oriented object detection](obb.md), and [pose estimation](pose.md). Each task has unique objectives and applications. For instance, detection involves identifying and classifying multiple objects within an image, while segmentation divides an image into different regions. You can apply these tasks by following the specific examples and detailed documentation provided for each [here](https://docs.ultralytics.com/tasks/).

### How accurate and fast is YOLOv8 in various computer vision tasks?

Ultralytics YOLOv8 is known for its high accuracy and speed across various computer vision tasks, including object detection, segmentation, classification, oriented object detection, and pose estimation. For example, in object detection, YOLOv8 can identify and classify multiple objects within a single frame with outstanding precision and minimal latency. You can find more specific details on performance metrics and benchmarks in the [detection](detect.md) and [segmentation](segment.md) pages.

### Why should I use Ultralytics YOLOv8 for object detection?

Ultralytics YOLOv8 offers superior accuracy and rapid inference, making it ideal for real-time applications. Its ability to handle multiple objects and provide precise classification results makes it suitable for diverse applications such as surveillance, healthcare, and automotive industries. For practical implementations, refer to the [detection examples](detect.md).

### What makes Ultralytics YOLOv8 suitable for segmentation tasks?

YOLOv8 uses a variant of the U-Net architecture to perform segmentation tasks, enabling it to segment images into different regions with high precision. This is particularly useful in medical imaging and other applications requiring detailed analysis. For a deeper dive, check out the [segmentation examples](segment.md) in the documentation.

### How can Ultralytics YOLOv8 be used for pose estimation?

Pose estimation with YOLOv8 involves identifying specific keypoints in an image or video frame, which can be used to track human movement or body posture accurately. This makes it highly useful in sports analytics, motion capture, and health monitoring. Learn more about pose estimation with YOLOv8 by exploring the [pose estimation section](pose.md).

For further guidance on using YOLOv8 and its features, explore more [Ultralytics YOLO tasks](https://docs.ultralytics.com/tasks/).
