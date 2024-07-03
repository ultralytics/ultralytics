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

### How do I get started with YOLOv8 for object detection?

To get started with YOLOv8 for object detection, first, you need to install the necessary dependencies and set up the environment. You can follow the [quickstart guide](https://docs.ultralytics.com/quickstart/) to install YOLOv8 using pip, conda, or Docker. Once installed, you can use the [detection](detect.md) mode to perform object detection on your images or videos with high accuracy and speed.

### What tasks does YOLOv8 support besides object detection?

YOLOv8 supports multiple computer vision tasks beyond object detection, including [segmentation](segment.md), [classification](classify.md), [oriented object detection (OBB)](obb.md), and [pose estimation](pose.md). Each of these tasks serves different applications and can be utilized based on your project requirements. For example, segmentation is ideal for delineating regions in images, while pose estimation is used for tracking specific points in human movement.

### Why should I use YOLOv8 for pose estimation?

Using YOLOv8 for pose estimation offers several advantages. YOLOv8's pose estimation feature can detect specific keypoints in images or video frames with high accuracy and speed. This makes it ideal for applications like human activity recognition, sports analysis, and biometric monitoring. You can learn more about implementing pose estimation with YOLOv8 in the [pose](pose.md) section of the documentation.

### What is the benefit of oriented object detection (OBB) in YOLOv8?

Oriented object detection (OBB) in YOLOv8 adds an angle component to traditional bounding box detection, allowing the model to more accurately locate objects regardless of their rotation. This feature is particularly useful for applications such as aerial imagery analysis, where objects like buildings and vehicles may appear in various orientations. For more details, check out the [OBB](obb.md) section.

### Where can I find examples of YOLOv8 applications?

You can find various examples of YOLOv8 applications in the respective sections of the documentation. For instance, visit the [Detection Examples](detect.md) and [Segmentation Examples](segment.md) pages to see how YOLOv8 is used in real-world scenarios. Each section provides practical examples and code snippets to help you better understand and implement these tasks in your projects.