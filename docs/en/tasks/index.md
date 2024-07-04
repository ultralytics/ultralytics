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

### What tasks can Ultralytics YOLOv8 perform?

Ultralytics YOLOv8 supports multiple computer vision tasks, including object detection, segmentation, classification, oriented object detection (OBB), and pose estimation. Each task caters to different use cases:
- **[Detection](detect.md)**: Identifies objects in images or videos, drawing bounding boxes around them.
- **[Segmentation](segment.md)**: Divides an image into different regions and labels them based on content.
- **[Classification](classify.md)**: Classifies images into various categories.
- **[Pose Estimation](pose.md)**: Detects keypoints in images or videos for tracking movement or estimating poses.
- **[Oriented Object Detection (OBB)](obb.md)**: Detects objects with orientation, useful for rotated object recognition.

Learn more about each task on their respective documentation pages.

### How accurate is Ultralytics YOLOv8 for object detection?

Ultralytics YOLOv8 is known for its high accuracy and speed in object detection, making it ideal for real-time applications. The model's architecture includes enhancements such as better bounding box predictions and improved anchor-free detection, contributing to its accuracy. You can see detailed benchmarks and accuracy metrics on the [detection](detect.md) page.

### Why should I use Ultralytics YOLOv8 for segmentation tasks?

Ultralytics YOLOv8 uses advanced architectures like U-Net variants for segmentation tasks, which offer high precision and performance. This makes it suitable for applications such as medical imaging and object segmentation in various industries. For practical implementation details, visit the [segmentation](segment.md) page.

### Can Ultralytics YOLOv8 handle pose estimation for real-time applications?

Yes, Ultralytics YOLOv8 performs pose estimation with high accuracy and real-time speeds. It detects specific keypoints in images or video frames, which is essential for applications in sports analytics, human-computer interaction, and security. Learn how to set up and use pose estimation with Ultralytics YOLOv8 on the [pose](pose.md) page.

### How does Ultralytics YOLOv8 manage oriented object detection (OBB)?

Ultralytics YOLOv8 supports oriented object detection (OBB), which includes detecting the orientation of objects in images. This feature is useful for applications requiring angle-specific object detection, such as aerial imagery analysis and industrial automation. To explore more about oriented object detection, refer to the [OBB](obb.md) page.