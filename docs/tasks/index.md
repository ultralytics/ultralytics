---
comments: true
description: Learn how Ultralytics YOLOv8 AI framework supports detection, segmentation, classification, and pose/keypoint estimation tasks.
---

# Ultralytics YOLOv8 Tasks

YOLOv8 is an AI framework that supports multiple computer vision **tasks**. The framework can be used to
perform [detection](detect.md), [segmentation](segment.md), [classification](classify.md),
and [pose](pose.md) estimation. Each of these tasks has a different objective and use case.

<img width="1024" src="https://user-images.githubusercontent.com/26833433/212094133-6bb8c21c-3d47-41df-a512-81c5931054ae.png">

## [Detection](detect.md)

Detection is the primary task supported by YOLOv8. It involves detecting objects in an image or video frame and drawing
bounding boxes around them. The detected objects are classified into different categories based on their features.
YOLOv8 can detect multiple objects in a single image or video frame with high accuracy and speed.

[Detection Examples](detect.md){ .md-button .md-button--primary}

## [Segmentation](segment.md)

Segmentation is a task that involves segmenting an image into different regions based on the content of the image. Each
region is assigned a label based on its content. This task is useful in applications such as image segmentation and
medical imaging. YOLOv8 uses a variant of the U-Net architecture to perform segmentation.

[Segmentation Examples](segment.md){ .md-button .md-button--primary}

## [Classification](classify.md)

Classification is a task that involves classifying an image into different categories. YOLOv8 can be used to classify
images based on their content. It uses a variant of the EfficientNet architecture to perform classification.

[Classification Examples](classify.md){ .md-button .md-button--primary}

## [Pose](pose.md)

Pose/keypoint detection is a task that involves detecting specific points in an image or video frame. These points are
referred to as keypoints and are used to track movement or pose estimation. YOLOv8 can detect keypoints in an image or
video frame with high accuracy and speed.

[Pose Examples](pose.md){ .md-button .md-button--primary}

## Conclusion

YOLOv8 supports multiple tasks, including detection, segmentation, classification, and keypoints detection. Each of
these tasks has different objectives and use cases. By understanding the differences between these tasks, you can choose
the appropriate task for your computer vision application.