---
comments: true
description: Explore Ultralytics YOLO11 for detection, segmentation, classification, OBB, and pose estimation with high accuracy and speed. Learn how to apply each task.
keywords: Ultralytics YOLO11, detection, segmentation, classification, oriented object detection, pose estimation, computer vision, AI framework
---

# Ultralytics YOLO11 Tasks

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-tasks-banner.avif" alt="Ultralytics YOLO supported tasks">

YOLO11 is an AI framework that supports multiple [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) **tasks**. The framework can be used to perform [detection](detect.md), [segmentation](segment.md), [obb](obb.md), [classification](classify.md), and [pose](pose.md) estimation. Each of these tasks has a different objective and use case.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/NAs-cfq9BDw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Explore Ultralytics YOLO Tasks: <a href="https://www.ultralytics.com/glossary/object-detection">Object Detection</a>, Segmentation, OBB, Tracking, and Pose Estimation.
</p>

## [Detection](detect.md)

Detection is the primary task supported by YOLO11. It involves detecting objects in an image or video frame and drawing bounding boxes around them. The detected objects are classified into different categories based on their features. YOLO11 can detect multiple objects in a single image or video frame with high [accuracy](https://www.ultralytics.com/glossary/accuracy) and speed.

[Detection Examples](detect.md){ .md-button }

## [Segmentation](segment.md)

Segmentation is a task that involves segmenting an image into different regions based on the content of the image. Each region is assigned a label based on its content. This task is useful in applications such as [image segmentation](https://www.ultralytics.com/glossary/image-segmentation) and medical imaging. YOLO11 uses a variant of the U-Net architecture to perform segmentation.

[Segmentation Examples](segment.md){ .md-button }

## [Classification](classify.md)

Classification is a task that involves classifying an image into different categories. YOLO11 can be used to classify images based on their content. It uses a variant of the EfficientNet architecture to perform classification.

[Classification Examples](classify.md){ .md-button }

## [Pose](pose.md)

Pose/keypoint detection is a task that involves detecting specific points in an image or video frame. These points are referred to as keypoints and are used to track movement or pose estimation. YOLO11 can detect keypoints in an image or video frame with high accuracy and speed.

[Pose Examples](pose.md){ .md-button }

## [OBB](obb.md)

Oriented object detection goes a step further than regular object detection with introducing an extra angle to locate objects more accurate in an image. YOLO11 can detect rotated objects in an image or video frame with high accuracy and speed.

[Oriented Detection](obb.md){ .md-button }

## Conclusion

YOLO11 supports multiple tasks, including detection, segmentation, classification, oriented object detection and keypoints detection. Each of these tasks has different objectives and use cases. By understanding the differences between these tasks, you can choose the appropriate task for your computer vision application.

## FAQ

### What tasks can Ultralytics YOLO11 perform?

Ultralytics YOLO11 is a versatile AI framework capable of performing various computer vision tasks with high accuracy and speed. These tasks include:

- **[Detection](detect.md):** Identifying and localizing objects in images or video frames by drawing bounding boxes around them.
- **[Segmentation](segment.md):** Segmenting images into different regions based on their content, useful for applications like medical imaging.
- **[Classification](classify.md):** Categorizing entire images based on their content, leveraging variants of the EfficientNet architecture.
- **[Pose estimation](pose.md):** Detecting specific keypoints in an image or video frame to track movements or poses.
- **[Oriented Object Detection (OBB)](obb.md):** Detecting rotated objects with an added orientation angle for enhanced accuracy.

### How do I use Ultralytics YOLO11 for object detection?

To use Ultralytics YOLO11 for object detection, follow these steps:

1. Prepare your dataset in the appropriate format.
2. Train the YOLO11 model using the detection task.
3. Use the model to make predictions by feeding in new images or video frames.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pre-trained YOLO model (adjust model type as needed)
        model = YOLO("yolo11n.pt")  # n, s, m, l, x versions available

        # Perform object detection on an image
        results = model.predict(source="image.jpg")  # Can also use video, directory, URL, etc.

        # Display the results
        results[0].show()  # Show the first image results
        ```

    === "CLI"

        ```bash
        # Run YOLO detection from the command line
        yolo detect model=yolo11n.pt source="image.jpg"  # Adjust model and source as needed
        ```

For more detailed instructions, check out our [detection examples](detect.md).

### What are the benefits of using YOLO11 for segmentation tasks?

Using YOLO11 for segmentation tasks provides several advantages:

1. **High Accuracy:** The segmentation task leverages a variant of the U-Net architecture to achieve precise segmentation.
2. **Speed:** YOLO11 is optimized for real-time applications, offering quick processing even for high-resolution images.
3. **Multiple Applications:** It is ideal for medical imaging, autonomous driving, and other applications requiring detailed image segmentation.

Learn more about the benefits and use cases of YOLO11 for segmentation in the [segmentation section](segment.md).

### Can Ultralytics YOLO11 handle pose estimation and keypoint detection?

Yes, Ultralytics YOLO11 can effectively perform pose estimation and keypoint detection with high accuracy and speed. This feature is particularly useful for tracking movements in sports analytics, healthcare, and human-computer interaction applications. YOLO11 detects keypoints in an image or video frame, allowing for precise pose estimation.

For more details and implementation tips, visit our [pose estimation examples](pose.md).

### Why should I choose Ultralytics YOLO11 for oriented object detection (OBB)?

Oriented Object Detection (OBB) with YOLO11 provides enhanced [precision](https://www.ultralytics.com/glossary/precision) by detecting objects with an additional angle parameter. This feature is beneficial for applications requiring accurate localization of rotated objects, such as aerial imagery analysis and warehouse automation.

- **Increased Precision:** The angle component reduces false positives for rotated objects.
- **Versatile Applications:** Useful for tasks in geospatial analysis, robotics, etc.

Check out the [Oriented Object Detection section](obb.md) for more details and examples.
