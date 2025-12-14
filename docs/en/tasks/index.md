---
comments: true
description: Explore Ultralytics YOLO11 for detection, segmentation, classification, OBB, and pose estimation with high accuracy and speed. Learn how to apply each task.
keywords: Ultralytics YOLO11, detection, segmentation, classification, oriented object detection, pose estimation, computer vision, AI framework
---

# Computer Vision Tasks Supported by Ultralytics YOLO11

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-tasks-banner.avif" alt="Ultralytics YOLO supported computer vision tasks">

Ultralytics YOLO11 is a versatile AI framework that supports multiple [computer vision](https://www.ultralytics.com/blog/everything-you-need-to-know-about-computer-vision-in-2025) **tasks**. The framework can be used to perform [detection](detect.md), [segmentation](segment.md), [OBB](obb.md), [classification](classify.md), and [pose](pose.md) estimation. Each of these tasks has a different objective and use case, allowing you to address various computer vision challenges with a single framework.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/NAs-cfq9BDw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Explore Ultralytics YOLO Tasks: <a href="https://www.ultralytics.com/blog/a-guide-to-deep-dive-into-object-detection-in-2025">Object Detection</a>, Segmentation, OBB, Tracking, and Pose Estimation.
</p>

## [Detection](detect.md)

Detection is the primary task supported by YOLO11. It involves identifying objects in an image or video frame and drawing bounding boxes around them. The detected objects are classified into different categories based on their features. YOLO11 can detect multiple objects in a single image or video frame with high [accuracy](https://www.ultralytics.com/glossary/accuracy) and speed, making it ideal for real-time applications like [surveillance systems](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai) and [autonomous vehicles](https://www.ultralytics.com/solutions/ai-in-automotive).

[Detection Examples](detect.md){ .md-button }

## [Image segmentation](segment.md)

Segmentation takes object detection further by producing pixel-level masks for each object. This precision is useful for applications such as [medical imaging](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency), [agricultural analysis](https://www.ultralytics.com/blog/from-farm-to-table-how-ai-drives-innovation-in-agriculture), and [manufacturing quality control](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision).

[Segmentation Examples](segment.md){ .md-button }

## [Classification](classify.md)

Classification involves categorizing entire images based on their content. This task is essential for applications like [product categorization](https://www.ultralytics.com/blog/understanding-vision-language-models-and-their-applications) in e-commerce, [content moderation](https://www.ultralytics.com/blog/ai-in-document-authentication-with-image-segmentation), and [wildlife monitoring](https://www.ultralytics.com/blog/monitoring-animal-behavior-using-ultralytics-yolov8).

[Classification Examples](classify.md){ .md-button }

## [Pose estimation](pose.md)

Pose estimation detects specific keypoints in images or video frames to track movements or estimate poses. These keypoints can represent human joints, facial features, or other significant points of interest. YOLO11 excels at keypoint detection with high accuracy and speed, making it valuable for [fitness applications](https://www.ultralytics.com/blog/ai-in-our-day-to-day-health-and-fitness), [sports analytics](https://www.ultralytics.com/blog/exploring-the-applications-of-computer-vision-in-sports), and [human-computer interaction](https://www.ultralytics.com/blog/custom-training-ultralytics-yolo11-for-dog-pose-estimation).

[Pose Examples](pose.md){ .md-button }

## [OBB](obb.md)

Oriented Bounding Box (OBB) detection enhances traditional object detection by adding an orientation angle to better locate rotated objects. This capability is particularly valuable for [aerial imagery analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery), [document processing](https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-smart-document-analysis), and [industrial applications](https://www.ultralytics.com/blog/yolo11-enhancing-efficiency-conveyor-automation) where objects appear at various angles. YOLO11 delivers high accuracy and speed for detecting rotated objects in diverse scenarios.

[Oriented Detection](obb.md){ .md-button }

## Conclusion

Ultralytics YOLO11 supports multiple computer vision tasks, including detection, segmentation, classification, oriented object detection, and keypoint detection. Each task addresses specific needs in the computer vision landscape, from basic object identification to detailed pose analysis. By understanding the capabilities and applications of each task, you can select the most appropriate approach for your specific computer vision challenges and leverage YOLO11's powerful features to build effective solutions.

## FAQ

### What computer vision tasks can Ultralytics YOLO11 perform?

Ultralytics YOLO11 is a versatile AI framework capable of performing various computer vision tasks with high accuracy and speed. These tasks include:

- **[Object Detection](detect.md):** Identifying and localizing objects in images or video frames by drawing bounding boxes around them.
- **[Image segmentation](segment.md):** Segmenting images into different regions based on their content, useful for applications like medical imaging.
- **[Classification](classify.md):** Categorizing entire images based on their content.
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

        # Load a pretrained YOLO model (adjust model type as needed)
        model = YOLO("yolo11n.pt")  # n, s, m, l, x versions available

        # Perform object detection on an image
        results = model.predict(source="image.jpg")  # Can also use video, directory, URL, etc.

        # Display the results
        results[0].show()  # Show the first image results
        ```

    === "CLI"

        ```bash
        # Run YOLO detection from the command line
        yolo detect model=yolo11n.pt source="image.jpg" # Adjust model and source as needed
        ```

For more detailed instructions, check out our [detection examples](detect.md).

### What are the benefits of using YOLO11 for segmentation tasks?

Using YOLO11 for segmentation tasks provides several advantages:

1. **High Accuracy:** The segmentation task provides precise, pixel-level masks.
2. **Speed:** YOLO11 is optimized for real-time applications, offering quick processing even for high-resolution images.
3. **Multiple Applications:** It is ideal for medical imaging, autonomous driving, and other applications requiring detailed image segmentation.

Learn more about the benefits and use cases of YOLO11 for segmentation in the [image segmentation section](segment.md).

### Can Ultralytics YOLO11 handle pose estimation and keypoint detection?

Yes, Ultralytics YOLO11 can effectively perform pose estimation and keypoint detection with high accuracy and speed. This feature is particularly useful for tracking movements in sports analytics, healthcare, and human-computer interaction applications. YOLO11 detects keypoints in an image or video frame, allowing for precise pose estimation.

For more details and implementation tips, visit our [pose estimation examples](pose.md).

### Why should I choose Ultralytics YOLO11 for oriented object detection (OBB)?

Oriented Object Detection (OBB) with YOLO11 provides enhanced [precision](https://www.ultralytics.com/glossary/precision) by detecting objects with an additional angle parameter. This feature is beneficial for applications requiring accurate localization of rotated objects, such as aerial imagery analysis and warehouse automation.

- **Increased Precision:** The angle component reduces false positives for rotated objects.
- **Versatile Applications:** Useful for tasks in geospatial analysis, robotics, etc.

Check out the [Oriented Object Detection section](obb.md) for more details and examples.
