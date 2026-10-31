---
title: YOLO26 Computer Vision Tasks Overview
comments: true
description: "Ultralytics YOLO26 supports seven vision tasks: object detection, instance and semantic segmentation, depth estimation, classification, pose, and OBB."
keywords: Ultralytics YOLO26, detection, segmentation, semantic segmentation, depth estimation, classification, pose estimation, oriented object detection, computer vision, AI framework
---

# Computer Vision Tasks Supported by Ultralytics YOLO26

<img width="1024" src="https://cdn.ul.run/i/c99d914c3958d0755b5a3d7204b6f24a.avif" alt="Ultralytics YOLO supported computer vision tasks">

Ultralytics YOLO26 supports multiple [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) **tasks**: [detection](detect.md), [segmentation](segment.md), [semantic segmentation](semantic.md), [depth estimation](depth.md), [classification](classify.md), [pose](pose.md) estimation, and [OBB](obb.md). Each task has a different [objective and use case](https://academy.ultralytics.com/courses/computer-vision-foundations/vision-tasks), so you can address a range of computer vision problems with a single framework.

For planned YOLO27 support, see the [YOLO27 preview](../models/yolo27.md#supported-tasks-and-modes). YOLO27 is coming soon and is not yet available; this guide uses released YOLO26 models.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/3Z0_Fxhm030"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Explore Ultralytics YOLO26 Tasks: Object Detection, Instance Segmentation, Semantic Segmentation, Depth Estimation, Classification, Pose Estimation, and OBB.
</p>

## [Detection](detect.md)

[Object detection](https://www.ultralytics.com/glossary/object-detection) is the primary task supported by YOLO26. It involves identifying objects in an image or video frame and drawing [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) around them. The detected objects are classified into different categories based on their features. YOLO26 detects multiple objects in a single image or video frame quickly enough for real-time applications like [surveillance systems](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai) and [autonomous vehicles](https://www.ultralytics.com/solutions/computer-vision-in-automotive).

[Detection Examples](detect.md){ .md-button }

## [Instance Segmentation](segment.md)

[Instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) takes object detection further by producing pixel-level masks for each object. This precision is useful for applications such as [medical imaging](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency), [agricultural analysis](https://www.ultralytics.com/blog/from-farm-to-table-how-ai-drives-innovation-in-agriculture), and [manufacturing quality control](https://www.ultralytics.com/blog/improving-manufacturing-with-computer-vision).

[Segmentation Examples](segment.md){ .md-button }

## [Semantic Segmentation](semantic.md)

[Semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) assigns a class label to every pixel in an image, producing a dense class map of the entire scene. Unlike instance segmentation, it does not distinguish between individual objects of the same class. This makes it well-suited for [autonomous driving](https://www.ultralytics.com/glossary/autonomous-vehicles), scene parsing, and land-cover mapping where understanding the full spatial layout matters more than identifying individual objects.

[Semantic Segmentation Examples](semantic.md){ .md-button }

## [Depth Estimation](depth.md)

Monocular [depth estimation](https://www.ultralytics.com/glossary/depth-estimation) predicts a per-pixel depth map in meters from a single RGB image. The output is a dense float map aligned to the input image, making it suitable for 3D scene reconstruction, robot navigation, and AR/VR applications where spatial layout must be inferred from a single camera.

[Depth Estimation Examples](depth.md){ .md-button }

## [Classification](classify.md)

[Image classification](https://www.ultralytics.com/glossary/image-classification) involves categorizing entire images based on their content. This task is useful for applications like product categorization in e-commerce, content moderation, and [wildlife monitoring](https://www.ultralytics.com/blog/monitoring-animal-behavior-using-ultralytics-yolov8).

[Classification Examples](classify.md){ .md-button }

## [Pose Estimation](pose.md)

[Pose estimation](https://www.ultralytics.com/glossary/pose-estimation) detects specific [keypoints](https://www.ultralytics.com/glossary/keypoints) in images or video frames to track movements or estimate poses. These keypoints can represent human joints, facial features, or other significant points of interest. YOLO26 pose models detect keypoints in real time, which makes them useful for [fitness applications](https://www.ultralytics.com/blog/ai-in-our-day-to-day-health-and-fitness), [sports analytics](https://www.ultralytics.com/blog/exploring-the-applications-of-computer-vision-in-sports), and [human-computer interaction](https://www.ultralytics.com/blog/vision-ai-enables-touch-free-gesture-recognition-technology).

[Pose Examples](pose.md){ .md-button }

## [OBB](obb.md)

Oriented Bounding Box (OBB) detection enhances traditional object detection by adding an orientation angle to better locate rotated objects. This capability is particularly valuable for [aerial imagery analysis](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery), [document processing](https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-smart-document-analysis), and [industrial applications](https://www.ultralytics.com/blog/yolo11-enhancing-efficiency-conveyor-automation) where objects appear at various angles. YOLO26 OBB models are pretrained on the DOTAv1 aerial imagery dataset.

[OBB Examples](obb.md){ .md-button }

## What's Next

Picked a task? Format your data for it with the [Datasets guide](../datasets/index.md), then [train on your own images](../modes/train.md).

## FAQ

### What computer vision tasks can Ultralytics YOLO26 perform?

Ultralytics YOLO26 supports the following computer vision tasks:

- **[Object Detection](detect.md):** Identifying and localizing objects in images or video frames by drawing bounding boxes around them.
- **[Instance Segmentation](segment.md):** Producing a pixel-level mask for each detected object, useful for applications like medical imaging.
- **[Semantic Segmentation](semantic.md):** Assigning a class label to every pixel in an image for dense scene understanding.
- **[Depth Estimation](depth.md):** Predicting a per-pixel depth map in meters from a single RGB image.
- **[Classification](classify.md):** Categorizing entire images based on their content.
- **[Pose Estimation](pose.md):** Detecting specific keypoints in an image or video frame to track movements or poses.
- **[Oriented Object Detection (OBB)](obb.md):** Detecting rotated objects with an added orientation angle for enhanced accuracy.

### How do I use Ultralytics YOLO26 for object detection?

To use Ultralytics YOLO26 for object detection, follow these steps:

1. Prepare your dataset in the appropriate format.
2. Train the YOLO26 model using the detection task.
3. Use the model to make predictions by feeding in new images or video frames.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO model (adjust model type as needed)
        model = YOLO("yolo26n.pt")  # n, s, m, l, x versions available

        # Perform object detection on an image
        results = model.predict(source="image.jpg")  # Can also use video, directory, URL, etc.

        # Display the results
        results[0].show()  # Show the first image results
        ```

    === "CLI"

        ```bash
        # Run YOLO detection from the command line
        yolo detect predict model=yolo26n.pt source="image.jpg" # Adjust model and source as needed
        ```

For more detailed instructions, check out our [detection examples](detect.md).

### What are the benefits of using YOLO26 for segmentation tasks?

Using YOLO26 for segmentation tasks provides several advantages:

1. **High Accuracy:** The segmentation task provides precise, pixel-level masks.
2. **Speed:** YOLO26 is optimized for real-time applications, offering quick processing even for high-resolution images.
3. **Multiple Applications:** It is ideal for medical imaging, autonomous driving, and other applications requiring detailed image segmentation.

Learn more about the benefits and use cases of YOLO26 for segmentation in the [image segmentation section](segment.md).

### Can Ultralytics YOLO26 handle pose estimation and keypoint detection?

Yes. YOLO26 pose models detect keypoints, such as human joints, in images and video frames, and connect them into poses. This is useful for tracking movements in sports analytics, healthcare, and human-computer interaction applications.

For more details and implementation tips, visit our [pose estimation examples](pose.md).

### Why should I choose Ultralytics YOLO26 for oriented object detection (OBB)?

Oriented Object Detection (OBB) with YOLO26 predicts an additional angle parameter for each box, so rotated objects are localized more tightly than with axis-aligned boxes. This is beneficial for applications requiring accurate localization of rotated objects, such as aerial imagery analysis and warehouse automation.

- **Tighter Boxes:** Rotated boxes include less background, which helps separate densely packed or angled objects.
- **Versatile Applications:** Useful for tasks in geospatial analysis, robotics, etc.

Check out the [Oriented Object Detection section](obb.md) for more details and examples.
