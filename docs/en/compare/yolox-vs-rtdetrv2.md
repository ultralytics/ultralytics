---
comments: true
description: Explore the comprehensive comparison between YOLOX and RTDETRv2, two cutting-edge models in real-time object detection and computer vision. Learn about their performance, efficiency, and suitability for applications in edge AI and advanced real-time AI tasks.
keywords: YOLOX, RTDETRv2, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOX VS RTDETRv2

When evaluating state-of-the-art object detection models, YOLOX and RTDETRv2 stand out as competitive solutions designed for speed and accuracy. This comparison aims to shed light on their unique architectures, strengths, and applications, helping users make informed decisions for their computer vision projects.

YOLOX is known for its exceptional real-time performance and efficiency, while RTDETRv2 excels in combining Vision Transformer-based designs with adaptable inference speed. By exploring their features and benchmarks, this page provides valuable insights into how these models perform across diverse scenarios. For additional context, you can also explore [RT-DETR's capabilities](https://docs.ultralytics.com/reference/models/rtdetr/model/) and learn more about [YOLOX advancements](https://docs.ultralytics.com/guides/).

## mAP Comparison

This section highlights the mAP values of YOLOX and RTDETRv2, showcasing their accuracy across various benchmarks. mAP, a key performance metric in [object detection](https://www.ultralytics.com/glossary/object-detection), evaluates both precision and recall, offering insights into model effectiveness for detecting and classifying objects. Learn more about [mAP metrics here](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOX | mAP<sup>val<br>50<br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 40.5 | 48.1 |
    	| m | 46.9 | 51.9 |
    	| l | 49.7 | 53.4 |
    	| x | 51.1 | 54.3 |


## Speed Comparison

Explore how YOLOX and RTDETRv2 compare in terms of speed, measured in milliseconds, across various model sizes. These metrics highlight the efficiency of each model for tasks requiring rapid inference, making them crucial for real-time applications. Learn more about [YOLOX models](https://github.com/Megvii-BaseDetection/YOLOX) and [RTDETRv2](https://docs.ultralytics.com/modes/benchmark/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| s | 2.56 | 5.03 |
    	| m | 5.43 | 7.51 |
    	| l | 9.04 | 9.76 |
    	| x | 16.1 | 15.03 |

## YOLO11 Feature: SAHI Tiled Inference

SAHI (Sliced Aided Hyper Inference) Tiled Inference is a powerful feature of Ultralytics YOLO11 designed to improve object detection in high-resolution images. This functionality is particularly useful for scenarios like satellite imagery, medical imaging, and wildlife monitoring, where large images need to be processed without losing detail.

<<<<<<< HEAD
By breaking down large images into smaller tiles, SAHI Tiled Inference allows YOLO11 to maintain high detection accuracy and performance. It ensures that even small objects in high-resolution images are not missed during detection. This feature is fully compatible with YOLO11's segmentation and detection capabilities, offering flexibility across various industries and applications.

# Learn more about [SAHI Tiled Inference](https://docs.ultralytics.com/guides/sahi-tiled-inference/) and how to integrate it into your projects for enhanced image analysis. For additional resources, explore the [Ultralytics Guides](https://docs.ultralytics.com/guides/) to maximize the potential of YOLO11 in your workflows.

By leveraging YOLO11's real-time detection and segmentation functionalities, users can easily identify and blur objects dynamically. The process is efficient and integrates seamlessly into workflows, making it ideal for industries like security, media, and compliance.

For further insights into YOLO11's solutions like object blurring, visit the [Ultralytics YOLO documentation](https://docs.ultralytics.com/guides/).

### Python Code Snippet for Object Blurring

```python
from ultralytics import YOLO

# Load YOLO11 model
model = YOLO("yolo11.pt")

# Perform object detection and apply blurring
results = model.predict(source="video.mp4", save=True, blur=True)

# Save the output video with blurred objects
results.save("output_blurred.mp4")
```

> > > > > > > 95d73b193a43ffc9b65d81b5b93f6d2acf3cb195
