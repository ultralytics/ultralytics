---
comments: true
description: Compare YOLOv9 and Ultralytics YOLOv8 to uncover advancements in object detection, real-time AI, and edge AI. Explore their performance, accuracy, and applications in computer vision to identify the best model for your needs.
keywords: YOLOv9, Ultralytics YOLOv8, object detection, real-time AI, edge AI, computer vision, YOLO models, Ultralytics
---

# YOLOv9 VS Ultralytics YOLOv8

The comparison between YOLOv9 and Ultralytics YOLOv8 underscores the evolution of object detection models in terms of accuracy, speed, and efficiency. Each model represents a significant milestone in computer vision, offering tailored solutions for diverse real-world applications like real-time detection and segmentation.

Ultralytics YOLOv8, known for its cutting-edge performance and ease of use, has set a high standard for object detection tasks. Meanwhile, YOLOv9 builds on this legacy with further enhancements in architectural design and efficiency, making it a worthy contender in the realm of advanced AI models. For more on YOLOv8, explore its [key features](https://docs.ultralytics.com/models/yolov8/).

## mAP Comparison

The mAP values for YOLOv9 and Ultralytics YOLOv8 provide a detailed measure of model accuracy, reflecting their ability to detect and classify objects across various scenarios. This comparison highlights the performance of different variants, showcasing advancements in precision and efficiency. Learn more about [mAP metrics](https://www.ultralytics.com/glossary/mean-average-precision-map) and their significance in evaluating models.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.8 | 37.3 |
    	| s | 46.5 | 44.9 |
    	| m | 51.5 | 50.2 |
    	| l | 52.8 | 52.9 |
    	| x | 55.1 | 53.9 |

## Speed Comparison

This section highlights a detailed speed analysis of YOLOv9 and Ultralytics YOLOv8 models across various sizes, emphasizing their inference times in milliseconds. These metrics provide insights into the real-time performance capabilities of both models in diverse application scenarios. For more details on YOLOv8's optimizations, visit the [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.3 | 1.47 |
    	| s | 3.54 | 2.66 |
    	| m | 6.43 | 5.86 |
    	| l | 7.16 | 9.06 |
    	| x | 16.77 | 14.37 |

## Object Counting with Ultralytics YOLO11

Ultralytics YOLO11 empowers users to solve advanced computer vision challenges, including **object counting**. This functionality is particularly valuable in scenarios like retail analytics, traffic monitoring, and inventory management, where accurate counting is critical. By leveraging YOLO11’s real-time detection capabilities, users can achieve precise object counts even in dynamic environments.

With its robust performance, YOLO11 integrates seamlessly into workflows, offering support for both pre-trained and custom-trained models. By fine-tuning on specific datasets, such as COCO8 or domain-specific datasets, users can enhance the accuracy of object counting tasks. For more insights on implementing object counting, visit the [Ultralytics Documentation on Object Counting](https://docs.ultralytics.com/guides/object-counting/).

This feature ensures scalability and efficiency, making YOLO11 a powerful tool for practical applications in diverse industries.
