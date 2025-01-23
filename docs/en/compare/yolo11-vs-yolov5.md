---
comments: true
description: Compare Ultralytics YOLO11 and YOLOv5 to discover how the latest advancements in object detection and computer vision redefine real-time AI. Explore key differences in speed, accuracy, and efficiency, making YOLO11 a game-changer for edge AI and diverse applications.
keywords: Ultralytics YOLO11, YOLOv5, object detection, real-time AI, edge AI, computer vision, model comparison, AI advancements
---

# Ultralytics YOLO11 VS Ultralytics YOLOv5

Ultralytics YOLO11 and YOLOv5 represent two significant milestones in the evolution of object detection models. This comparison highlights the advancements in precision, speed, and efficiency that each model brings to modern computer vision challenges.

While YOLOv5 set the standard for real-time object detection with its robust performance, YOLO11 takes it further with enhanced feature extraction and optimized architecture. Dive into the specifics to explore how these models meet diverse application needs, from edge deployment to large-scale enterprise solutions. Learn more about [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) and [YOLOv5 architecture](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/).

## mAP Comparison

This section highlights the mAP performance of Ultralytics YOLO11 compared to Ultralytics YOLOv5, showcasing their accuracy across multiple variants. Mean Average Precision (mAP) evaluates how effectively models detect and classify objects, offering a detailed view of their precision and recall. For more on mAP metrics, visit the [Ultralytics Glossary on mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) or explore [YOLO Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLO11 | mAP<sup>val<br>50<br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.5 | N/A |
    	| s | 47.0 | 37.4 |
    	| m | 51.4 | 45.4 |
    	| l | 53.2 | 49.0 |
    	| x | 54.7 | 50.7 |


## Speed Comparison

Ultralytics YOLO11 demonstrates faster inference times compared to YOLOv5 across various model sizes, reflecting its optimized architecture for real-time applications. For example, [YOLO11n](https://docs.ultralytics.com/models/yolo11/) achieves a speed of 1.5 ms on TensorRT10 with a T4 GPU, outperforming YOLOv5-N's 4.5 ms. Explore more on [YOLO11 performance](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.55 | N/A |
    	| s | 2.63 | 1.92 |
    	| m | 5.27 | 4.03 |
    	| l | 6.84 | 6.61 |
    	| x | 12.49 | 11.89 |

## YOLO Performance Metrics

Understanding performance metrics is crucial when working with Ultralytics YOLO11 to ensure accurate and efficient model evaluation. Metrics like **mean Average Precision (mAP)**, **Intersection over Union (IoU)**, and **F1 Score** provide insights into model performance across different scenarios.

For instance, **mAP** evaluates detection accuracy by comparing predicted bounding boxes with ground truth boxes, offering a comprehensive performance indicator. **IoU** measures the overlap between predicted and actual bounding boxes, while the **F1 Score** balances precision and recall, making it ideal for datasets with imbalanced classes.

Learn more about these metrics and how to optimize them in the [YOLO Performance Metrics Guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/). This guide includes practical examples and recommendations to improve detection and segmentation tasks effectively. Whether you're working on object detection, segmentation, or pose estimation, understanding these metrics helps fine-tune your YOLO11 models for better results.

For additional insights, refer to the [Ultralytics YOLO Documentation](https://docs.ultralytics.com/).
