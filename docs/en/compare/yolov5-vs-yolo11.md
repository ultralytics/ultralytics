---
comments: true
description: Compare Ultralytics YOLOv5 and YOLO11 to discover how these cutting-edge models differ in speed, accuracy, and efficiency for object detection and other computer vision tasks. Explore advancements in real-time AI and edge AI capabilities that set YOLO11 apart as the next-generation solution.
keywords: Ultralytics, YOLOv5, YOLO11, object detection, real-time AI, edge AI, computer vision, AI models
---

# Ultralytics YOLOv5 VS Ultralytics YOLO11

The evolution of object detection models has brought us to an exciting point of comparison between two groundbreaking innovations: Ultralytics YOLOv5 and Ultralytics YOLO11. Both models represent significant milestones in the YOLO series, offering unique capabilities tailored to a wide range of computer vision applications.

Ultralytics YOLOv5 is renowned for its speed and scalability, making it a favorite among developers for real-time use cases. On the other hand, Ultralytics YOLO11 builds on this foundation, delivering enhanced accuracy, optimized feature extraction, and greater computational efficiency, as highlighted in its [documentation](https://docs.ultralytics.com/models/yolo11/). This comparison will delve into their strengths to help you determine the best fit for your project.

## mAP Comparison

This section compares the mAP (Mean Average Precision) values of Ultralytics YOLOv5 and Ultralytics YOLO11, showcasing their performance across different variants. mAP serves as a critical metric to evaluate the accuracy of object detection models, balancing precision and recall to highlight improvements in detection capabilities. Learn more about [mAP in object detection](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>YOLO11 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 39.5 |
    	| s | 37.4 | 47.0 |
    	| m | 45.4 | 51.4 |
    	| l | 49.0 | 53.2 |
    	| x | 50.7 | 54.7 |

## Speed Comparison

This section highlights the performance differences between Ultralytics YOLOv5 and Ultralytics YOLO11 in terms of speed metrics. Measured in milliseconds, these values demonstrate how both models perform across various sizes, showcasing YOLO11's enhanced efficiency for real-time applications. For more details, visit [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 1.55 |
    	| s | 1.92 | 2.63 |
    	| m | 4.03 | 5.27 |
    	| l | 6.61 | 6.84 |
    	| x | 11.89 | 12.49 |

## YOLO Performance Metrics

Performance metrics are key to evaluating the effectiveness of your YOLO11 models. Metrics such as mAP (mean Average Precision), IoU (Intersection over Union), and F1 score provide insights into your model's accuracy and reliability across different tasks. These metrics help you understand how well your model detects objects, classifies them, and distinguishes between overlapping instances.

For example, mAP measures the precision and recall balance, making it a widely used benchmark in object detection tasks. YOLO11 allows you to evaluate these metrics seamlessly during training and validation, helping you optimize your model for real-world applications.

To dive deeper into performance metrics and learn how to improve your model's accuracy and speed, explore the [YOLO Performance Metrics Guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/). This guide includes practical examples and tips tailored for Ultralytics YOLO users.
