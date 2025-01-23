---
comments: true
description: Explore an in-depth comparison between Ultralytics YOLOv5 and PP-YOLOE+, two cutting-edge models in object detection. Discover their performance in terms of speed, accuracy, and efficiency, making them ideal for real-time AI, edge AI, and computer vision applications.
keywords: YOLOv5, PP-YOLOE+, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, AI performance
---

# Ultralytics YOLOv5 VS PP-YOLOE+

The comparison between Ultralytics YOLOv5 and PP-YOLOE+ emphasizes the advancements in real-time object detection, a cornerstone of modern computer vision applications. These models are benchmarked for their accuracy, speed, and computational efficiency, catering to diverse use cases such as surveillance, autonomous vehicles, and smart retail.

Ultralytics YOLOv5 stands out with its streamlined architecture and optimized training pipeline, ensuring robust performance across various deployment environments. Meanwhile, PP-YOLOE+ integrates innovative design elements for enhanced feature extraction and precision, offering a competitive edge in challenging detection scenarios. Explore how these models redefine [object detection](https://www.ultralytics.com/glossary/object-detection) and their applicability in AI-driven solutions.

## mAP Comparison

This section highlights the mAP values of Ultralytics YOLOv5 versus PP-YOLOE+, demonstrating each model's accuracy in detecting and classifying objects across different variants. mAP, or Mean Average Precision, offers a comprehensive evaluation of model performance by balancing [precision and recall](https://www.ultralytics.com/glossary/mean-average-precision-map) across classes and thresholds.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>PP-YOLOE+ |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 39.9 |
    	| s | 37.4 | 43.7 |
    	| m | 45.4 | 49.8 |
    	| l | 49.0 | 52.9 |
    	| x | 50.7 | 54.7 |

## Speed Comparison

This section highlights the speed metrics of Ultralytics YOLOv5 and PP-YOLOE+ models, measured in milliseconds, across various sizes. These comparisons provide insights into performance efficiency, helping users evaluate real-time applications. Learn more about [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) and [PP-YOLOE+](https://github.com/PaddlePaddle/PaddleDetection).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 2.84 |
    	| s | 1.92 | 2.62 |
    	| m | 4.03 | 5.56 |
    	| l | 6.61 | 8.36 |
    	| x | 11.89 | 14.3 |

## YOLO Thread-Safe Inference

Thread-safe inference is an essential concept when deploying models like Ultralytics YOLO11 in multi-threaded environments. By ensuring that inference processes do not interfere with each other, thread safety helps prevent race conditions and ensures consistent predictions. This is particularly important when deploying YOLO11 for real-time applications, such as surveillance systems or autonomous vehicles, where multiple threads may simultaneously access the model.

Ultralytics provides detailed guidelines for implementing thread-safe inference with YOLO11. These include best practices for managing resources, such as memory allocation and synchronization, to optimize performance without compromising reliability. Explore the [YOLO Thread-Safe Inference Guide](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/) to learn more about these techniques and how to implement them effectively.

By following these practices, you can achieve stable and efficient inference for your multi-threaded applications.
