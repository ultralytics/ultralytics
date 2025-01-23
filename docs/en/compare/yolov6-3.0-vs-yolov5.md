---
comments: true
description: Explore the key differences between YOLOv6-3.0 and Ultralytics YOLOv5 in this detailed comparison. Learn how these models excel in object detection, real-time AI, and edge AI applications, while pushing the boundaries of computer vision technology.
keywords: YOLOv6-3.0, Ultralytics YOLOv5, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOv6-3.0 VS Ultralytics YOLOv5

Comparing YOLOv6-3.0 and Ultralytics YOLOv5 provides valuable insights into the evolution of object detection models. Both models are renowned for their accuracy and speed, making them essential tools for real-time computer vision tasks.

YOLOv6-3.0 introduces advanced optimizations tailored for high-speed applications, while Ultralytics YOLOv5 stands as a benchmark for usability and performance balance. By analyzing their distinct strengths, this comparison aims to guide users in selecting the ideal model for their specific needs. For more on YOLO models, explore the [Ultralytics documentation](https://docs.ultralytics.com/models/yolov8/).

## mAP Comparison

This section highlights the mAP values of YOLOv6-3.0 and Ultralytics YOLOv5, showcasing their accuracy in detecting objects across various thresholds. Mean Average Precision (mAP) is a crucial metric that evaluates the balance between precision and recall, providing insights into the performance of these models. For a deeper understanding of mAP, visit the [Ultralytics Glossary on mAP](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv6-3.0 | mAP<sup>val<br>50<br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.5 | N/A |
    	| s | 45.0 | 37.4 |
    	| m | 50.0 | 45.4 |
    	| l | 52.8 | 49.0 |
    	| x | N/A | 50.7 |

## Speed Comparison

This section highlights the speed performance of YOLOv6-3.0 and Ultralytics YOLOv5 across multiple sizes. Speed metrics, measured in milliseconds, provide insight into the inference efficiency of these models, enabling a direct comparison of their real-time application capabilities. For more details on model profiling, visit the [Ultralytics Benchmarks page](https://docs.ultralytics.com/reference/utils/benchmarks/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.17 | N/A |
    	| s | 2.66 | 1.92 |
    	| m | 5.28 | 4.03 |
    	| l | 8.95 | 6.61 |
    	| x | N/A | 11.89 |

## YOLO Thread-Safe Inference

Thread-safe inference is critical for deploying models in multi-threaded environments, especially for applications requiring high concurrency. Ultralytics YOLO11 introduces support for thread-safe inference, ensuring consistent and reliable predictions even in complex systems. This feature is essential for tasks like real-time object detection in robotics or video analytics.

To implement thread-safe inference with YOLO11, follow best practices such as initializing models within individual threads and avoiding shared state variables. These practices help prevent race conditions and ensure optimal model performance.

For more detailed guidance, check out the [YOLO Thread-Safe Inference guide](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/). This resource provides step-by-step instructions and examples tailored to YOLO users, helping you seamlessly integrate thread-safe techniques into your projects.
