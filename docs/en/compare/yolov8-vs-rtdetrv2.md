---
comments: true
description: Explore an in-depth comparison between Ultralytics YOLOv8 and RTDETRv2, highlighting their performance in real-time object detection, edge AI applications, and advancements in computer vision technology. Discover which model excels in speed, accuracy, and versatility for your machine learning needs.  
keywords: Ultralytics YOLOv8, RTDETRv2, object detection, real-time AI, edge AI, computer vision, machine learning, model comparison, AI performance
---

# Ultralytics YOLOv8 VS RTDETRv2

The comparison between Ultralytics YOLOv8 and RTDETRv2 showcases two of the most advanced models in object detection. Both models excel in delivering high accuracy and efficiency, making them top choices for real-time applications across diverse industries.

Ultralytics YOLOv8 is renowned for its state-of-the-art performance, combining speed and precision optimized for tasks like object detection, segmentation, and more. On the other hand, RTDETRv2 shines with its transformer-based architecture, offering robust capabilities for complex detection scenarios. Learn more about [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [RTDETRv2](https://github.com/ultralytics/ultralytics).


## mAP Comparison

This section compares the mAP (Mean Average Precision) values of Ultralytics YOLOv8 and RTDETRv2 across different variants, showcasing their accuracy in object detection tasks. mAP is a critical metric that evaluates both precision and recall, providing a comprehensive measure of model performance. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in evaluating object detection models.


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>RTDETRv2 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 37.3 | N/A |
		| s | 44.9 | 48.1 |
		| m | 50.2 | 51.9 |
		| l | 52.9 | 53.4 |
		| x | 53.9 | 54.3 |
		

## Speed Comparison

This section highlights the speed metrics of Ultralytics YOLOv8 and RTDETRv2 across different model sizes, measured in milliseconds. These comparisons demonstrate the efficiency of each model in terms of latency, providing valuable insights into real-world performance. For more details on YOLOv8 performance, visit the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 1.47 | N/A |
		| s | 2.66 | 5.03 |
		| m | 5.86 | 7.51 |
		| l | 9.06 | 9.76 |
		| x | 14.37 | 15.03 |

## YOLO Thread-Safe Inference

Thread-safe inference is critical when deploying YOLO models in multi-threaded environments, such as real-time applications or server-based deployments. Ultralytics YOLO11 offers robust support for thread-safe operations, ensuring consistent and reliable predictions across concurrent threads. By implementing best practices for thread-safe inference, you can prevent race conditions and achieve optimal model performance.

To get started, explore the [YOLO Thread-Safe Inference Guide](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/) for detailed insights and actionable steps. This guide provides clear explanations of the challenges involved and how YOLO11 addresses them with advanced features.

For additional resources, check out the [Ultralytics Python package documentation](https://pypi.org/project/ultralytics/), which includes examples and tools to help you implement thread-safe YOLO11 inference seamlessly. By following these practices, you can maximize the efficiency and reliability of your computer vision applications.
