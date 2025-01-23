---
comments: true
description: Discover the key differences between YOLOv9 and YOLOv7 in this comprehensive comparison. Explore their performance, advancements, and efficiency in real-time AI, edge AI, and computer vision applications. Learn how these models stack up in terms of object detection accuracy and computational efficiency. 
keywords: YOLOv9, YOLOv7, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, AI performance
---

# YOLOv9 VS YOLOv7

The comparison between YOLOv9 and YOLOv7 highlights the evolution of object detection technologies within the YOLO series. Each model represents a significant milestone, with YOLOv9 building upon the foundational strengths of YOLOv7 to push boundaries in speed, accuracy, and efficiency.

YOLOv7 introduced groundbreaking features designed for optimal real-time performance, while YOLOv9 further refines these capabilities with enhanced architectures and processing power. This page explores their unique strengths, offering insights into their suitability for a range of [object detection tasks](https://www.ultralytics.com/glossary/object-detection).


## mAP Comparison

This section compares the mAP values of YOLOv9 and YOLOv7 across various model variants, showcasing their accuracy in detecting and localizing objects. Mean Average Precision (mAP) serves as a key metric, reflecting the balance between precision and recall for each model. For more details on mAP, visit the [Ultralytics Glossary](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>YOLOv7 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 37.8 | N/A |
		| s | 46.5 | N/A |
		| m | 51.5 | N/A |
		| l | 52.8 | 51.4 |
		| x | 55.1 | 53.1 |
		

## Speed Comparison

This section highlights the speed performance of YOLOv9 and YOLOv7 across various model sizes, measured in milliseconds. The detailed metrics illustrate how each model balances computational efficiency and real-time inference capabilities. For more on YOLOv7's efficiency, refer to [YOLOv7 Documentation](https://docs.ultralytics.com/models/yolov7/), and explore YOLOv9's advancements in [YOLOv9 Performance](https://docs.ultralytics.com/models/yolov9/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 2.3 | N/A |
		| s | 3.54 | N/A |
		| m | 6.43 | N/A |
		| l | 7.16 | 6.84 |
		| x | 16.77 | 11.57 |

## YOLO Thread-Safe Inference

Thread-safe inference with Ultralytics YOLO11 ensures reliable and consistent predictions when running multiple inference operations simultaneously. This feature is particularly valuable for applications requiring high concurrency, such as video analytics or large-scale real-time monitoring systems.

By implementing best practices for thread safety, YOLO11 minimizes race conditions and ensures optimal performance across threads. For example, users can configure separate instances of the model for each thread, preventing data collisions and maintaining efficiency. 

To dive deeper into the importance of thread safety and learn how to implement it effectively, refer to the [YOLO Thread-Safe Inference guide](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/). This guide provides step-by-step instructions and practical examples to help you achieve thread-safe inference setups.

For more information on YOLO11’s capabilities, visit the [Ultralytics documentation](https://docs.ultralytics.com/).
