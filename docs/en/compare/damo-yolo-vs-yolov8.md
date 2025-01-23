---
comments: true
description: Explore a detailed comparison between DAMO-YOLO and Ultralytics YOLOv8, two leading models in real-time object detection and computer vision. Discover their performance, speed, and capabilities in edge AI applications.
keywords: DAMO-YOLO, Ultralytics YOLOv8, object detection, real-time AI, edge AI, computer vision, Ultralytics, YOLO, AI models
---

# DAMO-YOLO VS Ultralytics YOLOv8

The comparison of DAMO-YOLO and Ultralytics YOLOv8 brings into focus two cutting-edge models in the realm of real-time object detection. Both models are designed to deliver exceptional performance, balancing speed and accuracy across a variety of computer vision tasks.  

While DAMO-YOLO emphasizes efficiency through innovative architecture optimizations, Ultralytics YOLOv8 stands out with its user-friendly design, pre-trained model versatility, and seamless integration. This evaluation will explore their unique strengths, helping you determine the best fit for your specific application needs. Learn more about [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) and its advancements.


## mAP Comparison

This section evaluates the mAP values of DAMO-YOLO and Ultralytics YOLOv8, showcasing their object detection accuracy across different variants. The mAP metric, crucial for assessing model performance, balances precision and recall to provide a comprehensive measure of effectiveness. Learn more about [Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>DAMO-YOLO | mAP<sup>val<br>50<br>YOLOv8 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 42.0 | 37.3 |
		| s | 46.0 | 44.9 |
		| m | 49.2 | 50.2 |
		| l | 50.8 | 52.9 |
		| x | N/A | 53.9 |
		

## Speed Comparison

This section highlights the speed performance of DAMO-YOLO and Ultralytics YOLOv8 models across different sizes, measured in milliseconds per inference. These metrics underline the efficiency of each model, providing critical insights into their suitability for real-time applications. For more details on YOLOv8's speed and accuracy, visit the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 2.32 | 1.47 |
		| s | 3.45 | 2.66 |
		| m | 5.09 | 5.86 |
		| l | 7.18 | 9.06 |
		| x | N/A | 14.37 |

## YOLO Thread-Safe Inference

Thread-safe inference is essential when performing predictions with YOLO models in multi-threaded environments. It ensures consistent and accurate results while preventing potential race conditions that could compromise your model's reliability. Ultralytics YOLO11 supports thread-safe inference, making it ideal for deployment in real-time applications such as robotics, security systems, and autonomous vehicles.

To learn more about thread safety and best practices, check out the [YOLO Thread-Safe Inference Guide](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/). This guide provides detailed insights and step-by-step instructions to help you integrate YOLO11 seamlessly into multi-threaded systems.

For more implementation tips, explore the [Ultralytics Python package](https://pypi.org/project/ultralytics/) documentation, which includes essential examples for ensuring thread-safe operations during inference.
