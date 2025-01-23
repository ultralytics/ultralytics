---
comments: true
description: Explore the key differences between Ultralytics YOLOv5 and YOLOv6-3.0, two cutting-edge models in object detection and real-time AI. Compare their performance, speed, and features to discover which is best suited for your computer vision and edge AI applications.
keywords: Ultralytics, YOLOv5, YOLOv6, object detection, real-time AI, edge AI, computer vision, model comparison
---

# Ultralytics YOLOv5 VS YOLOv6-3.0

Choosing the right object detection model is crucial for achieving optimal performance in real-time applications. This comparison between Ultralytics YOLOv5 and YOLOv6-3.0 highlights their unique strengths and the advancements they bring to the field of computer vision.

Ultralytics YOLOv5 is renowned for its ease of use, speed, and memory efficiency, making it a preferred choice for accessible AI solutions. Meanwhile, YOLOv6-3.0 introduces innovative features like the Bi-directional Concatenation (BiC) module and Anchor-Aided Training (AAT), delivering exceptional accuracy and scalability for complex tasks. Explore [YOLOv5](https://www.ultralytics.com/blog/yolov5-v6-0-is-here) and [YOLOv6](https://docs.ultralytics.com/models/yolov6/) to understand their capabilities further.


## mAP Comparison

This section evaluates the performance of Ultralytics YOLOv5 and YOLOv6-3.0 by comparing their mean Average Precision (mAP) values. mAP is a critical metric that reflects model accuracy across various classes and IoU thresholds, providing insights into their object detection capabilities. Learn more about [mAP and its importance](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>YOLOv6-3.0 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 37.5 |
		| s | 37.4 | 45.0 |
		| m | 45.4 | 50.0 |
		| l | 49.0 | 52.8 |
		| x | 50.7 | N/A |
		

## Speed Comparison

This section highlights the performance differences between Ultralytics YOLOv5 and YOLOv6-3.0 by comparing their speed metrics in milliseconds across various model sizes. These measurements provide insights into the efficiency of each model for real-time applications like [object detection](https://www.ultralytics.com/glossary/object-detection) and video analytics.


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 1.17 |
		| s | 1.92 | 2.66 |
		| m | 4.03 | 5.28 |
		| l | 6.61 | 8.95 |
		| x | 11.89 | N/A |

## Predict Functionality in YOLO11

Ultralytics YOLO11's **Predict** functionality enables developers to perform real-time inference on images, videos, or streams with exceptional speed and accuracy. This feature is crucial for applications such as surveillance, retail analytics, and wildlife monitoring. With YOLO11, predictions are streamlined and can be processed effortlessly using the Ultralytics Python package.

To explore further, learn more about [using YOLO11 for object detection](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-object-detection).

### Python Code Example

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO('yolo11.pt')

# Run prediction on an image
results = model.predict(source='image.jpg', save=True)

# Display predictions
results.show()
``` 

This streamlined prediction pipeline ensures an intuitive experience for developers and seamless integration into diverse workflows.
