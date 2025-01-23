---
comments: true
description: Explore the in-depth comparison between Ultralytics YOLOv8 and PP-YOLOE+, two cutting-edge models in object detection. Discover their performance in real-time AI applications, edge AI deployments, and computer vision tasks, highlighting speed, accuracy, and versatility.
keywords: Ultralytics YOLOv8, PP-YOLOE+, object detection, real-time AI, edge AI, computer vision, YOLO models, AI comparison
---

# Ultralytics YOLOv8 VS PP-YOLOE+

In the ever-evolving landscape of AI-driven object detection, comparing models like Ultralytics YOLOv8 and PP-YOLOE+ is essential for understanding their unique capabilities. These two state-of-the-art architectures cater to a wide range of applications, from real-time detection tasks to high-accuracy scenarios.

Ultralytics YOLOv8, with its optimized accuracy-speed tradeoff and enhanced extensibility, has redefined benchmarks in computer vision. On the other hand, PP-YOLOE+ brings its own innovations, focusing on efficiency and precision. This comparison unpacks their strengths to help you choose the best model for your needs. Learn more about [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) or explore [PP-YOLOE+](https://github.com/PaddlePaddle/PaddleDetection) for further details.

## mAP Comparison

This section highlights the mAP values of Ultralytics YOLOv8 and PP-YOLOE+, providing a clear metric to compare their accuracy across different model variants. Mean Average Precision (mAP) serves as a comprehensive evaluation standard, balancing precision and recall, offering insights into each model's object detection performance. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in model evaluation.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>PP-YOLOE+ |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.3 | 39.9 |
    	| s | 44.9 | 43.7 |
    	| m | 50.2 | 49.8 |
    	| l | 52.9 | 52.9 |
    	| x | 53.9 | 54.7 |


## Speed Comparison

This section highlights the speed performance of Ultralytics YOLOv8 and PP-YOLOE+ across various model sizes, measured in milliseconds. The comparison emphasizes YOLOv8's real-time efficiency, making it a standout choice for applications like [object detection](https://docs.ultralytics.com/tasks/detect/) and [real-time tracking](https://docs.ultralytics.com/modes/track/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.47 | 2.84 |
    	| s | 2.66 | 2.62 |
    	| m | 5.86 | 5.56 |
    	| l | 9.06 | 8.36 |
    	| x | 14.37 | 14.3 |

## Predict With Ultralytics YOLO11

The **Predict** functionality of Ultralytics YOLO11 empowers users to efficiently perform inference tasks across various domains. With its seamless integration, YOLO11 handles object detection, segmentation, and classification with remarkable accuracy and speed. Whether you are working with standard datasets or custom-trained models, the predict mode provides reliable outputs for real-time applications.

For instance, you can leverage the Ultralytics Python package to load your YOLO11 model and perform predictions on images, videos, or streams. This makes it ideal for tasks like real-time surveillance, wildlife monitoring, and retail analytics.

Learn more about [how to use YOLO11 for object detection](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-object-detection) to enhance your prediction workflows.

### Python Code Example

```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11.pt")

# Perform prediction on an image
results = model.predict(source="image.jpg", save=True, save_txt=True)

# Display prediction results
results.show()
```
