---
comments: true
description: Compare the performance of YOLOv10 and Ultralytics YOLOv8 to uncover advancements in object detection, real-time AI, and computer vision. Explore their strengths in edge AI applications and see which model excels in speed, accuracy, and versatility for diverse use cases.
keywords: YOLOv10, Ultralytics YOLOv8, object detection, real-time AI, edge AI, computer vision, model comparison, YOLO models
---

# YOLOv10 VS Ultralytics YOLOv8

YOLOv10 and Ultralytics YOLOv8 represent significant milestones in the evolution of object detection models, each offering unique advancements tailored for modern AI applications. This comparison highlights their performance in terms of speed, accuracy, and efficiency, providing key insights for developers and researchers.

Ultralytics YOLOv8 introduced groundbreaking features like anchor-free detection and optimized accuracy-speed tradeoffs, making it a versatile choice for real-time tasks. Meanwhile, YOLOv10 builds on these advancements with enhanced efficiency, dual label assignments, and reduced computational overhead, setting new benchmarks in [object detection](https://www.ultralytics.com/glossary/object-detection).

## mAP Comparison

This section highlights the mAP values of YOLOv10 and Ultralytics YOLOv8, which measure their accuracy across various model variants. Mean Average Precision (mAP) evaluates the balance between precision and recall, providing a comprehensive metric for object detection performance. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in evaluating model effectiveness.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv10 | mAP<sup>val<br>50<br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.5 | 37.3 |
    	| s | 46.7 | 44.9 |
    	| m | 51.3 | 50.2 |
    	| b | 52.7 | N/A |
    	| l | 53.3 | 52.9 |
    	| x | 54.4 | 53.9 |


## Speed Comparison

This section highlights the speed performance of YOLOv10 and Ultralytics YOLOv8 across various model sizes. Measured in milliseconds, the latency metrics showcase the efficiency of these models in real-time applications. Learn more about [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [YOLOv10](https://docs.ultralytics.com/models/yolov10/) for detailed technical insights.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv10 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.56 | 1.47 |
    	| s | 2.66 | 2.66 |
    	| m | 5.48 | 5.86 |
    	| b | 6.54 | N/A |
    	| l | 8.33 | 9.06 |
    	| x | 12.2 | 14.37 |

## Leveraging Ultralytics YOLO11 for Prediction

Ultralytics YOLO11 offers seamless prediction functionality, enabling real-time analysis of images and videos for a variety of [computer vision tasks](https://docs.ultralytics.com/tasks/). Its advanced design ensures high-speed and accurate predictions, even in complex scenarios such as object detection, segmentation, and classification. The prediction capability is highly versatile, supporting diverse datasets and deployment environments.

### Python Code Example: Prediction with YOLO11

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11.pt")

# Run prediction on an image
results = model.predict(source="image.jpg", save=True)

# Display results
results.show()
```

With YOLO11's prediction mode, users can easily analyze visual data across industries like retail, healthcare, and transportation. Explore more about YOLO's capabilities in the [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/guides/).
