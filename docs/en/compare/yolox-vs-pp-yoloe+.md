---
comments: true
description: Explore an in-depth comparison between YOLOX and PP-YOLOE+, two state-of-the-art models in the realm of object detection. Discover how these models perform in terms of speed, accuracy, and efficiency, and learn which one excels in real-time AI applications and edge AI deployments. Perfect for professionals in computer vision seeking the best solution for their projects.
keywords: YOLOX, PP-YOLOE+, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, AI performance.
---

# YOLOX VS PP-YOLOE+

Understanding the capabilities of advanced object detection models like YOLOX and PP-YOLOE+ is crucial for selecting the right solution for your computer vision needs. This comparison highlights their unique strengths, helping you make an informed decision tailored to your application requirements.

YOLOX offers a balanced mix of speed and accuracy, making it a versatile choice for real-time tasks. On the other hand, PP-YOLOE+ excels in precision and performance optimization, leveraging advancements in feature extraction and inference efficiency. Explore how these models stack up across key metrics and use cases.

## mAP Comparison

mAP (Mean Average Precision) serves as a critical evaluation metric for object detection models, reflecting their ability to accurately detect and localize objects. This section compares YOLOX and PP-YOLOE+ across their variants, highlighting differences in mAP values to assess their performance. Learn more about [Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in evaluating object detection models.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOX | mAP<sup>val<br>50<br>PP-YOLOE+ |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 39.9 |
    	| s | 40.5 | 43.7 |
    	| m | 46.9 | 49.8 |
    	| l | 49.7 | 52.9 |
    	| x | 51.1 | 54.7 |


## Speed Comparison

This section highlights the speed performance of YOLOX and PP-YOLOE+ models across different sizes, measured in milliseconds. These metrics provide valuable insights into how efficiently each model operates, supporting optimal deployment in real-time applications. For more details, explore YOLOX [here](https://github.com/Megvii-BaseDetection/YOLOX) and PP-YOLOE+ [here](https://github.com/PaddlePaddle/PaddleDetection).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 2.84 |
    	| s | 2.56 | 2.62 |
    	| m | 5.43 | 5.56 |
    	| l | 9.04 | 8.36 |
    	| x | 16.1 | 14.3 |

## Using Ultralytics YOLO11 for Tracking

Ultralytics YOLO11 offers advanced **tracking** capabilities, enabling you to follow objects across frames in real-time. This functionality is particularly valuable for applications such as surveillance, sports analytics, and traffic management. By leveraging YOLO11's robust performance, you can enhance tracking precision and reliability even in challenging scenarios like crowded environments or low-light conditions.

YOLO11 supports seamless integration with the Ultralytics HUB, simplifying the setup of tracking workflows. Additionally, the model can be fine-tuned on custom datasets for specialized tracking requirements, such as wildlife monitoring or retail analytics.

For more insights into YOLO11's features and tracking applications, visit the [Ultralytics YOLO Docs](https://docs.ultralytics.com/guides/).

### Python Code Example for Tracking

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Perform tracking on a video
results = model.track(source="video.mp4", save=True, show=True)

# Display tracking results
for result in results:
    print(result.boxes.xyxy)  # Bounding box coordinates
    print(result.boxes.id)  # Unique object IDs
```
