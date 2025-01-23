---
comments: true
description: Explore a detailed comparison between DAMO-YOLO and YOLOv9, two cutting-edge models in real-time object detection. Discover their performance, efficiency, and application potential in computer vision and edge AI, powered by Ultralytics innovation. 
keywords: DAMO-YOLO, YOLOv9, Ultralytics, object detection, real-time AI, edge AI, computer vision
---

# DAMO-YOLO VS YOLOv9

The comparison between DAMO-YOLO and YOLOv9 highlights the advancements in object detection and the strides made in balancing speed and accuracy. Both models cater to diverse use cases, showcasing their potential to redefine real-time applications in industries like healthcare, retail, and autonomous systems.

DAMO-YOLO emphasizes cutting-edge efficiency and accuracy through innovative architectural designs, while YOLOv9 builds upon the strong foundation of YOLO models, offering significant improvements in speed and parameter optimization. This analysis explores their unique strengths, helping users identify the best model for their specific needs. Explore more about [YOLO models](https://www.ultralytics.com/blog/the-evolution-of-object-detection-and-ultralytics-yolo-models) and [object detection tasks](https://docs.ultralytics.com/tasks/).


## mAP Comparison

This section compares the models DAMO-YOLO and YOLOv9 based on their mean average precision (mAP) scores, a critical metric that evaluates detection accuracy across different classes and thresholds. Higher mAP values indicate better performance, reflecting the precision and recall of these models in object detection tasks. Learn more about [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>DAMO-YOLO | mAP<sup>val<br>50<br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 42.0 | 37.8 |
		| s | 46.0 | 46.5 |
		| m | 49.2 | 51.5 |
		| l | 50.8 | 52.8 |
		| x | N/A | 55.1 |
		

## Speed Comparison

This section highlights the speed performance of DAMO-YOLO versus YOLOv9 across various model sizes, measured in milliseconds. These metrics provide a clear perspective on how efficiently each model processes data, offering valuable insights for real-time applications. Learn more about [YOLOv9's efficiency](https://docs.ultralytics.com/models/yolov9/) and its impact on [AI deployment](https://docs.ultralytics.com/guides/model-deployment-options/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 2.32 | 2.3 |
		| s | 3.45 | 3.54 |
		| m | 5.09 | 6.43 |
		| l | 7.18 | 7.16 |
		| x | N/A | 16.77 |

## Segment: Car Parts Segmentation

Ultralytics YOLO11 excels in custom applications like car parts segmentation, enabling precise identification and categorization of automotive components. This functionality is particularly beneficial in industries such as automotive manufacturing, repair, and e-commerce cataloging. By leveraging advanced segmentation capabilities, YOLO11 helps streamline workflows and improve accuracy in tasks like inventory management and quality control.

For car parts segmentation, YOLO11 can be fine-tuned on specialized datasets, such as the Roboflow Car Parts Segmentation dataset. This ensures high precision and relevance for industry-specific requirements. Learn more about how YOLO11 performs segmentation tasks effectively in [this guide](https://docs.ultralytics.com/datasets/segment/carparts-seg/).

### Python Code for Custom Training

```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11-seg.pt")

# Train the model on a custom car parts segmentation dataset
model.train(data="carparts.yaml", epochs=50, imgsz=640)
```
This code demonstrates how to train YOLO11 for car parts segmentation, ensuring optimal performance for your specific use case.
