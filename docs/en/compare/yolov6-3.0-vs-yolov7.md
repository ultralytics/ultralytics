---
comments: true
description: Compare the performance and innovations of YOLOv6-3.0 and YOLOv7, two state-of-the-art models in real-time object detection. Discover their unique features, efficiency, and accuracy in advancing edge AI and computer vision applications.
keywords: YOLOv6-3.0, YOLOv7, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, performance metrics
---

# YOLOv6-3.0 VS YOLOv7

Real-time object detection continues to evolve, with YOLOv6-3.0 and YOLOv7 standing as pivotal advancements in this domain. This comparison explores their unique capabilities, helping users determine the best fit for their specific applications.

YOLOv6-3.0 emphasizes efficiency with optimized architectures for deployment on diverse hardware, while YOLOv7 introduces innovations like dynamic label assignment for improved training accuracy. Dive deeper into their features to identify the ideal model for your needs, from edge devices to cloud integrations. For further details, visit the [YOLOv7 GitHub repository](https://github.com/WongKinYiu/yolov7) or learn about real-time applications with [Ultralytics tools](https://www.ultralytics.com).

## mAP Comparison

This section highlights the mAP differences between YOLOv6-3.0 and YOLOv7, showcasing their accuracy across various model sizes. Mean Average Precision (mAP) serves as a critical metric to evaluate and compare the object detection performance of these models. Learn more about [mAP in object detection](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv6-3.0 | mAP<sup>val<br>50<br>YOLOv7 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.5 | N/A |
    	| s | 45.0 | N/A |
    	| m | 50.0 | N/A |
    	| l | 52.8 | 51.4 |
    	| x | N/A | 53.1 |


## Speed Comparison

This section highlights the speed performance of YOLOv6-3.0 and YOLOv7 across various model sizes, measured in milliseconds. These metrics reflect the efficiency of inference, crucial for applications requiring real-time object detection. For further details on YOLOv7's innovations, visit the [YOLOv7 documentation](https://docs.ultralytics.com/models/yolov7/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.17 | N/A |
    	| s | 2.66 | N/A |
    	| m | 5.28 | N/A |
    	| l | 8.95 | 6.84 |
    	| x | N/A | 11.57 |

## Fine-Tuning with African Wildlife Dataset

Ultralytics YOLO11 offers the flexibility to fine-tune pre-trained models on custom datasets, such as the African Wildlife dataset. This dataset enables applications in wildlife monitoring, conservation, and anti-poaching efforts. By using YOLO11's advanced functionalities, users can enhance detection accuracy for species unique to the African ecosystem, including elephants, lions, and zebras. This customization ensures precise recognition and tracking in real-world scenarios.

For more on fine-tuning YOLO models, check out the [Custom Training Guide](https://docs.ultralytics.com/modes/train/).

### Example Python Code for Fine-Tuning

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Train the model on African Wildlife dataset
model.train(data="african_wildlife.yaml", epochs=50, imgsz=640)

# Validate the model
metrics = model.val()

# Export the trained model
model.export(format="onnx")
```

Explore more about dataset-specific training and evaluation on the [YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/).
