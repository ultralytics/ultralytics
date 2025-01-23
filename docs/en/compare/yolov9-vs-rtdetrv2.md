---
comments: true
description: Discover the key differences between YOLOv9 and RTDETRv2 in this comprehensive comparison. Explore their performance, efficiency, and real-world applications for object detection and real-time AI, and learn which model best suits your computer vision needs, from edge AI to cloud deployment.
keywords: YOLOv9, RTDETRv2, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, AI performance
---

# YOLOv9 VS RTDETRv2

The comparison of YOLOv9 and RTDETRv2 highlights the advancements in modern object detection models, focusing on speed, accuracy, and efficiency. Both models bring unique innovations to the table, making them ideal for diverse applications in computer vision.

YOLOv9, part of the YOLO family, is optimized for real-time performance with reduced latency and parameter efficiency. On the other hand, RTDETRv2 leverages Vision Transformer-based architecture for high accuracy and flexibility, as detailed in its [model reference](https://docs.ultralytics.com/reference/models/rtdetr/model/). This page explores their capabilities to help you select the best fit for your use case.

## mAP Comparison

The mAP comparison between YOLOv9 and RT-DETRv2 highlights the accuracy of these models across different variants. Mean Average Precision (mAP) serves as a key metric to evaluate how well models like [YOLOv9](https://docs.ultralytics.com/models/yolov9/) and RT-DETRv2 detect and classify objects, providing a comprehensive measure of performance on datasets such as [COCO](https://docs.ultralytics.com/datasets/detect/coco/).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.8 | N/A |
    	| s | 46.5 | 48.1 |
    	| m | 51.5 | 51.9 |
    	| l | 52.8 | 53.4 |
    	| x | 55.1 | 54.3 |


## Speed Comparison

This section highlights the speed performance of YOLOv9 and RTDETRv2 models, measured in milliseconds, across various sizes. These metrics provide a comprehensive understanding of their efficiency, crucial for applications requiring real-time processing. Explore more about [RTDETRv2](https://docs.ultralytics.com/models/yolov7/) and [YOLOv9](https://docs.ultralytics.com/models/yolov10/) for further insights.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.3 | N/A |
    	| s | 3.54 | 5.03 |
    	| m | 6.43 | 7.51 |
    	| l | 7.16 | 9.76 |
    	| x | 16.77 | 15.03 |

## Fine-Tuning on the COCO8 Dataset

Ultralytics YOLO11 offers seamless fine-tuning capabilities for a variety of datasets, including COCO8. The COCO8 dataset, derived from the renowned COCO dataset, provides a compact yet diverse set of annotated images ideal for object detection and segmentation tasks. Fine-tuning YOLO11 on COCO8 allows users to achieve optimized performance for specific applications while leveraging pre-trained weights for faster convergence and improved accuracy.

To start fine-tuning on COCO8, YOLO11 provides an intuitive interface for loading datasets, managing hyperparameters, and monitoring training metrics. This flexibility ensures that both beginners and experts can efficiently adapt the model to their unique use cases.

For more details on fine-tuning and dataset support, visit the [Ultralytics guide](https://docs.ultralytics.com/modes/train/).

### Python Code Example

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO('yolo11.pt')

# Fine-tune on the COCO8 dataset
model.train(data='coco8.yaml', epochs=50, imgsz=640)

# Validate the model
metrics = model.val()

# Export the model for deployment
model.export(format='onnx')
```

This code snippet demonstrates how to fine-tune YOLO11 on the COCO8 dataset using Ultralytics' streamlined training pipeline.
