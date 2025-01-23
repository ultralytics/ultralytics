---
comments: true
description: Explore the detailed comparison between YOLOv6-3.0 and YOLOv9, highlighting their performance, efficiency, and advancements in real-time object detection and computer vision. Learn how these models cater to edge AI applications with cutting-edge accuracy and speed optimizations.
keywords: YOLOv6-3.0, YOLOv9, Ultralytics, object detection, real-time AI, edge AI, computer vision, AI models, performance comparison
---

# YOLOv6-3.0 VS YOLOv9

The comparison between YOLOv6-3.0 and YOLOv9 highlights the evolution of object detection models and their growing importance in AI-driven applications. These models represent significant advancements in speed, accuracy, and efficiency, making them pivotal solutions for diverse use cases such as real-time tracking and autonomous systems.

YOLOv6-3.0 is celebrated for its optimized lightweight architecture, enabling high-speed inference on resource-constrained devices. On the other hand, YOLOv9 builds upon its predecessors with enhanced feature extraction capabilities and cutting-edge backbone designs, offering unparalleled precision for intricate scenarios. For more details on YOLO advancements, explore [Ultralytics YOLO documentation](https://docs.ultralytics.com/models/yolov8/).

## mAP Comparison

This section compares mAP values for YOLOv6-3.0 and YOLOv9 models, highlighting their accuracy in detecting objects across various scenarios. mAP, a critical metric in object detection, evaluates the precision of these models at different IoU thresholds. Learn more about [mean average precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map) and its significance in model evaluation.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv6-3.0 | mAP<sup>val<br>50<br>YOLOv9 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.5 | 37.8 |
    	| s | 45.0 | 46.5 |
    	| m | 50.0 | 51.5 |
    	| l | 52.8 | 52.8 |
    	| x | N/A | 55.1 |

## Speed Comparison

This section highlights the performance differences between YOLOv6-3.0 and YOLOv9, showcasing speed metrics measured in milliseconds across various model sizes. These comparisons illustrate how efficiently each model processes data, providing valuable insights for real-time applications. Learn more about [Ultralytics benchmarks](https://docs.ultralytics.com/reference/utils/benchmarks/) for additional context.

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.17 | 2.3 |
    	| s | 2.66 | 3.54 |
    	| m | 5.28 | 6.43 |
    	| l | 8.95 | 7.16 |
    	| x | N/A | 16.77 |

## Fine-Tuning on the African Wildlife Dataset

Ultralytics YOLO11 enables seamless fine-tuning on various datasets, including the African Wildlife dataset. This dataset is designed to help users create AI models tailored for wildlife conservation and monitoring applications, like identifying and tracking species in their natural habitats. By leveraging YOLO11's robust capabilities, users can enhance the accuracy of their models for wildlife detection tasks.

Fine-tuning with the African Wildlife dataset involves using pre-trained YOLO11 weights as a starting point and adapting them to this specialized dataset. This process not only saves time but also improves the performance of detection models in niche applications.

For more insights into custom training and dataset preparation, explore the [Ultralytics YOLO11 Custom Training Guide](https://docs.ultralytics.com/modes/train/).

### Python Code Snippet for Fine-Tuning

```python
from ultralytics import YOLO

# Load pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Fine-tune the model on the African Wildlife dataset
model.train(data="african_wildlife.yaml", epochs=50, imgsz=640)

# Validate the model to check performance
metrics = model.val()

# Export the fine-tuned model
model.export(format="onnx")
```

This script demonstrates how to fine-tune YOLO11 on the African Wildlife dataset, validate its performance, and export it for deployment.
