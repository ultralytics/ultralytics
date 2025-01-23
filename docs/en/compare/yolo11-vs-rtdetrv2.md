---
comments: true
description: Explore a detailed comparison between Ultralytics YOLO11 and RTDETRv2, two leading models in real-time AI. Discover their strengths in object detection, speed, accuracy, and deployment capabilities for edge and cloud-based computer vision applications.
keywords: Ultralytics, YOLO11, RTDETRv2, object detection, real-time AI, edge AI, computer vision, model comparison
---

# Ultralytics YOLO11 VS RTDETRv2

The comparison between Ultralytics YOLO11 and RTDETRv2 showcases two cutting-edge models at the forefront of real-time object detection. Both models have been engineered to deliver exceptional performance, balancing speed, accuracy, and computational efficiency across diverse applications.

Ultralytics YOLO11 introduces refined architecture and advanced feature extraction, making it a leader in precision and efficiency for real-time tasks. On the other hand, RTDETRv2 emphasizes its strengths in latency optimization and robust deployment capabilities, particularly on edge devices. Explore [Ultralytics YOLO11 features](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai) to see how it redefines AI applications.

## mAP Comparison

Mean Average Precision (mAP) is a critical metric for evaluating object detection model accuracy across different classes and confidence thresholds. This section compares the mAP performance of Ultralytics YOLO11 and RTDETRv2, highlighting their capabilities on benchmarks like the [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/). Explore how these models balance precision and recall for superior detection accuracy.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLO11 | mAP<sup>val<br>50<br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.5 | N/A |
    	| s | 47.0 | 48.1 |
    	| m | 51.4 | 51.9 |
    	| l | 53.2 | 53.4 |
    	| x | 54.7 | 54.3 |


## Speed Comparison

This section highlights the speed efficiency of Ultralytics YOLO11 and RTDETRv2 models across various sizes, with performance measured in milliseconds. The comparison underscores the advantages of YOLO11's optimized architecture for tasks requiring rapid inference. Learn more about [benchmarking metrics](https://docs.ultralytics.com/modes/benchmark/) and [YOLO11 models](https://docs.ultralytics.com/tasks/obb/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.55 | N/A |
    	| s | 2.63 | 5.03 |
    	| m | 5.27 | 7.51 |
    	| l | 6.84 | 9.76 |
    	| x | 12.49 | 15.03 |

## Predict Using YOLO11

Ultralytics YOLO11 provides a robust and efficient framework for making predictions on images or videos. The predict functionality is designed to deliver real-time object detection and segmentation results, making it ideal for applications across industries like surveillance, retail, and wildlife monitoring. YOLO11 supports a wide range of pre-trained models, including COCO8, and also enables users to fine-tune the model for specific datasets.

To learn more about how Ultralytics YOLO11 enhances prediction capabilities, check out the [documentation](https://docs.ultralytics.com/guides/).

### Python Code Example

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Make predictions on an image
results = model.predict(source="image.jpg", save=True)

# Display results
results.show()
```

This example demonstrates how to load a YOLO11 model and perform predictions on an image. For further details, explore the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics).
