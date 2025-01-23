---
comments: true
description: Compare Ultralytics YOLOv5 and YOLO11 to discover how these cutting-edge models excel in object detection, real-time AI, and computer vision. Explore advancements in accuracy, speed, and edge AI deployment for modern applications.
keywords: Ultralytics, YOLOv5, YOLO11, object detection, real-time AI, edge AI, computer vision, AI models comparison, YOLO series, Ultralytics models
---

# Ultralytics YOLOv5 VS YOLO11

# Ultralytics YOLOv5 VS Ultralytics YOLO11

Selecting the right model for your computer vision tasks is critical, and comparing Ultralytics YOLOv5 with Ultralytics YOLO11 highlights their evolution and unique capabilities. Both models represent milestones in real-time object detection, excelling in speed, accuracy, and efficiency.

Ultralytics YOLOv5 is celebrated for its versatility and ease of deployment, while Ultralytics YOLO11 introduces groundbreaking improvements in architecture, feature extraction, and scalability. Dive into this comparison to understand how these models cater to diverse applications, from [real-time detection](https://docs.ultralytics.com/guides/model-deployment-options/) to [large-scale AI solutions](https://www.ultralytics.com/blog/measuring-ai-performance-to-weigh-the-impact-of-your-innovations).

## mAP Comparison

This section compares the mAP (Mean Average Precision) values of Ultralytics YOLOv5 and Ultralytics YOLO11 across their variants, highlighting advancements in detection accuracy. mAP is a critical metric that evaluates how effectively models identify and localize objects, offering insights into their performance across different tasks. For more on mAP, explore [this detailed guide](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>YOLO11 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 39.5 |
    	| s | 37.4 | 47.0 |
    	| m | 45.4 | 51.4 |
    	| l | 49.0 | 53.2 |
    	| x | 50.7 | 54.7 |


## Speed Comparison

This section highlights the speed metrics in milliseconds for Ultralytics YOLOv5 and Ultralytics YOLO11, showcasing their real-time performance across different model sizes. Ultralytics YOLO11 demonstrates faster processing times, particularly with smaller models like YOLO11n, ideal for efficient deployment on edge devices. [Learn more about YOLO11 models](https://docs.ultralytics.com/models/yolo11/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 1.55 |
    	| s | 1.92 | 2.63 |
    	| m | 4.03 | 5.27 |
    	| l | 6.61 | 6.84 |
    	| x | 11.89 | 12.49 |

## Benchmarking With Ultralytics YOLO11

Ultralytics YOLO11 provides robust benchmarking capabilities, enabling users to evaluate model performance effectively. Benchmarking is essential for comparing inference speed, memory usage, and accuracy across different hardware or datasets. This feature is particularly valuable when optimizing models for real-world applications where performance consistency is critical.

To start benchmarking, use the `benchmark` mode in the Ultralytics YOLO11 Python package. This allows you to test the model on datasets like COCO8 or custom datasets, ensuring it meets your project requirements. Learn more about benchmarking best practices in the [Ultralytics YOLO Docs](https://docs.ultralytics.com/guides/).

### Python Code Snippet

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO('yolov11.pt')

# Benchmark model performance
results = model.benchmark(data='coco8.yaml', device='0')

# Print benchmarking results
print(results)
```

Leverage benchmarking to fine-tune your model for optimal performance across various tasks and hardware setups.
