---
comments: true
description: Explore the detailed comparison between PP-YOLOE+ and YOLOX, two state-of-the-art models in real-time AI and object detection. Discover their performance metrics, speed-accuracy trade-offs, and suitability for edge AI and computer vision applications.
keywords: PP-YOLOE+, YOLOX, Ultralytics, object detection, real-time AI, edge AI, computer vision
---

# PP-YOLOE+ VS YOLOX

The comparison between PP-YOLOE+ and YOLOX sheds light on two advanced object detection models, each designed to push the boundaries of computer vision. By evaluating their performance on metrics like speed, accuracy, and efficiency, this page will help you determine which model best suits your specific use case.

PP-YOLOE+ is renowned for its balanced optimization, delivering high accuracy alongside computational efficiency, making it ideal for resource-constrained scenarios. On the other hand, YOLOX stands out for its versatility and robust architecture, excelling in real-time tasks across various environments. Learn more about YOLO advancements on [Ultralytics' YOLO models page](https://docs.ultralytics.com/models/yolo11/) or explore [real-time applications](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai).

## mAP Comparison

This section evaluates the mAP (Mean Average Precision) performance of PP-YOLOE+ and YOLOX, providing insights into their accuracy across different object detection scenarios. mAP values, as a comprehensive metric, reflect the models' balance of precision and recall, making it essential for understanding their real-world effectiveness. Learn more about [mAP here](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>PP-YOLOE+ | mAP<sup>val<br>50<br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.9 | N/A |
    	| s | 43.7 | 40.5 |
    	| m | 49.8 | 46.9 |
    	| l | 52.9 | 49.7 |
    	| x | 54.7 | 51.1 |

## Speed Comparison

The speed comparison highlights the performance differences between PP-YOLOE+ and YOLOX models, measured in milliseconds across various sizes. These metrics provide insights into their efficiency, with PP-YOLOE+ demonstrating faster inference speeds in certain configurations. Explore more about [PP-YOLOE+ here](https://github.com/PaddlePaddle/PaddleDetection) and [YOLOX details here](https://github.com/Megvii-BaseDetection/YOLOX).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.84 | N/A |
    	| s | 2.62 | 2.56 |
    	| m | 5.56 | 5.43 |
    	| l | 8.36 | 9.04 |
    	| x | 14.3 | 16.1 |

## Benchmarking With YOLO11

Ultralytics YOLO11 offers a robust benchmarking functionality that allows users to assess model performance across various metrics such as accuracy, inference speed, and memory utilization. This feature is particularly useful for comparing YOLO11's effectiveness with other models or configurations, ensuring optimal deployment strategies for real-world scenarios.

For detailed insights into benchmarking and performance metrics like mAP, IoU, and F1 score, explore [YOLO Performance Metrics](https://docs.ultralytics.com/guides/).

### Python Code for Benchmarking

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11.pt")

# Benchmark the model on COCO8 dataset
results = model.benchmark(data="coco8.yaml")

# Print benchmarking results
print(results)
```

Leverage YOLO11's benchmarking capabilities to fine-tune your workflows and achieve superior results in computer vision projects.
