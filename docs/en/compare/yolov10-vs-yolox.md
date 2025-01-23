---
comments: true
description: Explore the detailed comparison between YOLOv10 and YOLOX, two cutting-edge models in real-time object detection. Discover how these models differ in architecture, performance, and efficiency, catering to diverse applications in computer vision and edge AI.
keywords: YOLOv10, YOLOX, Ultralytics, object detection, real-time AI, edge AI, computer vision
---

# YOLOv10 VS YOLOX

The comparison between YOLOv10 and YOLOX highlights the evolution of real-time object detection models, emphasizing their unique approaches and technological advancements. Both models aim to optimize accuracy and efficiency, making them highly suitable for diverse applications in computer vision.

YOLOv10, developed with a focus on NMS-free training and holistic design, delivers exceptional performance with reduced computational overhead. Meanwhile, YOLOX, an open-source model, stands out with its anchor-free design and decoupled head architecture, ensuring robust and scalable solutions for modern AI challenges. Explore their capabilities in more detail through [Ultralytics resources](https://docs.ultralytics.com/models/).

## mAP Comparison

This section compares the mean Average Precision (mAP) values of YOLOv10 and YOLOX across various model variants, showcasing their accuracy in object detection tasks. mAP is a key metric that evaluates a model's precision and recall, making it a crucial factor for assessing performance. Learn more about [mAP evaluation](https://www.ultralytics.com/glossary/mean-average-precision-map) and its importance in object detection.

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv10 | mAP<sup>val<br>50<br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.5 | N/A |
    	| s | 46.7 | 40.5 |
    	| m | 51.3 | 46.9 |
    	| b | 52.7 | N/A |
    	| l | 53.3 | 49.7 |
    	| x | 54.4 | 51.1 |

## Speed Comparison

This section highlights the speed performance of YOLOv10 and YOLOX models across various sizes, measured in milliseconds. By comparing inference times, discover how YOLOv10's optimized architecture provides faster processing than YOLOX, especially on modern GPUs like the T4. Learn more about YOLOv10's efficiency [here](https://docs.ultralytics.com/models/yolov10/) and explore YOLOX's benchmarks [here](https://github.com/Megvii-BaseDetection/YOLOX).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv10 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.56 | N/A |
    	| s | 2.66 | 2.56 |
    	| m | 5.48 | 5.43 |
    	| b | 6.54 | N/A |
    	| l | 8.33 | 9.04 |
    	| x | 12.2 | 16.1 |

## Benchmark Model Performance With YOLO11

Ultralytics YOLO11 provides a robust functionality for **benchmarking**, enabling users to evaluate the performance of their models across various metrics and datasets. This feature is crucial for understanding the efficiency and accuracy of your YOLO11 implementations in real-world scenarios. By leveraging YOLO11's benchmarking tools, you can assess metrics like latency, throughput, and mean Average Precision (mAP) to identify areas for optimization.

For more insights into performance metrics and evaluation, refer to the [YOLO Performance Metrics guide](https://docs.ultralytics.com/guides/). This guide also includes practical examples to improve model accuracy and speed, ensuring your YOLO11 deployment is finely tuned for your needs.

### Example: Benchmarking With YOLO11 in Python

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Benchmark the model on a dataset
results = model.benchmark(data="coco128.yaml", imgsz=640, batch_size=16)

# Print benchmarking results
print(results)
```

This example demonstrates how to run a benchmark on the COCO128 dataset, providing valuable performance statistics to guide your model optimization efforts. For advanced benchmarking techniques, explore the [Ultralytics YOLO Docs](https://docs.ultralytics.com/).
