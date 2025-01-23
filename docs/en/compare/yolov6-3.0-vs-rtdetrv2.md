---
comments: true
description: Explore a detailed comparison between YOLOv6-3.0 and RTDETRv2, analyzing their performance, efficiency, and suitability for real-time AI and edge AI applications. Discover how these cutting-edge object detection models excel in computer vision tasks and adapt to diverse scenarios with optimized speed and accuracy.
keywords: YOLOv6-3.0, RTDETRv2, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, performance analysis
---

# YOLOv6-3.0 VS RTDETRv2

The comparison between YOLOv6-3.0 and RTDETRv2 showcases the evolution of real-time object detection models, emphasizing their respective advancements in speed and accuracy. These models cater to diverse applications, making it essential to understand their unique strengths and performance metrics.

YOLOv6-3.0 emphasizes efficiency with its lightweight design and optimized architecture for real-time tasks, while RTDETRv2 leverages Vision Transformer-based enhancements for improved accuracy and adaptability. Learn more about [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) and [YOLOv6](https://github.com/meituan/YOLOv6) to explore their cutting-edge features.

## mAP Comparison

This section compares the mAP (Mean Average Precision) values of YOLOv6-3.0 and RTDETRv2, highlighting their accuracy across various model variants. mAP serves as a key metric for evaluating the ability of object detection models to correctly locate and classify objects. Learn more about [mAP and its importance](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv6-3.0 | mAP<sup>val<br>50<br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 37.5 | N/A |
    	| s | 45.0 | 48.1 |
    	| m | 50.0 | 51.9 |
    	| l | 52.8 | 53.4 |
    	| x | N/A | 54.3 |


## Speed Comparison

This section highlights the speed performance of YOLOv6-3.0 and RTDETRv2 across various sizes, measured in milliseconds. These metrics provide an essential benchmark for evaluating real-time capabilities and efficiency in diverse deployment scenarios. Learn more about how speed impacts performance on [Ultralytics YOLO Docs](https://docs.ultralytics.com/reference/utils/benchmarks/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv6-3.0 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.17 | N/A |
    	| s | 2.66 | 5.03 |
    	| m | 5.28 | 7.51 |
    	| l | 8.95 | 9.76 |
    	| x | N/A | 15.03 |

## Benchmarking with Ultralytics YOLO11

Benchmarking is a critical functionality of Ultralytics YOLO11, enabling users to evaluate model performance across various metrics such as speed, accuracy, and memory utilization. This feature allows developers to compare YOLO11 against other models or validate its efficiency across different hardware configurations. Whether using GPU or CPU environments, benchmarking ensures you achieve optimal results tailored to your specific use case.

Ultralytics YOLO11's benchmarking tools help identify bottlenecks and provide actionable insights for improving deployment efficiency. The feature is especially valuable for real-time applications like autonomous vehicles and surveillance systems, where performance is paramount.

For more details on YOLO11's capabilities, refer to [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/).

### Python Code Snippet: Benchmarking YOLO11

```python
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11.pt")

# Benchmark the model on a sample dataset
results = model.benchmark(data="coco8.yaml", imgsz=640, device="cuda")

# Print the benchmarking results
print(results)
```

By leveraging benchmarking, you can maximize YOLO11's potential for your computer vision projects.
