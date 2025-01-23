---
comments: true
description: Compare YOLOv9 and Ultralytics YOLO11, two powerful models in computer vision. Discover how Ultralytics YOLO11 outperforms with its enhanced accuracy, faster real-time object detection capabilities, and efficiency optimized for edge AI and cloud deployment. Dive into their key differences and advancements in real-time AI technology.
keywords: YOLOv9, Ultralytics YOLO11, object detection, real-time AI, edge AI, computer vision, YOLO models, AI comparison
---

# YOLOv9 VS Ultralytics YOLO11

As computer vision technology evolves, comparing models like YOLOv9 and Ultralytics YOLO11 is essential to understanding advancements in speed, accuracy, and efficiency. This page highlights the unique strengths of each model, offering insights into their capabilities for diverse applications, including real-time detection and large-scale deployment.

YOLOv9 laid the foundation with its reliable performance and robust architecture, while Ultralytics YOLO11 takes innovation further with enhanced feature extraction and optimized training techniques. Whether you're exploring cutting-edge [augmentation pipelines](https://www.ultralytics.com/ru/blog/what-are-diffusion-models-a-quick-and-comprehensive-guide#data-preprocessing) or seeking superior [deployment options](https://docs.ultralytics.com/guides/model-deployment-options/), this comparison provides a comprehensive perspective on the advancements shaping modern AI solutions.


## mAP Comparison

The mAP (mean Average Precision) values highlight the accuracy of YOLOv9 and Ultralytics YOLO11 across their respective variants, showcasing advancements in object detection performance. While YOLOv9 set new standards with its efficiency, Ultralytics YOLO11 further improves on these benchmarks, achieving higher mAP scores with fewer parameters and faster processing. Learn more about [mAP and its significance](https://www.ultralytics.com/glossary/mean-average-precision-map) in evaluating object detection models.


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>YOLO11 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 37.8 | 39.5 |
		| s | 46.5 | 47.0 |
		| m | 51.5 | 51.4 |
		| l | 52.8 | 53.2 |
		| x | 55.1 | 54.7 |
		

## Speed Comparison

This section highlights the speed performance of YOLOv9 and Ultralytics YOLO11 across various model sizes, measured in milliseconds. With optimized designs, Ultralytics YOLO11 demonstrates faster inference times, offering significant efficiency improvements for real-time applications. For more details, explore the [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 2.3 | 1.55 |
		| s | 3.54 | 2.63 |
		| m | 6.43 | 5.27 |
		| l | 7.16 | 6.84 |
		| x | 16.77 | 12.49 |

## Benchmarking With Ultralytics YOLO11

Ultralytics YOLO11 provides robust benchmarking tools to evaluate model performance across various scenarios. Benchmarking allows users to measure metrics such as speed, accuracy, and efficiency, making it easier to optimize models for specific use cases. For example, you can assess performance on datasets like COCO8 or African wildlife to understand how well YOLO11 generalizes across different tasks.

This functionality ensures that you can compare YOLO11 against other models or fine-tuned versions to make data-driven decisions. YOLO11’s benchmarking tools integrate seamlessly with popular frameworks like TensorFlow Lite and ONNX, providing flexibility in deployment and testing.

To learn more about performance metrics such as mAP, IoU, and F1 score, visit the [YOLO Performance Metrics Guide](https://docs.ultralytics.com/guides/).

### Python Code Snippet for Benchmarking

```python
from ultralytics import YOLO

# Load a pretrained YOLO11 model
model = YOLO('yolo11.pt')

# Benchmark the model on a dataset
results = model.benchmark(data='coco8.yaml', imgsz=640, batch=32)

# Print benchmark results
print(results)
```

This script demonstrates how to benchmark the YOLO11 model on the COCO8 dataset, providing insights into its performance.
