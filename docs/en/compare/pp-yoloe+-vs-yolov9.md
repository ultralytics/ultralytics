---
comments: true  
description: Discover a detailed comparison between PP-YOLOE+ and YOLOv9, two cutting-edge models in the field of real-time object detection. Explore their performance metrics, efficiency, and suitability for diverse computer vision tasks, including edge AI applications, powered by Ultralytics innovation.  
keywords: PP-YOLOE+, YOLOv9, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison
---

# PP-YOLOE+ VS YOLOv9

Comparing PP-YOLOE+ and YOLOv9 provides valuable insights into the evolution of object detection models, showcasing advancements in speed, accuracy, and efficiency. These models represent significant milestones in the field, offering distinct features tailored to diverse real-world applications.

PP-YOLOE+ is known for its balance of computational efficiency and high accuracy, making it suitable for resource-constrained deployments. On the other hand, YOLOv9, part of the renowned Ultralytics YOLO series, introduces innovative architectural improvements and optimized training techniques for superior performance across various computer vision tasks. Learn more about [YOLO models](https://www.ultralytics.com/blog/the-evolution-of-object-detection-and-ultralytics-yolo-models).


## mAP Comparison

The mAP comparison highlights the accuracy of PP-YOLOE+ and YOLOv9 across various model variants, showcasing their performance on object detection tasks. Mean Average Precision ([mAP](https://www.ultralytics.com/glossary/mean-average-precision-map)) is a comprehensive metric that evaluates a model's ability to detect and classify objects with precision and recall, making it a key benchmark for these architectures. Explore more on [YOLOv9's advancements](https://docs.ultralytics.com/models/yolov9/).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>PP-YOLOE+ | mAP<sup>val<br>50<br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 39.9 | 37.8 |
		| s | 43.7 | 46.5 |
		| m | 49.8 | 51.5 |
		| l | 52.9 | 52.8 |
		| x | 54.7 | 55.1 |
		

## Speed Comparison

This section highlights the speed performance of PP-YOLOE+ and YOLOv9 across various model sizes, measured in milliseconds. These metrics emphasize the efficiency and real-time capabilities of each model, offering insights into their deployment potential for diverse applications. Learn more about [YOLOv9 performance](https://docs.ultralytics.com/models/yolov9/) and [PP-YOLOE+ details](https://github.com/PaddlePaddle/PaddleDetection).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 2.84 | 2.3 |
		| s | 2.62 | 3.54 |
		| m | 5.56 | 6.43 |
		| l | 8.36 | 7.16 |
		| x | 14.3 | 16.77 |

## Benchmarking With YOLO11

Ultralytics YOLO11 offers robust benchmarking capabilities, allowing users to evaluate the performance of their models across various datasets and tasks. Benchmarking is essential for comparing metrics like speed, accuracy, and efficiency to ensure your model meets project-specific requirements. This functionality is especially useful for identifying areas for optimization during model development and deployment.

YOLO11 provides built-in tools to streamline the benchmarking process, offering insights into metrics such as mAP, inference time, and resource utilization. These tools can be used seamlessly across supported platforms, including ONNX, OpenVINO, and TensorFlow Lite, providing flexibility in deployment scenarios.

For a deeper understanding of performance metrics and their impact, refer to the [YOLO Performance Metrics Guide](https://docs.ultralytics.com/guides/). This guide includes practical examples and tips for enhancing model performance.

### Python Code Example for Benchmarking

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO('yolo11.pt')

# Benchmark the model on a dataset
results = model.benchmark(data='coco8.yaml', imgsz=640, device='0')

# Display benchmarking results
print(results)
```
