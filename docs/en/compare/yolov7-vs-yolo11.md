---
comments: true
description: Compare YOLOv7 and Ultralytics YOLO11 to explore advancements in object detection, speed, and accuracy. Discover how these models perform in real-time AI, edge AI, and computer vision applications for various industries.
keywords: YOLOv7, Ultralytics YOLO11, object detection, real-time AI, edge AI, computer vision, model comparison
---

# YOLOv7 VS Ultralytics YOLO11

Comparing YOLOv7 and Ultralytics YOLO11 provides a unique opportunity to explore the evolution of real-time object detection models. Both models represent significant advancements in computer vision, offering cutting-edge features tailored for diverse applications.

YOLOv7 is celebrated for its optimized speed and performance, making it a strong contender in real-time tasks. Meanwhile, Ultralytics YOLO11 sets a new benchmark with its improved architecture, enhanced feature extraction, and higher efficiency, as highlighted in its [official documentation](https://docs.ultralytics.com/models/yolo11/).


## mAP Comparison

This section highlights the mAP values for YOLOv7 and Ultralytics YOLO11, showcasing their accuracy in object detection tasks across different model variants. mAP, or Mean Average Precision, is a comprehensive metric that evaluates model performance by balancing precision and recall. Learn more about [mAP here](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv7 | mAP<sup>val<br>50<br>YOLO11 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 39.5 |
		| s | N/A | 47.0 |
		| m | N/A | 51.4 |
		| l | 51.4 | 53.2 |
		| x | 53.1 | 54.7 |
		

## Speed Comparison

This section highlights the speed performance of YOLOv7 and Ultralytics YOLO11 across different model sizes. Speed metrics, measured in milliseconds, demonstrate how YOLO11 achieves faster inference times, making it ideal for real-time applications. Learn more about [YOLOv7](https://docs.ultralytics.com/models/yolov7/) and [Ultralytics YOLO11](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv7 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 1.55 |
		| s | N/A | 2.63 |
		| m | N/A | 5.27 |
		| l | 6.84 | 6.84 |
		| x | 11.57 | 12.49 |

## Leveraging YOLO11 for Benchmarking  

Ultralytics YOLO11 offers robust benchmarking tools to evaluate model performance effectively across various tasks. Benchmarking helps users assess the accuracy, speed, and efficiency of their YOLO11 models in real-world scenarios. This feature is particularly useful for comparing results across datasets or hardware setups. For instance, you can measure performance on datasets like COCO8 or custom datasets to ensure your model meets project requirements.  

To learn more about YOLO11’s benchmarking capabilities and how to optimize your workflow, explore the detailed [Ultralytics Documentation](https://docs.ultralytics.com/guides/yolo-performance-metrics/).  

### Example: Benchmarking With YOLO11  

```python  
from ultralytics import YOLO  

# Load the YOLO11 model  
model = YOLO('yolo11n.pt')  

# Run benchmarking on a dataset  
results = model.benchmark(data='coco.yaml', imgsz=640, batch=16)  

# Print benchmarking results  
print(results)  
```  

This code snippet demonstrates how to benchmark a YOLO11 model using a COCO-like dataset. For more insights on performance metrics like mAP, IoU, and latency, check the [Performance Metrics Guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/).
