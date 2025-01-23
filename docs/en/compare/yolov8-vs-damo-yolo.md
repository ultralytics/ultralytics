---
comments: true  
description: Explore a detailed comparison between Ultralytics YOLOv8 and DAMO-YOLO, showcasing their performance, speed, and accuracy in real-time object detection and edge AI applications. Discover which model excels in computer vision tasks for cutting-edge innovation.  
keywords: Ultralytics YOLOv8, DAMO-YOLO, object detection, real-time AI, edge AI, computer vision, AI model comparison, Ultralytics
---

# Ultralytics YOLOv8 VS DAMO-YOLO

In the rapidly evolving field of computer vision, comparing state-of-the-art models like Ultralytics YOLOv8 and DAMO-YOLO is critical for understanding their unique strengths and real-world applications. Both models excel in object detection, segmentation, and classification but take distinct approaches to optimize accuracy, speed, and efficiency.

Ultralytics YOLOv8 is celebrated for its flexibility, seamless compatibility with previous YOLO versions, and intuitive workflows, making it ideal for diverse use cases. Meanwhile, DAMO-YOLO offers robust performance and innovative architectural choices, catering to specific needs in high-performance computing environments. Learn more about [Ultralytics YOLOv8](https://www.ultralytics.com/blog/introducing-ultralytics-yolov8) and its advancements here.


## mAP Comparison

This section compares the mAP (Mean Average Precision) values of Ultralytics YOLOv8 and DAMO-YOLO to highlight their accuracy across different variants. mAP serves as a comprehensive metric, balancing precision and recall to evaluate object detection performance effectively. Learn more about [mAP metrics here](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>DAMO-YOLO |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 37.3 | 42.0 |
		| s | 44.9 | 46.0 |
		| m | 50.2 | 49.2 |
		| l | 52.9 | 50.8 |
		| x | 53.9 | N/A |
		

## Speed Comparison

This section highlights the speed performance of Ultralytics YOLOv8 versus DAMO-YOLO across various model sizes. Speed metrics, measured in milliseconds, demonstrate the efficiency of these models in terms of real-time inference, providing insights into their suitability for diverse applications. For more details, explore [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/) or learn about DAMO-YOLO's advancements [here](https://github.com/damo-yolo).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 1.47 | 2.32 |
		| s | 2.66 | 3.45 |
		| m | 5.86 | 5.09 |
		| l | 9.06 | 7.18 |
		| x | 14.37 | N/A |

## Using the Export Functionality in YOLO11

The Export feature in Ultralytics YOLO11 allows users to convert their trained models into various formats compatible with diverse deployment environments. Supported formats include ONNX, OpenVINO, TensorFlow Lite, and more, enabling seamless integration into edge devices, cloud services, or production pipelines. This functionality ensures flexibility and optimizes the model for real-world applications.

For instance, leveraging the Export feature with formats like OpenVINO enhances performance by reducing latency, crucial for real-time applications. To learn more about model deployment options and the pros and cons of each format, refer to the [Model Deployment Options Guide](https://docs.ultralytics.com/guides/).

### Python Code Example: Export YOLO11 Model

```python
from ultralytics import YOLO

# Load a trained YOLO11 model
model = YOLO("path/to/your/model.pt")

# Export the model to ONNX format
model.export(format="onnx")
```

This simple process ensures your YOLO11 model is ready for deployment across various platforms, making it an indispensable tool for developers and businesses alike.
