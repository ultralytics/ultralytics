---
comments: true
description: Explore a detailed comparison between Ultralytics YOLOv5 and YOLOv9, highlighting advancements in object detection, real-time AI capabilities, and efficiency for edge AI and computer vision applications. Learn how these models perform across various metrics and use cases.  
keywords: YOLOv5, YOLOv9, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, AI advancements
---
# Ultralytics YOLOv5 VS YOLOv9


Ultralytics YOLOv5 and YOLOv9 represent pivotal milestones in the evolution of object detection technology. This comparison highlights the advancements made between these two versions, showcasing how YOLOv9 builds on the legacy of YOLOv5 to deliver superior performance and versatility for real-world applications.  

While YOLOv5 is celebrated for its accessibility and widespread adoption, YOLOv9 introduces cutting-edge enhancements in speed, accuracy, and efficiency. By examining their unique strengths, such as YOLOv5's simplicity and YOLOv9's advanced feature extraction, we aim to provide a clear understanding of their capabilities. Explore more about these models in the [Ultralytics YOLOv5 documentation](https://docs.ultralytics.com/models/yolov5/) and the [YOLOv9 advancements](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai).


## mAP Comparison

This section compares the mAP (mean average precision) values of Ultralytics YOLOv5 and YOLOv9, showcasing their accuracy across various model sizes. mAP serves as a key metric for evaluating object detection performance, balancing precision and recall to provide a comprehensive assessment. Learn more about [mAP and its significance](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 37.8 |
		| s | 37.4 | 46.5 |
		| m | 45.4 | 51.5 |
		| l | 49.0 | 52.8 |
		| x | 50.7 | 55.1 |
		

## Speed Comparison

The speed metrics of Ultralytics YOLOv5 and YOLOv9 highlight their efficiency across various model sizes, with performance measured in milliseconds. YOLOv9 demonstrates advancements in latency reduction and computational efficiency, offering faster inference times compared to YOLOv5, especially in real-time applications. Explore more about [Ultralytics YOLOv5](https://docs.ultralytics.com/models/yolov5/) and [YOLOv9](https://docs.ultralytics.com/models/yolov9/) to understand their detailed performance metrics.


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | N/A | 2.3 |
		| s | 1.92 | 3.54 |
		| m | 4.03 | 6.43 |
		| l | 6.61 | 7.16 |
		| x | 11.89 | 16.77 |

## Training With Ultralytics YOLO11

Ultralytics YOLO11 simplifies the training process, allowing users to fine-tune models for various tasks using custom datasets. With its robust framework, YOLO11 supports diverse datasets like COCO8, African wildlife, and signature detection, enabling high adaptability for unique use cases. The model’s efficient architecture ensures faster convergence and improved accuracy during training.

For more details on training YOLO11, refer to the [custom training guide](https://docs.ultralytics.com/modes/train/), which provides step-by-step instructions and best practices.

### Python Code Example for Training YOLO11

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov11.pt')  

# Train the model
model.train(data='custom_dataset.yaml', epochs=50, imgsz=640, batch=16)
```

This code snippet demonstrates how to train a YOLO11 model using a custom dataset. Adjust the parameters like `epochs` and `imgsz` to fit your project requirements.
