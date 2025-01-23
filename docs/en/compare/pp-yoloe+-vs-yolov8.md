---
comments: true
description: Compare PP-YOLOE+ and Ultralytics YOLOv8 to discover their strengths in object detection, real-time AI performance, and edge AI applications. Explore how these state-of-the-art computer vision models excel in speed, accuracy, and versatility for diverse use cases.
keywords: PP-YOLOE+, Ultralytics YOLOv8, object detection, real-time AI, edge AI, computer vision, model comparison, advanced AI models
---

# PP-YOLOE+ VS Ultralytics YOLOv8

In the rapidly evolving field of computer vision, comparing state-of-the-art models like PP-YOLOE+ and Ultralytics YOLOv8 is crucial for understanding their unique strengths. Both models bring innovation to object detection, offering advanced capabilities for diverse real-world applications.

PP-YOLOE+ emphasizes efficient performance and accuracy, making it suitable for resource-constrained environments. Meanwhile, Ultralytics YOLOv8 delivers unmatched speed and flexibility, supporting seamless integration across tasks like object detection and segmentation. Learn more about [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) and its cutting-edge features.


## mAP Comparison

This section compares the mAP values of PP-YOLOE+ and Ultralytics YOLOv8, highlighting their accuracy in detecting and localizing objects across different variants. Learn more about how [mean average precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map) serves as a critical metric for evaluating model performance.


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>PP-YOLOE+ | mAP<sup>val<br>50<br>YOLOv8 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 39.9 | 37.3 |
		| s | 43.7 | 44.9 |
		| m | 49.8 | 50.2 |
		| l | 52.9 | 52.9 |
		| x | 54.7 | 53.9 |
		

## Speed Comparison

This section evaluates the speed performance of PP-YOLOE+ and Ultralytics YOLOv8 models across various sizes, measured in milliseconds. These metrics highlight the efficiency of each model in delivering real-time object detection results, providing insights into their suitability for time-sensitive applications. For more details on YOLOv8's advancements, refer to the [Ultralytics YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 2.84 | 1.47 |
		| s | 2.62 | 2.66 |
		| m | 5.56 | 5.86 |
		| l | 8.36 | 9.06 |
		| x | 14.3 | 14.37 |

## Train Ultralytics YOLO11 Models  

Ultralytics YOLO11 provides seamless functionality for training custom models on your specific datasets. Whether you're working with datasets like COCO8, African wildlife, or car parts segmentation, YOLO11 enables you to fine-tune pre-trained models for maximum accuracy. The training process ensures the model adapts to your unique application, from wildlife monitoring to industrial applications. 

For a detailed guide on using YOLO11 for training, check out the [Ultralytics YOLO11 Training Documentation](https://docs.ultralytics.com/modes/train/). It includes step-by-step instructions for loading datasets, setting hyperparameters, and monitoring metrics during the training process.

### Python Code Example  

```python
from ultralytics import YOLO  

# Load a pre-trained YOLO11 model  
model = YOLO('yolo11.pt')  

# Train the model on a custom dataset  
results = model.train(data='custom_dataset.yaml', epochs=50, imgsz=640)  

# Save and evaluate the trained model  
model.save('trained_model.pt')  
```

This code snippet demonstrates how to load a YOLO11 model, train it on a custom dataset, and save the results. For more examples, visit the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics).
