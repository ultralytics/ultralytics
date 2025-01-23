---
comments: true
description: Explore a detailed comparison between PP-YOLOE+ and Ultralytics YOLOv5, highlighting their performance in object detection, real-time AI applications, and edge AI scenarios. Dive into their speed, accuracy, and parameter efficiency to uncover which model excels in modern computer vision tasks.  
keywords: PP-YOLOE+, Ultralytics YOLOv5, object detection, real-time AI, edge AI, computer vision, performance comparison, AI models.
---

# PP-YOLOE+ VS Ultralytics YOLOv5

When it comes to state-of-the-art object detection, comparing models like PP-YOLOE+ and Ultralytics YOLOv5 showcases the advancements in performance, efficiency, and versatility. Both models are designed to address complex computer vision challenges, making them essential tools in real-world AI applications.

PP-YOLOE+ highlights its strengths in optimized inference speed and resource efficiency, making it a solid choice for edge devices. On the other hand, [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) emphasizes ease of use and integration, backed by a robust [community](https://discord.com/invite/ultralytics) and extensive support for diverse tasks.


## mAP Comparison

This section highlights the mAP values of PP-YOLOE+ and Ultralytics YOLOv5, showcasing their accuracy across different model variants. mAP, or [Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map), is a key metric in object detection, providing a comprehensive evaluation of a model's precision and recall across multiple classes.


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>PP-YOLOE+ | mAP<sup>val<br>50<br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 39.9 | N/A |
		| s | 43.7 | 37.4 |
		| m | 49.8 | 45.4 |
		| l | 52.9 | 49.0 |
		| x | 54.7 | 50.7 |
		

## Speed Comparison

This section highlights the speed performance of PP-YOLOE+ and Ultralytics YOLOv5 models across various sizes. Speed metrics, measured in milliseconds, demonstrate how efficiently each model processes data, offering valuable insights for real-time applications. For more details on YOLOv5's architecture, visit the [Ultralytics YOLOv5 documentation](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 2.84 | N/A |
		| s | 2.62 | 1.92 |
		| m | 5.56 | 4.03 |
		| l | 8.36 | 6.61 |
		| x | 14.3 | 11.89 |

## Train with Ultralytics YOLO11  

Training a custom model with Ultralytics YOLO11 is straightforward, offering the flexibility to fine-tune on specific datasets such as COCO8, African wildlife, or more. YOLO11 supports streamlined training workflows that leverage pre-trained weights and advanced optimization techniques. This ensures high accuracy and adaptability for diverse use cases.  

For a comprehensive guide, explore [YOLO11 Training Documentation](https://docs.ultralytics.com/modes/train/).  

### Python Example  

```python  
from ultralytics import YOLO  

# Load a pre-trained YOLO11 model  
model = YOLO('yolo11.pt')  

# Train the model on a custom dataset  
model.train(data='config.yaml', epochs=50, batch=16, imgsz=640)  

# Evaluate model performance  
metrics = model.val()  
print(metrics)  
```  

Train on datasets like COCO8 or custom annotations to meet your project’s specific requirements. Learn more about dataset preparation [here](https://docs.ultralytics.com/datasets/).
