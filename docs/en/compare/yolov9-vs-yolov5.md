---
comments: true
description: Compare YOLOv9 and Ultralytics YOLOv5 to discover their strengths in object detection, real-time AI, and edge AI. Explore performance benchmarks, architecture advancements, and suitability for various computer vision applications. 
keywords: YOLOv9, Ultralytics YOLOv5, Ultralytics, object detection, real-time AI, edge AI, computer vision, AI models, performance comparison
---

# YOLOv9 VS Ultralytics YOLOv5

The comparison between YOLOv9 and Ultralytics YOLOv5 showcases the evolution of object detection technologies, highlighting key advancements in the YOLO series. Both models represent significant milestones in AI, offering unique strengths tailored to various real-world applications.

YOLOv9 introduces cutting-edge innovations in accuracy and efficiency, building upon its predecessors to handle complex tasks with ease. On the other hand, Ultralytics YOLOv5 remains a widely adopted model due to its simplicity, speed, and robust performance across diverse use cases. Learn more about [YOLOv5's architecture](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/) and its impact on modern AI solutions.


## mAP Comparison

This section highlights the mAP values of YOLOv9 and Ultralytics YOLOv5, showcasing their accuracy across different variants. Mean Average Precision (mAP) serves as a comprehensive metric to evaluate the object detection performance of these models, balancing precision and recall. Learn more about [mAP calculations](https://www.ultralytics.com/glossary/mean-average-precision-map) to understand its impact on model evaluation.


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv9 | mAP<sup>val<br>50<br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 37.8 | N/A |
		| s | 46.5 | 37.4 |
		| m | 51.5 | 45.4 |
		| l | 52.8 | 49.0 |
		| x | 55.1 | 50.7 |
		

## Speed Comparison

This section compares the speed performance of YOLOv9 and Ultralytics YOLOv5 across various model sizes, highlighting inference times in milliseconds. These metrics illustrate the efficiency and real-time capabilities of each model, crucial for applications like [autonomous driving](https://www.ultralytics.com/blog/ultralytics-yolov8-for-speed-estimation-in-computer-vision-projects) and [smart retail](https://www.ultralytics.com/blog/ai-for-smarter-retail-inventory-management).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv9 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 2.3 | N/A |
		| s | 3.54 | 1.92 |
		| m | 6.43 | 4.03 |
		| l | 7.16 | 6.61 |
		| x | 16.77 | 11.89 |

## Segmentation With Ultralytics YOLO11  

Ultralytics YOLO11 excels at segmentation tasks, enabling users to identify and isolate specific objects within an image. This functionality is particularly useful in applications like car parts segmentation, where precise object boundaries are necessary for tasks such as automotive repairs, manufacturing, or e-commerce cataloging. YOLO11’s segmentation capabilities are powered by its robust architecture, ensuring high accuracy and efficiency.  

Custom training on datasets like [Car Parts Segmentation](https://docs.ultralytics.com/datasets/segment/carparts-seg/) allows YOLO11 to adapt to specialized requirements, making it suitable for industries with unique use cases. The model supports pre-trained weights and fine-tuning, streamlining the training process for both general and specific segmentation tasks.  

For more details on segmentation with YOLO11, check out this [image segmentation guide](https://www.ultralytics.com/blog/image-segmentation-with-ultralytics-yolo11-on-google-colab).  

### Python Code Example  

```python  
from ultralytics import YOLO  

# Load a pre-trained YOLO11 model  
model = YOLO('yolo11-seg.pt')  

# Perform segmentation on an image  
results = model('car_parts.jpg', task='segment')  

# Display results  
results.show()  
```
