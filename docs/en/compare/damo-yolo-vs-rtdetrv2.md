---
comments: true  
description: Compare DAMO-YOLO and RTDETRv2 in this detailed analysis of real-time object detection models. Explore their performance, accuracy, and suitability for edge AI and computer vision applications, powered by Ultralytics.  
keywords: DAMO-YOLO, RTDETRv2, Ultralytics, object detection, real-time AI, edge AI, computer vision, model comparison, AI performance, detection models
---

# DAMO-YOLO VS RTDETRv2

This comparison page dives into the nuances of DAMO-YOLO and RTDETRv2, two leading-edge object detection models shaping the future of computer vision. By evaluating their performance, efficiency, and scalability, we aim to provide clarity for professionals seeking the best solution for advanced AI applications.

DAMO-YOLO brings exceptional speed and precision, leveraging innovative techniques for real-time object detection. On the other hand, RTDETRv2, built on Vision Transformer-based architecture, emphasizes accuracy and adaptability across diverse use cases. Explore their unique capabilities to determine which model best fits your needs. For more details, check out [Ultralytics YOLO models](https://docs.ultralytics.com/models/) and [RT-DETR resources](https://docs.ultralytics.com/reference/models/rtdetr/model/).


## mAP Comparison

This section highlights the mAP values of DAMO-YOLO and RTDETRv2, showcasing their accuracy across different variants. Mean Average Precision (mAP) evaluates detection performance by balancing precision and recall, offering a comprehensive measure of a model's effectiveness. Learn more about [mAP metrics](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>DAMO-YOLO | mAP<sup>val<br>50<br>RTDETRv2 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 42.0 | N/A |
		| s | 46.0 | 48.1 |
		| m | 49.2 | 51.9 |
		| l | 50.8 | 53.4 |
		| x | N/A | 54.3 |
		

## Speed Comparison

This section highlights the speed performance of DAMO-YOLO and RTDETRv2 models, measured in milliseconds across various sizes. These metrics provide critical insights into the efficiency and deployment suitability of each model on different hardware setups. Learn more about [benchmarking models](https://docs.ultralytics.com/modes/benchmark/) for optimal performance.


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 2.32 | N/A |
		| s | 3.45 | 5.03 |
		| m | 5.09 | 7.51 |
		| l | 7.18 | 9.76 |
		| x | N/A | 15.03 |

## YOLO11 Functionalities: Predict  

Ultralytics YOLO11 introduces advanced **predict** functionalities, enabling users to perform real-time object detection with exceptional accuracy. This feature supports a wide range of applications, from monitoring surveillance footage to analyzing wildlife behavior. With YOLO11, you can easily predict outcomes on both custom and pre-trained datasets, ensuring flexibility for varied use cases.

The predict functionality is streamlined for ease of use and can be executed with a single command in Python. For additional details on setting up predictions, you can refer to the [Ultralytics YOLO documentation](https://docs.ultralytics.com/guides/).

### Python Code Snippet  

```python
from ultralytics import YOLO  

# Load the YOLO11 model
model = YOLO('yolo11.pt')  

# Perform prediction on an image
results = model.predict(source='image.jpg', show=True)  

# Display results
results.show()
```  

This predictive capability supports various file types, including images and video streams, making it suitable for dynamic environments. For more information about YOLO11's other functionalities, check out [How to Use YOLO11 for Object Detection](https://www.ultralytics.com/blog/how-to-use-ultralytics-yolo11-for-object-detection).
