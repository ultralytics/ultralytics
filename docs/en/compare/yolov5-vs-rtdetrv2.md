---
comments: true
description: Explore the in-depth comparison between Ultralytics YOLOv5 and RTDETRv2, two leading models in object detection and real-time AI. Discover their performance, efficiency, and applications in edge AI and computer vision to determine the best fit for your needs.
keywords: Ultralytics, YOLOv5, RTDETRv2, object detection, real-time AI, edge AI, computer vision, model comparison
---

# Ultralytics YOLOv5 VS RTDETRv2

The comparison of Ultralytics YOLOv5 and RTDETRv2 highlights two leading models in the field of real-time object detection. Both models are celebrated for their performance, but they cater to different use cases and excel in unique aspects, making this evaluation crucial for understanding their strengths.

Ultralytics YOLOv5 is known for its exceptional speed, scalability, and ease of deployment, making it a favorite for diverse applications. On the other hand, RTDETRv2 emphasizes accuracy and end-to-end efficiency, particularly in scenarios requiring detailed detection performance. For more details on YOLO models, visit the [Ultralytics YOLOv5 documentation](https://docs.ultralytics.com/models/yolov5/).


## mAP Comparison

This section highlights the mAP (Mean Average Precision) values of Ultralytics YOLOv5 and RT-DETRv2 models to evaluate their accuracy in object detection. mAP serves as a critical metric for understanding the effectiveness of these models across various sizes and configurations. Learn more about [mAP and its significance](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>RTDETRv2 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| s | 37.4 | 48.1 |
		| m | 45.4 | 51.9 |
		| l | 49.0 | 53.4 |
		| x | 50.7 | 54.3 |
		

## Speed Comparison

This section highlights the speed performance of Ultralytics YOLOv5 and RTDETRv2 models across different sizes. Speed metrics, measured in milliseconds, demonstrate the efficiency of each model, providing insights for real-time applications. Explore [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) and other benchmarks for detailed comparisons.


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>RTDETRv2 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| s | 1.92 | 5.03 |
		| m | 4.03 | 7.51 |
		| l | 6.61 | 9.76 |
		| x | 11.89 | 15.03 |

## Fine-Tuning with Car Parts Segmentation Dataset

Ultralytics YOLO11 offers incredible flexibility for custom applications, including fine-tuning on specialized datasets like the **Car Parts Segmentation Dataset**. This capability is especially beneficial for industries like automotive manufacturing, repair, and e-commerce, where identifying and categorizing vehicle components can optimize workflows.

The Car Parts Segmentation dataset allows YOLO11 to be trained for detailed segmentation tasks. By leveraging custom training, you can improve the model’s accuracy for recognizing specific car parts, enhancing its performance in real-world applications. Learn more about using YOLO11 for segmentation tasks on [Google Colab](https://www.ultralytics.com/blog/image-segmentation-with-ultralytics-yolo11-on-google-colab).

### Python Code Example for Custom Training with YOLO11:
```python
from ultralytics import YOLO

# Load YOLO11 model
model = YOLO("yolo11-segmentation.pt")

# Train the model on the Car Parts Segmentation dataset
model.train(data="carparts.yaml", epochs=50, imgsz=640)

# Evaluate the model's performance
metrics = model.val()
print(metrics)
```

For detailed guidance on custom training, visit the [Ultralytics Training Documentation](https://docs.ultralytics.com/modes/train/).
