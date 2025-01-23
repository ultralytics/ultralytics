---
comments: true
description: Discover the key differences between Ultralytics YOLO11 and YOLOX in this comprehensive comparison. Learn how these cutting-edge models perform in object detection, real-time AI, and edge AI applications, and explore their efficiency in various computer vision tasks.
keywords: Ultralytics, YOLO11, YOLOX, object detection, real-time AI, edge AI, computer vision, model comparison
---

# Ultralytics YOLO11 VS YOLOX

The comparison between Ultralytics YOLO11 and YOLOX highlights the evolution of object detection models, showcasing advancements in accuracy, speed, and computational efficiency. These two cutting-edge models are pivotal in shaping the future of computer vision, catering to diverse real-time applications across industries.

Ultralytics YOLO11 builds on its legacy with enhanced feature extraction, optimized training pipelines, and remarkable efficiency, making it a top choice for challenging tasks. On the other hand, YOLOX offers innovative decoupled head architecture and strong performance benchmarks, delivering robust results for high-demand scenarios. For a deeper dive into YOLO11's features, explore the [official documentation](https://docs.ultralytics.com/models/yolo11/).

## mAP Comparison

This section evaluates the mAP performance of Ultralytics YOLO11 versus YOLOX, showcasing how well each model balances precision and recall across all object classes. For more on the significance of mAP in model evaluation, explore [Mean Average Precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLO11 | mAP<sup>val<br>50<br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.5 | N/A |
    	| s | 47.0 | 40.5 |
    	| m | 51.4 | 46.9 |
    	| l | 53.2 | 49.7 |
    	| x | 54.7 | 51.1 |


## Speed Comparison

This section highlights the speed performance of Ultralytics YOLO11 versus YOLOX, measured in milliseconds across various model sizes. Leveraging advancements like TensorRT integration, Ultralytics YOLO11 exhibits faster inference times, making it ideal for real-time applications on platforms such as [edge devices](https://docs.ultralytics.com/guides/model-deployment-options/) or [NVIDIA GPUs](https://docs.ultralytics.com/guides/triton-inference-server/).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOX |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.55 | N/A |
    	| s | 2.63 | 2.56 |
    	| m | 5.27 | 5.43 |
    	| l | 6.84 | 9.04 |
    	| x | 12.49 | 16.1 |

## Leveraging YOLO11 for Custom Training

Ultralytics YOLO11 provides an exceptional **Train** functionality, enabling you to fine-tune the model on custom datasets. Whether you're working on datasets like COCO8 or specific ones such as car parts segmentation or tiger pose, YOLO11's training workflow is seamless and efficient. By leveraging pre-trained weights and fine-tuning, you can achieve high accuracy tailored to your unique use cases.

Explore the [custom training documentation](https://docs.ultralytics.com/modes/train/) for step-by-step guidance on initializing your dataset and optimizing your model for better performance.

### Python Code Example

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Train the model on your custom dataset
model.train(data="custom_dataset.yaml", epochs=50, imgsz=640, batch=16)

# Validate and save the trained model
results = model.val()
model.save("custom_model.pt")
```

This functionality is ideal for applications requiring specialized datasets, offering flexibility and precision. Learn more about dataset preparation and training workflows in [Ultralytics guides](https://docs.ultralytics.com/guides/).
