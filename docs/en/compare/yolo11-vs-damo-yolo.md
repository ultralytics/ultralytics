---
comments: true
description: Compare Ultralytics YOLO11 and DAMO-YOLO to discover how these cutting-edge models perform in object detection, real-time AI, and edge AI applications. Dive into their accuracy, speed, and efficiency to see which one excels in computer vision tasks.
keywords: Ultralytics YOLO11, DAMO-YOLO, object detection, real-time AI, edge AI, computer vision, model comparison, AI performance, Ultralytics
---

# Ultralytics YOLO11 VS DAMO-YOLO

Ultralytics YOLO11 and DAMO-YOLO represent cutting-edge advancements in computer vision, each offering unique capabilities tailored for real-time applications. This comparison delves into their performance, efficiency, and adaptability, helping you choose the best model for your AI projects.

Ultralytics YOLO11 is celebrated for its enhanced accuracy, speed, and flexibility, making it ideal for diverse tasks like object detection and pose estimation. On the other hand, DAMO-YOLO brings its own innovations, excelling in resource efficiency and deployment versatility. Explore their strengths to find the perfect match for your needs. Learn more about YOLO11’s key features in [Ultralytics' documentation](https://docs.ultralytics.com/models/yolo11/) and the evolution of YOLO models [here](https://www.ultralytics.com/blog/the-evolution-of-object-detection-and-ultralytics-yolo-models).

## mAP Comparison

This section highlights the mAP performance of Ultralytics YOLO11 versus DAMO-YOLO, showcasing their ability to accurately detect and localize objects across different model variants. mAP, or Mean Average Precision, is a critical metric for evaluating the precision and recall balance of these advanced object detection systems. Learn more about [mAP here](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLO11 | mAP<sup>val<br>50<br>DAMO-YOLO |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.5 | 42.0 |
    	| s | 47.0 | 46.0 |
    	| m | 51.4 | 49.2 |
    	| l | 53.2 | 50.8 |
    	| x | 54.7 | N/A |

## Speed Comparison

Explore the performance of Ultralytics YOLO11 versus DAMO-YOLO, with speed metrics in milliseconds highlighting their efficiency across model sizes. These benchmarks emphasize real-time capability, ideal for applications requiring rapid response times. Learn more about [Ultralytics YOLO11's capabilities](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>DAMO-YOLO |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 1.55 | 2.32 |
    	| s | 2.63 | 3.45 |
    	| m | 5.27 | 5.09 |
    	| l | 6.84 | 7.18 |
    	| x | 12.49 | N/A |

## Fine-Tuning With African Wildlife Dataset

Ultralytics YOLO11 supports fine-tuning with diverse datasets, including the African Wildlife dataset. This dataset is instrumental in training models for wildlife conservation, biodiversity monitoring, and anti-poaching efforts. By leveraging YOLO11's advanced capabilities, users can efficiently detect and classify animals in real-time, even in challenging environments like dense forests or open savannahs.

Fine-tuning allows you to adapt YOLO11's pretrained weights to specific tasks, ensuring higher accuracy and relevance for your project. The flexibility of YOLO11 makes it an excellent choice for ecological and environmental applications.

To learn more and access the African Wildlife dataset, explore [this guide](https://docs.ultralytics.com/datasets/).

### Python Code Example

```python
from ultralytics import YOLO

# Load a pretrained YOLO11 model
model = YOLO("yolo11.pt")

# Train the model on the African Wildlife dataset
model.train(data="african_wildlife.yaml", epochs=50, imgsz=640, batch=16)

# Validate the model
metrics = model.val()

# Save the fine-tuned model
model.save("yolo11_african_wildlife.pt")
```
