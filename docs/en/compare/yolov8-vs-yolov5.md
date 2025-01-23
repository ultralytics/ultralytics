---
comments: true
description: Compare Ultralytics YOLOv8 and YOLOv5 to discover advancements in real-time object detection, speed, and accuracy. Explore how these state-of-the-art models revolutionize computer vision and edge AI applications, providing seamless integration and robust performance for diverse industries.  
keywords: Ultralytics, YOLOv8, YOLOv5, object detection, real-time AI, edge AI, computer vision, AI models, model comparison
---

# Ultralytics YOLOv8 VS Ultralytics YOLOv5

Ultralytics YOLOv8 and YOLOv5 represent significant milestones in the evolution of object detection models. This comparison highlights the advancements each model brings to real-time object detection, segmentation, and classification tasks.

While YOLOv5 gained widespread popularity for its speed and ease of use, YOLOv8 pushes the boundaries with enhanced accuracy, simplified workflows, and compatibility with all YOLO versions. Explore their unique strengths to determine the best fit for your AI projects. Learn more about [YOLOv8's release](https://www.ultralytics.com/blog/introducing-ultralytics-yolov8) and its key features.


## mAP Comparison

This section evaluates the mAP values of Ultralytics YOLOv8 and YOLOv5, providing insights into their accuracy across different variants. Mean Average Precision (mAP) is a key metric for assessing object detection performance, balancing precision and recall to reflect model effectiveness. Learn more about [mAP in object detection](https://www.ultralytics.com/glossary/mean-average-precision-map).


!!! tip "Accuracy"

	=== "Detection (COCO)"

		| Variant | mAP<sup>val<br>50<br>YOLOv8 | mAP<sup>val<br>50<br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 37.3 | N/A |
		| s | 44.9 | 37.4 |
		| m | 50.2 | 45.4 |
		| l | 52.9 | 49.0 |
		| x | 53.9 | 50.7 |
		

## Speed Comparison

Compare the performance of Ultralytics YOLOv8 and YOLOv5 across various model sizes, focusing on speed metrics in milliseconds. These measurements highlight the advancements in real-time object detection capabilities, as seen in Ultralytics' [YOLO documentation](https://docs.ultralytics.com/models/yolov8/).


!!! tip "Speed"

	=== "Detection (COCO)"

		| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv8 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 |
		|---------------------|-------------------------------------------------------|-------------------------------------------------------|
		| n | 1.47 | N/A |
		| s | 2.66 | 1.92 |
		| m | 5.86 | 4.03 |
		| l | 9.06 | 6.61 |
		| x | 14.37 | 11.89 |

## YOLO Thread-Safe Inference

Thread-safe inference is crucial when deploying YOLO models in multi-threaded environments to ensure consistent and reliable predictions. Ultralytics YOLO11 simplifies this process with best practices and tools designed to prevent race conditions during execution.

By enabling thread-safe operations, YOLO11 facilitates seamless integration into production systems requiring parallel processing, such as video analytics and real-time monitoring. Developers can confidently scale their applications across multiple threads without compromising accuracy or performance.

Learn more about implementing thread-safe inference in your projects by exploring the [YOLO Thread-Safe Inference guide](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/). This guide provides practical examples and tips on achieving robust and efficient deployments.

For Python users, the following snippet demonstrates how to use the `predict()` method in a thread-safe manner:

```python
from ultralytics import YOLO
import threading

# Load the YOLO11 model
model = YOLO("yolo11.pt")

# Define a thread-safe prediction function
def thread_safe_predict(image_path):
    results = model.predict(source=image_path)
    print(results)

# Run predictions in multiple threads
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
threads = [threading.Thread(target=thread_safe_predict, args=(path,)) for path in image_paths]

for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
```

This approach ensures consistent results even when handling multiple inputs concurrently.
