---
comments: true
description: Explore the comprehensive comparison between Ultralytics YOLOv5 and PP-YOLOE+, two leading models in object detection and real-time AI. Dive into their performance, speed, and efficiency for applications in computer vision and edge AI.
keywords: Ultralytics, YOLOv5, PP-YOLOE+, object detection, real-time AI, edge AI, computer vision, model comparison
---

# Ultralytics YOLOv5 VS PP-YOLOE+

The comparison between Ultralytics YOLOv5 and PP-YOLOE+ highlights two leading-edge object detection models that have set benchmarks in speed, accuracy, and efficiency. Both models cater to diverse computer vision tasks, making this analysis crucial for those seeking optimal solutions for real-world applications.

Ultralytics YOLOv5 stands out with its streamlined architecture and robust deployment capabilities across platforms, including edge devices and cloud environments. In contrast, PP-YOLOE+ is recognized for its advanced design and impressive performance across challenging datasets, appealing to researchers and industry professionals alike. Explore how these models redefine possibilities in AI-driven object detection.

## mAP Comparison

This section highlights the mAP values of Ultralytics YOLOv5 and PP-YOLOE+, providing a detailed measure of their detection accuracy across multiple variants. mAP, or Mean Average Precision, evaluates the models' capability to balance precision and recall, making it a key metric in object detection performance. Learn more about [mAP in object detection](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>YOLOv5 | mAP<sup>val<br>50<br>PP-YOLOE+ |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 39.9 |
    	| s | 37.4 | 43.7 |
    	| m | 45.4 | 49.8 |
    	| l | 49.0 | 52.9 |
    	| x | 50.7 | 54.7 |


## Speed Comparison

This section highlights the performance of Ultralytics YOLOv5 and PP-YOLOE+ in terms of inference speed across various model sizes. Speed metrics, measured in milliseconds, showcase how these models balance efficiency and accuracy, providing insights into their deployment suitability. Learn more about [YOLOv5 architecture](https://docs.ultralytics.com/yolov5/tutorials/architecture_description/) and [PP-YOLOE+ details](https://github.com/PaddlePaddle/PaddleDetection).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLOv5 | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | N/A | 2.84 |
    	| s | 1.92 | 2.62 |
    	| m | 4.03 | 5.56 |
    	| l | 6.61 | 8.36 |
    	| x | 11.89 | 14.3 |

## Track Functionality in YOLO11

Ultralytics YOLO11 excels in real-time object tracking, providing robust solutions for applications like surveillance, sports analytics, and autonomous vehicles. The **Track** functionality enables seamless integration of object detection and tracking, ensuring efficient and accurate monitoring across video streams. With advanced algorithms, YOLO11 maintains high-speed performance and precision, making it suitable for dynamic environments.

To explore more about YOLO11's tracking capabilities, visit the [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/).

### Python Code: YOLO11 Tracking

```python
from ultralytics import YOLO

# Load YOLO11 model
model = YOLO('yolo11n.pt')

# Run object tracking on a video file
results = model.track(source='video.mp4', save=True, show=True)

# Display results
for frame in results:
    print(frame.boxes)
```

This code initiates video tracking using a YOLO11 model, saving and displaying results in real-time.
