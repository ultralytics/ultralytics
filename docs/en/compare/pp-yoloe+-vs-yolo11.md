---
comments: true
description: Compare PP-YOLOE+ and Ultralytics YOLO11 to discover which model excels in object detection, real-time AI, and edge AI applications. Explore their performance, accuracy, and efficiency for cutting-edge computer vision tasks.
keywords: PP-YOLOE+, Ultralytics YOLO11, object detection, real-time AI, edge AI, computer vision, model comparison, AI performance, YOLO series, Ultralytics models
---

# PP-YOLOE+ VS Ultralytics YOLO11

The comparison between PP-YOLOE+ and Ultralytics YOLO11 showcases two cutting-edge models in the field of object detection. Both models are designed to push the boundaries of speed and precision, making them essential tools for real-time computer vision applications.

Ultralytics YOLO11 introduces significant advancements in efficiency and accuracy, leveraging an optimized architecture and proprietary datasets. Meanwhile, PP-YOLOE+ emphasizes its high-performance capabilities with a robust feature extraction pipeline. Learn more about [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) and its innovative features or explore the evolution of object detection through [PP-YOLOE+ insights](https://github.com/PaddlePaddle/PaddleDetection).

## mAP Comparison

This section compares the mAP scores of PP-YOLOE+ and Ultralytics YOLO11, showcasing their accuracy across different variants. mAP, a vital metric in object detection, evaluates how well these models balance precision and recall. Learn more about [mAP and its significance](https://www.ultralytics.com/glossary/mean-average-precision-map).

!!! tip "Accuracy"

    === "Detection (COCO)"

    	| Variant | mAP<sup>val<br>50<br>PP-YOLOE+ | mAP<sup>val<br>50<br>YOLO11 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 39.9 | 39.5 |
    	| s | 43.7 | 47.0 |
    	| m | 49.8 | 51.4 |
    	| l | 52.9 | 53.2 |
    	| x | 54.7 | 54.7 |

## Speed Comparison

This section highlights the speed metrics of PP-YOLOE+ and Ultralytics YOLO11 across various model sizes, measured in milliseconds. These benchmarks emphasize the efficiency of each model, with YOLO11 demonstrating optimized inference times for real-time applications. For detailed specifications, explore the [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/) or the [PP-YOLOE+ repository](https://github.com/PaddlePaddle/PaddleDetection).

!!! tip "Speed"

    === "Detection (COCO)"

    	| Variant | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>PP-YOLOE+ | Speed<br><sup>T4 TensorRT10<br>(ms)</sup><br>YOLO11 |
    	|---------------------|-------------------------------------------------------|-------------------------------------------------------|
    	| n | 2.84 | 1.55 |
    	| s | 2.62 | 2.63 |
    	| m | 5.56 | 5.27 |
    	| l | 8.36 | 6.84 |
    	| x | 14.3 | 12.49 |

## YOLO Performance Metrics

Ultralytics YOLO11 delivers powerful capabilities for object detection, segmentation, and more, but evaluating its performance effectively is key to optimizing your models. Metrics such as mAP (mean Average Precision), IoU (Intersection over Union), and F1 score are essential for understanding the model's accuracy and efficiency. These metrics help you assess how well your model is identifying and localizing objects in your dataset.

<<<<<<< HEAD
To dive deeper into these metrics and learn how to interpret them, refer to the [YOLO Performance Metrics Guide](https://docs.ultralytics.com/guides/yolo-performance-metrics/). This guide also provides practical examples on how to improve your model's detection accuracy and speed, ensuring you achieve the best results for your specific application. Whether you're working on real-time object detection or large-scale dataset analysis, understanding these metrics is crucial for success.
=======
To learn more about how to utilize YOLO11's prediction capabilities, explore the [Comprehensive Tutorials to Ultralytics YOLO](https://docs.ultralytics.com/guides/).

### Example Code: Using Predict in Ultralytics YOLO11

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO("yolo11.pt")

# Perform predictions on an image
results = model.predict(source="image.jpg", save=True)

# Display results
results.show()
```

This snippet demonstrates how to quickly generate predictions and visualize the output, showcasing YOLO11's efficiency and ease of use.

> > > > > > > 95d73b193a43ffc9b65d81b5b93f6d2acf3cb195
