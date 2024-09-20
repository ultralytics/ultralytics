---
comments: true
description: Discover the diverse modes of Ultralytics YOLOv8, including training, validation, prediction, export, tracking, and benchmarking. Maximize model performance and efficiency.
keywords: Ultralytics, YOLOv8, machine learning, model training, validation, prediction, export, tracking, benchmarking, object detection
---

# Ultralytics YOLOv8 Modes

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Ultralytics YOLOv8 is not just another object detection model; it's a versatile framework designed to cover the entire lifecycle of machine learning models—from data ingestion and model training to validation, deployment, and real-world tracking. Each mode serves a specific purpose and is engineered to offer you the flexibility and efficiency required for different tasks and use-cases.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics Modes Tutorial: Train, Validate, Predict, Export & Benchmark.
</p>

### Modes at a Glance

Understanding the different **modes** that Ultralytics YOLOv8 supports is critical to getting the most out of your models:

- **Train** mode: Fine-tune your model on custom or preloaded datasets.
- **Val** mode: A post-training checkpoint to validate model performance.
- **Predict** mode: Unleash the predictive power of your model on real-world data.
- **Export** mode: Make your model deployment-ready in various formats.
- **Track** mode: Extend your object detection model into real-time tracking applications.
- **Benchmark** mode: Analyze the speed and accuracy of your model in diverse deployment environments.

This comprehensive guide aims to give you an overview and practical insights into each mode, helping you harness the full potential of YOLOv8.

## [Train](train.md)

Train mode is used for training a YOLOv8 model on a custom dataset. In this mode, the model is trained using the specified dataset and hyperparameters. The training process involves optimizing the model's parameters so that it can accurately predict the classes and locations of objects in an image.

[Train Examples](train.md){ .md-button }

## [Val](val.md)

Val mode is used for validating a YOLOv8 model after it has been trained. In this mode, the model is evaluated on a validation set to measure its accuracy and generalization performance. This mode can be used to tune the hyperparameters of the model to improve its performance.

[Val Examples](val.md){ .md-button }

## [Predict](predict.md)

Predict mode is used for making predictions using a trained YOLOv8 model on new images or videos. In this mode, the model is loaded from a checkpoint file, and the user can provide images or videos to perform inference. The model predicts the classes and locations of objects in the input images or videos.

[Predict Examples](predict.md){ .md-button }

## [Export](export.md)

Export mode is used for exporting a YOLOv8 model to a format that can be used for deployment. In this mode, the model is converted to a format that can be used by other software applications or hardware devices. This mode is useful when deploying the model to production environments.

[Export Examples](export.md){ .md-button }

## [Track](track.md)

Track mode is used for tracking objects in real-time using a YOLOv8 model. In this mode, the model is loaded from a checkpoint file, and the user can provide a live video stream to perform real-time object tracking. This mode is useful for applications such as surveillance systems or self-driving cars.

[Track Examples](track.md){ .md-button }

## [Benchmark](benchmark.md)

Benchmark mode is used to profile the speed and accuracy of various export formats for YOLOv8. The benchmarks provide information on the size of the exported format, its `mAP50-95` metrics (for object detection, segmentation, and pose) or `accuracy_top5` metrics (for classification), and the inference time in milliseconds per image across various formats like ONNX, OpenVINO, TensorRT, and others. This information can help users choose the optimal export format for their specific use case based on their requirements for speed and accuracy.

[Benchmark Examples](benchmark.md){ .md-button }

## FAQ

### How do I train a custom object detection model with Ultralytics YOLOv8?

Training a custom object detection model with Ultralytics YOLOv8 involves using the train mode. You need a dataset formatted in YOLO format, containing images and corresponding annotation files. Use the following command to start the training process:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Train a custom model
        model = YOLO("yolov8n.pt")
        model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo train data=path/to/dataset.yaml epochs=100 imgsz=640
        ```

For more detailed instructions, you can refer to the [Ultralytics Train Guide](../modes/train.md).

### What metrics does Ultralytics YOLOv8 use to validate the model's performance?

Ultralytics YOLOv8 uses various metrics during the validation process to assess model performance. These include:

- **mAP (mean Average Precision)**: This evaluates the accuracy of object detection.
- **IOU (Intersection over Union)**: Measures the overlap between predicted and ground truth bounding boxes.
- **Precision and Recall**: Precision measures the ratio of true positive detections to the total detected positives, while recall measures the ratio of true positive detections to the total actual positives.

You can run the following command to start the validation:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Validate the model
        model = YOLO("yolov8n.pt")
        model.val(data="path/to/validation.yaml")
        ```

    === "CLI"

        ```bash
        yolo val data=path/to/validation.yaml
        ```

Refer to the [Validation Guide](../modes/val.md) for further details.

### How can I export my YOLOv8 model for deployment?

Ultralytics YOLOv8 offers export functionality to convert your trained model into various deployment formats such as ONNX, TensorRT, CoreML, and more. Use the following example to export your model:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Export the model
        model = YOLO("yolov8n.pt")
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolov8n.pt format=onnx
        ```

Detailed steps for each export format can be found in the [Export Guide](../modes/export.md).

### What is the purpose of the benchmark mode in Ultralytics YOLOv8?

Benchmark mode in Ultralytics YOLOv8 is used to analyze the speed and accuracy of various export formats such as ONNX, TensorRT, and OpenVINO. It provides metrics like model size, `mAP50-95` for object detection, and inference time across different hardware setups, helping you choose the most suitable format for your deployment needs.

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark on GPU
        benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

    === "CLI"

        ```bash
        yolo benchmark model=yolov8n.pt data='coco8.yaml' imgsz=640 half=False device=0
        ```

For more details, refer to the [Benchmark Guide](../modes/benchmark.md).

### How can I perform real-time object tracking using Ultralytics YOLOv8?

Real-time object tracking can be achieved using the track mode in Ultralytics YOLOv8. This mode extends object detection capabilities to track objects across video frames or live feeds. Use the following example to enable tracking:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Track objects in a video
        model = YOLO("yolov8n.pt")
        model.track(source="path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        yolo track source=path/to/video.mp4
        ```

For in-depth instructions, visit the [Track Guide](../modes/track.md).
