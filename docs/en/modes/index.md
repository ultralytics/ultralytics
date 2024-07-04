---
comments: true
description: Discover the diverse modes of Ultralytics YOLOv8, including training, validation, prediction, export, tracking, and benchmarking. Maximize model performance and efficiency.
keywords: Ultralytics, YOLOv8, machine learning, model training, validation, prediction, export, tracking, benchmarking, object detection
---

# Ultralytics YOLOv8 Modes

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO ecosystem and integrations">

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

Benchmark mode is used to profile the speed and accuracy of various export formats for YOLOv8. The benchmarks provide information on the size of the exported format, its `mAP50-95` metrics (for object detection, segmentation and pose) or `accuracy_top5` metrics (for classification), and the inference time in milliseconds per image across various export formats like ONNX, OpenVINO, TensorRT and others. This information can help users choose the optimal export format for their specific use case based on their requirements for speed and accuracy.

[Benchmark Examples](benchmark.md){ .md-button }



## FAQ

### What are the different modes available in Ultralytics YOLOv8?

Ultralytics YOLOv8 offers diverse modes to cover the complete lifecycle of machine learning models:

- **[Train](train.md)**: Fine-tunes models on custom or preloaded datasets.
- **[Val](val.md)**: Validates model performance using a post-training checkpoint.
- **[Predict](predict.md)**: Uses trained models to make predictions on new images or videos.
- **[Export](export.md)**: Converts models to formats suitable for deployment.
- **[Track](track.md)**: Extends object detection into real-time tracking applications.
- **[Benchmark](benchmark.md)**: Analyzes model speed and accuracy across various deployment environments.

These modes ensure you can maximize model performance and efficiency for different tasks and use cases.

### How can I export my YOLOv8 model for deployment?

Ultralytics YOLOv8's **[Export](../modes/export.md)** mode makes your model deployment-ready by converting it into various formats like ONNX, TensorRT, CoreML, and more. This ensures compatibility with different software applications and hardware devices for seamless production deployment. Simply follow the steps outlined in the [Export Examples](../modes/export.md) to perform the conversion.

### How can I use YOLOv8 for real-time object tracking?

Ultralytics YOLOv8 supports **[Track](../modes/track.md)** mode, enabling real-time object tracking using a loaded model. This mode is particularly useful for applications such as surveillance systems or self-driving cars. By providing a live video stream, the model can track and identify objects in real-time. Detailed examples can be found in the [Track Examples](../modes/track.md).

### Why should I use the Benchmark mode in YOLOv8?

The **[Benchmark](../modes/benchmark.md)** mode is crucial for profiling the speed and accuracy of YOLOv8 models across various export formats. It helps you understand the performance metrics like `mAP50-95` for object detection and `accuracy_top5` for classification, helping you choose the best format for your deployment needs. This ensures optimal performance in your specific use case.

### What is the purpose of the Validate mode in YOLOv8?

**[Validate](../modes/val.md)** mode in YOLOv8 helps measure the model's accuracy and generalization performance after training. It evaluates the model using a validation set and provides insights into how well the model performs on unseen data, allowing for hyperparameter tuning to enhance performance. Explore more through the detailed [Val Examples](../modes/val.md).
