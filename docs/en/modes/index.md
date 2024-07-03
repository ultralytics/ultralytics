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

### What are the different modes available in Ultralytics YOLOv8 and their purposes?
Ultralytics YOLOv8 supports several modes that cater to different aspects of machine learning model lifecycle:
- **Train**: Fine-tune your model on custom or preloaded datasets. [Learn more](../modes/train.md)
- **Val**: Validate model performance after training using a validation set. [Explore Val mode](../modes/val.md)
- **Predict**: Use a trained model to make predictions on new images or videos. [Discover Predict mode](../modes/predict.md)
- **Export**: Convert your model for deployment in various formats. [Read about Export mode](../modes/export.md)
- **Track**: Extend object detection models to real-time tracking applications. [Understand Track mode](../modes/track.md)
- **Benchmark**: Analyze the speed and accuracy of your model in different deployment environments. [Learn about Benchmark mode](../modes/benchmark.md)

### How can I train my custom YOLOv8 model using Train Mode?
Training a custom YOLOv8 model involves using your dataset and specifying hyperparameters. The YOLOv8 Train Mode optimizes the model to accurately predict object classes and locations. Practical examples and detailed steps can be found in our [Train Mode documentation](../modes/train.md).

### Why should I use the Export Mode in YOLOv8?
Export Mode in YOLOv8 allows you to convert trained models into formats suitable for deployment across different environments, such as mobile devices or other software applications. This capability is crucial for integrating AI models into real-world applications. Find detailed instructions in the [Export Mode documentation](../modes/export.md).

### What is the purpose of Benchmark Mode in YOLOv8?
Benchmark Mode is designed to evaluate the performance of YOLOv8 models by analyzing speed, accuracy, and resource utilization across various export formats like ONNX, TensorRT, and OpenVINO. This helps users choose the optimal format for their specific use case. Learn more about benchmarking in the [Benchmark Mode documentation](../modes/benchmark.md).

### Can YOLOv8 models be used for real-time object tracking?
Yes, YOLOv8 models can be extended for real-time object tracking using the Track Mode. This is ideal for applications like surveillance systems and autonomous vehicles. The model loads from a checkpoint file and can perform tracking on live video streams. Detailed usage examples are available in the [Track Mode documentation](../modes/track.md).