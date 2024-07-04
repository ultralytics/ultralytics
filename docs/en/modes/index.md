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

### What is the purpose of the different modes in Ultralytics YOLOv8?

Ultralytics YOLOv8 offers various modes to streamline the entire model lifecycle, including **Train**, **Val**, **Predict**, **Export**, **Track**, and **Benchmark**. Each mode serves a unique purpose:
- **[Train](train.md)**: Fine-tuning your model on custom or preloaded datasets.
- **[Val](val.md)**: Validating model performance post-training.
- **[Predict](predict.md)**: Performing inference on new data.
- **[Export](export.md)**: Making models deployment-ready in different formats.
- **[Track](track.md)**: Extending the object detection model into real-time tracking.
- **[Benchmark](benchmark.md)**: Evaluating the speed and accuracy of the model in various environments.

### How do I use the Train mode in Ultralytics YOLOv8 to fine-tune my model?

The **[Train](train.md)** mode in Ultralytics YOLOv8 is designed for training your model using a specific dataset and hyperparameters. This involves optimizing the model's parameters to accurately predict object classes and locations in images. You start the training via the command line or a Python script, specifying the dataset path, configuration, and hyperparameters. For a hands-on example, visit our [Train Examples](train.md).

### Why is validation necessary after training a YOLOv8 model?

Validation, facilitated by the **[Val](val.md)** mode, is crucial for measuring how well your model generalizes to new, unseen data. This mode evaluates the model on a validation set to provide metrics like accuracy and loss, which helps in tuning the hyperparameters for better performance. Regular validation ensures that your model is not overfitting and maintains high accuracy across different datasets. For detailed steps, check our [Val Examples](val.md).

### What formats can I export my YOLOv8 model to using Export mode?

In the **[Export](export.md)** mode, Ultralytics YOLOv8 allows you to convert your trained model into various deployment-ready formats such as ONNX, TensorRT, CoreML, and more. This makes it easier to integrate the model into different applications and hardware. The Export mode ensures that the deployment process is seamless and efficient. For more information on exporting, view our [Export Examples](export.md).

### How does the Track mode in YOLOv8 aid in real-time object tracking?

The **[Track](track.md)** mode extends the capabilities of YOLOv8 by providing real-time object tracking. This mode loads the model from a checkpoint file and uses it to track objects in a live video stream. It is particularly useful for applications like surveillance systems, self-driving cars, and more. For implementation details and examples, see our [Track Examples](track.md).