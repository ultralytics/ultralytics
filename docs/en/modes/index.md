---
comments: true
description: Ultralytics YOLO26 modes for training, validation, prediction, export, tracking, and benchmarking, with what each mode does and Python and CLI examples.
keywords: Ultralytics, YOLO26, machine learning, model training, validation, prediction, export, tracking, benchmarking, object detection
---

# Ultralytics YOLO26 Modes

<img width="1024" src="https://cdn.ul.run/i/f874ab850f33f361d01a01e9a8c98655.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Ultralytics YOLO26 is more than an [object detection](https://www.ultralytics.com/glossary/object-detection) model: the Ultralytics framework covers the [full lifecycle](https://www.ultralytics.com/blog/how-to-train-validate-predict-export-benchmark-with-ultralytics-yolo-models) of [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models, from training and validation to prediction, export, tracking, and benchmarking. Each mode serves a specific purpose in that workflow.

For planned YOLO27 support, see the [YOLO27 preview](../models/yolo27.md#usage-examples). YOLO27 is coming soon and is not yet available; this guide uses released YOLO26 models.

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

Ultralytics YOLO26 supports the following **modes**:

- **Train** mode: Train or fine-tune your model on custom or built-in datasets.
- **Val** mode: Validate model performance after training.
- **Predict** mode: Run inference with your model on new images, videos, and streams.
- **Export** mode: Convert your model to formats ready for [model deployment](https://www.ultralytics.com/glossary/model-deployment).
- **Track** mode: Track objects across video frames and live streams in real time.
- **Benchmark** mode: Analyze the speed and accuracy of your model in diverse deployment environments.

The sections below summarize each mode and link to its full guide.

## Train

Train mode is used for [training a YOLO26 model](https://academy.ultralytics.com/courses/train-your-first-yolo) on a custom dataset. In this mode, the model is trained using the specified dataset and hyperparameters. The training process involves optimizing the model's parameters so that it can accurately predict the classes and locations of objects in an image. Training is essential for creating models that can recognize specific objects relevant to your application.

[Train Examples](train.md){ .md-button }

## Val

Val mode is used for validating a YOLO26 model after it has been trained. In this mode, the model is evaluated on a validation set to measure its accuracy and generalization performance. Validation helps identify potential issues like [overfitting](https://www.ultralytics.com/glossary/overfitting) and provides metrics such as [mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) to quantify model performance. Validation results guide [hyperparameter tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning) and other model improvements.

[Val Examples](val.md){ .md-button }

## Predict

Predict mode is used for making predictions using a trained YOLO26 model on new images or videos. In this mode, the model is loaded from a checkpoint file, and the user can provide images or videos to perform inference. The model identifies and localizes objects in the input media. Predict mode is how you apply your trained model to practical problems.

[Predict Examples](predict.md){ .md-button }

## Export

Export mode is used for converting a YOLO26 model to formats suitable for deployment across different platforms and devices. This mode transforms your PyTorch model into optimized formats like ONNX, TensorRT, or CoreML, enabling deployment in production environments. Exporting is essential for integrating your model with various software applications or hardware devices, often resulting in significant performance improvements.

[Export Examples](export.md){ .md-button }

## Track

Track mode extends YOLO26's object detection capabilities to [track objects](https://www.ultralytics.com/glossary/object-tracking) across video frames or live streams. This mode is particularly valuable for applications requiring persistent object identification, such as [surveillance systems](https://www.ultralytics.com/blog/shattering-the-surveillance-status-quo-with-vision-ai) or [self-driving cars](https://www.ultralytics.com/solutions/computer-vision-in-automotive). It includes six trackers: TrackTrack (default), BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT, and FastTracker.

[Track Examples](track.md){ .md-button }

## Benchmark

Benchmark mode profiles the speed and accuracy of various export formats for YOLO26. This mode provides comprehensive metrics on model size, accuracy (mAP<sub>50-95</sub> for detection, segmentation, pose, and OBB; mIoU for semantic segmentation; δ<sub>1</sub> for depth; or accuracy_top1 for classification), and inference time across different formats like ONNX, [OpenVINO](../integrations/openvino.md), and TensorRT. Benchmarking helps you select the optimal export format based on your specific requirements for speed and accuracy in your deployment environment.

[Benchmark Examples](benchmark.md){ .md-button }

## FAQ

### How do I train a custom object detection model with Ultralytics YOLO26?

Training a custom object detection model with Ultralytics YOLO26 involves using the train mode. You need a dataset formatted in [YOLO format](../datasets/detect/index.md#ultralytics-yolo-format), containing images and corresponding annotation files. Use the following command to start the training process:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO model (you can choose n, s, m, l, or x versions)
        model = YOLO("yolo26n.pt")

        # Start training on your custom dataset
        model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train a YOLO model from the command line
        yolo detect train data=path/to/dataset.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For more detailed instructions, you can refer to the [Ultralytics Train Guide](../modes/train.md).

### What metrics does Ultralytics YOLO26 use to validate the model's performance?

Ultralytics YOLO26 uses various metrics during the validation process to assess model performance. These include:

- **mAP (mean Average Precision)**: This evaluates the accuracy of object detection.
- **[IoU (Intersection over Union)](https://www.ultralytics.com/glossary/intersection-over-union-iou)**: Measures the overlap between predicted and ground truth bounding boxes.
- **[Precision](https://www.ultralytics.com/glossary/precision) and [Recall](https://www.ultralytics.com/glossary/recall)**: Precision measures the ratio of true positive detections to the total detected positives, while recall measures the ratio of true positive detections to the total actual positives.

You can run the following command to start the validation:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained or custom YOLO model
        model = YOLO("yolo26n.pt")

        # Run validation on your dataset
        model.val(data="path/to/validation.yaml")
        ```

    === "CLI"

        ```bash
        # Validate a YOLO model from the command line
        yolo val model=yolo26n.pt data=path/to/validation.yaml
        ```

Refer to the [Validation Guide](../modes/val.md) for further details.

### How can I export my YOLO26 model for deployment?

Ultralytics YOLO26 offers export functionality to convert your trained model into various deployment formats such as ONNX, TensorRT, CoreML, and more. Use the following example to export your model:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load your trained YOLO model
        model = YOLO("yolo26n.pt")

        # Export the model to ONNX format (you can specify other formats as needed)
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        # Export a YOLO model to ONNX format from the command line
        yolo export model=yolo26n.pt format=onnx
        ```

Detailed steps for each export format can be found in the [Export Guide](../modes/export.md).

### What is the purpose of the benchmark mode in Ultralytics YOLO26?

Benchmark mode in Ultralytics YOLO26 is used to analyze the speed and [accuracy](https://www.ultralytics.com/glossary/accuracy) of various export formats such as ONNX, TensorRT, and OpenVINO. It provides metrics like model size, `mAP50-95` for object detection, and inference time across different hardware setups, helping you choose the most suitable format for your deployment needs.

!!! example

    === "Python"

        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Run benchmark on GPU (device 0)
        # You can adjust parameters like model, dataset, image size, and precision as needed
        benchmark(model="yolo26n.pt", data="coco8.yaml", imgsz=640, device=0)
        ```

    === "CLI"

        ```bash
        # Benchmark a YOLO model from the command line
        # Adjust parameters as needed for your specific use case
        yolo benchmark model=yolo26n.pt data='coco8.yaml' imgsz=640 device=0
        ```

For more details, refer to the [Benchmark Guide](../modes/benchmark.md).

### How can I perform real-time object tracking using Ultralytics YOLO26?

Real-time object tracking can be achieved using the track mode in Ultralytics YOLO26. This mode extends object detection capabilities to track objects across video frames or live feeds. Use the following example to enable tracking:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO model
        model = YOLO("yolo26n.pt")

        # Start tracking objects in a video
        # You can also use live video streams or webcam input
        model.track(source="path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # Perform object tracking on a video from the command line
        # You can specify different sources like webcam (0) or RTSP streams
        yolo track model=yolo26n.pt source=path/to/video.mp4
        ```

For in-depth instructions, visit the [Track Guide](../modes/track.md).
