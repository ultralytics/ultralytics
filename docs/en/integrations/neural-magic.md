---
comments: true
description: Enhance YOLOv8 performance using Neural Magic's DeepSparse Engine. Learn how to deploy and benchmark YOLOv8 models on CPUs for efficient object detection.
keywords: YOLOv8, DeepSparse, Neural Magic, model optimization, object detection, inference speed, CPU performance, sparsity, pruning, quantization
---

# Optimizing YOLOv8 Inferences with Neural Magic's DeepSparse Engine

When deploying [object detection](https://www.ultralytics.com/glossary/object-detection) models like [Ultralytics YOLOv8](https://www.ultralytics.com/) on various hardware, you can bump into unique issues like optimization. This is where YOLOv8's integration with Neural Magic's DeepSparse Engine steps in. It transforms the way YOLOv8 models are executed and enables GPU-level performance directly on CPUs.

This guide shows you how to deploy YOLOv8 using Neural Magic's DeepSparse, how to run inferences, and also how to benchmark performance to ensure it is optimized.

## Neural Magic's DeepSparse

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/neural-magic-deepsparse-overview.avif" alt="Neural Magic's DeepSparse Overview">
</p>

[Neural Magic's DeepSparse](https://neuralmagic.com/deepsparse/) is an inference run-time designed to optimize the execution of neural networks on CPUs. It applies advanced techniques like sparsity, pruning, and quantization to dramatically reduce computational demands while maintaining accuracy. DeepSparse offers an agile solution for efficient and scalable [neural network](https://www.ultralytics.com/glossary/neural-network-nn) execution across various devices.

## Benefits of Integrating Neural Magic's DeepSparse with YOLOv8

Before diving into how to deploy YOLOV8 using DeepSparse, let's understand the benefits of using DeepSparse. Some key advantages include:

- **Enhanced Inference Speed**: Achieves up to 525 FPS (on YOLOv8n), significantly speeding up YOLOv8's inference capabilities compared to traditional methods.

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/enhanced-inference-speed.avif" alt="Enhanced Inference Speed">
</p>

- **Optimized Model Efficiency**: Uses pruning and quantization to enhance YOLOv8's efficiency, reducing model size and computational requirements while maintaining [accuracy](https://www.ultralytics.com/glossary/accuracy).

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/optimized-model-efficiency.avif" alt="Optimized Model Efficiency">
</p>

- **High Performance on Standard CPUs**: Delivers GPU-like performance on CPUs, providing a more accessible and cost-effective option for various applications.

- **Streamlined Integration and Deployment**: Offers user-friendly tools for easy integration of YOLOv8 into applications, including image and video annotation features.

- **Support for Various Model Types**: Compatible with both standard and sparsity-optimized YOLOv8 models, adding deployment flexibility.

- **Cost-Effective and Scalable Solution**: Reduces operational expenses and offers scalable deployment of advanced object detection models.

## How Does Neural Magic's DeepSparse Technology Works?

Neural Magic's Deep Sparse technology is inspired by the human brain's efficiency in neural network computation. It adopts two key principles from the brain as follows:

- **Sparsity**: The process of sparsification involves pruning redundant information from [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) networks, leading to smaller and faster models without compromising accuracy. This technique reduces the network's size and computational needs significantly.

- **Locality of Reference**: DeepSparse uses a unique execution method, breaking the network into Tensor Columns. These columns are executed depth-wise, fitting entirely within the CPU's cache. This approach mimics the brain's efficiency, minimizing data movement and maximizing the CPU's cache use.

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/neural-magic-deepsparse-technology.avif" alt="How Neural Magic's DeepSparse Technology Works ">
</p>

For more details on how Neural Magic's DeepSparse technology work, check out [their blog post](https://neuralmagic.com/blog/how-neural-magics-deep-sparse-technology-works/).

## Creating A Sparse Version of YOLOv8 Trained on a Custom Dataset

SparseZoo, an open-source model repository by Neural Magic, offers [a collection of pre-sparsified YOLOv8 model checkpoints](https://sparsezoo.neuralmagic.com/?modelSet=computer_vision&searchModels=yolo). With SparseML, seamlessly integrated with Ultralytics, users can effortlessly fine-tune these sparse checkpoints on their specific datasets using a straightforward command-line interface.

Checkout [Neural Magic's SparseML YOLOv8 documentation](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov8) for more details.

## Usage: Deploying YOLOV8 using DeepSparse

Deploying YOLOv8 with Neural Magic's DeepSparse involves a few straightforward steps. Before diving into the usage instructions, be sure to check out the range of [YOLOv8 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements. Here's how you can get started.

### Step 1: Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required packages
        pip install deepsparse[yolov8]
        ```

### Step 2: Exporting YOLOv8 to ONNX Format

DeepSparse Engine requires YOLOv8 models in ONNX format. Exporting your model to this format is essential for compatibility with DeepSparse. Use the following command to export YOLOv8 models:

!!! tip "Model Export"

    === "CLI"

        ```bash
        # Export YOLOv8 model to ONNX format
        yolo task=detect mode=export model=yolov8n.pt format=onnx opset=13
        ```

This command will save the `yolov8n.onnx` model to your disk.

### Step 3: Deploying and Running Inferences

With your YOLOv8 model in ONNX format, you can deploy and run inferences using DeepSparse. This can be done easily with their intuitive Python API:

!!! tip "Deploying and Running Inferences"

    === "Python"

        ```python
        from deepsparse import Pipeline

        # Specify the path to your YOLOv8 ONNX model
        model_path = "path/to/yolov8n.onnx"

        # Set up the DeepSparse Pipeline
        yolo_pipeline = Pipeline.create(task="yolov8", model_path=model_path)

        # Run the model on your images
        images = ["path/to/image.jpg"]
        pipeline_outputs = yolo_pipeline(images=images)
        ```

### Step 4: Benchmarking Performance

It's important to check that your YOLOv8 model is performing optimally on DeepSparse. You can benchmark your model's performance to analyze throughput and latency:

!!! tip "Benchmarking"

    === "CLI"

        ```bash
        # Benchmark performance
        deepsparse.benchmark model_path="path/to/yolov8n.onnx" --scenario=sync --input_shapes="[1,3,640,640]"
        ```

### Step 5: Additional Features

DeepSparse provides additional features for practical integration of YOLOv8 in applications, such as image annotation and dataset evaluation.

!!! tip "Additional Features"

    === "CLI"

        ```bash
        # For image annotation
        deepsparse.yolov8.annotate --source "path/to/image.jpg" --model_filepath "path/to/yolov8n.onnx"

        # For evaluating model performance on a dataset
        deepsparse.yolov8.eval --model_path "path/to/yolov8n.onnx"
        ```

Running the annotate command processes your specified image, detecting objects, and saving the annotated image with bounding boxes and classifications. The annotated image will be stored in an annotation-results folder. This helps provide a visual representation of the model's detection capabilities.

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/image-annotation-feature.avif" alt="Image Annotation Feature">
</p>

After running the eval command, you will receive detailed output metrics such as [precision](https://www.ultralytics.com/glossary/precision), [recall](https://www.ultralytics.com/glossary/recall), and mAP (mean Average Precision). This provides a comprehensive view of your model's performance on the dataset. This functionality is particularly useful for fine-tuning and optimizing your YOLOv8 models for specific use cases, ensuring high accuracy and efficiency.

## Summary

This guide explored integrating Ultralytics' YOLOv8 with Neural Magic's DeepSparse Engine. It highlighted how this integration enhances YOLOv8's performance on CPU platforms, offering GPU-level efficiency and advanced neural network sparsity techniques.

For more detailed information and advanced usage, visit [Neural Magic's DeepSparse documentation](https://docs.neuralmagic.com/products/deepsparse/). Also, check out Neural Magic's documentation on the integration with YOLOv8 [here](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/yolov8#yolov8-inference-pipelines) and watch a great session on it [here](https://www.youtube.com/watch?v=qtJ7bdt52x8).

Additionally, for a broader understanding of various YOLOv8 integrations, visit the [Ultralytics integration guide page](../integrations/index.md), where you can discover a range of other exciting integration possibilities.

## FAQ

### What is Neural Magic's DeepSparse Engine and how does it optimize YOLOv8 performance?

Neural Magic's DeepSparse Engine is an inference runtime designed to optimize the execution of neural networks on CPUs through advanced techniques such as sparsity, pruning, and quantization. By integrating DeepSparse with YOLOv8, you can achieve GPU-like performance on standard CPUs, significantly enhancing inference speed, model efficiency, and overall performance while maintaining accuracy. For more details, check out the [Neural Magic's DeepSparse section](#neural-magics-deepsparse).

### How can I install the needed packages to deploy YOLOv8 using Neural Magic's DeepSparse?

Installing the required packages for deploying YOLOv8 with Neural Magic's DeepSparse is straightforward. You can easily install them using the CLI. Here's the command you need to run:

```bash
pip install deepsparse[yolov8]
```

Once installed, follow the steps provided in the [Installation section](#step-1-installation) to set up your environment and start using DeepSparse with YOLOv8.

### How do I convert YOLOv8 models to ONNX format for use with DeepSparse?

To convert YOLOv8 models to the ONNX format, which is required for compatibility with DeepSparse, you can use the following CLI command:

```bash
yolo task=detect mode=export model=yolov8n.pt format=onnx opset=13
```

This command will export your YOLOv8 model (`yolov8n.pt`) to a format (`yolov8n.onnx`) that can be utilized by the DeepSparse Engine. More information about model export can be found in the [Model Export section](#step-2-exporting-yolov8-to-onnx-format).

### How do I benchmark YOLOv8 performance on the DeepSparse Engine?

Benchmarking YOLOv8 performance on DeepSparse helps you analyze throughput and latency to ensure your model is optimized. You can use the following CLI command to run a benchmark:

```bash
deepsparse.benchmark model_path="path/to/yolov8n.onnx" --scenario=sync --input_shapes="[1,3,640,640]"
```

This command will provide you with vital performance metrics. For more details, see the [Benchmarking Performance section](#step-4-benchmarking-performance).

### Why should I use Neural Magic's DeepSparse with YOLOv8 for object detection tasks?

Integrating Neural Magic's DeepSparse with YOLOv8 offers several benefits:

- **Enhanced Inference Speed:** Achieves up to 525 FPS, significantly speeding up YOLOv8's capabilities.
- **Optimized Model Efficiency:** Uses sparsity, pruning, and quantization techniques to reduce model size and computational needs while maintaining accuracy.
- **High Performance on Standard CPUs:** Offers GPU-like performance on cost-effective CPU hardware.
- **Streamlined Integration:** User-friendly tools for easy deployment and integration.
- **Flexibility:** Supports both standard and sparsity-optimized YOLOv8 models.
- **Cost-Effective:** Reduces operational expenses through efficient resource utilization.

For a deeper dive into these advantages, visit the [Benefits of Integrating Neural Magic's DeepSparse with YOLOv8 section](#benefits-of-integrating-neural-magics-deepsparse-with-yolov8).
