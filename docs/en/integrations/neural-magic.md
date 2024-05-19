---
comments: true
description: Learn how to deploy your YOLOv8 models rapidly using Neural Magic’s DeepSparse. This guide focuses on integrating Ultralytics YOLOv8 with the DeepSparse Engine for high-speed, CPU-based inference, leveraging advanced neural network sparsity techniques.
keywords: YOLOv8, DeepSparse Engine, Ultralytics, CPU Inference, Neural Network Sparsity, Object Detection, Model Optimization
---

# Optimizing YOLOv8 Inferences with Neural Magic’s DeepSparse Engine

When deploying object detection models like [Ultralytics YOLOv8](https://ultralytics.com) on various hardware, you can bump into unique issues like optimization. This is where YOLOv8’s integration with Neural Magic’s DeepSparse Engine steps in. It transforms the way YOLOv8 models are executed and enables GPU-level performance directly on CPUs.

This guide shows you how to deploy YOLOv8 using Neural Magic's DeepSparse, how to run inferences, and also how to benchmark performance to ensure it is optimized.

## Neural Magic’s DeepSparse

<p align="center">
  <img width="640" src="https://docs.neuralmagic.com/assets/images/nm-flows-55d56c0695a30bf9ecb716ea98977a95.png" alt="Neural Magic’s DeepSparse Overview">
</p>

[Neural Magic’s DeepSparse](https://neuralmagic.com/deepsparse/) is an inference run-time designed to optimize the execution of neural networks on CPUs. It applies advanced techniques like sparsity, pruning, and quantization to dramatically reduce computational demands while maintaining accuracy. DeepSparse offers an agile solution for efficient and scalable neural network execution across various devices.

## Benefits of Integrating Neural Magic’s DeepSparse with YOLOv8

Before diving into how to deploy YOLOV8 using DeepSparse, let’s understand the benefits of using DeepSparse. Some key advantages include:

- **Enhanced Inference Speed**: Achieves up to 525 FPS (on YOLOv8n), significantly speeding up YOLOv8's inference capabilities compared to traditional methods.

<p align="center">
  <img width="640" src="https://neuralmagic.com/wp-content/uploads/2023/04/image1.png" alt="Enhanced Inference Speed">
</p>

- **Optimized Model Efficiency**: Uses pruning and quantization to enhance YOLOv8's efficiency, reducing model size and computational requirements while maintaining accuracy.

<p align="center">
  <img width="640" src="https://neuralmagic.com/wp-content/uploads/2023/04/YOLOv8-charts-Page-1.drawio.png" alt="Optimized Model Efficiency">
</p>

- **High Performance on Standard CPUs**: Delivers GPU-like performance on CPUs, providing a more accessible and cost-effective option for various applications.

- **Streamlined Integration and Deployment**: Offers user-friendly tools for easy integration of YOLOv8 into applications, including image and video annotation features.

- **Support for Various Model Types**: Compatible with both standard and sparsity-optimized YOLOv8 models, adding deployment flexibility.

- **Cost-Effective and Scalable Solution**: Reduces operational expenses and offers scalable deployment of advanced object detection models.

## How Does Neural Magic's DeepSparse Technology Works?

Neural Magic’s Deep Sparse technology is inspired by the human brain’s efficiency in neural network computation. It adopts two key principles from the brain as follows:

- **Sparsity**: The process of sparsification involves pruning redundant information from deep learning networks, leading to smaller and faster models without compromising accuracy. This technique reduces the network's size and computational needs significantly.

- **Locality of Reference**: DeepSparse uses a unique execution method, breaking the network into Tensor Columns. These columns are executed depth-wise, fitting entirely within the CPU's cache. This approach mimics the brain's efficiency, minimizing data movement and maximizing the CPU's cache use.

<p align="center">
  <img width="640" src="https://neuralmagic.com/wp-content/uploads/2021/03/Screen-Shot-2021-03-16-at-11.09.45-AM.png" alt="How Neural Magic's DeepSparse Technology Works ">
</p>

For more details on how Neural Magic's DeepSparse technology work, check out [their blog post](https://neuralmagic.com/blog/how-neural-magics-deep-sparse-technology-works/).

## Creating A Sparse Version of YOLOv8 Trained on a Custom Dataset

SparseZoo, an open-source model repository by Neural Magic, offers [a collection of pre-sparsified YOLOv8 model checkpoints](https://sparsezoo.neuralmagic.com/?modelSet=computer_vision&searchModels=yolo). With SparseML, seamlessly integrated with Ultralytics, users can effortlessly fine-tune these sparse checkpoints on their specific datasets using a straightforward command-line interface.

Checkout [Neural Magic's SparseML YOLOv8 documentation](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov8) for more details.

## Usage: Deploying YOLOV8 using DeepSparse

Deploying YOLOv8 with Neural Magic's DeepSparse involves a few straightforward steps. Before diving into the usage instructions, be sure to check out the range of [YOLOv8 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements. Here's how you can get started.

### Step 1: Installation

To install the required packages, run:

!!! Tip "Installation"

    === "CLI"

        ```bash
        # Install the required packages
        pip install deepsparse[yolov8]
        ```

### Step 2: Exporting YOLOv8 to ONNX Format

DeepSparse Engine requires YOLOv8 models in ONNX format. Exporting your model to this format is essential for compatibility with DeepSparse. Use the following command to export YOLOv8 models:

!!! Tip "Model Export"

    === "CLI"

        ```bash
        # Export YOLOv8 model to ONNX format
        yolo task=detect mode=export model=yolov8n.pt format=onnx opset=13
        ```

This command will save the `yolov8n.onnx` model to your disk.

### Step 3: Deploying and Running Inferences

With your YOLOv8 model in ONNX format, you can deploy and run inferences using DeepSparse. This can be done easily with their intuitive Python API:

!!! Tip "Deploying and Running Inferences"

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

!!! Tip "Benchmarking"

    === "CLI"

        ```bash
        # Benchmark performance
        deepsparse.benchmark model_path="path/to/yolov8n.onnx" --scenario=sync --input_shapes="[1,3,640,640]"
        ```

### Step 5: Additional Features

DeepSparse provides additional features for practical integration of YOLOv8 in applications, such as image annotation and dataset evaluation.

!!! Tip "Additional Features"

    === "CLI"

        ```bash
        # For image annotation
        deepsparse.yolov8.annotate --source "path/to/image.jpg" --model_filepath "path/to/yolov8n.onnx"

        # For evaluating model performance on a dataset
        deepsparse.yolov8.eval --model_path "path/to/yolov8n.onnx"
        ```

Running the annotate command processes your specified image, detecting objects, and saving the annotated image with bounding boxes and classifications. The annotated image will be stored in an annotation-results folder. This helps provide a visual representation of the model's detection capabilities.

<p align="center">
  <img width="640" src="https://user-images.githubusercontent.com/3195154/211942937-1d32193a-6dda-473d-a7ad-e2162bbb42e9.png" alt="Image Annotation Feature">
</p>

After running the eval command, you will receive detailed output metrics such as precision, recall, and mAP (mean Average Precision). This provides a comprehensive view of your model's performance on the dataset. This functionality is particularly useful for fine-tuning and optimizing your YOLOv8 models for specific use cases, ensuring high accuracy and efficiency.

## Summary

This guide explored integrating Ultralytics’ YOLOv8 with Neural Magic's DeepSparse Engine. It highlighted how this integration enhances YOLOv8's performance on CPU platforms, offering GPU-level efficiency and advanced neural network sparsity techniques.

For more detailed information and advanced usage, visit [Neural Magic’s DeepSparse documentation](https://docs.neuralmagic.com/products/deepsparse/). Also, check out Neural Magic’s documentation on the integration with YOLOv8 [here](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/yolov8#yolov8-inference-pipelines) and watch a great session on it [here](https://www.youtube.com/watch?v=qtJ7bdt52x8).

Additionally, for a broader understanding of various YOLOv8 integrations, visit the [Ultralytics integration guide page](../integrations/index.md), where you can discover a range of other exciting integration possibilities.
