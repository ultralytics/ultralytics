---
comments: true
description: Enhance YOLO26 performance using Neural Magic's DeepSparse Engine. Learn how to deploy and benchmark YOLO26 models on CPUs for efficient object detection.
keywords: YOLO26, DeepSparse, Neural Magic, model optimization, object detection, inference speed, CPU performance, sparsity, pruning, quantization
---

# Optimizing YOLO26 Inferences with Neural Magic's DeepSparse Engine

When deploying [object detection](https://www.ultralytics.com/glossary/object-detection) models like [Ultralytics YOLO26](https://www.ultralytics.com/) on various hardware, you can bump into unique issues like optimization. This is where YOLO26's integration with Neural Magic's DeepSparse Engine steps in. It transforms the way YOLO26 models are executed and enables GPU-level performance directly on CPUs.

This guide shows you how to deploy YOLO26 using Neural Magic's DeepSparse, how to run inferences, and also how to benchmark performance to ensure it is optimized.

!!! danger "SparseML EOL"

    Neural Magic was [acquired by Red Hat in January 2025](https://www.redhat.com/en/about/press-releases/red-hat-completes-acquisition-neural-magic-fuel-optimized-generative-ai-innovation-across-hybrid-cloud), and is deprecating the community versions of their `deepsparse`, `sparseml`, `sparsezoo`, and `sparsify` libraries. For additional information, see the notice posted [in the Readme on the `sparseml` GitHub repository](https://github.com/neuralmagic/sparsify/blob/5eb26a4e21b497ce573d10024e318a5ce48a7f9c/README.md#-2025-end-of-life-announcement-deepsparse-sparseml-sparsezoo-and-sparsify).

## Neural Magic's DeepSparse

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/neural-magic-deepsparse-overview.avif" alt="Neural Magic's DeepSparse Overview">
</p>

[Neural Magic's DeepSparse](https://github.com/neuralmagic/deepsparse/blob/main/README.md) is an inference run-time designed to optimize the execution of neural networks on CPUs. It applies advanced techniques like sparsity, pruning, and quantization to dramatically reduce computational demands while maintaining accuracy. DeepSparse offers an agile solution for efficient and scalable [neural network](https://www.ultralytics.com/glossary/neural-network-nn) execution across various devices.

## Benefits of Integrating Neural Magic's DeepSparse with YOLO26

Before diving into how to deploy YOLO26 using DeepSparse, let's understand the benefits of using DeepSparse. Some key advantages include:

- **Enhanced Inference Speed**: Achieves up to 525 FPS (on YOLO11n), significantly speeding up YOLO's inference capabilities compared to traditional methods.

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/enhanced-inference-speed.avif" alt="Neural Magic DeepSparse inference acceleration">
</p>

- **Optimized Model Efficiency**: Uses pruning and quantization to enhance YOLO26's efficiency, reducing model size and computational requirements while maintaining [accuracy](https://www.ultralytics.com/glossary/accuracy).

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/optimized-model-efficiency.avif" alt="Neural Magic model optimization and pruning">
</p>

- **High Performance on Standard CPUs**: Delivers GPU-like performance on CPUs, providing a more accessible and cost-effective option for various applications.

- **Streamlined Integration and Deployment**: Offers user-friendly tools for easy integration of YOLO26 into applications, including image and video annotation features.

- **Support for Various Model Types**: Compatible with both standard and sparsity-optimized YOLO26 models, adding deployment flexibility.

- **Cost-Effective and Scalable Solution**: Reduces operational expenses and offers scalable deployment of advanced object detection models.

## How Does Neural Magic's DeepSparse Technology Work?

Neural Magic's DeepSparse technology is inspired by the human brain's efficiency in neural network computation. It adopts two key principles from the brain as follows:

- **Sparsity**: The process of sparsification involves pruning redundant information from [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) networks, leading to smaller and faster models without compromising accuracy. This technique reduces the network's size and computational needs significantly.

- **Locality of Reference**: DeepSparse uses a unique execution method, breaking the network into Tensor Columns. These columns are executed depth-wise, fitting entirely within the CPU's cache. This approach mimics the brain's efficiency, minimizing data movement and maximizing the CPU's cache use.

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/neural-magic-deepsparse-technology.avif" alt="How Neural Magic's DeepSparse Technology Works ">
</p>

## Creating A Sparse Version of YOLO26 Trained on a Custom Dataset

[SparseZoo](https://github.com/neuralmagic/sparsezoo/blob/main/README.md), an open-source model repository by Neural Magic, offers [a collection of pre-sparsified YOLO26 model checkpoints](https://github.com/neuralmagic/sparsezoo/blob/main/README.md). With [SparseML](https://github.com/neuralmagic/sparseml), seamlessly integrated with Ultralytics, users can effortlessly fine-tune these sparse checkpoints on their specific datasets using a straightforward command-line interface.

Check out [Neural Magic's SparseML YOLO26 documentation](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov8) for more details.

## Usage: Deploying YOLO26 using DeepSparse

Deploying YOLO26 with Neural Magic's DeepSparse involves a few straightforward steps. Before diving into the usage instructions, be sure to check out the range of [YOLO26 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements. Here's how you can get started.

### Step 1: Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required packages
        pip install deepsparse[yolov8]
        ```

### Step 2: Exporting YOLO26 to ONNX Format

DeepSparse Engine requires YOLO26 models in [ONNX format](../integrations/onnx.md). Exporting your model to this format is essential for compatibility with DeepSparse. Use the following command to export YOLO26 models:

!!! tip "Model Export"

    === "CLI"

        ```bash
        # Export YOLO26 model to ONNX format
        yolo task=detect mode=export model=yolo26n.pt format=onnx opset=13
        ```

This command will save the `yolo26n.onnx` model to your disk.

### Step 3: Deploying and Running Inferences

With your YOLO26 model in ONNX format, you can deploy and run inferences using DeepSparse. This can be done easily with their intuitive Python API:

!!! tip "Deploying and Running Inferences"

    === "Python"

        ```python
        from deepsparse import Pipeline

        # Specify the path to your YOLO26 ONNX model
        model_path = "path/to/yolo26n.onnx"

        # Set up the DeepSparse Pipeline
        yolo_pipeline = Pipeline.create(task="yolov8", model_path=model_path)

        # Run the model on your images
        images = ["path/to/image.jpg"]
        pipeline_outputs = yolo_pipeline(images=images)
        ```

### Step 4: Benchmarking Performance

It's important to check that your YOLO26 model is performing optimally on DeepSparse. You can [benchmark](../modes/benchmark.md) your model's performance to analyze throughput and latency:

!!! tip "Benchmarking"

    === "CLI"

        ```bash
        # Benchmark performance
        deepsparse.benchmark model_path="path/to/yolo26n.onnx" --scenario=sync --input_shapes="[1,3,640,640]"
        ```

### Step 5: Additional Features

DeepSparse provides additional features for practical integration of YOLO26 in applications, such as image annotation and dataset evaluation.

!!! tip "Additional Features"

    === "CLI"

        ```bash
        # For image annotation
        deepsparse.yolov8.annotate --source "path/to/image.jpg" --model_filepath "path/to/yolo26n.onnx"

        # For evaluating model performance on a dataset
        deepsparse.yolov8.eval --model_path "path/to/yolo26n.onnx"
        ```

Running the annotate command processes your specified image, detecting objects, and saving the annotated image with bounding boxes and classifications. The annotated image will be stored in an annotation-results folder. This helps provide a visual representation of the model's detection capabilities.

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/image-annotation-feature.avif" alt="Neural Magic annotation feature interface">
</p>

After running the eval command, you will receive detailed output metrics such as [precision](https://www.ultralytics.com/glossary/precision), [recall](https://www.ultralytics.com/glossary/recall), and [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) (mean Average Precision). This provides a comprehensive view of your model's performance on the dataset and is particularly useful for fine-tuning and optimizing your YOLO26 models for specific use cases, ensuring high accuracy and efficiency.

## Summary

This guide explored integrating Ultralytics' YOLO26 with Neural Magic's DeepSparse Engine. It highlighted how this integration enhances YOLO26's performance on CPU platforms, offering GPU-level efficiency and advanced neural network sparsity techniques.

For more detailed information and advanced usage, visit the [DeepSparse documentation by Neural Magic](https://www.redhat.com/en/about/press-releases/red-hat-completes-acquisition-neural-magic-fuel-optimized-generative-ai-innovation-across-hybrid-cloud). You can also [explore the YOLO26 integration guide](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/yolov8#yolov8-inference-pipelines) and [watch a walkthrough session on YouTube](https://www.youtube.com/watch?v=qtJ7bdt52x8).

Additionally, for a broader understanding of various YOLO26 integrations, visit the [Ultralytics integration guide page](../integrations/index.md), where you can discover a range of other exciting integration possibilities.

## FAQ

### What is Neural Magic's DeepSparse Engine and how does it optimize YOLO26 performance?

Neural Magic's DeepSparse Engine is an inference runtime designed to optimize the execution of neural networks on CPUs through advanced techniques such as sparsity, pruning, and quantization. By integrating DeepSparse with YOLO26, you can achieve GPU-like performance on standard CPUs, significantly enhancing inference speed, model efficiency, and overall performance while maintaining accuracy. For more details, check out the [Neural Magic's DeepSparse section](#neural-magics-deepsparse).

### How can I install the needed packages to deploy YOLO26 using Neural Magic's DeepSparse?

Installing the required packages for deploying YOLO26 with Neural Magic's DeepSparse is straightforward. You can easily install them using the CLI. Here's the command you need to run:

```bash
pip install deepsparse[yolov8]
```

Once installed, follow the steps provided in the [Installation section](#step-1-installation) to set up your environment and start using DeepSparse with YOLO26.

### How do I convert YOLO26 models to ONNX format for use with DeepSparse?

To convert YOLO26 models to the ONNX format, which is required for compatibility with DeepSparse, you can use the following CLI command:

```bash
yolo task=detect mode=export model=yolo26n.pt format=onnx opset=13
```

This command will export your YOLO26 model (`yolo26n.pt`) to a format (`yolo26n.onnx`) that can be utilized by the DeepSparse Engine. More information about model export can be found in the [Model Export section](#step-2-exporting-yolo26-to-onnx-format).

### How do I benchmark YOLO26 performance on the DeepSparse Engine?

Benchmarking YOLO26 performance on DeepSparse helps you analyze throughput and latency to ensure your model is optimized. You can use the following CLI command to run a benchmark:

```bash
deepsparse.benchmark model_path="path/to/yolo26n.onnx" --scenario=sync --input_shapes="[1,3,640,640]"
```

This command will provide you with vital performance metrics. For more details, see the [Benchmarking Performance section](#step-4-benchmarking-performance).

### Why should I use Neural Magic's DeepSparse with YOLO26 for object detection tasks?

Integrating Neural Magic's DeepSparse with YOLO26 offers several benefits:

- **Enhanced Inference Speed:** Achieves up to 525 FPS (on YOLO11n), demonstrating DeepSparse's optimization capabilities.
- **Optimized Model Efficiency:** Uses sparsity, pruning, and quantization techniques to reduce model size and computational needs while maintaining accuracy.
- **High Performance on Standard CPUs:** Offers GPU-like performance on cost-effective CPU hardware.
- **Streamlined Integration:** User-friendly tools for easy deployment and integration.
- **Flexibility:** Supports both standard and sparsity-optimized YOLO26 models.
- **Cost-Effective:** Reduces operational expenses through efficient resource utilization.

For a deeper dive into these advantages, visit the [Benefits of Integrating Neural Magic's DeepSparse with YOLO26 section](#benefits-of-integrating-neural-magics-deepsparse-with-yolo26).
