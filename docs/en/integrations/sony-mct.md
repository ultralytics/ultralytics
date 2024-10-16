---
comments: true
description: Learn to export YOLOv8 models to Sony's Model Compression Toolkit (MCT) format to optimize your models for efficient deployment.
keywords: YOLOv8, Sony MCT, model export, quantization, pruning, deep learning optimization
---

# Sony MCT Export

In this guide, we cover exporting YOLOv8 models to Sony's Model Compression Toolkit (MCT) format, which offers a comprehensive suite of features designed to optimize neural network models for efficient deployment. These features enhance model performance and compatibility across various platforms by leveraging advanced quantization and pruning techniques.

## Introduction to Sony MCT

Sony's Model Compression Toolkit (MCT) is a powerful tool for optimizing deep learning models through quantization and pruning. It supports various quantization methods and provides advanced algorithms to reduce model size and computational complexity without significantly sacrificing accuracy. MCT is particularly useful for deploying models on resource-constrained devices, ensuring efficient inference and reduced latency.

## Usage Examples

Export a YOLOv8 model to MCT format and run inference with the exported model.

!!! example

    === "Python"
        ```python
        from ultralytics import YOLO

        # Load the YOLOv8n model
        model = YOLO("yolov8n.pt")
        # Export the model to MCT format with Post-Training Quantization (PTQ)
        model.export(format="mct")  # exports with PTQ quantization by default
        # Alternatively, export with Gradient-based Post-Training Quantization (GPTQ)
        # model.export(format="mct", gptq=True)
        # Load the exported MCT ONNX model
        mct_model = YOLO("yolov8n_mct_model.onnx")
        # Run inference
        results = mct_model("https://ultralytics.com/images/bus.jpg")
        ```
    === "CLI"
        ```bash
        # Export YOLOv8n to MCT format with PTQ quantization
        yolo export model=yolov8n.pt format=mct
        # Alternatively, export with GPTQ quantization
        # yolo export model=yolov8n.pt format=mct gptq=True
        # Run inference with the exported model
        yolo predict model=yolov8n_mct_model.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

## Supported Features of MCT

Sony's MCT offers a range of features designed to optimize neural network models:

### Quantization

MCT supports several quantization methods to reduce model size and improve inference speed:

- **Post-Training Quantization (PTQ)**:
    - Available via Keras and PyTorch APIs.
    - Complexity: Low
    - Computational Cost: Low (minutes)
- **Gradient-based Post-Training Quantization (GPTQ)**:
    - Available via Keras and PyTorch APIs.
    - Complexity: Medium
    - Computational Cost: Moderate (2-3 hours)
- **Quantization-Aware Training (QAT)**:
    - Complexity: High
    - Computational Cost: High (12-36 hours)

MCT also supports various quantization schemes for weights and activations:

- **Power-of-Two** (hardware-friendly)
- **Symmetric**
- **Uniform**

### Main Features

- **Graph Optimizations**: Transforms models into more efficient versions by folding layers like batch normalization into preceding layers.
- **Quantization Parameter Search**: Minimizes quantization noise using metrics like Mean-Square-Error, No-Clipping, and Mean-Average-Error.
- **Advanced Quantization Algorithms**:
    - **Shift Negative Correction**: Addresses performance issues from symmetric activation quantization.
    - **Outliers Filtering**: Uses z-score to detect and remove outliers.
    - **Clustering**: Utilizes non-uniform quantization grids for better distribution matching.
- **Mixed-Precision Search**: Assigns different quantization bit-widths per layer based on sensitivity.
- **Visualization**: Use TensorBoard to observe model performance insights, quantization phases, and bit-width configurations.

#### Enhanced Post-Training Quantization (EPTQ)

As part of GPTQ, MCT includes the Enhanced Post-Training Quantization (EPTQ) algorithm for advanced optimization. EPTQ aims to further reduce quantization error without requiring labeled data. For more details, refer to the paper: [EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian](https://github.com/sony/model_optimization).

### Structured Pruning

MCT introduces structured, hardware-aware model pruning designed for specific hardware architectures. This technique leverages the target platform's Single Instruction, Multiple Data (SIMD) capabilities by pruning SIMD groups. This reduces model size and complexity while optimizing channel utilization, aligned with the SIMD architecture for targeted resource utilization of weights memory footprint. Available via Keras and PyTorch APIs.

## Arguments

When exporting a model to MCT format, you can specify various arguments:

| Key      | Value   | Description                                                         |
| -------- | ------- | ------------------------------------------------------------------- |
| `format` | `'mct'` | Format to export to (MCT)                                           |
| `gptq`   | `False` | Use Gradient-based Post-Training Quantization (GPTQ) instead of PTQ |

## Benefits of Using MCT

1. **Model Size Reduction**: Significantly reduces the model size through quantization and pruning.
2. **Inference Speedup**: Improves inference speed by optimizing computations.
3. **Hardware Compatibility**: Generates models optimized for specific hardware architectures.
4. **Advanced Algorithms**: Utilizes state-of-the-art quantization and pruning algorithms.
5. **Ease of Integration**: Seamlessly integrates with Keras and PyTorch models.

## Installation

To use MCT with YOLOv8, ensure you have the latest version of the Ultralytics package installed:

!!! tip "Installation"

    === "CLI"
        ```bash
        # Install the Ultralytics package
        pip install ultralytics
        ```

For detailed instructions and best practices, refer to the [YOLOv8 Installation Guide](../quickstart.md). If you encounter any issues, consult our [Common Issues Guide](../guides/yolo-common-issues.md) for solutions and tips.

## Using MCT Export in Deployment

After exporting your YOLOv8 model to MCT format, you can deploy it using standard ONNX runtime environments. The MCT export generates an optimized ONNX model that can be integrated into your deployment pipeline.

### Steps for Deployment

1. **Load the Model**: Use the ONNX Runtime or another compatible framework to load the exported model.
2. **Prepare Input Data**: Preprocess your input data to match the model's expected input format.
3. **Run Inference**: Execute the model on the input data to get predictions.
4. **Post-Processing**: Apply any necessary post-processing to interpret the model's outputs.

## Conclusion

Exporting YOLOv8 models to Sony's MCT format allows you to optimize your models for efficient deployment on various hardware platforms. By leveraging advanced quantization and pruning techniques, you can reduce model size and improve inference speed without significantly compromising accuracy.
For more information and detailed guidelines, refer to Sony's [Model Compression Toolkit documentation](https://github.com/sony/model_optimization).

## FAQ

### How do I export YOLOv8 models to MCT format?

You can export YOLOv8 models to MCT format using either Python or CLI commands:
!!! example
=== "Python"

````python
from ultralytics import YOLO

        # Load the YOLOv8n model
        model = YOLO("YOLOv8n.pt")
        # Export the model to MCT format
        model.export(format="mct")
        ```
    === "CLI"
        ```bash
        # Export YOLOv8n to MCT format
        yolo export model=YOLOv8n.pt format=mct
        ```

### What quantization methods does MCT support?

MCT supports several quantization methods:

- **Post-Training Quantization (PTQ)**
- **Gradient-based Post-Training Quantization (GPTQ)**
- **Quantization-Aware Training (QAT)**
    These methods vary in complexity and computational cost, allowing you to choose the one that best fits your needs.

### Can I apply structured pruning with MCT?

Yes, MCT supports structured, hardware-aware pruning to optimize models further. This technique reduces model size and complexity while optimizing for specific hardware architectures.

### How does MCT improve inference speed?

By reducing model size through quantization and pruning, MCT decreases the computational load during inference. This leads to faster inference times, making it suitable for deployment on resource-constrained devices.

### Where can I find more information about MCT?

For more detailed information, refer to Sony's [Model Compression Toolkit documentation](#) or the official [MCT GitHub repository](#).
````
