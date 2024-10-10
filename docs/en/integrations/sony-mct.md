---
comments: true
description: 
keywords: 
---

# Supported Features of MCT

MCT offers a comprehensive suite of features designed to optimize neural network models for efficient deployment. These features enhance model performance and compatibility across various platforms. Here's a detailed overview of the supported features:

## Quantization

MCT supports several quantization methods, each with varying levels of complexity and computational cost:

- **Post-training quantization (PTQ)**:
  - Available via Keras API and PyTorch API.
  - Complexity: Low
  - Computational Cost: Low (order of minutes)

- **Gradient-based post-training quantization (GPTQ)**:
  - Available via Keras API and PyTorch API.
  - Complexity: Mild
  - Computational Cost: Mild (order of 2-3 hours)

- **Quantization-aware training (QAT)**:
  - Complexity: High
  - Computational Cost: High (order of 12-36 hours)

In addition, MCT supports various quantization schemes for weights and activations:

- Power-Of-Two (hardware-friendly)
- Symmetric
- Uniform

### Main Features

- **Graph Optimizations**: Transform models into more efficient versions (e.g., folding batch-normalization layers into preceding linear layers).
- **Quantization Parameter Search**: Minimize quantization noise using methods like Mean-Square-Error or other metrics like No-Clipping and Mean-Average-Error.
- **Advanced Quantization Algorithms**:
  - **Shift Negative Correction**: Addresses performance issues from symmetric activation quantization.
  - **Outliers Filtering**: Uses z-score to detect and remove outliers.
  - **Clustering**: Utilizes non-uniform quantization grids for better distribution matching.
- **Mixed-Precision Search**: Assigns quantization bit-width per layer based on sensitivity to various bit-widths.
- **Visualization**: Use TensorBoard to observe model performance insights, like quantization phases and bit-width configurations.

### Enhanced Post-Training Quantization (EPTQ)

As part of the GPTQ, MCT includes the Enhanced Post-Training Quantization (EPTQ) algorithm for advanced optimization. Details can be found in the paper: "EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian". For usage instructions, refer to the [EPTQ guidelines](#).

### Structured Pruning

MCT introduces structured, hardware-aware model pruning designed for specific hardware architectures. This technique leverages the target platform's Single Instruction, Multiple Data (SIMD) capabilities. By pruning SIMD groups, it reduces model size and complexity while optimizing channel utilization, aligned with the SIMD architecture for a targeted resource utilization of weights memory footprint. Available via Keras API and PyTorch API.

## Exporting YOLO Models with MCT

### Installation

To install the required package, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO11
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [YOLO11 Installation guide](../quickstart.md). While installing the required packages for YOLO11, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Before diving into the usage instructions, be sure to check out the range of [YOLO11 models offered by Ultralytics](../models/index.md). This will help you choose the most appropriate model for your project requirements.

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export the model to MCT format
        model.export(format="mct") # export with ptq quantization by default 
        # or 
        # model.export(format="mct", gptq=True) # export with gptq quantization


        # Load the exported MCT ONNX model
        mct_onnx_model = YOLO("yolo11n_mct_model.onnx")

        # Run inference
        results = mct_onnx_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
       # Export yolo11n to MCT format
        yolo export model=yolo11n.pt format=mct

        # Run inference with the exported model
        yolo predict model=yolo11n_mct_model.onnx source='https://ultralytics.com/images/bus.jpg'
        ```

For more details about the export process, visit the [Ultralytics documentation page on exporting](../modes/export.md).
