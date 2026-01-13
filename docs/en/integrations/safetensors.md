---
comments: true
description: Export YOLO models to SafeTensors format for fast, safe, and efficient model loading. Learn how to convert and use SafeTensors with Ultralytics YOLO.
keywords: Ultralytics, YOLO, SafeTensors, model export, fast loading, secure weights, Hugging Face, model serialization, PyTorch
---

# Export YOLO Models to SafeTensors Format

<p align="center">
  <img width="75%" src="https://github.com/ultralytics/assets/releases/download/v0.0.0/safetensors-logo.avif" alt="SafeTensors logo">
</p>

[SafeTensors](https://github.com/huggingface/safetensors) is a fast, safe, and efficient file format for storing and loading model weights. Developed by [Hugging Face](https://huggingface.co/), it provides significant advantages over traditional pickle-based formats like PyTorch's `.pt` files, including protection against arbitrary code execution and dramatically faster loading times.

This guide covers how to export [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models to SafeTensors format, enabling you to benefit from faster model loading and enhanced security.

## Why Export to SafeTensors?

SafeTensors was designed to address security vulnerabilities and performance limitations of existing tensor serialization formats. Here's why you should consider using it:

| Feature                  | PyTorch (pickle) | SafeTensors |
| ------------------------ | :--------------: | :---------: |
| Safe (no code execution) |        ❌        |     ✅      |
| Zero-copy loading        |        ❌        |     ✅      |
| Lazy loading             |        ❌        |     ✅      |
| No file size limit       |        ✅        |     ✅      |
| Bfloat16/Fp8 support     |        ✅        |     ✅      |

## Key Features of SafeTensors

SafeTensors offers several compelling advantages for storing and loading YOLO model weights:

- **Security**: SafeTensors files cannot execute arbitrary code during loading, unlike pickle-based formats. This eliminates a major security vulnerability when loading models from untrusted sources.

- **Speed**: SafeTensors uses memory-mapped file I/O, enabling extremely fast loading times—up to **200-500x faster** than traditional PyTorch loading, especially beneficial for large models.

- **Zero-Copy Loading**: Weights can be loaded directly to GPU memory without intermediate CPU copies, reducing memory usage and improving efficiency.

- **Framework Agnostic**: While SafeTensors works seamlessly with PyTorch, it also supports TensorFlow, JAX, and other frameworks, making it ideal for cross-framework model sharing.

- **Simple Format**: The format is straightforward, consisting of a header with metadata followed by raw tensor data, making it easy to inspect and validate.

## Deployment Options

SafeTensors models exported from Ultralytics YOLO can be used in various scenarios:

- **Fast Model Loading**: When you need to quickly load models in production environments where startup time matters.

- **Secure Model Sharing**: When distributing models publicly or receiving models from external sources, SafeTensors eliminates code execution risks.

- **Cloud Deployment**: SafeTensors' efficient loading reduces cold-start times in serverless and containerized deployments.

- **Model Hubs**: Many model hubs (like Hugging Face Hub) prefer or require SafeTensors format for security and performance.

- **Research & Development**: Quick iteration during development with fast model loading and saving.

## Exporting YOLO Models to SafeTensors

Export your Ultralytics YOLO models to SafeTensors format for faster, safer model loading.

### Installation

To export to SafeTensors format, ensure you have the required packages installed:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install Ultralytics with SafeTensors support
        pip install ultralytics "safetensors>=0.7.0"
        ```

For detailed instructions on installation, check our [YOLO Installation guide](../quickstart.md). If you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions.

### Usage

Exporting YOLO models to SafeTensors is straightforward:

!!! example "Usage"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO model
        model = YOLO("yolo11n.pt")

        # Export the model to SafeTensors format
        model.export(format="safetensors")  # creates 'yolo11n.safetensors'

        # Load and run inference with the SafeTensors model
        safetensors_model = YOLO("yolo11n.safetensors")
        results = safetensors_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO model to SafeTensors format
        yolo export model=yolo11n.pt format=safetensors

        # Run inference with the exported model
        yolo predict model=yolo11n.safetensors source=https://ultralytics.com/images/bus.jpg
        ```

### Export Arguments

When exporting to SafeTensors format, you can specify the following arguments:

| Argument | Type            | Default | Description                                            |
| -------- | --------------- | ------- | ------------------------------------------------------ |
| `format` | `str`           | `None`  | Target format: `'safetensors'`                         |
| `imgsz`  | `int` or `list` | `640`   | Image size for model input                             |
| `half`   | `bool`          | `False` | Export model in FP16 (half precision) for smaller size |
| `int8`   | `bool`          | `False` | Enable INT8 quantization (requires calibration data)   |
| `batch`  | `int`           | `1`     | Batch size for export                                  |
| `device` | `str`           | `None`  | Device to use for export (`'cpu'`, `'cuda:0'`)         |

### Export Examples

Export with different configurations:

!!! example "Export Options"

    === "FP16 (Half Precision)"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        # Export with half precision for ~50% smaller file size
        model.export(format="safetensors", half=True)
        # Creates 'yolo11n_fp16.safetensors' file
        ```

    === "With Batch Size"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        # Export with specific batch size
        model.export(format="safetensors", batch=4)
        # Creates 'yolo11n_b4.safetensors' file
        ```

    === "Combined Options"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        # Export with FP16 and batch size
        model.export(format="safetensors", half=True, batch=2)
        # Creates 'yolo11n_b2_fp16.safetensors' file
        ```

### Output Structure

The SafeTensors export creates a single file containing the model weights and embedded metadata:

```text
yolo11n.safetensors    # Model weights with embedded configuration
```

For exports with parameters, the naming reflects the configuration:

```text
yolo11n_fp16.safetensors    # FP16 weights (single file)
```

The model configuration (task type, class names, arguments, etc.) is embedded directly in the SafeTensors file metadata, eliminating the need for separate configuration files.

## Using Exported SafeTensors Models

After exporting, you can use SafeTensors models directly with Ultralytics YOLO:

### Inference

Run inference with your SafeTensors model:

```python
from ultralytics import YOLO

# Load SafeTensors model
model = YOLO("yolo11n.safetensors")

# Run inference
results = model("https://ultralytics.com/images/bus.jpg")

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    print(f"Detected {len(boxes)} objects")
```

### Validation

Validate your SafeTensors model on a dataset:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.safetensors")
metrics = model.val(data="coco8.yaml")
print(f"mAP50-95: {metrics.box.map}")
```

## Supported Tasks

SafeTensors export works with all YOLO task types:

| Task           | Model Example     | Export Command                                         |
| -------------- | ----------------- | ------------------------------------------------------ |
| Detection      | `yolo11n.pt`      | `yolo export model=yolo11n.pt format=safetensors`      |
| Segmentation   | `yolo11n-seg.pt`  | `yolo export model=yolo11n-seg.pt format=safetensors`  |
| Classification | `yolo11n-cls.pt`  | `yolo export model=yolo11n-cls.pt format=safetensors`  |
| Pose           | `yolo11n-pose.pt` | `yolo export model=yolo11n-pose.pt format=safetensors` |
| OBB            | `yolo11n-obb.pt`  | `yolo export model=yolo11n-obb.pt format=safetensors`  |

!!! note

    SafeTensors file size is larger than compressed `.pt` files because it stores raw tensor data. However, loading is significantly faster due to memory-mapped I/O.

## Troubleshooting

### Common Issues

**Issue**: `safetensors not installed`

**Solution**: Install the safetensors package:

```bash
pip install "safetensors>=0.7.0"
```

**Issue**: `Model metadata not found`

**Solution**: Ensure you're using a SafeTensors file exported with Ultralytics. The model configuration is embedded in the file metadata:

```python
# Load SafeTensors model directly
model = YOLO("yolo11n.safetensors")
```

**Issue**: `Model architecture mismatch`

**Solution**: The model configuration must match the exported weights. Don't modify the SafeTensors file manually.

For more troubleshooting help, visit the [Ultralytics GitHub Issues](https://github.com/ultralytics/ultralytics/issues) or the [SafeTensors Documentation](https://huggingface.co/docs/safetensors/index).

## Summary

Exporting YOLO models to SafeTensors format provides a secure, fast way to store and load model weights. With dramatically improved loading times and protection against code execution vulnerabilities, SafeTensors is an excellent choice for production deployments and model sharing.

Key takeaways:

- SafeTensors provides **200-500x faster** loading compared to PyTorch's pickle format
- **Security**: No arbitrary code execution during model loading
- **Single file**: Everything is contained in one `.safetensors` file (metadata embedded)
- **Simple export**: Use `format='safetensors'` to export
- **Full compatibility**: Works with all YOLO tasks (detect, segment, classify, pose, obb)
- **Flexible options**: Support for FP16 and batch size configuration

## FAQ

### How do I export a YOLO model to SafeTensors format?

Export a YOLO model to SafeTensors using Python or CLI:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="safetensors")
```

or

```bash
yolo export model=yolo11n.pt format=safetensors
```

### What are the benefits of SafeTensors over PyTorch's .pt format?

SafeTensors offers several advantages:

1. **Security**: Cannot execute arbitrary code (unlike pickle-based `.pt` files)
2. **Zero-copy**: Direct loading to GPU without CPU intermediary
3. **Cross-framework**: Works with PyTorch, TensorFlow, JAX, and more

### Can I use SafeTensors models for training?

SafeTensors exported models are optimized for inference. For training, use the original PyTorch `.pt` format which includes optimizer states and training metadata. SafeTensors models can be loaded for inference or fine-tuning but don't include full training state.

### Why is the SafeTensors file larger than the .pt file?

PyTorch `.pt` files use compression, while SafeTensors stores raw tensor data for faster memory-mapped access. The larger file size is a tradeoff for significantly faster loading times. Use `half=True` during export to reduce file size by ~50% with FP16 precision.

### Is SafeTensors compatible with Hugging Face Hub?

Yes! SafeTensors is the preferred format on Hugging Face Hub. You can upload your exported SafeTensors models directly to the Hub for sharing and deployment. The format is widely supported across the Hugging Face ecosystem.

### What Python version is required for SafeTensors export?

SafeTensors export requires Python 3.9 or higher. Ensure your environment meets this requirement:

```bash
python --version # Should be 3.9+
pip install ultralytics "safetensors>=0.7.0"
```
