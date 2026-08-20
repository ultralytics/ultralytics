---
title: YOLO26 ExecuTorch for Mobile & Edge
comments: true
description: Export YOLO26 models to ExecuTorch format for efficient on-device inference on mobile and edge devices. Optimize your AI models for iOS, Android, and embedded systems.
keywords: Ultralytics, YOLO26, ExecuTorch, model export, PyTorch, edge AI, iOS, iPadOS, Android, Raspberry Pi, NVIDIA Jetson, mobile deployment, on-device inference, XNNPACK, embedded systems
---

# Deploy YOLO26 on Mobile & Edge with ExecuTorch

Deploying computer vision models on edge devices like smartphones, tablets, and embedded systems requires an optimized runtime that balances performance with resource constraints. ExecuTorch, PyTorch's solution for edge computing, enables efficient on-device inference for [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models.

This guide outlines how to export Ultralytics YOLO models to ExecuTorch format, enabling you to deploy your models on the following mobile and edge devices with optimized performance:

- **Mobile Applications**: Deploy on iOS and Android applications with native performance, enabling real-time object detection in mobile apps.

- **Embedded Systems**: Run on embedded Linux devices like Raspberry Pi, NVIDIA Jetson, and other ARM-based systems with optimized performance.

- **Edge AI Devices**: Deploy on specialized edge AI hardware with custom delegates for accelerated inference.

- **IoT Devices**: Integrate into IoT devices for on-device inference without cloud connectivity requirements.

## Why export to ExecuTorch?

<p align="center">
  <img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/executorch-pipeline.avif" alt="ExecuTorch pipeline diagram showing an Ultralytics YOLO model exported to .pte format for on-device inference">
</p>

[ExecuTorch](https://docs.pytorch.org/executorch/) is PyTorch's end-to-end solution for enabling on-device inference capabilities across mobile and edge devices. Built with the goal of being portable and efficient, ExecuTorch can be used to run PyTorch programs on a wide variety of computing platforms.

## Key features of ExecuTorch

ExecuTorch provides several powerful features for deploying Ultralytics YOLO models on edge devices:

- **Portable Model Format**: ExecuTorch uses the `.pte` (PyTorch ExecuTorch) format, which is optimized for size and loading speed on resource-constrained devices.

- **XNNPACK Backend**: Default integration with XNNPACK provides highly optimized inference on mobile CPUs, delivering excellent performance without requiring specialized hardware.

- **Quantization Ready**: The ExecuTorch ecosystem supports quantization techniques to reduce model size and improve inference speed; Ultralytics currently exports FP32 models via the XNNPACK backend.

- **Memory Efficiency**: Optimized memory management reduces runtime memory footprint, making it suitable for devices with limited RAM.

- **Model Metadata**: Exported models include metadata (image size, class names, etc.) in a separate YAML file for easy integration.

## Deployment Options with ExecuTorch

ExecuTorch models can be deployed across various edge and mobile platforms:

- **Mobile Applications**: Deploy on iOS and Android applications with native performance, enabling real-time object detection in mobile apps.

- **Embedded Systems**: Run on embedded Linux devices like Raspberry Pi, NVIDIA Jetson, and other ARM-based systems with optimized performance.

- **Edge AI Devices**: Deploy on specialized edge AI hardware with custom delegates for accelerated inference.

- **IoT Devices**: Integrate into IoT devices for on-device inference without cloud connectivity requirements.

## Supported Tasks

ExecuTorch export supports all seven Ultralytics tasks. Semantic segmentation and depth estimation are available only with YOLO26, the only family that ships those heads.

{% include "macros/supported-tasks.md" %}

## Exporting Ultralytics YOLO26 Models to ExecuTorch

Exporting Ultralytics YOLO26 models to ExecuTorch format enables efficient deployment on mobile and edge devices.

### Installation

ExecuTorch export requires Python 3.10-3.13 and PyTorch >= 2.9.0 along with the `executorch` package:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install Ultralytics package
        pip install ultralytics
        ```

For detailed instructions and best practices related to the installation process, check our [YOLO26 Installation guide](../quickstart.md). While installing the required packages for YOLO26, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

### Usage

Exporting YOLO26 models to ExecuTorch is straightforward:

The ExecuTorch format supports the [Export](../modes/export.md), [Predict](../modes/predict.md), and [Validate](../modes/val.md) modes. Export your model, then load the exported model to run inference or validate its accuracy.

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to ExecuTorch format
        model.export(format="executorch")  # creates 'yolo26n_executorch_model'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to ExecuTorch format
        yolo export model=yolo26n.pt format=executorch # creates 'yolo26n_executorch_model'
        ```

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ExecuTorch model
        model = YOLO("yolo26n_executorch_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported ExecuTorch model
        yolo predict model=yolo26n_executorch_model source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported ExecuTorch model
        model = YOLO("yolo26n_executorch_model")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported ExecuTorch model
        yolo val model=yolo26n_executorch_model data=coco8.yaml
        ```

    ExecuTorch exports generate a directory that includes a `.pte` file and metadata. Use the ExecuTorch runtime in your mobile or embedded application to load the `.pte` model and perform inference. To dynamically lookup human-readable detection class names and output-tensor shapes, embed the YAML metadata file in your app.

### Export Arguments

When exporting to ExecuTorch format, you can specify the following arguments:

| Argument   | Type             | Default        | Description                                                                                                                             |
| ---------- | ---------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'executorch'` | Target format for the exported model, defining compatibility with various deployment environments.                                      |
| `imgsz`    | `int` or `tuple` | `640`          | Desired image size for the model input. Can be an integer for square images or a tuple `(height, width)` for specific dimensions.       |
| `quantize` | `int` or `str`   | `None`         | Fixed FP32 export. ExecuTorch export does not support export-time FP16, INT8, or W8A16 precision conversion.                            |
| `batch`    | `int`            | `1`            | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `device`   | `str`            | `None`         | Specifies the device for exporting: GPU (`device=0`), CPU (`device=cpu`), MPS for Apple silicon (`device=mps`).                         |

### Output Structure

The ExecuTorch export creates a directory containing the model and metadata:

```text
yolo26n_executorch_model/
├── model.pte               # ExecuTorch model file
└── metadata.yaml           # Model metadata (classes, image size, etc.)
```

## Using Exported ExecuTorch Models

After exporting your model, you'll need to integrate it into your target application using the ExecuTorch runtime.

### Mobile Integration

For mobile applications (iOS/Android), you'll need to:

1. **Add ExecuTorch Runtime**: Include the ExecuTorch runtime library in your mobile project
2. **Load Model**: Load the `.pte` file in your application
3. **Run Inference**: Process images and get predictions

Example iOS integration (Objective-C/C++):

```objc
// iOS uses C++ APIs for model loading and inference
// See https://pytorch.org/executorch/stable/using-executorch-ios.html for complete examples

#include <executorch/extension/module/module.h>

using namespace ::executorch::extension;

// Load the model
Module module("/path/to/model.pte");

// Create input tensor
float input[1 * 3 * 640 * 640];
auto tensor = from_blob(input, {1, 3, 640, 640});

// Run inference
const auto result = module.forward(tensor);
```

Example Android integration (Kotlin Activity). ExecuTorch export disables the end-to-end (NMS-free) branch, so YOLO11 and YOLO26 models both emit the raw `(1, 4 + nc, 8400)` tensor and your app must run its own NMS. The sample below assumes the 80-class COCO shape `(1, 84, 8400)`:

```kotlin
// ...
import android.graphics.BitmapFactory
import androidx.core.graphics.scale
import java.io.File
import java.io.FileOutputStream
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
// ...

fun runObjectDetection() {
    Log.d(TAG, "Starting object detection...")

    // Load the model.
    // Copy the model from the app's assets to storage (to get a file path to the model).
    // `model.pte` is the file produced inside the exported `yolo26n_executorch_model` directory.
    val modelFileName = "model.pte"
    val outputFile = File(filesDir, modelFileName)
    if (!outputFile.exists()) { // In your app, you may need a way to deploy newer models
        assets.open(modelFileName).use { inputStream ->
            FileOutputStream(outputFile).use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }
    }
    val module = Module.load(outputFile.absolutePath)

    // Get our data from an image in the app package.
    val bitmap = assets.open("bus.jpg").use { inputStream ->
        BitmapFactory.decodeStream(inputStream)
    }
    val resizedBitmap = bitmap.scale(640, 640, false)

    // Prepare input tensor.
    // YOLO models expect [0,1] float input, not ImageNet normalization.
    // See org.pytorch.torchvision.TensorImageUtils
    val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
        resizedBitmap,
        floatArrayOf(0.0f, 0.0f, 0.0f),
        floatArrayOf(1.0f, 1.0f, 1.0f)
    )
    val inputEValue = EValue.from(inputTensor)

    // Run inference
    Log.d(TAG, "Running inference...")
    val outputs = module.forward(inputEValue)

    // Process results.
    // In your app, you should do this on a background thread.
    Log.d(TAG, "Processing results...")
    val predictions = tensorTo2DPredictions(outputs[0].toTensor())
    val detections = yoloNmsSingleLabel(predictions)

    Log.d(TAG, "Found ${detections.size} detections")
    detections.forEach {
        // In your app, you may want to lookup the human-readable class names from the
        // metadata.yaml file included with your exported model.
        Log.d(
            TAG,
            "Cls:${it.cls}, Conf:${it.conf}, Box:(${it.x1},${it.y1})->(${it.x2},${it.y2})"
        )
    }
}

// Simple example of YOLO NMS. In your app, you may want to add the following:
// * Multi-label: this sample forces one class per box.
// * Coordinate mapping: this returns boxes in 640x640 model space, not original image space.
// * Aspect-ratio handling: this assumes input was stretched, not letterboxed.
// * Flexible output-tensor shapes: this assumes YOLO11 dimensions (8400 boxes, 84 attributes).
// * Clipping: these bounding boxes are not clamped to image boundaries.
// * Error handling: omitted because it's sample code.
fun yoloNmsSingleLabel(
    predictions: Array<FloatArray>,
    confThreshold: Float = 0.25f,
    iouThreshold: Float = 0.45f
): List<Detection> {
    val detections = mutableListOf<Detection>()
    for (p in predictions) {
        val x = p[0]
        val y = p[1]
        val w = p[2]
        val h = p[3]
        val classScores = p.copyOfRange(4, p.size)

        // Single-label, so pick max-score class
        val cls = classScores.indices.maxBy { classScores[it] }
        val score = classScores[cls]
        if (score < confThreshold) continue

        val (x1, y1, x2, y2) = centerToCornerCoords(x, y, w, h)
        detections.add(Detection(x1, y1, x2, y2, score, cls))
    }
    return nmsDetections(detections, iouThreshold)
}

// Detection-data class to represent a single-object-detection result using
// corner-based bounding-box coordinates.
data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val conf: Float,
    val cls: Int
)

// Box data helper class
data class Box(val x1: Float, val y1: Float, val x2: Float, val y2: Float)

// Converts a bounding box from center-based coordinates (x, y, w, h) to
// corner-based coordinates (x1, y1, x2, y2).
fun centerToCornerCoords(x: Float, y: Float, w: Float, h: Float): Box {
    return Box(x - w / 2f, y - h / 2f, x + w / 2f, y + h / 2f)
}

// Performs Non-Maximum Suppression (NMS) to remove redundant overlapping detections,
// keeping only the highest-confidence boxes.
fun nmsDetections(
    detections: List<Detection>,
    iouThreshold: Float = 0.45f,
    maxDet: Int = 50
): List<Detection> {

    val sorted = detections.sortedByDescending { it.conf }.toMutableList()
    val keep = mutableListOf<Detection>()

    while (sorted.isNotEmpty()) {
        val best = sorted.removeAt(0)
        keep.add(best)

        val it = sorted.iterator()
        while (it.hasNext()) {
            val det = it.next()
            if (best.cls == det.cls && intersectionOverUnion(best, det) > iouThreshold) it.remove()
        }

        if (keep.size >= maxDet) break
    }

    return keep
}

// Calculates the Intersection over Union (IoU) between two detections.
// Used in object-detection tasks to measure how closely two detections overlap.
fun intersectionOverUnion(a: Detection, b: Detection): Float {
    val xx1 = maxOf(a.x1, b.x1)
    val yy1 = maxOf(a.y1, b.y1)
    val xx2 = minOf(a.x2, b.x2)
    val yy2 = minOf(a.y2, b.y2)

    val w = maxOf(0f, xx2 - xx1)
    val h = maxOf(0f, yy2 - yy1)
    val inter = w * h

    val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
    val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)

    return inter / (areaA + areaB - inter)
}

// Reshapes tensor to 2D. Uses YOLO11 output tensor of shape (1, 84, 8400).
// In your app, you may want to dynamically lookup the shape from the
// metadata.yaml file included with your exported model.
fun tensorTo2DPredictions(outputTensor: Tensor): Array<FloatArray> {
    val flat = outputTensor.dataAsFloatArray
    val numBoxes = 8400
    val numAttrs = 84
    val predictions = Array(numBoxes) { FloatArray(numAttrs) }

    for (i in 0 until numBoxes) {
        for (j in 0 until numAttrs) {
            predictions[i][j] = flat[j * numBoxes + i]
        }
    }
    return predictions
}
```

### Embedded Linux

For embedded Linux systems, use the ExecuTorch C++ API:

```cpp
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using namespace ::executorch::extension;

// Load model
Module module("model.pte");

// Prepare input
std::vector<float> input_data = preprocessImage(image);
auto input_tensor = from_blob(input_data.data(), {1, 3, 640, 640});

// Run inference
const auto outputs = module.forward(input_tensor);
```

For more details on integrating ExecuTorch into your applications, visit the [ExecuTorch Documentation](https://docs.pytorch.org/executorch/).

## Performance Optimization

### Model Size Optimization

To reduce model size for deployment:

- **Use Smaller Models**: Start with YOLO26n (nano) for the smallest footprint
- **Lower Input Resolution**: Use smaller image sizes (e.g., `imgsz=320` or `imgsz=416`)
- **Quantization**: Apply quantization techniques (supported in future ExecuTorch versions)

### Inference Speed Optimization

For faster inference:

- **XNNPACK Backend**: The default XNNPACK backend provides optimized CPU inference
- **Hardware Acceleration**: Use platform-specific delegates (e.g., CoreML for iOS)
- **Batch Processing**: Process multiple images when possible

## Benchmarks

The Ultralytics team benchmarked YOLO26 models, comparing speed and accuracy between PyTorch and ExecuTorch.

!!! tip "Performance"

    === "Raspberry Pi 5"

        | Model   | Format      | Status | Size (MB) | metrics/mAP50-95(B) | Inference time (ms/im) |
        | ------- | ----------- | ------ | --------- | ------------------- | ---------------------- |
        | YOLO26n | PyTorch     | ✅     | 5.3       | 0.4790              | 314.80                  |
        | YOLO26n | ExecuTorch  | ✅     | 9.4        | 0.4800              | 142                    |
        | YOLO26s | PyTorch     | ✅     | 19.5       | 0.5730             | 930.90                 |
        | YOLO26s | ExecuTorch  | ✅     | 36.5        | 0.5780              | 376.1                 |

    === "More devices coming soon!"

    Benchmarked with Ultralytics 8.4.9

    !!! note

        Inference time does not include pre/ post-processing.

## Troubleshooting

### Common Issues

**Issue**: `Python version error`

**Solution**: ExecuTorch requires Python 3.10 or higher. Upgrade your Python installation:

```bash
# Using conda
conda create -n executorch python=3.10
conda activate executorch
```

**Issue**: `Export fails during first run`

**Solution**: Ensure you have the latest prebuilt `executorch` wheel installed:

```bash
pip install --upgrade executorch
```

**Issue**: `Import errors for ExecuTorch modules`

**Solution**: Ensure ExecuTorch is properly installed:

```bash
pip install executorch --force-reinstall
```

For more troubleshooting help, visit the [Ultralytics GitHub Issues](https://github.com/ultralytics/ultralytics/issues) or the [ExecuTorch Documentation](https://docs.pytorch.org/executorch/stable/getting-started-setup.html).

## Summary

Exporting YOLO26 models to ExecuTorch format enables efficient deployment on mobile and edge devices. With PyTorch-native integration, cross-platform support, and optimized performance, ExecuTorch is an excellent choice for edge AI applications.

Key takeaways:

- ExecuTorch provides PyTorch-native edge deployment with excellent performance
- Export is simple with `format='executorch'` parameter
- Models are optimized for mobile CPUs via XNNPACK backend
- Supports iOS, Android, and embedded Linux platforms
- Requires Python 3.10-3.13 and PyTorch >= 2.9.0

## FAQ

### How do I export a YOLO26 model to ExecuTorch format?

Export a YOLO26 model to ExecuTorch using either Python or CLI:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="executorch")
```

or

```bash
yolo export model=yolo26n.pt format=executorch
```

### What are the system requirements for ExecuTorch export?

ExecuTorch export requires:

- Python 3.10 or higher
- `executorch` package (install via `pip install executorch`)
- PyTorch (installed automatically with ultralytics)

Note: The `executorch` package ships prebuilt wheels (with the XNNPACK backend), so no extra compilation step is required during export.

### Can I run inference with ExecuTorch models directly in Python?

ExecuTorch models can be loaded directly with `YOLO()` for inference and validation in Python (see the Predict/Validate examples above), and they can also be deployed on mobile and edge devices using the ExecuTorch runtime libraries.

### What platforms are supported by ExecuTorch?

ExecuTorch supports:

- **Mobile**: iOS and Android
- **Embedded Linux**: Raspberry Pi, NVIDIA Jetson, and other ARM devices
- **Desktop**: Linux, macOS, and Windows (for development)

### How does ExecuTorch compare to LiteRT for mobile deployment?

Both ExecuTorch and [LiteRT](litert.md) are excellent for mobile deployment:

- **ExecuTorch**: Better PyTorch integration, native PyTorch workflow, growing ecosystem
- **LiteRT**: More mature, wider hardware support, more deployment examples, and runs the same model on Android, iOS, and the browser

Choose ExecuTorch if you're already using PyTorch and want a native deployment path. Choose LiteRT for maximum compatibility and mature tooling.

### Can I use ExecuTorch models with GPU acceleration?

Yes! ExecuTorch supports hardware acceleration through various backends:

- **Mobile GPU**: Via Vulkan, Metal, or OpenCL delegates
- **NPU/DSP**: Via platform-specific delegates
- **Default**: XNNPACK for optimized CPU inference

Refer to the [ExecuTorch Documentation](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html) for backend-specific setup.
