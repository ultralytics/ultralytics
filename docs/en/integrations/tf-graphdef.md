---
comments: true
description: Learn about TF GraphDef format (deprecated). For TensorFlow deployment, use TFLite format instead.
keywords: YOLO, TensorFlow, GraphDef, TFLite, model deployment, deprecated, machine learning, AI, computer vision
---

# TF GraphDef Export (DEPRECATED)

!!! warning "Deprecation Notice"

    **TensorFlow GraphDef (.pb) export is deprecated and no longer supported.**

    The GraphDef format is a legacy TensorFlow 1.x format. With the migration to Google's `ai-edge-torch` library for TensorFlow exports, this format is no longer available.

    **Please use TensorFlow Lite (TFLite) format instead:**

    ```bash
    yolo export model=yolo11n.pt format=tflite
    ```

    See the [TFLite integration guide](../integrations/tflite.md) for more information.

## Migration Guide

If you were previously using TF GraphDef format, here's how to migrate:

| Previous (Deprecated) | New (Recommended)                            |
| --------------------- | -------------------------------------------- |
| `format="pb"`         | `format="tflite"`                            |
| `yolo11n.pb`          | `yolo11n_saved_model/yolo11n_float32.tflite` |

### Updated Export Example

!!! example "Export to TFLite"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO model
        model = YOLO("yolo11n.pt")

        # Export to TFLite format (recommended)
        model.export(format="tflite")

        # Load and run inference
        tflite_model = YOLO("yolo11n_saved_model/yolo11n_float32.tflite")
        results = tflite_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export to TFLite format
        yolo export model=yolo11n.pt format=tflite

        # Run inference
        yolo predict model=yolo11n_saved_model/yolo11n_float32.tflite source=https://ultralytics.com/images/bus.jpg
        ```

## Why TFLite Instead of GraphDef?

TensorFlow Lite offers several advantages over the legacy GraphDef format:

- **Direct PyTorch conversion**: Uses Google's `ai-edge-torch` for direct PyTorch to TFLite conversion
- **Better mobile support**: Optimized for mobile and embedded deployment
- **Simpler dependency tree**: No longer requires `onnx2tf` and its dependencies
- **Active development**: TFLite is actively maintained by Google
- **Hardware acceleration**: Better support for Edge TPU, GPU delegates, and NPUs

## Inference with Existing .pb Files

If you have existing .pb files, they can still be loaded for inference:

```python
from ultralytics import YOLO

# Load existing .pb file for inference only
model = YOLO("existing_model.pb")
results = model("image.jpg")
```

However, creating new .pb exports is no longer supported.

## Related Resources

- [TFLite Integration Guide](../integrations/tflite.md)
- [Edge TPU Integration Guide](../integrations/edge-tpu.md)
- [TensorFlow.js Integration Guide](../integrations/tfjs.md)
- [Export Mode Documentation](../modes/export.md)

## FAQ

### Why was TF GraphDef export deprecated?

The GraphDef (.pb) format is a legacy TensorFlow 1.x format. Ultralytics has migrated to using Google's `ai-edge-torch` library, which provides direct PyTorch to TFLite conversion without requiring the complex `onnx2tf` dependency chain. This results in a cleaner, more reliable export process.

### What should I use instead of TF GraphDef?

Use TensorFlow Lite (TFLite) format: `yolo export model=yolo11n.pt format=tflite`. TFLite is the recommended format for TensorFlow-based deployment and offers better mobile/embedded device support.

### Can I still run inference on existing .pb files?

Yes, existing .pb files can still be loaded for inference using `YOLO("model.pb")`. However, creating new .pb exports is no longer supported.

### What about TensorFlow.js export?

TensorFlow.js export (`format=tfjs`) is still supported and now uses TFLite as an intermediate format instead of GraphDef.
