---
comments: true
description: TensorFlow SavedModel export is deprecated. Use TFLite format with ai-edge-torch for TensorFlow deployment instead.
keywords: YOLO, TensorFlow SavedModel, TFLite, ai-edge-torch, model export, deprecated
---

# TensorFlow SavedModel Export (DEPRECATED)

!!! warning "Deprecation Notice"

    **TensorFlow SavedModel export is deprecated and no longer supported.**

    For TensorFlow/TFLite deployment, use **TFLite format** directly instead, which offers:

    - **Direct conversion**: PyTorch to TFLite without intermediate formats
    - **Simpler dependencies**: Uses Google's ai-edge-torch library
    - **Better performance**: Optimized for edge deployment
    - **Active development**: ai-edge-torch is actively maintained by Google

    ```bash
    # Export to TFLite for TensorFlow deployment
    yolo export model=yolo11n.pt format=tflite
    ```

## Migration to TFLite

### Step 1: Export to TFLite

```bash
yolo export model=yolo11n.pt format=tflite
```

Or in Python:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="tflite")
```

### Step 2: Use TFLite for Inference

```python
from ultralytics import YOLO

# Load TFLite model
model = YOLO("yolo11n.tflite")

# Run inference
results = model("https://ultralytics.com/images/bus.jpg")
```

## Why was TensorFlow SavedModel deprecated?

TensorFlow SavedModel export was deprecated because:

1. **Dependency conflicts**: The previous `onnx2tf` converter had heavy dependencies and version conflicts
2. **Simpler path available**: ai-edge-torch provides direct PyTorch to TFLite conversion
3. **TFLite is sufficient**: For most edge deployment use cases, TFLite provides everything needed
4. **Better maintenance**: ai-edge-torch is officially maintained by Google

## What about TensorFlow Serving?

For TensorFlow Serving deployment, consider these alternatives:

- **ONNX Runtime**: Use ONNX export with ONNX Runtime Server
- **TensorRT**: For NVIDIA GPU deployment
- **OpenVINO**: For Intel hardware deployment

## Related Resources

- [TFLite Export Guide](tflite.md)
- [Edge TPU Guide](edge-tpu.md)
- [Export Modes Documentation](../modes/export.md)
