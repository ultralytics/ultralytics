---
comments: true
description: Export Ultralytics YOLO11 to ONNX, TensorRT, CoreML, TFLite, and more. Deployment-ready guide with CLI and Python examples for every major format.
keywords: YOLO11, export, ONNX, TensorRT, CoreML, TFLite, model deployment, Ultralytics, edge inference
canonical: https://docs.ultralytics.com/models/yolo11/tutorials/model-export/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Export YOLO11 to ONNX, TensorRT, CoreML, and More",
  "description": "Export Ultralytics YOLO11 to ONNX, TensorRT, CoreML, TFLite, and more. Deployment-ready guide with CLI and Python examples for every major format.",
  "url": "https://docs.ultralytics.com/models/yolo11/tutorials/model-export/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-export.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo11/tutorials/model-export/"
}
</script>

# Export YOLO11 to ONNX, TensorRT, CoreML, and More

<!-- NOTE FOR MURAT: Please verify the full list of export formats supported by YOLO11 and confirm any format-specific limitations. Add measured inference speed benchmarks per format on representative hardware (e.g. RTX 4090 for TensorRT, iPhone 15 for CoreML, Raspberry Pi 5 for TFLite). Confirm `imgsz`, `half`, and `dynamic` flags work as expected. Add screenshots of successful export terminal output if available. -->

Once you have a trained [Ultralytics YOLO11](../../../models/yolo11.md) model, the next step is deploying it. Exporting converts the PyTorch `.pt` checkpoint into a self-contained format optimised for your target hardware or runtime. YOLO11 supports all standard Ultralytics export targets across cloud, edge, and mobile deployments.

## Supported Export Formats

| Format | CLI `format` arg | File suffix | Notes |
|---|---|---|---|
| PyTorch | — | `.pt` | Default checkpoint; used for further training |
| ONNX | `onnx` | `.onnx` | Portable; works with ONNX Runtime, OpenVINO |
| TensorRT | `engine` | `.engine` | Fastest on NVIDIA GPUs; requires TensorRT SDK |
| CoreML | `coreml` | `.mlpackage` | Apple Silicon / iOS / macOS deployment |
| TFLite | `tflite` | `.tflite` | Android, microcontrollers, Coral edge TPU |
| TF SavedModel | `saved_model` | `saved_model/` | TensorFlow Serving and TF-Lite conversion |
| TF.js | `tfjs` | `_web_model/` | In-browser inference via TensorFlow.js |
| OpenVINO | `openvino` | `_openvino_model/` | Intel CPU/VPU acceleration |
| PaddlePaddle | `paddle` | `_paddle_model/` | Baidu ecosystem deployment |
| NCNN | `ncnn` | `_ncnn_model/` | Mobile-optimised C++ runtime (Tencent) |

## Export Commands

=== "CLI"

    ```bash
    # Export to ONNX (default opset 17)
    yolo export model=yolo11n.pt format=onnx

    # Export to TensorRT FP16 (requires NVIDIA GPU + TensorRT)
    yolo export model=yolo11n.pt format=engine half=True

    # Export to CoreML for Apple Silicon
    yolo export model=yolo11n.pt format=coreml

    # Export to TFLite INT8 (quantised)
    yolo export model=yolo11n.pt format=tflite int8=True

    # Export with custom image size
    yolo export model=yolo11n.pt format=onnx imgsz=640
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    # Export to ONNX
    model.export(format="onnx")

    # Export to TensorRT FP16
    model.export(format="engine", half=True)

    # Export to CoreML
    model.export(format="coreml")

    # Export to TFLite INT8
    model.export(format="tflite", int8=True)

    # Dynamic axes ONNX (variable batch size)
    model.export(format="onnx", dynamic=True)
    ```

## Export Arguments

| Argument | Default | Description |
|---|---|---|
| `format` | `"onnx"` | Target export format (see table above) |
| `imgsz` | `640` | Input image size (single int or `[h, w]`) |
| `half` | `False` | FP16 quantisation — TensorRT and ONNX |
| `int8` | `False` | INT8 quantisation — TFLite, OpenVINO, TensorRT |
| `dynamic` | `False` | Dynamic batch/spatial axes — ONNX, TensorRT |
| `simplify` | `True` | Simplify ONNX graph with `onnxsim` |
| `opset` | `None` | ONNX opset version (default: latest supported) |
| `batch` | `1` | Static batch size for export |
| `device` | `None` | Device for export (e.g. `0` for GPU) |

## Deployment Tips by Format

### ONNX

ONNX is the most portable format and the best starting point if you are unsure which runtime you will use. Use `dynamic=True` when your inference service needs variable batch sizes or image dimensions.

!!! tip "ONNX Runtime inference"

    ```python
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession("yolo11n.onnx")
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.random.rand(1, 3, 640, 640).astype(np.float32)})
    ```

### TensorRT

TensorRT delivers the lowest latency on NVIDIA hardware. Always export with `half=True` on Ampere+ GPUs for maximum throughput. The exported `.engine` file is tied to the specific GPU architecture and TensorRT version — rebuild when upgrading hardware.

!!! warning "Architecture-specific engines"

    A TensorRT engine compiled on an RTX 4090 will not run on a T4. Export on the same hardware you intend to deploy to.

### CoreML

CoreML is required for native iOS and macOS deployment via the `Vision` framework or `CoreML` API. Export produces a `.mlpackage` directory. Use Xcode to integrate into your app.

### TFLite

TFLite is the standard format for Android and embedded Linux devices. Use `int8=True` for microcontrollers or when memory is constrained; this requires a representative calibration dataset to be passed via the `data` argument.

## Run Inference After Export

After exporting, you can run inference directly with the Ultralytics CLI on the exported file:

```bash
# ONNX inference
yolo predict model=yolo11n.onnx source=image.jpg

# TensorRT inference
yolo predict model=yolo11n.engine source=image.jpg
```

## See Also

- [YOLO11 Model Overview](../../../models/yolo11.md)
- [Ultralytics Export Mode Docs](../../../modes/export.md)
- [Train YOLO11 on a Custom Dataset](train-custom-dataset.md)
