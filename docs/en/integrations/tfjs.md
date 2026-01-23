---
comments: true
description: TensorFlow.js export is deprecated. Use ONNX format with ONNX Runtime Web for browser deployment instead.
keywords: YOLO, TensorFlow.js, TF.js, ONNX, ONNX Runtime Web, browser ML, WebGPU, model export, deprecated
---

# TensorFlow.js Export (DEPRECATED)

!!! warning "Deprecation Notice"

    **TensorFlow.js export is deprecated and no longer supported.**

    For browser and Node.js deployment, use **ONNX format with ONNX Runtime Web** instead, which offers:

    - **Better performance**: WebGPU acceleration (~20x faster than CPU)
    - **No TensorFlow dependency**: Pure JavaScript/WebAssembly runtime
    - **Active development**: Microsoft actively maintains ONNX Runtime Web
    - **Already supported**: Ultralytics ONNX export works out of the box

    ```bash
    # Export to ONNX for web deployment
    yolo export model=yolo11n.pt format=onnx
    ```

## Migration to ONNX Runtime Web

### Step 1: Export to ONNX

```bash
yolo export model=yolo11n.pt format=onnx
```

### Step 2: Use ONNX Runtime Web in Browser

```javascript
import * as ort from "onnxruntime-web";

// Load model
const session = await ort.InferenceSession.create("yolo11n.onnx", {
    executionProviders: ["webgpu", "webgl", "wasm"], // Falls back automatically
});

// Run inference
const feeds = { images: inputTensor };
const results = await session.run(feeds);
```

### NPM Installation

```bash
npm install onnxruntime-web
```

## Why ONNX Runtime Web?

| Feature           | TensorFlow.js (old) | ONNX Runtime Web (recommended)  |
| ----------------- | ------------------- | ------------------------------- |
| Export dependency | Full TensorFlow     | None (uses existing ONNX)       |
| GPU backends      | WebGL, WASM         | **WebGPU**, WebGL, WASM, WebNN  |
| Performance       | Good                | **Better** (WebGPU ~20x faster) |
| Maintenance       | Google              | Microsoft (active development)  |
| YOLO support      | Required conversion | Direct ONNX loading             |

## Example Projects

Several production-ready implementations exist:

- [yolo-object-detection-onnxruntime-web](https://github.com/nomi30701/yolo-object-detection-onnxruntime-web) - WebGPU/WASM support
- [yolov8-onnxruntime-web](https://github.com/Hyuto/yolov8-onnxruntime-web) - YOLOv8 implementation
- [ONNX Runtime Web Tutorial](https://onnxruntime.ai/docs/tutorials/web/) - Official documentation

## Browser Compatibility

| Browser      | WebGPU  | WebGL | WASM |
| ------------ | ------- | ----- | ---- |
| Chrome 113+  | Yes     | Yes   | Yes  |
| Edge 113+    | Yes     | Yes   | Yes  |
| Firefox 141+ | Yes     | Yes   | Yes  |
| Safari 18+   | Partial | Yes   | Yes  |

## Related Resources

- [ONNX Export Guide](../modes/export.md)
- [ONNX Runtime Web Documentation](https://onnxruntime.ai/docs/tutorials/web/)
- [WebGPU Tutorial](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)

## FAQ

### Why was TensorFlow.js export deprecated?

TensorFlow.js export required the full TensorFlow Python package, which conflicted with the new `ai-edge-torch` library used for TFLite exports. ONNX Runtime Web provides better performance and requires no additional Python dependencies since ONNX export is already supported.

### Can I still use TensorFlow.js with YOLO models?

Yes, but you'll need to manually convert the ONNX model to TensorFlow.js format outside of Ultralytics. However, we recommend using ONNX Runtime Web directly for better performance and simpler workflow.

### What about existing TensorFlow.js models?

Existing TensorFlow.js models will continue to work in browsers. This deprecation only affects the export pipeline - you can no longer create new TF.js exports from Ultralytics.
