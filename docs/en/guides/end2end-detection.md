---
comments: true
description: Learn how YOLO26 end-to-end NMS-free detection works, what changes for your deployment pipeline, which export formats support it, and how to migrate from YOLOv8 or YOLO11.
keywords: YOLO26, end-to-end detection, NMS-free inference, model export, deployment guide, object detection, Ultralytics, YOLOv8 migration, YOLO11 migration, ONNX, TensorRT, CoreML, post-processing, computer vision
---

# Understanding End-to-End Detection in Ultralytics YOLO26

## Introduction

If you're upgrading to [YOLO26](../models/yolo26.md) from an earlier model like [YOLOv8](../models/yolov8.md) or [YOLO11](../models/yolo11.md), one of the biggest changes you'll notice is the removal of [Non-Maximum Suppression](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) (NMS). Traditional YOLO models produce thousands of overlapping predictions that need a separate NMS post-processing step to filter down to final detections. This adds latency, complicates export graphs, and can behave inconsistently across different hardware platforms.

YOLO26 takes a different approach. It outputs final detections directly from the model — no external filtering required. This is known as **end-to-end [object detection](https://www.ultralytics.com/glossary/object-detection)**, and it's enabled by default in all YOLO26 models. The result is a simpler deployment pipeline, lower latency, and up to **43% faster [inference](https://www.ultralytics.com/glossary/inference-time) on CPUs**.

This guide walks you through what changed, whether you need to update your code, which export formats support end-to-end inference, and how to migrate smoothly from older YOLO models.

For a deeper look at the motivation behind this architectural shift, see the [Ultralytics blog post on why YOLO26 removes NMS](https://www.ultralytics.com/blog/why-ultralytics-yolo26-removes-nms-and-how-that-changes-deployment).

!!! summary "Quick Summary"

    - **Using the Ultralytics API or CLI?** No changes needed — just swap your model name to `yolo26n.pt`.
    - **Using custom inference code (ONNX Runtime, TensorRT, etc.)?** Update your post-processing — output is now `(N, 300, 6)` in `[x1, y1, x2, y2, conf, class_id]` format, no NMS required.
    - **Exporting?** Most formats support end-to-end. A few (NCNN, RKNN, PaddlePaddle, ExecuTorch, IMX, Edge TPU) auto-fallback to traditional output.

## How End-to-End Detection Works

YOLO26 uses a **dual-head architecture** during [training](../modes/train.md). Both heads share the same backbone and neck, but produce outputs in different ways:

| Head | Purpose | Output Shape | Post-Processing |
| --- | --- | --- | --- |
| **One-to-One** (default) | End-to-end inference | `(N, 300, 6)` | Confidence threshold only |
| **One-to-Many** | Traditional YOLO output | `(N, nc + 4, 8400)` | Requires NMS |

During training, both heads run simultaneously — the one-to-many head provides a richer learning signal, while the one-to-one head learns to produce clean, non-overlapping predictions. During [inference](../modes/predict.md) and [export](../modes/export.md), only the **one-to-one head** is active by default, producing up to 300 detections per image in the format `[x1, y1, x2, y2, confidence, class_id]`.

When you call `model.fuse()`, the one-to-many head is automatically removed, reducing model size and FLOPs. For more details on the dual-head architecture, see the [YOLO26 model page](../models/yolo26.md).

## Do I Need to Change My Code?

### Using the Ultralytics Python API or CLI

**No changes needed.** If you use the standard [Ultralytics Python API](../usage/python.md) or [CLI](../usage/cli.md), everything works automatically — [prediction](../modes/predict.md), [validation](../modes/val.md), and [export](../modes/export.md) all handle end-to-end models out of the box.

!!! example "No code changes required with the Ultralytics API"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Predict — no NMS step, no code changes
        results = model.predict("image.jpg")
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n.pt source=image.jpg
        ```

### Using Custom Inference Code

**Yes, the output format is different.** If you wrote custom post-processing logic for [YOLOv8](../models/yolov8.md) or [YOLO11](../models/yolo11.md) (for example, when running inference with [ONNX Runtime](../integrations/onnx.md) or [TensorRT](../integrations/tensorrt.md)), you'll need to update it to handle the new output shape:

| | YOLOv8 / YOLO11 | YOLO26 (end-to-end) |
| --- | --- | --- |
| **Output shape** | `(N, nc+4, 8400)` | `(N, 300, 6)` |
| **Format** | Box coords + class scores per anchor | `[x1, y1, x2, y2, conf, class_id]` |
| **NMS required** | Yes | No |
| **Post-processing** | NMS + confidence filter | Confidence filter only |

Where `nc` is the number of classes (e.g., 80 for [COCO](https://www.ultralytics.com/glossary/coco-dataset)).

With end-to-end models, post-processing becomes much simpler — for example, when using [ONNX Runtime](../integrations/onnx.md):

```python
import onnxruntime as ort

# Load and run the exported end-to-end model
session = ort.InferenceSession("yolo26n.onnx")
output = session.run(None, {session.get_inputs()[0].name: input_tensor})

# End-to-end output: (batch, 300, 6) → [x1, y1, x2, y2, confidence, class_id]
detections = output[0][0]  # first image in batch
detections = detections[detections[:, 4] > conf_threshold]  # confidence filter — that's it!
```

### Switching to the One-to-Many Head

If you need the traditional YOLO output format (for example, to reuse existing NMS-based post-processing code), you can switch to the one-to-many head at any time by setting `end2end=False`:

!!! example "Using the one-to-many head for traditional NMS-based output"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Prediction with NMS (traditional behavior)
        results = model.predict("image.jpg", end2end=False)

        # Validation with NMS
        metrics = model.val(data="coco.yaml", end2end=False)

        # Export without end-to-end
        model.export(format="onnx", end2end=False)
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n.pt source=image.jpg end2end=False
        yolo val model=yolo26n.pt data=coco.yaml end2end=False
        yolo export model=yolo26n.pt format=onnx end2end=False
        ```

## Export Format Compatibility

Most [export formats](../modes/export.md#export-formats) support end-to-end inference out of the box, including [ONNX](../integrations/onnx.md), [TensorRT](../integrations/tensorrt.md), [CoreML](../integrations/coreml.md), [OpenVINO](../integrations/openvino.md), [TF Lite](../integrations/tflite.md), [TF.js](../integrations/tfjs.md), and [MNN](../integrations/mnn.md).

The following formats **do not** support end-to-end and automatically fall back to the one-to-many head: [NCNN](../integrations/ncnn.md), RKNN, [PaddlePaddle](../integrations/paddlepaddle.md), ExecuTorch, IMX, and [Edge TPU](../integrations/edge-tpu.md).

!!! tip "What happens when end-to-end isn't supported"

    When you export to one of these formats, Ultralytics automatically switches to the one-to-many head and logs a warning — no manual intervention needed. This means **you'll need NMS in your inference pipeline** for these formats, just like with [YOLOv8](../models/yolov8.md) or [YOLO11](../models/yolo11.md).

!!! note "TensorRT + INT8"

    [TensorRT](../integrations/tensorrt.md) supports end-to-end, but it is **auto-disabled** when exporting with `int8=True` on TensorRT ≤10.3.0.

## Accuracy and Speed Tradeoffs

End-to-end detection provides significant deployment benefits with minimal impact on [accuracy](https://www.ultralytics.com/glossary/accuracy):

| Metric | End-to-End (default) | One-to-Many + NMS (`end2end=False`) |
| --- | --- | --- |
| **CPU Inference Speed** | Up to **43% faster** | Baseline |
| **mAP Impact** | ~0.5 mAP lower | Matches or exceeds YOLO11 |
| **Post-Processing** | Confidence filter only | Full NMS pipeline |
| **Deployment Complexity** | Minimal | Requires NMS implementation |

For most real-world applications, the ~0.5 [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) difference is negligible, especially when considering the speed and simplicity gains. If maximum accuracy is your top priority, you can always fall back to the one-to-many head using `end2end=False`.

See the [YOLO26 performance metrics](../models/yolo26.md#performance-metrics) for detailed benchmarks across all model sizes (n, s, m, l, x).

## Migrating from YOLOv8 or YOLO11

If you're upgrading an existing project to YOLO26, here's a quick checklist to ensure a smooth transition:

- **Ultralytics API / CLI users:** No changes needed — just update the model name to `yolo26n.pt` (or your preferred size)
- **Custom post-processing code:** Update to handle `(N, 300, 6)` output shape instead of `(N, nc+4, 8400)` + NMS
- **Export pipelines:** Check the [format compatibility](#export-format-compatibility) section above for your target format
- **TensorRT + INT8:** Verify your TensorRT version is >10.3.0 for end-to-end support
- **FP16 exports:** If you need all outputs in FP16, export with `end2end=False` — see [why output0 stays FP32](../modes/export.md#why-is-output0-fp32-when-exporting-with-halftrue-and-end2endtrue)
- **iOS / CoreML:** End-to-end is fully supported — no changes needed for CoreML deployments
- **Edge devices (NCNN, RKNN):** These formats auto-fallback to one-to-many, so include NMS in your on-device pipeline

## FAQ

### Can I use end2end=True and nms=True together?

No. These options are mutually exclusive. If you set `nms=True` on an end-to-end model during [export](../modes/export.md), it will be automatically forced to `nms=False` with a warning. The end-to-end head already handles duplicate filtering internally, so external NMS is unnecessary.

### What does the max_det parameter control in end-to-end models?

The `max_det` parameter (default: 300) sets the maximum number of detections the one-to-one head can output per image. If your use case involves scenes with many objects, you can increase it. For simpler scenes, reducing it may slightly improve speed:

```python
model.predict("image.jpg", max_det=100)  # fewer detections, slightly faster
model.predict("image.jpg", max_det=500)  # more detections for dense scenes
```

### My exported ONNX model outputs (1, 300, 6) — is that correct?

Yes, that's the expected end-to-end output format: [batch size](https://www.ultralytics.com/glossary/batch-size) of 1, up to 300 detections, each with 6 values `[x1, y1, x2, y2, confidence, class_id]`. Simply filter by confidence threshold and you're done — no NMS needed.

### How do I check if my exported model is end-to-end?

You can inspect the ONNX model metadata directly to verify whether end-to-end inference is enabled:

```python
import onnxruntime as ort

session = ort.InferenceSession("yolo26n.onnx")
metadata = session.get_modelmeta().custom_metadata_map
print(metadata.get("end2end"))  # 'True' if end-to-end is enabled
```

Alternatively, check the output shape — end-to-end models output `(1, 300, 6)`, while traditional models output `(1, nc+4, 8400)`.

### Is end-to-end detection supported for segmentation, pose, and OBB tasks?

Yes. All YOLO26 task variants — [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [pose estimation](../tasks/pose.md), and [oriented object detection (OBB)](../tasks/obb.md) — support end-to-end inference by default. The `end2end=False` fallback is available across all tasks as well.
