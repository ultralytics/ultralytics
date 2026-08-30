---
title: YOLO26 End-to-End NMS-Free Detection
comments: true
description: Learn how YOLO26 end-to-end NMS-free detection works, what changes in your deployment pipeline, which export formats support it, and how to migrate.
keywords: YOLO26, end-to-end detection, NMS-free inference, model export, deployment guide, object detection, Ultralytics, YOLOv8 migration, YOLO11 migration, ONNX, TensorRT, CoreML, post-processing, computer vision
---

# Understanding End-to-End Detection in Ultralytics YOLO26

[YOLO26 detection-style models](../models/yolo26.md) — detection, segmentation, pose, and OBB — are **NMS-free** by default: they output final detections directly from the model, with no [Non-Maximum Suppression](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) (NMS) post-processing step. Earlier models like [YOLOv8](../models/yolov8.md) and [YOLO11](../models/yolo11.md) produce thousands of overlapping predictions that a separate NMS step has to filter down, which adds latency, complicates export graphs, and can behave inconsistently across hardware platforms.

This is known as **end-to-end [object detection](https://www.ultralytics.com/glossary/object-detection)**, and it is enabled by default. The result is a simpler deployment pipeline and lower latency — YOLO26n runs up to **43% faster than YOLO11n** on CPU ONNX inference (Intel Xeon CPU @ 2.00 GHz).

This guide walks you through what changed, whether you need to update your code, which export formats support end-to-end inference, and how to migrate smoothly from older YOLO models.

For a deeper look at the motivation behind this architectural shift, see the [Ultralytics blog post on why YOLO26 removes NMS](https://www.ultralytics.com/blog/why-ultralytics-yolo26-removes-nms-and-how-that-changes-deployment).

!!! summary "Quick Summary"

    - **Using the Ultralytics API or CLI?** No changes needed — just swap your model name to `yolo26n.pt`.
    - **Using custom inference code (ONNX Runtime, TensorRT, etc.)?** Update your post-processing — detection output is now `(N, 300, 6)` in `xyxy` format, no NMS required. Other tasks append extra data (mask coefficients, keypoints, or angle).
    - **Exporting?** Most formats keep end-to-end output natively; a few fall back to traditional output, and quantization can disable it — see [Export Format Compatibility](#export-format-compatibility).

## How End-to-End Detection Works

YOLO26 uses a **dual-head architecture** during [training](../modes/train.md). Both heads share the same backbone and neck, but produce outputs in different ways:

| Head                     | Purpose                 | Detection Output    | Post-Processing           |
| ------------------------ | ----------------------- | ------------------- | ------------------------- |
| **One-to-One** (default) | End-to-end inference    | `(N, 300, 6)`       | Confidence threshold only |
| **One-to-Many**          | Traditional YOLO output | `(N, nc + 4, 8400)` | Requires NMS              |

The shapes above are for [detection](../tasks/detect.md), where `N` is the [batch size](https://www.ultralytics.com/glossary/batch-size), `nc` is the number of classes (e.g., 80 for [COCO](../datasets/detect/coco.md)), and the 8400 anchor count is the value at `imgsz=640`. Other tasks extend the one-to-one output with additional data per detection:

| Task                                         | End-to-End Output                          | Extra Data                          |
| -------------------------------------------- | ------------------------------------------ | ----------------------------------- |
| [Detection](../tasks/detect.md)              | `(N, 300, 6)`                              | —                                   |
| [Instance Segmentation](../tasks/segment.md) | `(N, 300, 6 + nm)` + proto `(N, nm, H, W)` | `nm` mask coefficients (default 32) |
| [Pose](../tasks/pose.md)                     | `(N, 300, 57)`                             | 17 keypoints × 3 (x, y, visibility) |
| [OBB](../tasks/obb.md)                       | `(N, 300, 7)`                              | Rotation angle                      |

During training, both heads run simultaneously — the one-to-many head provides a richer learning signal, while the one-to-one head learns to produce clean, non-overlapping predictions. During [inference](../modes/predict.md) and [export](../modes/export.md), only the **one-to-one head** is active by default, producing up to 300 detections per image in the format `[x1, y1, x2, y2, confidence, class_id]`.

When you call `model.fuse()`, it folds Conv + BatchNorm layers for faster inference and, on end-to-end models, also removes the one-to-many head — reducing model size and FLOPs. For more details on the dual-head architecture, see the [YOLO26 model page](../models/yolo26.md).

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

|                      | YOLOv8 / YOLO11                            | YOLO26 (end-to-end)                                             |
| -------------------- | ------------------------------------------ | --------------------------------------------------------------- |
| **Detection output** | `(N, nc + 4, 8400)`                        | `(N, 300, 6)`                                                   |
| **Box format**       | `xywh` (center x, center y, width, height) | `xyxy` (top-left x, top-left y, bottom-right x, bottom-right y) |
| **Layout**           | Box coords + class scores per anchor       | `[x1, y1, x2, y2, conf, class_id]`                              |
| **NMS required**     | Yes                                        | No                                                              |
| **Post-processing**  | NMS + confidence filter                    | Confidence filter only                                          |

For [segmentation](../tasks/segment.md), [pose](../tasks/pose.md), and [OBB](../tasks/obb.md) tasks, YOLO26 appends task-specific data to each detection — see the [output shapes table](#how-end-to-end-detection-works).

With end-to-end models, post-processing becomes much simpler — for example, when using [ONNX Runtime](../integrations/onnx.md):

```python
import onnxruntime as ort

# Load and run the exported end-to-end model
session = ort.InferenceSession("yolo26n.onnx")
output = session.run(None, {session.get_inputs()[0].name: input_tensor})

# End-to-end output: (batch, 300, 6) → [x1, y1, x2, y2, confidence, class_id]
detections = output[0][0]  # first image in batch
detections = detections[detections[:, 4] > 0.25]  # confidence filter, no NMS
```

### Switching to the One-to-Many Head

If you need the traditional YOLO output format (for example, to reuse existing NMS-based post-processing code), you can switch to the one-to-many head when it is available by setting `end2end=False`:

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

Most [export formats](../modes/export.md#export-formats) support end-to-end inference out of the box, including [ONNX](../integrations/onnx.md), [TensorRT](../integrations/tensorrt.md), [CoreML](../integrations/coreml.md), [OpenVINO](../integrations/openvino.md), [LiteRT](../integrations/litert.md), and [MNN](../integrations/mnn.md).

The following formats **do not** support end-to-end and automatically fall back to the one-to-many head: [NCNN](../integrations/ncnn.md), [RKNN](../integrations/rockchip-rknn.md), [PaddlePaddle](../integrations/paddlepaddle.md), [ExecuTorch](../integrations/executorch.md), [IMX](../integrations/sony-imx500.md), [Edge TPU](../integrations/edge-tpu.md), and [Qualcomm QNN](../integrations/qnn.md).

!!! tip "What happens when end-to-end isn't supported"

    When you export to one of these formats, Ultralytics automatically switches to the one-to-many head and logs a warning. This means **you'll need NMS in your inference pipeline** for these formats, just like with [YOLOv8](../models/yolov8.md) or [YOLO11](../models/yolo11.md).

For [Hailo](../integrations/hailo.md), the exporter picks the output path from the loaded head rather than from an argument, so `end2end` is rejected if passed: a default YOLO26 detection model keeps its NMS-free one-to-one outputs, while a checkpoint whose head is already `end2end=False` compiles the traditional path with HailoRT NMS.

!!! note "Quantization and runtime version can disable end-to-end"

    [TensorRT](../integrations/tensorrt.md) and [LiteRT](../integrations/litert.md) support end-to-end, but the branch is **auto-disabled** on TensorRT older than 8.5.0, on TensorRT 10.3.0 with `quantize=8` on JetPack 6, and on LiteRT with `quantize=8` or `quantize="w8a16"`. Each case logs a warning and exports the one-to-many head.

## Accuracy and Speed Trade-offs

End-to-end detection provides significant deployment benefits with minimal impact on [accuracy](https://www.ultralytics.com/glossary/accuracy):

| Metric                     | End-to-End (default)   | One-to-Many + NMS (`end2end=False`) |
| -------------------------- | ---------------------- | ----------------------------------- |
| **COCO mAP<sup>val</sup>** | 0.6-0.8 lower          | Baseline                            |
| **Post-Processing**        | Confidence filter only | Full NMS pipeline                   |
| **Deployment Complexity**  | Minimal                | Requires NMS implementation         |

Across the five detection scales the one-to-one head costs 0.6-0.8 [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map) on COCO — 40.9 to 40.1 for YOLO26n and 57.5 to 56.9 for YOLO26x — in exchange for dropping the NMS pass entirely. If maximum accuracy is your priority, fall back to the one-to-many head with `end2end=False`.

See the [YOLO26 performance metrics](../models/yolo26.md#performance-metrics) for detailed benchmarks across all model sizes (n, s, m, l, x).

## Migrating from YOLOv8 or YOLO11

If you're upgrading an existing project to YOLO26:

- **Ultralytics API / CLI users:** No changes needed — just update the model name to `yolo26n.pt` (or `yolo26n-seg.pt`, `yolo26n-pose.pt`, `yolo26n-obb.pt`)
- **Custom post-processing code:** Update to handle the new output shapes — `(N, 300, 6)` for detection, plus task-specific data for [segmentation](../tasks/segment.md), [pose](../tasks/pose.md), and [OBB](../tasks/obb.md). Also note the box format change from `xywh` to `xyxy`
- **Export pipelines:** Check the [format compatibility](#export-format-compatibility) section for your target format
- **TensorRT below 8.5.0:** end-to-end is disabled at every precision — upgrade TensorRT to 8.5.0 or later to keep it
- **Quantized exports:** TensorRT 10.3.0 with `quantize=8` on JetPack 6, and LiteRT with `quantize=8` or `quantize="w8a16"`, auto-disable end-to-end — export at a higher precision to keep it
- **FP16 exports:** If you need all outputs in FP16, export with `end2end=False` — see [why output0 stays FP32](../modes/export.md#why-is-output0-fp32-when-exporting-quantized-models-with-end2endtrue)
- **iOS / CoreML:** End-to-end is fully supported. If you need Xcode Preview support, use `end2end=False` with `nms=True`
- **Edge devices (NCNN, RKNN):** These formats auto-fallback to one-to-many, so include NMS in your on-device pipeline

## Conclusion

End-to-end detection is the default in YOLO26 and needs no code changes if you use the [Ultralytics Python API](../usage/python.md) or [CLI](../usage/cli.md). Only custom post-processing pipelines need updating to read the new `(N, 300, 6)` output and drop the NMS step — except for export formats that fall back to one-to-many output (such as NCNN and RKNN), which still require NMS on-device. For detailed speed and accuracy benchmarks across all model sizes, see the [YOLO26 model page](../models/yolo26.md), and for the full set of export options and formats, see the [Export mode](../modes/export.md) documentation.

## FAQ

### Can I use end2end=True and nms=True together?

No. These options are mutually exclusive. If you set `nms=True` on an end-to-end model during [export](../modes/export.md), it will be automatically forced to `nms=False` with a warning. The end-to-end head already handles duplicate filtering internally, so external NMS is unnecessary.

However, `end2end=False` combined with `nms=True` is a valid configuration — it bakes traditional NMS into the export graph. This can be useful for [CoreML](../integrations/coreml.md) exports because it lets you use the Preview function in Xcode with the detection model directly.

### What does the max_det parameter control in end-to-end models?

The [`max_det`](../modes/predict.md#inference-arguments) parameter (default: 300) sets the maximum number of detections returned per image. You can adjust it at inference or export time:

```python
model.predict("image.jpg", max_det=100)  # fewer detections
model.export(format="onnx", max_det=500)  # more detections for dense scenes
```

The value is baked into an exported graph as the head's top-k, so `max_det=500` widens the output tensor to up to `(1, 500, 6)`, capped by the anchor count.

### My exported ONNX model outputs (1, 300, 6) — is that correct?

Yes, that's the expected end-to-end output format for detection: [batch size](https://www.ultralytics.com/glossary/batch-size) of 1, up to 300 detections, each with 6 values `[x1, y1, x2, y2, confidence, class_id]`. Simply filter by confidence threshold — no NMS needed.

For other tasks, the output shape differs:

| Task         | Output Shape                         | Description                                                       |
| ------------ | ------------------------------------ | ----------------------------------------------------------------- |
| Detection    | `(1, 300, 6)`                        | `[x1, y1, x2, y2, conf, class_id]`                                |
| Segmentation | `(1, 300, 38)` + `(1, 32, 160, 160)` | 6 box values + 32 mask coefficients, plus a prototype mask tensor |
| Pose         | `(1, 300, 57)`                       | 6 box values + 17 keypoints × 3 (x, y, visibility)                |
| OBB          | `(1, 300, 7)`                        | 6 box values + 1 rotation angle                                   |

### How do I check if my exported model is end-to-end?

You can check from the Ultralytics Python API or from the exported ONNX model metadata:

!!! example "Check if a model is end-to-end"

    === "Python API"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.onnx")
        model.predict(verbose=False)  # run predict to setup predictor first
        print(model.predictor.model.end2end)  # True if end-to-end is enabled
        ```

    === "ONNX Runtime"

        ```python
        import onnxruntime as ort

        session = ort.InferenceSession("yolo26n.onnx")
        metadata = session.get_modelmeta().custom_metadata_map
        print(metadata.get("end2end"))  # 'True' if end-to-end is enabled
        ```

The two checks answer different questions: the ONNX metadata records the head that was exported, while `predictor.model.end2end` reports that the backend output is already post-processed and needs no external NMS. They disagree for a model exported with `end2end=False, nms=True`, which uses the one-to-many head but bakes NMS into the graph and also outputs `(1, 300, 6)` — so neither the flag nor the output shape alone identifies the head. For other task shapes, see the [output shapes FAQ](#my-exported-onnx-model-outputs-1-300-6-is-that-correct).

### Is end-to-end supported for instance segmentation, pose, and OBB tasks?

Yes. YOLO26 detection-style task variants — [detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), [pose estimation](../tasks/pose.md), and [oriented object detection (OBB)](../tasks/obb.md) — support end-to-end inference by default. The `end2end=False` fallback is available across these tasks as well.

Each task extends the base detection output with task-specific data; `yolo26n.pt`, `yolo26n-seg.pt`, `yolo26n-pose.pt`, and `yolo26n-obb.pt` output shapes are listed under [How End-to-End Detection Works](#how-end-to-end-detection-works).
