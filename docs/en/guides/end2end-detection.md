---
title: YOLO26 End-to-End NMS-Free Detection
comments: true
description: Choose conventional NMS or end-to-end inference with one nms argument, understand exported outputs, and migrate existing end2end and nms options.
keywords: YOLO26, end-to-end detection, NMS-free inference, NMS, model export, deployment guide, Ultralytics, ONNX, TensorRT, CoreML
---

# Understanding End-to-End Detection in Ultralytics YOLO26

[YOLO26](../models/yolo26.md) trains both a **one-to-many** head and a **one-to-one** head. Prediction and validation use the one-to-many head with [Non-Maximum Suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) by default. This favors accuracy using the same trained weights. Set `nms=False` to use the faster, NMS-free one-to-one head instead.

One argument controls the choice across [prediction](../modes/predict.md), [validation](../modes/val.md), [tracking](../modes/track.md), [export](../modes/export.md), and [benchmarking](../modes/benchmark.md):

| `nms`            | Prediction and validation               | Export                                             |
| ---------------- | --------------------------------------- | -------------------------------------------------- |
| `None` (default) | One-to-many head; Ultralytics runs NMS  | Raw one-to-many outputs; the consumer runs NMS     |
| `True`           | Same as `None`                          | One-to-many head with NMS embedded where supported |
| `False`          | One-to-one head without IoU suppression | NMS-free one-to-one outputs where supported        |

`None` means no optional post-processing is embedded in the model. It does **not** remove the normal processing that converts model outputs into prediction results or validation metrics. Classification, semantic segmentation, depth, and models without selectable detection heads keep their task's native behavior.

!!! example "Choose the output path"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        results = model.predict("image.jpg")  # one-to-many + NMS
        metrics = model.val(data="coco.yaml")  # one-to-many + NMS
        results = model.predict("image.jpg", nms=False)  # opt into NMS-free inference
        results = model.predict("image.jpg", nms=None)  # switch back to one-to-many + NMS

        model.export(format="onnx")  # raw one-to-many outputs
        model.export(format="onnx", nms=True)  # embed NMS
        model.export(format="onnx", nms=False)  # NMS-free one-to-one outputs
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n.pt source=image.jpg
        yolo val model=yolo26n.pt data=coco.yaml
        yolo predict model=yolo26n.pt source=image.jpg nms=False
        yolo export model=yolo26n.pt format=onnx nms=None
        yolo export model=yolo26n.pt format=onnx nms=True
        yolo export model=yolo26n.pt format=onnx nms=False
        ```

## How End-to-End Detection Works

Both heads share the backbone and neck and are optimized during [training](../modes/train.md). The one-to-many head provides multiple candidate predictions per object; NMS removes overlapping detections. The one-to-one head learns to produce a single prediction per object. Selecting the inference path does not disable dual-head supervision. Validation during training uses the selected inference head, so checkpoint selection and early stopping follow the same predictions as deployment.

| Head                  | Detection output before external processing | Processing                               |
| --------------------- | ------------------------------------------- | ---------------------------------------- |
| One-to-many (default) | `(N, nc + 4, 8400)`                         | Confidence filtering and NMS             |
| One-to-one            | `(N, 300, 6)`                               | Confidence filtering; no IoU suppression |

Here `N` is batch size, `nc` is the number of classes, and 8400 is the candidate count at `imgsz=640`. The one-to-one detection rows contain `[x1, y1, x2, y2, confidence, class_id]`. Other detection tasks carry additional outputs:

| Task                                         | End-to-end output                      | Extra data                       |
| -------------------------------------------- | -------------------------------------- | -------------------------------- |
| [Detection](../tasks/detect.md)              | `(N, 300, 6)`                          | —                                |
| [Instance segmentation](../tasks/segment.md) | `(N, 300, 6 + nm)` and `(N, nm, H, W)` | Mask coefficients and prototypes |
| [Pose](../tasks/pose.md)                     | `(N, 300, 57)`                         | 17 keypoints × 3 values          |
| [OBB](../tasks/obb.md)                       | `(N, 300, 7)`                          | Rotation angle                   |

Fusion removes unused inference branches as well as folding Conv and BatchNorm layers. Keep the original training checkpoint if you need to switch heads: fusion cannot reconstruct a branch that has already been removed. A model with only its one-to-one head remaining keeps that available path.

## Do I Need to Change My Code?

### Using the Ultralytics Python API or CLI

Prediction and validation continue to return the same result and metric types. The default for models with both heads now selects one-to-many predictions with NMS. To keep the previous YOLO26 NMS-free behavior, add `nms=False`.

### Using Custom Inference Code

Default YOLOv8 and YOLO11 exports retain their existing raw output layout. Default YOLO26 exports now use that layout too. If your application expects YOLO26's previous `(N, 300, 6)` output, export with `nms=False` explicitly.

|                     | `nms=None`                        | `nms=False`                           |
| ------------------- | --------------------------------- | ------------------------------------- |
| Detection output    | `(N, nc + 4, 8400)`               | `(N, 300, 6)`                         |
| Box format          | `xywh`                            | `xyxy`                                |
| Scores              | One score per class per candidate | Confidence and class ID per detection |
| External processing | Confidence filtering and NMS      | Confidence filtering                  |

`nms=True` also produces processed detections, but uses the one-to-many head and embeds traditional NMS. It is useful when your deployment runtime should receive detections without implementing suppression itself.

Existing exported models keep their original graphs and outputs. Passing a different `nms` value when loading an exported graph does not rebuild it; export the source checkpoint again to change its output path. Ultralytics uses the artifact's metadata to avoid applying NMS twice.

### Migrating `end2end`

`end2end` remains accepted with a deprecation warning. Use `nms` instead:

| Older arguments            | Replacement |
| -------------------------- | ----------- |
| `end2end=True`             | `nms=False` |
| `end2end=False`            | `nms=None`  |
| `end2end=False, nms=True`  | `nms=True`  |
| `end2end=False, nms=False` | `nms=None`  |

Existing explicit `nms=False` calls keep NMS-free inference on dual-head models and raw exports on models without that head. A standalone `nms=True` now selects one-to-many predictions on dual-head models instead of being discarded by the exporter. When both older arguments are supplied, the `end2end` selection retains its previous precedence: `end2end=True, nms=True` selects the NMS-free head.

## Export Format Compatibility

ONNX, TensorRT, CoreML, OpenVINO and several other formats support NMS-free exports. NCNN, RKNN, PaddlePaddle, ExecuTorch, IMX, Edge TPU and Qualcomm QNN fall back to the one-to-many path when their operators cannot support end-to-end output. Existing format warnings explain the fallback.

- **Embedded NMS:** `nms=True` remains subject to each format's existing task, precision and dynamic-shape restrictions. Formats without embedded NMS support export native outputs for external processing.
- **CoreML:** Embedded NMS supports detect, segment and pose with static shapes. Use `nms=True` for detection models that need Xcode Preview's NMS pipeline.
- **MNN:** Embedded NMS supports detect and pose with `dynamic=False`.
- **IMX:** Detection, instance segmentation and pose require embedded NMS, selected automatically.
- **Hailo:** The default one-to-many detection path uses HailoRT NMS. `nms=False` selects the available one-to-one detection path. Other tasks retain their existing Hailo output contracts.
- **Quantization:** TensorRT versions before 8.5.0, TensorRT 10.3.0 INT8 on JetPack 6, and LiteRT INT8 or `w8a16` retain their existing end-to-end fallback restrictions.

See the individual [integration guides](../integrations/index.md) for hardware requirements. For full FP16 output tensors, use `nms=None`; end-to-end class indices can keep output tensors in FP32 even when the model is quantized.

## Accuracy and Speed Trade-offs

The [published YOLO26 COCO results](../models/yolo26.md) show that the one-to-many head improves detection mAP by 0.6–0.8 points across the five scales: for example, 40.9 versus 40.1 for YOLO26n, and 57.5 versus 56.9 for YOLO26x. The one-to-one head avoids the NMS pass and favors latency. These results motivate the default; they do not guarantee a gain on every dataset.

Published NMS-free speed measurements use `nms=False`. Compare accuracy and latency using the same head selection, image size, precision and hardware.

## FAQ

### What does `max_det` control?

It limits detections returned by prediction and validation. For end-to-end and embedded-NMS exports, the limit is part of the graph; re-export to change it. End-to-end output can contain fewer candidates when the image supplies fewer than `max_det` anchors.

### My exported ONNX model outputs `(1, 300, 6)` — is that correct?

Yes, for `nms=False` or `nms=True` with the default detection limit. The shape alone does not identify which head was exported. A default raw COCO detection export instead normally has shape `(1, 84, 8400)` at `imgsz=640`.

### How can I tell which output path was exported?

The export metadata's `end2end` field identifies the selected head. Its `args.nms=True` field identifies embedded NMS, as in older runtimes. By contrast, `model.predictor.model.end2end` means the backend output is already processed and should not receive external NMS. It can therefore be true for either NMS-free or embedded-NMS exports.

### Does this apply to segmentation, pose and OBB?

Yes. The same `nms` argument selects the available detection head for detect, instance segmentation, pose and OBB. It does not replace mask reconstruction, keypoint decoding, rotated-box handling, classification probabilities, semantic class maps or depth decoding.
