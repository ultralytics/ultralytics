---
comments: true
description: Master instance segmentation using YOLO26. Learn how to detect, segment and outline objects in images with detailed guides and examples.
keywords: instance segmentation, YOLO26, object detection, image segmentation, machine learning, deep learning, computer vision, COCO dataset, Ultralytics
model_name: yolo26n-seg
---

# Instance Segmentation

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/instance-segmentation-examples.avif" alt="Instance segmentation examples">

[Instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) goes a step further than object detection and involves identifying individual objects in an image and segmenting them from the rest of the image.

The output of an instance segmentation model is a set of masks or contours that outline each object in the image, along with class labels and confidence scores for each object. Instance segmentation is useful when you need to know not only where objects are in an image, but also what their exact shape is.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/o4Zd-IeMlSY?si=37nusCzDTd74Obsp"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Run Segmentation with Pretrained Ultralytics YOLO Model in Python.
</p>

!!! tip

    YOLO26 Segment models use the `-seg` suffix, i.e., `yolo26n-seg.pt`, and are pretrained on [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 pretrained Segment models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) dataset, [Semantic](semantic.md) models are pretrained on [Cityscapes](../datasets/semantic/cityscapes.md), and Classify models are pretrained on the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-seg-perf.md" %}

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org/) dataset. <br>Reproduce by `yolo val segment data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val segment data=coco.yaml batch=1 device=0|cpu`
- **Params** and **FLOPs** values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers and, for end2end models, removes the auxiliary one-to-many detection head. Pretrained checkpoints retain the full training architecture and may show higher counts.

## Train

Train YOLO26n-seg on the COCO8-seg dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n-seg.yaml").load("yolo26n-seg.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo segment train data=coco8-seg.yaml model=yolo26n-seg.yaml pretrained=yolo26n-seg.pt epochs=100 imgsz=640
        ```

See full `train` mode details in the [Train](../modes/train.md) page. Segmentation models can also be trained on cloud GPUs through [Ultralytics Platform](https://platform.ultralytics.com).

### Dataset format

YOLO segmentation dataset format can be found in detail in the [Dataset Guide](../datasets/segment/index.md). To convert your existing dataset from other formats (like COCO etc.) to YOLO format, please use [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) tool by Ultralytics. You can also create segmentation masks on [Ultralytics Platform](https://platform.ultralytics.com) using polygon tools and SAM-powered smart annotation.

## Val

Validate trained YOLO26n-seg model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8-seg dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95(B)
        metrics.box.map50  # map50(B)
        metrics.box.map75  # map75(B)
        metrics.box.maps  # a list containing mAP50-95(B) for each category
        metrics.box.image_metrics  # per-image metrics dictionary for det with precision, recall, F1, TP, FP, and FN
        metrics.seg.map  # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps  # a list containing mAP50-95(M) for each category
        metrics.seg.image_metrics  # per-image metrics dictionary for seg with precision, recall, F1, TP, FP, and FN
        ```

    === "CLI"

        ```bash
        yolo segment val model=yolo26n-seg.pt  # val official model
        yolo segment val model=path/to/best.pt # val custom model
        ```

## Predict

Use a trained YOLO26n-seg model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

        # Access the results
        for result in results:
            xy = result.masks.xy  # mask polygons in pixel coordinates
            xyn = result.masks.xyn  # normalized mask polygons
            masks = result.masks.data  # binary masks, shape (N,H,W), dtype torch.uint8
        ```

    === "CLI"

        ```bash
        yolo segment predict model=yolo26n-seg.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo segment predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

### Results Output

YOLO instance segmentation returns one `Results` object per image. Each result stores object-level predictions, where
each detected instance has its own binary mask, class, confidence, and box.

| Attribute           | Type            | Shape         | Description                         |
| ------------------- | --------------- | ------------- | ----------------------------------- |
| `result.masks`      | `Masks`         | `(N)`         | Instance masks.                     |
| `result.masks.data` | `torch.uint8`   | `(N,H,W)`     | Binary masks, values `0` or `1`.    |
| `result.masks.xy`   | `np.float32`    | `list[(P,2)]` | Pixel polygons.                     |
| `result.masks.xyn`  | `np.float32`    | `list[(P,2)]` | Normalized polygons.                |
| `result.boxes`      | `Boxes`         | `(N)`         | Instance boxes/classes/confidences. |
| `result.boxes.cls`  | `torch.float32` | `(N,)`        | Class IDs; cast to `int` for names. |

For task-specific `Results` fields across every task, see the [Predict Results by Task](../modes/predict.md#results-by-task) section.

### How This Differs from Semantic Segmentation

Instance segmentation is object-level segmentation: two cars produce two masks, two boxes, and two confidence scores.
[Semantic segmentation](semantic.md) is pixel-level classification: those same cars become pixels with the same class ID
in one image-sized class map, with no per-object boxes, confidences, or default polygon list.

## Export

Export a YOLO26n-seg model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-seg.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx # export custom-trained model
        ```

Available YOLO26-seg export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-seg.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### How do I train a YOLO26 segmentation model on a custom dataset?

To train a YOLO26 segmentation model on a custom dataset, you first need to prepare your dataset in the YOLO segmentation format. You can use tools like [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) to convert datasets from other formats. Once your dataset is ready, you can train the model using Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26 segment model
        model = YOLO("yolo26n-seg.pt")

        # Train the model
        results = model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo segment train data=path/to/your_dataset.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
        ```

Check the [Configuration](../usage/cfg.md) page for more available arguments.

### What is the difference between [object detection](https://www.ultralytics.com/glossary/object-detection) and instance segmentation in YOLO26?

Object detection identifies and localizes objects within an image by drawing bounding boxes around them, whereas instance segmentation not only identifies the bounding boxes but also delineates the exact shape of each object. YOLO26 instance segmentation models provide masks or contours that outline each detected object, which is particularly useful for tasks where knowing the precise shape of objects is important, such as medical imaging or autonomous driving.

### Why use YOLO26 for instance segmentation?

Ultralytics YOLO26 is a state-of-the-art model recognized for its high accuracy and real-time performance, making it ideal for instance segmentation tasks. YOLO26 Segment models come pretrained on the [COCO dataset](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), ensuring robust performance across a variety of objects. Additionally, YOLO supports training, validation, prediction, and export functionalities with seamless integration, making it highly versatile for both research and industry applications.

### How do I load and validate a pretrained YOLO segmentation model?

Loading and validating a pretrained YOLO segmentation model is straightforward. Here's how you can do it using both Python and CLI:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-seg.pt")

        # Validate the model
        metrics = model.val()
        print("Mean Average Precision for boxes:", metrics.box.map)
        print("Mean Average Precision for masks:", metrics.seg.map)
        ```

    === "CLI"

        ```bash
        yolo segment val model=yolo26n-seg.pt
        ```

These steps will provide you with validation metrics like [Mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP), crucial for assessing model performance.

### How can I export a YOLO segmentation model to ONNX format?

Exporting a YOLO segmentation model to ONNX format is simple and can be done using Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-seg.pt")

        # Export the model to ONNX format
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-seg.pt format=onnx
        ```

For more details on exporting to various formats, refer to the [Export](../modes/export.md) page.

### How do I expose the auxiliary semantic segmentation output during inference?

YOLO26-seg includes an auxiliary semantic segmentation branch used to aid training, but this branch is not exposed in the default inference outputs. To expose it during inference, save the helper below as `semseg_wrapper.py`:

```python
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from ultralytics.nn.modules import Segment26
from ultralytics.nn.modules.block import Proto
from ultralytics.nn.modules.head import Detect
from ultralytics.utils import LOGGER, ops


def _segment26_forward(self, x):
    """Patched Segment26.forward: emit (det_output, proto, semseg) at inference."""
    outputs = Detect.forward(self, x)
    preds = outputs[1] if isinstance(outputs, tuple) else outputs

    pm = self.proto
    feat = x[0]
    for i, f in enumerate(pm.feat_refine):
        up_feat = f(x[i + 1])
        up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
        feat = feat + up_feat
    proto = Proto.forward(pm, pm.feat_fuse(feat))
    semseg = pm.semseg(feat) if pm.semseg is not None else None

    if isinstance(preds, dict):
        if self.end2end:
            preds["one2many"]["proto"] = proto
            preds["one2one"]["proto"] = tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
        else:
            preds["proto"] = proto
    if self.training:
        return preds
    extras = (proto,) if semseg is None else (proto, semseg)
    if self.export:
        return (outputs, *extras)
    return ((outputs[0], *extras), preds)


def _patch_predictor(predictor):
    """Patch SegmentationPredictor.postprocess to surface semseg on Results."""
    orig_postprocess = predictor.postprocess

    def postprocess(self, preds, img, orig_imgs):
        first = preds[0]
        semseg_batch = first[2] if isinstance(first, tuple) and len(first) >= 3 else None
        if semseg_batch is None:
            return orig_postprocess(preds, img, orig_imgs)
        patched = ((first[0], first[1]), *preds[1:])
        results = orig_postprocess(patched, img, orig_imgs)
        sem_up = F.interpolate(semseg_batch, size=img.shape[2:], mode="bilinear", align_corners=False)
        for r, s in zip(results, sem_up):
            oh, ow = r.orig_shape
            # 1-channel (binary): foreground where logit > 0 (== sigmoid > 0.5).
            # Multi-channel: argmax over channels.
            scaled = ops.scale_masks(s.unsqueeze(0), (oh, ow), padding=True)[0]  # (C, H, W)
            if scaled.shape[0] == 1:
                cls_map = (scaled[0] > 0).to(torch.int32)
            else:
                cls_map = scaled.argmax(0).to(torch.int32)
                cls_map[scaled.amax(0) <= 0] = 255
            r.update(semantic_mask=cls_map)
        return results

    predictor.postprocess = types.MethodType(postprocess, predictor)
    return predictor


def enable_semseg(yolo_or_module):
    """Enable semseg output for all Segment26 heads on a YOLO wrapper or raw nn.Module."""
    inner = (
        yolo_or_module.model
        if hasattr(yolo_or_module, "model") and isinstance(yolo_or_module.model, torch.nn.Module)
        else yolo_or_module
    )

    patched = 0
    for m in inner.modules():
        if isinstance(m, Segment26):
            if getattr(m.proto, "semseg", None) is None:
                LOGGER.warning("Proto26.semseg is None (model was fused). Reload an un-fused checkpoint.")
                continue
            m.forward = types.MethodType(_segment26_forward, m)
            m.proto.fuse = types.MethodType(lambda self: None, m.proto)
            patched += 1
    LOGGER.info(f"enable_semseg: patched {patched} Segment26 head(s).")

    if hasattr(yolo_or_module, "_smart_load"):
        orig_smart_load = yolo_or_module._smart_load

        def smart_load(key):
            obj = orig_smart_load(key)
            if key == "predictor":
                if isinstance(obj, type):
                    orig_setup = obj.setup_model

                    def setup_model(self_p, *a, **kw):
                        out = orig_setup(self_p, *a, **kw)
                        _patch_predictor(self_p)
                        return out

                    obj.setup_model = setup_model
                else:
                    _patch_predictor(obj)
            return obj

        yolo_or_module._smart_load = smart_load

    return yolo_or_module
```

Then run inference. The semantic segmentation result is stored on each `Results` object as `r.semantic_mask`:

```python
from semseg_wrapper import enable_semseg

from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
enable_semseg(model)
results = model.predict("img.jpg")
for r in results:
    semantic_mask = r.semantic_mask
    r.masks = None
    r.save("semantic_result.jpg", boxes=False)
```
