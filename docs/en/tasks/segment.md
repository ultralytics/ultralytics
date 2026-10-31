---
comments: true
description: Instance segmentation with Ultralytics YOLO26. Explore COCO-pretrained models, then train, validate, predict, and export per-object mask models in Python or CLI.
keywords: instance segmentation, YOLO26, object detection, image segmentation, machine learning, deep learning, computer vision, COCO dataset, Ultralytics
model_name: yolo26n-seg
---

# Instance Segmentation with Ultralytics YOLO {#instance-segmentation}

<img width="1024" src="https://cdn.ul.run/i/cf52e30c0986666e4e850b17e0514a54.avif" alt="Ultralytics YOLO instance segmentation examples">

[Instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) goes a step further than object detection and involves identifying individual objects in an image and segmenting them from the rest of the image.

The output of an instance segmentation model is a set of masks or contours that outline each object in the image, along with class labels and confidence scores for each object. Instance segmentation is useful when you need to know not only where objects are in an image, but also what their [exact shape](https://academy.ultralytics.com/courses/computer-vision-foundations/instance-segmentation) is.

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

    YOLO26 Segment models use the `-seg` suffix, e.g., `yolo26n-seg.pt`, and are pretrained on [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 Segment models pretrained on the [COCO](../datasets/segment/coco.md) dataset are shown below.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-seg-perf.md" %}

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org/) dataset. <br>Reproduce with `yolo segment val data=coco.yaml device=0 nms=False`
- **Speed** averaged over COCO val images with ONNX on CPU and TensorRT10 on an NVIDIA T4 GPU. <br>Reproduce with `yolo segment val data=coco.yaml batch=1 device=0|cpu nms=False`
- **Params** and **FLOPs** values are for fused models after Conv/BatchNorm folding and removal of the unused detection branch. Pretrained checkpoints retain the full training architecture and may show higher counts.

These checkpoints segment the 80 COCO classes. To segment categories outside that list without retraining, see [YOLOE](../models/yoloe.md), which takes the classes as a text prompt, a visual example, or a built-in vocabulary.

See the [unreleased YOLO27 preview](../models/yolo27.md#performance-metrics) for preliminary COCO box and mask results.

## Train

Train [YOLO26n-seg](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n-seg) on the [COCO8-seg](https://platform.ultralytics.com/ultralytics/datasets/coco8-seg) dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

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

See full `train` mode details in the [Train](../modes/train.md) page. Segmentation models can also be trained with [Ultralytics Platform cloud training](../platform/train/cloud-training.md).

### Dataset format

YOLO segmentation dataset format can be found in detail in the [Dataset Guide](../datasets/segment/index.md). To convert an existing COCO JSON dataset to YOLO format, use the built-in `convert_coco` function with `use_segments=True`, as described in the [COCO to YOLO guide](../guides/coco-to-yolo.md). You can also create segmentation masks with [Ultralytics Platform annotation](../platform/data/annotation.md) using polygon tools and SAM-powered smart annotation.

## Val

Validate trained YOLO26n-seg model accuracy, measured as box and mask [mean average precision (mAP)](https://www.ultralytics.com/glossary/mean-average-precision-map). No arguments are needed, as the `model` retains its training `data` and arguments as model attributes: `path/to/best.pt` from the [Train](#train) example validates on COCO8-seg. Official weights record a training dataset path that doesn't exist on your machine, so they fall back to the task default `coco8-seg.yaml` with a warning. Pass `data` to validate on another dataset.

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
        metrics.box.maps  # mAP50-95(B) per class; classes absent from the val set hold the overall mAP(B)
        metrics.box.ap_class_index  # class indices that were actually evaluated
        metrics.box.image_metrics  # per-image metrics dictionary for det with precision, recall, F1, TP, FP, and FN
        metrics.seg.map  # map50-95(M)
        metrics.seg.map50  # map50(M)
        metrics.seg.map75  # map75(M)
        metrics.seg.maps  # mAP50-95(M) per class; classes absent from the val set hold the overall mAP(M)
        metrics.seg.ap_class_index  # class indices that were actually evaluated
        metrics.seg.image_metrics  # per-image metrics dictionary for seg with precision, recall, F1, TP, FP, and FN
        ```

    === "CLI"

        ```bash
        yolo segment val model=yolo26n-seg.pt data=coco8-seg.yaml     # val official model
        yolo segment val model=path/to/best.pt data=path/to/data.yaml # val custom model
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

### Instance vs Semantic Segmentation

Instance segmentation is object-level segmentation: two cars produce two masks, two boxes, and two confidence scores.
[Semantic segmentation](semantic.md) is pixel-level classification: those same cars become pixels with the same class ID
in one image-sized class map, with no per-object boxes, confidences, or default polygon list. See the
[field-by-field comparison](semantic.md#instance-vs-semantic-segmentation) on the semantic segmentation page.

## Export

Export a YOLO26n-seg model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-seg.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx # export custom model
        ```

Available YOLO26-seg export formats are in the table below. You can export to any format using the `format` argument, e.g., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, e.g., `yolo predict model=yolo26n-seg.onnx`. Usage examples are shown for your model after export completes.

!!! note

    CoreML supports embedded NMS (`nms=True`) for detection, instance segmentation and pose with static shapes. Default segmentation exports (`nms=None`) leave NMS to the consumer.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### How do I train a YOLO26 segmentation model on a custom dataset?

To [train a YOLO26 segmentation model on a custom dataset](https://www.ultralytics.com/blog/how-to-custom-train-ultralytics-yolo26-for-instance-segmentation), you first need to prepare your dataset in the YOLO segmentation format. You can use the built-in [`convert_coco`](../guides/coco-to-yolo.md) utility to convert COCO JSON datasets. Once your dataset is ready, you can train the model using Python or CLI commands:

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

### What is the difference between object detection and instance segmentation in YOLO26?

[Object detection](https://www.ultralytics.com/glossary/object-detection) identifies and localizes objects within an image by drawing bounding boxes around them, whereas instance segmentation not only identifies the bounding boxes but also delineates the exact shape of each object. YOLO26 instance segmentation models provide masks or contours that outline each detected object, which is particularly useful for tasks where knowing the precise shape of objects is important, such as medical imaging or autonomous driving.

### Why use YOLO26 for instance segmentation?

Ultralytics YOLO26 combines high accuracy with real-time inference speed, which suits instance segmentation on video streams and edge devices. YOLO26 Segment models come pretrained on the [COCO dataset](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml), so they segment 80 common object classes out of the box and provide a strong starting point for fine-tuning on custom data. The same `YOLO` API covers training, validation, prediction, and export for both research and industry applications.

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
        yolo segment val model=yolo26n-seg.pt data=coco8-seg.yaml
        ```

These steps will provide you with validation metrics like mean average precision (mAP), the standard measure of segmentation performance.

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
