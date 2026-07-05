---
comments: true
description: Learn about semantic segmentation using YOLO26. Assign class labels to every pixel for dense scene understanding with Cityscapes and ADE20K support.
keywords: semantic segmentation, YOLO26, pixel-wise classification, scene parsing, Cityscapes, ADE20K, VOC, dense prediction, Ultralytics
model_name: yolo26n-sem
---

# Semantic Segmentation

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/semantic-segmentation-examples.avif" alt="Semantic segmentation examples">

[Semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) assigns a class label to every pixel in an image, producing a dense class map that covers the entire scene. Unlike [instance segmentation](segment.md), which separates individual objects, semantic segmentation groups all pixels of the same class together regardless of how many distinct objects are present.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/zF2T17ppKIE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO26 Semantic Segmentation Model on Custom Dataset | Ultralytics Platform
</p>

The output of a semantic segmentation model is a single height-by-width class map where each pixel value corresponds to a predicted class ID. This makes semantic segmentation ideal for scene parsing tasks such as autonomous driving, medical imaging, and land-cover mapping.

!!! tip

    Use `task=semantic` or the `yolo semantic` CLI task for semantic segmentation. YOLO26 semantic segmentation model files use the `-sem` suffix, such as `yolo26n-sem.pt`.

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 semantic segmentation models pretrained on the [Cityscapes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/cityscapes.yaml) dataset are shown below.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-semantic-perf.md" %}

- **mIoU<sup>val</sup>** values are for single-model single-scale on the [Cityscapes](https://www.cityscapes-dataset.com/) validation set. <br>Reproduce with `yolo semantic val data=cityscapes.yaml device=0 imgsz=2048`
- **Speed** metrics are averaged over Cityscapes validation images using an RTX3090 instance. <br>Reproduce with `yolo semantic val data=cityscapes.yaml batch=1 device=0|cpu imgsz=2048`
- **Params** and **FLOPs** values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers. Pretrained checkpoints retain the full training architecture and may show higher counts.

YOLO26 semantic segmentation models pretrained on the [ADE20K](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ade20k.yaml) dataset are shown below.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-semantic-ade20k-perf.md" %}

- **mIoU<sup>val</sup>** values are for single-model single-scale on the [ADE20K](https://ade20k.csail.mit.edu/) validation set. <br>Reproduce with `yolo semantic val model=yolo26n-sem-ade20k.pt data=ade20k.yaml device=0 imgsz=640`, replacing `yolo26n-sem-ade20k.pt` with the desired `yolo26*-sem-ade20k.pt` checkpoint.
- **Speed** metrics are averaged over ADE20K validation images using an RTX3090 instance. <br>Reproduce with `yolo semantic val model=yolo26n-sem-ade20k.pt data=ade20k.yaml batch=1 device=0|cpu imgsz=640`, replacing `yolo26n-sem-ade20k.pt` with the desired `yolo26*-sem-ade20k.pt` checkpoint.
- **Params** and **FLOPs** values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers. Pretrained checkpoints retain the full training architecture and may show higher counts.

## Train

Train YOLO26n-sem on the Cityscapes8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 1024. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-sem.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-sem.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n-sem.yaml").load("yolo26n-sem.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="cityscapes8.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo semantic train data=cityscapes8.yaml model=yolo26n-sem.yaml epochs=100 imgsz=1024

        # Start training from a pretrained *.pt model
        yolo semantic train data=cityscapes8.yaml model=yolo26n-sem.pt epochs=100 imgsz=1024

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo semantic train data=cityscapes8.yaml model=yolo26n-sem.yaml pretrained=yolo26n-sem.pt epochs=100 imgsz=1024
        ```

See full `train` mode details in the [Train](../modes/train.md) page.

### Dataset format

Semantic segmentation datasets use single-channel mask images, typically PNG, where each pixel value represents a class ID. Pixels with value 255 are treated as "ignore" and excluded from loss computation. The dataset YAML should specify paths to images and their corresponding mask directories. See the [Semantic Segmentation Dataset Guide](../datasets/semantic/index.md) for format details. Supported datasets include [Cityscapes](../datasets/semantic/cityscapes.md) and [ADE20K](../datasets/semantic/ade20k.md).

## Val

Validate trained YOLO26n-sem model [accuracy](https://www.ultralytics.com/glossary/accuracy) on a semantic segmentation dataset. Pass `data` explicitly so validation uses the intended dataset YAML.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-sem.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val(data="cityscapes.yaml")
        metrics.miou  # mean Intersection over Union
        metrics.pixel_accuracy  # overall pixel accuracy
        ```

    === "CLI"

        ```bash
        yolo semantic val model=yolo26n-sem.pt data=cityscapes.yaml    # validate official model
        yolo semantic val model=path/to/best.pt data=path/to/data.yaml # validate custom model
        ```

## Predict

Use a trained YOLO26n-sem model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-sem.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

        # Access the results
        for result in results:
            semantic_mask = result.semantic_mask.data  # class map, shape (H,W), integer dtype selected by class count
        ```

    === "CLI"

        ```bash
        yolo semantic predict model=yolo26n-sem.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo semantic predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg' # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

### Results Output

YOLO semantic segmentation returns one `Results` object per image. Each result stores one dense class map for the full
image instead of a list of object masks. Pixels with the same predicted class share the same class ID, even when they
belong to separate objects.

| Attribute                   | Type                                            | Shape   | Description                               |
| --------------------------- | ----------------------------------------------- | ------- | ----------------------------------------- |
| `result.semantic_mask`      | `SemanticMask`                                  | `(H,W)` | Dense class map.                          |
| `result.semantic_mask.data` | `torch.uint8`<br>`torch.int16`<br>`torch.int32` | `(H,W)` | Class IDs; dtype selected by class count. |
| `result.masks`              | -                                               | -       | No instance masks.                        |
| `result.boxes`              | -                                               | -       | No instance boxes/confidences.            |
| `result.masks.xy`           | -                                               | -       | No default polygons.                      |

For task-specific `Results` fields across every task, see the [Predict Results by Task](../modes/predict.md#results-by-task) section.

!!! tip "Mask boundary quality"

    Semantic segmentation predicts a dense class map, then resizes that map back to the image shape for visualization and
    downstream use. Very thin structures, such as lane markings, court lines, poles, or wires, can therefore look
    stair-stepped when inference runs at a much lower `imgsz` than the original image resolution. If boundaries appear
    jagged, first retest the native PyTorch `.pt` model with a larger `imgsz`, such as `1024`, `1280`, or the closest
    practical value to the source image size. Use exported models only after confirming the `.pt` output is acceptable,
    since lower-resolution inputs cannot recover fine detail that was not present in the predicted class map.

### Instance vs Semantic Segmentation

| Aspect               | Instance Segmentation (`task="segment"`)               | Semantic Segmentation (`task="semantic"`)                        |
| -------------------- | ------------------------------------------------------ | ---------------------------------------------------------------- |
| Prediction goal      | Segment each detected object separately                | Assign one class ID to every pixel                               |
| Output field         | `result.masks`                                         | `result.semantic_mask`                                           |
| Main data            | `result.masks.data`                                    | `result.semantic_mask.data`                                      |
| Shape                | `(N,H,W)`                                              | `(H,W)`                                                          |
| Pixel values         | Binary mask values: `0` or `1`                         | Class IDs: `0`, `1`, `2`, ...                                    |
| Dtype                | `torch.uint8`                                          | `torch.uint8`<br>`torch.int16`<br>`torch.int32`                  |
| Same-class objects   | Kept as separate instances                             | Merged into the same class region                                |
| Polygons             | Yes, through `result.masks.xy` and `result.masks.xyn`  | No polygon output by default                                     |
| Boxes and confidence | Yes, through `result.boxes`                            | No per-instance boxes or confidence scores                       |
| Typical use          | Counting, tracking, cropping, object-level measurement | Dense scene labeling, drivable area, land cover, medical regions |

## Export

Export a YOLO26n-sem model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-sem.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-sem.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx # export custom model
        ```

Available YOLO26 semantic segmentation export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-sem.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### How do I train a YOLO26 semantic segmentation model on a custom dataset?

To train a YOLO26 semantic segmentation model on a custom dataset, you need to prepare PNG mask images where each pixel value represents a class ID (0, 1, 2, ...) and pixels with value 255 are ignored during training. Create a dataset YAML file pointing to your images and masks directories, then train the model:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO26 semantic segmentation model
        model = YOLO("yolo26n-sem.pt")

        # Train the model
        results = model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=512)
        ```

    === "CLI"

        ```bash
        yolo semantic train data=path/to/your_dataset.yaml model=yolo26n-sem.pt epochs=100 imgsz=512
        ```

Check the [Configuration](../usage/cfg.md) page for more available arguments.

### What is the difference between instance segmentation and semantic segmentation?

Instance segmentation and semantic segmentation are both pixel-level tasks but differ in a key way:

- **[Semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation)** assigns a class label to every pixel but does not distinguish between individual objects of the same class. For example, all cars in a scene share the same class label.
- **[Instance segmentation](segment.md)** identifies each individual object separately, producing distinct masks for each object even if they belong to the same class.

Semantic segmentation is best suited for scene understanding tasks like autonomous driving and land-cover mapping, while instance segmentation is preferred when counting or tracking individual objects matters.

### Can I use instance segmentation data to train semantic segmentation?

Yes. If your dataset uses Ultralytics YOLO polygon labels (one `.txt` per image), **omit** `masks_dir` from the dataset YAML and the loader will convert polygons to per-image semantic masks on the fly. For multi-class datasets (`N > 1`) an extra `background` class is appended to `names` automatically. For single-class datasets (`N == 1`) training stays at 1 class — your declared class becomes `1` in the mask and uncovered pixels become `0`. See the [Semantic Segmentation Dataset Guide](../datasets/semantic/index.md#yolo-polygon-label-format) for details.

### What datasets are supported for semantic segmentation?

Ultralytics YOLO26 provides built-in configurations for several semantic segmentation datasets:

- **[Cityscapes](../datasets/semantic/cityscapes.md):** Urban street scenes with 19 classes, widely used for autonomous driving research.
- **[ADE20K](../datasets/semantic/ade20k.md):** A large-scale scene parsing dataset with 150 classes.

You can also use any custom dataset that provides PNG mask annotations where pixel values correspond to class IDs.

### How do I validate a pretrained YOLO26 semantic segmentation model?

Validate a pretrained YOLO26 semantic segmentation model with the dataset YAML used for evaluation:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-sem.pt")

        # Validate the model
        metrics = model.val(data="cityscapes.yaml")
        print("Mean IoU:", metrics.miou)
        print("Pixel Accuracy:", metrics.pixel_accuracy)
        ```

    === "CLI"

        ```bash
        yolo semantic val model=yolo26n-sem.pt data=cityscapes.yaml
        ```

These steps will provide you with validation metrics like mean Intersection over Union (mIoU) and pixel accuracy, which are standard measures for assessing semantic segmentation performance.

### How can I export a YOLO26 semantic segmentation model to ONNX format?

Export a YOLO26 semantic segmentation model to ONNX format with Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-sem.pt")

        # Export the model to ONNX format
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-sem.pt format=onnx
        ```

For more details on exporting to various formats, refer to the [Export](../modes/export.md) page.
