---
comments: true
description: Learn about semantic segmentation using YOLO26. Assign class labels to every pixel for dense scene understanding with Cityscapes and ADE20K support.
keywords: semantic segmentation, YOLO26, pixel-wise classification, scene parsing, Cityscapes, ADE20K, VOC, dense prediction, Ultralytics
model_name: yolo26n-semseg
---

# Semantic Segmentation

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/semantic-segmentation-examples.avif" alt="Semantic segmentation examples">

[Semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) assigns a class label to every pixel in an image, producing a dense class map that covers the entire scene. Unlike [instance segmentation](segment.md), which separates individual objects, semantic segmentation groups all pixels of the same class together regardless of how many distinct objects are present.

The output of a semantic segmentation model is a single H x W class map where each pixel value corresponds to a predicted class ID. This makes semantic segmentation ideal for scene parsing tasks such as autonomous driving, medical imaging, and land-cover mapping.

!!! tip

    YOLO26 Semantic Segmentation models use the `-semseg` suffix, i.e., `yolo26n-semseg.pt`, and are pretrained on [Cityscapes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/cityscapes.yaml).

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 pretrained Semantic Segmentation models are shown here, which are pretrained on the [Cityscapes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/cityscapes.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

{% include "macros/yolo-semseg-perf.md" %}

- **mIoU<sup>val</sup>** values are for single-model single-scale on [Cityscapes](https://www.cityscapes-dataset.com/) dataset. <br>Reproduce by `yolo val semseg data=cityscapes.yaml device=0 imgsz=2048`
- **Speed** averaged over cityscapes val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance. <br>Reproduce by `yolo val semseg data=cityscapes.yaml batch=1 device=0|cpu imgsz=2048`
- **Params** and **FLOPs** values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers. Pretrained checkpoints retain the full training architecture and may show higher counts.

## Train

Train YOLO26n-semseg on the Cityscapes8 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) at image size 1024. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-semseg.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-semseg.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolo26n-semseg.yaml").load("yolo26n-semseg.pt")  # build from YAML and transfer weights

        # Train the model
        results = model.train(data="cityscapes8.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo semseg train data=cityscapes8.yaml model=yolo26n-semseg.yaml epochs=100 imgsz=1024

        # Start training from a pretrained *.pt model
        yolo semseg train data=cityscapes8.yaml model=yolo26n-semseg.pt epochs=100 imgsz=1024

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo semseg train data=cityscapes8.yaml model=yolo26n-semseg.yaml pretrained=yolo26n-semseg.pt epochs=100 imgsz=1024
        ```

See full `train` mode details in the [Train](../modes/train.md) page.

### Dataset format

Semantic segmentation datasets use PNG mask images where each pixel value represents a class ID. Pixels with value 255 are treated as "ignore" and excluded from loss computation. The dataset YAML should specify paths to images and their corresponding mask directories. See the [Semantic Segmentation Dataset Guide](../datasets/semseg/index.md) for format details. Supported datasets include [Cityscapes](../datasets/semseg/cityscapes.md) and [ADE20K](../datasets/semseg/ade20k.md).

## Val

Validate trained YOLO26n-semseg model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the Cityscapes8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-semseg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.miou  # mean Intersection over Union
        metrics.pixel_accuracy  # overall pixel accuracy
        ```

    === "CLI"

        ```bash
        yolo semseg val model=yolo26n-semseg.pt  # val official model
        yolo semseg val model=path/to/best.pt  # val custom model
        ```

## Predict

Use a trained YOLO26n-semseg model to run predictions on images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-semseg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

        # Access the results
        for result in results:
            semantic_mask = result.semantic_mask  # H x W class map (torch.Tensor)
        ```

    === "CLI"

        ```bash
        yolo semseg predict model=yolo26n-semseg.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo semseg predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLO26n-semseg model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-semseg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-semseg.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom-trained model
        ```

Available YOLO26-semseg export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-semseg.onnx`. Usage examples are shown for your model after export completes.

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
        model = YOLO("yolo26n-semseg.pt")

        # Train the model
        results = model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=512)
        ```

    === "CLI"

        ```bash
        yolo semseg train data=path/to/your_dataset.yaml model=yolo26n-semseg.pt epochs=100 imgsz=512
        ```

Check the [Configuration](../usage/cfg.md) page for more available arguments.

### What is the difference between instance segmentation and semantic segmentation?

Instance segmentation and semantic segmentation are both pixel-level tasks but differ in a key way:

- **[Semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation)** assigns a class label to every pixel but does not distinguish between individual objects of the same class. For example, all cars in a scene share the same class label.
- **[Instance segmentation](segment.md)** identifies each individual object separately, producing distinct masks for each object even if they belong to the same class.

Semantic segmentation is best suited for scene understanding tasks like autonomous driving and land-cover mapping, while instance segmentation is preferred when counting or tracking individual objects matters.

### Can I use instance segmentation data to train semantic segmentation?

Yes. If your dataset uses Ultralytics YOLO polygon labels (one `.txt` per image), simply **omit** `masks_dir` from the dataset YAML and the loader will convert polygons to per-image semantic masks on the fly. For multi-class datasets (`N > 1`) an extra `background` class is appended to `names` automatically. For single-class datasets (`N == 1`) training stays at 1 class — your declared class becomes `1` in the mask and uncovered pixels become `0`. See the [Semantic Segmentation Dataset Guide](../datasets/semseg/index.md#yolo-polygon-label-format) for details.

### What datasets are supported for semantic segmentation?

Ultralytics YOLO26 supports several semantic segmentation datasets out of the box:

- **[Cityscapes](../datasets/semseg/cityscapes.md):** Urban street scenes with 19 classes, widely used for autonomous driving research.
- **[ADE20K](../datasets/semseg/ade20k.md):** A large-scale scene parsing dataset with 150 classes.

You can also use any custom dataset that provides PNG mask annotations where pixel values correspond to class IDs.

### How do I validate a pretrained YOLO26 semantic segmentation model?

Loading and validating a pretrained YOLO26 semantic segmentation model is straightforward:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-semseg.pt")

        # Validate the model
        metrics = model.val()
        print("Mean IoU:", metrics.miou)
        print("Pixel Accuracy:", metrics.pixel_accuracy)
        ```

    === "CLI"

        ```bash
        yolo semseg val model=yolo26n-semseg.pt
        ```

These steps will provide you with validation metrics like mean Intersection over Union (mIoU) and pixel accuracy, which are standard measures for assessing semantic segmentation performance.

### How can I export a YOLO26 semantic segmentation model to ONNX format?

Exporting a YOLO26 semantic segmentation model to ONNX format is simple and can be done using Python or CLI commands:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo26n-semseg.pt")

        # Export the model to ONNX format
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-semseg.pt format=onnx
        ```

For more details on exporting to various formats, refer to the [Export](../modes/export.md) page.
