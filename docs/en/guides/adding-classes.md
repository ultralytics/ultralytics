---
comments: true
description: Add a new class to a trained Ultralytics YOLO model, or tune an existing one, without changing the predictions of the other classes.
keywords: add class to YOLO, incremental learning, catastrophic forgetting, fine-tune single class, RefineDetectionTrainer, Ultralytics YOLO, object detection
---

# Adding Classes to a Trained Model

`RefineDetectionTrainer` tunes the classes you name and leaves every other class of a trained detection model untouched. Use it to add a class a pretrained model does not have, or to improve one it already predicts, without collecting the original dataset again.

## Before You Start

This trainer freezes the whole model except a small branch on the detection head, which is what keeps the other classes intact. That constraint comes with real limits.

!!! warning "Limitations"

    - **The [backbone](https://www.ultralytics.com/glossary/backbone) learns nothing new.** Your class is recognized from the features the pretrained model already has. This works when the new class looks like something the model has seen, such as a rhino against a [COCO](../datasets/detect/coco.md) model that knows elephants and cows. It works poorly for a domain the backbone has never encountered, such as X-ray, satellite or microscopy imagery.
    - **The dataset YAML must list every class of the pretrained model**, in the order the tuned model should use, plus the new names. Classes missing from it are dropped from the head.
    - **The base model must be trained.** An untrained or randomly initialized backbone has no features to reuse, so there is nothing to freeze and nothing to build on. Start from a pretrained checkpoint.
    - **Detect only.** Segment, Pose and OBB are work in progress.

!!! tip "No suitable base model?"

    If no trained model covers your domain, or the classes you need are far from anything a pretrained detector knows, use [YOLOE](../models/yoloe.md) instead. It detects classes from text prompts with no training at all, and `set_classes()` bakes your class list into the weights so [CLIP](https://www.ultralytics.com/glossary/contrastive-language-image-pre-training-clip) is not needed at inference:

    ```python
    from ultralytics import YOLOE

    names = ["person", "bus", "rhino"]  # every class the model should detect, in the order you want them indexed

    model = YOLOE("yoloe-26n-seg.pt")
    model.model.eval()
    model.model.get_vocab(names)  # embeds the names and fuses them into the cls head
    model.save("yoloe-fused.pt")

    model = YOLOE("yoloe-fused.pt")
    model.predict("https://ultralytics.com/images/bus.jpg")
    ```

    After fusing, the classification head is a plain convolution like a standard YOLO head, so inference costs the same.

    A fused YOLOE **Detect** model can then be tuned with `RefineDetectionTrainer` like any other base model. Convert the segmentation checkpoint to a detection model first, as described in [YOLOE training](../models/yoloe.md#train-usage):

    ```python
    from ultralytics import YOLOE
    from ultralytics.utils.patches import torch_load

    names = ["person", "bus", "rhino"]  # the same full list of class names

    model = YOLOE("yoloe-26n.yaml")
    model.load(torch_load("yoloe-26n-seg.pt")["model"])
    model.model.eval()
    model.model.get_vocab(names)
    model.save("yoloe-det-fused.pt")
    ```

    Segmentation checkpoints stay segmentation models and are not accepted, since this trainer is Detect only.

## Quickstart

Pass `RefineDetectionTrainer` as the `trainer` and select the classes to tune with `classes`:

!!! example

    === "Add a new class"

        ```python
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import RefineDetectionTrainer

        # data.yaml lists the 80 COCO names plus "rhino" at index 80
        model = YOLO("yolo26n.pt")
        model.train(data="data.yaml", epochs=50, classes=[80], trainer=RefineDetectionTrainer)
        ```

    === "Tune existing classes"

        ```python
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import RefineDetectionTrainer

        # improve "bowl" and "orange" on your images, leave the other 78 classes alone
        model = YOLO("yolo26n.pt")
        model.train(data="coco8.yaml", epochs=50, classes=[45, 49], trainer=RefineDetectionTrainer)
        ```

!!! note

    `trainer` takes a class, so this is a Python-only workflow. There is no CLI equivalent.

`classes` accepts any number of classes, new or existing. It also filters the dataset labels and validation to those classes, so only the classes you tune need annotating and the reported metrics cover only them.

## Preparing the Dataset YAML

Copy the class names of your base model and append the new ones. For a COCO model, take the 80 names from [coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) and add yours at the end:

```yaml
path: ../datasets/my-data
train: images/train
val: images/val

names:
    0: person
    1: bicycle
    # ... the remaining COCO names, unchanged ...
    79: toothbrush
    80: rhino # the new class
```

Label files only need boxes for the classes you tune. Everything else is ignored during training.

!!! note "Class names are matched, not positions"

    Rows are transferred by name, so the pretrained classes keep their weights even if your YAML reorders them. A name that does not appear in the base model is treated as new.

## What Changes in the Model

The detection head gains a small refinement branch, one block per detection level, built from the same depthwise layers as the existing classification branch at a quarter of its width. It predicts a class score adjustment for the tuned classes and a box adjustment gated by their confidence, and starts zero-initialized so an attached model predicts exactly what it did before training.

Everything else is frozen, including [batch normalization](https://www.ultralytics.com/glossary/batch-normalization) statistics. The classification output rows of the untuned classes are restored after every optimizer step, so weight decay and momentum cannot move them either.

The cost for one added class, measured at 640 pixels after `fuse`:

| Model                                                          | Parameters       | GFLOPs |
| -------------------------------------------------------------- | ---------------- | ------ |
| [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26) | 2.57M to 2.64M   | +1.2%  |
| [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26) | 10.01M to 10.25M | +1.1%  |
| [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26) | 21.90M to 22.31M | +1.0%  |

The result is a normal checkpoint. [Prediction](../modes/predict.md), [validation](../modes/val.md) and [export](../modes/export.md) all work as usual.

## Example Results

A YOLO26n COCO model tuned for 30 epochs on 559 `rhino` boxes reaches **0.649** mAP50 and **0.591** mAP50-95 on that class. COCO class scores differ from the pretrained model by 5.7e-14, which is floating point noise.

The pretrained model calls rhinos elephants, cows and horses. The tuned model adds `rhino` and leaves every COCO detection at its original confidence:

```text
base : [('elephant', 0.7)]
tuned: [('elephant', 0.7), ('rhino', 0.65)]

base : [('elephant', 0.77), ('horse', 0.42), ('elephant', 0.41)]
tuned: [('elephant', 0.77), ('rhino', 0.6), ('horse', 0.42), ('elephant', 0.41), ('rhino', 0.38)]
```

## FAQ

### Are the other classes really unchanged?

Their class scores are bit-identical. Boxes are shared by every class, so the box adjustment can shift a box on the few anchors where a tuned class is confident. Class predictions are never affected.

### How many images do I need?

Far fewer than a full retrain, because only a small branch is being trained. A few hundred boxes of the new class is a reasonable starting point. Add more if the class is visually close to one the model already predicts.

### Can I add a second class later?

Yes. Training a model that was already tuned stacks a second branch on it and freezes the first, so the classes of the earlier session keep their predictions:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import RefineDetectionTrainer

# session 1 adds class 80, session 2 adds class 81 and keeps class 80
model = YOLO("runs/detect/train/weights/best.pt")
model.train(data="data-2.yaml", epochs=50, classes=[81], trainer=RefineDetectionTrainer)
```

Each session adds another branch and about 1% inference cost, so pass the classes together as `classes=[80, 81]` when you know them up front.

### Why is my new class not detected at all?

Check that the class index in `classes` matches its position in the dataset YAML, and that the label files use that same index. If the trainer reports zero instances during training, the labels were filtered out because their indices do not match.

### Can I use this for Segment, Pose or OBB?

Not yet. Those tasks are work in progress. For related approaches see [Knowledge Distillation](knowledge-distillation.md) and the [fine-tuning guide](finetuning-guide.md).
