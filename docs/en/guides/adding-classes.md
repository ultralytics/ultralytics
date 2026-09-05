---
comments: true
description: Add a new class to a trained Ultralytics YOLO model, or tune an existing one, without changing the predictions of the other classes.
keywords: add class to YOLO, incremental learning, catastrophic forgetting, fine-tune single class, RefineDetectionTrainer, Ultralytics YOLO, object detection
---

# Adding and Fine-Tuning Classes on a Trained Model

`RefineDetectionTrainer` tunes the classes you name on a trained detection model and leaves every other class untouched. Use it to add a class a pretrained model does not have, or to improve one it already predicts, without the original dataset.

!!! warning "Limitations"

    - **The [backbone](https://www.ultralytics.com/glossary/backbone) learns nothing new.** The class is recognized from the features the pretrained model already has, so it works for a rhino on a [COCO](../datasets/detect/coco.md) model that knows elephants and cows, and poorly for a domain the backbone has never seen, such as X-ray or satellite imagery.
    - **The dataset YAML must list every class of the pretrained model**, in the order the tuned model should use, plus the new names. Classes missing from it are dropped from the head.
    - **The base model must be trained.** A randomly initialized model has no features to reuse.
    - **Tuning an existing class can make it worse elsewhere.** The guarantee covers the classes you do not name. Tuning `person` on ten images pulls it towards those ten images, so tune an existing class only with enough data to represent it, or add your own class instead.
    - **Detect only.** Segment, Semantic, Classify, Pose and OBB are not supported yet.

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

`trainer` takes a class, so this is a Python-only workflow. `classes` accepts any number of new or existing classes, and also filters the dataset labels and the validation to them, so only the tuned classes need annotating and the reported metrics cover only them.

## Preparing the Dataset YAML

Copy the class names of the base model and append the new ones. For a COCO model, take the 80 names from [coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) and add yours at the end:

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

Label files only need boxes for the classes you tune. Rows are transferred by name, so the pretrained classes keep their weights even if the YAML reorders them, and a name the base model does not have is treated as new.

## What Changes in the Model

The detection head gains a small refinement branch per detection level, built from the same depthwise blocks as the classification branch, a quarter as wide and twice as deep. It predicts a class score delta for the tuned classes and a box delta gated by their confidence, and starts zero-initialized, so the model predicts exactly what it did before training.

Everything else is frozen, including [batch normalization](https://www.ultralytics.com/glossary/batch-normalization) statistics, and the classification rows of the untuned classes are restored after every optimizer step so weight decay and momentum cannot move them. Their class scores are therefore bit-identical. Boxes are shared by all classes, so the box delta can move a box on the few anchors where a tuned class is confident.

The cost for one added class, measured at 640 pixels after `fuse`:

| Model                                                          | Parameters       | GFLOPs |
| -------------------------------------------------------------- | ---------------- | ------ |
| [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26) | 2.42M to 2.47M   | +1.7%  |
| [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26) | 9.51M to 9.68M   | +1.5%  |
| [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26) | 20.44M to 20.72M | +1.3%  |

The result is a normal checkpoint: [predict](../modes/predict.md), [val](../modes/val.md) and [export](../modes/export.md) work as usual.

## Example: Adding `rhino` to a COCO Model

COCO has no rhino, so a pretrained model calls rhinos elephants and cows. Relabel the [African Wildlife](../datasets/detect/african-wildlife.md) images to COCO indices (`rhino` becomes the new class 80), write the YAML above, and train:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import RefineDetectionTrainer

model = YOLO("yolo26n.pt")
model.train(data="rhino.yaml", epochs=30, classes=[80], trainer=RefineDetectionTrainer)
```

With 559 `rhino` boxes and 30 epochs this reaches 0.747 mAP50 and 0.672 mAP50-95 on `rhino`, while the maximum class score change of the 80 COCO classes against the base model is 5.7e-14, floating point noise.

<table border="0">
    <tr>
        <th>Pretrained YOLO26n</th>
        <th>rhino class added through <code>RefineDetectionTrainer</code></th>
    </tr>
    <tr>
        <td><img src="https://cdn.ul.run/i/5dd5d63c8f29dc5bbdbca9e4631e99b5.avif" alt="Pretrained model labels the rhino an elephant" width="640"></td>
        <td><img src="https://cdn.ul.run/i/0d478913546540e6c1c566f86a8687f3.avif" alt="Tuned model detects rhino" width="640"></td>
    </tr>
</table>

The wrong `elephant` and `cow` guesses stay, because nothing tells the model they are wrong. The new class is added next to them, and picking the higher score at inference is enough to separate them. A class unlike anything in COCO, such as licence plates on 2325 street images, reaches 0.564 mAP50 in 50 epochs and was still improving, so classes far from the pretrained features want a longer run.

## Compared With a Normal Fine-Tune

Training every layer on the new data reaches a slightly better score on the new class and destroys the old ones. Both runs below use `yolo26n`, `rhino` as class 80, 262 images annotated for `rhino` only and 100 epochs. Old-class accuracy is measured on COCO val2017, which no run trained on.

| Training                                      | `rhino` mAP50-95 | COCO mAP50-95 | COCO kept |
| --------------------------------------------- | ---------------- | ------------- | --------- |
| Pretrained model, for reference               | none             | 0.395         | 100%      |
| **`RefineDetectionTrainer`**                  | **0.831**        | **0.394**     | **99.8%** |
| Every layer trainable                         | 0.852            | 0.036         | 9%        |
| Every layer trainable, head rebuilt for 81    | 0.809            | 0.000         | 0%        |
| Every layer trainable, 1000 COCO images added | 0.846            | 0.272         | 69%       |

The damage lands early: by epoch 11 the unfrozen run is down to 0.188 COCO mAP50-95 and has not yet learned anything about `rhino`, so stopping early does not avoid the trade. Mixing the original data back in needs the original dataset, trains on five times the images, and still brings the old classes back at only 69%.

## FAQ

### Can I add a second class later?

Yes. Training an already tuned model stacks a second branch on it and freezes the first, so the classes of the earlier session keep their predictions:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import RefineDetectionTrainer

# session 1 added class 80, session 2 adds class 81 and keeps class 80
model = YOLO("runs/detect/train/weights/best.pt")
model.train(data="data-2.yaml", epochs=50, classes=[81], trainer=RefineDetectionTrainer)
```

Every branch stays in the model, about +2.0% parameters and +1.7% GFLOPs each on YOLO26n, while one session tuning several classes costs the same as one class. Pass the classes together whenever you know them up front.

### How do I resume an interrupted run?

Pass the trainer again with `resume=True` on `last.pt`, keeping the same `classes`:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import RefineDetectionTrainer

model = YOLO("runs/detect/train/weights/last.pt")
model.train(resume=True, trainer=RefineDetectionTrainer)
```

`trainer` is not stored in the checkpoint. Leaving it out on any further training of a tuned model falls back to the standard trainer, which trains the whole model and loses the guarantee without an error, so check the log for `Freezing layer` lines if unsure.

### Why is my new class not detected at all?

Check that the index in `classes` matches the position of the class in the dataset YAML and in the label files. If the trainer reports zero instances, the labels were filtered out because the indices do not match.

### No suitable base model?

Build one with [YOLOE](../models/yoloe.md), which names classes from text without training. Convert the segmentation checkpoint to a detection model as described in [YOLOE training](../models/yoloe.md#train-usage), fuse your class list into it with `get_vocab()`, and the result is a plain detector that needs no text encoder:

```python
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.detect import RefineDetectionTrainer
from ultralytics.utils.patches import torch_load

names = ["person", "bus", "rhino"]  # every class the model should detect, in the dataset YAML order

model = YOLOE("yoloe-26n.yaml")  # a Detect model, the released checkpoints are Segment
model.load(torch_load("yoloe-26n-seg.pt")["model"])
model.model.eval()
model.model.get_vocab(names)  # embeds the names and fuses them into the cls head
model.save("yoloe-det-fused.pt")

model = YOLO("yoloe-det-fused.pt")
model.train(data="data.yaml", epochs=50, classes=[2], trainer=RefineDetectionTrainer)  # tune "rhino"
```
