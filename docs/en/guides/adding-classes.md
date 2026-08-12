---
comments: true
description: Add a new class to a trained Ultralytics YOLO model, or tune an existing one, without changing the predictions of the other classes.
keywords: add class to YOLO, incremental learning, catastrophic forgetting, fine-tune single class, RefineDetectionTrainer, Ultralytics YOLO, object detection
---

# Adding and Fine-Tuning Classes on a Trained Model

`RefineDetectionTrainer` tunes the classes you name and leaves every other class of a trained detection model untouched. Use it to add a class a pretrained model does not have, or to improve one it already predicts, without collecting the original dataset again.

## Before You Start

This trainer freezes the whole model except a small branch on the detection head, which is what keeps the other classes intact. That constraint comes with real limits.

!!! warning "Limitations"

    - **The [backbone](https://www.ultralytics.com/glossary/backbone) learns nothing new.** Your class is recognized from the features the pretrained model already has. This works when the new class looks like something the model has seen, such as a rhino against a [COCO](../datasets/detect/coco.md) model that knows elephants and cows. It works poorly for a domain the backbone has never encountered, such as X-ray, satellite or microscopy imagery.
    - **The dataset YAML must list every class of the pretrained model**, in the order the tuned model should use, plus the new names. Classes missing from it are dropped from the head.
    - **The base model must be trained.** An untrained or randomly initialized backbone has no features to reuse, so there is nothing to freeze and nothing to build on. Start from a pretrained checkpoint.
    - **Tuning an existing class can make it worse elsewhere.** Nothing here protects a class from its own training data. Tuning `person`, learned from tens of thousands of COCO images, on ten images of your site pulls it towards those ten images, and it can lose accuracy on everything else. The guarantee covers the classes you do not name, not the ones you do. Tune an existing class only when you have enough images to represent it, or add your own class instead and leave the original alone.
    - **Detect only.** Segment, Semantic, Classify, Pose and OBB are work in progress.

!!! tip "No suitable base model?"

    If no trained model covers your domain, build the base model with [YOLOE](../models/yoloe.md). It names classes from text with no training, and `get_vocab()` bakes that class list into the classification head, leaving a plain detector that needs no text encoder at inference and costs the same to run.

    Convert the segmentation checkpoint to a detection model first, as described in [YOLOE training](../models/yoloe.md#train-usage), then fuse your class list into it:

    ```python
    from ultralytics import YOLOE
    from ultralytics.utils.patches import torch_load

    names = ["person", "bus", "rhino"]  # every class the model should detect, in the dataset YAML order

    model = YOLOE("yoloe-26n.yaml")  # a Detect model, the released checkpoints are Segment
    model.load(torch_load("yoloe-26n-seg.pt")["model"])
    model.model.eval()
    model.model.get_vocab(names)  # embeds the names and fuses them into the cls head
    model.save("yoloe-det-fused.pt")
    ```

    Load `yoloe-det-fused.pt` with `YOLO` and tune it as shown below. Keep `names` in the same order as the dataset YAML, so `classes` indexes the same list in both. Segmentation checkpoints stay segmentation models and are not accepted, since this trainer is Detect only.

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

The detection head gains a small refinement branch, one per detection level, built from the same depthwise blocks as the existing classification branch, a quarter as wide and twice as deep. It predicts a class score adjustment for the tuned classes and a box adjustment gated by their confidence, and starts zero-initialized so an attached model predicts exactly what it did before training.

Everything else is frozen, including [batch normalization](https://www.ultralytics.com/glossary/batch-normalization) statistics. The classification output rows of the untuned classes are restored after every optimizer step, so weight decay and momentum cannot move them either.

The cost for one added class, measured at 640 pixels after `fuse`:

| Model                                                          | Parameters       | GFLOPs |
| -------------------------------------------------------------- | ---------------- | ------ |
| [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26) | 2.42M to 2.47M   | +1.7%  |
| [YOLO26s](https://platform.ultralytics.com/ultralytics/yolo26) | 9.51M to 9.68M   | +1.5%  |
| [YOLO26m](https://platform.ultralytics.com/ultralytics/yolo26) | 20.44M to 20.72M | +1.3%  |

The result is a normal checkpoint. [Prediction](../modes/predict.md), [validation](../modes/val.md) and [export](../modes/export.md) all work as usual.

## Example Results

Two `yolo26n` runs added class 80 to COCO without changing the original 80 class scores:

| Added class     | Training data | Epochs | mAP50 | mAP50-95 | Maximum original-score change |
| --------------- | ------------- | ------ | ----- | -------- | ----------------------------- |
| `rhino`         | 559 boxes     | 30     | 0.747 | 0.672    | 5.7e-14                       |
| `license-plate` | 2325 images   | 50     | 0.564 | 0.244    | 2.3e-13                       |

The rhino run used the [African Wildlife](../datasets/detect/african-wildlife.md) images. COCO already has visually related large animals, so its frozen features transfer well:

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

The wrong `elephant` and `cow` guesses stay because they were not tuned; the new class is added beside them. The lower license-plate result shows the tradeoff when the frozen backbone has no closely related features: preserving old classes costs accuracy on the new one.

## FAQ

### Are the other classes really unchanged?

Their class scores are bit-identical. Boxes are shared by every class, so the box adjustment can shift a box on the few anchors where a tuned class is confident. Class predictions are never affected.

### Can I add a second class later?

Yes. Training a tuned checkpoint stacks and trains another branch while freezing the earlier one:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import RefineDetectionTrainer

model = YOLO("runs/detect/train/weights/best.pt")
model.train(data="data-2.yaml", epochs=50, classes=[81], trainer=RefineDetectionTrainer)
```

Each session adds about **2.0% parameters** and **1.7% GFLOPs** to YOLO26n. A single branch can tune several classes at nearly the same cost, so pass them together as `classes=[80, 81]` when possible.

### How do I resume an interrupted run?

Pass the trainer again along with `resume=True`:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import RefineDetectionTrainer

model = YOLO("runs/detect/train/weights/last.pt")
model.train(resume=True, trainer=RefineDetectionTrainer)
```

Resume from `last.pt`, which carries the optimizer state and epoch counter. Keep these constraints:

- **Pass `trainer` again.** It is not stored in the checkpoint.
- **Do not change `classes`.** The run being resumed already owns a branch, and the trainer refuses to continue if the two disagree. Change `epochs`, `data` or `classes` only by starting a new session.
- **Do not resume from `best.pt`.** It starts a new session and adds another branch; `last.pt` continues the interrupted one.

!!! warning

    Leaving `trainer` out falls back to the standard trainer, which trains the whole model and silently loses the guarantee that the other classes stay unchanged. There is no error, so check the training log for `Freezing layer` lines if you are unsure the right trainer ran. This applies to any further training of a tuned checkpoint, not only to resuming.

### Why is my new class not detected at all?

Check that the class index in `classes` matches its position in the dataset YAML, and that the label files use that same index. If the trainer reports zero instances during training, the labels were filtered out because their indices do not match.

### Can I use this for the other tasks?

Not yet. `RefineDetectionTrainer` covers Detect only. Segment, Semantic, Classify, Pose and OBB are work in progress. For related approaches see [Knowledge Distillation](knowledge-distillation.md) and the [fine-tuning guide](finetuning-guide.md).
