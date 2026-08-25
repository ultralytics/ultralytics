---
title: Fine-Tune YOLO26 on a Custom Dataset
comments: true
description: Fine-tune YOLO26 on a custom dataset using pretrained weights. Covers transfer learning, layer freezing, optimizers, two-stage training, and fixing low mAP.
keywords: fine-tune YOLO, finetune YOLO custom dataset, YOLO transfer learning, YOLO26 fine-tuning, freeze layers YOLO, pretrained YOLO custom data, YOLO training from scratch vs fine-tuning, catastrophic forgetting YOLO, two-stage fine-tuning, YOLO optimizer selection, fine-tune object detection model, custom object detection training, YOLO freeze backbone, how to finetune YOLO26
---

# How to Fine-Tune YOLO on a Custom Dataset

[Fine-tuning](https://www.ultralytics.com/glossary/fine-tuning) adapts a pretrained model to recognize new classes by starting from learned weights rather than random initialization. Instead of training from scratch for hundreds of epochs, fine-tuning leverages pretrained [COCO](../datasets/detect/coco.md) features and converges on custom data in a fraction of the time.

This guide covers fine-tuning [YOLO26](../models/yolo26.md) on custom datasets, from basic usage to advanced techniques like [layer freezing](#freezing-layers) and [two-stage training](#two-stage-fine-tuning).

## Fine-Tuning vs Training from Scratch

A pretrained model has already learned general visual features — edge detection, texture recognition, shape understanding — from millions of images. [Transfer learning](https://www.ultralytics.com/glossary/transfer-learning) through fine-tuning reuses that knowledge and only teaches the model what the new classes look like, which is why it converges faster and requires less data. Training from scratch discards all of that and forces the model to learn everything from pixel-level patterns up, which demands significantly more resources — it starts from an architecture-only [model YAML configuration](model-yaml-config.md) rather than a checkpoint.

|                       | Fine-Tuning                                          | Training from Scratch                                                 |
| --------------------- | ---------------------------------------------------- | --------------------------------------------------------------------- |
| **Starting weights**  | Pretrained on COCO (80 classes)                      | Random initialization                                                 |
| **Command**           | `YOLO("yolo26n.pt")`                                 | `YOLO("yolo26n.yaml")`                                                |
| **Convergence**       | Faster — backbone is already trained                 | Slower — all layers learn from zero                                   |
| **Data requirements** | Lower — pretrained features compensate for less data | Higher — model must learn all features from the dataset alone         |
| **When to use**       | Custom classes with natural images                   | Domains fundamentally different from COCO (medical, satellite, radar) |

!!! tip "Fine-tuning requires no extra code"

    When a `.pt` file is loaded with `YOLO("yolo26n.pt")`, the pretrained weights are stored in the model. Calling `.train(data="custom.yaml")` after that automatically transfers all compatible weights to the new model architecture, reinitializes any layers that don't match (such as the detection head when the number of classes differs), and begins training. No manual weight loading, layer manipulation, or custom transfer learning code is required.

### How Pretrained Weight Transfer Works

When a pretrained model is fine-tuned on a dataset with a different number of classes (for example, COCO's 80 classes to 5 custom classes), Ultralytics performs shape-aware weight transfer:

1. **Backbone and neck transfer fully** — these layers extract general visual features and their shapes are independent of the number of classes.
2. **Detection head is partially reinitialized** — the classification output layers (`cv3`, `one2one_cv3`) have shapes tied to the class count (80 vs 5), so they cannot transfer wholesale: rows whose class names match are remapped across first (see below), and only the unmatched or shape-incompatible remainder is randomly initialized. Box regression layers (`cv2`, `one2one_cv2`) in the head have fixed shapes regardless of class count, so they transfer normally.
3. **The vast majority of weights transfer** when changing class count. For example, fine-tuning YOLO26n from COCO (80 classes) onto a 5-class dataset whose names do not match any COCO class transfers 606 of 708 weight tensors: only the class-count-dependent classification layers are reinitialized, while the backbone, neck, and box regression branches remain intact. Names that do match transfer their classification rows too, as described below.

For datasets with the same number of classes as the pretrained model (for example, fine-tuning COCO-pretrained weights on another 80-class dataset), 100% of weights transfer including the detection head.

### Transfer Classes with Name Aliases

Ultralytics transfers matching classification-head rows by class name across datasets, ignoring case and surrounding whitespace. When equivalent classes use different names, rename the source checkpoint classes in memory before loading. This preserves pretrained classification weights for shared concepts that would otherwise be treated as unmatched and initialized randomly.

This [Objects365](../datasets/detect/objects365.md) v2-to-COCO example renames the source classes on the loaded checkpoint, which `train()` then passes through as the pretrained weights:

```python
from ultralytics import YOLO

# Source Objects365 v2 name (lowercased) -> target COCO name
ALIASES = {
    "wild bird": "bird",
    "handbag/satchel": "handbag",
    "luggage": "suitcase",
    "bowl/basin": "bowl",
    "orange/tangerine": "orange",
    "monitor/tv": "tv",
    "stuffed toy": "teddy bear",
    "hair dryer": "hair drier",
}

model = YOLO("path/to/yolo26s-objects365.pt")
# Objects365 class names are Title-Cased, so match on the lowercased name
model.model.names = {i: ALIASES.get(name.lower(), name) for i, name in model.model.names.items()}
model.train(data="coco.yaml", epochs=100, imgsz=640)
```

The matching itself is performed by `cls_remap`, which is enabled by default: for every class in the target dataset, the pretrained classification row whose name matches is copied across, whether the class counts differ or only the ordering does. Each head tensor transfers only where its shape still matches: on `yolo26n` the classifier width follows `max(64, min(nc, 100))`, so an 80-class to 5-class transfer moves the matched bias rows alone, while `yolo26s` and larger hold that width fixed and move the weights as well. The aliases above exist purely to make names that mean the same thing match textually. Training prints `Remapped N/M cls head rows from pretrained weights by class name` when the remap fires, which is the quickest way to confirm the aliases took effect; the separate `Transferred X/Y items from pretrained weights` line counts tensors rather than rows and looks the same either way. Set `cls_remap=False` to skip name matching entirely — rows then transfer positionally in the pretrained model's own class order wherever the shapes still match, and are reinitialized only where the class counts differ.

## Basic Fine-Tuning Example

!!! example "Fine-tune YOLO26 on a custom dataset"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")  # load pretrained model
        model.train(data="custom.yaml", epochs=50, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train model=yolo26n.pt data=custom.yaml epochs=50 imgsz=640
        ```

### Choosing a Model Size

Larger models have more capacity but also more parameters to update, which can increase the risk of overfitting when training data is limited. Starting with a smaller model (YOLO26n or YOLO26s) and scaling up only if validation metrics plateau is a practical approach. The optimal model size depends on the complexity of the task, the number of classes, the diversity of the dataset, and the hardware available for deployment. See the full [YOLO26 model page](../models/yolo26.md) for available sizes and performance benchmarks. If deployment constraints force a small model but a larger one fine-tunes better, close the gap with [knowledge distillation](knowledge-distillation.md) rather than scaling up.

## Optimizer and Learning Rate Selection

The default `optimizer=auto` setting selects the optimizer and learning rate based on the total number of training iterations:

- **10,000 iterations or fewer** (small datasets or few epochs): AdamW with a low, auto-calculated learning rate
- **More than 10,000 iterations** (large datasets): [MuSGD](../reference/optim/muon.md) (a hybrid Muon+SGD optimizer) with lr=0.01

Iterations are counted as `ceil(dataset_size / max(batch, nbs)) * epochs`, where `nbs` is the nominal batch size of 64 and `dataset_size` is the training split after `fraction` is applied. A batch smaller than 64 therefore does not raise the count, and the rounding up matters near the threshold: 65 images at `batch=64` count as 2 iterations per epoch, not 1.

For most fine-tuning tasks, the default setting works well without any manual tuning. Consider setting the optimizer explicitly when:

- **Training is unstable** (loss spikes or diverges): try `optimizer=AdamW, lr0=0.001` for more stable convergence
- **Fine-tuning a large model on a small dataset**: a lower learning rate like `lr0=0.001` helps preserve pretrained features

!!! warning "Auto optimizer overrides manual lr0"

    When `optimizer=auto`, the `lr0` and `momentum` values are ignored. To control the learning rate manually, set the optimizer explicitly: `optimizer=SGD, lr0=0.005`.

## Freezing Layers

The `freeze` parameter prevents specific YOLO layers from updating during training. This speeds up training and reduces [overfitting](https://www.ultralytics.com/glossary/overfitting) when the dataset is small relative to the model capacity.

The `freeze` parameter accepts either an integer or a list. An integer `freeze=10` freezes the first 10 layers (indices 0-9), which covers most of the YOLO26 backbone. The backbone spans layers 0-10, so `freeze=10` leaves the final C2PSA block (layer 10) trainable; use `freeze=11` to freeze the entire backbone. A list can contain layer indices like `freeze=[0, 3, 5]` for partial backbone freezing, or module name strings like `freeze=["23.cv2", "23.one2one_cv2"]` for fine-grained control over specific branches within a layer (here, both box regression branches of the detection head).

!!! example

    === "Freeze backbone"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.train(data="custom.yaml", epochs=50, freeze=10)
        ```

    === "Freeze specific layers"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.train(data="custom.yaml", epochs=50, freeze=[0, 1, 2, 3, 4])
        ```

    === "Freeze by module name"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Freeze both box regression branches (one-to-many and end-to-end) of the detection head
        model.train(data="custom.yaml", epochs=50, freeze=["23.cv2", "23.one2one_cv2"])
        ```

The right freeze depth depends on how similar the target domain is to the pretrained data and how much training data is available:

| Scenario                      | Recommendation          | Rationale                                                   |
| ----------------------------- | ----------------------- | ----------------------------------------------------------- |
| Large dataset, similar domain | `freeze=None` (default) | Enough data to adapt all layers without overfitting         |
| Small dataset, similar domain | `freeze=10`             | Preserves backbone features, reduces trainable parameters   |
| Very small dataset            | `freeze=23`             | Only the detection head trains, minimizing overfitting risk |
| Domain far from COCO          | `freeze=None`           | Backbone features may not transfer well and need retraining |

Freeze depth can also be treated as a hyperparameter — trying a few values (0, 5, 10) and comparing validation mAP is a practical way to find the best setting for a specific dataset.

The indices above describe YOLO26 detect checkpoints. Segmentation, pose, and OBB checkpoints have their own layer counts, so read the topology off the checkpoint you are fine-tuning rather than reusing a number from this page:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt").model
print(f"last index = {len(model.model) - 1} -> {type(model.model[-1]).__name__}")
```

Training logs one `Freezing layer '<name>'` line per frozen parameter, which is the authoritative record of what was actually held fixed.

## Key Hyperparameters for Fine-Tuning

Fine-tuning generally requires fewer hyperparameter adjustments than training from scratch. The parameters that matter most are:

- **`epochs`**: Fine-tuning converges faster than training from scratch. Start with a moderate value and use `patience` to stop early when validation metrics plateau.
- **`patience`**: The default of 100 is designed for long training runs. Reducing this to 10-20 avoids wasting time on runs that have already converged.
- **`warmup_epochs`**: The default warmup (3 epochs) gradually increases the learning rate from zero, which prevents large gradient updates from damaging pretrained features in early iterations. Keeping the default is recommended even for fine-tuning.
- **`close_mosaic`**: The default of 10 disables mosaic — along with `copy_paste`, `mixup`, and `cutmix` — for the final 10 epochs. On short fine-tuning runs that default lands badly: at `epochs=10` mosaic is off for the entire run, at `epochs=12` it is active for only the first two, and below 10 epochs it never switches off at all, so the stabilizing phase never happens. Set `close_mosaic` to roughly the last 10-20% of your `epochs` value whenever the run is short.

For the full list of training parameters, see the [training configuration reference](../usage/cfg.md). For behavior the parameters do not expose — per-layer learning rates, gradient clipping, or custom validation metrics — [subclass the trainer](custom-trainer.md).

## Two-Stage Fine-Tuning

Two-stage fine-tuning splits training into two phases. The first stage freezes the backbone and trains only the neck and head, allowing the detection layers to adapt to the new classes without disrupting pretrained features. The second stage unfreezes all layers and trains the full model with a lower learning rate to refine the backbone for the target domain.

This approach is particularly useful when the target domain differs significantly from COCO (medical images, aerial imagery, microscopy), where the backbone may need adaptation but training everything at once causes instability. For automatic unfreezing with a callback-based approach, see [Freezing and Unfreezing the Backbone](custom-trainer.md#freezing-and-unfreezing-the-backbone).

!!! example "Two-stage fine-tuning"

    ```python
    from ultralytics import YOLO

    # Stage 1: freeze backbone, train head and neck
    model = YOLO("yolo26n.pt")
    model.train(data="custom.yaml", epochs=20, freeze=10, name="stage1", exist_ok=True)

    # Stage 2: unfreeze all, fine-tune with lower lr
    model = YOLO("runs/detect/stage1/weights/best.pt")
    model.train(data="custom.yaml", epochs=30, lr0=0.001, name="stage2", exist_ok=True)
    ```

!!! warning "`resume` does not extend a finished run"

    `resume=True` continues a run that was interrupted mid-training, and `epochs` is not one of the arguments it accepts as an override. Once a run completes, its `last.pt` no longer carries optimizer or epoch state, so `resume=True` logs a warning about a non-resumable checkpoint and quietly starts a fresh run in a new directory. That run keeps only `imgsz`, `data`, `task`, and `single_cls` from the checkpoint; every other setting reverts to its default unless you pass it again. To train an already-converged model further, chain a new run from its weights the way stage 2 does above, pointing at `best.pt` and lowering `lr0` because the model is no longer starting cold. See [Resuming Interrupted Trainings](../modes/train.md#resuming-interrupted-trainings) for the interrupted-run case.

## Common Pitfalls

### Model produces no predictions

- **Insufficient training data**: training with very few samples is the most common cause — the model cannot learn or generalize from too little data. Ensure enough diverse examples per class before investigating other causes.
- **Check dataset paths**: incorrect paths in `data.yaml` silently produce zero labels. Validate the dataset before training to confirm labels load correctly:

    ```bash
    yolo detect val model=yolo26n.pt data=custom.yaml
    ```

- **Lower confidence threshold**: if predictions exist but are filtered out, try `conf=0.1` during inference.
- **Verify class count**: ensure `nc` in `data.yaml` matches the actual number of classes in the label files.

### Validation mAP plateaus early

- **Add more data**: fine-tuning benefits significantly from additional training data, especially diverse examples with varied angles, lighting, and backgrounds.
- **Check class balance**: underrepresented classes will have low AP. Use `cls_pw` to apply inverse frequency class weighting (start with `cls_pw=0.25` for moderate imbalance, increase to `1.0` for severe imbalance).
- **Reduce augmentation**: for very small datasets, heavy augmentation can hurt more than it helps. Try `mosaic=0.5` or `mosaic=0.0`.
- **Increase resolution**: for datasets with small objects, try `imgsz=1280` to preserve detail.

### Performance degrades on original classes after fine-tuning

A YOLO model that loses accuracy on its original COCO classes after fine-tuning is exhibiting catastrophic forgetting — previously learned knowledge is overwritten when the model trains exclusively on new data. Forgetting is mostly unavoidable without including original dataset images alongside new data. To mitigate this:

- **Merge datasets**: include examples of the original classes alongside the new classes during fine-tuning. This is the only reliable way to prevent forgetting.
- **Freeze backbone and neck**: freezing both the backbone and neck so only the detection head trains helps for short fine-tuning runs with a very low learning rate.
- **Train for fewer epochs**: the longer the model trains on new data exclusively, the more forgetting increases.

## Conclusion

Fine-tuning YOLO26 needs no custom transfer-learning code: load a `.pt` checkpoint, point `.train()` at your `data.yaml`, and Ultralytics transfers every compatible weight while reinitializing only the class-dependent head layers. Reach for `freeze` and a lower `lr0` when the dataset is small, and for two-stage training when the domain is far from COCO. From there, search the remaining settings automatically with the [hyperparameter tuning guide](hyperparameter-tuning.md), or start from the published defaults in the [YOLO26 training recipe](yolo26-training-recipe.md).

## FAQ

### How many images do I need to fine-tune YOLO?

Plan on at least a few hundred labeled images per class for a domain close to COCO, and more as the classes get visually similar or the domain gets further from natural images. There is no hard minimum — diversity of lighting, angles, and backgrounds matters more than raw count. On a small dataset, [K-fold cross-validation](kfold-cross-validation.md) gives a more reliable read on whether the model is actually improving than a single train/val split does.

### How do I fine-tune YOLO26 on a custom dataset?

Load a pretrained `.pt` file and call `.train()` with the path to a custom `data.yaml`. Ultralytics automatically handles [weight transfer](https://www.ultralytics.com/glossary/transfer-learning), detection head reinitialization, and optimizer selection. See the [Basic Fine-Tuning](#basic-fine-tuning-example) section for the complete code example.

### Why is my fine-tuned YOLO model not detecting anything?

The most common causes are incorrect paths in `data.yaml` (which silently produces zero labels), a mismatch between `nc` in the YAML and the actual label files, or a confidence threshold that is too high. See [Common Pitfalls](#model-produces-no-predictions) for a full troubleshooting checklist.

### Which YOLO layers should I freeze for fine-tuning?

It depends on the dataset size and domain similarity. For small datasets with a domain similar to COCO, freezing the backbone (`freeze=10`) prevents overfitting. For domains very different from COCO, leaving all layers unfrozen (`freeze=None`) allows the backbone to adapt. See [Freezing Layers](#freezing-layers) for detailed recommendations.

### Can I fine-tune YOLO26 for segmentation, pose, or OBB the same way?

Yes. Load the task-specific pretrained checkpoint — `yolo26n-seg.pt`, `yolo26n-pose.pt`, or `yolo26n-obb.pt` — and call `.train()` with a `data.yaml` in that task's label format. Weight transfer, head reinitialization, `freeze`, and optimizer selection all work the same way across tasks, though each task has its own head layers, so the indices in this guide describe detect checkpoints specifically. See the [task pages](../tasks/index.md) for the label format each one expects.

### How do I prevent catastrophic forgetting when fine-tuning YOLO on new classes?

Include examples of the original classes in the training data alongside the new classes. If that is not possible, freezing more layers (`freeze=10` or higher) and using a lower learning rate helps preserve the pretrained knowledge. See [Performance degrades on original classes](#performance-degrades-on-original-classes-after-fine-tuning) for more details.
