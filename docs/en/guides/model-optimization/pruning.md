---
comments: true
description: Learn how to prune YOLOv8 Detection models for faster inference and smaller size while maintaining accuracy.
keywords: Ultralytics, YOLO, pruning, model compression, optimization, computer vision
---

# Pruning

> **Note:** Pruning is currently supported only for **YOLOv8 detection models**.

Model pruning is an important optimization technique that reduces model size and inference latency by removing less significant parameters, while aiming to maintain accuracy.  
For background, see the Ultralytics Glossary entries on [Pruning](https://www.ultralytics.com/glossary/pruning) and [Model Pruning](https://www.ultralytics.com/glossary/model-pruning).

> **Note:** Structured pruning alters model architecture. Retraining is strongly recommended so the remaining weights can adapt to the reduced structure.

Ultralytics provides a built-in pruning utility that allows two modes of operation:

1. **Global pruning ratio** — apply a single pruning percentage to all supported layers.
2. **Layer-specific pruning** — define custom ratios for each model component in a YAML file.

---

## Setup

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
```

The `prune_detection_model()` function takes a YOLO model instance and returns a pruned version of it.

---

## 1. Global Pruning Ratio

```python
from ultralytics.utils.prune import prune_detection_model

# Prune 25% of channels globally
pruned = prune_detection_model(model, prune_ratio=0.25)
pruned.save("y8s-pruned.pt")
```

---

## 2. Per-Layer YAML Configuration

```python
from ultralytics.utils.prune import prune_detection_model

# Prune using a YAML-defined ratio for each component
pruned = prune_detection_model(model, prune_yaml="ultralytics/cfg/pruning/sample_prune.yaml")
pruned.save("y8s-pruned.pt")
```

Example structure of sample_prune.yaml:

```yaml
prune_ratios:
    # Backbone
    0: 0.1 # Conv 0 - P1/2
    1: 0.25 # Conv 1 - P2/4
    2: 0.25 # C2f 2 - first backbone C2f
    3: 0.25 # Conv 3 - P3/8
    4: 0.25 # C2f 4 - second backbone C2f
    5: 0.25 # Conv 5 - P4/16
    6: 0.25 # C2f 6 - third backbone C2f
    7: 0.25 # Conv 7 - P5/32
    8: 0.25 # C2f 8 - fourth backbone C2f
    9: 0.0 # SPPF 9

    # Head
    10: null # Upsample
    11: null # Concat - cat backbone P4
    12: 0.5 # C2f 12 - head P4
    13: null # Upsample
    14: null # Concat - cat backbone P3
    15: 0.5 # C2f 15 - head P3 / small
    16: 0.25 # Conv
    17: null # Concat - cat head P4
    18: 0.5 # C2f 18 - head P4 / medium
    19: 0.25 # Conv
    20: null # Concat - cat head P5
    21: 0.5 # C2f 21 - head P5 / large
    22: [0.0, 0.0] # Detect - [regression tower, classification tower]
```

Each index (0–22) corresponds to the components in the [YOLOv8 model definition](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml).
The optimal pruning ratios depend on your hardware, latency constraints, and acceptable accuracy drop. Experiment to find a suitable balance.

---

## Fine-Tuning Pruned Models

Retraining is recommended to recover accuracy after pruning:

```python
pruned.train(data="coco128.yaml", epochs=50)
```

---

## Exporting Pruned Models

> **Note:** ONNX export requires the `onnxscript` package (`pip install onnxscript`).

Export pruned models as usual:

```python
pruned.export(format="onnx")
```

---

## Trade-offs

- Pruning reduces model size and speeds up inference by removing less important parameters. However, this reduction in capacity leads to an immediate drop in accuracy since the model must re-learn how to represent patterns with fewer weights. Fine-tuning typically restores much of this loss, though some degradation may persist if the remaining model capacity is insufficient for the task.
- In many real-world cases—especially when pre-trained models are adapted to simpler downstream problems—pruned models achieve nearly the same accuracy while offering significant speed and memory benefits. This trade-off between compactness and performance is central to model optimization.
