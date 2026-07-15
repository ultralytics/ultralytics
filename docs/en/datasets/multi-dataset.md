---
title: Multi-Dataset Training & Semantic Class Resolution
comments: true
description: Learn how to train YOLO models on multiple datasets simultaneously using Native Semantic Class Resolution to automatically unify class IDs.
keywords: multi-dataset training, semantic class resolution, class ID remapping, joint training, YOLO, Ultralytics
---

# Multi-Dataset Training & Semantic Class Resolution

## Introduction

Fine-tuning or pre-training vision models on multiple distinct datasets simultaneously is a powerful way to enhance model generalization and robustness. However, different datasets often map the same semantic objects to different numeric class IDs. 

For instance, consider two independent datasets:

=== "Dataset A (`dataset_a.yaml`)"

    ```yaml
    names:
      0: blood
      1: stool
    ```

=== "Dataset B (`dataset_b.yaml`)"

    ```yaml
    names:
      0: stool
      1: blood
    ```

Here, the semantic class **blood** is class `0` in Dataset A but class `1` in Dataset B, while **stool** is class `1` in Dataset A and class `0` in Dataset B. Training directly on both datasets simultaneously would lead to catastrophic class ID conflicts and confuse the model.

Ultralytics natively solves this issue with **Semantic Class Resolution**. The framework automatically aligns and translates class IDs at runtime based on their semantic class names, requiring no manual rewriting of annotations or configuration tables.

---

## How It Works

During training initialization, the framework intercepts the multi-dataset configuration and resolves class IDs before annotations are processed by the training pipeline:

```
Dataset A                           Dataset B
0: blood                            0: stool
1: stool                            1: blood
   │                                   │
   └───────────────┬───────────────────┘
                   │
                   v
       Semantic Class Resolution
                   │
                   v
         Global Class Registry
             0: blood
             1: stool
                   │
                   v
         Unified Training Labels
```

1. **Class Names as Source of Truth**: The framework treats class name strings as the canonical source of truth for semantic classes.
2. **Automatic Remapping**: An internal, read-only global class registry is established. The framework maps the local numeric class IDs of each dataset to their corresponding global class IDs on-the-fly.
3. **No Annotation Modifications**: Label files (`*.txt` files containing annotations) are never modified. Caches are built and loaded using local IDs and mapped only at runtime, ensuring that your dataset caches remain reusable for single-dataset tasks.

---

## Dataset Configuration

To enable multi-dataset training, create a parent dataset configuration YAML that lists the target sub-dataset YAML files under the `datasets:` key.

### Automatic Class Discovery

If no global `names` mapping is provided in the parent YAML, the framework automatically builds a global registry from the union of all classes in the sub-datasets. The global ordering is determined deterministically by the sequence of datasets in the list and local class ID sorting.

!!! example "combined.yaml (Auto-Discovery)"

    ```yaml
    datasets:
      - path/to/dataset_a.yaml
      - path/to/dataset_b.yaml
    ```

### User-Defined Global Ordering

You can optionally specify a canonical global class names list in the parent configuration YAML. This forces a specific global ordering of class IDs in the trained model.

!!! example "combined.yaml (User-Defined)"

    ```yaml
    names:
      - blood
      - stool
      - cells

    datasets:
      - path/to/dataset_a.yaml
      - path/to/dataset_b.yaml
    ```

### Extensible List Schema

The `datasets:` list supports both simple string paths and dictionary definitions. This allows you to define per-dataset properties (e.g. sample weights or custom overrides) in the future:

!!! example "combined.yaml (Extensible Schema)"

    ```yaml
    datasets:
      - path/to/dataset_a.yaml
      - data: path/to/dataset_b.yaml
        weight: 2.0  # Optional future per-dataset configuration
    ```

---

## Training Example

To train a model on the combined dataset configuration, simply pass the parent YAML path to the `data` parameter of `.train()`.

!!! example "Multi-Dataset Training"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained detection model
        model = YOLO("yolo11n.pt")

        # Train on the multi-dataset configuration
        results = model.train(data="combined.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train on the multi-dataset configuration using the CLI
        yolo detect train data=combined.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

---

## Supported Tasks

Semantic Class Resolution is fully compatible and verified across the following computer vision tasks:

- **Object Detection** ✅
- **Instance Segmentation** ✅

---

## Validation Rules

To prevent training issues or silent misalignments, the framework strictly validates the following rules at configuration load time:

- **Duplicate Class Names**: Raises a `ValueError` if a dataset contains duplicate class name definitions.
- **Unknown Classes**: If a user-defined parent `names` list is provided, any sub-dataset containing a class name not defined in the parent list will trigger a `ValueError` with clear details.
- **Malformed Schemas**: Raises a `ValueError` if dictionary entries in the `datasets` list lack a `'data'` or `'path'` reference.

---

## Backward Compatibility

Single-dataset training configurations remain completely unaffected. The semantic resolution logic activates only when a `datasets:` list is defined in the primary data YAML.

```python
# Continues to work exactly as before, with no changes needed.
model.train(data="coco8.yaml")
```

---

## FAQ

### Do I need to modify my annotation label files?
No. All class remapping is executed dynamically inside PyTorch dataloading memory during training and validation. The original annotation files are never touched.

### Do all sub-datasets need to have the same class IDs?
No. Sub-datasets can define classes in different orders or map classes to different IDs. They only need matching semantic class names (e.g., `"blood"` matching `"blood"` case-sensitively).

### Are dataset cache files affected?
No. Cache files (`.cache` files) store the original dataset-local IDs. Because resolution is applied after cache loading, the same local dataset caches can be safely shared and reused between single-dataset and multi-dataset training runs without rebuilding.
