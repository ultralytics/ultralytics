---
comments: true
description: Learn how to run K-Fold Cross-Validation on object detection datasets with Ultralytics YOLO. Split data into folds, train on each, and aggregate metrics and confusion matrices for a reliable performance estimate.
keywords: Ultralytics YOLO, K-Fold Cross-Validation, object detection, cross validation, dataset split, sklearn KFold, model evaluation, mAP, confusion matrix, pandas, machine learning, model robustness
---

# K-Fold Cross-Validation with Ultralytics YOLO

K-Fold Cross-Validation splits your dataset into `k` folds and trains `k` models, each validated on a different fold so that every labeled image is used for validation exactly once. This produces a more reliable estimate of object detection performance, along with its variance, than a single train/val split can provide. This guide walks through preparing the splits, training Ultralytics YOLO on each fold, and aggregating the results.

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/k-fold-cross-validation-overview.avif" alt="K-fold cross validation data splitting">
</p>

The workflow uses the YOLO detection format with scikit-learn, pandas, and PyYAML. The examples apply `k=5` folds to the African Wildlife dataset, and the same steps work for any number of folds on datasets that use the standard `images/` and `labels/` directory layout.

## Why Use K-Fold Cross-Validation?

A single train/val split measures performance on just one slice of your data, which can mislead you if that slice happens to be unusually easy, hard, or unbalanced. K-Fold Cross-Validation addresses this by:

- **Using all labeled data for both training and validation** — every labeled image is validated exactly once, so none is locked away in a fixed holdout.
- **Reporting variance, not just a single number** — the spread of metrics across folds shows how stable the model is between data splits.
- **Reducing the risk of [overfitting](https://www.ultralytics.com/glossary/overfitting) to a lucky split** — averaging across folds gives a more generalizable estimate of real-world accuracy.

The workflow has four stages: [generate feature vectors](#generate-feature-vectors), [split the dataset into folds](#split-the-dataset-into-folds), [train on each fold](#train-on-each-fold), and [aggregate the results](#aggregating-results-across-folds).

## Setup

Install Ultralytics along with the libraries used for splitting and bookkeeping:

```bash
pip install -U ultralytics scikit-learn pandas pyyaml
```

The examples use the [African Wildlife](../datasets/detect/african-wildlife.md) dataset, a 4-class detection dataset of 1504 labeled images that Ultralytics downloads automatically. Its instance counts per class are:

| Class    | Instances |
| :------- | :-------: |
| Buffalo  |    554    |
| Elephant |    748    |
| Rhino    |    559    |
| Zebra    |    824    |

Download the dataset so its label files are available locally, then use the returned root as `dataset_path` in the following sections:

```python
from ultralytics.data.utils import check_det_dataset

data = check_det_dataset("african-wildlife.yaml")  # downloads to your datasets dir on first call
```

To run the workflow on your own data instead, point the paths below at any dataset in the [YOLO detection format](../datasets/detect/index.md).

!!! tip "Manage your datasets"

    You can store, annotate, and version detection datasets on the [Ultralytics Platform](../platform/data/index.md) and export them in YOLO format, ready for the K-Fold workflow below.

## Generate Feature Vectors

K-Fold splitting works on one row per image, but a detection image holds many objects across several classes. To summarize each image as a single row, build a feature vector that counts the instances of every class it contains. These per-image counts let you check how evenly classes fall across the folds created below.

1. Start by creating a new `example.py` Python file for the steps below.

2. Pool all label files across the dataset's existing train/val/test splits.

    ```python
    from pathlib import Path

    dataset_path = Path(data["path"])  # root returned by check_det_dataset; or your own dataset directory
    labels = sorted((dataset_path / "labels").rglob("*.txt"))  # pools train/val/test; ignores any prior fold output
    ```

3. Read the class names from the dataset config and extract the class indices.

    ```python
    classes = data["names"]  # e.g. {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}
    cls_idx = sorted(classes.keys())
    ```

4. Initialize an empty `pandas` DataFrame.

    ```python
    import pandas as pd

    index = [label.stem for label in labels]  # uses base filename as ID (no extension)
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)
    ```

5. Count the instances of each class-label present in the annotation files.

    ```python
    from collections import Counter

    for label in labels:
        lbl_counter = Counter()

        with open(label) as lf:
            lines = lf.readlines()

        for line in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(line.split(" ", 1)[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`
    ```

6. The resulting DataFrame has one row per image and one column per class, holding the count of each class in that image:

    ```
               0    1    2    3
    1 (103)  1.0  0.0  0.0  0.0
    1 (121)  1.0  0.0  0.0  0.0
    1 (128)  1.0  0.0  0.0  0.0
    ...      ...  ...  ...  ...
    4 (88)   0.0  0.0  0.0  1.0
    4 (91)   0.0  0.0  0.0  1.0
    4 (97)   0.0  0.0  0.0  2.0
    ```

Each row is a pseudo feature-vector that summarizes an image by its class composition, which later lets you verify that classes are spread reasonably across the folds.

!!! tip "Works for segmentation, pose, and OBB too"

    The class index is the first value on every YOLO label line across detection, segmentation, pose, and OBB, so the same feature-vector logic builds folds for any of these tasks without changes.

## Split the Dataset into Folds

1. Now we will use the `KFold` class from `sklearn.model_selection` to generate `k` splits of the dataset.
    - Important:
        - Setting `shuffle=True` ensures a randomized distribution of classes in your splits.
        - By setting `random_state=M` where `M` is a chosen integer, you can obtain repeatable results.

    ```python
    from sklearn.model_selection import KFold

    ksplit = 5
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results

    kfolds = list(kf.split(labels_df))
    ```

2. The dataset has now been split into `k` folds, each having a list of `train` and `val` indices. We will construct a DataFrame to display these results more clearly.

    ```python
    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df.loc[labels_df.iloc[train].index, f"split_{i}"] = "train"
        folds_df.loc[labels_df.iloc[val].index, f"split_{i}"] = "val"
    ```

3. Now we will calculate the distribution of class labels for each fold as a ratio of the classes present in `val` to those present in `train`.

    ```python
    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio
    ```

    The ideal scenario is for all class ratios to be reasonably similar for each split and across classes. This, however, will be subject to the specifics of your dataset.

4. Next, we create the directories and dataset YAML files for each split.

    ```python
    import datetime

    import yaml

    from ultralytics.data.utils import IMG_FORMATS

    # Gather images in any Ultralytics-supported format and map each label to its image by filename stem
    images = sorted(p for p in (dataset_path / "images").rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
    labels_by_stem = {label.stem: label for label in labels}

    # Create the necessary directories and dataset YAML files
    save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )
    ```

5. Lastly, copy images and labels into the respective `train` or `val` directory for each split.

    ```python
    import shutil

    from tqdm import tqdm

    for image in tqdm(images, total=len(images), desc="Copying files"):
        label = labels_by_stem.get(image.stem)  # pair by filename stem, not by list position
        if label is None:
            continue  # skip background/unlabeled images (no matching .txt)
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)
    ```

!!! note "How copying behaves"

    - Images and labels are matched by filename stem, so this workflow assumes filenames are unique across the dataset (the YOLO convention). If two images in different subdirectories share a basename, key the lookup by a path relative to the `images`/`labels` roots instead.
    - Runtime scales with dataset size and disk speed.
    - Each image is copied into every split, duplicating the dataset about `k` times on disk. For large datasets, use symlinks (`os.symlink`) instead of `shutil.copy` to save space (on Windows, symlinks may require elevated permissions).
    - Background images with no label file are not part of the label-based folds and are skipped; add them to your training splits separately if you use them.

## Save Records (Optional)

Optionally, you can save the records of the K-Fold split and label distribution DataFrames as CSV files for future reference.

```python
folds_df.to_csv(save_path / "kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")
```

## Train on Each Fold

Train one model per fold, loading a fresh copy of the pretrained weights each time so the folds stay independent. Each run is saved under `project/name` (by default `runs/detect/...`), and every fold's metrics object is kept in the `results` dictionary for aggregation in the next section.

```python
from ultralytics import YOLO

weights = "yolo26n.pt"  # pretrained weights; swap for any YOLO model
results = {}

for k, dataset_yaml in enumerate(ds_yamls):
    model = YOLO(weights)  # fresh model per fold
    results[k] = model.train(
        data=dataset_yaml, epochs=100, batch=16, project="kfold_demo", name=f"fold_{k + 1}"
    )  # add any other train arguments here
```

!!! tip "Need a simple split instead of K-Fold?"

    If you only need a one-off random train/val/test split rather than cross-validation, use [`autosplit`](../reference/data/split.md):

    ```python
    from ultralytics.data.split import autosplit

    # Random one-off split (not K-Fold cross-validation)
    autosplit(path="path/to/images", weights=(0.8, 0.2, 0.0), annotated_only=True)
    ```

## Aggregating Results Across Folds

The goal of K-Fold Cross-Validation is a single, more reliable performance estimate together with its variance. Aggregate the per-fold results stored in the `results` dictionary above in two complementary ways:

- **Scalar metrics (mAP, precision, recall) are not additive** across folds, so report their mean and standard deviation. The spread across folds tells you how stable the model is between splits.
- **Confusion matrices hold raw counts**, and because the fold validation sets are disjoint and together cover the labeled set exactly once, summing the per-fold matrices yields a single cross-validated confusion matrix over every labeled image.

```python
import os

import numpy as np
import pandas as pd

from ultralytics.utils.metrics import ConfusionMatrix

# 1. Scalar metrics: mean and standard deviation across folds
metrics_df = pd.DataFrame({f"fold_{k + 1}": r.results_dict for k, r in results.items()}).T
summary = pd.concat([metrics_df, metrics_df.agg(["mean", "std"])]).round(4)
print(summary)
summary.to_csv("kfold_metrics_summary.csv")

# 2. Composite confusion matrix: sum per-fold matrices (disjoint val sets cover the dataset once)
cms = [r.confusion_matrix for r in results.values()]
composite = ConfusionMatrix(names=cms[0].names, task=cms[0].task)
composite.matrix = np.sum([cm.matrix for cm in cms], axis=0)

os.makedirs("kfold_composite", exist_ok=True)  # plot() does not create the directory
composite.plot(normalize=True, save_dir="kfold_composite")
```

!!! note

    The confusion matrix is only populated when validation runs with `plots=True`, which is the default during training. If you validate separately with `model.val(plots=False)`, `confusion_matrix.matrix` stays all zeros. This aggregation also assumes single-device training; multi-GPU DDP runs do not return a metrics object from `model.train()`.

## Conclusion

K-Fold Cross-Validation turns a single train/val split into a robust, variance-aware estimate of model performance: randomly split a detection dataset into folds, inspect their class distributions, train on each, and aggregate the results. Use the mean and standard deviation across folds to judge how reliably your model generalizes, especially when working from a limited dataset. The same steps apply to any [YOLO task](../tasks/index.md) and transfer cleanly to other machine learning workflows.

## FAQ

### What is K-Fold Cross-Validation and why is it useful in object detection?

K-Fold Cross-Validation divides the dataset into `k` subsets (folds) to evaluate model performance more reliably. Each fold serves in turn as validation data while the remaining folds are used for training. For object detection, this confirms that your model is robust and generalizes across different data splits, rather than being tied to one lucky partition.

### How do I implement K-Fold Cross-Validation using Ultralytics YOLO?

Follow these steps:

1. Verify annotations are in the [YOLO detection format](../datasets/detect/index.md).
2. Install `scikit-learn`, `pandas`, and `pyyaml` alongside `ultralytics`.
3. Generate feature vectors that count classes per image.
4. Split the dataset with `KFold` from `sklearn.model_selection`.
5. Train on each fold, then aggregate the metrics and confusion matrices.

### How do I combine results across folds?

Scalar metrics such as mAP, precision, and recall are not additive, so report their mean and standard deviation across folds. Confusion matrices hold raw counts and the fold validation sets are disjoint, so summing the per-fold matrices yields a single cross-validated confusion matrix over the whole dataset. See [Aggregating Results Across Folds](#aggregating-results-across-folds) for the code.

### Can I use K-Fold Cross-Validation with datasets other than African Wildlife?

Yes. K-Fold Cross-Validation works with any dataset whose annotations are in the YOLO detection format, and because the class index is the first value on each label line, it extends to segmentation, pose, and OBB datasets too. Point the paths in the feature-vector step at your own dataset to get started.
