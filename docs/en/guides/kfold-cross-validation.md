---
comments: true
description: Learn to implement K-Fold Cross Validation for object detection datasets using Ultralytics YOLO. Improve your model's reliability and robustness.
keywords: Ultralytics, YOLO, K-Fold Cross Validation, object detection, sklearn, pandas, PyYAML, machine learning, dataset split
---

# K-Fold Cross Validation with Ultralytics

## Introduction

This comprehensive guide illustrates the implementation of K-Fold Cross Validation for [object detection](https://www.ultralytics.com/glossary/object-detection) datasets within the Ultralytics ecosystem. We'll leverage the YOLO detection format and key Python libraries such as sklearn, pandas, and PyYAML to guide you through the necessary setup, the process of generating feature vectors, and the execution of a K-Fold dataset split.

<p align="center">
  <img width="800" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/k-fold-cross-validation-overview.avif" alt="K-fold cross validation data splitting">
</p>

Whether your project involves the Fruit Detection dataset or a custom data source, this tutorial aims to help you comprehend and apply K-Fold Cross Validation to bolster the reliability and robustness of your [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models. While we're applying `k=5` folds for this tutorial, keep in mind that the optimal number of folds can vary depending on your dataset and the specifics of your project.

Let's get started.

## Setup

- Your annotations should be in the [YOLO detection format](../datasets/detect/index.md).

- This guide assumes that annotation files are locally available.

- For our demonstration, we use the [Fruit Detection](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection/code) dataset.
    - This dataset contains a total of 8479 images.
    - It includes 6 class labels, each with its total instance counts listed below.

| Class Label | Instance Count |
| :---------- | :------------: |
| Apple       |      7049      |
| Grapes      |      7202      |
| Pineapple   |      1613      |
| Orange      |     15549      |
| Banana      |      3536      |
| Watermelon  |      1976      |

- Necessary Python packages include:
    - `ultralytics`
    - `sklearn`
    - `pandas`
    - `pyyaml`

- This tutorial operates with `k=5` folds. However, you should determine the best number of folds for your specific dataset.

1. Initiate a new Python virtual environment (`venv`) for your project and activate it. Use `pip` (or your preferred package manager) to install:
    - The Ultralytics library: `pip install -U ultralytics`. Alternatively, you can clone the official [repo](https://github.com/ultralytics/ultralytics).
    - Scikit-learn, pandas, and PyYAML: `pip install -U scikit-learn pandas pyyaml`.

2. Verify that your annotations are in the [YOLO detection format](../datasets/detect/index.md).
    - For this tutorial, all annotation files are found in the `Fruit-Detection/labels` directory.

## Generating Feature Vectors for Object Detection Dataset

1. Start by creating a new `example.py` Python file for the steps below.

2. Proceed to retrieve all label files for your dataset.

    ```python
    from pathlib import Path

    dataset_path = Path("./Fruit-detection")  # replace with 'path/to/dataset' for your custom data
    labels = sorted(dataset_path.rglob("*labels/*.txt"))  # all data in 'labels'
    ```

3. Now, read the contents of the dataset YAML file and extract the indices of the class labels.

    ```python
    import yaml

    yaml_file = "path/to/data.yaml"  # your data YAML with data directories and names dictionary
    with open(yaml_file, encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
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

6. The following is a sample view of the populated DataFrame:

    ```
                                                           0    1    2    3    4    5
    '0000a16e4b057580_jpg.rf.00ab48988370f64f5ca8ea4...'  0.0  0.0  0.0  0.0  0.0  7.0
    '0000a16e4b057580_jpg.rf.7e6dce029fb67f01eb19aa7...'  0.0  0.0  0.0  0.0  0.0  7.0
    '0000a16e4b057580_jpg.rf.bc4d31cdcbe229dd022957a...'  0.0  0.0  0.0  0.0  0.0  7.0
    '00020ebf74c4881c_jpg.rf.508192a0a97aa6c4a3b6882...'  0.0  0.0  0.0  1.0  0.0  0.0
    '00020ebf74c4881c_jpg.rf.5af192a2254c8ecc4188a25...'  0.0  0.0  0.0  1.0  0.0  0.0
     ...                                                  ...  ...  ...  ...  ...  ...
    'ff4cd45896de38be_jpg.rf.c4b5e967ca10c7ced3b9e97...'  0.0  0.0  0.0  0.0  0.0  2.0
    'ff4cd45896de38be_jpg.rf.ea4c1d37d2884b3e3cbce08...'  0.0  0.0  0.0  0.0  0.0  2.0
    'ff5fd9c3c624b7dc_jpg.rf.bb519feaa36fc4bf630a033...'  1.0  0.0  0.0  0.0  0.0  0.0
    'ff5fd9c3c624b7dc_jpg.rf.f0751c9c3aa4519ea3c9d6a...'  1.0  0.0  0.0  0.0  0.0  0.0
    'fffe28b31f2a70d4_jpg.rf.7ea16bd637ba0711c53b540...'  0.0  6.0  0.0  0.0  0.0  0.0
    ```

The rows index the label files, each corresponding to an image in your dataset, and the columns correspond to your class-label indices. Each row represents a pseudo feature-vector, with the count of each class-label present in your dataset. This data structure enables the application of [K-Fold Cross Validation](https://www.ultralytics.com/glossary/cross-validation) to an object detection dataset.

## K-Fold Dataset Split

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

5. Lastly, copy images and labels into the respective directory ('train' or 'val') for each split.
    - **NOTE:** The time required for this portion of the code will vary based on the size of your dataset and your system hardware.
    - **NOTE:** Each image is copied into every split, so this duplicates the dataset roughly `k` times on disk. For large datasets, consider symlinks (`os.symlink`) instead of `shutil.copy` to save space (on Windows, symlinks may require elevated permissions).

    ```python
    import shutil

    from tqdm import tqdm

    for image in tqdm(images, total=len(images), desc="Copying files"):
        label = labels_by_stem[image.stem]  # pair by filename stem, not by list position
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)
    ```

## Save Records (Optional)

Optionally, you can save the records of the K-Fold split and label distribution DataFrames as CSV files for future reference.

```python
folds_df.to_csv(save_path / "kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")
```

## Train YOLO using K-Fold Data Splits

1. First, load the YOLO model.

    ```python
    from ultralytics import YOLO

    weights_path = "path/to/weights.pt"  # use yolo26n.pt for a small model
    model = YOLO(weights_path, task="detect")
    ```

2. Next, iterate over the dataset YAML files to run training. The results will be saved to a directory specified by the `project` and `name` arguments. By default, this directory is 'runs/detect/train#' where # is an integer index.

    ```python
    results = {}

    # Define your additional arguments here
    batch = 16
    project = "kfold_demo"
    epochs = 100

    for k, dataset_yaml in enumerate(ds_yamls):
        model = YOLO(weights_path, task="detect")
        results[k] = model.train(
            data=dataset_yaml, epochs=epochs, batch=batch, project=project, name=f"fold_{k + 1}"
        )  # include any additional train arguments
    ```

!!! tip "Need a simple split instead of K-Fold?"

    If you only need a one-off random train/val/test split rather than cross-validation, use [`autosplit`](https://docs.ultralytics.com/reference/data/split):

    ```python
    from ultralytics.data.split import autosplit

    # Random one-off split (not K-Fold cross-validation)
    autosplit(path="path/to/images", weights=(0.8, 0.2, 0.0), annotated_only=True)
    ```

## Aggregating Results Across Folds

The goal of K-Fold cross-validation is a single, more reliable performance estimate together with its variance. Aggregate the per-fold results stored in the `results` dictionary above in two complementary ways:

- **Scalar metrics (mAP, precision, recall) are not additive** across folds, so report their mean and standard deviation. The spread across folds tells you how stable the model is between splits.
- **Confusion matrices hold raw counts**, and because the fold validation sets are disjoint and together cover the dataset exactly once, summing the per-fold matrices yields a single cross-validated confusion matrix over every image.

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

In this guide, we have explored the process of using K-Fold cross-validation for training the YOLO object detection model. We learned how to split our dataset into K partitions, ensuring a balanced class distribution across the different folds.

We also explored the procedure for creating report DataFrames to visualize the data splits and label distributions across these splits, providing us a clear insight into the structure of our training and validation sets.

Optionally, we saved our records for future reference, which could be particularly useful in large-scale projects or when troubleshooting model performance.

Finally, we implemented the actual model training using each split in a loop, saving our training results for further analysis and comparison.

This technique of K-Fold cross-validation is a robust way of making the most out of your available data, and it helps to ensure that your model performance is reliable and consistent across different data subsets. This results in a more generalizable and reliable model that is less likely to [overfit](https://www.ultralytics.com/glossary/overfitting) to specific data patterns.

The same workflow also applies to segmentation, pose, and OBB datasets, since the class index is the first value on each label line, so the class-count feature vectors are built the same way.

Remember that although we used YOLO in this guide, these steps are mostly transferable to other machine learning models. Understanding these steps allows you to apply cross-validation effectively in your own machine learning projects.

## FAQ

### What is K-Fold Cross Validation and why is it useful in object detection?

K-Fold Cross Validation is a technique where the dataset is divided into 'k' subsets (folds) to evaluate model performance more reliably. Each fold serves as both training and [validation data](https://www.ultralytics.com/glossary/validation-data). In the context of object detection, using K-Fold Cross Validation helps to ensure your Ultralytics YOLO model's performance is robust and generalizable across different data splits, enhancing its reliability. For detailed instructions on setting up K-Fold Cross Validation with Ultralytics YOLO, refer to [K-Fold Cross Validation with Ultralytics](#introduction).

### How do I implement K-Fold Cross Validation using Ultralytics YOLO?

To implement K-Fold Cross Validation with Ultralytics YOLO, you need to follow these steps:

1. Verify annotations are in the [YOLO detection format](../datasets/detect/index.md).
2. Use Python libraries like `sklearn`, `pandas`, and `pyyaml`.
3. Create feature vectors from your dataset.
4. Split your dataset using `KFold` from `sklearn.model_selection`.
5. Train the YOLO model on each split.

For a comprehensive guide, see the [K-Fold Dataset Split](#k-fold-dataset-split) section in our documentation.

### Why should I use Ultralytics YOLO for object detection?

Ultralytics YOLO offers state-of-the-art, real-time object detection with high [accuracy](https://www.ultralytics.com/glossary/accuracy) and efficiency. It's versatile, supporting multiple [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks such as [detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), and [classification](../tasks/classify.md). Additionally, it integrates seamlessly with tools like [Ultralytics Platform](https://docs.ultralytics.com/platform) for no-code model training and deployment. For more details, explore the benefits and features on our [Ultralytics YOLO page](https://www.ultralytics.com/yolo).

### How can I ensure my annotations are in the correct format for Ultralytics YOLO?

Your annotations should follow the YOLO detection format. Each annotation file must list the object class, alongside its [bounding box](https://www.ultralytics.com/glossary/bounding-box) coordinates in the image. The YOLO format ensures streamlined and standardized data processing for training object detection models. For more information on proper annotation formatting, visit the [YOLO detection format guide](../datasets/detect/index.md).

### Can I use K-Fold Cross Validation with custom datasets other than Fruit Detection?

Yes, you can use K-Fold Cross Validation with any custom dataset as long as the annotations are in the YOLO detection format. Replace the dataset paths and class labels with those specific to your custom dataset. This flexibility ensures that any object detection project can benefit from robust model evaluation using K-Fold Cross Validation. For a practical example, review our [Generating Feature Vectors](#generating-feature-vectors-for-object-detection-dataset) section.
