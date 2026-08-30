---
comments: true
description: Learn to implement K-Fold Cross Validation for object detection datasets using Ultralytics YOLO. Improve your model's reliability and robustness.
keywords: Ultralytics, YOLO, K-Fold Cross Validation, object detection, sklearn, pandas, PyYAML, machine learning, dataset split
---

# K-Fold Cross Validation with Ultralytics

## Introduction

This comprehensive guide illustrates the implementation of K-Fold Cross Validation for [object detection](https://www.ultralytics.com/glossary/object-detection) datasets within the Ultralytics ecosystem. We'll leverage the YOLO detection format and key Python libraries such as sklearn, pandas, and PyYAML to guide you through the necessary setup, the process of generating feature vectors, and the execution of a K-Fold dataset split.

<p align="center">
  <img width="800" src="https://cdn.ul.run/i/457d0a77dc06d7204322ec056248c4b5.avif" alt="K-fold cross validation data splitting">
</p>

Whether your project involves the Fruit Detection dataset or a custom data source, this tutorial aims to help you comprehend and apply K-Fold Cross Validation to bolster the reliability and robustness of your [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models. While we're applying `k=5` folds for this tutorial, keep in mind that the optimal number of folds can vary depending on your dataset and the specifics of your project. K-Fold Cross Validation delivers the most value when your dataset is small, noisy, or highly variable; for large, diverse datasets, a well-constructed train/val/test split is usually sufficient.

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

2. Load the configured training and validation images through the package dataset owner, leaving the test split untouched.

    ```python
    from pathlib import Path

    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    yaml_file = "path/to/data.yaml"
    data = check_det_dataset(yaml_file)
    dataset_path = Path(data["path"])
    sources = []
    for split in ("train", "val"):
        source = data.get(split)
        if source:
            sources.extend(source if isinstance(source, list) else [source])
    dataset = YOLODataset(sources, data=data, augment=False)
    images = [Path(p) for p in dataset.im_files]
    ```

3. Now, read the contents of the dataset YAML file and extract the indices of the class labels.

    ```python
    classes = data["names"]
    cls_idx = sorted(classes.keys())
    ```

4. Initialize an empty `pandas` DataFrame.

    ```python
    import pandas as pd

    labels_df = pd.DataFrame(0.0, columns=cls_idx, index=images)
    ```

5. Count the instances of each class-label present in the annotation files.

    ```python
    from collections import Counter

    for image, label in zip(images, dataset.labels):
        lbl_counter = Counter(label["cls"].flatten().astype(int))
        labels_df.loc[image, list(lbl_counter)] = list(lbl_counter.values())
    ```

6. The following is a sample view of the populated DataFrame:

    ```text
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

The rows use absolute image paths, and the columns correspond to class-label indices. Missing labels remain all-zero background rows. This data structure enables the application of [K-Fold Cross Validation](https://www.ultralytics.com/glossary/cross-validation) to an object detection dataset.

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
    folds_df = pd.DataFrame(index=labels_df.index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        folds_df.loc[labels_df.index[train], f"split_{i}"] = "train"
        folds_df.loc[labels_df.index[val], f"split_{i}"] = "val"
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

4. Write image lists and a dataset YAML for each split. Text lists avoid copying the dataset `k` times.

    ```python
    import yaml

    save_path = dataset_path.parent / f"{ksplit}-Fold_Cross-val"
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        for partition in ("train", "val"):
            split_images = folds_df.index[folds_df[split] == partition]
            paths = "\n".join(map(str, split_images))
            (save_path / f"{split}_{partition}.txt").write_text(f"{paths}\n")

        dataset_yaml = save_path / f"{split}.yaml"
        ds_yamls.append(dataset_yaml)
        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": save_path.as_posix(),
                    "train": f"{split}_train.txt",
                    "val": f"{split}_val.txt",
                    "names": classes,
                },
                ds_y,
            )
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

3. You can also use [Ultralytics data.split.autosplit](../reference/data/split.md) function for automatic dataset splitting:

    ```python
    from ultralytics.data.split import autosplit

    # Automatically split dataset into train/val/test
    autosplit(path="path/to/images", weights=(0.8, 0.2, 0.0), annotated_only=True)
    ```

## Conclusion

In this guide, we have explored the process of using K-Fold cross-validation for training the YOLO object detection model. We learned how to split the training and validation pool into K partitions and use the generated ratio table to inspect class balance after random splitting.

We also explored the procedure for creating report DataFrames to visualize the data splits and label distributions across these splits, providing us a clear insight into the structure of our training and validation sets.

Optionally, we saved our records for future reference, which could be particularly useful in large-scale projects or when troubleshooting model performance.

Finally, we implemented the actual model training using each split in a loop, saving our training results for further analysis and comparison.

This technique of K-Fold cross-validation is a robust way of making the most out of your available data, and it helps to ensure that your model performance is reliable and consistent across different data subsets. This results in a more generalizable and reliable model that is less likely to [overfit](https://www.ultralytics.com/glossary/overfitting) to specific data patterns.

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

### How should I design folds for other YOLO tasks like segmentation, classification, pose, or OBB?

The workflow in this guide targets the YOLO detection format, but the same approach adapts to every YOLO task — the task changes how you compose the folds, not whether cross-validation helps:

| Task       | Fold design                                                                                                                                                                |
| :--------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `detect`   | Split at the image level, balancing object and class distributions across folds. Keep related images (same patient, video sequence, camera, or site) within a single fold. |
| `segment`  | Use the same image-level strategy as detection, additionally preserving mask and class coverage in every fold.                                                             |
| `classify` | Prefer stratified folds so class frequencies stay balanced between training and validation.                                                                                |
| `pose`     | Split by subject or sequence so the same person or animal never appears on both sides of a fold.                                                                           |
| `obb`      | Split at the image level, keeping tiles or crops from the same scene together — especially important for aerial imagery.                                                   |

Whatever the task, keep near-duplicate and related samples out of opposing folds: that kind of leakage inflates validation metrics well beyond what the model will achieve in production.

### Why should I use Ultralytics YOLO for object detection?

Ultralytics YOLO offers state-of-the-art, real-time object detection with high [accuracy](https://www.ultralytics.com/glossary/accuracy) and efficiency. It's versatile, supporting multiple [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks such as [detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), and [classification](../tasks/classify.md). Additionally, it integrates seamlessly with tools like [Ultralytics Platform](../platform/index.md) for no-code model training and deployment. For more details, explore the benefits and features on our [Ultralytics YOLO page](https://www.ultralytics.com/yolo).

### How can I ensure my annotations are in the correct format for Ultralytics YOLO?

Your annotations should follow the YOLO detection format. Each annotation file must list the object class, alongside its [bounding box](https://www.ultralytics.com/glossary/bounding-box) coordinates in the image. The YOLO format ensures streamlined and standardized data processing for training object detection models. For more information on proper annotation formatting, visit the [YOLO detection format guide](../datasets/detect/index.md).

### Can I use K-Fold Cross Validation with custom datasets other than Fruit Detection?

Yes, you can use K-Fold Cross Validation with any custom dataset as long as the annotations are in the YOLO detection format. Replace the dataset paths and class labels with those specific to your custom dataset. This flexibility ensures that any object detection project can benefit from robust model evaluation using K-Fold Cross Validation. For a practical example, review our [Generating Feature Vectors](#generating-feature-vectors-for-object-detection-dataset) section.
