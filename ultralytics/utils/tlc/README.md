<img src="https://3lc.ai/wp-content/uploads/2023/09/3LC-Logo_Footer.svg">

# 3LC Integration

This document outlines how to use the 3LC integration available for YOLOv8 classification and object detection.

For any questions or problems, please reach out on the [3LC Discord](https://discord.com/channels/1236027984150794290/1236118620002586655).

## About 3LC

[3LC](https://3lc.ai) is a tool which enables data scientists to improve machine learning models in a data-centric fashion. It collects per-sample predictions and metrics, allows viewing and modifying the dataset in the context of those predictions in the 3LC Dashboard, and rerunning training with the revised dataset.

3LC is free for non-commercial use.

![3LC Dashboard Overview](_static/dashboard.png)

## Getting Started

The first step is to clone this fork, change directory into it and install the package and requirements into a virtual environment:
```bash
git clone https://github.com/3lc-ai/ultralytics.git
cd ultralytics
python -m venv .venv
source .venv/bin/activate # or .venv/Scripts/activate in Git Bash / Windows
pip install -e . # install the local ultralytics fork package
pip install pacmap # or umap-learn (optional, only required for embeddings collection)
```

In order to create a `tlc.Run` with the integration, instantiate `TLCYOLO` (instead of `YOLO`) and call the method `.train()` like usual:
<details open>
<summary>Code Example</summary>
The following code example shows basic usage of the 3LC integration. The `Settings` object can be used to specify 3LC specific settings. For more details, see the
section called 3LC Settings.

```python
from ultralytics.utils.tlc import Settings, TLCYOLO

# Set 3LC specific settings
settings = Settings(
    project_name="my_yolo_project",
    run_name="my_yolo_run",
    run_description="my_yolo_run_description",
)

# Initialize and run training
model = TLCYOLO("yolov8n.pt") # Or e.g. "yolov8n-cls.pt" for classification
model.train(data="coco128.yaml", settings=settings) # See the section 'Dataset Specification' for how to specify which data to use
```
</details>

In the background, 3LC will create `tlc.Table`s and collect metrics with the trained model after training completes, which can be opened in the 3LC Dashboard.

> **⚠️ NOTE:** Make sure not to have your calling script in the same directory as the cloned `ultralytics` repository. If they are in the same directory, Python will directly import the module instead of the installed package, causing `import ultralytics` to fail.

## Dataset specification

![Banner image with tasks](https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png)

A good starting point for using the integration is usually to use the dataset you are already using with YOLOv8. In this case, you can get started by setting `data=<path to your dataset>` like you are already doing. See the [Ultralytics Documentation](https://docs.ultralytics.com/datasets/) to learn more. 3LC parses these datasets and creates a table for each split, which can be viewed in the Dashboard. Once you make some new versions of your data in the 3LC Dashboard you can use the same command with `data=<path to your dataset>`, and the latest version will be used automatically.

As an alternative, if you would like to train again with a specific version, or have your own `tlc.Table`s you would like to use, there are two ways to specify this:

1. When calling `model.train()`, `model.val()` or `model.collect()`, provide a keyword argument `tables` which is a dictionary mapping split names to `tlc.Table` instances or `tlc.Url`s to tables. For example, for table instances it could look like this: `tables={"train": my_train_table, "val": my_val_table}`. When `tables` is provided, any value of `data` is ignored. In training the table for the key `"train"` is used for training, and `"val"` or `"test"`for validation (val takes precedence). 

2. Use a so-called 3LC YAML file. To signal that you are providing a 3LC YAML file, add a `3LC://`-prefix to the path to the file. If, for example, you create a 3LC YAML file named `my_3lc_yaml_file.yaml`, pass it as `model.train(data="3LC://my_3lc_yaml_file.yaml")`. A 3LC YAML file should look something like the following:
```yaml
train: /path/to/train/table
val: s3://path/to/val/table # The table is on s3
```
It is also possible to specify that the latest version of your dataset should be used by attaching `:latest` to the end of your path:
```yaml
train: /path/to/train/table:latest # Use the latest version of this data
val: s3://path/to/val/table # The table is on s3
```
`names` and `nc` are not needed since the `tlc.Table`s themselves contain the category names and indices.

There are some specific settings and options for the different tasks, check out the relevant dropdowns for this:
<details>
<summary>Classification</summary>
For image classification, it is possible to provide `image_column_name` and `label_column_name` when calling `model.train()`, `model.val()` and `model.collect()` if you are providing your own table which has different column names to those expected by 3LC.
</details>

<details>
<summary>Object Detection</summary>
In addition to tables created with `Table.from_yolo()` (which is called internally when you provide a yolo dataset), it is also possible to use tables with the COCO format used in the 3LC Detectron2 integration. If you have created 3LC tables in the Detectron2 integration, you can also use this `tlc.Table` in this integration!
</details>

<details>
<summary>Segmentation (not supported)</summary>
The 3LC integration does not yet support the Segmentation task. Let us know on Discord if you would like us to add it.
</details>

<details>
<summary>Pose Estimation (not supported)</summary>
The 3LC integration does not yet support the Pose Estimation task. Let us know on Discord if you would like us to add it.
</details>

<details>
<summary>OBB (oriented object detection) (not supported)</summary>
The 3LC integration does not yet support the Oriented Object Detection task. Let us know on Discord if you would like us to add it.
</details>

## Metrics collection only

It is possible to create runs where only metrics collection, and no training, is performed. This is useful when you already have a trained model and would like to collect metrics, or if you would like to collect metrics on a different dataset to the one you trained and validated on.

Use the method `model.collect()` to perform metrics collection only. Either pass `data` (a path to a yaml file) and `splits` (an iterable of split names to collect metrics for), or a dictionary `tables` like detailed in the previous section, to define which data to collect metrics on. This will create a run, collect the metrics on each split by calling `model.val()` and finally reduce any embeddings that were collected. Any additional arguments, such as `imgsz` and `batch`, are forwarded as `model.val(**kwargs)`.

The following code snippet shows how to collect metrics on the train and validation splits of the `coco128` dataset with `yolov8m.pt`:
```python
from ultralytics.utils.tlc import Settings, TLCYOLO

model = TLCYOLO("yolov8m.pt")

settings = Settings(
    image_embeddings_dim=2,
    conf_thres=0.2,
)

model.collect(
    data="coco128.yaml",
    splits=("train", "val"),
    settings=settings,
    batch=32,
    imgsz=320
)
```

## 3LC Settings

The integration offers a rich set of settings and features which can be set through an instance of `Settings`, which are in addition to the regular YOLOv8 settings. They allow specifying which metrics to collect, how often to collect them, and whether to use sampling weights during training.

The available 3LC settings can be seen in the `Settings` class in [settings.py](settings.py).

Providing invalid values (or combinations of values) will either log an appropriate warning or raise an error, depending on the case.

### Image Embeddings

Image embeddings can be collected by setting `image_embeddings_dim` to 2 or 3. Similar images, as seen by the model, tend to be close to each other in this space. In the 3LC Dashboard these embeddings can be visualized, allowing you to find similar images, duplicates and imbalances in your dataset, and take appropriate actions to mitigate these issues.

The way in which embeddings are collected is different for the different tasks. For more details, see the drop-downs:
<details>
<summary>Classification</summary>
For classification, the integration scans your model for the first occurrence of a `torch.nn.Linear` layer. The inputs to this layer are used to extract image embeddings.
</details>
<details>
<summary>Object Detection</summary>
For object detection, the output of the spatial pooling function is used to extract embeddings.
</details>

You can change which `3lc`-supported reducer to use by setting `image_embeddings_dim`. `pacmap` is the default.

### Run properties
Use `project_name`, `run_name` and `run_description` to customize the `tlc.Run` that is created. Any tables created by the integration will be under the `project_name` provided here. If these settings are not set, appropriate defaults are used instead.

### Sampling Weights
Use `sampling_weights=True` to enable the usage of sampling weights. This resamples the data presented to the model according to the weight column in the `Table`. If a sample has weight 2.0, it is twice as likely to appear as a particular sample with weight 1.0. Any given sample can occur multiple times in one epoch. This setting only applies to training.

### Exclude zero weight samples
Use `exclude_zero_weight_training=True` (only applies to training) and `exclude_zero_weight_collection=True` to eliminate rows with weight 0.0. If your table has samples with weight 0.0, this will effectively reduce the size of the dataset (i.e. reduce the number of iterations per epoch).

### Metrics collection settings
Use `collection_val_only=True` to disable metrics collection on the training set. This only applies to training.

Use `collection_disable=True` to disable metrics collection entirely. This only applies to training. A run will still be created, and hyperparameters and aggregate metrics will be logged to 3LC.

Use `collection_epoch_start` and `collection_epoch_interval` to define when to collect metrics during training. The start epoch is 1-based, i.e. 1 means after the first epoch. As an example, `collection_epoch_start=1` with `collection_epoch_interval=2` means metrics collection will occur after the first epoch and then every other epoch after that.

## Other output

When viewing all your YOLOv8 runs in the 3LC Dashboard, charts will show up with per-epoch aggregate metrics produced by YOLOv8 for each run. This allows you to follow your runs in real-time, and compare them with each other.

# Frequently Asked Questions

## What is the difference between before and after training metrics?

By default, the 3LC integration collects metrics only after training with the `best.pt` weights written by YOLOv8. These are the after training metrics.

If a starting metrics collection epoch is provided (optionally with an interval), metrics are also collected during training, this time with the exponential moving average that YOLOv8 uses for its validation passes.

## What happens if I use early stopping? Does it interfere with 3LC?

Early stopping can be used just like before. Unless metrics collection is disabled, final validation passes are performed over the train and validation sets after training, regardless of whether that is due to early stopping or completing all the epochs.

## Why is embeddings collection disabled by default?

Embeddings collection has an extra dependency for the library used for reduction, and a performance implication (fitting and applying the reducer) at the end of a run. It is therefore disabled by default.

## How do I collect embeddings for each bounding box?

In order to collect embeddings (or other additional metrics) for each bounding box, refer to the [3LC Bounding Box Example Notebooks](https://docs.3lc.ai/3lc/latest/public-notebooks/add-bb-embeddings.html).

## Can I use the YOLOv8 CLI commands in the integration to train and collect metrics?

This is not supported yet, but will be added in a future commit!