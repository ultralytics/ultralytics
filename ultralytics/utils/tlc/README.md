<img src="https://3lc.ai/wp-content/uploads/2023/09/3LC-Logo_Footer.svg">

# 3LC Integration

This document outlines how to use the 3LC integration available for YOLOv8 object detection.

## About 3LC

[3LC](https://3lc.ai) is a tool which enables data scientists to improve machine learning models in a data-centric fashion. It collects per-sample predictions and metrics, allows viewing and modifying the dataset in the context of those predictions in the 3LC Dashboard, and rerunning training with the revised dataset.

3LC is free for non-commercial use.

<img src="https://docs.3lc.ai/3lc/latest/_images/2_hardhat_run1.PNG">

## Getting Started

The first step is to clone this fork, change directory into it and install the package and requirements into a virtual environment:
```bash
git clone https://github.com/3lc-ai/ultralytics.git
cd ultralytics
python -m venv .venv
source .venv/bin/activate # or .venv/Scripts/activate in Git Bash
pip install -e . # install the local ultralytics fork package
pip install 3lc # install 3lc
```

In order to run training with the integration, instantiate `TLCYOLO` (instead of `YOLO`) and call the method `.train()` just like you are used to. The most simple example, which also shows how to specify 3LC settings, looks like this:

```python
from ultralytics.utils.tlc.detect.model import TLCYOLO
from ultralytics.utils.tlc.detect.settings import Settings

# Set 3LC specific settings
settings = Settings(
    project_name="my_yolo_project",
    run_name="my_yolo_run",
    run_description="my_yolo_run_description",
    image_embeddings_dim=2,
    collection_epoch_start=0,
    collection_epoch_interval=2,
    conf_thres=0.2,
)

# Initialize and run training
model = TLCYOLO("yolov8n.pt")
model.train(data="coco128.yaml", settings=settings)
```

In order to run metrics collection only (no training) in a single 3LC run, you can use `.val()`, where the same run is reused across calls:

```python
from ultralytics.utils.tlc.detect.model import TLCYOLO
from ultralytics.utils.tlc.detect.settings import Settings

model = TLCYOLO("yolov8n.pt")

# Set 3LC specific settings
settings = Settings(
    image_embeddings_dim=2,
    conf_thres=0.2,
)

# Run metrics collection on 
for split in ("train", "val"):
    results = model.val(data="coco128.yaml", split=split, settings=settings)
```

### First Time

For your first run, 3LC creates `Table`s for your training and validation sets provided through the `data` kwarg, and collects metrics after the final epoch for every sample in your dataset. A new YAML file is written next to the one that was used, which can be used for later runs, more on that in [Later Runs](#later-runs).

You can then open the 3LC Dashboard to view your run!

### Later Runs

For later runs, in order to specify that you would like to continue working with the same 3LC tables, there are two ways to proceed:

#### Regular YAML file

You can keep using the same YAML file pointed to by `data`. As long as this file does not change, the integration will resolve to the same 3LC tables and always get the latest revision for each split. The specific revisions used are logged to the console, and a line is printed stating that a 3LC YAML is printed with instructions on how to use it.

The integration uses only the YAML file name to resolve to the relevant tables if they exist. Therefore, if any changes are made to the original YAML file name, this reference to the tables is lost (and new tables are instead created on your next run).

#### 3LC YAML File

For more flexibility and to explicitly select which tables to use, you should use a 3LC YAML file, like the one written during your first run.

The file should simply contain keys for the relevant splits (`train` and `val` in most cases), with values set to the 3LC Urls of the corresponding tables. Once your 3LC YAML is populated with these it will look like the following example:

```yaml
train: my_train_table
val: my_val_table
```

In order to use it, prepend a 3LC prefix `3LC://` to the path. If the 3LC YAML file is named `my_dataset.yaml`, you should provide `data="3LC://my_dataset.yaml` to `.train()` and `.val()`. After running your first run with the regular YAML file, a 3LC YAML is written next to it which you can use immediately (just remember to prepend the 3LC prefix).

In order to train on different revisions, simply change the paths in the file to your desired revision.

If you would like to train on the latest revisions, you can add `:latest` to one or both of the paths, and 3LC will find the `Table`s for the latest revision in the same lineage. For the above example, with latest on both, the 3LC YAML would look like this:

```yaml
train: my_train_table:latest
val: my_val_table:latest
```

______________________________________________________________________

**NOTE**: We recommend using a 3LC YAML to specify which revisions to use, as this enables using specific revisions of the dataset, and adding `:latest` in order to use the latest table in the lineage. It removes the dependency on the original YAML file to find the corresponding tables.

______________________________________________________________________

<details>
<summary>3LC YAML Example</summary>
<br>
The following example highlights the behavior when using 3LC YAML files.

Let's assume that you made a new revision in the 3LC Dashboard where you edited two bounding boxes. You would then have the following tables in your lineage:
```
my_train_table ---> Edited2BoundingBoxes (latest)
my_val_table (latest)
```

If you were to reuse the original YAML file, `Edited2BoundingBoxes` would be the latest revision of your train set and `my_val_table` the latest val set. These would be used for your run.

In order to train on a specific revision, in this case the original data, you can provide a 3LC YAML file `my_3lc_dataset.yaml` with `--data 3LC://my_3lc_dataset.yaml`, with the following content:

```yaml
train: my_train_table
val: my_val_table
```

Specifying to use the latest revisions instead can be done by adding `:latest` to one or both of these `Url`s:

```yaml
train: my_train_table:latest # resolves to the latest revision of my_train_table, which is Edited1BoundingBoxes
val: my_val_table:latest # resolves to the latest revision of my_val_table, which is my_val_table
```

</details>

## 3LC Settings

The integration offers a rich set of settings and features which can be set through an instance of `Settings`. They allow specifying which metrics to collect, how often to collect them, and whether to use sampling weights during training.

The available 3LC settings can be seen in the `Settings` class in [settings.py](detect/settings.py).

Providing invalid values (or combinations of values) will either log an appropriate warning or raise an error, depending on the case.

### Image Embeddings

Image embeddings can be collected by setting `image_embeddings_dim` to 2 or 3, and are based on the output of the spatial pooling function output from the YOLOv8 architectures. Similar images, as seen by the model, tend to be close to each other in this space. In the 3LC Dashboard these embeddings can be visualized, allowing you to find similar images, duplicates and imbalances in your dataset and determine if your validation set is representative of your training data (and vice-versa).

Note that when collecting image embeddings for validation only runs, `reduce_all_embeddings()` must be called at the end to produce embeddings which can be visualized in the Dashboard.

## Other output

When viewing all your YOLOv8 runs in the 3LC Dashboard, charts will show up with per-epoch validation metrics for each run. This allows you to follow your runs in real-time, and compare them with each other.

# Frequently Asked Questions

## What is the difference between before and after training metrics?

By default, the 3LC integration collects metrics only after training with the `best.pt` weights written by YOLOv8. These are the after training metrics.

If a starting metrics collection epoch is provided (optionally with an interval), metrics are also collected during training, this time with the exponential moving average that YOLOv8 uses for its validation passes.

## What happens if I use early stopping? Does it interfere with 3LC?

Early stopping can be used just like before. Unless metrics collection is disabled, final validation passes are performed over the train and validation sets after training, regardless of whether that is due to early stopping or completing all the epochs.

## Why is embeddings collection disabled by default?

Embeddings collection has an extra dependency for the library used for reduction, and a performance implication (fitting and applying the reducer) at the end of your run. It is therefore disabled by default.

## How do I collect embeddings for each bounding box?

In order to collect embeddings (or other additional metrics) for each bounding box, refer to the [3LC Bounding Box Example Notebooks](https://docs.3lc.ai/3lc/latest/public-notebooks/add-bb-embeddings.html).