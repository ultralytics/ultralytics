---
comments: true
description: Learn how to use ClearML for tracking YOLOv5 experiments, data versioning, hyperparameter optimization, and remote execution with ease.
keywords: ClearML, YOLOv5, machine learning, experiment tracking, data versioning, hyperparameter optimization, remote execution, ML pipeline
---

# ClearML Integration

<img align="center" src="https://github.com/thepycoder/clearml_screenshots/raw/main/logos_dark.png" alt="ClearML MLOps experiment tracking platform">

## About ClearML

[ClearML](https://clear.ml/) is an [open-source](https://github.com/clearml/clearml) MLOps platform built to streamline machine learning workflows and save engineering time.

- 🔨 Track every YOLOv5 training run in the **experiment manager**.
- 🔧 Version and access your custom [training data](https://www.ultralytics.com/glossary/training-data) with the integrated ClearML **data versioning tool**.
- 🔦 **Remotely train and monitor** YOLOv5 runs using the ClearML Agent.
- 🔬 Find the best mAP with ClearML **hyperparameter optimization**.
- 🔭 Turn your trained **YOLOv5 model into an API** with a few commands using ClearML Serving.

Use as many or as few of these tools as you need — start with the experiment manager alone, or chain everything together into a full pipeline.

![ClearML scalars dashboard showing YOLOv5 training metrics](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/clearml-scalars-dashboard.avif)

## 🦾 Setting Things Up

ClearML needs to communicate with a server to track your experiments and data. You have two options:

- Sign up for the free [ClearML Hosted Service](https://clear.ml/), or
- Deploy your own [ClearML server](https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server) — it is open-source, so it remains a viable option even for sensitive data.

Then install the `clearml` Python package and connect the SDK to your server:

```bash
pip install clearml
```

Generate credentials at [Settings → Workspace → Create new credentials](https://app.clear.ml/settings/workspace-configuration) (top right of the ClearML UI), then run:

```bash
clearml-init
```

Follow the prompts. That's it — setup is complete.

## 🚀 Training YOLOv5 With ClearML

To enable experiment tracking, install the ClearML pip package if you haven't already:

```bash
pip install clearml
```

This will enable integration with the YOLOv5 training script. Every training run from now on will be captured and stored by the ClearML [experiment manager](https://docs.ultralytics.com/integrations/clearml).

To customize the project and task names, pass `--project` and `--name` to `train.py`. The defaults are `YOLOv5` and `Training`. ClearML uses `/` as a subproject delimiter, so avoid `/` in custom project names.

```bash
python train.py --img 640 --batch 16 --epochs 3 --data coco8.yaml --weights yolov5s.pt --cache
```

Or with custom names:

```bash
python train.py --project my_project --name my_training --img 640 --batch 16 --epochs 3 --data coco8.yaml --weights yolov5s.pt --cache
```

Each run captures:

- Source code and uncommitted changes
- Installed packages
- Hyperparameters
- Model checkpoints (use `--save-period n` to save every `n` epochs)
- Console output
- Scalars (mAP_0.5, mAP_0.5:0.95, precision, recall, losses, learning rates)
- Machine details, runtime, and creation date
- Generated plots such as the label correlogram and [confusion matrix](https://www.ultralytics.com/glossary/confusion-matrix)
- Images with bounding boxes per [epoch](https://www.ultralytics.com/glossary/epoch)
- Mosaic visualizations per epoch
- Validation images per epoch

Everything appears in the ClearML UI so you can monitor training in one place. Add custom columns (for example, `mAP_0.5`) to sort by the best-performing model, or select multiple experiments to compare them side by side.

Keep reading for [hyperparameter optimization](https://www.ultralytics.com/glossary/hyperparameter-tuning) and remote execution.

### 🔗 Dataset Version Management

Versioning data separately from code makes it easy to pull the latest version and ensures full reproducibility. This repository accepts a dataset version ID, fetches the data automatically if it is missing, and records the ID as a task parameter so you always know which data was used in which experiment.

![ClearML dataset version management interface](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/clearml-dataset-interface.avif)

### Prepare Your Dataset

The YOLOv5 repository supports many datasets via YAML configuration files. By default, datasets download to the `../datasets` folder relative to the repository root. After downloading `coco128`, the folder structure looks like:

```text
..
|_ yolov5
|_ datasets
    |_ coco128
        |_ images
        |_ labels
        |_ LICENSE
        |_ README.txt
```

Any dataset works, provided you preserve this structure.

Next, **copy the dataset's YAML file into the dataset root folder** — ClearML reads this file to use the dataset correctly. You can write your own YAML by following the example layout, ensuring it defines `path`, `train`, `test`, `val`, `nc`, and `names`.

```text
..
|_ yolov5
|_ datasets
    |_ coco128
        |_ images
        |_ labels
        |_ coco128.yaml  # <---- HERE
        |_ LICENSE
        |_ README.txt
```

### Upload Your Dataset

To register the dataset as a versioned ClearML dataset, change into its root folder and run:

```bash
cd ../datasets/coco128
clearml-data sync --project YOLOv5 --name coco128 --folder .
```

`clearml-data sync` is shorthand for the following sequence, which you can also run explicitly:

```bash
# Add --parent <parent_dataset_id> to base this version on a previous one.
# Duplicate files are not re-uploaded.
clearml-data create --name coco128 --project YOLOv5
clearml-data add --files .
clearml-data close
```

### Train On A ClearML Dataset

With the dataset registered, point training at it by ID:

```bash
python train.py --img 640 --batch 16 --epochs 3 --data clearml://YOUR_DATASET_ID --weights yolov5s.pt --cache
```

### 👀 Hyperparameter Optimization

With experiments and data versioned, you can build on top of them. Because each tracked experiment captures the full environment — code, installed packages, and configuration — runs are **fully reproducible**. ClearML lets you clone an experiment, change its parameters, and rerun it automatically, which is the foundation of hyperparameter optimization (HPO).

To run HPO locally, use the bundled script. First make sure a training task exists in the experiment manager — the script clones it and varies its hyperparameters.

Fill in the template task ID in `utils/loggers/clearml/hpo.py`, then run:

```bash
# Install Optuna or change the optimizer to RandomSearch.
pip install optuna
python utils/loggers/clearml/hpo.py
```

Switch `task.execute_locally()` to `task.execute()` to push the job to a ClearML queue for a remote agent to pick up.

![ClearML HPO dashboard with YOLOv5 metrics](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/hpo-clearml-experiment.avif)

## 🤯 Remote Execution (Advanced)

Running HPO locally is convenient, but you'll often want experiments on more powerful hardware — an on-prem GPU machine or a cloud instance. That's the role of the [ClearML Agent](https://clear.ml/docs/latest/docs/clearml_agent):

- [YouTube overview](https://youtu.be/MX3BrXnaULs)
- [Documentation](https://clear.ml/docs/latest/docs/clearml_agent)

Each tracked experiment contains everything needed to reproduce it on another machine (installed packages, uncommitted changes, and configuration). A ClearML agent listens to a queue, picks up incoming tasks, recreates the environment, runs the job, and streams scalars and plots back to the experiment manager.

Turn any machine — a cloud VM, a local GPU box, or a laptop — into a ClearML agent with:

```bash
clearml-agent daemon --queue QUEUES_TO_LISTEN_TO [--docker]
```

### Cloning, Editing, And Enqueuing

With an agent running, you can assign it work directly from the UI:

- 🪄 Right-click an experiment and clone it.
- 🎯 Edit its hyperparameters.
- ⏳ Right-click the cloned task and enqueue it to a target queue.

![Enqueue a task from the UI](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/enqueue-task-ui.avif)

### Executing A Task Remotely

You can also flag a running script for remote execution programmatically by adding `task.execute_remotely()` after the ClearML logger has been instantiated. Add the highlighted line to `train.py`:

```python
# ...
# Loggers
data_dict = None
if RANK in {-1, 0}:
    loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
    if loggers.clearml:
        loggers.clearml.task.execute_remotely(queue="my_queue")  # <------ ADD THIS LINE
        # data_dict is None unless the user selected a ClearML dataset, in which case ClearML fills it in.
        data_dict = loggers.clearml.data_dict
# ...
```

After this change, running the training script executes up to that line, packages the code, and ships it to the queue.

### Autoscaling Workers

ClearML ships with [autoscalers](https://clear.ml/docs/latest/docs/guides/services/aws_autoscaler) that spin up remote machines in AWS, GCP, or Azure when a queue has pending experiments, convert them into ClearML agents, and shut them down when work is finished — so you only pay for compute that is actually running.

Watch the getting-started video below:

[![Watch the video](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/clearml-autoscalers-video-thumbnail.avif)](https://youtu.be/j4XVMAaUt3E)

## Learn More

For more information about integrating ClearML with Ultralytics models, check out our [ClearML integration guide](https://docs.ultralytics.com/integrations/clearml) and explore how you can enhance your [MLOps workflow](https://www.ultralytics.com/blog/exploring-yolov8-ml-experiment-tracking-integrations) with other experiment tracking tools.
