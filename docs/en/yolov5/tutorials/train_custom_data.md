---
comments: true
description: Learn how to train YOLOv5 on your own custom datasets with easy-to-follow steps. Detailed guide on dataset preparation, model selection, and training process.
keywords: YOLOv5, custom dataset, model training, object detection, machine learning, AI, YOLO model, PyTorch, dataset preparation
---

📚 This guide explains how to train your own **custom dataset** with [YOLOv5](https://github.com/ultralytics/yolov5) 🚀.

## Before You Start

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/). [Models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

## Train On Custom Data

<a href="https://www.ultralytics.com/hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-active-learning-loop.avif" alt="Ultralytics active learning"></a>
<br>
<br>

Creating a custom model to detect your objects is an iterative process of collecting and organizing images, labeling your objects of interest, training a model, deploying it into the wild to make predictions, and then using that deployed model to collect examples of edge cases to repeat and improve.

!!! question "Licensing"

    Ultralytics offers two licensing options:

    - The [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), an [OSI-approved](https://opensource.org/license) open-source license ideal for students and enthusiasts.
    - The [Enterprise License](https://www.ultralytics.com/license) for businesses seeking to incorporate our AI models into their products and services.

    For more details see [Ultralytics Licensing](https://www.ultralytics.com/license).

YOLOv5 models must be trained on labelled data in order to learn classes of objects in that data. There are two options for creating your dataset before you start training:

## Option 1: Create a <a href="https://roboflow.com/?ref=ultralytics">Roboflow</a> Dataset

### 1.1 Collect Images

Your model will learn by example. Training on images similar to the ones it will see in the wild is of the utmost importance. Ideally, you will collect a wide variety of images from the same configuration (camera, angle, lighting, etc.) as you will ultimately deploy your project.

If this is not possible, you can start from [a public dataset](https://universe.roboflow.com/?ref=ultralytics) to train your initial model and then [sample images from the wild during inference](https://blog.roboflow.com/what-is-active-learning/?ref=ultralytics) to improve your dataset and model iteratively.

### 1.2 Create Labels

Once you have collected images, you will need to annotate the objects of interest to create a ground truth for your model to learn from.

<p align="center"><a href="https://app.roboflow.com/?model=yolov5&ref=ultralytics" title="Create a Free Roboflow Account"><img width="450" src="https://github.com/ultralytics/docs/releases/download/0/roboflow-annotate.avif" alt="YOLOv5 accuracies"></a></p>

[Roboflow Annotate](https://roboflow.com/annotate?ref=ultralytics) is a simple web-based tool for managing and labeling your images with your team and exporting them in [YOLOv5's annotation format](https://roboflow.com/formats/yolov5-pytorch-txt?ref=ultralytics).

### 1.3 Prepare Dataset for YOLOv5

Whether you [label your images with Roboflow](https://roboflow.com/annotate?ref=ultralytics) or not, you can use it to convert your dataset into YOLO format, create a YOLOv5 YAML configuration file, and host it for importing into your training script.

[Create a free Roboflow account](https://app.roboflow.com/?model=yolov5&ref=ultralytics) and upload your dataset to a `Public` workspace, label any unannotated images, then generate and export a version of your dataset in `YOLOv5 Pytorch` format.

Note: YOLOv5 does online augmentation during training, so we do not recommend applying any augmentation steps in Roboflow for training with YOLOv5. But we recommend applying the following preprocessing steps:

<p align="center"><img width="450" src="https://github.com/ultralytics/docs/releases/download/0/roboflow-preprocessing-steps.avif" alt="Recommended Preprocessing Steps"></p>

- **Auto-Orient** - to strip EXIF orientation from your images.
- **Resize (Stretch)** - to the square input size of your model (640x640 is the YOLOv5 default).

Generating a version will give you a snapshot of your dataset, so you can always go back and compare your future model training runs against it, even if you add more images or change its configuration later.

<p align="center"><img width="450" src="https://github.com/ultralytics/docs/releases/download/0/roboflow-export.avif" alt="Export in YOLOv5 Format"></p>

Export in `YOLOv5 Pytorch` format, then copy the snippet into your training script or notebook to download your dataset.

<p align="center"><img width="450" src="https://github.com/ultralytics/docs/releases/download/0/roboflow-dataset-download-snippet.avif" alt="Roboflow dataset download snippet"></p>

## Option 2: Create a Manual Dataset

### 2.1 Create `dataset.yaml`

[COCO128](https://www.kaggle.com/datasets/ultralytics/coco128) is an example small tutorial dataset composed of the first 128 images in [COCO](https://cocodataset.org/) train2017. These same 128 images are used for both training and validation to verify our training pipeline is capable of [overfitting](https://www.ultralytics.com/glossary/overfitting). [data/coco128.yaml](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml), shown below, is the dataset config file that defines 1) the dataset root directory `path` and relative paths to `train` / `val` / `test` image directories (or `*.txt` files with image paths) and 2) a class `names` dictionary:

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128 # dataset root dir
train: images/train2017 # train images (relative to 'path') 128 images
val: images/train2017 # val images (relative to 'path') 128 images
test: # test images (optional)

# Classes (80 COCO classes)
names:
    0: person
    1: bicycle
    2: car
    # ...
    77: teddy bear
    78: hair drier
    79: toothbrush
```

### 2.2 Create Labels

After using an annotation tool to label your images, export your labels to **YOLO format**, with one `*.txt` file per image (if no objects in image, no `*.txt` file is required). The `*.txt` file specifications are:

- One row per object
- Each row is `class x_center y_center width height` format.
- Box coordinates must be in **normalized xywh** format (from 0 to 1). If your boxes are in pixels, divide `x_center` and `width` by image width, and `y_center` and `height` by image height.
- Class numbers are zero-indexed (start from 0).

<p align="center"><img width="750" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie.avif" alt="Roboflow annotations"></p>

The label file corresponding to the above image contains 2 persons (class `0`) and a tie (class `27`):

<p align="center"><img width="428" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie-1.avif" alt="Roboflow dataset preprocessing"></p>

### 2.3 Organize Directories

Organize your train and val images and labels according to the example below. YOLOv5 assumes `/coco128` is inside a `/datasets` directory **next to** the `/yolov5` directory. **YOLOv5 locates labels automatically for each image** by replacing the last instance of `/images/` in each image path with `/labels/`. For example:

```bash
../datasets/coco128/images/im0.jpg  # image
../datasets/coco128/labels/im0.txt  # label
```

<p align="center"><img width="700" src="https://github.com/ultralytics/docs/releases/download/0/yolov5-dataset-structure.avif" alt="YOLOv5 dataset structure"></p>

## 3. Select a Model

Select a pretrained model to start training from. Here we select [YOLOv5s](https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml), the second-smallest and fastest model available. See our README [table](https://github.com/ultralytics/yolov5#pretrained-checkpoints) for a full comparison of all models.

<p align="center"><img width="800" alt="YOLOv5 models" src="https://github.com/ultralytics/docs/releases/download/0/yolov5-model-comparison.avif"></p>

## 4. Train

Train a YOLOv5s model on COCO128 by specifying dataset, batch-size, image size and either pretrained `--weights yolov5s.pt` (recommended), or randomly initialized `--weights '' --cfg yolov5s.yaml` (not recommended). Pretrained weights are auto-downloaded from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases).

```bash
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt
```

!!! tip

    💡 Add `--cache ram` or `--cache disk` to speed up training (requires significant RAM/disk resources).

!!! tip

    💡 Always train from a local dataset. Mounted or network drives like Google Drive will be very slow.

All training results are saved to `runs/train/` with incrementing run directories, i.e. `runs/train/exp2`, `runs/train/exp3` etc. For more details see the Training section of our tutorial notebook. <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>

## 5. Visualize

### Comet Logging and Visualization 🌟 NEW

[Comet](https://bit.ly/yolov5-readme-comet) is now fully integrated with YOLOv5. Track and visualize model metrics in real time, save your hyperparameters, datasets, and model checkpoints, and visualize your model predictions with [Comet Custom Panels](https://bit.ly/yolov5-colab-comet-panels)! Comet makes sure you never lose track of your work and makes it easy to share results and collaborate across teams of all sizes!

Getting started is easy:

```shell
pip install comet_ml  # 1. install
export COMET_API_KEY=<Your API Key>  # 2. paste API key
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt  # 3. train
```

To learn more about all the supported Comet features for this integration, check out the [Comet Tutorial](./comet_logging_integration.md). If you'd like to learn more about Comet, head over to our [documentation](https://bit.ly/yolov5-colab-comet-docs). Get started by trying out the Comet Colab Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RG0WOQyxlDlo5Km8GogJpIEJlg_5lyYO?usp=sharing)

<img width="1920" alt="YOLO UI" src="https://github.com/ultralytics/docs/releases/download/0/yolo-ui.avif">

### ClearML Logging and Automation 🌟 NEW

[ClearML](https://clear.ml/) is completely integrated into YOLOv5 to track your experimentation, manage dataset versions and even remotely execute training runs. To enable ClearML:

- `pip install clearml`
- run `clearml-init` to connect to a ClearML server

You'll get all the great expected features from an experiment manager: live updates, model upload, experiment comparison etc. but ClearML also tracks uncommitted changes and installed packages for example. Thanks to that ClearML Tasks (which is what we call experiments) are also reproducible on different machines! With only 1 extra line, we can schedule a YOLOv5 training task on a queue to be executed by any number of ClearML Agents (workers).

You can use ClearML Data to version your dataset and then pass it to YOLOv5 simply using its unique ID. This will help you keep track of your data without adding extra hassle. Explore the [ClearML Tutorial](./clearml_logging_integration.md) for details!

<a href="https://clear.ml/">
<img alt="ClearML Experiment Management UI" src="https://github.com/ultralytics/docs/releases/download/0/clearml-experiment-management-ui.avif" width="1280"></a>

### Local Logging

Training results are automatically logged with [Tensorboard](https://www.tensorflow.org/tensorboard) and [CSV](https://github.com/ultralytics/yolov5/pull/4148) loggers to `runs/train`, with a new experiment directory created for each new training as `runs/train/exp2`, `runs/train/exp3`, etc.

This directory contains train and val statistics, mosaics, labels, predictions and augmented mosaics, as well as metrics and charts including [precision](https://www.ultralytics.com/glossary/precision)-[recall](https://www.ultralytics.com/glossary/recall) (PR) curves and confusion matrices.

<img alt="Local logging results" src="https://github.com/ultralytics/docs/releases/download/0/local-logging-results.avif" width="1280">

Results file `results.csv` is updated after each [epoch](https://www.ultralytics.com/glossary/epoch), and then plotted as `results.png` (below) after training completes. You can also plot any `results.csv` file manually:

```python
from utils.plots import plot_results

plot_results("path/to/results.csv")  # plot 'results.csv' as 'results.png'
```

<p align="center"><img width="800" alt="results.png" src="https://github.com/ultralytics/docs/releases/download/0/results.avif"></p>

## Next Steps

Once your model is trained you can use your best checkpoint `best.pt` to:

- Run [CLI](https://github.com/ultralytics/yolov5#quick-start-examples) or [Python](./pytorch_hub_model_loading.md) inference on new images and videos
- [Validate](https://github.com/ultralytics/yolov5/blob/master/val.py) [accuracy](https://www.ultralytics.com/glossary/accuracy) on train, val and test splits
- [Export](./model_export.md) to [TensorFlow](https://www.ultralytics.com/glossary/tensorflow), Keras, ONNX, TFlite, TF.js, CoreML and TensorRT formats
- [Evolve](./hyperparameter_evolution.md) hyperparameters to improve performance
- [Improve](https://docs.roboflow.com/adding-data/upload-api?ref=ultralytics) your model by sampling real-world images and adding them to your dataset

## Supported Environments

Ultralytics provides a range of ready-to-use environments, each pre-installed with essential dependencies such as [CUDA](https://developer.nvidia.com/cuda-zone), [CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/), to kickstart your projects.

- **Free GPU Notebooks**: <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a> <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud**: [GCP Quickstart Guide](../environments/google_cloud_quickstart_tutorial.md)
- **Amazon**: [AWS Quickstart Guide](../environments/aws_quickstart_tutorial.md)
- **Azure**: [AzureML Quickstart Guide](../environments/azureml_quickstart_tutorial.md)
- **Docker**: [Docker Quickstart Guide](../environments/docker_image_quickstart_tutorial.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Project Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 CI"></a>

This badge indicates that all [YOLOv5 GitHub Actions](https://github.com/ultralytics/yolov5/actions) Continuous Integration (CI) tests are successfully passing. These CI tests rigorously check the functionality and performance of YOLOv5 across various key aspects: [training](https://github.com/ultralytics/yolov5/blob/master/train.py), [validation](https://github.com/ultralytics/yolov5/blob/master/val.py), [inference](https://github.com/ultralytics/yolov5/blob/master/detect.py), [export](https://github.com/ultralytics/yolov5/blob/master/export.py), and [benchmarks](https://github.com/ultralytics/yolov5/blob/master/benchmarks.py). They ensure consistent and reliable operation on macOS, Windows, and Ubuntu, with tests conducted every 24 hours and upon each new commit.

## FAQ

### How do I train YOLOv5 on my custom dataset?

Training YOLOv5 on a custom dataset involves several steps:

1. **Prepare Your Dataset**: Collect and label images. Use tools like [Roboflow](https://roboflow.com/?ref=ultralytics) to organize data and export in [YOLOv5 format](https://roboflow.com/formats/yolov5-pytorch-txt?ref=ultralytics).
2. **Setup Environment**: Clone the YOLOv5 repo and install dependencies:
    ```bash
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt
    ```
3. **Create Dataset Configuration**: Write a `dataset.yaml` file defining train/val paths and class names.
4. **Train the Model**:
    ```bash
    python train.py --img 640 --epochs 3 --data dataset.yaml --weights yolov5s.pt
    ```

### What tools can I use to annotate my YOLOv5 dataset?

You can use [Roboflow Annotate](https://roboflow.com/annotate?ref=ultralytics), an intuitive web-based tool for labeling images. It supports team collaboration and exports in YOLOv5 format. After collecting images, use Roboflow to create and manage annotations efficiently. Other options include tools like LabelImg and CVAT for local annotations.

### Why should I use Ultralytics HUB for training my YOLO models?

Ultralytics HUB offers an end-to-end platform for training, deploying, and managing YOLO models without needing extensive coding skills. Benefits of using Ultralytics HUB include:

- **Easy Model Training**: Simplifies the training process with preconfigured environments.
- **Data Management**: Effortlessly manage datasets and version control.
- **Real-time Monitoring**: Integrates tools like [Comet](https://bit.ly/yolov5-readme-comet) for real-time metrics tracking and visualization.
- **Collaboration**: Ideal for team projects with shared resources and easy management.

### How do I convert my annotated data to YOLOv5 format?

To convert annotated data to YOLOv5 format using Roboflow:

1. **Upload Your Dataset** to a Roboflow workspace.
2. **Label Images** if not already labeled.
3. **Generate and Export** the dataset in `YOLOv5 Pytorch` format. Ensure preprocessing steps like Auto-Orient and Resize (Stretch) to the square input size (e.g., 640x640) are applied.
4. **Download the Dataset** and integrate it into your YOLOv5 training script.

### What are the licensing options for using YOLOv5 in commercial applications?

Ultralytics offers two licensing options:

- **AGPL-3.0 License**: An open-source license suitable for non-commercial use, ideal for students and enthusiasts.
- **Enterprise License**: Tailored for businesses seeking to integrate YOLOv5 into commercial products and services. For detailed information, visit our [Licensing page](https://www.ultralytics.com/license).

For more details, refer to our guide on [Ultralytics Licensing](https://www.ultralytics.com/license).
