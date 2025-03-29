---
comments: true
description: Learn how to train YOLOv5 on your own custom datasets with easy-to-follow steps. Detailed guide on dataset preparation, model selection, and training process.
keywords: YOLOv5, custom dataset, model training, object detection, machine learning, AI, YOLO model, PyTorch, dataset preparation, Ultralytics
---

# Train YOLOv5 on Custom Data

📚 This guide explains how to train your own **custom dataset** with [YOLOv5](https://github.com/ultralytics/yolov5) 🚀. Training custom models is a crucial step in applying computer vision to specific use cases beyond general object detection.

## Before You Start

First, clone the YOLOv5 repository and install the necessary dependencies from `requirements.txt`. Ensure you have a [**Python>=3.8.0**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) installed. Models and datasets will be downloaded automatically from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) if they are not found locally.

```bash
git clone https://github.com/ultralytics/yolov5 # Clone the repository
cd yolov5
pip install -r requirements.txt # Install dependencies
```

## Train On Custom Data

<a href="https://www.ultralytics.com/hub" target="_blank">
<img width="100%" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-active-learning-loop.avif" alt="Ultralytics active learning loop diagram showing data collection, labeling, training, deployment, and edge case collection"></a>
<br>
<br>

Creating a custom [object detection](https://docs.ultralytics.com/tasks/detect/) model involves an iterative process:

1.  **Collect & Organize Images**: Gather images relevant to your specific task.
2.  **Label Objects**: Annotate the objects of interest within your images.
3.  **Train a Model**: Use the labeled data to train your YOLOv5 model.
4.  **Deploy & Predict**: Use the trained model for inference on new data.
5.  **Collect Edge Cases**: Identify scenarios where the model performs poorly and add similar data to improve robustness. Repeat the cycle.

[Ultralytics HUB](https://docs.ultralytics.com/hub/) offers a streamlined, no-code solution for this entire process, including dataset management, model training, and deployment.

!!! question "Licensing"

    Ultralytics provides two licensing options to accommodate different usage needs:

    - **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license) open-source license is ideal for students, researchers, and enthusiasts who wish to share their work openly. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for details.
    - **Enterprise License**: Designed for commercial use, this license allows for the integration of Ultralytics software and AI models into commercial products and services without the open-source requirements of AGPL-3.0. Request an [Enterprise License](https://www.ultralytics.com/license) if this suits your needs.

    Learn more about our licensing options on the [Ultralytics Licensing](https://www.ultralytics.com/license) page.

Before training, you need to prepare your dataset.

## 1. Create a Dataset

YOLOv5 models require labeled data to learn object classes. There are several ways to create and organize your dataset.

### 1.1 Create `dataset.yaml`

The dataset configuration file (e.g., `coco128.yaml`) defines the structure and classes of your dataset. [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/) is a small example dataset using the first 128 images from the larger [COCO](https://docs.ultralytics.com/datasets/detect/coco/) dataset, useful for testing the training pipeline and checking for [overfitting](https://www.ultralytics.com/glossary/overfitting).

The `dataset.yaml` file specifies:

- `path`: The root directory for the dataset.
- `train`, `val`, `test`: Relative paths to directories containing images or text files listing image paths.
- `names`: A dictionary mapping class indices (starting from 0) to class names.

Here's the structure for `coco128.yaml` ([view on GitHub](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml)):

```yaml
# Dataset root directory
path: ../datasets/coco128

# Train/val/test sets: specify directories, *.txt files, or lists
train: images/train2017 # 128 images
val: images/train2017 # 128 images
test: # Optional test images

# Classes (80 COCO classes)
names:
    0: person
    1: bicycle
    2: car
    # ... (rest of the classes)
    77: teddy bear
    78: hair drier
    79: toothbrush
```

### 1.2 Create Labels

Use an annotation tool (like CVAT or LabelImg) for [data labeling](https://www.ultralytics.com/glossary/data-labeling). Export your annotations in the **YOLO format**. This format requires one `*.txt` file for each image. If an image has no objects, no `*.txt` file is needed.

The `*.txt` file format specifications are:

- One row per object [bounding box](https://www.ultralytics.com/glossary/bounding-box).
- Each row follows the format: `class_index x_center y_center width height`.
- Coordinates must be **normalized** (values between 0 and 1). To normalize, divide the pixel values of `x_center` and `width` by the image width, and `y_center` and `height` by the image height.
- Class indices are zero-indexed (the first class is `0`, the second is `1`, and so on).

<p align="center"><img width="750" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie.avif" alt="Example image with two persons and a tie annotated"></p>

The label file corresponding to the image above would contain two lines for the 'person' class (index `0`) and one line for the 'tie' class (index `27`):

<p align="center"><img width="428" src="https://github.com/ultralytics/docs/releases/download/0/two-persons-tie-1.avif" alt="Example YOLO format label file content for the annotated image"></p>

### 1.3 Organize Directories

Structure your [datasets](https://docs.ultralytics.com/datasets/) directory as shown below. YOLOv5 expects the dataset directory (e.g., `/coco128`) to be inside a `/datasets` folder located **next to** the `/yolov5` repository directory.

YOLOv5 automatically finds the labels for each image by replacing the last occurrence of `/images/` in the image path with `/labels/`. For example:

```bash
../datasets/coco128/images/im0.jpg # Image file path
../datasets/coco128/labels/im0.txt # Corresponding label file path
```

The expected directory structure:

```
/datasets/
└── coco128/
    ├── images/
    │   ├── train2017/
    │   │   ├── 000000000009.jpg
    │   │   └── ...
    │   └── val2017/  # (Optional if using same images for train/val)
    │       └── ...
    └── labels/
        ├── train2017/
        │   ├── 000000000009.txt
        │   └── ...
        └── val2017/  # (Optional if using same labels for train/val)
            └── ...
```

<p align="center"><img width="700" src="https://github.com/ultralytics/docs/releases/download/0/yolov5-dataset-structure.avif" alt="Diagram showing the recommended YOLOv5 dataset directory structure"></p>

## 2. Select a Model

Choose a [pretrained model](https://docs.ultralytics.com/models/#pretrained-models) as a starting point for training. YOLOv5 offers various models, differing in size and speed. For instance, [YOLOv5s](https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml) is the second-smallest and fastest option. Refer to the [README table](https://github.com/ultralytics/yolov5#pretrained-checkpoints) for a comparison of all available [models](https://docs.ultralytics.com/models/).

<p align="center"><img width="800" alt="Comparison chart of YOLOv5 models showing size, speed, and accuracy" src="https://github.com/ultralytics/docs/releases/download/0/yolov5-model-comparison.avif"></p>

## 3. Train

Start the [training process](https://docs.ultralytics.com/modes/train/) using the `train.py` script. Key arguments include:

- `--img`: Defines the input [image size](https://docs.ultralytics.com/usage/cfg/#image-size) (e.g., `--img 640`).
- `--batch`: Determines the [batch size](https://www.ultralytics.com/glossary/batch-size) (e.g., `--batch 16`).
- `--epochs`: Specifies the number of training [epochs](https://www.ultralytics.com/glossary/epoch) (e.g., `--epochs 100`).
- `--data`: Path to your `dataset.yaml` file (e.g., `--data coco128.yaml`).
- `--weights`: Path to the initial weights. Use pretrained weights (e.g., `--weights yolov5s.pt`) for faster convergence and better results (recommended). To train from scratch, use `--weights '' --cfg yolov5s.yaml` (not recommended).

Pretrained weights are automatically downloaded from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases).

```bash
# Train YOLOv5s on COCO128 dataset for 3 epochs
python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt
```

!!! tip "Optimize Training Speed"

    💡 Use `--cache ram` or `--cache disk` to cache dataset images in [RAM](https://en.wikipedia.org/wiki/Random-access_memory) or local disk, respectively. This significantly speeds up training, especially if dataset I/O is a bottleneck. Requires substantial RAM/disk space.

!!! tip "Local Data Storage"

    💡 Always train using datasets stored locally. Accessing data from mounted or network drives (like Google Drive) can be extremely slow and hinder training performance.

All training outputs are saved in the `runs/train/` directory, with subdirectories for each run (e.g., `runs/train/exp`, `runs/train/exp2`, etc.). For a hands-on experience, explore the training section in our tutorial notebooks: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>

## 4. Visualize

YOLOv5 integrates with various tools for visualizing training progress and results.

### Comet Logging and Visualization 🌟 NEW

[Comet](https://bit.ly/yolov5-readme-comet) is fully integrated for experiment tracking. Visualize metrics in real-time, save hyperparameters, manage datasets and model checkpoints, and analyze predictions using [Comet Custom Panels](https://bit.ly/yolov5-colab-comet-panels).

Get started easily:

```shell
pip install comet_ml                                                          # 1. Install Comet
export COMET_API_KEY=<Your API Key>                                           # 2. Set API key (create free account at Comet.ml)
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt # 3. Train model
```

Explore the [Comet Integration Guide](https://docs.ultralytics.com/integrations/comet/) for more details on supported features. Learn more about Comet from their [documentation](https://bit.ly/yolov5-colab-comet-docs). Try the Comet Colab Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RG0WOQyxlDlo5Km8GogJpIEJlg_5lyYO?usp=sharing)

<img width="1920" alt="Comet UI showing YOLOv5 training metrics and visualizations" src="https://github.com/ultralytics/docs/releases/download/0/yolo-ui.avif">

### ClearML Logging and Automation 🌟 NEW

[ClearML](https://clear.ml/) integration allows tracking experiments, managing dataset versions, and even executing training runs remotely. Enable ClearML with:

- `pip install clearml`
- Run `clearml-init` to connect to your ClearML server.

ClearML tracks experiments, model uploads, comparisons, uncommitted code changes, and installed packages, ensuring reproducibility. Schedule training tasks on remote agents easily. Use ClearML Data for dataset versioning. Check out the [ClearML Integration Guide](https://docs.ultralytics.com/integrations/clearml/) for more information.

<a href="https://clear.ml/">
<img alt="ClearML Experiment Management UI showing charts and logs for a YOLOv5 training run" src="https://github.com/ultralytics/docs/releases/download/0/clearml-experiment-management-ui.avif" width="1280"></a>

### Local Logging

Training results are automatically logged using [TensorBoard](https://docs.ultralytics.com/integrations/tensorboard/) and as [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) files in the `runs/train/exp` directory. This includes:

- Training and validation statistics.
- Sample images with augmentations (mosaics).
- Ground truth labels and model predictions.
- Metrics like [Precision](https://www.ultralytics.com/glossary/precision)-[Recall](https://www.ultralytics.com/glossary/recall) (PR) curves.
- [Confusion matrices](https://www.ultralytics.com/glossary/confusion-matrix).

<img alt="Example of local logging results including charts and image mosaics from YOLOv5 training" src="https://github.com/ultralytics/docs/releases/download/0/local-logging-results.avif" width="1280">

The `results.csv` file is updated after each epoch and plotted as `results.png` upon completion. You can plot any `results.csv` file manually:

```python
from utils.plots import plot_results

# Plot results from a specific training run
plot_results("runs/train/exp/results.csv")  # Generates 'results.png'
```

<p align="center"><img width="800" alt="Example results.png plot showing training metrics like mAP, precision, recall, and loss over epochs" src="https://github.com/ultralytics/docs/releases/download/0/results.avif"></p>

## 5. Next Steps

After successfully training your model, the best checkpoint (`best.pt`) can be used for various tasks:

- Run [inference](https://docs.ultralytics.com/modes/predict/) on new images or videos using [CLI](https://github.com/ultralytics/yolov5#quick-start-examples) or [Python](./pytorch_hub_model_loading.md).
- Perform [validation](https://docs.ultralytics.com/modes/val/) to assess model [accuracy](https://www.ultralytics.com/glossary/accuracy) on different data splits.
- [Export](https://docs.ultralytics.com/modes/export/) the model to formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/), [TensorFlow SavedModel](https://docs.ultralytics.com/integrations/tf-savedmodel/), or [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) for deployment.
- Use [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/) techniques to potentially improve performance.
- Enhance your model further by following our [Tips for Best Training Results](https://docs.ultralytics.com/guides/model-training-tips/) and incorporating more diverse data.

## Supported Environments

Ultralytics offers pre-configured environments with necessary dependencies like [CUDA](https://developer.nvidia.com/cuda-zone), [CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/) to get you started quickly.

- **Free GPU Notebooks**:
    - <a href="https://bit.ly/yolov5-paperspace-notebook"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"></a>
    - <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    - <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Cloud Platforms**:
    - **Google Cloud**: [GCP Quickstart Guide](https://docs.ultralytics.com/integrations/google-colab/)
    - **Amazon AWS**: [AWS Quickstart Guide](https://docs.ultralytics.com/integrations/amazon-sagemaker/)
    - **Microsoft Azure**: [AzureML Quickstart Guide](https://docs.ultralytics.com/guides/azureml-quickstart/)
- **Local Setup**:
    - **Docker**: [Docker Quickstart Guide](https://docs.ultralytics.com/guides/docker-quickstart/) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## Project Status

<a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="YOLOv5 Continuous Integration Status Badge"></a>

This badge confirms that all YOLOv5 [GitHub Actions](https://github.com/ultralytics/yolov5/actions) [Continuous Integration (CI)](https://www.ultralytics.com/glossary/continuous-integration-ci) tests pass successfully. These tests cover core functionalities like [training](https://docs.ultralytics.com/modes/train/), [validation](https://docs.ultralytics.com/modes/val/), [inference](https://docs.ultralytics.com/modes/predict/), [export](https://docs.ultralytics.com/modes/export/), and [benchmarks](https://docs.ultralytics.com/modes/benchmark/) across macOS, Windows, and Ubuntu environments. Tests run automatically every 24 hours and on each code commit to ensure ongoing stability and performance.

## FAQ

## Frequently Asked Questions

### How do I train YOLOv5 on my custom dataset?

Training YOLOv5 on a custom dataset involves these main steps:

1.  **Prepare Your Dataset**: Collect images and annotate them using an annotation tool. Export the annotations in the required [YOLO format](https://docs.ultralytics.com/datasets/detect/).
2.  **Set Up Your Environment**: Clone the YOLOv5 repository and install the dependencies listed in `requirements.txt`.
    ```bash
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt
    ```
3.  **Create Dataset Configuration**: Define your dataset paths and class names in a `dataset.yaml` file.
4.  **Start Training**: Run the `train.py` script, specifying your dataset configuration file, desired model weights (e.g., `yolov5s.pt`), image size, batch size, and number of epochs.
    ```bash
    python train.py --img 640 --batch 16 --epochs 100 --data path/to/your/dataset.yaml --weights yolov5s.pt
    ```

### Why should I use Ultralytics HUB for training my YOLO models?

[Ultralytics HUB](https://docs.ultralytics.com/hub/) provides a comprehensive platform designed to simplify the entire lifecycle of YOLO model development, from training to deployment, often without requiring any code. Key advantages include:

- **Simplified Training**: Easily train models using pre-configured environments and intuitive interfaces.
- **Integrated Data Management**: Upload, version, and manage your datasets efficiently.
- **Real-time Monitoring**: Track training progress and visualize metrics using integrated tools like [Comet](https://docs.ultralytics.com/integrations/comet/).
- **Collaboration Features**: Facilitates teamwork through shared resources and project management capabilities.

For a practical guide, see our blog post on [How to Train Your Custom Models with Ultralytics HUB](https://www.ultralytics.com/blog/how-to-train-your-custom-models-with-ultralytics-hub).

### How do I convert my annotated data to the YOLOv5 format?

After annotating your images using an annotation tool (many are available, such as CVAT, LabelImg, or others), you need to export the labels into the specific **YOLO format** required by YOLOv5. This format involves:

- One `.txt` file per image, placed in a parallel `labels/` directory relative to your `images/` directory.
- Each `.txt` file contains one line per object annotation: `class_index center_x center_y width height`.
- Coordinates must be **normalized** (ranging from 0.0 to 1.0) relative to the image's width and height.
- Class indices are **zero-based** (e.g., the first class is `0`, the second is `1`).

Many annotation tools offer direct export capabilities to the YOLO format. If not, you might need to write a simple script to convert your annotations. Ensure your final dataset structure matches the example provided in the guide. For more detailed guidance on preparing your data, refer to our [Data Collection and Annotation Guide](https://docs.ultralytics.com/guides/data-collection-and-annotation/).

### What are the licensing options for using YOLOv5 in commercial applications?

Ultralytics offers flexible licensing:

- **AGPL-3.0 License**: An open-source license ideal for academic research, personal projects, and situations where open-source compliance is feasible. See the [AGPL-3.0 License details](https://www.ultralytics.com/legal/agpl-3-0-software-license).
- **Enterprise License**: A commercial license designed for businesses integrating YOLOv5 into products or services. This license avoids the open-source requirements of AGPL-3.0. Visit our [Licensing page](https://www.ultralytics.com/license) for more information or to request an [Enterprise License](https://www.ultralytics.com/legal/enterprise-software-license).

Choose the license that best fits your project's needs.
