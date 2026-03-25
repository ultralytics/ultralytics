---
comments: true
description: Learn how to train YOLOv5 on your own custom datasets with easy-to-follow steps. Detailed guide on dataset preparation, model selection, and training process.
keywords: YOLOv5, custom dataset, model training, object detection, machine learning, AI, YOLO model, PyTorch, dataset preparation, Ultralytics
---

# Train YOLOv5 on Custom Data

📚 This guide explains how to train your own **custom dataset** using the [YOLOv5](https://github.com/ultralytics/yolov5) model 🚀. Training custom models is a fundamental step in tailoring [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) solutions to specific real-world applications beyond generic [object detection](https://docs.ultralytics.com/tasks/detect/).

## Before You Start

First, ensure you have the necessary environment set up. Clone the YOLOv5 repository and install the required dependencies from `requirements.txt`. A [**Python>=3.8.0**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) is essential. Models and datasets are automatically downloaded from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) if they are not found locally.

```bash
git clone https://github.com/ultralytics/yolov5 # Clone the repository
cd yolov5
pip install -r requirements.txt # Install dependencies
```

## Train On Custom Data

<a href="https://platform.ultralytics.com" target="_blank">
<img width="100%" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-active-learning-loop.avif" alt="Ultralytics active learning loop diagram"></a>
<br>
<br>

Developing a custom [object detection](https://docs.ultralytics.com/tasks/detect/) model is an iterative process:

1.  **Collect & Organize Images**: Gather images relevant to your specific task. High-quality, diverse data is crucial. See our guide on [Data Collection and Annotation](https://docs.ultralytics.com/guides/data-collection-and-annotation/).
2.  **Label Objects**: Annotate the objects of interest within your images accurately.
3.  **Train a Model**: Use the labeled data to [train](https://docs.ultralytics.com/modes/train/) your YOLOv5 model. Leverage [transfer learning](https://www.ultralytics.com/glossary/transfer-learning) by starting with pretrained weights.
4.  **Deploy & Predict**: Utilize the trained model for [inference](https://docs.ultralytics.com/modes/predict/) on new, unseen data.
5.  **Collect Edge Cases**: Identify scenarios where the model performs poorly ([edge cases](https://en.wikipedia.org/wiki/Edge_case)) and add similar data to your dataset to improve robustness. Repeat the cycle.

[Ultralytics Platform](https://docs.ultralytics.com/platform/) offers a streamlined, no-code solution for this entire [machine learning operations (MLOps)](https://www.ultralytics.com/glossary/machine-learning-operations-mlops) cycle, including dataset management, model training, and deployment.

!!! question "Licensing"

    Ultralytics provides two licensing options to accommodate diverse usage scenarios:

    - **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license/agpl-v3) open-source license is ideal for students, researchers, and enthusiasts passionate about open collaboration and knowledge sharing. It requires derived works to be shared under the same license. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for full details.
    - **Enterprise License**: Designed for commercial applications, this license permits the seamless integration of Ultralytics software and AI models into commercial products and services without the open-source stipulations of AGPL-3.0. If your project requires commercial deployment, request an [Enterprise License](https://www.ultralytics.com/license).

    Explore our licensing options further on the [Ultralytics Licensing](https://www.ultralytics.com/license) page.

Before initiating the training, dataset preparation is essential.

## 1. Create a Dataset

YOLOv5 models require labeled data to learn the visual characteristics of object classes. Organizing your dataset correctly is key.

### 1.1 Create `dataset.yaml`

The dataset configuration file (e.g., `coco128.yaml`) outlines the dataset's structure, class names, and paths to image directories. [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/) serves as a small example dataset, comprising the first 128 images from the extensive [COCO](https://docs.ultralytics.com/datasets/detect/coco/) dataset. It's useful for quickly testing the training pipeline and diagnosing potential issues like [overfitting](https://www.ultralytics.com/glossary/overfitting).

The `dataset.yaml` file structure includes:

- `path`: The root directory containing the dataset.
- `train`, `val`, `test`: Relative paths from `path` to directories containing images or text files listing image paths for training, validation, and testing sets.
- `names`: A dictionary mapping class indices (starting from 0) to their corresponding class names.

You can set `path` to either an absolute directory (e.g., `/home/user/datasets/coco128`) or a relative path such as `../datasets/coco128` when launching training from the YOLOv5 repository root.

Below is the structure for `coco128.yaml` ([view on GitHub](https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml)):

```yaml
# Dataset root directory relative to the yolov5 directory
path: coco128

# Train/val/test sets: specify directories, *.txt files, or lists
train: images/train2017 # 128 images for training
val: images/train2017 # 128 images for validation
test: # Optional path to test images

# Classes (example using 80 COCO classes)
names:
    0: person
    1: bicycle
    2: car
    # ... (remaining COCO classes)
    77: teddy bear
    78: hair drier
    79: toothbrush
```

### 1.2 Leverage Models for Automated Labeling

While manual labeling using tools is a common approach, the process can be time-consuming. Recent advancements in foundation models offer possibilities for automating or semi-automating the annotation process, potentially speeding up dataset creation significantly. Here are a few examples of models that can assist with generating labels:

- **[Google Gemini](https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-use-google-gemini-models-for-object-detection-image-captioning-and-ocr.ipynb)**: Large multimodal models like Gemini possess powerful image understanding capabilities. They can be prompted to identify and locate objects within images, generating bounding boxes or descriptions that can be converted into YOLO format labels. Explore its potential in the provided tutorial notebook.
- **[SAM2 (Segment Anything Model 2)](https://docs.ultralytics.com/models/sam-2/)**: Foundation models focused on segmentation, like SAM2, can identify and delineate objects with high precision. While primarily for segmentation, the resulting masks can often be converted into bounding box annotations suitable for object detection tasks.
- **[YOLOWorld](https://docs.ultralytics.com/models/yolo-world/)**: This model offers open-vocabulary detection capabilities. You can provide text descriptions of the objects you're interested in, and YOLOWorld can locate them in images _without_ prior training on those specific classes. This can be used as a starting point for generating initial labels, which can then be refined.

Using these models can provide a "pre-labeling" step, reducing the manual effort required. However, it's crucial to review and refine automatically generated labels to ensure accuracy and consistency, as the quality directly impacts the performance of your trained YOLOv5 model. After generating (and potentially refining) your labels, ensure they adhere to the **YOLO format**: one `*.txt` file per image, with each line representing an object as `class_index x_center y_center width height` (normalized coordinates, zero-indexed class). If an image has no objects of interest, no corresponding `*.txt` file is needed.

The YOLO format `*.txt` file specifications are precise:

- One row per object [bounding box](https://www.ultralytics.com/glossary/bounding-box).
- Each row must contain: `class_index x_center y_center width height`.
- Coordinates must be **normalized** to a range between 0 and 1. To achieve this, divide the pixel values of `x_center` and `width` by the image's total width, and divide `y_center` and `height` by the image's total height.
- Class indices are zero-indexed (i.e., the first class is represented by `0`, the second by `1`, and so forth).

<p align="center"><img width="750" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/two-persons-tie.avif" alt="Example image with two persons and a tie annotated"></p>

The label file corresponding to the image above, containing two 'person' objects (class index `0`) and one 'tie' object (class index `27`), would look like this:

<p align="center"><img width="428" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/two-persons-tie-1.avif" alt="YOLO format label file content example"></p>

### 1.3 Organize Directories

Structure your [datasets](https://docs.ultralytics.com/datasets/) directory as illustrated below. By default, YOLOv5 anticipates the dataset directory (e.g., `/coco128`) to reside within a `/datasets` folder located **adjacent to** the `/yolov5` repository directory.

YOLOv5 automatically locates the labels for each image by substituting the last instance of `/images/` in the image path with `/labels/`. For example:

```bash
../datasets/coco128/images/im0.jpg # Path to the image file
../datasets/coco128/labels/im0.txt # Path to the corresponding label file
```

The recommended directory structure is:

```
/datasets/
└── coco128/  # Dataset root
    ├── images/
    │   ├── train2017/  # Training images
    │   │   ├── 000000000009.jpg
    │   │   └── ...
    │   └── val2017/    # Validation images (optional if using same set for train/val)
    │       └── ...
    └── labels/
        ├── train2017/  # Training labels
        │   ├── 000000000009.txt
        │   └── ...
        └── val2017/    # Validation labels (optional if using same set for train/val)
            └── ...
```

<p align="center"><img width="700" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolov5-dataset-structure.avif" alt="YOLOv5 recommended dataset directory structure"></p>

## 2. Select a Model

Choose a [pretrained model](https://docs.ultralytics.com/models/) to initiate the training process. Starting with pretrained weights significantly accelerates learning and improves performance compared to training from scratch. YOLOv5 offers various model sizes, each balancing speed and accuracy differently. For example, [YOLOv5s](https://github.com/ultralytics/yolov5/blob/master/models/yolov5s.yaml) is the second-smallest and fastest model, suitable for resource-constrained environments. Consult the [README table](https://github.com/ultralytics/yolov5#pretrained-checkpoints) for a detailed comparison of all available [models](https://docs.ultralytics.com/models/).

<p align="center"><img width="800" alt="Comparison chart of YOLOv5 models showing size, speed, and accuracy" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolov5-model-comparison.avif"></p>

## 3. Train

Begin the [model training](https://docs.ultralytics.com/modes/train/) using the `train.py` script. Essential arguments include:

- `--img`: Defines the input [image size](https://docs.ultralytics.com/usage/cfg/#image-size) (e.g., `--img 640`). Larger sizes generally yield better accuracy but require more GPU memory.
- `--batch`: Determines the [batch size](https://www.ultralytics.com/glossary/batch-size) (e.g., `--batch 16`). Choose the largest size your GPU can handle.
- `--epochs`: Specifies the total number of training [epochs](https://www.ultralytics.com/glossary/epoch) (e.g., `--epochs 100`). One epoch represents a full pass over the entire training dataset.
- `--data`: Path to your `dataset.yaml` file (e.g., `--data coco128.yaml`).
- `--weights`: Path to the initial weights file. Using pretrained weights (e.g., `--weights yolov5s.pt`) is highly recommended for faster convergence and superior results. To train from scratch (not advised unless you have a very large dataset and specific needs), use `--weights '' --cfg yolov5s.yaml`.

Pretrained weights are automatically downloaded from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) if not found locally.

```bash
# Example: Train YOLOv5s on the COCO128 dataset for 3 epochs
python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt
```

!!! tip "Optimize Training Speed"

    💡 Employ `--cache ram` or `--cache disk` to cache dataset images in [RAM](https://en.wikipedia.org/wiki/Random-access_memory) or local disk, respectively. This dramatically accelerates training, particularly when dataset I/O (Input/Output) operations are a bottleneck. Note that this requires substantial RAM or disk space.

!!! tip "Local Data Storage"

    💡 Always train using datasets stored locally. Accessing data from network drives (like Google Drive) or remote storage can be significantly slower and impede training performance. Copying your dataset to a local SSD is often the best practice.

All training outputs, including weights and logs, are saved in the `runs/train/` directory. Each training session creates a new subdirectory (e.g., `runs/train/exp`, `runs/train/exp2`, etc.). For an interactive, hands-on experience, explore the training section in our official tutorial notebooks: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/models/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>

## 4. Visualize

YOLOv5 seamlessly integrates with various tools for visualizing training progress, evaluating results, and monitoring performance in real-time.

### Comet Logging and Visualization 🌟 NEW

[Comet](https://docs.ultralytics.com/integrations/comet/) is fully integrated for comprehensive experiment tracking. Visualize metrics live, save hyperparameters, manage datasets and model checkpoints, and analyze model predictions using interactive [Comet Custom Panels](https://bit.ly/yolov5-colab-comet-panels).

Getting started is straightforward:

```bash
pip install comet_ml                                                          # 1. Install Comet library
export COMET_API_KEY=YOUR_API_KEY_HERE                                        # 2. Set your Comet API key (create a free account at Comet.ml)
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt # 3. Train your model - Comet automatically logs everything!
```

Dive deeper into the supported features in our [Comet Integration Guide](https://docs.ultralytics.com/integrations/comet/). Learn more about Comet's capabilities from their official [documentation](https://bit.ly/yolov5-colab-comet-docs). Try the Comet Colab Notebook for a live demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RG0WOQyxlDlo5Km8GogJpIEJlg_5lyYO?usp=sharing)

<img width="1920" alt="Comet UI showing YOLOv5 training metrics and visualizations" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolo-ui.avif">

### ClearML Logging and Automation 🌟 NEW

[ClearML](https://docs.ultralytics.com/integrations/clearml/) integration enables detailed experiment tracking, dataset version management, and even remote execution of training runs. Activate ClearML with these simple steps:

- Install the package: `pip install clearml`
- Initialize ClearML: Run `clearml-init` once to connect to your ClearML server (either self-hosted or the [free tier](https://clear.ml/)).

ClearML automatically captures experiment details, model uploads, comparisons, uncommitted code changes, and installed packages, ensuring full reproducibility. You can easily schedule training tasks on remote agents and manage dataset versions using ClearML Data. Explore the [ClearML Integration Guide](https://docs.ultralytics.com/integrations/clearml/) for comprehensive details.

<a href="https://clear.ml/">
<img alt="ClearML experiment management UI for YOLOv5" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/clearml-experiment-management-ui.avif" width="1280"></a>

### Local Logging

Training results are automatically logged using [TensorBoard](https://docs.ultralytics.com/integrations/tensorboard/) and saved as [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) files within the specific experiment directory (e.g., `runs/train/exp`). Logged data includes:

- Training and validation loss and performance metrics.
- Sample images showing applied augmentations (like mosaics).
- Ground truth labels alongside model predictions for visual inspection.
- Key evaluation metrics such as [Precision](https://www.ultralytics.com/glossary/precision)-[Recall](https://www.ultralytics.com/glossary/recall) (PR) curves.
- [Confusion matrices](https://www.ultralytics.com/glossary/confusion-matrix) for detailed class-wise performance analysis.

<img alt="YOLOv5 local logging results with charts and mosaics" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/local-logging-results.avif" width="1280">

The `results.csv` file is updated after every epoch and is plotted as `results.png` once training concludes. You can also plot any `results.csv` file manually using the provided utility function:

```python
from utils.plots import plot_results

# Plot results from a specific training run directory
plot_results("runs/train/exp/results.csv")  # This will generate 'results.png' in the same directory
```

<p align="center"><img width="800" alt="YOLOv5 results.png training metrics plot" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/yolov5-training-results-plot.avif"></p>

## 5. Next Steps

Upon successful completion of training, the best performing model checkpoint (`best.pt`) is saved and ready for deployment or further refinement. Potential next steps include:

- Run [inference](https://docs.ultralytics.com/modes/predict/) on new images or videos using the trained model via the [CLI](https://github.com/ultralytics/yolov5#quick-start-examples) or [Python](./pytorch_hub_model_loading.md).
- Perform [validation](https://docs.ultralytics.com/modes/val/) to evaluate the model's [accuracy](https://www.ultralytics.com/glossary/accuracy) and generalization capabilities on different data splits (e.g., a held-out test set).
- [Export](https://docs.ultralytics.com/modes/export/) the model to various deployment formats like [ONNX](https://docs.ultralytics.com/integrations/onnx/), [TensorFlow SavedModel](https://docs.ultralytics.com/integrations/tf-savedmodel/), or [TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) for optimized inference on diverse platforms.
- Employ [hyperparameter tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/) techniques to potentially squeeze out additional performance gains.
- Continue improving your model by following our [Tips for Best Training Results](https://docs.ultralytics.com/guides/model-training-tips/) and iteratively adding more diverse and challenging data based on performance analysis.

## Supported Environments

Ultralytics provides ready-to-use environments equipped with essential dependencies like [CUDA](https://developer.nvidia.com/cuda), [cuDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/), and [PyTorch](https://pytorch.org/), facilitating a smooth start.

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

This badge indicates that all YOLOv5 [GitHub Actions](https://github.com/ultralytics/yolov5/actions) [Continuous Integration (CI)](https://www.ultralytics.com/glossary/continuous-integration-ci) tests are passing successfully. These rigorous CI tests cover the core functionalities, including [training](https://docs.ultralytics.com/modes/train/), [validation](https://docs.ultralytics.com/modes/val/), [inference](https://docs.ultralytics.com/modes/predict/), [export](https://docs.ultralytics.com/modes/export/), and [benchmarks](https://docs.ultralytics.com/modes/benchmark/), across macOS, Windows, and Ubuntu operating systems. Tests are executed automatically every 24 hours and upon each code commit, ensuring consistent stability and optimal performance.

## FAQ

### How do I train YOLOv5 on my custom dataset?

Training YOLOv5 on a custom dataset involves several key steps:

1.  **Prepare Your Dataset**: Collect images and annotate them. Ensure annotations are in the required [YOLO format](https://docs.ultralytics.com/datasets/detect/). Organize images and labels into `train/` and `val/` (and optionally `test/`) directories. Consider using models like [Google Gemini](https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-use-google-gemini-models-for-object-detection-image-captioning-and-ocr.ipynb), [SAM2](https://docs.ultralytics.com/models/sam-2/), or [YOLOWorld](https://docs.ultralytics.com/models/yolo-world/) to assist with or automate the labeling process (see Section 1.2).
2.  **Set Up Your Environment**: Clone the YOLOv5 repository and install dependencies using `pip install -r requirements.txt`.
    ```bash
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt
    ```
3.  **Create Dataset Configuration**: Define dataset paths, number of classes, and class names in a `dataset.yaml` file.
4.  **Start Training**: Execute the `train.py` script, providing paths to your `dataset.yaml`, desired pretrained weights (e.g., `yolov5s.pt`), image size, batch size, and the number of epochs.
    ```bash
    python train.py --img 640 --batch 16 --epochs 100 --data path/to/your/dataset.yaml --weights yolov5s.pt
    ```

### Why should I use Ultralytics Platform for training my YOLO models?

[Ultralytics Platform](https://docs.ultralytics.com/platform/) is a comprehensive platform designed to streamline the entire YOLO model development lifecycle, often without needing to write any code. Key benefits include:

- **Simplified Training**: Easily train models using pre-configured environments and an intuitive user interface.
- **Integrated Data Management**: Upload, version control, and manage your datasets efficiently within the platform.
- **Real-time Monitoring**: Track training progress and visualize performance metrics using integrated tools like [Comet](https://docs.ultralytics.com/integrations/comet/) or TensorBoard.
- **Collaboration Features**: Facilitates teamwork through shared resources, project management tools, and easy model sharing.
- **No-Code Deployment**: Deploy trained models directly to various targets.

For a practical walkthrough, check out our blog post: [How to Train Your Custom Models with Ultralytics Platform](https://www.ultralytics.com/blog/how-to-train-your-custom-models-with-ultralytics-hub).

### How do I convert my annotated data to the YOLOv5 format?

Whether you annotate manually or use automated tools (like those mentioned in Section 1.2), the final labels must be in the specific **YOLO format** required by YOLOv5:

- Create one `.txt` file for each image. The filename should match the image filename (e.g., `image1.jpg` corresponds to `image1.txt`). Place these files in a `labels/` directory parallel to your `images/` directory (e.g., `../datasets/mydataset/labels/train/`).
- Each line within a `.txt` file represents one object annotation and follows the format: `class_index center_x center_y width height`.
- Coordinates (`center_x`, `center_y`, `width`, `height`) must be **normalized** (values between 0.0 and 1.0) relative to the image's dimensions.
- Class indices are **zero-based** (the first class is `0`, the second is `1`, etc.).

Many manual annotation tools offer direct export to YOLO format. If using automated models, you will need scripts or processes to convert their output (e.g., bounding box coordinates, segmentation masks) into this specific normalized text format. Ensure your final dataset structure adheres to the example provided in the guide. For more details, see our [Data Collection and Annotation Guide](https://docs.ultralytics.com/guides/data-collection-and-annotation/).

### What are the licensing options for using YOLOv5 in commercial applications?

Ultralytics provides flexible licensing tailored to different needs:

- **AGPL-3.0 License**: This open-source license is suitable for academic research, personal projects, and situations where open-source compliance is acceptable. It mandates that modifications and derivative works also be open-sourced under AGPL-3.0. Review the [AGPL-3.0 License details](https://www.ultralytics.com/legal/agpl-3-0-software-license).
- **Enterprise License**: A commercial license designed for businesses integrating YOLOv5 into proprietary products or services. This license removes the open-source obligations of AGPL-3.0, allowing for closed-source distribution. Visit our [Licensing page](https://www.ultralytics.com/license) for further details or to request an [Enterprise License](https://www.ultralytics.com/legal/enterprise-software-license).

Select the license that aligns best with your project's requirements and distribution model.
