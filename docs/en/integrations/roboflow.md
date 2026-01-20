---
comments: true
description: Learn how to label data and export datasets in YOLO format using Roboflow for training Ultralytics models.
keywords: Roboflow, Ultralytics YOLO, data labeling, computer vision, dataset export
---

# Roboflow

[Roboflow](https://roboflow.com/?ref=ultralytics) provides tools for [data labeling](https://www.ultralytics.com/glossary/data-labeling) and dataset export in various formats, including YOLO. This guide covers labeling, exporting, and deploying data for [Ultralytics YOLO](../models/index.md) models.

!!! question "Licensing"

    Ultralytics offers two licensing options to accommodate different use cases:

    - **AGPL-3.0 License**: This [OSI-approved open-source license](https://www.ultralytics.com/legal/agpl-3-0-software-license) is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details.
    - **Enterprise License**: Designed for commercial use, this license allows for the seamless integration of Ultralytics software and AI models into commercial products and services. If your scenario involves commercial applications, please reach out via [Ultralytics Licensing](https://www.ultralytics.com/license).

    For more details see the [Ultralytics Licensing page](https://www.ultralytics.com/license).

This guide demonstrates how to find, label, and organize data for training a custom [Ultralytics YOLO26](../models/yolo26.md) model using Roboflow.

- [Gather Data for Training](#gather-data-for-training-a-custom-yolo26-model)
- [Label Data](#upload-convert-and-label-data-for-yolo26-format)
- [Dataset Management](#dataset-management-for-yolo26)
- [Export Data](#export-data-in-40-formats-for-model-training)
- [Deploy Models](#upload-custom-yolo26-model-weights-for-testing-and-deployment)
- [Evaluate Models](#how-to-evaluate-yolo26-models)
- [FAQ](#faq)

## Gather Data for Training a Custom YOLO26 Model

Roboflow offers two primary services to assist in data collection for Ultralytics [YOLO models](../models/index.md): Universe and Collect. For more general information on data collection strategies, refer to our [Data Collection and Annotation Guide](../guides/data-collection-and-annotation.md).

### Roboflow Universe

Roboflow Universe is an online repository of vision [datasets](../datasets/index.md). You can export datasets in YOLO format for use with Ultralytics models.

### Roboflow Collect

If you prefer to gather images yourself, Roboflow Collect is an open-source project enabling automatic image collection via a webcam on edge devices. You can use text or image prompts to specify the data to be collected, helping capture only the necessary images for your vision model.

## Upload, Convert and Label Data for YOLO26 Format

Roboflow Annotate is an online tool for labeling images for various computer vision tasks, including [object detection](../tasks/detect.md), [classification](../tasks/classify.md), and [segmentation](../tasks/segment.md).

To label data for an Ultralytics [YOLO](../models/index.md) model, create a project in Roboflow, upload your images, and start annotating.

### Annotation Tools

- **Bounding Box Annotation**: Press `B` or click the box icon. Click and drag to create the [bounding box](https://www.ultralytics.com/glossary/bounding-box). A pop-up will prompt you to select a class for the annotation.
- **Polygon Annotation**: Used for [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation). Press `P` or click the polygon icon. Click points around the object to draw the polygon.

### Label Assistant (SAM Integration)

Roboflow integrates a [Segment Anything Model (SAM)](../models/sam.md)-based label assistant to potentially speed up annotation.

To use the label assistant, click the cursor icon in the sidebar. SAM will be enabled for your project.

Hover over an object, and SAM may suggest an annotation. Click to accept the annotation. You can refine the annotation's specificity by clicking inside or outside the suggested area.

### Tagging

You can add tags to images using the Tags panel in the sidebar. Tags can represent attributes like location, camera source, etc. These tags allow you to search for specific images and generate dataset versions containing images with particular tags.

### Label Assist (Model-Based)

Models hosted on Roboflow can be used with Label Assist to suggest annotations. Upload your YOLO model weights to Roboflow (see instructions below), then activate Label Assist via the magic wand icon in the sidebar.

## Dataset Management for YOLO26

Roboflow provides several tools for understanding and managing your computer vision [datasets](../datasets/index.md).

### Dataset Search

Use dataset search to find images based on text descriptions or specific labels/tags. Access this feature by clicking "Dataset" in the sidebar.

### Health Check

Before training, use Roboflow Health Check to gain insights into your dataset and identify potential improvements. Access it via the "Health Check" sidebar link. It provides statistics on image sizes, class balance, annotation heatmaps, and more.

<p align="center">
<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/rf-dataset-health-check.avif" alt="Roboflow Health Check analysis dashboard" width="800">
</p>

Health Check might suggest changes to enhance performance, such as addressing class imbalances identified in the class balance feature. Understanding dataset health is crucial for effective [model training](../modes/train.md).

## Pre-process and Augment Data for Model Robustness

To export your data, you need to create a dataset version, which is a snapshot of your dataset at a specific point in time. Click "Versions" in the sidebar, then "Create New Version." Here, you can apply preprocessing steps and [data augmentations](https://www.ultralytics.com/glossary/data-augmentation) to potentially enhance model robustness.

<p align="center">
<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/creating-dataset-version-on-roboflow.avif" alt="Creating Roboflow dataset version with augmentation" width="800">
</p>

For each selected augmentation, a pop-up allows you to fine-tune its parameters such as brightness. Proper augmentation can significantly improve model generalization, a key concept discussed in our [model training tips guide](../guides/model-training-tips.md).

## Export Data in 40+ Formats for Model Training

Once your dataset version is generated, you can export it in various formats suitable for model training. Click the "Export Dataset" button on the version page.

<p align="center">
<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/exporting-dataset.avif" alt="Roboflow dataset export to YOLO format" width="800">
</p>

Select the "YOLO26" format for compatibility with Ultralytics training pipelines. You are now ready to train your custom [YOLO26](../models/yolo26.md) model. Refer to the [Ultralytics Train mode documentation](../modes/train.md) for detailed instructions on initiating training with your exported dataset.

## Upload Custom YOLO26 Model Weights for Testing and Deployment

Roboflow offers a scalable API for deployed models and SDKs compatible with devices like [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing), [Luxonis OAK](https://www.luxonis.com/), [Raspberry Pi](../guides/raspberry-pi.md), and GPU-based systems. Explore various [model deployment options](../guides/model-deployment-options.md) in our guides.

You can deploy YOLO26 models by uploading their weights to Roboflow using a simple [Python](https://www.python.org/) script.

Create a new Python file and add the following code:

```python
import roboflow  # install with 'pip install roboflow'

# Log in to Roboflow (requires API key)
roboflow.login()

# Initialize Roboflow client
rf = roboflow.Roboflow()

# Define your workspace and project details
WORKSPACE_ID = "your-workspace-id"  # Replace with your actual Workspace ID
PROJECT_ID = "your-project-id"  # Replace with your actual Project ID
VERSION = 1  # Replace with your desired dataset version number
MODEL_PATH = "path/to/your/runs/detect/train/"  # Replace with the path to your YOLO26 training results directory

# Get project and version
project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
dataset = project.version(VERSION)

# Upload model weights for deployment
# Ensure MODEL_PATH points to the directory containing 'best.pt'
dataset.deploy(
    model_type="yolov8",
    model_path=MODEL_PATH,
)  # Note: Use "yolov8" as model_type for YOLO26 compatibility in Roboflow deployment

print(f"Model from {MODEL_PATH} uploaded to Roboflow project {PROJECT_ID}, version {VERSION}.")
print("Deployment may take up to 30 minutes.")
```

In this code, replace `your-workspace-id`, `your-project-id`, the `VERSION` number, and the `MODEL_PATH` with the values specific to your Roboflow account, project, and local training results directory. Ensure the `MODEL_PATH` correctly points to the directory containing your trained `best.pt` weights file.

When you run the code above, you will be asked to authenticate (usually via an API key). Then, your model will be uploaded, and an API endpoint will be created for your project. This process can take up to 30 minutes to complete.

To test your model and find deployment instructions for supported SDKs, go to the "Deploy" tab in the Roboflow sidebar. At the top of this page, a widget will appear allowing you to test your model using your webcam or by uploading images or videos.

<p align="center">
<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/running-inference-example-image.avif" alt="Roboflow deployment widget for model inference" width="800">
</p>

Your uploaded model can also be used as a labeling assistant, suggesting annotations on new images based on its training.

## How to Evaluate YOLO26 Models

Roboflow provides features for evaluating model performance. Understanding [performance metrics](../guides/yolo-performance-metrics.md) is crucial for model iteration.

After uploading a model, access the model evaluation tool via your model page on the Roboflow dashboard. Click "View Detailed Evaluation."

<p align="center">
<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/roboflow-model-evaluation.avif" alt="Initiating a Roboflow model evaluation" width="800">
</p>

This tool displays a [confusion matrix](https://www.ultralytics.com/glossary/confusion-matrix) illustrating model performance and an interactive vector analysis plot using [CLIP](https://openai.com/research/clip) embeddings. These features help identify areas for model improvement.

The confusion matrix pop-up:

<p align="center">
<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/confusion-matrix.avif" alt="A confusion matrix displayed in Roboflow" width="800">
</p>

Hover over cells to see values, and click cells to view corresponding images with model predictions and ground truth data.

Click "Vector Analysis" for a scatter plot visualizing image similarity based on CLIP embeddings. Images closer together are semantically similar. Dots represent images, colored from white (good performance) to red (poor performance).

<p align="center">
<img src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/vector-analysis-plot.avif" alt="Roboflow vector analysis plot using CLIP embeddings" width="800">
</p>

Vector Analysis helps:

- Identify image clusters.
- Pinpoint clusters where the model performs poorly.
- Understand commonalities among images causing poor performance.

## Learning Resources

- **[Train YOLO on a Custom Dataset (Colab)](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)**: Interactive [Google Colab](../integrations/google-colab.md) notebook for training on your data.
- **[Ultralytics YOLO Documentation](../models/index.md)**: Training, exporting, and deploying YOLO models.
- **[Ultralytics Blog](https://www.ultralytics.com/blog)**: Articles on computer vision and model training.
- **[Ultralytics YouTube](https://www.youtube.com/@Ultralytics)**: Video guides on model training and deployment.

## FAQ

### How do I label data for YOLO26 models using Roboflow?

Use Roboflow Annotate. Create a project, upload images, and use the annotation tools (`B` for [bounding boxes](https://www.ultralytics.com/glossary/bounding-box), `P` for polygons) or the SAM-based label assistant for faster labeling. Detailed steps are available in the [Upload, Convert and Label Data section](#upload-convert-and-label-data-for-yolo26-format).

### What services does Roboflow offer for collecting YOLO26 training data?

Roboflow provides Universe (access to numerous [datasets](../datasets/index.md)) and Collect (automated image gathering via webcam). These can help acquire the necessary [training data](https://www.ultralytics.com/glossary/training-data) for your YOLO26 model, complementing strategies outlined in our [Data Collection Guide](../guides/data-collection-and-annotation.md).

### How can I manage and analyze my YOLO26 dataset using Roboflow?

Utilize Roboflow's dataset search, tagging, and Health Check features. Search finds images by text or tags, while Health Check analyzes dataset quality (class balance, image sizes, etc.) to guide improvements before training. See the [Dataset Management section](#dataset-management-for-yolo26) for details.

### How do I export my YOLO26 dataset from Roboflow?

Create a dataset version in Roboflow, apply desired preprocessing and [augmentations](https://www.ultralytics.com/glossary/data-augmentation), then click "Export Dataset" and select the YOLO26 format. The process is outlined in the [Export Data section](#export-data-in-40-formats-for-model-training). This prepares your data for use with Ultralytics [training pipelines](../modes/train.md).

### How can I integrate and deploy YOLO26 models with Roboflow?

Upload your trained YOLO26 weights to Roboflow using the provided Python script. This creates a deployable API endpoint. Refer to the [Upload Custom Weights section](#upload-custom-yolo26-model-weights-for-testing-and-deployment) for the script and instructions. Explore further [deployment options](../guides/model-deployment-options.md) in our documentation.
