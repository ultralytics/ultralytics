---
comments: true
description: Learn how to gather, label, and deploy data for custom Ultralytics YOLO models using Roboflow's powerful tools. Optimize your computer vision pipeline effortlessly.
keywords: Roboflow, Ultralytics YOLO, data labeling, computer vision, model training, model deployment, dataset management, automated image annotation, AI tools
---

# Roboflow Integration

[Roboflow](https://roboflow.com/?ref=ultralytics) provides a suite of tools designed for building and deploying [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models. You can integrate Roboflow at various stages of your development pipeline using their APIs and SDKs, or utilize its end-to-end interface to manage the process from image collection to inference. Roboflow offers functionalities for [data labeling](https://www.ultralytics.com/glossary/data-labeling), [model training](https://docs.ultralytics.com/modes/train/), and [model deployment](https://docs.ultralytics.com/guides/model-deployment-options/), providing components for developing custom computer vision solutions alongside Ultralytics tools.

!!! question "Licensing"

    Ultralytics offers two licensing options to accommodate different use cases:

    - **AGPL-3.0 License**: This [OSI-approved open-source license](https://www.ultralytics.com/legal/agpl-3-0-software-license) is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details.
    - **Enterprise License**: Designed for commercial use, this license allows for the seamless integration of Ultralytics software and AI models into commercial products and services. If your scenario involves commercial applications, please reach out via [Ultralytics Licensing](https://www.ultralytics.com/license).

    For more details see the [Ultralytics Licensing page](https://www.ultralytics.com/license).

This guide demonstrates how to find, label, and organize data for training a custom [Ultralytics YOLO11](../models/yolo11.md) model using Roboflow.

- [Gather Data for Training a Custom YOLO11 Model](#gather-data-for-training-a-custom-yolo11-model)
- [Upload, Convert and Label Data for YOLO11 Format](#upload-convert-and-label-data-for-yolo11-format)
- [Pre-process and Augment Data for Model Robustness](#pre-process-and-augment-data-for-model-robustness)
- [Dataset Management for YOLO11](#dataset-management-for-yolo11)
- [Export Data in 40+ Formats for Model Training](#export-data-in-40-formats-for-model-training)
- [Upload Custom YOLO11 Model Weights for Testing and Deployment](#upload-custom-yolo11-model-weights-for-testing-and-deployment)
- [How to Evaluate YOLO11 Models](#how-to-evaluate-yolo11-models)
- [Learning Resources](#learning-resources)
- [Project Showcase](#project-showcase)
- [FAQ](#faq)

## Gather Data for Training a Custom YOLO11 Model

Roboflow offers two primary services to assist in data collection for Ultralytics [YOLO models](../models/index.md): Universe and Collect. For more general information on data collection strategies, refer to our [Data Collection and Annotation Guide](../guides/data-collection-and-annotation.md).

### Roboflow Universe

Roboflow Universe is an online repository featuring a large number of vision [datasets](../datasets/index.md).

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-universe.avif" alt="Roboflow Universe" width="800">
</p>

With a Roboflow account, you can export datasets available on Universe. To export a dataset, use the "Download this Dataset" button on the relevant dataset page.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-universe-dataset-export.avif" alt="Roboflow Universe dataset export" width="800">
</p>

For compatibility with Ultralytics [YOLO11](../models/yolo11.md), select "YOLO11" as the export format:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-universe-dataset-export-1.avif" alt="Roboflow Universe dataset export format selection" width="800">
</p>

Universe also features a page aggregating public fine-tuned YOLO models uploaded to Roboflow. This can be useful for exploring pre-trained models for testing or automated data labeling.

### Roboflow Collect

If you prefer to gather images yourself, Roboflow Collect is an open-source project enabling automatic image collection via a webcam on edge devices. You can use text or image prompts to specify the data to be collected, helping capture only the necessary images for your vision model.

## Upload, Convert and Label Data for YOLO11 Format

Roboflow Annotate is an online tool for labeling images for various computer vision tasks, including [object detection](../tasks/detect.md), [classification](../tasks/classify.md), and [segmentation](../tasks/segment.md).

To label data for an Ultralytics [YOLO](../models/index.md) model (which supports detection, instance segmentation, classification, pose estimation, and OBB), begin by creating a project in Roboflow.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/create-roboflow-project.avif" alt="Create a Roboflow project" width="400">
</p>

Next, upload your images and any existing annotations from other tools into Roboflow.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/upload-images-to-roboflow.avif" alt="Upload images to Roboflow" width="800">
</p>

After uploading, you'll be directed to the Annotate page. Select the batch of uploaded images and click "Start Annotating" to begin labeling.

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

Models hosted on Roboflow can be used with Label Assist, an automated annotation tool that leverages your trained [YOLO11](../models/yolo11.md) model to suggest annotations. First, upload your YOLO11 model weights to Roboflow (see instructions below). Then, activate Label Assist by clicking the magic wand icon in the left sidebar and selecting your model.

Choose your model and click "Continue" to enable Label Assist:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-label-assist.avif" alt="Enabling Label Assist in Roboflow" width="800">
</p>

When you open new images for annotation, Label Assist may automatically suggest annotations based on your model's predictions.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-label-assist.avif" alt="Label Assist recommending an annotation based on a trained model" width="800">
</p>

## Dataset Management for YOLO11

Roboflow provides several tools for understanding and managing your computer vision [datasets](../datasets/index.md).

### Dataset Search

Use dataset search to find images based on semantic text descriptions (e.g., "find all images containing people") or specific labels/tags. Access this feature by clicking "Dataset" in the sidebar and using the search bar and filters.

For example, searching for images containing people:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/searching-for-an-image.avif" alt="Searching for an image in a Roboflow dataset" width="800">
</p>

You can refine searches using tags via the "Tags" selector:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/filter-images-by-tag.avif" alt="Filtering images by tag in Roboflow" width="350">
</p>

### Health Check

Before training, use Roboflow Health Check to gain insights into your dataset and identify potential improvements. Access it via the "Health Check" sidebar link. It provides statistics on image sizes, class balance, annotation heatmaps, and more.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-dataset-health-check.avif" alt="Roboflow Health Check analysis dashboard" width="800">
</p>

Health Check might suggest changes to enhance performance, such as addressing class imbalances identified in the class balance feature. Understanding dataset health is crucial for effective [model training](../modes/train.md).

## Pre-process and Augment Data for Model Robustness

To export your data, you need to create a dataset version, which is a snapshot of your dataset at a specific point in time. Click "Versions" in the sidebar, then "Create New Version." Here, you can apply preprocessing steps and [data augmentations](https://www.ultralytics.com/glossary/data-augmentation) to potentially enhance model robustness.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/creating-dataset-version-on-roboflow.avif" alt="Creating a dataset version on Roboflow with preprocessing and augmentation options" width="800">
</p>

For each selected augmentation, a pop-up allows you to fine-tune its parameters such as brightness. Proper augmentation can significantly improve model generalization, a key concept discussed in our [model training tips guide](../guides/model-training-tips.md).

## Export Data in 40+ Formats for Model Training

Once your dataset version is generated, you can export it in various formats suitable for model training. Click the "Export Dataset" button on the version page.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/exporting-dataset.avif" alt="Exporting a dataset from Roboflow" width="800">
</p>

Select the "YOLO11" format for compatibility with Ultralytics training pipelines. You are now ready to train your custom [YOLO11](../models/yolo11.md) model. Refer to the [Ultralytics Train mode documentation](../modes/train.md) for detailed instructions on initiating training with your exported dataset.

## Upload Custom YOLO11 Model Weights for Testing and Deployment

Roboflow offers a scalable API for deployed models and SDKs compatible with devices like [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing), [Luxonis OAK](https://www.luxonis.com/), [Raspberry Pi](../guides/raspberry-pi.md), and GPU-based systems. Explore various [model deployment options](../guides/model-deployment-options.md) in our guides.

You can deploy YOLO11 models by uploading their weights to Roboflow using a simple [Python](https://www.python.org/) script.

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
MODEL_PATH = "path/to/your/runs/detect/train/"  # Replace with the path to your YOLO11 training results directory

# Get project and version
project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
dataset = project.version(VERSION)

# Upload model weights for deployment
# Ensure MODEL_PATH points to the directory containing 'best.pt'
dataset.deploy(
    model_type="yolov8",
    model_path=MODEL_PATH,
)  # Note: Use "yolov8" as model_type for YOLO11 compatibility in Roboflow deployment

print(f"Model from {MODEL_PATH} uploaded to Roboflow project {PROJECT_ID}, version {VERSION}.")
print("Deployment may take up to 30 minutes.")
```

In this code, replace `your-workspace-id`, `your-project-id`, the `VERSION` number, and the `MODEL_PATH` with the values specific to your Roboflow account, project, and local training results directory. Ensure the `MODEL_PATH` correctly points to the directory containing your trained `best.pt` weights file.

When you run the code above, you will be asked to authenticate (usually via an API key). Then, your model will be uploaded, and an API endpoint will be created for your project. This process can take up to 30 minutes to complete.

To test your model and find deployment instructions for supported SDKs, go to the "Deploy" tab in the Roboflow sidebar. At the top of this page, a widget will appear allowing you to test your model using your webcam or by uploading images or videos.

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/running-inference-example-image.avif" alt="Running inference on an example image using the Roboflow deployment widget" width="800">
</p>

Your uploaded model can also be used as a labeling assistant, suggesting annotations on new images based on its training.

## How to Evaluate YOLO11 Models

Roboflow provides features for evaluating model performance. Understanding [performance metrics](../guides/yolo-performance-metrics.md) is crucial for model iteration.

After uploading a model, access the model evaluation tool via your model page on the Roboflow dashboard. Click "View Detailed Evaluation."

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/roboflow-model-evaluation.avif" alt="Initiating a Roboflow model evaluation" width="800">
</p>

This tool displays a [confusion matrix](https://www.ultralytics.com/glossary/confusion-matrix) illustrating model performance and an interactive vector analysis plot using [CLIP](https://openai.com/research/clip) embeddings. These features help identify areas for model improvement.

The confusion matrix pop-up:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/confusion-matrix.avif" alt="A confusion matrix displayed in Roboflow" width="800">
</p>

Hover over cells to see values, and click cells to view corresponding images with model predictions and ground truth data.

Click "Vector Analysis" for a scatter plot visualizing image similarity based on CLIP embeddings. Images closer together are semantically similar. Dots represent images, colored from white (good performance) to red (poor performance).

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/vector-analysis-plot.avif" alt="A vector analysis plot in Roboflow using CLIP embeddings" width="800">
</p>

Vector Analysis helps:

- Identify image clusters.
- Pinpoint clusters where the model performs poorly.
- Understand commonalities among images causing poor performance.

## Learning Resources

Explore these resources to learn more about using Roboflow with Ultralytics YOLO11:

- **[Train YOLO11 on a Custom Dataset (Colab)](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)**: An interactive [Google Colab](../integrations/google-colab.md) notebook guiding you through training YOLO11 on your data.
- **[YOLO11 Documentation](../models/yolo11.md)**: Learn about training, exporting, and deploying YOLO11 models within the Ultralytics framework.
- **[Ultralytics Blog](https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai)**: Features articles on computer vision, including [YOLO11 training](../modes/train.md) and annotation best practices.
- **[Ultralytics YouTube Channel](https://www.youtube.com/@Ultralytics)**: Offers in-depth video guides on computer vision topics, from model training to automated labeling and [deployment](../guides/model-deployment-options.md).

## Project Showcase

Feedback from users combining Ultralytics YOLO11 and Roboflow:

<p align="center">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-showcase-1.avif" alt="Showcase image 1" width="500">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-showcase-2.avif" alt="Showcase image 2" width="500">
<img src="https://github.com/ultralytics/docs/releases/download/0/rf-showcase-3.avif" alt="Showcase image 3" width="500">
</p>

## FAQ

## Frequently Asked Questions

### How do I label data for YOLO11 models using Roboflow?

Use Roboflow Annotate. Create a project, upload images, and use the annotation tools (`B` for [bounding boxes](https://www.ultralytics.com/glossary/bounding-box), `P` for polygons) or the SAM-based label assistant for faster labeling. Detailed steps are available in the [Upload, Convert and Label Data section](#upload-convert-and-label-data-for-yolo11-format).

### What services does Roboflow offer for collecting YOLO11 training data?

Roboflow provides Universe (access to numerous [datasets](../datasets/index.md)) and Collect (automated image gathering via webcam). These can help acquire the necessary [training data](https://www.ultralytics.com/glossary/training-data) for your YOLO11 model, complementing strategies outlined in our [Data Collection Guide](../guides/data-collection-and-annotation.md).

### How can I manage and analyze my YOLO11 dataset using Roboflow?

Utilize Roboflow's dataset search, tagging, and Health Check features. Search finds images by text or tags, while Health Check analyzes dataset quality (class balance, image sizes, etc.) to guide improvements before training. See the [Dataset Management section](#dataset-management-for-yolo11) for details.

### How do I export my YOLO11 dataset from Roboflow?

Create a dataset version in Roboflow, apply desired preprocessing and [augmentations](https://www.ultralytics.com/glossary/data-augmentation), then click "Export Dataset" and select the YOLO11 format. The process is outlined in the [Export Data section](#export-data-in-40-formats-for-model-training). This prepares your data for use with Ultralytics [training pipelines](../modes/train.md).

### How can I integrate and deploy YOLO11 models with Roboflow?

Upload your trained YOLO11 weights to Roboflow using the provided Python script. This creates a deployable API endpoint. Refer to the [Upload Custom Weights section](#upload-custom-yolo11-model-weights-for-testing-and-deployment) for the script and instructions. Explore further [deployment options](../guides/model-deployment-options.md) in our documentation.
