---
comments: true
description: Learn how to use Roboflow with Ultralytics for labeling and managing images for use in training, and for evaluating model performance.
keywords: Ultralytics, YOLOv8, Roboflow, vector analysis, confusion matrix, data management, image labeling
---

# Roboflow

[Roboflow](https://roboflow.com/?ref=ultralytics) has everything you need to build and deploy computer vision models. Connect Roboflow at any step in your pipeline with APIs and SDKs, or use the end-to-end interface to automate the entire process from image to inference. Whether you’re in need of [data labeling](https://roboflow.com/annotate?ref=ultralytics), [model training](https://roboflow.com/train?ref=ultralytics), or [model deployment](https://roboflow.com/deploy?ref=ultralytics), Roboflow gives you building blocks to bring custom computer vision solutions to your project.

!!! Question "Licensing"

    Ultralytics offers two licensing options:

    - The [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), an OSI-approved open-source license ideal for students and enthusiasts.
    - The [Enterprise License](https://ultralytics.com/license) for businesses seeking to incorporate our AI models into their products and services.

    For more details see [Ultralytics Licensing](https://ultralytics.com/license).

In this guide, we are going to showcase how to find, label, and organize data for use in training a custom Ultralytics YOLOv8 model. Use the table of contents below to jump directly to a specific section:

- Gather data for training a custom YOLOv8 model
- Upload, convert and label data for YOLOv8 format
- Pre-process and augment data for model robustness
- Dataset management for [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- Export data in 40+ formats for model training
- Upload custom YOLOv8 model weights for testing and deployment
- Gather Data for Training a Custom YOLOv8 Model

Roboflow provides two services that can help you collect data for YOLOv8 models: [Universe](https://universe.roboflow.com/?ref=ultralytics) and [Collect](https://roboflow.com/collect?ref=ultralytics).

Universe is an online repository with over 250,000 vision datasets totalling over 100 million images.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_universe.png" alt="Roboflow Universe" width="800">
</p>

With a [free Roboflow account](https://app.roboflow.com/?ref=ultralytics), you can export any dataset available on Universe. To export a dataset, click the "Download this Dataset" button on any dataset.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_dataset.png" alt="Roboflow Universe dataset export" width="800">
</p>

For YOLOv8, select "YOLOv8" as the export format:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_data_format.png" alt="Roboflow Universe dataset export" width="800">
</p>

Universe also has a page that aggregates all [public fine-tuned YOLOv8 models uploaded to Roboflow](https://universe.roboflow.com/search?q=model:yolov8). You can use this page to explore pre-trained models you can use for testing or [for automated data labeling](https://docs.roboflow.com/annotate/use-roboflow-annotate/model-assisted-labeling) or to prototype with [Roboflow inference](https://roboflow.com/inference?ref=ultralytics).

If you want to gather images yourself, try [Collect](https://github.com/roboflow/roboflow-collect), an open source project that allows you to automatically gather images using a webcam on the edge. You can use text or image prompts with Collect to instruct what data should be collected, allowing you to capture only the useful data you need to build your vision model.

## Upload, Convert and Label Data for YOLOv8 Format

[Roboflow Annotate](https://docs.roboflow.com/annotate/use-roboflow-annotate) is an online annotation tool for use in labeling images for object detection, classification, and segmentation.

To label data for a YOLOv8 object detection, instance segmentation, or classification model, first create a project in Roboflow.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_create_project.png" alt="Create a Roboflow project" width="400">
</p>

Next, upload your images, and any pre-existing annotations you have from other tools ([using one of the 40+ supported import formats](https://roboflow.com/formats?ref=ultralytics)), into Roboflow.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_upload_data.png" alt="Upload images to Roboflow" width="800">
</p>

Select the batch of images you have uploaded on the Annotate page to which you are taken after uploading images. Then, click "Start Annotating" to label images.

To label with bounding boxes, press the `B` key on your keyboard or click the box icon in the sidebar. Click on a point where you want to start your bounding box, then drag to create the box:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_annotate.png" alt="Annotating an image in Roboflow" width="800">
</p>

A pop-up will appear asking you to select a class for your annotation once you have created an annotation.

To label with polygons, press the `P` key on your keyboard, or the polygon icon in the sidebar. With the polygon annotation tool enabled, click on individual points in the image to draw a polygon.

Roboflow offers a SAM-based label assistant with which you can label images faster than ever. SAM (Segment Anything Model) is a state-of-the-art computer vision model that can precisely label images. With SAM, you can significantly speed up the image labeling process. Annotating images with polygons becomes as simple as a few clicks, rather than the tedious process of precisely clicking points around an object.

To use the label assistant, click the cursor icon in the sidebar, SAM will be loaded for use in your project.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_annotate_interactive.png" alt="Annotating an image in Roboflow with SAM-powered label assist" width="800">
</p>

Hover over any object in the image and SAM will recommend an annotation. You can hover to find the right place to annotate, then click to create your annotation. To amend your annotation to be more or less specific, you can click inside or outside the annotation SAM has created on the document.

You can also add tags to images from the Tags panel in the sidebar. You can apply tags to data from a particular area, taken from a specific camera, and more. You can then use these tags to search through data for images matching a tag and generate versions of a dataset with images that contain a particular tag or set of tags.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_tags.png" alt="Adding tags to an image in Roboflow" width="300">
</p>

Models hosted on Roboflow can be used with Label Assist, an automated annotation tool that uses your YOLOv8 model to recommend annotations. To use Label Assist, first upload a YOLOv8 model to Roboflow (see instructions later in the guide). Then, click the magic wand icon in the left sidebar and select your model for use in Label Assist.

Choose a model, then click "Continue" to enable Label Assist:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_label_assist.png" alt="Enabling Label Assist" width="800">
</p>

When you open new images for annotation, Label Assist will trigger and recommend annotations.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_label_assist.png" alt="ALabel Assist recommending an annotation" width="800">
</p>

## Dataset Management for YOLOv8

Roboflow provides a suite of tools for understanding computer vision datasets.

First, you can use dataset search to find images that meet a semantic text description (i.e. find all images that contain people), or that meet a specified label (i.e. the image is associated with a specific tag). To use dataset search, click "Dataset" in the sidebar. Then, input a search query using the search bar and associated filters at the top of the page.

For example, the following text query finds images that contain people in a dataset:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_dataset_management.png" alt="Searching for an image" width="800">
</p>

You can narrow your search to images with a particular tag using the "Tags" selector:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_filter_by_tag.png" alt="Filter images by tag" width="350">
</p>

Before you start training a model with your dataset, we recommend using Roboflow [Health Check](https://docs.roboflow.com/datasets/dataset-health-check), a web tool that provides an insight into your dataset and how you can improve the dataset prior to training a vision model.

To use Health Check, click the "Health Check" sidebar link. A list of statistics will appear that show the average size of images in your dataset, class balance, a heatmap of where annotations are in your images, and more.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_dataset_health_check.png" alt="Roboflow Health Check analysis" width="800">
</p>

Health Check may recommend changes to help enhance dataset performance. For example, the class balance feature may show that there is an imbalance in labels that, if solved, may boost performance or your model.

## Export Data in 40+ Formats for Model Training

To export your data, you will need a dataset version. A version is a state of your dataset frozen-in-time. To create a version, first click "Versions" in the sidebar. Then, click the "Create New Version" button. On this page, you will be able to choose augmentations and preprocessing steps to apply to your dataset:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_generate_dataset.png" alt="Creating a dataset version on Roboflow" width="800">
</p>

For each augmentation you select, a pop-up will appear allowing you to tune the augmentation to your needs. Here is an example of tuning a brightness augmentation within specified parameters:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_augmentations.png" alt="Applying augmentations to a dataset" width="800">
</p>

When your dataset version has been generated, you can export your data into a range of formats. Click the "Export Dataset" button on your dataset version page to export your data:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_export_data.png" alt="Exporting a dataset" width="800">
</p>

You are now ready to train YOLOv8 on a custom dataset. Follow this [written guide](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/) and [YouTube video](https://www.youtube.com/watch?v=wuZtUMEiKWY) for step-by-step instructions or refer to the [Ultralytics documentation](https://docs.ultralytics.com/modes/train/).

## Upload Custom YOLOv8 Model Weights for Testing and Deployment

Roboflow offers an infinitely scalable API for deployed models and SDKs for use with NVIDIA Jetsons, Luxonis OAKs, Raspberry Pis, GPU-based devices, and more.

You can deploy YOLOv8 models by uploading YOLOv8 weights to Roboflow. You can do this in a few lines of Python code. Create a new Python file and add the following code:

```python
import roboflow  # install with 'pip install roboflow'

roboflow.login()

rf = roboflow.Roboflow()

project = rf.workspace(WORKSPACE_ID).project("football-players-detection-3zvbc")
dataset = project.version(VERSION).download("yolov8")

project.version(dataset.version).deploy(model_type="yolov8", model_path=f"{HOME}/runs/detect/train/")
```

In this code, replace the project ID and version ID with the values for your account and project. [Learn how to retrieve your Roboflow API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

When you run the code above, you will be asked to authenticate. Then, your model will be uploaded and an API will be created for your project. This process can take up to 30 minutes to complete.

To test your model and find deployment instructions for supported SDKs, go to the "Deploy" tab in the Roboflow sidebar. At the top of this page, a widget will appear with which you can test your model. You can use your webcam for live testing or upload images or videos.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_test_project.png" alt="Running inference on an example image" width="800">
</p>

You can also use your uploaded model as a [labeling assistant](https://docs.roboflow.com/annotate/use-roboflow-annotate/model-assisted-labeling). This feature uses your trained model to recommend annotations on images uploaded to Roboflow.

## How to Evaluate YOLOv8 Models

Roboflow provides a range of features for use in evaluating models.

Once you have uploaded a model to Roboflow, you can access our model evaluation tool, which provides a confusion matrix showing the performance of your model as well as an interactive vector analysis plot. These features can help you find opportunities to improve your model.

To access a confusion matrix, go to your model page on the Roboflow dashboard, then click "View Detailed Evaluation":

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_model_eval.png" alt="Start a Roboflow model evaluation" width="800">
</p>

A pop-up will appear showing a confusion matrix:

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_confusion_matrix.png" alt="A confusion matrix" width="800">
</p>

Hover over a box on the confusion matrix to see the value associated with the box. Click on a box to see images in the respective category. Click on an image to view the model predictions and ground truth data associated with that image.

For more insights, click Vector Analysis. This will show a scatter plot of the images in your dataset, calculated using CLIP. The closer images are in the plot, the more similar they are, semantically. Each image is represented as a dot with a color between white and red. The more red the dot, the worse the model performed.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_vector_analysis.png" alt="A vector analysis plot" width="800">
</p>

You can use Vector Analysis to:

- Find clusters of images;
- Identify clusters where the model performs poorly, and;
- Visualize commonalities between images on which the model performs poorly.

## Learning Resources

Want to learn more about using Roboflow for creating YOLOv8 models? The following resources may be helpful in your work.

- [Train YOLOv8 on a Custom Dataset](https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb): Follow our interactive notebook that shows you how to train a YOLOv8 model on a custom dataset.
- [Autodistill](https://autodistill.github.io/autodistill/): Use large foundation vision models to label data for specific models. You can label images for use in training YOLOv8 classification, detection, and segmentation models with Autodistill.
- [Supervision](https://roboflow.github.io/supervision/): A Python package with helpful utilities for use in working with computer vision models. You can use supervision to filter detections, compute confusion matrices, and more, all in a few lines of Python code.
- [Roboflow Blog](https://blog.roboflow.com/): The Roboflow Blog features over 500 articles on computer vision, covering topics from how to train a YOLOv8 model to annotation best practices.
- [Roboflow YouTube channel](https://www.youtube.com/@Roboflow): Browse dozens of in-depth computer vision guides on our YouTube channel, covering topics from training YOLOv8 models to automated image labeling.

## Project Showcase

Below are a few of the many pieces of feedback we have received for using YOLOv8 and Roboflow together to create computer vision models.

<p align="center">
<img src="https://media.roboflow.com/ultralytics/rf_showcase_1.png" alt="Showcase image" width="500">
<img src="https://media.roboflow.com/ultralytics/rf_showcase_2.png" alt="Showcase image" width="500">
<img src="https://media.roboflow.com/ultralytics/rf_showcase_3.png" alt="Showcase image" width="500">
</p>
