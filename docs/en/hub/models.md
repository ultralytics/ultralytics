---
comments: true
description: Learn how to efficiently train AI models using Ultralytics HUB, a streamlined solution for model creation, training, evaluation, and deployment.
keywords: Ultralytics, HUB Models, AI model training, model creation, model training, model evaluation, model deployment
---

# Ultralytics HUB Models

[Ultralytics HUB](https://hub.ultralytics.com/) models provide a streamlined solution for training vision AI models on custom datasets.

The process is user-friendly and efficient, involving a simple three-step creation and accelerated training powered by Ultralytics YOLOv8. Real-time updates on model metrics are available during training, allowing users to monitor progress at each step. Once training is completed, models can be previewed and easily deployed to real-world applications. Therefore, Ultralytics HUB offers a comprehensive yet straightforward system for model creation, training, evaluation, and deployment.

The entire process of training a model is detailed on our [Cloud Training Page](cloud-training.md).

![Preview of the Models](https://github.com/ultralytics/ultralytics/assets/19519529/a02e1441-f5f6-4935-ad75-ec18e425d8bd)

## Train Model

Navigate to the [Models](https://hub.ultralytics.com/models) page by clicking on the **Models** button in the sidebar.

Training a model using HUB is a 4-step process:

- **Execute the pre-requisites script**: Run the provided scripts to prepare the virtual environment.
- **Provide the API and start Training**: Once the model is prepared, provide the API key as instructed and execute the code block.
- **Check the results and Metrics**: Upon successful execution, a link is provided to the Metrics Page. This page offers comprehensive details on the trained model, including specifications, loss metrics, dataset information, and image distributions. Additionally, the 'Deploy' tab provides access to the trained model's documentation and license details.
- **Test your model**: Ultralytics HUB offers testing using custom images, device cameras, or links to test on `iPhone` or `Android` devices.

![Ultralytics HUB screenshot of the Home page](https://github.com/ultralytics/ultralytics/assets/19519529/61428720-aa93-4689-b209-ead7f06fa488)

!!! tip "Tip"

    You can also train a model directly from the [Home](https://hub.ultralytics.com/home) page.

    ![Ultralytics HUB screenshot of the Home page with an arrow pointing to the Train Model card](https://github.com/ultralytics/ultralytics/assets/19519529/6f9f06f7-e663-4fa7-800c-98675bf1405b)

Click on the **Train Model** button on the top right of the page to trigger the **Train Model** dialog.

The **Train Model** dialog has three simple steps:

### 1. Dataset

Select the dataset for training and click **Continue**.

![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to a dataset and one to the Continue button](https://github.com/ultralytics/ultralytics/assets/19519529/7ff90f2a-c61e-472f-a573-f725a5bddc1c)

### 2. Model

Choose the project, model name, and architecture. Read more about available architectures in our [YOLOv8](https://docs.ultralytics.com/models/yolov8) (and [YOLOv5](https://docs.ultralytics.com/models/yolov5)) documentation.

Click **Continue** when satisfied with the configuration.

![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to a model architecture and one to the Continue button](https://github.com/ultralytics/ultralytics/assets/19519529/a7a412b3-3e87-48de-b117-c506338f36fb)

!!! note "Note"

    By default, your model will use a pre-trained model (trained on the [COCO](https://docs.ultralytics.com/datasets/detect/coco) dataset) to reduce training time.

    Advanced options are available to modify this behavior.

## Preview Model

Ultralytics HUB offers various ways to preview trained models.

You can upload an image in the **Test** card under the **Preview** tab to preview your model.

![Ultralytics HUB screenshot of the Preview tab (Test card) inside the Model page](https://github.com/ultralytics/ultralytics/assets/19519529/a732d13a-8da9-40a8-9f5e-c766bec3fbe9)

Use our Ultralytics Cloud API to effortlessly [run inference](inference-api.md) with your custom model.

![Ultralytics HUB screenshot of the Preview tab (Ultralytics Cloud API card) inside the Model page](https://github.com/ultralytics/ultralytics/assets/19519529/77ae0f6c-d89e-433c-b404-77f71c06def5)

Preview your model in real-time on your [iOS](https://apps.apple.com/xk/app/ultralytics/id1583935240) or [Android](https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app) device by [downloading](https://ultralytics.com/app_install) our [Ultralytics HUB Mobile Application](app/index.md).

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with an arrow pointing to the Real-Time Preview card](https://github.com/ultralytics/ultralytics/assets/19519529/8d711052-5ab1-43bc-bc25-a8802a24b301)

## Train the model

Ultralytics HUB offers three training options:

- **Ultralytics Cloud** - Learn more about training via the Ultralytics [Cloud Training Page](cloud-training.md)
- **Google Colab**
- **Bring your own agent**

## Training the Model on Google Colab

To start training using Google Colab, follow the instructions on the Google Colab notebook.

<a href="https://colab.research.google.com/github/ultralytics/hub/blob/main/hub.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

![Google Colab Screenshot](https://github.com/ultralytics/ultralytics/assets/19519529/f19d2e04-d33c-446b-91f9-80396e02b68f)

## Bring your own Agent

Create an API endpoint through Ultralytics HUB to train the Model locally. Follow the provided steps, and access training details via the link generated on the Agent terminal.

![Bring your own agent screenshot](https://github.com/ultralytics/ultralytics/assets/19519529/7d8dcd7a-19ec-4ada-87bf-1a1ba1d01ceb)

## Deploy Model

Export your model to 13 different formats, including ONNX, OpenVINO, CoreML, TensorFlow, Paddle, and more.

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with all formats exported](https://github.com/ultralytics/ultralytics/assets/19519529/083a929d-2bbd-45f8-9dec-2767949caaba)

## Share Model

Ultralytics HUB's sharing functionality provides a convenient way to share models. Control the general access of your models, setting them to "Private" or "Unlisted".

Navigate to the Model page, open the model actions dropdown, and click on the **Share** option.

![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Share option](https://github.com/ultralytics/ultralytics/assets/19519529/ac98724e-9267-4557-a792-33073c47bbff)

Set the general access and click **Save**.

![Ultralytics HUB screenshot of the Share Model dialog with an arrow pointing to the dropdown and one to the Save button](https://github.com/ultralytics/ultralytics/assets/19519529/65afcd99-1f9e-4be8-b287-096a7c74fc0e)

Now, anyone with the direct link can view your model.

!!! tip "Tip"

    Easily copy the model's link shown in the **Share Model** dialog by clicking on it.

    ![Ultralytics HUB screenshot of the Share Model dialog with an arrow pointing to the model's link](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_share_model_4.jpg)

## Edit and Delete Model

Navigate to the Model page, open the model actions dropdown, and click on the **Edit** option to update the model. To delete the model, select the **Delete** option.

![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Edit option](https://github.com/ultralytics/ultralytics/assets/19519529/5c2db731-45dc-4f04-ac0f-9ad600c140a1)
