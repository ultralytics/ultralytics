---
comments: true
description: Learn how to use Ultralytics HUB models for efficient and user-friendly AI model training. For easy model creation, training, evaluation and deployment, follow our detailed guide.
keywords: Ultralytics, HUB Models, AI model training, model creation, model training, model evaluation, model deployment
---

# Ultralytics HUB Models

[Ultralytics HUB](https://hub.ultralytics.com/) models provide a streamlined solution for training vision AI models on your custom datasets.

The process is user-friendly and efficient, involving a simple three-step creation and accelerated training powered by Ultralytics YOLOv8. During training, real-time updates on model metrics are available so that you can monitor each step of the progress. Once training is completed, you can preview your model and easily deploy it to real-world applications. Therefore, Ultralytics HUB offers a comprehensive yet straightforward system for model creation, training, evaluation, and deployment.

The entire process of training a Model is detailed on our [Cloud Training Page](cloudtraining.md)

![Preview of the Models](https://github.com/ultralytics/ultralytics/assets/19519529/a02e1441-f5f6-4935-ad75-ec18e425d8bd)

## Train Model

Navigate to the [Models](https://hub.ultralytics.com/models) page by clicking on the **Models** button in the sidebar.
Training the Model using HUB is a 4 step process <br />
**Execute the pre-requisites script** - Run the already mention scripts to prepare the virtual Environment.<br />
**Provide the API and start Training** - Once the model has been prepared, we can provide the API key as provided in the previous model (by simple copying and pasting the code block) and executing it.<br />
**Check the results and Metrics** - Upon successful code execution, a link is presented that directs the user to the Metrics Page. This page provides comprehensive details regarding the trained model, including model specifications, box loss, class loss, object loss, dataset information, and image distributions. Additionally, the deploy tab offers access to the trained model's documentation and license details.<br />
**Test your model** - Ultralytics HUB offers testing the model using custom Image, device camera or even links to test it using your `iPhone` or `Android` device.<br />

![Ultralytics HUB screenshot of the Home page](https://github.com/ultralytics/ultralytics/assets/19519529/61428720-aa93-4689-b209-ead7f06fa488)

??? tip "Tip"

    You can also train a model directly from the [Home](https://hub.ultralytics.com/home) page.

    ![Ultralytics HUB screenshot of the Home page with an arrow pointing to the Train Model card](https://github.com/ultralytics/ultralytics/assets/19519529/6f9f06f7-e663-4fa7-800c-98675bf1405b)

Click on the **Train Model** button on the top right of the page. This action will trigger the **Train Model** dialog.

The **Train Model** dialog has three simple steps, explained below.

### 1. Dataset

In this step, you have to select the dataset you want to train your model on. After you selected a dataset, click **Continue**.

![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to a dataset and one to the Continue button](https://github.com/ultralytics/ultralytics/assets/19519529/7ff90f2a-c61e-472f-a573-f725a5bddc1c)

### 2. Model

In this step, you have to choose the project in which you want to create your model, the name of your model and your model's architecture.

!!! Info "Info"

    You can read more about the available [YOLOv8](https://docs.ultralytics.com/models/yolov8) (and [YOLOv5](https://docs.ultralytics.com/models/yolov5)) architectures in our documentation.

When you're happy with your model configuration, click **Continue**.

![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to a model architecture and one to the Continue button](https://github.com/ultralytics/ultralytics/assets/19519529/a7a412b3-3e87-48de-b117-c506338f36fb)

??? note "Note"

    By default, your model will use a pre-trained model (trained on the [COCO](https://docs.ultralytics.com/datasets/detect/coco) dataset) to reduce training time.

    You can change this behavior by opening the **Advanced Options** accordion.

## Preview Model

Ultralytics HUB offers a variety of ways to preview your trained model.

You can preview your model if you click on the **Preview** tab and upload an image in the **Test** card.

![Ultralytics HUB screenshot of the Preview tab (Test card) inside the Model page](https://github.com/ultralytics/ultralytics/assets/19519529/a732d13a-8da9-40a8-9f5e-c766bec3fbe9)

You can also use our Ultralytics Cloud API to effortlessly [run inference](inference-api.md) with your custom model.

![Ultralytics HUB screenshot of the Preview tab (Ultralytics Cloud API card) inside the Model page](https://github.com/ultralytics/ultralytics/assets/19519529/77ae0f6c-d89e-433c-b404-77f71c06def5)

Furthermore, you can preview your model in real-time directly on your [iOS](https://apps.apple.com/xk/app/ultralytics/id1583935240) or [Android](https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app) mobile device by [downloading](https://ultralytics.com/app_install) our [Ultralytics HUB Mobile Application](app/index.md).

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with arrow pointing to the Real-Time Preview card](https://github.com/ultralytics/ultralytics/assets/19519529/8d711052-5ab1-43bc-bc25-a8802a24b301)

## Train the model

Ultralytics HUB offers three training options:

- **Ultralytics Cloud** - Read more about training via Ultralytics Cloud [Cloud Training Page](cloudtraining.md)
- **Google Colab**
- **Bring your own agent**

## Training the Model on Google Colab

To start training your model using Google Colab, simply follow the instructions shown above or on the Google Colab notebook.

<a href="https://colab.research.google.com/github/ultralytics/hub/blob/master/hub.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

![Google Colab Screenshot](https://github.com/ultralytics/ultralytics/assets/19519529/f19d2e04-d33c-446b-91f9-80396e02b68f)

## Bring your own Agent

A user can create API end point through Ultralytics HUB and use their own agent to train the Model locally. Simply follow the steps on the section and then we can see the details of the training by a link generated on the terminal on the Agent. <br />
The link takes us to the Metrics Information and Deployment completion page to know more about the model and deploy / share it.

![Bring your own agent screenshot](https://github.com/ultralytics/ultralytics/assets/19519529/7d8dcd7a-19ec-4ada-87bf-1a1ba1d01ceb)

## Deploy Model

You can export your model to 13 different formats, including ONNX, OpenVINO, CoreML, TensorFlow, Paddle and many others.

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with all formats exported](https://github.com/ultralytics/ultralytics/assets/19519529/083a929d-2bbd-45f8-9dec-2767949caaba)

## Share Model

!!! Info "Info"

    Ultralytics HUB's sharing functionality provides a convenient way to share models with others. This feature is designed to accommodate both existing Ultralytics HUB users and those who have yet to create an account.

??? note "Note"

    You have control over the general access of your models.

    You can choose to set the general access to "Private", in which case, only you will have access to it. Alternatively, you can set the general access to "Unlisted" which grants viewing access to anyone who has the direct link to the model, regardless of whether they have an Ultralytics HUB account or not.

Navigate to the Model page of the model you want to share, open the model actions dropdown and click on the **Share** option. This action will trigger the **Share Model** dialog.

![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Share option](https://github.com/ultralytics/ultralytics/assets/19519529/ac98724e-9267-4557-a792-33073c47bbff)

Set the general access to "Unlisted" and click **Save**.

![Ultralytics HUB screenshot of the Share Model dialog with an arrow pointing to the dropdown and one to the Save button](https://github.com/ultralytics/ultralytics/assets/19519529/65afcd99-1f9e-4be8-b287-096a7c74fc0e)

Now, anyone who has the direct link to your model can view it.

??? tip "Tip"

    You can easily click on the model's link shown in the **Share Model** dialog to copy it.

    ![Ultralytics HUB screenshot of the Share Model dialog with an arrow pointing to the model's link](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_share_model_4.jpg)

## Edit and Delete Model

Navigate to the Model page of the model you want to edit, open the model actions dropdown and click on the **Edit** option. This action will trigger the **Update Model** dialog. Navigate to the Model page of the model you want to delete, open the model actions dropdown and click on the **Delete** option. This action will delete the model.
![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Edit option](https://github.com/ultralytics/ultralytics/assets/19519529/5c2db731-45dc-4f04-ac0f-9ad600c140a1)
