---
comments: true
description: Learn how to use Ultralytics HUB models for efficient and user-friendly AI model training. For easy model creation, training, evaluation and deployment, follow our detailed guide.
keywords: Ultralytics, HUB Models, AI model training, model creation, model training, model evaluation, model deployment
---

# Ultralytics HUB Models

[Ultralytics HUB](https://hub.ultralytics.com/) models provide a streamlined solution for training vision AI models on your custom datasets.

The process is user-friendly and efficient, involving a simple three-step creation and accelerated training powered by Ultralytics YOLOv8. During training, real-time updates on model metrics are available so that you can monitor each step of the progress. Once training is completed, you can preview your model and easily deploy it to real-world applications. Therefore, Ultralytics HUB offers a comprehensive yet straightforward system for model creation, training, evaluation, and deployment.

## Train Model

Navigate to the [Models](https://hub.ultralytics.com/models) page by clicking on the **Models** button in the sidebar.

![Ultralytics HUB screenshot of the Home page with an arrow pointing to the Models button in the sidebar](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_1.jpg)

??? tip "Tip"

    You can also train a model directly from the [Home](https://hub.ultralytics.com/home) page.

    ![Ultralytics HUB screenshot of the Home page with an arrow pointing to the Train Model card](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_2.jpg)

Click on the **Train Model** button on the top right of the page. This action will trigger the **Train Model** dialog.

![Ultralytics HUB screenshot of the Models page with an arrow pointing to the Train Model button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_3.jpg)

The **Train Model** dialog has three simple steps, explained below.

### 1. Dataset

In this step, you have to select the dataset you want to train your model on. After you selected a dataset, click **Continue**.

![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to a dataset and one to the Continue button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_4.jpg)

??? tip "Tip"

    You can skip this step if you train a model directly from the Dataset page.

    ![Ultralytics HUB screenshot of the Dataset page with an arrow pointing to the Train Model button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_5.jpg)

### 2. Model

In this step, you have to choose the project in which you want to create your model, the name of your model and your model's architecture.

??? note "Note"

    Ultralytics HUB will try to pre-select the project.

    If you opened the **Train Model** dialog as described above, Ultralytics HUB will pre-select the last project you used.

    If you opened the **Train Model** dialog from the Project page, Ultralytics HUB will pre-select the project you were inside of.

    ![Ultralytics HUB screenshot of the Project page with an arrow pointing to the Train Model button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_6.jpg)

    In case you don't have a project created yet, you can set the name of your project in this step and it will be created together with your model.

    ![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to the project name](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_7.jpg)

!!! Info "Info"

    You can read more about the available [YOLOv8](https://docs.ultralytics.com/models/yolov8) (and [YOLOv5](https://docs.ultralytics.com/models/yolov5)) architectures in our documentation.

When you're happy with your model configuration, click **Continue**.

![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to a model architecture and one to the Continue button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_8.jpg)

??? note "Note"

    By default, your model will use a pre-trained model (trained on the [COCO](https://docs.ultralytics.com/datasets/detect/coco) dataset) to reduce training time.

    You can change this behaviour by opening the **Advanced Options** accordion.

### 3. Train

In this step, you will start training you model.

Ultralytics HUB offers three training options:

- Ultralytics Cloud **(COMING SOON)**
- Google Colab
- Bring your own agent

In order to start training your model, follow the instructions presented in this step.

![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to each step](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_9.jpg)

??? note "Note"

    When you are on this step, before the training starts, you can change the default training configuration by opening the **Advanced Options** accordion.

    ![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to the Train Advanced Options](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_10.jpg)

??? note "Note"

    When you are on this step, you have the option to close the **Train Model** dialog and start training your model from the Model page later.

    ![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Start Training card](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_11.jpg)

To start training your model using Google Colab, simply follow the instructions shown above or on the Google Colab notebook.

<a href="https://colab.research.google.com/github/ultralytics/hub/blob/master/hub.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

When the training starts, you can click **Done** and monitor the training progress on the Model page.

![Ultralytics HUB screenshot of the Train Model dialog with an arrow pointing to the Done button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_12.jpg)

![Ultralytics HUB screenshot of the Model page of a model that is currently training](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_13.jpg)

??? note "Note"

    In case the training stops and a checkpoint was saved, you can resume training your model from the Model page.

    ![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Resume Training card](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_train_model_14.jpg)

## Preview Model

Ultralytics HUB offers a variety of ways to preview your trained model.

You can preview your model if you click on the **Preview** tab and upload an image in the **Test** card.

![Ultralytics HUB screenshot of the Preview tab (Test card) inside the Model page](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_preview_model_1.jpg)

You can also use our Ultralytics Cloud API to effortlessly [run inference](https://docs.ultralytics.com/hub/inference_api) with your custom model.

![Ultralytics HUB screenshot of the Preview tab (Ultralytics Cloud API card) inside the Model page](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_preview_model_2.jpg)

Furthermore, you can preview your model in real-time directly on your [iOS](https://apps.apple.com/xk/app/ultralytics/id1583935240) or [Android](https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app) mobile device by [downloading](https://ultralytics.com/app_install) our [Ultralytics HUB Mobile Application](app/index.md).

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with arrow pointing to the Real-Time Preview card](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_preview_model_3.jpg)

## Deploy Model

You can export your model to 13 different formats, including ONNX, OpenVINO, CoreML, TensorFlow, Paddle and many others.

![Ultralytics HUB screenshot of the Deploy tab inside the Model page with all formats exported](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_deploy_model_1.jpg)

??? tip "Tip"

    You can customize the export options of each format if you open the export actions dropdown and click on the **Advanced** option.

    ![Ultralytics HUB screenshot of the Deploy tab inside the Model page with an arrow pointing to the Advanced option](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_deploy_model_2.jpg)

## Share Model

!!! Info "Info"

    Ultralytics HUB's sharing functionality provides a convenient way to share models with others. This feature is designed to accommodate both existing Ultralytics HUB users and those who have yet to create an account.

??? note "Note"

    You have control over the general access of your models.

    You can choose to set the general access to "Private", in which case, only you will have access to it. Alternatively, you can set the general access to "Unlisted" which grants viewing access to anyone who has the direct link to the model, regardless of whether they have an Ultralytics HUB account or not.

Navigate to the Model page of the model you want to share, open the model actions dropdown and click on the **Share** option. This action will trigger the **Share Model** dialog.

![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Share option](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_share_model_1.jpg)

??? tip "Tip"

    You can also share a model directly from the [Models](https://hub.ultralytics.com/models) page or from the Project page of the project where your model is located.

    ![Ultralytics HUB screenshot of the Models page with an arrow pointing to the Share option of one of the models](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_share_model_2.jpg)

Set the general access to "Unlisted" and click **Save**.

![Ultralytics HUB screenshot of the Share Model dialog with an arrow pointing to the dropdown and one to the Save button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_share_model_3.jpg)

Now, anyone who has the direct link to your model can view it.

??? tip "Tip"

    You can easily click on the model's link shown in the **Share Model** dialog to copy it.

    ![Ultralytics HUB screenshot of the Share Model dialog with an arrow pointing to the model's link](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_share_model_4.jpg)

## Edit Model

Navigate to the Model page of the model you want to edit, open the model actions dropdown and click on the **Edit** option. This action will trigger the **Update Model** dialog.

![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Edit option](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_edit_model_1.jpg)

??? tip "Tip"

    You can also edit a model directly from the [Models](https://hub.ultralytics.com/models) page or from the Project page of the project where your model is located.

    ![Ultralytics HUB screenshot of the Models page with an arrow pointing to the Edit option of one of the models](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_edit_model_2.jpg)

Apply the desired modifications to your model and then confirm the changes by clicking **Save**.

![Ultralytics HUB screenshot of the Update Model dialog with an arrow pointing to the Save button](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_edit_model_3.jpg)

## Delete Model

Navigate to the Model page of the model you want to delete, open the model actions dropdown and click on the **Delete** option. This action will delete the model.

![Ultralytics HUB screenshot of the Model page with an arrow pointing to the Delete option](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_delete_model_1.jpg)

??? tip "Tip"

    You can also delete a model directly from the [Models](https://hub.ultralytics.com/models) page or from the Project page of the project where your model is located.

    ![Ultralytics HUB screenshot of the Models page with an arrow pointing to the Delete option of one of the models](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_delete_model_2.jpg)

??? note "Note"

    If you change your mind, you can restore the model from the [Trash](https://hub.ultralytics.com/trash) page.

    ![Ultralytics HUB screenshot of the Trash page with an arrow pointing to the Restore option of one of the models](https://raw.githubusercontent.com/ultralytics/assets/main/docs/hub/models/hub_delete_model_3.jpg)
