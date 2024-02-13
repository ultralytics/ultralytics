---
comments: true
description: Kickstart your journey with Ultralytics HUB. Learn how to train and deploy YOLOv5 and YOLOv8 models in seconds with our Quickstart guide.
keywords: Ultralytics HUB, Quickstart, YOLOv5, YOLOv8, model training, quick deployment, drag-and-drop interface, real-time object detection
---

# Quickstart Guide for Ultralytics HUB

HUB is designed to be user-friendly and intuitive, with a drag-and-drop interface that allows users to easily upload their data and train new models quickly. It offers a range of pre-trained models and templates to choose from, making it easy for users to get started with training their own models. Once a model is trained, it can be easily deployed and used for real-time object detection, instance segmentation and classification tasks.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/lveF9iCMIzc?si=_Q4WB5kMB5qNe7q6"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Train Your Custom YOLO Models In A Few Clicks with Ultralytics HUB.
</p>

## Creating an Account

[Ultralytics HUB](https://hub.ultralytics.com/) offers multiple and easy account creation options to get started with. A user can user can register and sign-in using `Google`, `Apple` or `Github` account or a `work email` address(preferably for corporate users).

![Creating an Account](https://github.com/ultralytics/ultralytics/assets/19519529/1dcf454a-68ab-4821-9779-ee33a6e300cf)

## The Dashboard and details

After completing the registration and login process on the HUB, users are directed to the HUB dashboard. This dashboard provides a comprehensive overview, with the Welcome tutorial prominently displayed. Additionally, the left pane conveniently offers links for tasks such as Uploading Datasets, Creating Projects, Training Models, Integrating Third-party Applications, Accessing Support, and Managing Trash.

![HUB Dashboard](https://github.com/ultralytics/ultralytics/assets/19519529/108de60e-1b21-4f07-8d46-ed51d8439f67)

## Select the Model

Once we have decided on a Dataset, it's time to train the model. We first pick the Project name and Model name (or leave it to default, if they are not label specific), then pick an Architecture. Ultralytics provide a wide range of YOLOv8, YOLOv5 and YOLOv5u6 Architectures. You can also pick from previously trained or custom model.
The latter option allows us to fine tune option likes Pre-trained, Epochs, Image Size, Caching Strategy, Type of Device, Number of GPUs, Batch Size, AMP status and Freeze option. Read more about Models [HUB Models page](models.md).

## Train the Model

Once we reach the Model Training page, we are offered three-way option to train our model. We can either use Google Colab to simply follow the steps and use the API key provided at the page, or follow the steps to actually train the model locally. The third way is our upcoming Ultralytics Cloud , which enables you to directly train your model over cloud even more efficiently. Read more about Training the model at [Cloud Training Page](cloudtraining.md)

## Integrating the Model

`Ultralytics Hub` supports integrating the model with other third-party applications or to connect HUB from an external agent. Currently we support `Roboflow`, with very simple one click API Integration. Read more about Integrating the model at [Integration Page](integrations.md)

## Stuck? We got you!

We at Ultralytics we have a strong faith in user feedbacks and complaints. You can `Report a bug`, `Request a Feature` and/or `Ask question`.

![Support Page](https://github.com/ultralytics/ultralytics/assets/19519529/c29bf5c5-72d8-4be4-9f3f-b504968d0bef)

## Data restoration and deletion

Your Dataset in your account can be restored and/or deleted permanently from the `Trash` section in the left column.

![Trash Page](https://github.com/ultralytics/ultralytics/assets/19519529/c3d46107-aa58-4b05-a7a8-44db1ad61bb2)
