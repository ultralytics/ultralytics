---
comments: true
description: Learn how to use Ultralytics HUB for efficient and user-friendly AI model training in the cloud. Follow our detailed guide for easy model creation, training, evaluation, and deployment.
keywords: Ultralytics, HUB Models, AI model training, model creation, model training, model evaluation, model deployment
---

# Cloud Training

[Ultralytics HUB](https://hub.ultralytics.com/) provides a powerful and user-friendly cloud platform to train custom object detection models. Easily select your dataset and the desired training method, then kick off the process with just a few clicks. Ultralytics HUB offers pre-built options and various model architectures to streamline your workflow.

![cloud training cover](https://github.com/ultralytics/ultralytics/assets/19519529/cbfdb3b8-ad35-44a6-afe6-61ec0b8e8b8d)

Read more about creating and other details of a Model at our [HUB Models page](models.md)

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ie3vLUDNYZo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> New Feature 🌟 Introducing Ultralytics HUB Cloud Training
</p>

## Selecting an Instance

For details on picking a model and instances for it, please read our [Instances guide Page](models.md)

## Steps to Train the Model

Once the instance has been selected, training a model using Ultralytics HUB is a three-step process, as below:

1. Picking a Dataset - Read more about datasets, steps to add/remove datasets from the [Dataset page](datasets.md)
2. Picking a Model - Read more about models, steps to create/share and handle a model on the [HUB Models page](models.md)
3. Training the Model on the Chosen Dataset

Ultralytics HUB offers three training options:

- **Ultralytics Cloud** - Explained in this page.
- **Google Colab** - Train on Google's popular Colab notebooks.
- **Bring your own agent** - Train models locally on your own hardware or on-premise GPU servers.

In order to start training your model, follow the instructions presented in these steps.

## Training via Ultralytics Cloud

To start training your model using Ultralytics Cloud, simply select the Training Duration, Available Instances, and Payment options.

**Training Duration** - Ultralytics offers two kinds of training durations:

1. Training based on `Epochs`: This option allows you to train your model based on the number of times your dataset needs to go through the cycle of train, label, and test. The exact pricing based on the number of epochs is hard to determine. Hence, if the credit gets exhausted before the intended number of epochs, the training pauses, and you get a prompt to top-up and resume training.
2. Timed Training: The timed training feature allows you to fix the time duration of the entire training process and also determines the estimated amount before the start of training.

![Ultralytics cloud screenshot of training duration options](https://github.com/ultralytics/ultralytics/assets/19519529/47b96f3f-a9ea-441a-b065-cba97edc333f)

When the training starts, you can click **Done** and monitor the training progress on the Model page.

## Monitor Your Training

Once the model and mode of training have been selected, you can monitor the training procedure on the `Train` section with the link provided in the terminal (on your agent/Google Colab) or a button from Ultralytics Cloud.

![Monitor your Training](https://github.com/ultralytics/ultralytics/assets/19519529/316f8301-0d60-465e-8c99-aa3daf66433c)

## Stopping and Resuming Your Training

Once the training has started, you can `Stop` the training, which will also correspondingly pause the credit usage. You can then `Resume` the training from the point where it stopped.

![Pausing and Resuming Training](https://github.com/ultralytics/ultralytics/assets/19519529/b2707a93-fa5c-4ee2-8443-6be9e1c2857d)

## Payments and Billing Options

Ultralytics HUB offers `Pay Now` as upfront and/or using `Ultralytics HUB Account` as a wallet to top up and fulfill the billing. You can choose from two types of accounts: `Free` and `Pro` user.

To access your profile, click on the profile picture in the bottom left corner.

![Clicking profile picture](https://github.com/ultralytics/ultralytics/assets/19519529/53e5410e-06f5-4b40-b29d-ef00b5779163)

Click on the Billing tab to view your current plan and options to upgrade it.

![Clicking Upgrade button](https://github.com/ultralytics/ultralytics/assets/19519529/361b43c7-a9d4-4d05-b80b-dc1fa8bce829)

You will be prompted with different available plans, and you can pick from the available plans as shown below.

![Picking a plan](https://github.com/ultralytics/ultralytics/assets/19519529/4326b01c-0d7d-4850-ac4f-ced2de3339ee)

Navigate to the Payment page, fill in the details, and complete the payment.

![Payment Page](https://github.com/ultralytics/ultralytics/assets/19519529/5deebabe-1d8a-485a-b290-e038729c849f)
