---
comments: true
description: Learn how to use Ultralytics HUB for cloud for efficient and user-friendly AI model training. For easy model creation, training, evaluation and deployment, follow our detailed guide.
keywords: Ultralytics, HUB Models, AI model training, model creation, model training, model evaluation, model deployment
---

# Cloud Training

Ultralytics provides a web-based cloud training platform, enabling rapid and streamlined deployment of custom object detection models. Users benefit from a straightforward interface that facilitates the selection of their desired dataset and training method. Ultralytics further streamlines the process by offering a diverse array of pre-built options and architectural configurations.

![cloud training cover](https://github.com/ultralytics/ultralytics/assets/19519529/cbfdb3b8-ad35-44a6-afe6-61ec0b8e8b8d)

Read more about creating and other details of a Model at our [HUB Models page](models.md)

## Selecting an Instance

For details on Picking a model, and instances for it, please read [Instances guide Page](models.md)

## Steps to train the Model

Once the instance has been selected, training a model using ultralytics Hub is a three step process, as below: <br />

1. Picking a Dataset - Read more about Dataset, steps to add/remove dataset from [Dataset page](datasets.md) <br />
2. Picking a Model - Read more about Models, steps to create / share and handle a model [HUB Models page](models.md) <br />
3. Training the Model on the chosen Dataset <br />

Ultralytics HUB offers three training options:

- **Ultralytics Cloud**
- **Google Colab** - Read more about training via Google Colab [HUB Models page](models.md)
- **Bring your own agent** - Read more about training via your own Agent [HUB Models page](models.md)

In order to start training your model, follow the instructions presented in these steps.

## Training via Ultralytics Cloud

To start training your model using Ultralytics Cloud, we need to simply select the Training Duration, Available Instances, and Payment options.<br />

**Training Duration** - The Ultralytics offers two kind of training durations <br />

1. Training based on `Epochs` - This option lets you train your model based on number of times your Dataset needs to go through the cycle of Train, Label and Test. The exact pricing based on number of Epochs is hard to determine. Hence, if the credit gets exhausted before intended number of Epochs, the training pauses and we get a prompt to Top-up and resume Training. <br />
2. Timed Training - The timed training features allows you to fix the time duration of the entire Training process and also determines the estimated amount before the start of Training. <br />

![Ultralytics cloud screenshot of training Duration options](https://github.com/ultralytics/ultralytics/assets/19519529/47b96f3f-a9ea-441a-b065-cba97edc333f)

When the training starts, you can click **Done** and monitor the training progress on the Model page.

## Monitor your training

Once the model and mode of the training has been selected, a User can monitor the training procedure on the `Train` section with the link provided in the terminal (on your agent / Google colab) or a button from Ultralytics Cloud.

![Monitor your Training](https://github.com/ultralytics/ultralytics/assets/19519529/316f8301-0d60-465e-8c99-aa3daf66433c)

## Stopping and resuming your training

Once the training has started a user can `Stop` the training, which will also correspondingly pause the credit usage for the user. A user can again `Resume` the training from the point as described in the below screenshot.

![Pausing and Resuming Training](https://github.com/ultralytics/ultralytics/assets/19519529/b2707a93-fa5c-4ee2-8443-6be9e1c2857d)

## Payments and Billing options

Ultralytics HUB offers `Pay Now` as upfront and/or use `Ultralytics HUB Account` as a wallet to top-up and fulfil the billing. A user can pick from amongst two types of Account namely `Free` and `Pro` user. <br />
The user can navigate to the profile by clicking the Profile picture in the bottom left corner
![Clicking profile picture](https://github.com/ultralytics/ultralytics/assets/19519529/53e5410e-06f5-4b40-b29d-ef00b5779163)

Click on the Billing tab to know about your current plan and option to upgrade it.
![Clicking Upgrade button](https://github.com/ultralytics/ultralytics/assets/19519529/361b43c7-a9d4-4d05-b80b-dc1fa8bce829)

User is prompted about different available plans, and can pick from the available plans as stated below.
![Picking a plan](https://github.com/ultralytics/ultralytics/assets/19519529/4326b01c-0d7d-4850-ac4f-ced2de3339ee)

The user will then Navigate to the Payment page, fill in the details and payment is done.
![Payment Page](https://github.com/ultralytics/ultralytics/assets/19519529/5deebabe-1d8a-485a-b290-e038729c849f)
