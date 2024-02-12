---
comments: true
description: Kickstart your journey with Ultralytics HUB. Learn how to train and deploy YOLOv5 and YOLOv8 models in seconds with our Quickstart guide.
keywords: Ultralytics HUB, Quickstart, YOLOv5, YOLOv8, model training, quick deployment, drag-and-drop interface, real-time object detection
---

# Quickstart Guide for Ultralytics HUB

## Creating an Account

[Ultralytics HUB](https://hub.ultralytics.com/)! offers multiple and easy account creation options to get started with. A user can user can register and sign-in using `Google`, `Apple` or `Github` account or a `work email` address(preferably for corporate users).

![alt text](/docs/en/hub/screenshots/Signin-page.png)

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

## The Dashboard and details

After completing the registration and login process on the HUB, users are directed to the HUB dashboard. This dashboard provides a comprehensive overview, with the Welcome tutorial prominently displayed. Additionally, the left pane conveniently offers links for tasks such as Uploading Datasets, Creating Projects, Training Models, Integrating Third-party Applications, Accessing Support, and Managing Trash.

![alt text](/docs/en/hub/screenshots/Dashboard.png)

## Datasets - Uploading and Usage

The Ultralytics HUB offers a swift and dependable method for uploading Datasets, encompassing a three-step process. This involves selecting a Task, providing a descriptive Name for the Dataset, and then uploading the file. Alternatively, users have the option to download and utilize a real example dataset for further exploration of the HUB's capabilities.

![alt text](/docs/en/hub/screenshots/Upload%20Dataset.png)

## Dataset Centre

Once the dataset has been uploaded, we can view them from the Datasets section in the left column. The Page displays all the Private and Publicly available Datasets.
As you browse the datasets, each one awaits your exploration. Click on any dataset to unveil its secrets, including training, validation, and test labels - the very DNA of its learning potential. Gain insights into the class distribution and description, understanding the who and what behind the data. Need more details? Explore its essence - classes, image sizes, and other intricate characteristics. Ready to unleash the power within? Simply select a dataset and embark on your model training journey.

![alt text](/docs/en/hub/screenshots/Dataset%20Centre.png)

## Explore the Dataset

The Data Exploration interface allows users to delve into the details of available datasets before initiating model training. By clicking on any dataset, you can access a comprehensive overview, including:<br />

**Labels**: Training, validation, and test labels provide essential information about data partitioning for model learning.<br />
**Class Distribution & Description**: Gain insights into the distribution of classes within the dataset and understand its subject matter through a clear description.<br />
**Dataset Details**: Uncover essential characteristics such as class structure, image dimensions, and other pertinent information.
This comprehensive exploration empowers users to make informed decisions before selecting a dataset for model training. Simply choose the desired dataset from the interface and proceed to the model training section to begin your journey.<br />

![alt text](/docs/en/hub/screenshots/Dataset%20Centre.png)

## Select the Model

Once we have decided on a Dataset, it's time to train the model. We first pick the Project name and Model name (or leave it to default, if they are not label specific), then pick an Architecture. Ultralytics provide a wide range of YOLOv8, YOLOv5 and YOLOv5u6 Architectures. You can also pick from previously trained or custom model.
The latter option allows us to fine tune option likes Pre-trained, Epochs, Image Size, Caching Strategy, Type of Device, Number of GPUs, Batch Size, AMP status and Freeze option.

![alt text](/docs/en/hub/screenshots/Training%20a%20Model.png)

## Train the Model

Once we reach the Model Training page, we are offered three-way option to train our model. We can either use Google Colab to simply follow the steps and use the API key provided at the page, or follow the steps to actually train the model locally. The third way is our upcoming Ultralytics Cloud , which enables you to directly train your model over cloud even more efficiently.

![alt text](/docs/en/hub/screenshots/Training%20Options.png)

## Training the Model on Google Colab

Training the model on Google colab has the following steps,<br />
**Execute the pre-requisites script** - Run the already mention scripts to prepare the virtual Environment.<br />
**Provide the API and start Training** - Once the model has been prepared, we can provide the API key as provided in the previous model (by simple copying and pasting the code block) and executing it.<br />
**Check the results and Metrics** - Upon successful code execution, a link is presented that directs the user to the Metrics Page. This page provides comprehensive details regarding the trained model, including model specifications, box loss, class loss, object loss, dataset information, and image distributions. Additionally, the deploy tab offers access to the trained model's documentation and license details.<br />
**Test your model** - Ultralytics HUB offers testing the model using custom Image, device camera or even links to test it using your `iPhone` or `Android` device.<br />

![alt text](/docs/en/hub/screenshots/Google_Colab.png)

## Integrating the Model

`Ultralytics Hub` supports integrating the model with other third-party applications or to connect HUB from an external agent. Currently we support `Roboflow`, with very simple one click API Integration.

![alt text](/docs/en/hub/screenshots/Integrations.png)

## Stuck? We got you!

Here at Ultralytics we have a strong faith in user feedbacks and complaints. You can `Report a bug`, `Request a Feature` and/or `Ask question`.

![alt text](/docs/en/hub/screenshots/Support%20Page.png)

## Data restoration and deletion

Your Dataset in your account can be restored and/or deleted permanently from the `Trash` section in the left column.

![alt text](/docs/en/hub/screenshots/Trash.png)
