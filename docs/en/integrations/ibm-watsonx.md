---
title: Train YOLO26 Models with IBM Watsonx
comments: true
description: Train Ultralytics YOLO26 object detection models in IBM watsonx.ai Jupyter notebooks, from project setup and Kaggle data loading to fine-tuning and evaluation.
keywords: IBM Watsonx, IBM Watsonx AI, IBM Watson Integration, IBM Watson Features, YOLO26, Ultralytics, Model Training, GPU, TPU, cloud computing
---

# A Step-by-Step Guide to Training YOLO26 Models with IBM Watsonx

Scalable [computer vision solutions](../guides/steps-of-a-cv-project.md) need infrastructure for data, training, and deployment. IBM Watsonx is an AI and data platform that supports the development, deployment, and management of AI models. It covers the entire AI lifecycle and integrates with IBM Cloud services.

You can train [Ultralytics YOLO26 models](https://platform.ultralytics.com/ultralytics/yolo26) using IBM Watsonx. It's a good option for enterprises interested in efficient [model training](../modes/train.md), fine-tuning for specific tasks, and improving [model performance](../guides/model-evaluation-insights.md) with robust tools and a user-friendly setup. This guide walks through training YOLO26 with IBM Watsonx, from setting up your environment to evaluating your trained models.

## What is IBM Watsonx?

[Watsonx](https://www.ibm.com/products/watsonx) is IBM's cloud-based platform designed for commercial [generative AI](https://www.ultralytics.com/glossary/generative-ai) and scientific data. IBM Watsonx's three components - `watsonx.ai`, `watsonx.data`, and `watsonx.governance` - come together to create an end-to-end, trustworthy AI platform that can accelerate AI projects aimed at solving business problems. It provides tools for building, training, and [deploying machine learning models](../guides/model-deployment-options.md) and makes it easy to connect with various data sources.

<p align="center">
  <img width="800" src="https://cdn.ul.run/i/4bdc3428aca8607280cf61dea931dd67.avif" alt="IBM Watsonx AI platform architecture overview">
</p>

Its interface and collaboration features support model development, management, and deployment. IBM Watsonx serves computer vision, predictive analytics, [natural language processing](https://www.ultralytics.com/glossary/natural-language-processing-nlp), and other AI applications.

## Key Features of IBM Watsonx

IBM Watsonx is made of three main components: `watsonx.ai`, `watsonx.data`, and `watsonx.governance`. Each component covers a different aspect of AI and data management.

### Watsonx.ai

[Watsonx.ai](https://www.ibm.com/products/watsonx-ai) provides tools for AI development and offers access to IBM-supported custom models, third-party models like [Llama 3](https://www.ultralytics.com/blog/getting-to-know-metas-llama-3), and IBM's own Granite models. It includes the Prompt Lab for experimenting with AI prompts, the Tuning Studio for improving model performance with labeled data, and the Flows Engine for simplifying generative AI application development. Also, it offers comprehensive tools for automating the AI model lifecycle and connecting to various APIs and libraries.

### Watsonx.data

[Watsonx.data](https://www.ibm.com/products/watsonx-data) supports both cloud and on-premises deployments through the IBM Storage Fusion HCI integration. Its user-friendly console provides centralized access to data across environments and makes data exploration easy with common SQL. It optimizes workloads with efficient query engines like Presto and Spark, accelerates data insights with an AI-powered semantic layer, includes a [vector database](https://www.ultralytics.com/glossary/vector-database) for AI relevance, and supports open data formats for easy sharing of analytics and AI data.

### Watsonx.governance

[Watsonx.governance](https://www.ibm.com/products/watsonx-governance) makes compliance easier by automatically identifying regulatory changes and enforcing policies. It links requirements to internal risk data and provides up-to-date AI factsheets. The platform helps manage risk with alerts and tools to detect issues such as [bias and drift](../guides/model-monitoring-and-maintenance.md). It also automates the monitoring and documentation of the AI lifecycle, organizes AI development with a model inventory, and enhances collaboration with user-friendly dashboards and reporting tools.

## How to Train YOLO26 Using IBM Watsonx

You can use IBM Watsonx to accelerate your YOLO26 model training workflow.

### Prerequisites

You need an [IBM Cloud account](https://cloud.ibm.com/registration) to create a watsonx.ai project, and you'll also need a [Kaggle](kaggle.md) account to load the dataset.

### Step 1: Set Up Your Environment

First, you'll need to set up an IBM account to use a Jupyter Notebook. Log in to [watsonx.ai](https://eu-de.dataplatform.cloud.ibm.com/registration/stepone?preselect_region=true) using your IBM Cloud account.

Then, create a [watsonx.ai project](https://www.ibm.com/docs/en/watsonx/saas?topic=projects-creating-project), and a [Jupyter Notebook](https://www.ibm.com/docs/en/watsonx/saas?topic=editor-creating-managing-notebooks).

Once you do so, a notebook environment will open for you to load your dataset. You can use the code from this tutorial to train a simple [object detection](https://www.ultralytics.com/glossary/object-detection) model.

### Step 2: Install and Import Relevant Libraries

Next, you can install and import the necessary Python libraries.

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required packages
        pip install torch torchvision torchaudio
        pip install ultralytics-opencv-headless
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLO26, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

Then, you can import the needed packages.

!!! example "Import Relevant Libraries"

    === "Python"

        ```python
        import glob
        import os
        import shutil

        import yaml
        from IPython.display import display
        from PIL import Image

        import ultralytics

        ultralytics.checks()
        ```

### Step 3: Load the Data

For this tutorial, we will use a [marine litter dataset](https://www.kaggle.com/datasets/atiqishrak/trash-dataset-icra19) available on Kaggle (also available as [Trash-ICRA19 on Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/trash-icra19)). With this dataset, we will custom-train a YOLO26 model to detect and classify litter and biological objects in underwater images.

We can load the dataset directly into the notebook using the Kaggle API. First, create a free Kaggle account. Once you have created an account, you'll need to generate an API key. Directions for generating your key can be found in the [Kaggle API documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md) under the section "API credentials".

Copy and paste your Kaggle username and API key into the following code. Then run the code to install the API and load the dataset into Watsonx.

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install kaggle
        pip install kaggle
        ```

After installing Kaggle, we can load the dataset into Watsonx.

!!! example "Load the Data"

    === "Python"

        ```python
        # Replace "username" string with your username
        os.environ["KAGGLE_USERNAME"] = "username"
        # Replace "apiKey" string with your key
        os.environ["KAGGLE_KEY"] = "apiKey"

        # Load dataset
        os.system("kaggle datasets download atiqishrak/trash-dataset-icra19 --unzip")

        # Store working directory path as work_dir
        work_dir = os.getcwd()

        # Print work_dir path
        print(os.getcwd())

        # Print work_dir contents
        print(os.listdir(f"{work_dir}"))

        # Print trash_ICRA19 subdirectory contents
        print(os.listdir(f"{work_dir}/trash_ICRA19"))
        ```

After loading the dataset, we printed and saved our working directory. We have also printed the contents of our working directory to confirm the "trash_ICRA19" dataset was loaded properly.

If you see "trash_ICRA19" among the directory's contents, then it has loaded successfully. You should see three files/folders: a `config.yaml` file, a `videos_for_testing` directory, and a `dataset` directory. We will ignore the `videos_for_testing` directory, so feel free to delete it.

We will use the `config.yaml` file and the contents of the dataset directory to train our object detection model. Here is a sample image from our marine litter dataset.

<p align="center">
  <img width="400" src="https://cdn.ul.run/i/c5011609236bd52b55cb4bdda131c483.avif" alt="Underwater marine litter sample image with a bounding box annotation">
</p>

### Step 4: Preprocess the Data

All labels in the marine litter dataset are already formatted as YOLO .txt files. However, we need to rearrange the image and label directories so the model can find them. Right now, our loaded dataset directory follows this structure:

<p align="center">
  <img width="400" src="https://cdn.ul.run/i/5c09bf4043815ed8207c803923d2a913.avif" alt="Original trash_ICRA19 dataset directory structure">
</p>

By default, YOLO models require separate `images` and `labels` subdirectories within each train/val/test split. We need to reorganize the directory into the following structure:

<p align="center">
  <img width="400" src="https://cdn.ul.run/i/f851268766ed65e402efed6395cb6058.avif" alt="Required YOLO dataset directory structure with images and labels subdirectories">
</p>

To reorganize the dataset directory, we can run the following script:

!!! example "Preprocess the Data"

    === "Python"

        ```python
        # Function to reorganize dir
        def organize_files(directory):
            for subdir in ["train", "test", "val"]:
                subdir_path = os.path.join(directory, subdir)
                if not os.path.exists(subdir_path):
                    continue

                images_dir = os.path.join(subdir_path, "images")
                labels_dir = os.path.join(subdir_path, "labels")

                # Create image and label subdirs if non-existent
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(labels_dir, exist_ok=True)

                # Move images and labels to respective subdirs
                for filename in os.listdir(subdir_path):
                    if filename.endswith(".txt"):
                        shutil.move(os.path.join(subdir_path, filename), os.path.join(labels_dir, filename))
                    elif filename.endswith((".jpg", ".png", ".jpeg")):
                        shutil.move(os.path.join(subdir_path, filename), os.path.join(images_dir, filename))
                    # Delete .xml files
                    elif filename.endswith(".xml"):
                        os.remove(os.path.join(subdir_path, filename))


        if __name__ == "__main__":
            directory = f"{work_dir}/trash_ICRA19/dataset"
            organize_files(directory)
        ```

Next, we need to modify the [dataset YAML](../datasets/detect/index.md#ultralytics-yolo-format) file. This is the setup we will use in our .yaml file. Class ID numbers start from 0:

```yaml
path: /path/to/dataset/directory # root directory for dataset
train: train/images # train images subdirectory
val: val/images # validation images subdirectory
test: test/images # test images subdirectory

# Classes
names:
    0: plastic
    1: bio
    2: rov
```

Run the following script to delete the current contents of `config.yaml` and replace it with the configuration that reflects our new dataset directory structure. The script automatically uses the `work_dir` variable we defined earlier, so be sure it points to your dataset before execution, and leave the train, val, and test subdirectory definitions unchanged.

!!! example "Edit the .yaml File"

    === "Python"

        ```python
        # Contents of new config.yaml file
        def update_yaml_file(file_path):
            data = {
                "path": f"{work_dir}/trash_ICRA19/dataset",
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {0: "plastic", 1: "bio", 2: "rov"},
            }

            # Ensures the "names" list appears after the sub/directories
            names_data = data.pop("names")
            with open(file_path, "w") as yaml_file:
                yaml.dump(data, yaml_file)
                yaml_file.write("\n")
                yaml.dump({"names": names_data}, yaml_file)


        if __name__ == "__main__":
            file_path = f"{work_dir}/trash_ICRA19/config.yaml"  # .yaml file path
            update_yaml_file(file_path)
            print(f"{file_path} updated successfully.")
        ```

### Step 5: Train the YOLO26 model

Run the following command to [fine-tune](https://www.ultralytics.com/glossary/fine-tuning) a pretrained [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) model.

!!! example "Train the YOLO26 model"

    === "CLI"

        ```bash
        !yolo detect train data={work_dir}/trash_ICRA19/config.yaml model=yolo26n.pt epochs=2 batch=32 plots=True
        ```

Here's a closer look at the parameters in the model training command:

- **detect**: The [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) task for which you are using the specified YOLO model and dataset.
- **train**: The mode, which denotes the purpose for which you are loading the specified model and data. Later, when we test our model's performance, we will use `predict`.
- **epochs**: The number of times YOLO26 will pass through the entire dataset.
- **batch**: The training [batch size](https://www.ultralytics.com/glossary/batch-size), which is the number of images the model processes before it updates its parameters.
- **plots**: Directs YOLO to generate and save plots of our model's training and evaluation metrics.

For a detailed understanding of the model training process and best practices, refer to the [YOLO26 Model Training guide](../modes/train.md). This guide will help you get the most out of your experiments and ensure you're using YOLO26 effectively.

### Step 6: Test the Model

We can now run inference to test the performance of our fine-tuned model:

!!! example "Test the YOLO26 model"

    === "CLI"

        ```bash
        !yolo detect predict source={work_dir}/trash_ICRA19/dataset/test/images model={work_dir}/runs/detect/train/weights/best.pt conf=0.5 iou=.5 save=True save_txt=True
        ```

This command generates predicted labels for each image in our test set, as well as new output image files that overlay the predicted [bounding box](https://www.ultralytics.com/glossary/bounding-box) atop the original image.

Predicted .txt labels for each image are saved via the `save_txt=True` argument and the output images with bounding box overlays are generated through the `save=True` argument.
The parameter `conf=0.5` tells the model to ignore all predictions with a confidence level of less than 50%.

Lastly, `iou=.5` sets the [non-maximum suppression (NMS)](https://www.ultralytics.com/glossary/non-maximum-suppression-nms) threshold: boxes of the same class that overlap a higher-scoring box by 50% or more are discarded. This reduces duplicate boxes generated for the same object.
We can load the images with predicted bounding box overlays to view how our model performs on a handful of images.

!!! example "Display Predictions"

    === "Python"

        ```python
        # Show the first ten images from the preceding prediction task
        for pred_dir in glob.glob(f"{work_dir}/runs/detect/predict/*.jpg")[:10]:
            img = Image.open(pred_dir)
            display(img)
        ```

The code above displays ten images from the test set with their predicted bounding boxes, accompanied by class name labels and confidence levels.

### Step 7: Evaluate the Model

We can produce visualizations of the model's [precision](https://www.ultralytics.com/glossary/precision) and recall for each class. These visualizations are saved in the training run directory (`{work_dir}/runs/detect/train/`). The precision score is displayed in `BoxP_curve.png`:

<p align="center">
  <img width="800" src="https://cdn.ul.run/i/fc3ef091a58db523dbbb7d7eb11f83fb.avif" alt="Model precision-confidence evaluation curve">
</p>

The graph shows precision rising as the confidence threshold for predictions increases. However, the model precision has not yet leveled out at a certain confidence level after two [epochs](https://www.ultralytics.com/glossary/epoch).

The [recall](https://www.ultralytics.com/glossary/recall) graph (`BoxR_curve.png`) displays an inverse trend:

<p align="center">
  <img width="800" src="https://cdn.ul.run/i/d7910d2e3174c447d49e9cff14de040a.avif" alt="Model recall-confidence evaluation curve">
</p>

Unlike precision, recall is highest at low confidence thresholds and falls as the threshold increases. This illustrates the trade-off between precision and recall.

### Step 8: Calculating Intersection Over Union

You can measure localization accuracy by calculating the [Intersection over Union (IoU)](https://www.ultralytics.com/glossary/intersection-over-union-iou) between a predicted bounding box and a ground truth bounding box for the same object. Check out [IBM's tutorial on training a YOLO object detection model](https://developer.ibm.com/tutorials/awb-train-yolo-object-detection-model-in-python/) for more details.

## Summary

We explored the key features of IBM Watsonx and how to train a YOLO26 model with it. IBM Watsonx also provides tools for model building, data management, and compliance.

For further details on usage, visit [IBM Watsonx official documentation](https://www.ibm.com/products/watsonx).

Also, check out the [Ultralytics integrations overview](index.md) to learn about other integrations.

## FAQ

### How do I train a YOLO26 model using IBM Watsonx?

To train a YOLO26 model using IBM Watsonx, follow these steps:

1. **Set Up Your Environment**: Create an IBM Cloud account and set up a Watsonx.ai project. Use a Jupyter Notebook for your coding environment.
2. **Install Libraries**: Install necessary libraries like `torch`, `opencv`, and `ultralytics`.
3. **Load Data**: Use the Kaggle API to load your dataset into Watsonx.
4. **Preprocess Data**: Organize your dataset into the required directory structure and update the `.yaml` configuration file.
5. **Train the Model**: Use the YOLO command-line interface to train your model with parameters such as `epochs` and `batch`.
6. **Test and Evaluate**: Run inference to test the model and evaluate its performance using metrics like precision and recall.

For detailed instructions, refer to our [YOLO26 Model Training guide](../modes/train.md).

### What are the key features of IBM Watsonx for AI model training?

IBM Watsonx offers several key features for AI model training:

- **Watsonx.ai**: Provides tools for AI development, including access to IBM-supported custom models and third-party models like Llama 3. It includes the Prompt Lab, Tuning Studio, and Flows Engine for comprehensive AI lifecycle management.
- **Watsonx.data**: Supports cloud and on-premises deployments, offering centralized data access, efficient query engines like Presto and Spark, and an AI-powered semantic layer.
- **Watsonx.governance**: Automates compliance, manages risk with alerts, and provides tools for detecting issues like bias and drift. It also includes dashboards and reporting tools for collaboration.

For more information, visit the [IBM Watsonx official documentation](https://www.ibm.com/products/watsonx).

### Why should I use IBM Watsonx for training Ultralytics YOLO26 models?

IBM Watsonx suits Ultralytics YOLO26 training because its tools cover the full AI lifecycle. Key benefits include:

- **Scalability**: Easily scale your model training with IBM Cloud services.
- **Integration**: Connect to various data sources and APIs.
- **User-Friendly Interface**: Simplifies the development process with a collaborative and intuitive interface.
- **Advanced Tools**: Access tools like the Prompt Lab, Tuning Studio, and Flows Engine for enhancing model performance.

Learn more about [Ultralytics YOLO26](../models/yolo26.md) and other training platforms in the [integrations overview](index.md).

### How can I preprocess my dataset for YOLO26 training on IBM Watsonx?

To preprocess your dataset for YOLO26 training on IBM Watsonx:

1. **Organize Directories**: Ensure your dataset follows the YOLO directory structure with separate subdirectories for images and labels within the train/val/test split.
2. **Update .yaml File**: Modify the `.yaml` configuration file to reflect the new directory structure and class names.
3. **Run Preprocessing Script**: Use a Python script to reorganize your dataset and update the `.yaml` file accordingly.

Here's a sample script to organize your dataset:

```python
import os
import shutil


def organize_files(directory):
    for subdir in ["train", "test", "val"]:
        subdir_path = os.path.join(directory, subdir)
        if not os.path.exists(subdir_path):
            continue

        images_dir = os.path.join(subdir_path, "images")
        labels_dir = os.path.join(subdir_path, "labels")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for filename in os.listdir(subdir_path):
            if filename.endswith(".txt"):
                shutil.move(os.path.join(subdir_path, filename), os.path.join(labels_dir, filename))
            elif filename.endswith((".jpg", ".png", ".jpeg")):
                shutil.move(os.path.join(subdir_path, filename), os.path.join(images_dir, filename))


if __name__ == "__main__":
    directory = f"{work_dir}/trash_ICRA19/dataset"
    organize_files(directory)
```

For more details, refer to our [data preprocessing guide](../guides/preprocessing-annotated-data.md).

### What are the prerequisites for training a YOLO26 model on IBM Watsonx?

Before you start training a YOLO26 model on IBM Watsonx, ensure you have the following prerequisites:

- **IBM Cloud Account**: Create an account on IBM Cloud to access Watsonx.ai.
- **Kaggle Account**: For loading datasets, you'll need a Kaggle account and an API key.
- **Jupyter Notebook**: Set up a Jupyter Notebook environment within Watsonx.ai for coding and model training.

For more information on setting up your environment, visit our [Ultralytics Installation guide](../quickstart.md).
