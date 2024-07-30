---
comments: true
description: Dive into our detailed integration guide on using IBM Watson to train a YOLOv8 model. Uncover key features and step-by-step instructions on model training.
keywords: IBM Watsonx, IBM Watsonx AI, What is Watson?, IBM Watson Integration, IBM Watson Features, YOLOv8, Ultralytics, Model Training, GPU, TPU, cloud computing
---

# A Step-by-Step Guide to Training YOLOv8 Models with IBM Watsonx

Nowadays, scalable [computer vision solutions](../guides/steps-of-a-cv-project.md) are becoming more common and transforming the way we handle visual data. A great example is IBM Watsonx, an advanced AI and data platform that simplifies the development, deployment, and management of AI models. It offers a complete suite for the entire AI lifecycle and seamless integration with IBM Cloud services.

You can train [Ultralytics YOLOv8 models](https://github.com/ultralytics/ultralytics) using IBM Watsonx. It's a good option for enterprises interested in efficient [model training](../modes/train.md), fine-tuning for specific tasks, and improving [model performance](../guides/model-evaluation-insights.md) with robust tools and a user-friendly setup. In this guide, we'll walk you through the process of training YOLOv8 with IBM Watsonx, covering everything from setting up your environment to evaluating your trained models. Let's get started!

## What is IBM Watsonx?

[Watsonx](https://www.ibm.com/watsonx) is IBM's cloud-based platform designed for commercial generative AI and scientific data. IBM Watsonx's three components - watsonx.ai, watsonx.data, and watsonx.governance - come together to create an end-to-end, trustworthy AI platform that can accelerate AI projects aimed at solving business problems. It provides powerful tools for building, training, and [deploying machine learning models](../guides/model-deployment-options.md) and makes it easy to connect with various data sources.

<p align="center">
  <img width="800" src="https://cdn.stackoverflow.co/images/jo7n4k8s/production/48b67e6aec41f89031a3426cbd1f78322e6776cb-8800x4950.jpg?auto=format" alt="Overview of IBM Watsonx">
</p>

Its user-friendly interface and collaborative capabilities streamline the development process and help with efficient model management and deployment. Whether for computer vision, predictive analytics, natural language processing, or other AI applications, IBM Watsonx provides the tools and support needed to drive innovation.

## Key Features of IBM Watsonx

IBM Watsonx is made of three main components: watsonx.ai, watsonx.data, and watsonx.governance. Each component offers features that cater to different aspects of AI and data management. Let's take a closer look at them.

### [Watsonx.ai](https://www.ibm.com/products/watsonx-ai)

Watsonx.ai provides powerful tools for AI development and offers access to IBM-supported custom models, third-party models like [Llama 3](https://www.ultralytics.com/blog/getting-to-know-metas-llama-3), and IBM's own Granite models. It includes the Prompt Lab for experimenting with AI prompts, the Tuning Studio for improving model performance with labeled data, and the Flows Engine for simplifying generative AI application development. Also, it offers comprehensive tools for automating the AI model lifecycle and connecting to various APIs and libraries.

### [Watsonx.data](https://www.ibm.com/products/watsonx-data)

Watsonx.data supports both cloud and on-premises deployments through the IBM Storage Fusion HCI integration. Its user-friendly console provides centralized access to data across environments and makes data exploration easy with common SQL. It optimizes workloads with efficient query engines like Presto and Spark, accelerates data insights with an AI-powered semantic layer, includes a vector database for AI relevance, and supports open data formats for easy sharing of analytics and AI data.

### [Watsonx.governance](https://www.ibm.com/products/watsonx-governance)

Watsonx.governance makes compliance easier by automatically identifying regulatory changes and enforcing policies. It links requirements to internal risk data and provides up-to-date AI factsheets. The platform helps manage risk with alerts and tools to detect issues such as [bias and drift](../guides/model-monitoring-and-maintenance.md). It also automates the monitoring and documentation of the AI lifecycle, organizes AI development with a model inventory, and enhances collaboration with user-friendly dashboards and reporting tools.

## How to Train YOLOv8 Using IBM Watsonx

You can use IBM Watsonx to accelerate your YOLOv8 model training workflow.

### Prerequisites

You need an [IBM Cloud account](https://cloud.ibm.com/registration) to create a [watsonx.ai](https://www.ibm.com/products/watsonx-ai) project, and you'll also need a [Kaggle](./kaggle.md) account to load the data set.

### Step 1: Set Up Your Environment

First, you'll need to set up an IBM account to use a Jupyter Notebook. Log in to [watsonx.ai](https://eu-de.dataplatform.cloud.ibm.com/registration/stepone?preselect_region=true) using your IBM Cloud account.

Then, create a [watsonx.ai project](https://www.ibm.com/docs/en/watsonx/saas?topic=projects-creating-project), and a [Jupyter Notebook](https://www.ibm.com/docs/en/watsonx/saas?topic=editor-creating-managing-notebooks).

Once you do so, a notebook environment will open for you to load your data set. You can use the code from this tutorial to tackle a simple object detection model training task.

### Step 2: Install and Import Relevant Libraries

Next, you can install and import the necessary Python libraries.

!!! Tip "Installation"

    === "CLI"

        ```bash
        # Install the required packages
        pip install torch torchvision torchaudio
        pip install opencv-contrib-python-headless
        pip install ultralytics==8.0.196
        ```

For detailed instructions and best practices related to the installation process, check our [Ultralytics Installation guide](../quickstart.md). While installing the required packages for YOLOv8, if you encounter any difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md) for solutions and tips.

Then, you can import the needed packages.

!!! Example "Import Relevant Libraries"

    === "Python"

        ```python
        # Import ultralytics
        import ultralytics

        ultralytics.checks()

        # Import packages to retrieve and display image files
        ```

### Step 3: Load the Data

For this tutorial, we will use a [marine litter dataset](https://www.kaggle.com/datasets/atiqishrak/trash-dataset-icra19) available on Kaggle. With this dataset, we will custom-train a YOLOv8 model to detect and classify litter and biological objects in underwater images.

We can load the dataset directly into the notebook using the Kaggle API. First, create a free Kaggle account. Once you have created an account, you'll need to generate an API key. Directions for generating your key can be found in the [Kaggle API documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md) under the section "API credentials".

Copy and paste your Kaggle username and API key into the following code. Then run the code to install the API and load the dataset into Watsonx.

!!! Tip "Installation"

    === "CLI"

        ```bash
        # Install kaggle
        pip install kaggle
        ```

After installing Kaggle, we can load the dataset into Watsonx.

!!! Example "Load the Data"

    === "Python"

        ```python
        # Replace "username" string with your username
        os.environ["KAGGLE_USERNAME"] = "username"
        # Replace "apiKey" string with your key
        os.environ["KAGGLE_KEY"] = "apiKey"

        # Load dataset
        !kaggle datasets download atiqishrak/trash-dataset-icra19 --unzip

        # Store working directory path as work_dir
        work_dir = os.getcwd()

        # Print work_dir path
        print(os.getcwd())

        # Print work_dir contents
        print(os.listdir(f"{work_dir}"))

        # Print trash_ICRA19 subdirectory contents
        print(os.listdir(f"{work_dir}/trash_ICRA19"))
        ```

After loading the dataset, we printed and saved our working directory. We have also printed the contents of our working directory to confirm the "trash_ICRA19" data set was loaded properly.

If you see "trash_ICRA19" among the directory's contents, then it has loaded successfully. You should see three files/folders: a `config.yaml` file, a `videos_for_testing` directory, and a `dataset` directory. We will ignore the `videos_for_testing` directory, so feel free to delete it.

We will use the config.yaml file and the contents of the dataset directory to train our object detection model. Here is a sample image from our marine litter data set.

<p align="center">
  <img width="400" src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/sQy6asArOJ2weUuQ_POiVA.jpg" alt="Marine Litter with Bounding Box">
</p>

### Step 4: Preprocess the Data

Fortunately, all labels in the marine litter data set are already formatted as YOLO .txt files. However, we need to rearrange the structure of the image and label directories in order to help our model process the image and labels. Right now, our loaded data set directory follows this structure:

<p align="center">
  <img width="400" src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VfgvRT7vdgkeTQNqVMs_CQ.png" alt="Loaded Dataset Directory">
</p>

But, YOLO models by default require separate images and labels in subdirectories within the train/val/test split. We need to reorganize the directory into the following structure:

<p align="center">
  <img width="400" src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/uUk1YopS94mytGaCav3ZaQ.png" alt="Yolo Directory Structure">
</p>

To reorganize the data set directory, we can run the following script:

!!! Example "Preprocess the Data"

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
                    elif filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                        shutil.move(os.path.join(subdir_path, filename), os.path.join(images_dir, filename))
                    # Delete .xml files
                    elif filename.endswith(".xml"):
                        os.remove(os.path.join(subdir_path, filename))


        if __name__ == "__main__":
            directory = f"{work_dir}/trash_ICRA19/dataset"
            organize_files(directory)
        ```

Next, we need to modify the .yaml file for the data set. This is the setup we will use in our .yaml file. Class ID numbers start from 0:

```yaml
path: /path/to/dataset/directory # root directory for dataset
train: train/images # train images subdirectory
val: train/images # validation images subdirectory
test: test/images # test images subdirectory

# Classes
names:
    0: plastic
    1: bio
    2: rov
```

Run the following script to delete the current contents of config.yaml and replace it with the above contents that reflect our new data set directory structure. Be certain to replace the work_dir portion of the root directory path in line 4 with your own working directory path we retrieved earlier. Leave the train, val, and test subdirectory definitions. Also, do not change {work_dir} in line 23 of the code.

!!! Example "Edit the .yaml File"

    === "Python"

        ```python
        # Contents of new confg.yaml file
        def update_yaml_file(file_path):
            data = {
                "path": "work_dir/trash_ICRA19/dataset",
                "train": "train/images",
                "val": "train/images",
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

### Step 5: Train the YOLOv8 model

Run the following command-line code to fine tune a pretrained default YOLOv8 model.

!!! Example "Train the YOLOv8 model"

    === "CLI"

        ```bash
        !yolo task=detect mode=train data={work_dir}/trash_ICRA19/config.yaml model=yolov8s.pt epochs=2 batch=32 lr0=.04 plots=True
        ```

Here's a closer look at the parameters in the model training command:

- **task**: It specifies the computer vision task for which you are using the specified YOLO model and data set.
- **mode**: Denotes the purpose for which you are loading the specified model and data. Since we are training a model, it is set to "train." Later, when we test our model's performance, we will set it to "predict."
- **epochs**: This delimits the number of times YOLOv8 will pass through our entire data set.
- **batch**: The numerical value stipulates the training batch sizes. Batches are the number of images a model processes before it updates its parameters.
- **lr0**: Specifies the model's initial learning rate.
- **plots**: Directs YOLO to generate and save plots of our model's training and evaluation metrics.

For a detailed understanding of the model training process and best practices, refer to the [YOLOv8 Model Training guide](../modes/train.md). This guide will help you get the most out of your experiments and ensure you're using YOLOv8 effectively.

### Step 6: Test the Model

We can now run inference to test the performance of our fine-tuned model:

!!! Example "Test the YOLOv8 model"

    === "CLI"

        ```bash
        !yolo task=detect mode=predict source={work_dir}/trash_ICRA19/dataset/test/images model={work_dir}/runs/detect/train/weights/best.pt conf=0.5 iou=.5 save=True save_txt=True
        ```

This brief script generates predicted labels for each image in our test set, as well as new output image files that overlay the predicted bounding box atop the original image.

Predicted .txt labels for each image are saved via the `save_txt=True` argument and the output images with bounding box overlays are generated through the `save=True` argument.  
The parameter `conf=0.5` informs the model to ignore all predictions with a confidence level of less than 50%.

Lastly, `iou=.5` directs the model to ignore boxes in the same class with an overlap of 50% or greater. It helps to reduce potential duplicate boxes generated for the same object.  
we can load the images with predicted bounding box overlays to view how our model performs on a handful of images.

!!! Example "Display Predictions"

    === "Python"

        ```python
        # Show the first ten images from the preceding prediction task
        for pred_dir in glob.glob(f"{work_dir}/runs/detect/predict/*.jpg")[:10]:
            img = Image.open(pred_dir)
            display(img)
        ```

The code above displays ten images from the test set with their predicted bounding boxes, accompanied by class name labels and confidence levels.

### Step 7: Evaluate the Model

We can produce visualizations of the model's precision and recall for each class. These visualizations are saved in the home directory, under the train folder. The precision score is displayed in the P_curve.png:

<p align="center">
  <img width="800" src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/EvQpqt4D6VI2And1T86Fww.png" alt="Precision Confidence Curve">
</p>

The graph shows an exponential increase in precision as the model's confidence level for predictions increases. However, the model precision has not yet leveled out at a certain confidence level after two epochs.

The recall graph (R_curve.png) displays an inverse trend:

<p align="center">
  <img width="800" src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/NS0pQDHuEWM-WlpBpxTydw.png" alt="Recall Confidence Curve">
</p>

Unlike precision, recall moves in the opposite direction, showing greater recall with lower confidence instances and lower recall with higher confidence instances. This is an apt example of the trade-off in precision and recall for classification models.

### Step 8: Calculating Intersection Over Union

You can measure the prediction accuracy by calculating the IoU between a predicted bounding box and a ground truth bounding box for the same object. Check out [IBM's tutorial on training YOLOv8](https://developer.ibm.com/tutorials/awb-train-yolo-object-detection-model-in-python/) for more details.

## Summary

We explored IBM Watsonx key features, and how to train a YOLOv8 model using IBM Watsonx. We also saw how IBM Watsonx can enhance your AI workflows with advanced tools for model building, data management, and compliance.

For further details on usage, visit [IBM Watsonx official documentation](https://www.ibm.com/watsonx).

Also, be sure to check out the [Ultralytics integration guide page](./index.md), to learn more about different exciting integrations.

## FAQ

### How do I train a YOLOv8 model using IBM Watsonx?

To train a YOLOv8 model using IBM Watsonx, follow these steps:

1. **Set Up Your Environment**: Create an IBM Cloud account and set up a Watsonx.ai project. Use a Jupyter Notebook for your coding environment.
2. **Install Libraries**: Install necessary libraries like `torch`, `opencv`, and `ultralytics`.
3. **Load Data**: Use the Kaggle API to load your dataset into Watsonx.
4. **Preprocess Data**: Organize your dataset into the required directory structure and update the `.yaml` configuration file.
5. **Train the Model**: Use the YOLO command-line interface to train your model with specific parameters like `epochs`, `batch size`, and `learning rate`.
6. **Test and Evaluate**: Run inference to test the model and evaluate its performance using metrics like precision and recall.

For detailed instructions, refer to our [YOLOv8 Model Training guide](../modes/train.md).

### What are the key features of IBM Watsonx for AI model training?

IBM Watsonx offers several key features for AI model training:

- **Watsonx.ai**: Provides tools for AI development, including access to IBM-supported custom models and third-party models like Llama 3. It includes the Prompt Lab, Tuning Studio, and Flows Engine for comprehensive AI lifecycle management.
- **Watsonx.data**: Supports cloud and on-premises deployments, offering centralized data access, efficient query engines like Presto and Spark, and an AI-powered semantic layer.
- **Watsonx.governance**: Automates compliance, manages risk with alerts, and provides tools for detecting issues like bias and drift. It also includes dashboards and reporting tools for collaboration.

For more information, visit the [IBM Watsonx official documentation](https://www.ibm.com/watsonx).

### Why should I use IBM Watsonx for training Ultralytics YOLOv8 models?

IBM Watsonx is an excellent choice for training Ultralytics YOLOv8 models due to its comprehensive suite of tools that streamline the AI lifecycle. Key benefits include:

- **Scalability**: Easily scale your model training with IBM Cloud services.
- **Integration**: Seamlessly integrate with various data sources and APIs.
- **User-Friendly Interface**: Simplifies the development process with a collaborative and intuitive interface.
- **Advanced Tools**: Access to powerful tools like the Prompt Lab, Tuning Studio, and Flows Engine for enhancing model performance.

Learn more about [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and how to train models using IBM Watsonx in our [integration guide](./index.md).

### How can I preprocess my dataset for YOLOv8 training on IBM Watsonx?

To preprocess your dataset for YOLOv8 training on IBM Watsonx:

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
            elif filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                shutil.move(os.path.join(subdir_path, filename), os.path.join(images_dir, filename))


if __name__ == "__main__":
    directory = f"{work_dir}/trash_ICRA19/dataset"
    organize_files(directory)
```

For more details, refer to our [data preprocessing guide](../guides/preprocessing_annotated_data.md).

### What are the prerequisites for training a YOLOv8 model on IBM Watsonx?

Before you start training a YOLOv8 model on IBM Watsonx, ensure you have the following prerequisites:

- **IBM Cloud Account**: Create an account on IBM Cloud to access Watsonx.ai.
- **Kaggle Account**: For loading datasets, you'll need a Kaggle account and an API key.
- **Jupyter Notebook**: Set up a Jupyter Notebook environment within Watsonx.ai for coding and model training.

For more information on setting up your environment, visit our [Ultralytics Installation guide](../quickstart.md).
