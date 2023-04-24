# Roboflow Datasets

You can now use Roboflow to organize, label, prepare, version, and host your datasets for training YOLOv5 🚀 models. Roboflow is free to use with YOLOv5 if you make your workspace public.  
UPDATED 30 September 2021.

## Upload
You can upload your data to Roboflow via [web UI](https://docs.roboflow.com/adding-data), [rest API](https://docs.roboflow.com/adding-data/upload-api), or [python](https://docs.roboflow.com/python).

## Labeling
After uploading data to Roboflow, you can label your data and review previous labels.

[![Roboflow Annotate](https://roboflow-darknet.s3.us-east-2.amazonaws.com/roboflow-annotate.gif)](https://roboflow.com/annotate)

## Versioning
You can make versions of your dataset with different preprocessing and offline augmentation options. YOLOv5 does online augmentations natively, so be intentional when layering Roboflow's offline augs on top.

![Roboflow Preprocessing](https://roboflow-darknet.s3.us-east-2.amazonaws.com/robolfow-preprocessing.png)

## Exporting Data
You can download your data in YOLOv5 format to quickly begin training.

```
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR API KEY HERE")
project = rf.workspace().project("YOUR PROJECT")
dataset = project.version("YOUR VERSION").download("yolov5")
```

## Custom Training
We have released a custom training tutorial demonstrating all of the above capabilities. You can access the code here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/yolov5-custom-training-tutorial/blob/main/yolov5-custom-training.ipynb)

## Active Learning
The real world is messy and your model will invariably encounter situations your dataset didn't anticipate. Using [active learning](https://blog.roboflow.com/what-is-active-learning/) is an important strategy to iteratively improve your dataset and model. With the Roboflow and YOLOv5 integration, you can quickly make improvements on your model deployments by using a battle tested machine learning pipeline.

<p align=""><a href="https://roboflow.com/?ref=ultralytics"><img width="1000" src="https://uploads-ssl.webflow.com/5f6bc60e665f54545a1e52a5/615627e5824c9c6195abfda9_computer-vision-cycle.png"/></a></p>
