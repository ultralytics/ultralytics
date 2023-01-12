# Ultralytics HUB

<div align="center">
  <a href="https://hub.ultralytics.com" target="_blank">
    <img width="1024" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-hub.png"></a>
<br>
  <a href="https://github.com/ultralytics/hub/actions/workflows/ci.yaml">
    <img src="https://github.com/ultralytics/hub/actions/workflows/ci.yaml/badge.svg" alt="CI CPU"></a>
</div>



[Ultralytics HUB](https://hub.ultralytics.com) is a new no-code online tool developed
by [Ultralytics](https://ultralytics.com), the creators of the popular [YOLOv5](https://github.com/ultralytics/yolov5)
object detection and image segmentation models. With Ultralytics HUB, users can easily train and deploy YOLOv5 models
without any coding or technical expertise.

Ultralytics HUB is designed to be user-friendly and intuitive, with a drag-and-drop interface that allows users to
easily upload their data and select their model configurations. It also offers a range of pre-trained models and
templates to choose from, making it easy for users to get started with training their own models. Once a model is
trained, it can be easily deployed and used for real-time object detection and image segmentation tasks. Overall,
Ultralytics HUB is an essential tool for anyone looking to use YOLOv5 for their object detection and image segmentation
projects.

**[Get started now](https://hub.ultralytics.com)** and experience the power and simplicity of Ultralytics HUB for
yourself. Sign up for a free account and
start building, training, and deploying YOLOv5 and YOLOv8 models today.

## 1. Upload a Dataset

Ultralytics HUB datasets are just like YOLOv5 🚀 datasets, they use the same structure and the same label formats to keep
everything simple.

When you upload a dataset to Ultralytics HUB, make sure to **place your dataset YAML inside the dataset root directory**
as in the example shown below, and then zip for upload to https://hub.ultralytics.com/. Your **dataset YAML, directory
and zip** should all share the same name. For example, if your dataset is called 'coco6' as in our
example [ultralytics/hub/coco6.zip](https://github.com/ultralytics/hub/blob/master/coco6.zip), then you should have a
coco6.yaml inside your coco6/ directory, which should zip to create coco6.zip for upload:

```bash
zip -r coco6.zip coco6
```

The example [coco6.zip](https://github.com/ultralytics/hub/blob/master/coco6.zip) dataset in this repository can be
downloaded and unzipped to see exactly how to structure your custom dataset.

<p align="center"><img width="80%" src="https://user-images.githubusercontent.com/26833433/201424843-20fa081b-ad4b-4d6c-a095-e810775908d8.png" title="COCO6" /></p>

The dataset YAML is the same standard YOLOv5 YAML format. See
the [YOLOv5 Train Custom Data tutorial](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) for full details.

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path:  # dataset root dir (leave empty for HUB)
train: images/train  # train images (relative to 'path') 8 images
val: images/val  # val images (relative to 'path') 8 images
test:  # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  ...
```

After zipping your dataset, sign in to [Ultralytics HUB](https://bit.ly/ultralytics_hub) and click the Datasets tab.
Click 'Upload Dataset' to upload, scan and visualize your new dataset before training new YOLOv5 models on it!

<img width="100%" alt="HUB Dataset Upload" src="https://user-images.githubusercontent.com/26833433/198611715-540c9856-49d7-4069-a2fd-7c9eb70e772e.png">

## 2. Train a Model

Connect to the Ultralytics HUB notebook and use your model API key to begin
training! <a href="https://colab.research.google.com/github/ultralytics/hub/blob/master/hub.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## 3. Deploy to Real World

Export your model to 13 different formats, including TensorFlow, ONNX, OpenVINO, CoreML, Paddle and many others. Run
models directly on your mobile device by downloading the [Ultralytics App](https://ultralytics.com/app_install)!

<a align="center" href="https://ultralytics.com/app_install" target="_blank">
<img width="100%" alt="Ultralytics mobile app" src="https://github.com/ultralytics/assets/raw/main/im/ultralytics-app.png"></a>

## ❓ Issues

If you are a new [Ultralytics HUB](https://bit.ly/ultralytics_hub) user and have questions or comments, you are in the
right place! Please raise a [New Issue](https://github.com/ultralytics/hub/issues/new/choose) and let us know what we
can do to make your life better 😃!
