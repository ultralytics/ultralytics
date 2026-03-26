<div align="center">
  <p>
    <a href="https://platform.ultralytics.com/?utm_source=github&utm_medium=referral&utm_campaign=platform_launch&utm_content=banner&utm_term=ultralytics_github" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
  </p>

[中文](https://docs.ultralytics.com/zh/) | [한국어](https://docs.ultralytics.com/ko/) | [日本語](https://docs.ultralytics.com/ja/) | [Русский](https://docs.ultralytics.com/ru/) | [Deutsch](https://docs.ultralytics.com/de/) | [Français](https://docs.ultralytics.com/fr/) | [Español](https://docs.ultralytics.com/es) | [Português](https://docs.ultralytics.com/pt/) | [Türkçe](https://docs.ultralytics.com/tr/) | [Tiếng Việt](https://docs.ultralytics.com/vi/) | [العربية](https://docs.ultralytics.com/ar/)

</div>

# EV-Ultralytics: An event-based object detector based on YOLO

This repository is a fork of [the ultralytics repo](https://github.com/ultralytics/ultralytics) with minimal modifications required to train and evaluate a YOLO object detector on event-based histograms in different file formats (HDF5, npz, npy).

## Introduction

The goal of this repo is to serve as a tutorial for training and evalutaing an event-based object detection model based on the YOLO architecture and training code, with minimal modifications. The models used by [ultralytics](https://www.ultralytics.com/) are for classical computer vision, meaning they are frame-based and use RGB images as inputs.  Processed event frames, such as event histograms for example, encode similar visual information as RGB images and can be used as their substitutions for the input of the networks.

In this tutorial, we use yolo26 as an example to show how to train and evaluate object detection models with event histograms instead of RGB images. The main differences between the properties of event histograms and RGB images are:

- An event histogram has 2 channels (for ON and OFF polarities) while a RGB image has 3 channels.
- Event histograms are often continuous and stored together in either an .npy, .npz or .h5 (which is what we use in this tutorial) file when they are converted from an events stream. RGB images, by contrast, are separately stored and often irrelevant to each other.

The modification of the source code mainly targets these two issues.

### Example Result

Here is an example output of a trained model following this scheme:
![Example predictions on event histograms](ultralytics/assets/prediction_test_data.gif)

## Train or test a model

If you want to train your own model, or test an already-trained model, you first need to acquire some data. This event data needs to be precomputed from raw event data to be histogram data. Here are the steps to acquire such data.

### Acquire an Event-based Object Detection Dataset

You now have two choices to acquire an event-based object detection dataset:

1. You can download the example dataset we provide and follow the instructions to train and test a detection model with it. The dataset is already in the correct format for training and testing.
2. You can prepare your own dataset with your recordings of an event camera. This is outside the scope of this tutorial however.

#### Download the Example dataset

We provide an example dataset for you to quickly test the training and evaluation pipeline. You can
download it from this [Hugging Face repo](https://huggingface.co/datasets/prophesee-ai/gen4-automotive-histos):

```bash
hf download prophesee-ai/gen4-automotive-histos --repo-type dataset --local-dir <YOUR_LOCAL_DIR>
```

This dataset is recorded with a Prophesee Gen4 event camera and is an automotive dataset.
It contains 6 classes:

- pedestrian
- two-wheeler
- car
- truck
- bus
- traffic light
- traffic sign

The sequences have already been converted to event histograms and saved in the [HDF5 tensor format](https://docs.prophesee.ai/stable/data/file_formats/hdf5_tensor.html) and the labels are in the correct `.npy` format.

#### Acquire your own dataset (outside of scope of this tutorial)

If you wish to acquire your own event-based dataset or get help to devise new models geared toward event-based vision, please [contact us](support@prophesee.ai) or check [this page](https://www.prophesee.ai/metavision-sdk-pro/) for further information.

### Training and Testing the model

Now that you have your data, you can train your own model, or test an already existing one.

#### Prepare your environment

To get started, clone the repository and prepare your Python Environment:

```bash
uv sync &&  source .venv/bin/activate
# or (recommended to use pip in a virtual environment)
pip install .
```

Then you need to update the config file to add your path:

In `scripts/cfg/config.yaml` change `<path_to_histos_directory>` to your own path that leads to the base folder containing train and val directories. You can further modify this file according to your own dataset.

#### Training

To train simply run:

```bash
python scripts/train.py
```

The network will be trained for only 10 epochs. Increase the number if you want to train more. The trained model will be saved in the `runs/detect/train/weights` folder by default. You can also find stats on your training as well as images with predictions and labels in the `runs/detect/train` folder.

You can change the saving path and other training parameters by referring to the [ultralytics documentation](https://docs.ultralytics.com/modes/train/). For example, for a bigger model you can rename `yolo26n.yaml` to `yolo26m.yaml` or `yolo26l.yaml`. You can also change the batch size, learning rate, etc. in the `model.train()` function (arguments are found in the [table here](https://docs.ultralytics.com/modes/train/#musgd-optimizer>))

#### Make Predictions with a Trained Detection Model

If you wish to test our already trained model, it can be found on [HuggingFace](https://huggingface.co/prophesee-ai/ev-yolo-detector). You can download it using Hugging Face's CLI tool:

```bash
hf download prophesee-ai/ev-yolo-detector weights/ev-yolo26.pt --local-dir <YOUR_LOCAL_DIR>
```

You can now run your trained model on a test data sequence. To do so, first update the `scripts/test.py` script:

- Change `YOUR_TRAINED_MODEL.pt` to the path to your model weights.
- Change `YOUR_TEST_EVENTS_FILE.h5` to the path to your test sequence.
- Change `OUTPUT_FOLDER` to the directory you would like the output video to be saved.

Then simply run:

```bash
python scripts/test.py
```

The output will be a video with the detections made by your trained model.

## Contact

This repository is meant to serve as an example of event-based object detection, and not to be used for commercial purposes. For further information on event-based technology and its possibilities, please contact [Prophesee](https://www.prophesee.ai/) at <support@prophesee.ai>.
