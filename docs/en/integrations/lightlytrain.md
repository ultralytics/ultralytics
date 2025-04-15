---
comments: true
description: Learn how to leverage distillation and self-supervised pretraining with LightlyTrain to leverage all your unlabeled data.
keywords: LightlyTrain, YOLO11, Ultralytics, machine learning, model training, data science, computer vision, self-supervised learning, distillation, DINOv2, DINO, object detection
---

# Unlock the Power of Unlabeled Data with Lightly**Train** and YOLO

Ever wondered what to do with all those unlabeled images in your dataset? Great news! [Lightly**Train**](https://github.com/lightly-ai/lightly-train) helps you make the most of every single image - labeled or not.⚡️

In this hands-on tutorial, we'll show you how to supercharge your object detector by combining Lightly**Train**'s pretraining capabilities with Ultralytics' YOLO. We'll work with the classic [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) to demonstrate both pretraining (without labels) and fine-tuning (with labels).

You can also run this tutorial in [Google Colab](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/ultralytics_yolo.ipynb) directly.

## Install Dependencies

Let's install the required dependencies:

- `lightly-train` for pretraining, with support for `ultralytics`' YOLO models
- [`supervision`](https://github.com/roboflow/supervision) to visualize some of the annotated pictures

```bash
pip install "lightly-train[ultralytics]" "supervision==0.25.1"
```

## Download the Dataset

We can download the dataset directly using Ultralytics' API with the `check_det_dataset` function:

```python
from ultralytics.data.utils import check_det_dataset

dataset = check_det_dataset("VOC.yaml")
```

Ultralytics always downloads your datasets to a fixed location, which you can fetch via their `settings` module:

```python
from ultralytics import settings

print(settings["datasets_dir"])
```

Inside that directory (`<DATASET-DIR>`), you will now have the following structure of images and labels:

```bash
tree -d <DATASET-DIR>/VOC -I VOCdevkit

>    datasets/VOC
>    ├── images
>    │   ├── test2007
>    │   ├── train2007
>    │   ├── train2012
>    │   └── val2007
>    │   └── val2012
>    └── labels
>        ├── test2007
>        ├── train2007
>        ├── train2012
>        ├── val2007
>        └── val2012
```

!!! note

    The labels are not required for pre-training. We will use the labels only for fine-tuning.

## Get to Know Your Data

Let's take a peek at what we're working with! We'll use `supervision` to visualize some samples from our dataset:

```python
import random

import matplotlib.pyplot as plt
import supervision as sv
import yaml
from ultralytics import settings
from ultralytics.data.utils import check_det_dataset

dataset = check_det_dataset("VOC.yaml")

detections = sv.DetectionDataset.from_yolo(
    data_yaml_path=dataset["yaml_file"],
    images_directory_path=f"{settings['datasets_dir']}/VOC/images/train2012",
    annotations_directory_path=f"{settings['datasets_dir']}/VOC/labels/train2012",
)

with open(dataset["yaml_file"], "r") as f:
    data = yaml.safe_load(f)

names = data["names"]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()

detections = [detections[random.randint(0, len(detections))] for _ in range(4)]

for i, (path, image, annotation) in enumerate(detections):
    annotated_image = box_annotator.annotate(scene=image, detections=annotation)
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=annotation,
        labels=[names[elem] for elem in annotation.class_id],
    )
    ax[i].imshow(annotated_image[..., ::-1])
    ax[i].axis("off")

fig.tight_layout()
fig.show()
```

![VOC2012 Training Samples](https://github.com/lightly-ai/lightly-train/blob/main/docs/source/tutorials/yolo/samples_VOC_train2012.png)

## Pre-training and Fine-tuning

Now comes the exciting part! We'll take a randomly initialized YOLO11 model and transform it into a powerful object detector through:
1. Smart pretraining with unlabeled data using Lightly**Train**'s distillation techniques
2. Focused fine-tuning to object detection, using labeled data together with Ultralytics

Here's how to work this magic:

=== "Python"

    ```python
    # pretrain_yolo.py
    import lightly_train
    from ultralytics import settings

    data_path = f"{settings["datasets_dir"]}/VOC/images/train2012"

    if __name__ == "__main__":
        # Pre-train with lightly-train.
        lightly_train.train(
            out="out/my_experiment",            # Output directory.
            model="ultralytics/yolo11s.yaml",   # Pass the YOLO model.
            data=data_path,                     # Path to a directory with training images.
            epochs=100,                         # Adjust epochs for faster training.
            batch_size=64,                      # Adjust batch size based on hardware.
        )
    ```

    ```python
    # finetune_yolo.py

    from ultralytics import YOLO

    if __name__ == "__main__":
        # Load the exported model.
        model = YOLO("out/my_experiment/exported_models/exported_last.pt")

        # Fine-tune with ultralytics.
        model.train(data="VOC.yaml", epochs=100)
    ```

=== "Command Line"

    ```bash
    lightly-train train out="out/my_experiment" data="<DATASET-DIR>/VOC/images/train2012" model="ultralytics/yolo11s.yaml" epochs=100 batch_size=64
    ```

    ```bash
    yolo detect train model="out/my_experiment/exported_models/exported_last.pt" data="VOC.yaml" epochs=100
    ```

Congratulations!🥳 You have successfully pre-trained a model using `lightly-train` and fine-tuned it for object detection using `ultralytics`.

For more advanced options, explore Lightly**Train**'s [Python API](https://docs.lightly.ai/train/stable/python_api/index.html) and [Ultralytics documentation](https://docs.ultralytics.com).

## Ready for More?

Now that you've got the basics down, here are some cool ways to level up your model:

- Dive into the world of self-supervised learning with [DINO and SimCLR in `lightly-train`](https://docs.lightly.ai/train/stable/methods/index.html)
- Experiment with different YOLO flavors (`YOLOv5`, `YOLOv6`, `YOLOv8`)
- Take your pre-trained model for a spin with [image embeddings and similarity search](https://docs.lightly.ai/train/stable/embed.html)

Happy experimenting! 🚀