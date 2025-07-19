---
comments: true
description: Learn how to leverage distillation and self-supervised pretraining with LightlyTrain to leverage all your unlabeled data.
keywords: LightlyTrain, YOLO11, Ultralytics, machine learning, model training, data science, computer vision, self-supervised learning, distillation, DINOv2, DINO, object detection
---

# Unlock the Power of Unlabeled Data with Lightly**Train** and YOLO

Ever wondered what to do with all those unlabeled images in your dataset? Great news! [Lightly**Train**](https://github.com/lightly-ai/lightly-train) helps you make the most of every single image - labeled or not. ⚡️

In this step-by-step guide, we'll show you how to:

1. Pretrain a YOLO model on unlabeled COCO images using Lightly**Train**'s distillation techniques
2. Fine-tune it for object detection on PASCAL VOC using Ultralytics

You can also run this tutorial in directly in [Google Colab](https://colab.research.google.com/github/lightly-ai/lightly-train/blob/main/examples/notebooks/ultralytics_yolo.ipynb).

## Install Dependencies

Let's install the required dependencies:

- `lightly-train` for pretraining, with support for `ultralytics`' YOLO models
- [`supervision`](https://github.com/roboflow/supervision) to visualize some of the annotated pictures

```bash
pip install "lightly-train[ultralytics]" "supervision==0.25.1"
```

## Pretraining YOLO with Lightly**Train**

### Getting our Unlabeled Data

Let's start by downloading a subset of COCO (25k images) that we'll use for pretraining:

```bash
# Download COCO-minitrain
wget https://huggingface.co/datasets/bryanbocao/coco_minitrain/resolve/main/coco_minitrain_25k.zip

# Unzip it
unzip coco_minitrain_25k.zip

# Remove labels (we don't need them for pretraining!)
rm -rf coco_minitrain_25k/labels
```

### Pretraining the YOLO Model

Now for the exciting part - pretraining! With Lightly**Train**, it's as simple as specifying:

- Where to save outputs (`out`)
- Which model to train (`model`)
- Where your images are (`data`)

=== "Python"

    ```python
    # pretrain_yolo.py
    import lightly_train

    if __name__ == "__main__":
        # Pre-train with LightlyTrain.
        lightly_train.train(
            out="out/coco_minitrain_pretrain",  # Output directory.
            model="ultralytics/yolo11s.yaml",  # Pass the YOLO model (use .yaml ending to start with random weights).
            data="coco_minitrain_25k/images",  # Path to a directory with training images.
            epochs=100,  # Adjust epochs for shorter training.
            batch_size=128,  # Adjust batch size based on hardware.
        )
    ```

=== "Command Line"

    ```bash
    lightly-train train out="out/coco_minitrain_pretrain" model="ultralytics/yolo11s.yaml" data="coco_minitrain_25k/images" epochs=100 batch_size=128
    ```

## Fine-tuning YOLO for Object Detection

### Getting the Labeled VOC Dataset

Now let's get our labeled dataset for fine-tuning:

```python
from ultralytics import settings
from ultralytics.data.utils import check_det_dataset

# Download VOC
dataset = check_det_dataset("VOC.yaml")

# Check where it was saved
print(settings["datasets_dir"])
```

We can have a look at some of the samples in the dataset with `supervision`:

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

with open(dataset["yaml_file"]) as f:
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

![VOC Samples](https://raw.githubusercontent.com/lightly-ai/lightly-train/refs/heads/main/docs/source/tutorials/yolo/samples_VOC_train2012.png)

### Fine-tuning for Object Detection

Time to transform our pretrained model into an object detector! We'll fine-tune two models on the VOC dataset, one from random weights and one from our pretrained model:

=== "From Pretrained"

    ```python
    from ultralytics import YOLO

    # Load our pretrained model
    model = YOLO("out/coco_minitrain_pretrain/exported_models/exported_last.pt")

    # Fine-tune on VOC
    model.train(data="VOC.yaml", epochs=30, project="logs/voc_yolo11s", name="from_pretrained")
    ```

=== "From Scratch (for comparison)"

    ```python
    from ultralytics import YOLO

    # Start with random weights
    model = YOLO("yolo11s.yaml")

    # Train on VOC
    model.train(data="VOC.yaml", epochs=30, project="logs/voc_yolo11s", name="from_scratch")
    ```

## Compare the Results

Want to see the magic of pretraining? Let's plot the results:

```python
import matplotlib.pyplot as plt
import pandas as pd

res_scratch = pd.read_csv("logs/voc_yolo11s/from_scratch/results.csv")
res_finetune = pd.read_csv("logs/voc_yolo11s/from_pretrained/results.csv")

fig, ax = plt.subplots()
ax.plot(res_scratch["epoch"], res_scratch["metrics/mAP50-95(B)"], label="scratch")
ax.plot(res_finetune["epoch"], res_finetune["metrics/mAP50-95(B)"], label="finetune")
ax.set_xlabel("Epoch")
ax.set_ylabel("mAP50-95")
max_pretrained = res_finetune["metrics/mAP50-95(B)"].max()
max_scratch = res_scratch["metrics/mAP50-95(B)"].max()
ax.set_title(f"Pretraining is {(max_pretrained - max_scratch) / max_scratch * 100:.2f}% better than scratch")
ax.legend()
plt.show()
```

![Pretraining vs Scratch](https://raw.githubusercontent.com/lightly-ai/lightly-train/refs/heads/main/docs/source/tutorials/yolo/results_VOC.png)

As you can see, pretraining gives us a significant boost in performance and much faster convergence compared to training from scratch! 🎉

## Ready to Level Up? 🚀

Now that you've seen the power of pretraining, here's what to try next:

- Explore different pretraining methods like [DINO and SimCLR](https://docs.lightly.ai/train/stable/methods/index.html)
- Try other YOLO flavors (`YOLOv5`, `YOLOv6`, `YOLOv8`)
- Use your pretrained model for [image embeddings](https://docs.lightly.ai/train/stable/embed.html)

Happy pretraining! ⚡️
