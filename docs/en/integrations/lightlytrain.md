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

## Step 1: Get the Unlabeled Data Ready

Let's start by downloading a subset of COCO (25k images) that we'll use for pretraining:

```bash
# Download COCO-minitrain
wget https://huggingface.co/datasets/bryanbocao/coco_minitrain/resolve/main/coco_minitrain_25k.zip

# Unzip it
unzip coco_minitrain_25k.zip

# Remove labels (we don't need them for pretraining!)
rm -rf coco_minitrain_25k/labels
```

## Step 2: Pretrain Your Model

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
            model="ultralytics/yolo11s.yaml",   # Pass the YOLO model (use .yaml ending to start with random weights).
            data="coco_minitrain_25k/images",   # Path to a directory with training images.
            epochs=100,                         # Adjust epochs for shorter training.
            batch_size=128,                     # Adjust batch size based on hardware.
        )
    ```

=== "Command Line"

    ```bash
    lightly-train train out="out/coco_minitrain_pretrain" model="ultralytics/yolo11s.yaml" data="coco_minitrain_25k/images" epochs=100 batch_size=128
    ```

## Step 3: Download PASCAL VOC

Now let's get our labeled dataset for fine-tuning:

```python
from ultralytics.data.utils import check_det_dataset
from ultralytics import settings

# Download VOC
dataset = check_det_dataset("VOC.yaml")

# Check where it was saved
print(settings["datasets_dir"])
```

## Step 4: Fine-tune for Object Detection 

Time to transform our pretrained model into an object detector! We'll train two models to show the power of pretraining:

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
ax.set_title(
    f"Pretraining is {(max_pretrained - max_scratch) / max_scratch * 100:.2f}% better than scratch"
)
ax.legend()
plt.show()
```
![Pretraining vs Scratch](https://raw.githubusercontent.com/lightly-ai/lightly-train/refs/heads/main/docs/source/tutorials/yolo/results_VOC.png)

## Ready to Level Up? 🚀

Now that you've seen the power of pretraining, here's what to try next:

- Explore different pretraining methods like [DINO and SimCLR](https://docs.lightly.ai/train/stable/methods/index.html)
- Try other YOLO flavors (`YOLOv5`, `YOLOv6`, `YOLOv8`)
- Use your pretrained model for [image embeddings](https://docs.lightly.ai/train/stable/embed.html)

Happy training! ⚡️