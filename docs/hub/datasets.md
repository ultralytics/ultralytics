---
comments: true
description: Upload custom datasets to Ultralytics HUB for YOLOv5 and YOLOv8 models. Follow YAML structure, zip and upload. Scan & train new models.
---

# HUB Datasets

## 1. Upload a Dataset

Ultralytics HUB datasets are just like YOLOv5 and YOLOv8 🚀 datasets, they use the same structure and the same label formats to keep
everything simple.

When you upload a dataset to Ultralytics HUB, make sure to **place your dataset YAML inside the dataset root directory**
as in the example shown below, and then zip for upload to [https://hub.ultralytics.com](https://hub.ultralytics.com/). Your **dataset YAML, directory
and zip** should all share the same name. For example, if your dataset is called 'coco8' as in our
example [ultralytics/hub/example_datasets/coco8.zip](https://github.com/ultralytics/hub/blob/master/example_datasets/coco8.zip), then you should have a `coco8.yaml` inside your `coco8/` directory, which should zip to create `coco8.zip` for upload:

```bash
zip -r coco8.zip coco8
```

The [example_datasets/coco8.zip](https://github.com/ultralytics/hub/blob/master/example_datasets/coco8.zip) dataset in this repository can be downloaded and unzipped to see exactly how to structure your custom dataset.

<p align="center">
<img width="80%" src="https://user-images.githubusercontent.com/26833433/201424843-20fa081b-ad4b-4d6c-a095-e810775908d8.png" title="COCO8" />
</p>

The dataset YAML is the same standard YOLOv5 and YOLOv8 YAML format. See
the [YOLOv5 and YOLOv8 Train Custom Data tutorial](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/) for full details.

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
Click 'Upload Dataset' to upload, scan and visualize your new dataset before training new YOLOv5 or YOLOv8 models on it!

<img width="100%" alt="HUB Dataset Upload" src="https://user-images.githubusercontent.com/26833433/216763338-9a8812c8-a4e5-4362-8102-40dad7818396.png">