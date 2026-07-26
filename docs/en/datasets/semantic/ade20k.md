---
title: ADE20K Segmentation Dataset
comments: true
creator:
    name: MIT CSAIL
    url: https://ade20k.csail.mit.edu/
license:
    name: Research-Only
    url: https://ade20k.csail.mit.edu/terms
description: Train Ultralytics YOLO on the ADE20K dataset — 20,210 training and 2,000 validation images across 150 scene-parsing classes for semantic segmentation.
keywords: ADE20K dataset, semantic segmentation, scene parsing, Ultralytics YOLO, YOLO26, ADEChallengeData2016, computer vision, deep learning
---

# ADE20K Dataset

The [ADE20K](http://sceneparsing.csail.mit.edu/) dataset is a large-scale [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation) benchmark from MIT CSAIL with 20,210 training and 2,000 validation images densely annotated across 150 indoor, outdoor, object, and stuff categories. It is a standard resource for training and evaluating dense scene-understanding models with [Ultralytics YOLO](../../models/index.md).

## Key Features

- ADE20K's full SceneParsing benchmark totals 25,562 images: 20,210 for training, 2,000 for validation, and 3,352 for testing. Test-image annotations are not publicly released, so the downloadable `ADEChallengeData2016` archive and the Ultralytics `ade20k.yaml` config use only the training and validation splits.
- The dataset covers 150 semantic classes spanning indoor, outdoor, object, and stuff categories.
- Annotations are dense pixel-level segmentation masks suitable for scene parsing.

## Dataset Structure

The Ultralytics configuration expects the official ADEChallengeData2016 layout:

```text
ADEChallengeData2016/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

!!! warning "Manual Download Required"

    ADE20K has no automatic download script. Download the ~1 GB [`ADEChallengeData2016.zip`](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) archive and extract it directly into your `datasets/` folder. The archive's own top-level folder is already named `ADEChallengeData2016/`, so this produces `datasets/ADEChallengeData2016/` matching the layout above — do not create an `ADEChallengeData2016` folder yourself and extract into it, or you'll end up with a nested `datasets/ADEChallengeData2016/ADEChallengeData2016/` directory that the YAML won't find.

The `masks_dir` field is set to `annotations`, so each image under `images/` is paired with its corresponding mask under `annotations/`. The original ADE20K masks use source label IDs where `0` is ignored, and the `label_mapping` section converts valid labels `1` through `150` to contiguous train IDs `0` through `149`, mapping ignored pixels to `255`.

## Applications

ADE20K is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in semantic segmentation and scene parsing. Its diverse set of categories and complex scenes make it valuable for applications such as autonomous navigation, robotics, augmented reality, and image editing.

The breadth of indoor and outdoor scenes also makes ADE20K a strong benchmark for evaluating model generalization across domains. Pretrained YOLO26 semantic segmentation models reach up to 51.5 mIoU on the ADE20K validation set — see the [semantic segmentation models](../../tasks/semantic.md) page for the full benchmark table. ADE20K-format datasets are also fully compatible with [Ultralytics Platform](https://platform.ultralytics.com/) for dataset management and training.

## Dataset YAML

A dataset YAML file defines the ADE20K paths, classes, mask directory, and label mapping. The `ade20k.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ade20k.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ade20k.yaml).

!!! example "ultralytics/cfg/datasets/ade20k.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/ade20k.yaml"
    ```

## Usage

To train a YOLO26n-sem model on the ADE20K dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 512, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-sem.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="ade20k.yaml", epochs=100, imgsz=512)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo semantic train data=ade20k.yaml model=yolo26n-sem.pt epochs=100 imgsz=512
        ```

## Citations, License and Acknowledgments

ADE20K images are released for [non-commercial research and educational use only](https://ade20k.csail.mit.edu/terms); the dataset's annotation software is separately licensed under BSD-3. Commercial use requires permission from MIT CSAIL.

If you use the ADE20K dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{zhou2017scene,
          title={Scene Parsing through ADE20K Dataset},
          author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
          booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
          year={2017}
        }
        ```

We would like to acknowledge the MIT CSAIL Computer Vision Group for creating and maintaining this valuable resource for the computer vision community. For more information about the ADE20K dataset and its creators, visit the [ADE20K dataset website](http://sceneparsing.csail.mit.edu/).

## FAQ

### What is the ADE20K dataset and why is it important for computer vision?

The [ADE20K](http://sceneparsing.csail.mit.edu/) dataset is a large-scale scene parsing benchmark used for [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation), with 20,210 training and 2,000 validation images publicly released across 150 categories covering indoor, outdoor, object, and stuff classes. Researchers use ADE20K because of its diverse scenes, fine-grained category set, and standardized evaluation metrics like mean Intersection over Union (mIoU), which make it ideal for benchmarking dense prediction models.

### How can I train a YOLO model using the ADE20K dataset?

To train a YOLO26n-sem model on the ADE20K dataset for 100 epochs with an image size of 512, you can use the following code snippets. For a detailed list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-sem.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="ade20k.yaml", epochs=100, imgsz=512)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo semantic train data=ade20k.yaml model=yolo26n-sem.pt epochs=100 imgsz=512
        ```

### How is the ADE20K dataset structured?

The ADE20K dataset follows the official ADEChallengeData2016 layout, with images organized under `images/training/` and `images/validation/`, and corresponding masks under `annotations/training/` and `annotations/validation/`. The Ultralytics YAML file pairs each image with its mask via the `masks_dir: annotations` field, and uses `label_mapping` to convert source label IDs `1`–`150` into contiguous train IDs `0`–`149`, mapping the ignore label to `255`.

### Do I need to download ADE20K manually?

Yes. Download the [`ADEChallengeData2016.zip`](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) archive (~1 GB) and extract it directly into your `datasets/` folder before training — the archive's own top-level folder is already named `ADEChallengeData2016/`, so extracting it there (not into a separate `ADEChallengeData2016` folder you create yourself) produces the `images/` and `annotations/` layout that `ade20k.yaml` expects.

### Why does ADE20K use `label_mapping`?

ADE20K annotation masks store source label IDs where `0` denotes the ignore or background class. The `label_mapping` section maps valid labels `1` through `150` to contiguous train IDs `0` through `149`, and assigns `255` to ignored pixels so they are excluded from the loss and metrics during training and validation.

### Is the ADE20K dataset free for commercial use?

No. ADE20K images are released under [terms restricting use to non-commercial research and education](https://ade20k.csail.mit.edu/terms); the accompanying annotation software is separately licensed under BSD-3. Contact MIT CSAIL for commercial licensing options.
