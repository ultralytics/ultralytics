---
title: xView Detection Dataset
comments: true
description: Train YOLO26 on the xView satellite dataset - 1M+ object instances across 60 classes in 0.3 m WorldView-3 imagery with automatic GeoJSON-to-YOLO conversion.
keywords: xView dataset, satellite imagery, overhead imagery, object detection, remote sensing, YOLO26, xView download, WorldView-3, bounding boxes, computer vision
---

# xView Dataset

The [xView](https://xviewdataset.org/) dataset is one of the largest publicly available satellite-imagery benchmarks for [object detection](../../tasks/detect.md), providing over 1 million object instances across 60 classes annotated with [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) in more than 1,400 km² of 0.3 m WorldView-3 imagery. It was released for the DIUx xView 2018 Challenge by the U.S. National Geospatial-Intelligence Agency (NGA) and requires a manual download of about 20.7 GB.

The dataset was created to push four [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) frontiers:

1. Reduce minimum resolution for detection.
2. Improve learning efficiency.
3. Enable discovery of more object classes.
4. Improve detection of fine-grained classes.

Building on benchmarks like [COCO](coco.md), xView targets overhead imagery, where objects are far smaller and more densely packed than in ground-level photos.

!!! warning "Manual Download Required"

    The xView dataset is **not** downloaded automatically. Register at the [DIUx xView 2018 Challenge](https://challenge.xviewdataset.org/) website to download `train_images.zip` (~15 GB), `train_labels.zip`, and `val_images.zip` (~5 GB), then extract them under `datasets/xView/` so that it contains:

    ```text
    datasets/xView/
    ├── train_images/          # 847 TIF satellite images
    ├── val_images/            # 282 TIF images (no public labels)
    └── xView_train.geojson    # bounding-box annotations
    ```

    On the first training run, Ultralytics converts the GeoJSON annotations to YOLO format and splits the labeled images roughly 90/10 into training and validation sets automatically — no manual conversion is needed.

## Key Features

- **Fine-grained classes**: 60 object classes spanning aircraft, vehicles, railway stock, maritime vessels, construction equipment, and buildings — many small, rare, and visually similar.
- **High resolution**: 0.3 m ground sample distance collected from WorldView-3 satellites.
- **Dense annotation**: over 1 million object instances across more than 1,400 km² of imagery, all labeled with horizontal bounding boxes.
- **Automatic conversion**: the Ultralytics download script converts the original GeoJSON labels to YOLO format and generates the train/val split on first use.

## Dataset Structure

xView images are large satellite scenes in TIF format, and only the 847 training images ship with public labels — the 282-image challenge validation set has none. The Ultralytics `xView.yaml` configuration therefore splits the labeled images automatically on first use:

| Split      | Images      | Description                                                                             |
| ---------- | ----------- | --------------------------------------------------------------------------------------- |
| Train      | ~90% of 847 | Labeled images listed in `autosplit_train.txt`, generated on the first run              |
| Validation | ~10% of 847 | Labeled images listed in `autosplit_val.txt`, used for [evaluation](../../modes/val.md) |

The 60 classes cover fine-grained categories such as Fixed-wing Aircraft, Cargo Plane, Small Car, Bus, Locomotive, Maritime Vessel, Excavator, Building, Aircraft Hangar, and Storage Tank; the full list is in the [Dataset YAML](#dataset-yaml) below. During conversion, the original challenge class IDs (11–94) are remapped to contiguous indices 0–59.

## Applications

xView's fine-grained classes and high-resolution overhead viewpoint make it a standard benchmark for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in [remote sensing](https://www.ultralytics.com/blog/using-computer-vision-to-analyze-satellite-imagery). Common applications include:

- Military and defense reconnaissance
- Urban planning and development
- Environmental monitoring
- Disaster response and assessment
- Infrastructure mapping and management

For other overhead-imagery benchmarks, see the drone-focused [VisDrone dataset](visdrone.md) or the oriented-box [DOTA-v2 dataset](../obb/dota-v2.md).

## Dataset YAML

The `xView.yaml` file defines the dataset configuration — the dataset paths, the 60 class names, and the download script that converts the GeoJSON annotations and generates the autosplit. It is maintained in the Ultralytics repository at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/xView.yaml).

!!! example "ultralytics/cfg/datasets/xView.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/xView.yaml"
    ```

## Usage

!!! note "20.7 GB manual download"

    Training expects the manual download described above to be extracted under `datasets/xView/`; annotation conversion and the train/val split then run automatically.

To train a model on the xView dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="xView.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=xView.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

To label additional satellite images and manage xView training runs in your browser, use [Ultralytics Platform](https://platform.ultralytics.com/).

## Sample Data and Annotations

The sample below shows a typical xView scene: high-resolution overhead imagery in which small objects such as vehicles and buildings are annotated with bounding boxes, illustrating why [object detection](https://www.ultralytics.com/glossary/object-detection) in satellite imagery demands fine-grained localization.

![xView dataset overhead satellite imagery with object detection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/overhead-imagery-object-detection.avif)

## Citations and Acknowledgments

If you use the xView dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lam2018xview,
              title={xView: Objects in Context in Overhead Imagery},
              author={Darius Lam and Richard Kuzma and Kevin McGee and Samuel Dooley and Michael Laielli and Matthew Klaric and Yaroslav Bulatov and Brendan McCord},
              year={2018},
              eprint={1802.07856},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

We would like to acknowledge the [Defense Innovation Unit](https://www.diu.mil/) (DIU) and the creators of the xView dataset for their valuable contribution to the computer vision research community. For more information, visit the [xView dataset website](https://xviewdataset.org/).

## FAQ

### What is the xView dataset and how does it benefit computer vision research?

The [xView](https://xviewdataset.org/) dataset is a satellite-imagery benchmark released for the DIUx xView 2018 Challenge by the U.S. National Geospatial-Intelligence Agency, providing over 1 million object instances across 60 fine-grained classes in 0.3 m WorldView-3 imagery. It supports research on detecting small, rare, and fine-grained objects in overhead views, which are far harder targets than those in ground-level photos.

### How do I download and set up the xView dataset?

xView requires a manual download: register at the [DIUx xView 2018 Challenge](https://challenge.xviewdataset.org/) website, download `train_images.zip` (~15 GB), `train_labels.zip`, and `val_images.zip` (~5 GB) — about 20.7 GB in total — and extract them under `datasets/xView/` following the layout shown in the warning at the top of this page. On the first training run, Ultralytics automatically converts the GeoJSON annotations to YOLO format and creates the train/validation split.

### How many images and classes does xView have?

xView contains 847 labeled training images and 282 validation images without public labels, all captured by WorldView-3 satellites at 0.3 m resolution. Annotations cover over 1 million object instances across 60 classes. Because only the training labels are public, the Ultralytics `xView.yaml` configuration splits the 847 labeled images roughly 90/10 into training and validation sets; see [Dataset Structure](#dataset-structure) for details.

### How do I train a YOLO26 model on the xView dataset?

Train a [YOLO26n](../../models/yolo26.md) model on xView for 100 epochs at an image size of 640:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="xView.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=xView.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For detailed arguments and settings, refer to the model [Training](../../modes/train.md) page.

### How do I cite the xView dataset in my research?

Cite the paper "xView: Objects in Context in Overhead Imagery" (Lam et al., arXiv:1802.07856, 2018); the full BibTeX entry is in the [Citations and Acknowledgments](#citations-and-acknowledgments) section above.
