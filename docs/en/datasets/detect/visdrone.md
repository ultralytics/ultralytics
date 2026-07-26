---
title: VisDrone Detection Dataset
comments: true
creator:
    name: VisDrone Team
    url: https://github.com/VisDrone/VisDrone-Dataset
license:
    name: None
description: Train YOLO26 on the VisDrone-DET aerial dataset - 6,471 train, 548 val, and 1,610 test drone images across 10 object classes with automatic download.
keywords: VisDrone, VisDrone-DET, drone dataset, aerial object detection, small object detection, UAV imagery, YOLO26, object detection dataset
---

# VisDrone Dataset

The [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) is a large-scale drone-imagery benchmark whose detection subset (VisDrone2019-DET) provides 8,629 aerial images — 6,471 train, 548 validation, and 1,610 test-dev — annotated with 10 object classes for [object detection](../../tasks/detect.md). It was created by the AISKYEYE team at the Lab of [Machine Learning](https://www.ultralytics.com/glossary/machine-learning-ml) and Data Mining, Tianjin University, China.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/rTblZN9IRDo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO on the VisDrone Dataset | Aerial Detection | Complete Tutorial 🚀
</p>

The full VisDrone benchmark comprises 288 video clips (261,908 frames) and 10,209 static images captured by drone-mounted cameras in 14 different cities across China, spanning urban and rural environments, sparse and crowded scenes, and varied weather and lighting conditions. Its frames carry over 2.6 million manually annotated [bounding boxes](https://www.ultralytics.com/glossary/bounding-box), with extra attributes such as scene visibility, object class, and occlusion. The Ultralytics `VisDrone.yaml` configuration uses the VisDrone2019-DET static-image subset of this benchmark.

## Key Features

- **Small, dense objects**: Aerial viewpoints make targets tiny and crowded — the 548 validation images alone contain 38,759 labeled boxes, an average of about 70 objects per image.
- **Scene diversity**: Imagery from 14 Chinese cities covering urban and rural locations, day and night, and different weather conditions.
- **Rich annotations**: Over 2.6 million boxes across the full benchmark, with occlusion and visibility attributes.
- **Pre-defined splits**: Fixed train / val / test-dev splits (6,471 / 548 / 1,610 images) for consistent evaluation.

## Dataset Structure

The Ultralytics VisDrone configuration covers the VisDrone2019-DET image subset, split into three parts:

| Split      | Images | Description                                                         |
| ---------- | ------ | ------------------------------------------------------------------- |
| Train      | 6,471  | Labeled aerial images used to train the detector                    |
| Validation | 548    | Images used for [evaluation](../../modes/val.md) during development |
| Test-dev   | 1,610  | Held-out images for final evaluation of the trained model           |

A fourth split, test-challenge (1,580 images), is withheld for the VisDrone competition and is not downloaded, which is why the full DET set totals 10,209 images.

The dataset annotates 10 object classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, and motor. VisDrone distinguishes **pedestrian** (a person standing or walking) from **people** (a person in any other posture).

!!! note "Automatic YOLO conversion"

    On first use the download script converts the original VisDrone annotations to YOLO format, skipping regions marked as ignored (which also excludes the unused "others" category).

## Applications

VisDrone's dense scenes and tiny targets make it a standard benchmark for [small-object detection](https://www.ultralytics.com/glossary/object-detection) from aerial viewpoints. Common applications include:

- Traffic monitoring and vehicle counting from UAVs
- Crowd analysis and public-safety surveillance
- Infrastructure and construction-site inspection
- [Computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) research on detecting small objects in cluttered scenes

For other aerial-imagery benchmarks, see the satellite-focused [xView dataset](xview.md) or the oriented-box [DOTA-v2 dataset](../obb/dota-v2.md).

## Dataset YAML

The `VisDrone.yaml` file defines the dataset configuration — the dataset paths, class names, and the automatic download-and-convert script. It is maintained in the Ultralytics repository at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml).

!!! example "ultralytics/cfg/datasets/VisDrone.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/VisDrone.yaml"
    ```

## Usage

!!! note "~2 GB download"

    VisDrone downloads automatically the first time you train — three archives totaling about 2 GB — and needs roughly 4 GB of free disk space during extraction and conversion.

To train a YOLO26n model on the VisDrone dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model - dataset will auto-download on first run
        results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        # Dataset will auto-download and convert on first run
        yolo detect train data=VisDrone.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

To label additional aerial images and manage VisDrone training runs in your browser, use [Ultralytics Platform](https://platform.ultralytics.com/).

## Sample Data and Annotations

The sample below shows a typical VisDrone scene: an aerial viewpoint over a busy road where pedestrians and vehicles appear as small, densely packed targets, many partially occluded by one another.

![VisDrone dataset aerial drone imagery with object detection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/visdrone-object-detection-sample.avif)

## Citations and Acknowledgments

If you use the VisDrone dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @ARTICLE{9573394,
          author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Detection and Tracking Meet Drones Challenge},
          year={2022},
          volume={44},
          number={11},
          pages={7380-7399},
          doi={10.1109/TPAMI.2021.3119563}}
        ```

We would like to acknowledge the AISKYEYE team at the Lab of Machine Learning and [Data Mining](https://www.ultralytics.com/glossary/data-mining), Tianjin University, China, for creating and maintaining the VisDrone dataset. For more information, visit the [VisDrone Dataset GitHub repository](https://github.com/VisDrone/VisDrone-Dataset).

## FAQ

### What is the VisDrone dataset used for?

VisDrone is used to train and benchmark detectors on drone-captured imagery, where objects are small, dense, and seen from above. Its combination of aerial viewpoints, crowded scenes, and varied conditions makes it a standard testbed for UAV-based traffic monitoring, crowd analysis, and small-object detection research.

### How many images and classes does VisDrone have?

The Ultralytics VisDrone configuration contains 8,629 images: 6,471 for training, 548 for validation, and 1,610 for testing (test-dev). All splits share the same 10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, and motor. See [Dataset Structure](#dataset-structure) for the full breakdown.

### How do I download the VisDrone dataset?

VisDrone downloads automatically the first time you train with `data="VisDrone.yaml"` — no manual steps are required. The script fetches three archives (about 2 GB) from Ultralytics GitHub release assets and converts the annotations to YOLO format. The competition's withheld test-challenge split is not included.

### How do I train a YOLO26 model on the VisDrone dataset?

Train a YOLO26n model on VisDrone for 100 epochs at an image size of 640:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=VisDrone.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For detailed configurations, see the [Training](../../modes/train.md) page and [model training tips](../../guides/model-training-tips.md).

### Why is VisDrone difficult for object detectors, and how can I improve accuracy?

Objects in VisDrone are tiny relative to the frame — often only a few dozen pixels — and appear in dense, heavily occluded groups, which strains detectors tuned on ground-level photos. Training and predicting at a higher resolution (for example `imgsz=1280` with a smaller batch) recovers small targets, and [SAHI tiled inference](../../guides/sahi-tiled-inference.md) slices large images so small objects occupy more of each inference window.

### What is the difference between VisDrone-DET and the full VisDrone benchmark?

The full VisDrone benchmark spans five tasks — object detection in images, object detection in videos, single-object tracking, [multi-object tracking](../index.md#multi-object-tracking), and crowd counting — across 288 video clips and 10,209 static images. The Ultralytics `VisDrone.yaml` configuration covers only the image detection task (VisDrone2019-DET), downloading its 6,471 train, 548 validation, and 1,610 test-dev images.

### How do I cite VisDrone in my research?

Cite the paper "Detection and Tracking Meet Drones Challenge" (IEEE TPAMI, vol. 44, no. 11, 2022, DOI 10.1109/TPAMI.2021.3119563); the full BibTeX entry is in the [Citations and Acknowledgments](#citations-and-acknowledgments) section above.
