---
title: Construction-PPE Detection Dataset
comments: true
creator:
    name: Ultralytics
    url: https://www.ultralytics.com
license:
    name: CC BY 4.0
    url: https://creativecommons.org/licenses/by/4.0/
description: Train YOLO26 on Ultralytics Construction-PPE — 1,000 human-collected and generated images with 12 classes for people, forklifts, protective equipment, and missing PPE.
keywords: Construction-PPE, Ultralytics dataset, PPE detection, construction safety, generated images, object detection, YOLO26, CC BY 4.0
---

# Ultralytics Construction-PPE Dataset

Ultralytics Construction-PPE is an original [object detection](../../tasks/detect.md) dataset of **1,000 images, 14,863 bounding boxes, and 12 classes** covering people, forklifts, personal protective equipment, and visible missing equipment in construction and industrial settings.

The dataset was **created and open-sourced in November 2026 by the Ultralytics team**. It combines **human-collected and generated images**, with **expert human annotation and AI auto-annotation completed in November 2026 by the Ultralytics team**. Images and annotations are released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

[Explore Construction-PPE on Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/construction-ppe) to browse the images and annotations, clone the dataset, or train a model.

## Images and Annotation

The images cover construction sites, workshops, warehouses, loading areas, and maintenance work. They include varied viewpoints, lighting, worker arrangements, protective equipment, and forklift activity.

Images include landscape, portrait, and square compositions at varied resolutions.

Annotations use bounding boxes in [Ultralytics YOLO format](index.md#ultralytics-yolo-format). Equipment classes identify visible items; missing-equipment classes identify the corresponding visible body region when equipment is absent. An occluded or out-of-frame body part does not by itself establish missing equipment.

## Dataset Structure

The dataset has a fixed 80/20 train/validation split:

| Split      | Images    | Purpose                       |
| ---------- | --------- | ----------------------------- |
| Train      | 800       | Model training                |
| Validation | 200       | Evaluation during development |
| **Total**  | **1,000** |                               |

There is no separate test split. Use independent images from the intended deployment environment for final evaluation, especially when assessing performance on real camera footage.

## Classes

The class IDs below match the dataset annotations and `construction-ppe.yaml`:

| ID  | Class       | Target                                                    |
| --- | ----------- | --------------------------------------------------------- |
| 0   | `helmet`    | Protective helmet or hard hat                             |
| 1   | `gloves`    | Work gloves                                               |
| 2   | `vest`      | High-visibility safety vest, jacket, or shirt             |
| 3   | `boots`     | Protective work boots or work shoes                       |
| 4   | `goggles`   | Protective eyewear                                        |
| 5   | `Person`    | Person                                                    |
| 6   | `no_helmet` | Visible head without a helmet                             |
| 7   | `no_goggle` | Visible eye region without protective eyewear             |
| 8   | `no_gloves` | Visible hand without a glove                              |
| 9   | `no_boots`  | Visible foot or footwear without protective work footwear |
| 10  | `forklift`  | Forklift                                                  |
| 11  | `no_vest`   | Visible torso without high-visibility safety clothing     |

## Dataset YAML

The [dataset configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/construction-ppe.yaml) defines the image paths, class order, and download URL:

!!! example "ultralytics/cfg/datasets/construction-ppe.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/construction-ppe.yaml"
    ```

The [November 2026 dataset archive](https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip) contains images, YOLO labels, dataset metadata, and the CC BY 4.0 license. It extracts into `construction-ppe`.

## Usage

Install or update the [Ultralytics package](../../quickstart.md), then train a pretrained YOLO26 model. YOLO downloads and extracts the dataset automatically on first use.

!!! example "Train on Construction-PPE"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")
        model.train(
            data="https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip",
            epochs=100,
            imgsz=640,
        )
        ```

    === "CLI"

        ```bash
        yolo detect train model=yolo26n.pt data=https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip epochs=100 imgsz=640
        ```

See the [training guide](../../modes/train.md) for configuration options and the [validation guide](../../modes/val.md) for evaluating a trained model. The [Construction-PPE notebook](https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-construction-ppe-detection-dataset.ipynb) provides a Colab walkthrough.

## Applications

- **PPE detection:** locate helmets, gloves, vests, boots, and protective eyewear.
- **Missing-equipment detection:** identify visible examples of missing PPE for review.
- **Worksite analysis:** detect people and forklifts in construction and industrial imagery.
- **Research and education:** compare models and study training with human-collected and generated images.

## License and Attribution

The dataset's images and annotations are released by Ultralytics under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), which permits sharing and adaptation, including commercial use, with attribution. Credit the Ultralytics team, link to the dataset and license, and indicate any changes.

Suggested attribution: **Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0.**

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @dataset{Ultralytics_Construction_PPE_2026,
            author = "{Ultralytics team}",
            title = {Ultralytics Construction-PPE Dataset},
            month = {November},
            year = {2026},
            license = {CC-BY-4.0},
            url = {https://docs.ultralytics.com/datasets/detect/construction-ppe},
            publisher = {Ultralytics}
        }
        ```
