---
title: Construction-PPE Detection Dataset
comments: true
creator:
    name: Ultralytics
    url: https://www.ultralytics.com/
license:
    name: AGPL-3.0
    url: https://www.ultralytics.com/license
description: Train YOLO26 on the Construction-PPE dataset — 1,416 images across 11 classes for detecting helmets, gloves, vests, boots, goggles, and missing safety gear.
keywords: Construction-PPE, PPE dataset, PPE detection, safety compliance, construction workers, object detection, YOLO26, workplace safety, computer vision
---

# Construction-PPE Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-construction-ppe-detection-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Construction-PPE Dataset In Colab"></a>

The Ultralytics Construction-PPE dataset is an [object detection](https://www.ultralytics.com/glossary/object-detection) dataset of 1,416 images (1,132 for training, 143 for validation, and 141 for testing) labeled across 11 classes for detecting personal protective equipment — helmets, gloves, vests, boots, and goggles — and flagging missing gear on construction sites. Curated from real construction environments, it includes both compliant and non-compliant cases, making it a practical resource for training [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models that monitor workplace safety.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/lFaVnrhMmaE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to train Ultralytics YOLO on Personal Protective Equipment Dataset | VisionAI in Construction 👷
</p>

## Dataset Structure

The Construction-PPE dataset contains 1,416 images split into three predefined subsets, defined by the `construction-ppe.yaml` configuration:

| Split      | Images | Annotations |
| ---------- | ------ | ----------- |
| Train      | 1,132  | Yes         |
| Validation | 143    | Yes         |
| Test       | 141    | Yes         |

Every image is annotated in the [Ultralytics YOLO](../detect/index.md#what-is-the-ultralytics-yolo-dataset-format-and-how-to-structure-it) format, ensuring compatibility with state-of-the-art [object detection](../../tasks/detect.md) and [tracking](../../modes/track.md) pipelines.

The dataset provides **11 classes** covering worn gear, missing gear, and people:

- **Worn PPE (5)**: `helmet`, `gloves`, `vest`, `boots`, `goggles`
- **Missing PPE (4)**: `no_helmet`, `no_gloves`, `no_boots`, `no_goggle`
- **Other (2)**: `Person`, `none`

Pairing worn and missing labels lets a model both detect properly worn gear **and** flag safety violations. Note that `vest` has no dedicated missing-vest label.

## Business Value

Construction is one of the most hazardous industries, and the challenge is usually enforcement rather than a lack of regulation. Health-and-safety teams are stretched thin and cannot watch every corner of a busy, ever-changing site in real time.

Computer-vision-based PPE detection helps close that gap. By automatically checking whether workers wear the required helmets, vests, and other gear, it enforces safety rules consistently across sites and surfaces leading indicators of risk — revealing compliance trends before incidents occur. PPE detection can also flag unauthorized site intruders, who are typically the first to appear without proper safety gear.

## Applications

Construction-PPE powers a variety of safety-focused computer vision applications:

- **Automated compliance monitoring**: Train AI models to instantly check if workers are wearing required safety gear like helmets, vests, or gloves, reducing risks on site.
- **Workplace safety analytics**: Track PPE usage over time, spot frequent violations, and generate insights to improve safety culture.
- **Smart surveillance systems**: Connect detection models with cameras to send real-time alerts when PPE is missing, preventing accidents before they happen.
- **Robotics and autonomous systems**: Enable drones or robots to perform PPE checks across large sites, supporting faster and safer inspections.
- **Research and education**: Provide a real-world dataset for students and researchers exploring workplace safety and human-object interactions.

To label, train, and deploy a PPE detection model without managing local infrastructure, run the full workflow in your browser with [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

The Construction-PPE dataset includes a YAML configuration file that defines the train, validation, and test image paths along with the full list of object classes. You can access the `construction-ppe.yaml` file directly in the Ultralytics repository here: [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/construction-ppe.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/construction-ppe.yaml)

!!! example "ultralytics/cfg/datasets/construction-ppe.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/construction-ppe.yaml"
    ```

## Usage

You can train a YOLO26n model on the Construction-PPE dataset for 100 epochs with an image size of 640. The following examples show how to get started quickly. For more options and advanced configurations, see the [Training guide](../../modes/train.md).

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load pretrained model
        model = YOLO("yolo26n.pt")

        # Train the model on Construction-PPE dataset
        model.train(data="construction-ppe.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=construction-ppe.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The dataset captures construction workers across varied environments, lighting conditions, and postures. Both **compliant** and **non-compliant** cases are included.

![Construction-PPE dataset sample with safety gear detection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/construction-ppe-dataset-sample.avif)

## License and Attribution

Construction-PPE is developed and released under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), supporting open-source research and commercial applications with proper attribution.

If you use this dataset in your research, please cite it:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Dalvi_Construction_PPE_Dataset_2025,
            author = {Mrunmayee Dalvi and Niyati Singh and Sahil Bhingarde and Ketaki Chalke},
            title = {Construction-PPE: Personal Protective Equipment Detection Dataset},
            month = {January},
            year = {2025},
            version = {1.0.0},
            license = {AGPL-3.0},
            url = {https://docs.ultralytics.com/datasets/detect/construction-ppe/},
            publisher = {Ultralytics}
        }
        ```

## FAQ

### What makes the Construction-PPE dataset unique?

Unlike generic construction datasets, Construction-PPE explicitly includes **missing-equipment classes** (`no_helmet`, `no_gloves`, `no_boots`, `no_goggle`). This dual-labeling approach lets a model not only detect worn PPE but also flag violations in real time.

### Which object categories are included?

The Construction-PPE dataset has 11 classes: five worn-PPE items (`helmet`, `gloves`, `vest`, `boots`, `goggles`), four missing-PPE labels (`no_helmet`, `no_gloves`, `no_boots`, `no_goggle`), plus `Person` and a generic `none` class. Note that `vest` has no dedicated missing-vest label.

### How many images and classes are in the Construction-PPE dataset?

The Construction-PPE dataset contains 1,416 images across 11 classes — 1,132 for training, 143 for validation, and 141 for testing. See the [Dataset Structure](#dataset-structure) section for the full split and class breakdown.

### How do I download the Construction-PPE dataset?

The dataset (178.4 MB) downloads automatically the first time you train with `data="construction-ppe.yaml"` — no manual step is required. Ultralytics fetches and unpacks it to your local datasets directory. You can browse related datasets in the [detection datasets overview](index.md).

### How can I train a YOLO model using the Construction-PPE dataset?

To train a YOLO26 model using the Construction-PPE dataset, you can use the following code snippets:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="construction-ppe.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=construction-ppe.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

### Is this dataset suitable for real-world applications?

Yes. Images are curated from real construction sites under diverse conditions, which makes the dataset highly effective for building deployable workplace safety monitoring systems.

### What are the benefits of using the Construction-PPE dataset in AI projects?

The dataset enables real-time detection of personal protective equipment, helping monitor worker safety on construction sites. With classes for both worn and missing gear, it supports AI systems that can automatically flag safety violations, generate compliance insights, and reduce risks. It also provides a practical resource for developing computer vision solutions in workplace safety, robotics, and academic research.
