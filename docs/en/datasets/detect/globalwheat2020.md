---
title: Global Wheat Head Detection Dataset
comments: true
description: Train YOLO26 on the Global Wheat Head Dataset — 3,422 train, 748 validation, and 1,276 test field images labeled with wheat head boxes for single-class detection.
keywords: Global Wheat Head Dataset, GWHD, wheat head detection, wheat spike detection, wheat phenotyping, crop management, object detection, YOLO26, agriculture
---

# Global Wheat Head Dataset

The [Global Wheat Head Dataset](https://www.global-wheat.com/) (GWHD) is a single-class [object detection](../../tasks/detect.md) dataset for detecting **wheat heads** — the grain-bearing spikes of the wheat plant — in outdoor field images. It provides 3,422 training, 748 validation, and 1,276 test images captured across multiple growing regions, and was created by a collaboration of nine research institutes from seven countries so that models generalize across different environments. Accurate wheat head detection underpins estimates of head density, size, and yield potential in [plant phenotyping](https://www.ultralytics.com/blog/computer-vision-in-agriculture-transforming-fruit-detection-and-precision-farming) and crop management.

## Key Features

- Real outdoor field images that capture the natural variability in wheat head appearance, lighting, and growth stage.
- Built by nine research institutes across seven countries, spanning European, North American, Asian, and Australian growing regions for strong cross-environment generalization.
- Bounding-box annotations for a single class, `wheat_head`, ready for [object detection](../../tasks/detect.md) and [tracking](../../modes/track.md) pipelines.
- Test images come from genotypes and regions unseen during training, providing a genuine generalization benchmark.

## Dataset Structure

The Global Wheat Head Dataset is organized into three subsets defined by the `GlobalWheat2020.yaml` configuration, all annotated with a single class, `wheat_head`:

| Split      | Images | Regions                                                  |
| ---------- | ------ | -------------------------------------------------------- |
| Train      | 3,422  | Europe (France, UK, Switzerland), North America (Canada) |
| Validation | 748    | Switzerland (ETH Zürich)                                 |
| Test       | 1,276  | Australia, Japan, China                                  |

!!! note "Validation split"

    The validation set (748 images) is the `ethz_1` subset, which is also part of the training domains — so validation metrics reflect in-domain performance. The held-out test set from Australia, Japan, and China measures generalization to environments unseen during training.

## Applications

The Global Wheat Head Dataset is widely used to train and evaluate [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models for wheat head detection. Its diverse imagery across regions, genotypes, and conditions makes it a valuable resource for [plant phenotyping](https://www.ultralytics.com/blog/from-farm-to-table-how-ai-drives-innovation-in-agriculture) and crop management — supporting yield estimation, crop-health monitoring, and phenotypic analysis.

To annotate field imagery, train, and manage dataset versions in your browser, run the full workflow with [Ultralytics Platform](https://platform.ultralytics.com/).

## Dataset YAML

A YAML file is used to define the dataset configuration. It defines the dataset's paths, classes, and other configuration details. For the Global Wheat Head Dataset, the `GlobalWheat2020.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml).

!!! example "ultralytics/cfg/datasets/GlobalWheat2020.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/GlobalWheat2020.yaml"
    ```

## Usage

To train a YOLO26n model on the Global Wheat Head Dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. The dataset (~7.0 GB) downloads automatically on first use. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=GlobalWheat2020.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The Global Wheat Head Dataset contains a diverse set of outdoor field images, capturing the natural variability in wheat head appearances, environments, and conditions. Here is an example image from the dataset, along with its corresponding annotations:

![Global Wheat dataset sample showing wheat head detection](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/wheat-head-detection-sample.avif)

- **Wheat Head Detection**: Wheat heads are annotated with bounding boxes for [object detection](../../tasks/detect.md), across a variety of field conditions that reflect the diversity and complexity of the dataset.

## Citations and Acknowledgments

If you use the Global Wheat Head Dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{david2020global,
                 title={Global Wheat Head Detection (GWHD) Dataset: A Large and Diverse Dataset of High-Resolution RGB-Labelled Images to Develop and Benchmark Wheat Head Detection Methods},
                 author={David, Etienne and Madec, Simon and Sadeghi-Tehran, Pouria and Aasen, Helge and Zheng, Bangyou and Liu, Shouyang and Kirchgessner, Norbert and Ishikawa, Goro and Nagasawa, Koichi and Badhon, Minhajul and others},
                 journal={arXiv preprint arXiv:2005.02162},
                 year={2020}
        }
        ```

We would like to acknowledge the researchers and institutions that contributed to the creation and maintenance of the Global Wheat Head Dataset as a valuable resource for the plant phenotyping and crop management research community. For more information about the dataset and its creators, visit the [Global Wheat Head Dataset website](https://www.global-wheat.com/).

## FAQ

### What is the Global Wheat Head Dataset used for?

The Global Wheat Head Dataset is primarily used for developing and training [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models aimed at wheat head detection. This is crucial for applications in [wheat phenotyping](https://www.ultralytics.com/blog/from-farm-to-table-how-ai-drives-innovation-in-agriculture) and crop management, allowing for more accurate estimations of wheat head density, size, and overall crop yield potential. Accurate detection methods help in assessing crop health and maturity, essential for efficient crop management.

### How many images and classes are in the Global Wheat Head Dataset?

The Global Wheat Head Dataset has a single class, `wheat_head`, and is split into three subsets: 3,422 training images, 748 validation images, and 1,276 test images. Training and validation images come from Europe and North America, while the test set is drawn from Australia, Japan, and China to evaluate generalization to unseen environments.

### How do I train a YOLO26n model on the Global Wheat Head Dataset?

To train a YOLO26n model on the Global Wheat Head Dataset, you can use the following code snippets. Make sure you have the `GlobalWheat2020.yaml` configuration file specifying dataset paths and classes:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model (recommended for training)
        model = YOLO("yolo26n.pt")

        # Train the model
        results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=GlobalWheat2020.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

### How do I download the Global Wheat Head Dataset?

The dataset (~7.0 GB) downloads automatically the first time you train with `data="GlobalWheat2020.yaml"` — no manual step is required. Ultralytics fetches the images and labels and unpacks them to your local datasets directory. You can browse related datasets in the [detection datasets overview](index.md).

### Where can I find the configuration YAML file for the Global Wheat Head Dataset?

The configuration YAML file for the Global Wheat Head Dataset, named `GlobalWheat2020.yaml`, is available on GitHub. You can access it at <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/GlobalWheat2020.yaml>. This file contains the dataset paths, classes, and other configuration details needed for model training in [Ultralytics YOLO](../../models/yolo26.md).

### Why is wheat head detection important in crop management?

Wheat head detection is critical in crop management because it enables accurate estimation of wheat head density and size, which are essential for evaluating crop health, maturity, and yield potential. By leveraging [deep learning models](../../models/index.md) trained on datasets like the Global Wheat Head Dataset, farmers and researchers can better monitor and manage crops, leading to improved productivity and optimized resource use in agricultural practices. This technological advancement supports [sustainable agriculture](https://www.ultralytics.com/blog/real-time-crop-health-monitoring-with-ultralytics-yolo11) and food security initiatives.

For more information on applications of AI in agriculture, visit [AI in Agriculture](https://www.ultralytics.com/solutions/computer-vision-in-agriculture).
