---
comments: true
description: Discover HomeObjects-3K, a rich indoor object detection dataset with 12 classes like bed, sofa, TV, and laptop. Ideal for computer vision in smart homes, robotics, and AR.
keywords: HomeObjects-3K, indoor dataset, household items, object detection, computer vision, YOLO11, smart home AI, robotics dataset
---

# HomeObjects-3K Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-homeobjects-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="HomeObjects-3K Dataset In Colab"></a>

The HomeObjects-3K dataset is a curated collection of common household object images, designed for training, testing, and [benchmarking](../../modes/benchmark.md) [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models. Featuring ~3,000 images and 12 distinct object classes, this dataset is ideal for research and applications in indoor scene understanding, smart home devices, [robotics](https://www.ultralytics.com/glossary/robotics), and augmented reality.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/v3iqOYoRBFQ"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO11 on HomeObjects-3K Dataset | Detection, Validation & ONNX Export 🚀
</p>

## Dataset Structure

The HomeObjects-3K dataset is organized into the following subsets:

- **Training Set**: Comprises 2,285 annotated images featuring objects such as sofas, chairs, tables, lamps, and more.
- **Validation Set**: Includes 404 annotated images designated for evaluating model performance.

Each image is labeled using bounding boxes aligned with the [Ultralytics YOLO](../detect/index.md/#what-is-the-ultralytics-yolo-dataset-format-and-how-to-structure-it) format. The diversity of indoor lighting, object scale, and orientations makes it robust for real-world deployment scenarios.

## Object Classes

The dataset supports 12 everyday object categories, covering furniture, electronics, and decorative items. These classes are chosen to reflect common items encountered in indoor domestic environments and support vision tasks like [object detection](../../tasks/detect.md) and [object tracking](../../modes/track.md).

!!! Tip "HomeObjects-3K classes"

    0. bed
    1. sofa
    2. chair
    3. table
    4. lamp
    5. tv
    6. laptop
    7. wardrobe
    8. window
    9. door
    10. potted plant
    11. photo frame

## Applications

HomeObjects-3K enables a wide spectrum of applications in indoor computer vision, spanning both research and real-world product development:

- **Indoor object detection**: Use models like [Ultralytics YOLO11](../../models/yolo11.md) to find and locate common home items like beds, chairs, lamps, and laptops in images. This helps with real-time understanding of indoor scenes.

- **Scene layout parsing**: In robotics and smart home systems, this helps devices understand how rooms are arranged, where objects like doors, windows, and furniture are, so they can navigate safely and interact with their environment properly.

- **AR applications**: Power [object recognition](http://ultralytics.com/glossary/image-recognition) features in apps that use augmented reality. For example, detect TVs or wardrobes and show extra information or effects on them.

- **Education and research**: Support learning and academic projects by giving students and researchers a ready-to-use dataset for practicing indoor object detection with real-world examples.

- **Home inventory and asset tracking**: Automatically detect and list home items in photos or videos, useful for managing belongings, organizing spaces, or visualizing furniture in real estate.

## Dataset YAML

The configuration for the HomeObjects-3K dataset is provided through a YAML file. This file outlines essential information such as image paths for train and validation directories, and the list of object classes.
You can access the `HomeObjects-3K.yaml` file directly from the Ultralytics repository at: [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/HomeObjects-3K.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/HomeObjects-3K.yaml)

!!! example "ultralytics/cfg/datasets/HomeObjects-3K.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/HomeObjects-3K.yaml"
    ```

## Usage

You can train a YOLO11n model on the HomeObjects-3K dataset for 100 epochs using an image size of 640. The examples below show how to get started. For more training options and detailed settings, check the [Training](../../modes/train.md) guide.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load pretrained model
        model = YOLO("yolo11n.pt")

        # Train the model on HomeObjects-3K dataset
        model.train(data="HomeObjects-3K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=HomeObjects-3K.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The dataset features a rich collection of indoor scene images that capture a wide range of household objects in natural home environments. Below are sample visuals from the dataset, each paired with its corresponding annotations to illustrate object positions, scales, and spatial relationships.

![HomeObjects-3K dataset sample image, highlighting different objects i.e, beds, chair, door, sofas, and plants](https://github.com/ultralytics/docs/releases/download/0/homeobjects-3k-dataset-sample.avif)

## License and Attribution

HomeObjects-3K is developed and released by the **[Ultralytics team](https://www.ultralytics.com/about)** under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), supporting open-source research and commercial use with proper attribution.

If you use this dataset in your research, please cite it using the mentioned details:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Jocher_Ultralytics_Datasets_2025,
            author = {Jocher, Glenn and Rizwan, Muhammad},
            license = {AGPL-3.0},
            month = {May},
            title = {Ultralytics Datasets: HomeObjects-3K Detection Dataset},
            url = {https://docs.ultralytics.com/datasets/detect/homeobject-3k/},
            version = {1.0.0},
            year = {2025}
        }
        ```

## FAQ

### What is the HomeObjects-3K dataset designed for?

HomeObjects-3K is crafted for advancing AI understanding of indoor scenes. It focuses on detecting everyday household items—like beds, sofas, TVs, and lamps—making it ideal for applications in smart homes, robotics, augmented reality, and interior monitoring systems. Whether you're training models for real-time edge devices or academic research, this dataset provides a balanced foundation.

### Which object categories are included, and why were they selected?

The dataset includes 12 of the most commonly encountered household items: bed, sofa, chair, table, lamp, tv, laptop, wardrobe, window, door, potted plant, and photo frame. These objects were chosen to reflect realistic indoor environments and to support multipurpose tasks such as robotic navigation, or scene generation in AR/VR applications.

### How can I train a YOLO model using the HomeObjects-3K dataset?

To train a YOLO model like YOLO11n, you'll just need the `HomeObjects-3K.yaml` configuration file and the [pretrained model](../../models/index.md) weights. Whether you're using Python or the CLI, training can be launched with a single command. You can customize parameters such as epochs, image size, and batch size depending on your target performance and hardware setup.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load pretrained model
        model = YOLO("yolo11n.pt")

        # Train the model on HomeObjects-3K dataset
        model.train(data="HomeObjects-3K.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=HomeObjects-3K.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

### Is this dataset suitable for beginner-level projects?

Absolutely. With clean labeling, and standardized YOLO-compatible annotations, HomeObjects-3K is an excellent entry point for students and hobbyists who want to explore real-world object detection in indoor scenarios. It also scales well for more complex applications in commercial environments.

### Where can I find the annotation format and YAML?

Refer to the [Dataset YAML](#dataset-yaml) section. The format is standard YOLO, making it compatible with most object detection pipelines.
