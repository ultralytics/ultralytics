---
comments: true
description: Explore the DOTA8 dataset - a small, versatile oriented object detection dataset ideal for testing and debugging object detection models using Ultralytics YOLO11.
keywords: DOTA8 dataset, Ultralytics, YOLO11, object detection, debugging, training models, oriented object detection, dataset YAML
---

# Cityscapes Dataset

## Introduction

Cityscapes dataset for YOLO semseg task is a largescale [semantic segment](https://www.ultralytics.com/glossary/semantic-segmentation) dataset composed of
5002 images and 20,000 annotation files. All of the images and files are obtained from about 50 cities. The annotation files contain color anotation image,
instance ID annotation image, class ID annotation image and polygon json file. In YOLO project, there is color annotation file just adopt for taining, validation and test.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the Cityscape dataset, the `CityEscapeYOLO.yaml` file is maintained at [https://github.com/kuazhangxiaoai/ultralytics-semantic-segment/blob/semseg/ultralytics/cfg/datasets/CityEscapeYOLO.yaml](https://github.com/kuazhangxiaoai/ultralytics-semantic-segment/blob/semseg/ultralytics/cfg/datasets/CityEscapeYOLO.yaml).

!!! example "ultralytics/cfg/datasets/CityEscapeYOLO.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/CityEscapeYOLO.yaml"
    ```

## Usage

To train a YOLO11n-semseg model on the Cityscaps dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 512, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-semseg.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="CityEscapeYOLO.yaml", epochs=100, imgsz=512)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo obb train data=CityEscapeYOLO.yaml model=yolo11n-semseg.pt epochs=100 imgsz=512
        ```

## Sample Images and Annotations

Here are some examples of images from the DOTA8 dataset, along with their corresponding annotations:

<img src="https://release-assets.githubusercontent.com/github-production-release-asset/1046723287/e2f15ea4-7ac9-40b7-b49a-b964748dcd4b?sp=r&sv=2018-11-09&sr=b&spr=https&se=2025-10-28T09%3A15%3A40Z&rscd=attachment%3B+filename%3Dmosaic.png&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2025-10-28T08%3A15%3A00Z&ske=2025-10-28T09%3A15%3A40Z&sks=b&skv=2018-11-09&sig=TInr7KgigLtpG2Lx%2B2taJ1q8UCTfPmmS0J527TATacE%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc2MTY0MDQ5NywibmJmIjoxNzYxNjQwMTk3LCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.jpx46c60iF4YoJseUNorW94ZUyAk3-LzQ6iJ6pe1t1U&response-content-disposition=attachment%3B%20filename%3Dmosaic.png&response-content-type=application%2Foctet-stream" alt="Dataset sample image" width="800">

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the DOTA8 dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the DOTA dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{Cordts2016Cityscapes,
            title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
            author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
            booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            year={2016}
        ```

A special note of gratitude to the team behind the DOTA datasets for their commendable effort in curating this dataset. For an exhaustive understanding of the dataset and its nuances, please visit the [official DOTA website](https://captain-whu.github.io/DOTA/index.html).

## FAQ

### What is the Cityscaps dataset and how can it be used?

The Cityscaps dataset is a largescale dataset captured from about 50 cities, which is designed for semantic and instance segment task.
The dataset contains 5002 fine-annotated images and their annotation files including color annotaion image, class ID annotation image,
instance ID image and polygon json file. For usage of YOLO semseg task, the image and color annotation image is adopt for taining, validation, and test.
Learn more about object detection with [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).

### How do I train a YOLO11 model using the Cityscaps dataset?

To train a YOLO11n-semseg model on the Cityscape dataset for 100 epochs with an image size of 512, you can use the following code snippets. For comprehensive argument options, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-semseg.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="CityEscapeYOLO.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo obb train data=CityEscapeYOLO.yaml model=yolo11n-semseg.pt epochs=100 imgsz=640
        ```

### What are the key features of the Cityscaps dataset and where can I access the YAML file?

The Cityscaps dataset is known for its large-scale benchmark and the challenges it presents for semantic segment. You can access the `CityEscapeYOLO.yaml` file, which contains paths, classes, and configuration details, at this [GitHub link](https://github.com/kuazhangxiaoai/ultralytics-semantic-segment/blob/semseg/ultralytics/cfg/datasets/CityEscapeYOLO.yaml).

### How do I convert the official format of Cityscape to YOLO format of it?

We designed the function, [`CityEcapse2YOLO`](../../reference/data/converter.md), to convert the offical format of Cityscapes dataset to YOLO format of it. This conversion ensures compatibility with the Ultralytics YOLO models
Here's a quick example:

```python
from ultralytics.data.converter import CityEcapse2YOLO

CityEcapse2YOLO("path/to/Cityscapes", "path/to/CityscapsYOLO")
```

### Why should I use Ultralytics YOLO11 for semantic segment tasks?

Ultralytics YOLO11 provides state-of-the-art real-time object detection capabilities, including features like oriented bounding boxes (OBB), [semantic segmentation](https://www.ultralytics.com/glossary/semantic-segmentation), and a highly versatile training pipeline. It's suitable for various applications and offers pretrained models for efficient fine-tuning. Explore further about the advantages and usage in the [Ultralytics YOLO11 documentation](https://github.com/ultralytics/ultralytics).
