---
comments: true
description: Delve into DOTA v2, a distinct dataset focusing on object detection in aerial imagery. Understand its synergy with advanced machine learning models.
keywords: DOTA v2, object detection, aerial images, computer vision, deep learning, annotations, oriented bounding boxes, OBB
---

# DOTA v2 Dataset with OBB

[DOTA v2](https://captain-whu.github.io/DOTA/index.html) stands as a specialized dataset, emphasizing object detection in aerial images. Pioneered by the collaborative efforts of academic researchers, it encapsulates a rich array of aerial scenes annotated meticulously with Oriented Bounding Boxes (OBB).

![DOTA v2 classes visual](https://user-images.githubusercontent.com/26833433/259461765-72fdd0d8-266b-44a9-8199-199329bf5ca9.jpg)

## Key Features

- Comprises a diverse set of aerial images, showcasing urban scenes, agricultural landscapes, and intricate infrastructural patterns.
- Features more than 1M Oriented Bounding Boxes across 18 categories, hand-crafted with precision to capture objects of varying sizes and orientations.
- The presence of OBB annotations provides a unique challenge, differentiating DOTA v2 from datasets with axis-aligned bounding boxes.
- Encompasses multiscale object detection, recognizing objects that may appear minuscule or colossal based on their altitude.
- Offers a balanced distribution across categories, ensuring robust model training without inherent biases.

## Dataset Structure

DOTA v2 exhibits a structured layout tailored for OBB object detection challenges:

- **Images**: A vast collection of high-resolution aerial images capturing diverse terrains and structures.
- **Oriented Bounding Boxes**: Annotations in the form of rotated rectangles encapsulating objects irrespective of their orientation, ideal for capturing objects like airplanes, ships, and buildings.

## Applications

DOTA v2 serves as a benchmark for training and evaluating models specifically tailored for aerial image analysis. With the inclusion of OBB annotations, it provides a unique challenge, enabling the development of specialized object detection models that cater to aerial imagery's nuances.

## Dataset YAML

Typically, datasets incorporate a YAML (Yet Another Markup Language) file detailing the dataset's configuration. For DOTA v2, a hypothetical `DOTAv2.yaml` could be used. For accurate paths and configurations, it's vital to consult the dataset's official repository or documentation.

!!! example "DOTAv2.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/DOTAv2.yaml"
    ```

## Usage

To train a model on the DOTA v2 dataset, you can utilize the following code snippets. Always refer to your model's documentation for a thorough list of available arguments.

!!! warning

    Please be aware that all images and associated annotations in the DOTAv2 dataset can be used for academic purposes only, but any commercial use is prohibited. Your understanding and respect for the dataset creators' wishes are greatly appreciated!


!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a new YOLOv8n-OBB model from scratch
        model = YOLO('yolov8n-obb.yaml')

        # Train the model on the DOTAv2 dataset
        results = model.train(data='DOTAv2.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train a new YOLOv8n-OBB model on the DOTAv2 dataset
        yolo detect train data=DOTAv2.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

Having a glance at the dataset illustrates its depth:

![Dataset sample image](https://captain-whu.github.io/DOTA/images/instances-DOTA.jpg)

- **DOTA v2**: This snapshot underlines the complexity of aerial scenes and the significance of Oriented Bounding Box annotations, capturing objects in their natural orientation.

The dataset's richness offers invaluable insights into object detection challenges exclusive to aerial imagery.

## Citations and Acknowledgments

For those leveraging DOTA v2 in their endeavors, it's pertinent to cite the relevant research papers:

```bibtex
@article{9560031,
  author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3117983}
}
```

A special note of gratitude to the team behind DOTA v2 for their commendable effort in curating this dataset. For an exhaustive understanding of the dataset and its nuances, please visit the [official DOTA v2 website](https://captain-whu.github.io/DOTA/index.html).