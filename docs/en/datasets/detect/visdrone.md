---
comments: true
description: Explore the VisDrone Dataset, a large-scale benchmark for drone-based image and video analysis with over 2.6 million annotations for objects like pedestrians and vehicles.
keywords: VisDrone, drone dataset, computer vision, object detection, object tracking, crowd counting, machine learning, deep learning
---

# VisDrone Dataset

The [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) is a large-scale benchmark created by the AISKYEYE team at the Lab of [Machine Learning](https://www.ultralytics.com/glossary/machine-learning-ml) and Data Mining, Tianjin University, China. It contains carefully annotated ground truth data for various computer vision tasks related to drone-based image and video analysis.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/9ymyH4H1fG4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO11 on the VisDrone Dataset | Aerial Detection | Complete Tutorial 🚀
</p>

VisDrone is composed of 288 video clips with 261,908 frames and 10,209 static images, captured by various drone-mounted cameras. The dataset covers a wide range of aspects, including location (14 different cities across China), environment (urban and rural), objects (pedestrians, vehicles, bicycles, etc.), and density (sparse and crowded scenes). The dataset was collected using various drone platforms under different scenarios and weather and lighting conditions. These frames are manually annotated with over 2.6 million bounding boxes of targets such as pedestrians, cars, bicycles, and tricycles. Attributes like scene visibility, object class, and occlusion are also provided for better data utilization.

## Dataset Structure

The VisDrone dataset is organized into five main subsets, each focusing on a specific task:

1. **Task 1**: Object detection in images
2. **Task 2**: Object detection in videos
3. **Task 3**: Single-object tracking
4. **Task 4**: [Multi-object tracking](../index.md#multi-object-tracking)
5. **Task 5**: Crowd counting

## Applications

The VisDrone dataset is widely used for training and evaluating deep learning models in drone-based [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks such as object detection, object tracking, and crowd counting. The dataset's diverse set of sensor data, object annotations, and attributes make it a valuable resource for researchers and practitioners in the field of drone-based computer vision.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the Visdrone dataset, the `VisDrone.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml).

!!! example "ultralytics/cfg/datasets/VisDrone.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/VisDrone.yaml"
    ```

## Usage

To train a YOLO11n model on the VisDrone dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=VisDrone.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## Sample Data and Annotations

The VisDrone dataset contains a diverse set of images and videos captured by drone-mounted cameras. Here are some examples of data from the dataset, along with their corresponding annotations:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/visdrone-object-detection-sample.avif)

- **Task 1**: [Object detection](https://www.ultralytics.com/glossary/object-detection) in images - This image demonstrates an example of object detection in images, where objects are annotated with [bounding boxes](https://www.ultralytics.com/glossary/bounding-box). The dataset provides a wide variety of images taken from different locations, environments, and densities to facilitate the development of models for this task.

The example showcases the variety and complexity of the data in the VisDrone dataset and highlights the importance of high-quality sensor data for drone-based computer vision tasks.

## Citations and Acknowledgments

If you use the VisDrone dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @ARTICLE{9573394,
          author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Detection and Tracking Meet Drones Challenge},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3119563}}
        ```

We would like to acknowledge the AISKYEYE team at the Lab of Machine Learning and [Data Mining](https://www.ultralytics.com/glossary/data-mining), Tianjin University, China, for creating and maintaining the VisDrone dataset as a valuable resource for the drone-based computer vision research community. For more information about the VisDrone dataset and its creators, visit the [VisDrone Dataset GitHub repository](https://github.com/VisDrone/VisDrone-Dataset).

## FAQ

### What is the VisDrone Dataset and what are its key features?

The [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset) is a large-scale benchmark created by the AISKYEYE team at Tianjin University, China. It is designed for various computer vision tasks related to drone-based image and video analysis. Key features include:

- **Composition**: 288 video clips with 261,908 frames and 10,209 static images.
- **Annotations**: Over 2.6 million bounding boxes for objects like pedestrians, cars, bicycles, and tricycles.
- **Diversity**: Collected across 14 cities, in urban and rural settings, under different weather and lighting conditions.
- **Tasks**: Split into five main tasks—object detection in images and videos, single-object and multi-object tracking, and crowd counting.

### How can I use the VisDrone Dataset to train a YOLO11 model with Ultralytics?

To train a YOLO11 model on the VisDrone dataset for 100 epochs with an image size of 640, you can follow these steps:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained model
        model = YOLO("yolo11n.pt")

        # Train the model
        results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=VisDrone.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

For additional configuration options, please refer to the model [Training](../../modes/train.md) page.

### What are the main subsets of the VisDrone dataset and their applications?

The VisDrone dataset is divided into five main subsets, each tailored for a specific computer vision task:

1. **Task 1**: Object detection in images.
2. **Task 2**: Object detection in videos.
3. **Task 3**: Single-object tracking.
4. **Task 4**: Multi-object tracking.
5. **Task 5**: Crowd counting.

These subsets are widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in drone-based applications such as surveillance, traffic monitoring, and public safety.

### Where can I find the configuration file for the VisDrone dataset in Ultralytics?

The configuration file for the VisDrone dataset, `VisDrone.yaml`, can be found in the Ultralytics repository at the following link:
[VisDrone.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml).

### How can I cite the VisDrone dataset if I use it in my research?

If you use the VisDrone dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @ARTICLE{9573394,
          author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Detection and Tracking Meet Drones Challenge},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3119563}
        }
        ```
