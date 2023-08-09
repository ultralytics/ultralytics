---
comments: true
description: Explore the possibilities of the COCO-Seg dataset, designed for object instance segmentation and YOLO model training. Discover key features, dataset structure, applications, and usage.
keywords: Ultralytics, YOLO, COCO-Seg, dataset, instance segmentation, model training, deep learning, computer vision
---

# COCO-Seg Dataset

The [COCO-Seg](https://cocodataset.org/#home) dataset, an extension of the COCO (Common Objects in Context) dataset, is specially designed to aid research in object instance segmentation. It uses the same images as COCO but introduces more detailed segmentation annotations. This dataset is a crucial resource for researchers and developers working on instance segmentation tasks, especially for training YOLO models.

## Key Features

- COCO-Seg retains the original 330K images from COCO.
- The dataset consists of the same 80 object categories found in the original COCO dataset.
- Annotations now include more detailed instance segmentation masks for each object in the images.
- COCO-Seg provides standardized evaluation metrics like mean Average Precision (mAP) for object detection, and mean Average Recall (mAR) for instance segmentation tasks, enabling effective comparison of model performance.

## Dataset Structure

The COCO-Seg dataset is partitioned into three subsets:

1. **Train2017**: This subset contains 118K images for training instance segmentation models.
2. **Val2017**: This subset includes 5K images used for validation purposes during model training.
3. **Test2017**: This subset encompasses 20K images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the [COCO evaluation server](https://codalab.lisn.upsaclay.fr/competitions/7383) for performance evaluation.

## Applications

COCO-Seg is widely used for training and evaluating deep learning models in instance segmentation, such as the YOLO models. The large number of annotated images, the diversity of object categories, and the standardized evaluation metrics make it an indispensable resource for computer vision researchers and practitioners.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO-Seg dataset, the `coco.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml).

!!! example "ultralytics/cfg/datasets/coco.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco.yaml"
    ```

## Usage

To train a YOLOv8n-seg model on the COCO-Seg dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

        # Train the model
        model.train(data='coco-seg.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco-seg.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

COCO-Seg, like its predecessor COCO, contains a diverse set of images with various object categories and complex scenes. However, COCO-Seg introduces more detailed instance segmentation masks for each object in the images. Here are some examples of images from the dataset, along with their corresponding instance segmentation masks:

![Dataset sample image](https://user-images.githubusercontent.com/26833433/239690696-93fa8765-47a2-4b34-a6e5-516d0d1c725b.jpg)

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This aids the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the COCO-Seg dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the COCO-Seg dataset in your research or development work, please cite the original COCO paper and acknowledge the extension to COCO-Seg:

```bibtex
@misc{lin2015microsoft,
      title={Microsoft COCO: Common Objects in Context},
      author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
      year={2015},
      eprint={1405.0312},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

We extend our thanks to the COCO Consortium for creating and maintaining this invaluable resource for the computer vision community. For more information about the COCO dataset and its creators, visit the [COCO dataset website](https://cocodataset.org/#home).
