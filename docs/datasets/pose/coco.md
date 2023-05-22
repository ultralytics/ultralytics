---
comments: true
description: Learn about the COCO-Pose dataset, designed to encourage research on pose estimation tasks with standardized evaluation metrics.
---

# COCO-Pose Dataset

The [COCO-Pose](https://cocodataset.org/#keypoints-2017) dataset is a specialized version of the COCO (Common Objects in Context) dataset, designed for pose estimation tasks. It leverages the COCO Keypoints 2017 images and labels to enable the training of models like YOLO for pose estimation tasks.

![Pose sample image](https://user-images.githubusercontent.com/26833433/239691398-d62692dc-713e-4207-9908-2f6710050e5c.jpg)

## Key Features

- COCO-Pose builds upon the COCO Keypoints 2017 dataset which contains 200K images labeled with keypoints for pose estimation tasks.
- The dataset supports 17 keypoints for human figures, facilitating detailed pose estimation.
- Like COCO, it provides standardized evaluation metrics, including Object Keypoint Similarity (OKS) for pose estimation tasks, making it suitable for comparing model performance.

## Dataset Structure

The COCO-Pose dataset is split into three subsets:

1. **Train2017**: This subset contains a portion of the 118K images from the COCO dataset, annotated for training pose estimation models.
2. **Val2017**: This subset has a selection of images used for validation purposes during model training.
3. **Test2017**: This subset consists of images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the [COCO evaluation server](https://competitions.codalab.org/competitions/5181) for performance evaluation.

## Applications

The COCO-Pose dataset is specifically used for training and evaluating deep learning models in keypoint detection and pose estimation tasks, such as OpenPose. The dataset's large number of annotated images and standardized evaluation metrics make it an essential resource for computer vision researchers and practitioners focused on pose estimation.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the COCO-Pose dataset, the `coco-pose.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/coco-pose.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/coco-pose.yaml).

!!! example "ultralytics/datasets/coco-pose.yaml"

    ```yaml
    --8<-- "ultralytics/datasets/coco-pose.yaml"
    ```

## Usage

To train a YOLOv8n-pose model on the COCO-Pose dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
        
        # Train the model
        model.train(data='coco-pose.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=coco-pose.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The COCO-Pose dataset contains a diverse set of images with human figures annotated with keypoints. Here are some examples of images from the dataset, along with their corresponding annotations:

![Dataset sample image](https://user-images.githubusercontent.com/26833433/239690150-a9dc0bd0-7ad9-4b78-a30f-189ed727ea0e.jpg)

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the COCO-Pose dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the COCO-Pose dataset in your research or development work, please cite the following paper:

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

We would like to acknowledge the COCO Consortium for creating and maintaining this valuable resource for the computer vision community. For more information about the COCO-Pose dataset and its creators, visit the [COCO dataset website](https://cocodataset.org/#home).