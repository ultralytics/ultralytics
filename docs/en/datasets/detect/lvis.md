---
comments: true
description: Learn how LVIS, a leading dataset for object detection and segmentation, integrates with Ultralytics. Discover ways to use it for training YOLO models.
keywords: Ultralytics, LVIS dataset, object detection, YOLO, YOLO model training, image segmentation, computer vision, deep learning models
---

# LVIS Dataset

The [LVIS](https://www.lvisdataset.org/dataset) dataset is a large-scale, fine-grained vocabulary-level annotation dataset developed and released by Facebook AI Research (FAIR). It is primarily used as a research benchmark for object detection and instance segmentation with a large vocabulary of categories, aiming to drive further advancements in computer vision field.

## Key Features

- LVIS contains 160k images and 2M instance annotations for object detection, segmentation, and captioning tasks.
- The dataset comprises 1203 object categories, including common objects like cars, bicycles, and animals, as well as more specific categories such as umbrellas, handbags, and sports equipment.
- Annotations include object bounding boxes, segmentation masks, and captions for each image.
- LVIS provides standardized evaluation metrics like mean Average Precision (mAP) for object detection, and mean Average Recall (mAR) for segmentation tasks, making it suitable for comparing model performance.
- LVIS uses the exactly the same images as [COCO](./coco.md) dataset, but with different splits and different annotations.

## Dataset Structure

The LVIS dataset is split into three subsets:

1. **Train**: This subset contains 100k images for training object detection, segmentation, and captioning models.
2. **Val**: This subset has 20k images used for validation purposes during model training.
3. **Minival**: This subset is exactly the same as COCO val2017 set which has 5k images used for validation purposes during model training.
4. **Test**: This subset consists of 20k images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the [LVIS evaluation server](https://eval.ai/web/challenges/challenge-page/675/overview) for performance evaluation.


## Applications

The LVIS dataset is widely used for training and evaluating deep learning models in object detection (such as YOLO, Faster R-CNN, and SSD), instance segmentation (such as Mask R-CNN). The dataset's diverse set of object categories, large number of annotated images, and standardized evaluation metrics make it an essential resource for computer vision researchers and practitioners.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the LVIS dataset, the `lvis.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml).

!!! Example "ultralytics/cfg/datasets/lvis.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/lvis.yaml"
    ```

## Usage

To train a YOLOv8n model on the LVIS dataset for 100 epochs with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='lvis.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=lvis.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The LVIS dataset contains a diverse set of images with various object categories and complex scenes. Here are some examples of images from the dataset, along with their corresponding annotations:

![Dataset sample image](https://private-user-images.githubusercontent.com/61612323/316485965-a88c2e62-58d0-4f67-bc69-1418e42175e9.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTEzNjcyNjYsIm5iZiI6MTcxMTM2Njk2NiwicGF0aCI6Ii82MTYxMjMyMy8zMTY0ODU5NjUtYTg4YzJlNjItNThkMC00ZjY3LWJjNjktMTQxOGU0MjE3NWU5LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzI1VDExNDI0NlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZmMTVlNzE5MTBkOTZmNDQwNzJjNWQzYzM2NmEyMGMxODQ4ZDEyMjYwYmMyY2JjZDU5YzBmMDIyZGEwMGEwZDAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.7thukPdnJKYuBmTk1ROUyqxxV3Ix5GeNLqyi4wSDYvA)


- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the LVIS dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the LVIS dataset in your research or development work, please cite the following paper:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{gupta2019lvis,
          title={{LVIS}: A Dataset for Large Vocabulary Instance Segmentation},
          author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
          booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
          year={2019}
        }
        ```

We would like to acknowledge the LVIS Consortium for creating and maintaining this valuable resource for the computer vision community. For more information about the LVIS dataset and its creators, visit the [LVIS dataset website](https://www.lvisdataset.org/dataset).
