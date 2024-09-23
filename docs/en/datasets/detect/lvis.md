---
comments: true
description: Discover the LVIS dataset by Facebook AI Research, a benchmark for object detection and instance segmentation with a large, diverse vocabulary. Learn how to utilize it.
keywords: LVIS dataset, object detection, instance segmentation, Facebook AI Research, YOLO, computer vision, model training, LVIS examples
---

# LVIS Dataset

The [LVIS dataset](https://www.lvisdataset.org/) is a large-scale, fine-grained vocabulary-level annotation dataset developed and released by Facebook AI Research (FAIR). It is primarily used as a research benchmark for object detection and [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) with a large vocabulary of categories, aiming to drive further advancements in computer vision field.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/cfTKj96TjSE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> YOLO World training workflow with LVIS dataset
</p>

<p align="center">
    <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/lvis-dataset-example-images.avif" alt="LVIS Dataset example images">
</p>

## Key Features

- LVIS contains 160k images and 2M instance annotations for object detection, segmentation, and captioning tasks.
- The dataset comprises 1203 object categories, including common objects like cars, bicycles, and animals, as well as more specific categories such as umbrellas, handbags, and sports equipment.
- Annotations include object bounding boxes, segmentation masks, and captions for each image.
- LVIS provides standardized evaluation metrics like [mean Average Precision](https://www.ultralytics.com/glossary/mean-average-precision-map) (mAP) for object detection, and mean Average [Recall](https://www.ultralytics.com/glossary/recall) (mAR) for segmentation tasks, making it suitable for comparing model performance.
- LVIS uses exactly the same images as [COCO](./coco.md) dataset, but with different splits and different annotations.

## Dataset Structure

The LVIS dataset is split into three subsets:

1. **Train**: This subset contains 100k images for training object detection, segmentation, and captioning models.
2. **Val**: This subset has 20k images used for validation purposes during model training.
3. **Minival**: This subset is exactly the same as COCO val2017 set which has 5k images used for validation purposes during model training.
4. **Test**: This subset consists of 20k images used for testing and benchmarking the trained models. Ground truth annotations for this subset are not publicly available, and the results are submitted to the [LVIS evaluation server](https://eval.ai/web/challenges/challenge-page/675/overview) for performance evaluation.

## Applications

The LVIS dataset is widely used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in object detection (such as YOLO, Faster R-CNN, and SSD), instance segmentation (such as Mask R-CNN). The dataset's diverse set of object categories, large number of annotated images, and standardized evaluation metrics make it an essential resource for computer vision researchers and practitioners.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the LVIS dataset, the `lvis.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml).

!!! example "ultralytics/cfg/datasets/lvis.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/lvis.yaml"
    ```

## Usage

To train a YOLOv8n model on the LVIS dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="lvis.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=lvis.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The LVIS dataset contains a diverse set of images with various object categories and complex scenes. Here are some examples of images from the dataset, along with their corresponding annotations:

![LVIS Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/lvis-mosaiced-training-batch.avif)

- **Mosaiced Image**: This image demonstrates a training batch composed of mosaiced dataset images. Mosaicing is a technique used during training that combines multiple images into a single image to increase the variety of objects and scenes within each training batch. This helps improve the model's ability to generalize to different object sizes, aspect ratios, and contexts.

The example showcases the variety and complexity of the images in the LVIS dataset and the benefits of using mosaicing during the training process.

## Citations and Acknowledgments

If you use the LVIS dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{gupta2019lvis,
          title={LVIS: A Dataset for Large Vocabulary Instance Segmentation},
          author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
          booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
          year={2019}
        }
        ```

We would like to acknowledge the LVIS Consortium for creating and maintaining this valuable resource for the [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) community. For more information about the LVIS dataset and its creators, visit the [LVIS dataset website](https://www.lvisdataset.org/).

## FAQ

### What is the LVIS dataset, and how is it used in computer vision?

The [LVIS dataset](https://www.lvisdataset.org/) is a large-scale dataset with fine-grained vocabulary-level annotations developed by Facebook AI Research (FAIR). It is primarily used for object detection and instance segmentation, featuring over 1203 object categories and 2 million instance annotations. Researchers and practitioners use it to train and benchmark models like Ultralytics YOLO for advanced computer vision tasks. The dataset's extensive size and diversity make it an essential resource for pushing the boundaries of model performance in detection and segmentation.

### How can I train a YOLOv8n model using the LVIS dataset?

To train a YOLOv8n model on the LVIS dataset for 100 epochs with an image size of 640, follow the example below. This process utilizes Ultralytics' framework, which offers comprehensive training features.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="lvis.yaml", epochs=100, imgsz=640)
        ```


    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=lvis.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

For detailed training configurations, refer to the [Training](../../modes/train.md) documentation.

### How does the LVIS dataset differ from the COCO dataset?

The images in the LVIS dataset are the same as those in the [COCO dataset](./coco.md), but the two differ in terms of splitting and annotations. LVIS provides a larger and more detailed vocabulary with 1203 object categories compared to COCO's 80 categories. Additionally, LVIS focuses on annotation completeness and diversity, aiming to push the limits of [object detection](https://www.ultralytics.com/glossary/object-detection) and instance segmentation models by offering more nuanced and comprehensive data.

### Why should I use Ultralytics YOLO for training on the LVIS dataset?

Ultralytics YOLO models, including the latest YOLOv8, are optimized for real-time object detection with state-of-the-art [accuracy](https://www.ultralytics.com/glossary/accuracy) and speed. They support a wide range of annotations, such as the fine-grained ones provided by the LVIS dataset, making them ideal for advanced computer vision applications. Moreover, Ultralytics offers seamless integration with various [training](../../modes/train.md), [validation](../../modes/val.md), and [prediction](../../modes/predict.md) modes, ensuring efficient model development and deployment.

### Can I see some sample annotations from the LVIS dataset?

Yes, the LVIS dataset includes a variety of images with diverse object categories and complex scenes. Here is an example of a sample image along with its annotations:

![LVIS Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/lvis-mosaiced-training-batch.avif)

This mosaiced image demonstrates a training batch composed of multiple dataset images combined into one. Mosaicing increases the variety of objects and scenes within each training batch, enhancing the model's ability to generalize across different contexts. For more details on the LVIS dataset, explore the [LVIS dataset documentation](#key-features).
