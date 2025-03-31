---
comments: true
description: Explore the Caltech-256 dataset, featuring 30,000 images across 257 categories, ideal for training and testing object recognition algorithms.
keywords: Caltech-256 dataset, object classification, image dataset, machine learning, computer vision, deep learning, YOLO, training dataset
---

# Caltech-256 Dataset

The [Caltech-256](https://data.caltech.edu/records/nyy15-4j048) dataset is an extensive collection of images used for object classification tasks. It contains around 30,000 images divided into 257 categories (256 object categories and 1 background category). The images are carefully curated and annotated to provide a challenging and diverse benchmark for object recognition algorithms.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/isc06_9qnM0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train <a href="https://www.ultralytics.com/glossary/image-classification">Image Classification</a> Model using Caltech-256 Dataset with Ultralytics HUB
</p>

## Key Features

- The Caltech-256 dataset comprises around 30,000 color images divided into 257 categories.
- Each category contains a minimum of 80 images.
- The categories encompass a wide variety of real-world objects, including animals, vehicles, household items, and people.
- Images are of variable sizes and resolutions.
- Caltech-256 is widely used for training and testing in the field of machine learning, particularly for object recognition tasks.

## Dataset Structure

Like [Caltech-101](../classify/caltech101.md), the Caltech-256 dataset does not have a formal split between training and testing sets. Users typically create their own splits according to their specific needs. A common practice is to use a random subset of images for training and the remaining images for testing.

## Applications

The Caltech-256 dataset is extensively used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in object recognition tasks, such as [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs), [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs), and various other machine learning algorithms. Its diverse set of categories and high-quality images make it an invaluable dataset for research and development in the field of machine learning and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).

## Usage

To train a YOLO model on the Caltech-256 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch), you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="caltech256", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech256 model=yolo11n-cls.pt epochs=100 imgsz=416
        ```

## Sample Images and Annotations

The Caltech-256 dataset contains high-quality color images of various objects, providing a comprehensive dataset for object recognition tasks. Here are some examples of images from the dataset ([credit](https://ml4a.github.io/demos/tsne_viewer.html)):

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/caltech256-sample-image.avif)

The example showcases the diversity and complexity of the objects in the Caltech-256 dataset, emphasizing the importance of a varied dataset for training robust object recognition models.

## Citations and Acknowledgments

If you use the Caltech-256 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{griffin2007caltech,
                 title={Caltech-256 object category dataset},
                 author={Griffin, Gregory and Holub, Alex and Perona, Pietro},
                 year={2007}
        }
        ```

We would like to acknowledge Gregory Griffin, Alex Holub, and Pietro Perona for creating and maintaining the Caltech-256 dataset as a valuable resource for the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and computer vision research community. For more information about the Caltech-256 dataset and its creators, visit the [Caltech-256 dataset website](https://data.caltech.edu/records/nyy15-4j048).

## FAQ

### What is the Caltech-256 dataset and why is it important for machine learning?

The [Caltech-256](https://data.caltech.edu/records/nyy15-4j048) dataset is a large image dataset used primarily for object classification tasks in machine learning and computer vision. It consists of around 30,000 color images divided into 257 categories, covering a wide range of real-world objects. The dataset's diverse and high-quality images make it an excellent benchmark for evaluating object recognition algorithms, which is crucial for developing robust machine learning models.

### How can I train a YOLO model on the Caltech-256 dataset using Python or CLI?

To train a YOLO model on the Caltech-256 dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch), you can use the following code snippets. Refer to the model [Training](../../modes/train.md) page for additional options.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-cls.pt")  # load a pretrained model

        # Train the model
        results = model.train(data="caltech256", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech256 model=yolo11n-cls.pt epochs=100 imgsz=416
        ```

### What are the most common use cases for the Caltech-256 dataset?

The Caltech-256 dataset is widely used for various object recognition tasks such as:

- Training Convolutional [Neural Networks](https://www.ultralytics.com/glossary/neural-network-nn) (CNNs)
- Evaluating the performance of Support Vector Machines (SVMs)
- Benchmarking new deep learning algorithms
- Developing [object detection](https://www.ultralytics.com/glossary/object-detection) models using frameworks like Ultralytics YOLO

Its diversity and comprehensive annotations make it ideal for research and development in machine learning and computer vision.

### How is the Caltech-256 dataset structured and split for training and testing?

The Caltech-256 dataset does not come with a predefined split for training and testing. Users typically create their own splits according to their specific needs. A common approach is to randomly select a subset of images for training and use the remaining images for testing. This flexibility allows users to tailor the dataset to their specific project requirements and experimental setups.

### Why should I use Ultralytics YOLO for training models on the Caltech-256 dataset?

Ultralytics YOLO models offer several advantages for training on the Caltech-256 dataset:

- **High Accuracy**: YOLO models are known for their state-of-the-art performance in object detection tasks.
- **Speed**: They provide real-time inference capabilities, making them suitable for applications requiring quick predictions.
- **Ease of Use**: With [Ultralytics HUB](https://www.ultralytics.com/hub), users can train, validate, and deploy models without extensive coding.
- **Pretrained Models**: Starting from pretrained models, like `yolo11n-cls.pt`, can significantly reduce training time and improve model [accuracy](https://www.ultralytics.com/glossary/accuracy).

For more details, explore our [comprehensive training guide](../../modes/train.md) and learn about [image classification](../../tasks/classify.md) with Ultralytics YOLO.
