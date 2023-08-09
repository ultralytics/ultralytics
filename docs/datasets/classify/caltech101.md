---
comments: true
description: Learn about the Caltech-101 dataset, its structure and uses in machine learning. Includes instructions to train a YOLO model using this dataset.
keywords: Caltech-101, dataset, YOLO training, machine learning, object recognition, ultralytics
---

# Caltech-101 Dataset

The [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) dataset is a widely used dataset for object recognition tasks, containing around 9,000 images from 101 object categories. The categories were chosen to reflect a variety of real-world objects, and the images themselves were carefully selected and annotated to provide a challenging benchmark for object recognition algorithms.

## Key Features

- The Caltech-101 dataset comprises around 9,000 color images divided into 101 categories.
- The categories encompass a wide variety of objects, including animals, vehicles, household items, and people.
- The number of images per category varies, with about 40 to 800 images in each category.
- Images are of variable sizes, with most images being medium resolution.
- Caltech-101 is widely used for training and testing in the field of machine learning, particularly for object recognition tasks.

## Dataset Structure

Unlike many other datasets, the Caltech-101 dataset is not formally split into training and testing sets. Users typically create their own splits based on their specific needs. However, a common practice is to use a random subset of images for training (e.g., 30 images per category) and the remaining images for testing.

## Applications

The Caltech-101 dataset is extensively used for training and evaluating deep learning models in object recognition tasks, such as Convolutional Neural Networks (CNNs), Support Vector Machines (SVMs), and various other machine learning algorithms. Its wide variety of categories and high-quality images make it an excellent dataset for research and development in the field of machine learning and computer vision.

## Usage

To train a YOLO model on the Caltech-101 dataset for 100 epochs, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data='caltech101', epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=caltech101 model=yolov8n-cls.pt epochs=100 imgsz=416
        ```

## Sample Images and Annotations

The Caltech-101 dataset contains high-quality color images of various objects, providing a well-structured dataset for object recognition tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://user-images.githubusercontent.com/26833433/239366386-44171121-b745-4206-9b59-a3be41e16089.png)

The example showcases the variety and complexity of the objects in the Caltech-101 dataset, emphasizing the significance of a diverse dataset for training robust object recognition models.

## Citations and Acknowledgments

If you use the Caltech-101 dataset in your research or development work, please cite the following paper:

!!! note ""

    === "BibTeX"

        ```bibtex
        @article{fei2007learning,
          title={Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories},
          author={Fei-Fei, Li and Fergus, Rob and Perona, Pietro},
          journal={Computer vision and Image understanding},
          volume={106},
          number={1},
          pages={59--70},
          year={2007},
          publisher={Elsevier}
        }
        ```

We would like to acknowledge Li Fei-Fei, Rob Fergus, and Pietro Perona for creating and maintaining the Caltech-101 dataset as a valuable resource for the machine learning and computer vision research community. For more information about the Caltech-101 dataset and its creators, visit the [Caltech-101 dataset website](https://data.caltech.edu/records/mzrjq-6wc02).
