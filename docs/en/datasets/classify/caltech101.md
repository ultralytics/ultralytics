---
comments: true
description: Explore the widely-used Caltech-101 dataset with 9,000 images across 101 categories. Ideal for object recognition tasks in machine learning and computer vision.
keywords: Caltech-101, dataset, object recognition, machine learning, computer vision, YOLO, deep learning, research, AI
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

The Caltech-101 dataset is extensively used for training and evaluating [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models in object recognition tasks, such as [Convolutional Neural Networks](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn) (CNNs), Support Vector Machines (SVMs), and various other machine learning algorithms. Its wide variety of categories and high-quality images make it an excellent dataset for research and development in the field of machine learning and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv).

## Usage

To train a YOLO model on the Caltech-101 dataset for 100 epochs, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="caltech101", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech101 model=yolov8n-cls.pt epochs=100 imgsz=416
        ```

## Sample Images and Annotations

The Caltech-101 dataset contains high-quality color images of various objects, providing a well-structured dataset for object recognition tasks. Here are some examples of images from the dataset:

![Dataset sample image](https://github.com/ultralytics/docs/releases/download/0/caltech101-sample-image.avif)

The example showcases the variety and complexity of the objects in the Caltech-101 dataset, emphasizing the significance of a diverse dataset for training robust object recognition models.

## Citations and Acknowledgments

If you use the Caltech-101 dataset in your research or development work, please cite the following paper:

!!! quote ""

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

## FAQ

### What is the Caltech-101 dataset used for in machine learning?

The [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) dataset is widely used in machine learning for object recognition tasks. It contains around 9,000 images across 101 categories, providing a challenging benchmark for evaluating object recognition algorithms. Researchers leverage it to train and test models, especially Convolutional [Neural Networks](https://www.ultralytics.com/glossary/neural-network-nn) (CNNs) and [Support Vector Machines](https://www.ultralytics.com/glossary/support-vector-machine-svm) (SVMs), in computer vision.

### How can I train an Ultralytics YOLO model on the Caltech-101 dataset?

To train an Ultralytics YOLO model on the Caltech-101 dataset, you can use the provided code snippets. For example, to train for 100 [epochs](https://www.ultralytics.com/glossary/epoch):

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="caltech101", epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo classify train data=caltech101 model=yolov8n-cls.pt epochs=100 imgsz=416
        ```

For more detailed arguments and options, refer to the model [Training](../../modes/train.md) page.

### What are the key features of the Caltech-101 dataset?

The Caltech-101 dataset includes:

- Around 9,000 color images across 101 categories.
- Categories covering a diverse range of objects, including animals, vehicles, and household items.
- Variable number of images per category, typically between 40 and 800.
- Variable image sizes, with most being medium resolution.

These features make it an excellent choice for training and evaluating object recognition models in [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) and computer vision.

### Why should I cite the Caltech-101 dataset in my research?

Citing the Caltech-101 dataset in your research acknowledges the creators' contributions and provides a reference for others who might use the dataset. The recommended citation is:

!!! quote ""

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

Citing helps in maintaining the integrity of academic work and assists peers in locating the original resource.

### Can I use Ultralytics HUB for training models on the Caltech-101 dataset?

Yes, you can use Ultralytics HUB for training models on the Caltech-101 dataset. Ultralytics HUB provides an intuitive platform for managing datasets, training models, and deploying them without extensive coding. For a detailed guide, refer to the [how to train your custom models with Ultralytics HUB](https://www.ultralytics.com/blog/how-to-train-your-custom-models-with-ultralytics-hub) blog post.
