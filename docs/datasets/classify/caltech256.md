---
comments: true
description: Learn about the Caltech-256 dataset, a broad collection of images used for object classification tasks in machine learning and computer vision algorithms.
keywords: Caltech-256, Dataset, Object Recognition, Image Classification, Convolutional Neural Networks, SVMs, YOLO, Deep Learning Models
---

# Caltech-256 Dataset

The [Caltech-256](https://data.caltech.edu/records/nyy15-4j048) dataset is an extensive collection of images used for object classification tasks. It contains around 30,000 images divided into 257 categories (256 object categories and 1 background category). The images are carefully curated and annotated to provide a challenging and diverse benchmark for object recognition algorithms.

## Key Features

- The Caltech-256 dataset comprises around 30,000 color images divided into 257 categories.
- Each category contains a minimum of 80 images.
- The categories encompass a wide variety of real-world objects, including animals, vehicles, household items, and people.
- Images are of variable sizes and resolutions.
- Caltech-256 is widely used for training and testing in the field of machine learning, particularly for object recognition tasks.

## Dataset Structure

Like Caltech-101, the Caltech-256 dataset does not have a formal split between training and testing sets. Users typically create their own splits according to their specific needs. A common practice is to use a random subset of images for training and the remaining images for testing.

## Applications

The Caltech-256 dataset is extensively used for training and evaluating deep learning models in object recognition tasks, such as Convolutional Neural Networks (CNNs), Support Vector Machines (SVMs), and various other machine learning algorithms. Its diverse set of categories and high-quality images make it an invaluable dataset for research and development in the field of machine learning and computer vision.

## Usage

To train a YOLO model on the Caltech-256 dataset for 100 epochs, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
        
        # Train the model
        model.train(data='caltech256', epochs=100, imgsz=416)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=caltech256 model=yolov8n-cls.pt epochs=100 imgsz=416
        ```

## Sample Images and Annotations

The Caltech-256 dataset contains high-quality color images of various objects, providing a comprehensive dataset for object recognition tasks. Here are some examples of images from the dataset ([credit](https://ml4a.github.io/demos/tsne_viewer.html)):

![Dataset sample image](https://user-images.githubusercontent.com/26833433/239365061-1e5f7857-b1e8-44ca-b3d7-d0befbcd33f9.jpg)

The example showcases the diversity and complexity of the objects in the Caltech-256 dataset, emphasizing the importance of a varied dataset for training robust object recognition models.

## Citations and Acknowledgments

If you use the Caltech-256 dataset in your research or development work, please cite the following paper:

```bibtex
@article{griffin2007caltech,
         title={Caltech-256 object category dataset},
         author={Griffin, Gregory and Holub, Alex and Perona, Pietro},
         year={2007}
}
```

We would like to acknowledge Gregory Griffin, Alex Holub, and Pietro Perona for creating and maintaining the Caltech-256 dataset as a valuable resource for the machine learning and computer vision research community. For more information about the

Caltech-256 dataset and its creators, visit the [Caltech-256 dataset website](https://data.caltech.edu/records/nyy15-4j048).