---
comments: true
description: Explore the medicinal pill detection dataset with labeled images. Essential for training AI models for pharmaceutical identification and automation.
keywords: medicinal pill dataset, pill detection, pharmaceutical imaging, AI in healthcare, computer vision, object detection, medical automation, dataset for training
---

# Medicinal Pill Dataset

The medicinal pill detection dataset consists of labeled images designed for identifying medicinal pills. This dataset is essential for training [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models to automate pharmaceutical identification, aiding in quality control, counterfeit detection, and packaging automation.

## Dataset Structure

The medicinal pill dataset is divided into two subsets:

- **Training set**: Consisting of 92 images, each annotated with the class `pill`.
- **Validation set**: Comprising 23 images with corresponding annotations.

## Applications

Using computer vision for medicinal pill detection enables automation in the pharmaceutical industry, supporting tasks like:

- **Pharmaceutical Sorting**: Automating the sorting of pills into specific categories based on physical characteristics like size, shape, or color.
- **Pill Counting**: Accurately counting pills for packaging and dispensing in pharmacies.
- **Regulatory Compliance**: Ensuring pills are properly labeled and packaged to meet pharmaceutical industry standards.

## Dataset YAML

A YAML configuration file is provided to define the dataset's structure, including paths and classes. For the medicinal pill dataset, the `medicinal-pill.yaml` file can be accessed at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medicinal-pill.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medicinal-pill.yaml).

!!! example "ultralytics/cfg/datasets/medicinal-pill.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/medicinal-pill.yaml"
    ```

## Usage

To train a YOLO11n model on the medicinal pill dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following examples. For detailed arguments, refer to the model's [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="medicinal-pill.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=medicinal-pill.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

!!! example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a fine-tuned model

        # Inference using the model
        results = model.predict("https://ultralytics.com/assets/medicinal-pill-sample.jpg")
        ```

    === "CLI"

        ```bash
        # Start prediction with a fine-tuned *.pt model
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/medicinal-pill-sample.jpg"
        ```

## Sample Images and Annotations

The medicinal pill dataset features labeled images showcasing the diversity of pills. Below is an example of a labeled image from the dataset:

![Medicinal pill dataset sample image](https://github.com/ultralytics/docs/releases/download/0/medicinal-pill-dataset-sample-image.avif)

- **Mosaiced Image**: Displayed is a training batch comprising mosaiced dataset images. Mosaicing enhances training diversity by consolidating multiple images into one, improving model generalization.

## Citations and Acknowledgments

The dataset is available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

## FAQ

### What is the structure of the medicinal pill dataset?

The dataset includes 92 images for training and 23 images for validation. Each image is annotated with the class `pill`, enabling effective training and evaluation of models.

### How can I train a YOLO11 model on the medicinal pill dataset?

You can train a YOLO11 model for 100 epochs with an image size of 640px using the Python or CLI methods provided. Refer to the [Training Example](#usage) section for detailed instructions.

### What are the benefits of using the medicinal pill dataset in AI projects?

The dataset enables automation in pill detection, contributing to counterfeit prevention, quality assurance, and pharmaceutical process optimization.

### How do I perform inference on the medicinal pill dataset?

Inference can be done using Python or CLI methods with a fine-tuned YOLO11 model. Refer to the [Inference Example](#usage) section for code snippets.

### Where can I find the YAML configuration file for the medicinal pill dataset?

The YAML file is available at [medicinal-pill.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/medicinal-pill.yaml), containing dataset paths, classes, and additional configuration details.
