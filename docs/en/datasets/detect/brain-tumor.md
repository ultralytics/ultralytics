---
comments: true
description: Explore the brain tumor detection dataset with MRI/CT images. Essential for training AI models for early diagnosis and treatment planning.
keywords: brain tumor dataset, MRI scans, CT scans, brain tumor detection, medical imaging, AI in healthcare, computer vision, early diagnosis, treatment planning
---

# Brain Tumor Dataset

A brain tumor detection dataset consists of medical images from MRI or CT scans, containing information about brain tumor presence, location, and characteristics. This dataset is essential for training computer vision algorithms to automate brain tumor identification, aiding in early diagnosis and treatment planning.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/ogTBBD8McRk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Brain Tumor Detection using Ultralytics HUB
</p>

## Dataset Structure

The brain tumor dataset is divided into two subsets:

- **Training set**: Consisting of 893 images, each accompanied by corresponding annotations.
- **Testing set**: Comprising 223 images, with annotations paired for each one.

## Applications

The application of brain tumor detection using computer vision enables early diagnosis, treatment planning, and monitoring of tumor progression. By analyzing medical imaging data like MRI or CT scans, computer vision systems assist in accurately identifying brain tumors, aiding in timely medical intervention and personalized treatment strategies.

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information. In the case of the brain tumor dataset, the `brain-tumor.yaml` file is maintained at [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml).

!!! Example "ultralytics/cfg/datasets/brain-tumor.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/brain-tumor.yaml"
    ```

## Usage

To train a YOLOv8n model on the brain tumor dataset for 100 epochs with an image size of 640, utilize the provided code snippets. For a detailed list of available arguments, consult the model's [Training](../../modes/train.md) page.

!!! Example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=brain-tumor.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

!!! Example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a brain-tumor fine-tuned model

        # Inference using the model
        results = model.predict("https://ultralytics.com/assets/brain-tumor-sample.jpg")
        ```

    === "CLI"

        ```bash
        # Start prediction with a finetuned *.pt model
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/brain-tumor-sample.jpg"
        ```

## Sample Images and Annotations

The brain tumor dataset encompasses a wide array of images featuring diverse object categories and intricate scenes. Presented below are examples of images from the dataset, accompanied by their respective annotations

![Brain tumor dataset sample image](https://github.com/RizwanMunawar/RizwanMunawar/assets/62513924/1741cbf5-2462-4e9a-b0b9-4a07d76cf7ef)

- **Mosaiced Image**: Displayed here is a training batch comprising mosaiced dataset images. Mosaicing, a training technique, consolidates multiple images into one, enhancing batch diversity. This approach aids in improving the model's capacity to generalize across various object sizes, aspect ratios, and contexts.

This example highlights the diversity and intricacy of images within the brain tumor dataset, underscoring the advantages of incorporating mosaicing during the training phase.

## Citations and Acknowledgments

The dataset has been released available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

## FAQ

### What is the structure of the brain tumor dataset available in Ultralytics documentation?

The brain tumor dataset is divided into two subsets: the **training set** consists of 893 images with corresponding annotations, while the **testing set** comprises 223 images with paired annotations. This structured division aids in developing robust and accurate computer vision models for detecting brain tumors. For more information on the dataset structure, visit the [Dataset Structure](#dataset-structure) section.

### How can I train a YOLOv8 model on the brain tumor dataset using Ultralytics?

You can train a YOLOv8 model on the brain tumor dataset for 100 epochs with an image size of 640px using both Python and CLI methods. Below are the examples for both:

!!! Example "Train Example"

    === "Python"
    
        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640)
        ```
    

    === "CLI"
    
        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=brain-tumor.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

For a detailed list of available arguments, refer to the [Training](../../modes/train.md) page.

### What are the benefits of using the brain tumor dataset for AI in healthcare?

Using the brain tumor dataset in AI projects enables early diagnosis and treatment planning for brain tumors. It helps in automating brain tumor identification through computer vision, facilitating accurate and timely medical interventions, and supporting personalized treatment strategies. This application holds significant potential in improving patient outcomes and medical efficiencies.

### How do I perform inference using a fine-tuned YOLOv8 model on the brain tumor dataset?

Inference using a fine-tuned YOLOv8 model can be performed with either Python or CLI approaches. Here are the examples:

!!! Example "Inference Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("path/to/best.pt")  # load a brain-tumor fine-tuned model

        # Inference using the model
        results = model.predict("https://ultralytics.com/assets/brain-tumor-sample.jpg")
        ```

    === "CLI"
    
        ```bash
        # Start prediction with a finetuned *.pt model
        yolo detect predict model='path/to/best.pt' imgsz=640 source="https://ultralytics.com/assets/brain-tumor-sample.jpg"
        ```

### Where can I find the YAML configuration for the brain tumor dataset?

The YAML configuration file for the brain tumor dataset can be found at [brain-tumor.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml). This file includes paths, classes, and additional relevant information necessary for training and evaluating models on this dataset.
