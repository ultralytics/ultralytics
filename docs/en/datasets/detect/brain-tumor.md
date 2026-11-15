---
title: Brain Tumor Detection Dataset
comments: true
creator:
    name: Ultralytics
    url: https://www.ultralytics.com
license:
    name: AGPL-3.0
    url: https://www.ultralytics.com/license
description: Train YOLO26 on the Ultralytics Brain Tumor dataset with 1,115 brain MRI images and 1,180 tumor boxes in one class, split into 994 train and 121 val images.
keywords: brain tumor dataset, MRI scans, brain tumor detection, medical imaging, AI in healthcare, computer vision, object detection, YOLO26, tumor localization
---

# Brain Tumor Dataset

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-train-ultralytics-yolo-on-brain-tumor-detection-dataset.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Brain Tumor Dataset In Colab"></a>

The Ultralytics Brain Tumor dataset is an [object detection](https://www.ultralytics.com/glossary/object-detection) dataset of 1,115 brain MRI images (994 for training and 121 for validation) with 1,180 [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) in a single class, `tumor`. Every image contains 1 to 3 tumor boxes. The validation split holds the images of 25 identified patients, and no identified patient appears in both splits. At 4.06 MB, it is a compact dataset for learning and prototyping [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models for [healthcare applications](https://www.ultralytics.com/solutions/computer-vision-in-healthcare), and it is not a clinically validated diagnostic resource.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/Jj7WpfiegD0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Brain Tumor Detection using Ultralytics Platform with Ultralytics YOLO26 | Object Detection 🚀
</p>

## Dataset Structure

The Ultralytics Brain Tumor dataset contains 1,115 brain MRI images and 1,180 bounding boxes in two splits defined by the `brain-tumor.yaml` configuration:

| Split      | Images    | Bounding boxes | Patients                                            |
| ---------- | --------- | -------------- | --------------------------------------------------- |
| Train      | 994       | 1,057          | 95 with an ID, plus 541 images without a patient ID |
| Validation | 121       | 123            | 25                                                  |
| **Total**  | **1,115** | **1,180**      |                                                     |

Every image contains 1 to 3 bounding boxes of the single class `tumor`, and image sizes range from 192x256 to 512x512 pixels. The configuration defines no separate test split.

!!! note "Validation split"

    The validation split is built by patient. Some file names start with a five-digit patient ID: for example, `00097_77.jpg` belongs to patient `00097`. The images carrying the IDs of 25 of these patients form the validation split and the images of the other 95 identified patients are in training, so no identified patient appears in both splits. The remaining 541 images have no patient ID in the file name and are all in the training split.

The dataset downloads automatically (4.06 MB) from Ultralytics GitHub assets the first time you train, so no manual download or preparation is required.

!!! warning "Update an older local copy"

    The dataset changed to a single `tumor` class in November 2026. If your local `brain-tumor/images/train` folder holds 893 images instead of 994, delete the `brain-tumor` folder in your datasets directory (see `datasets_dir` in `yolo settings`) and update the package with `pip install -U ultralytics`. The current version then downloads on the next training run.

Explore [Brain Tumor on Ultralytics Platform](https://platform.ultralytics.com/ultralytics/datasets/brain-tumor) to browse the images with their annotation overlays, view the class distribution and bounding-box heatmaps in the **Charts** tab, and clone it to train your own model in the cloud.

## Applications

Brain tumor [detection](../../tasks/detect.md) models locate tumors in brain MRI images. With 1,115 images and a 4.06 MB download, the Ultralytics Brain Tumor dataset suits:

- **Learning the workflow**: running the full train, [validate](../../modes/val.md), and [predict](../../modes/predict.md) cycle on medical images with a small dataset.
- **Prototyping**: building and testing a tumor localization pipeline before moving to a larger clinical dataset.
- **Held-out patient evaluation**: measuring detection on the 25 validation patients, with no identified patient in both splits.
- **Research and education**: comparing detection models and studying [medical image analysis](https://www.ultralytics.com/glossary/medical-image-analysis) and [AI in radiology](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency).

Models trained on this dataset are for research and education and are not validated for clinical diagnosis.

## Dataset YAML

The [`brain-tumor.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/brain-tumor.yaml) configuration defines the dataset path, the train and validation image folders, the single `tumor` class, and the download URL. Labels use the [Ultralytics YOLO format](index.md#ultralytics-yolo-format).

!!! example "ultralytics/cfg/datasets/brain-tumor.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/brain-tumor.yaml"
    ```

## Usage

To [train](../../modes/train.md) a [YOLO26n](https://platform.ultralytics.com/ultralytics/yolo26/yolo26n) model on the brain tumor dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the code snippets below.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=brain-tumor.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

!!! example "Inference Example"

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

Below are sample images from the Ultralytics Brain Tumor dataset with their annotations:

![Sample brain MRI images with tumor bounding boxes from the Ultralytics Brain Tumor dataset](https://cdn.ul.run/i/bd7eed1fc78ec679209ff0d402e5dcdf.avif)

- **Annotated Images**: The grid shows 16 training images, each with a `tumor` bounding box around the lesion. Tumors vary in size, shape, and location across the dataset, which helps a detection model generalize.

## Citations and Acknowledgments

The dataset is available under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

If you use the brain tumor dataset in your research or development work, please cite it:

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @dataset{Ultralytics_Brain_Tumor_Dataset_2023,
            author = {Ultralytics},
            title = {Brain Tumor Detection Dataset},
            year = {2023},
            publisher = {Ultralytics},
            url = {https://docs.ultralytics.com/datasets/detect/brain-tumor}
        }
        ```

## FAQ

### How many images and classes are in the brain tumor dataset?

The Ultralytics Brain Tumor dataset has 1,115 brain MRI images and 1 class, `tumor`: 994 training images with 1,057 bounding boxes and 121 validation images with 123 bounding boxes. Every image has 1 to 3 boxes, and there is no separate test split. See [Dataset Structure](#dataset-structure) for the split details.

### Is the brain tumor dataset MRI or CT?

The Ultralytics Brain Tumor dataset contains brain MRI images and no CT scans. Image sizes range from 192x256 to 512x512 pixels, and each image is labeled with 1 to 3 `tumor` bounding boxes in [Ultralytics YOLO format](index.md#ultralytics-yolo-format).

### Can the brain tumor dataset tell tumor from no-tumor scans?

No, the Ultralytics Brain Tumor dataset contains no tumor-free images and no negative class. All 1,115 images have 1 to 3 `tumor` bounding boxes, so a model trained on it learns where a tumor is, not whether a scan is healthy. Models trained on this dataset are for research and education, not clinical diagnosis.

### How is the brain tumor dataset split into training and validation sets?

The Ultralytics Brain Tumor dataset is split by patient: the 121 validation images carry the IDs of 25 patients, and no identified patient appears in both splits. Patients are identified by the five-digit prefix of the file name, such as `00097_77.jpg`. The training set holds the other 95 identified patients plus 541 images whose file names carry no patient ID.

### How do I download the brain tumor dataset?

The Ultralytics Brain Tumor dataset (4.06 MB) downloads automatically from Ultralytics GitHub assets the first time you train with `data="brain-tumor.yaml"`, so no manual download is required. If an earlier copy with 893 training images is already on disk, delete its `brain-tumor` folder and update the `ultralytics` package first so the current single-class version downloads. For another small medical imaging dataset, see the [Medical Pills detection dataset](medical-pills.md).

### How can I train a YOLO26 model on the brain tumor dataset using Ultralytics?

To train YOLO26 on the Ultralytics Brain Tumor dataset, pass `data="brain-tumor.yaml"` to the train command, and the dataset downloads automatically. The examples below train YOLO26n for 100 epochs at an image size of 640:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=brain-tumor.yaml model=yolo26n.pt epochs=100 imgsz=640
        ```

For a detailed list of available arguments, refer to the [Training](../../modes/train.md) page.

### How do I perform inference using a fine-tuned YOLO26 model on the brain tumor dataset?

To run inference with a YOLO26 model fine-tuned on the Ultralytics Brain Tumor dataset, load your `best.pt` weights and call [predict](../../modes/predict.md) on a brain MRI image. Each result contains the `tumor` bounding boxes with confidence scores:

!!! example "Inference Example"

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

### What license does the brain tumor dataset use?

The Ultralytics Brain Tumor dataset was created by Ultralytics and is released under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
