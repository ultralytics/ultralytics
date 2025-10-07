---
comments: true
description: Explore the Ultralytics COCO8-Multispectral dataset, an enhanced version of COCO8 with interpolated spectral channels, ideal for testing multispectral object detection models and training pipelines.
keywords: COCO8-Multispectral, Ultralytics, dataset, multispectral, object detection, YOLO11, training, validation, machine learning, computer vision
---

# COCO8-Multispectral Dataset

## Introduction

The [Ultralytics](https://www.ultralytics.com/) COCO8-Multispectral dataset is an advanced variant of the original COCO8 dataset, designed to facilitate experimentation with multispectral object detection models. It consists of the same 8 images from the COCO train 2017 set—4 for training and 4 for validation—but with each image transformed into a 10-channel multispectral format. By expanding beyond standard RGB channels, COCO8-Multispectral enables the development and evaluation of models that can leverage richer spectral information.

<p align="center">
  <img width="640" src="https://github.com/ultralytics/docs/releases/download/0/coco8-multispectral-overview.avif" alt="Multispectral Imagery Overview">
</p>

COCO8-Multispectral is fully compatible with [Ultralytics HUB](https://hub.ultralytics.com/) and [YOLO11](../../models/yolo11.md), ensuring seamless integration into your [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) workflows.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yw2Fo6qjJU4"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Train Ultralytics YOLO11 on Multispectral Datasets | Multi-Channel VisionAI 🚀
</p>

## Dataset Generation

The multispectral images in COCO8-Multispectral were created by interpolating the original RGB images across 10 evenly spaced spectral channels within the visible spectrum. The process includes:

- **Wavelength Assignment**: Assigning nominal wavelengths to the RGB channels—Red: 650 nm, Green: 510 nm, Blue: 475 nm.
- **Interpolation**: Using linear interpolation to estimate pixel values at intermediate wavelengths between 450 nm and 700 nm, resulting in 10 spectral channels.
- **Extrapolation**: Applying extrapolation with SciPy's `interp1d` function to estimate values beyond the original RGB wavelengths, ensuring a complete spectral representation.

This approach simulates a multispectral imaging process, providing a more diverse set of data for model training and evaluation. For further reading on multispectral imaging, see the [Multispectral Imaging Wikipedia article](https://en.wikipedia.org/wiki/Multispectral_imaging).

## Dataset YAML

The COCO8-Multispectral dataset is configured using a YAML file, which defines dataset paths, class names, and essential metadata. You can review the official `coco8-multispectral.yaml` file in the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8-multispectral.yaml).

!!! example "ultralytics/cfg/datasets/coco8-multispectral.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-multispectral.yaml"
    ```

!!! note

    Prepare your TIFF images in `(channel, height, width)` order and saved with `.tiff` or `.tif` extension for use with Ultralytics:

    ```python
    import cv2
    import numpy as np

    # Create and write 10-channel TIFF
    image = np.ones((10, 640, 640), dtype=np.uint8)  # CHW-order
    cv2.imwritemulti("example.tiff", image)

    # Read TIFF
    success, frames_list = cv2.imreadmulti("example.tiff")
    image = np.stack(frames_list, axis=2)
    print(image.shape)  # (640, 640, 10)  HWC-order for training and inference
    ```

## Usage

To train a YOLO11n model on the COCO8-Multispectral dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch) with an image size of 640, use the following examples. For a comprehensive list of training options, refer to the [YOLO Training documentation](../../modes/train.md).

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO11n model
        model = YOLO("yolo11n.pt")

        # Train the model on COCO8-Multispectral
        results = model.train(data="coco8-multispectral.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train YOLO11n on COCO8-Multispectral using the command line
        yolo detect train data=coco8-multispectral.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

For more details on model selection and best practices, explore the [Ultralytics YOLO model documentation](../../models/yolo11.md) and the [YOLO Model Training Tips guide](https://docs.ultralytics.com/guides/model-training-tips/).

## Sample Images and Annotations

Below is an example of a mosaiced training batch from the COCO8-Multispectral dataset:

<img src="https://github.com/ultralytics/docs/releases/download/0/coco8-multispectral-mosaic-batch.avif" alt="Dataset sample image" width="800">

- **Mosaiced Image**: This image demonstrates a training batch where multiple dataset images are combined using [mosaic augmentation](https://docs.ultralytics.com/reference/data/augment/). Mosaic augmentation increases the diversity of objects and scenes within each batch, helping the model generalize better to various object sizes, aspect ratios, and backgrounds.

This technique is especially valuable for small datasets like COCO8-Multispectral, as it maximizes the utility of each image during training.

## Citations and Acknowledgments

If you use the COCO dataset in your research or development, please cite the following paper:

!!! quote ""

    === "BibTeX"

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

Special thanks to the [COCO Consortium](https://cocodataset.org/#home) for their ongoing contributions to the [computer vision community](https://www.ultralytics.com/blog/a-history-of-vision-models).

## FAQ

### What Is the Ultralytics COCO8-Multispectral Dataset Used For?

The Ultralytics COCO8-Multispectral dataset is designed for rapid testing and debugging of [multispectral object detection](https://www.ultralytics.com/glossary/object-detection) models. With only 8 images (4 for training, 4 for validation), it is ideal for verifying your [YOLO](../../models/yolo11.md) training pipelines and ensuring everything works as expected before scaling to larger datasets. For more datasets to experiment with, visit the [Ultralytics Datasets Catalog](https://docs.ultralytics.com/datasets/).

### How Does Multispectral Data Improve Object Detection?

Multispectral data provides additional spectral information beyond standard RGB, enabling models to distinguish objects based on subtle differences in reflectance across wavelengths. This can enhance detection accuracy, especially in challenging scenarios. Learn more about [multispectral imaging](https://en.wikipedia.org/wiki/Multispectral_imaging) and its applications in [advanced computer vision](https://www.ultralytics.com/blog/ai-in-aviation-a-runway-to-smarter-airports).

### Is COCO8-Multispectral Compatible With Ultralytics HUB and YOLO Models?

Yes, COCO8-Multispectral is fully compatible with [Ultralytics HUB](https://hub.ultralytics.com/) and all [YOLO models](../../models/yolo11.md), including the latest YOLO11. This allows you to easily integrate the dataset into your training and validation workflows.

### Where Can I Find More Information on Data Augmentation Techniques?

For a deeper understanding of data augmentation methods such as mosaic and their impact on model performance, refer to the [YOLO Data Augmentation Guide](https://docs.ultralytics.com/guides/yolo-data-augmentation/) and the [Ultralytics Blog on Data Augmentation](https://www.ultralytics.com/blog/the-ultimate-guide-to-data-augmentation-in-2025).

### Can I Use COCO8-Multispectral for Benchmarking or Educational Purposes?

Absolutely! The small size and multispectral nature of COCO8-Multispectral make it ideal for benchmarking, educational demonstrations, and prototyping new model architectures. For more benchmarking datasets, see the [Ultralytics Benchmark Dataset Collection](https://docs.ultralytics.com/datasets/).
