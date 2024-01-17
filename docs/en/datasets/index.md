---
comments: true
description: Explore various computer vision datasets supported by Ultralytics for object detection, segmentation, pose estimation, image classification, and multi-object tracking.
keywords: computer vision, datasets, Ultralytics, YOLO, object detection, instance segmentation, pose estimation, image classification, multi-object tracking
---

# Datasets Overview

Ultralytics provides support for various datasets to facilitate computer vision tasks such as detection, instance segmentation, pose estimation, classification, and multi-object tracking. Below is a list of the main Ultralytics datasets, followed by a summary of each computer vision task and the respective datasets.

## 🌟 New: Ultralytics Explorer 🌟

Create embeddings for your dataset, search for similar images, run SQL queries, perform semantic search and even search using natural language! You can get started with our GUI app or build your own using the API. Learn more [here](explorer/index.md).

<p>
<img width="1709" alt="Screenshot 2024-01-08 at 7 19 48 PM (1)" src="https://github.com/AyushExel/assets/assets/15766192/e536b0eb-6bce-43fe-b800-3e79510d2e5b">
</p>

- Try the [GUI Demo](explorer/index.md)
- Learn more about the [Explorer API](explorer/index.md)

## [Detection Datasets](detect/index.md)

Bounding box object detection is a computer vision technique that involves detecting and localizing objects in an image by drawing a bounding box around each object.

- [Argoverse](detect/argoverse.md): A dataset containing 3D tracking and motion forecasting data from urban environments with rich annotations.
- [COCO](detect/coco.md): A large-scale dataset designed for object detection, segmentation, and captioning with over 200K labeled images.
- [COCO8](detect/coco8.md): Contains the first 4 images from COCO train and COCO val, suitable for quick tests.
- [Global Wheat 2020](detect/globalwheat2020.md): A dataset of wheat head images collected from around the world for object detection and localization tasks.
- [Objects365](detect/objects365.md): A high-quality, large-scale dataset for object detection with 365 object categories and over 600K annotated images.
- [OpenImagesV7](detect/open-images-v7.md): A comprehensive dataset by Google with 1.7M train images and 42k validation images.
- [SKU-110K](detect/sku-110k.md): A dataset featuring dense object detection in retail environments with over 11K images and 1.7 million bounding boxes.
- [VisDrone](detect/visdrone.md): A dataset containing object detection and multi-object tracking data from drone-captured imagery with over 10K images and video sequences.
- [VOC](detect/voc.md): The Pascal Visual Object Classes (VOC) dataset for object detection and segmentation with 20 object classes and over 11K images.
- [xView](detect/xview.md): A dataset for object detection in overhead imagery with 60 object categories and over 1 million annotated objects.

## [Instance Segmentation Datasets](segment/index.md)

Instance segmentation is a computer vision technique that involves identifying and localizing objects in an image at the pixel level.

- [COCO](segment/coco.md): A large-scale dataset designed for object detection, segmentation, and captioning tasks with over 200K labeled images.
- [COCO8-seg](segment/coco8-seg.md): A smaller dataset for instance segmentation tasks, containing a subset of 8 COCO images with segmentation annotations.

## [Pose Estimation](pose/index.md)

Pose estimation is a technique used to determine the pose of the object relative to the camera or the world coordinate system.

- [COCO](pose/coco.md): A large-scale dataset with human pose annotations designed for pose estimation tasks.
- [COCO8-pose](pose/coco8-pose.md): A smaller dataset for pose estimation tasks, containing a subset of 8 COCO images with human pose annotations.
- [Tiger-pose](pose/tiger-pose.md): A compact dataset consisting of 263 images focused on tigers, annotated with 12 keypoints per tiger for pose estimation tasks.

## [Classification](classify/index.md)

Image classification is a computer vision task that involves categorizing an image into one or more predefined classes or categories based on its visual content.

- [Caltech 101](classify/caltech101.md): A dataset containing images of 101 object categories for image classification tasks.
- [Caltech 256](classify/caltech256.md): An extended version of Caltech 101 with 256 object categories and more challenging images.
- [CIFAR-10](classify/cifar10.md): A dataset of 60K 32x32 color images in 10 classes, with 6K images per class.
- [CIFAR-100](classify/cifar100.md): An extended version of CIFAR-10 with 100 object categories and 600 images per class.
- [Fashion-MNIST](classify/fashion-mnist.md): A dataset consisting of 70,000 grayscale images of 10 fashion categories for image classification tasks.
- [ImageNet](classify/imagenet.md): A large-scale dataset for object detection and image classification with over 14 million images and 20,000 categories.
- [ImageNet-10](classify/imagenet10.md): A smaller subset of ImageNet with 10 categories for faster experimentation and testing.
- [Imagenette](classify/imagenette.md): A smaller subset of ImageNet that contains 10 easily distinguishable classes for quicker training and testing.
- [Imagewoof](classify/imagewoof.md): A more challenging subset of ImageNet containing 10 dog breed categories for image classification tasks.
- [MNIST](classify/mnist.md): A dataset of 70,000 grayscale images of handwritten digits for image classification tasks.

## [Oriented Bounding Boxes (OBB)](obb/index.md)

Oriented Bounding Boxes (OBB) is a method in computer vision for detecting angled objects in images using rotated bounding boxes, often applied to aerial and satellite imagery.

- [DOTAv2](obb/dota-v2.md): A popular OBB aerial imagery dataset with 1.7 million instances and 11,268 images.

## [Multi-Object Tracking](track/index.md)

Multi-object tracking is a computer vision technique that involves detecting and tracking multiple objects over time in a video sequence.

- [Argoverse](detect/argoverse.md): A dataset containing 3D tracking and motion forecasting data from urban environments with rich annotations for multi-object tracking tasks.
- [VisDrone](detect/visdrone.md): A dataset containing object detection and multi-object tracking data from drone-captured imagery with over 10K images and video sequences.

## Contribute New Datasets

Contributing a new dataset involves several steps to ensure that it aligns well with the existing infrastructure. Below are the necessary steps:

### Steps to Contribute a New Dataset

1. **Collect Images**: Gather the images that belong to the dataset. These could be collected from various sources, such as public databases or your own collection.

2. **Annotate Images**: Annotate these images with bounding boxes, segments, or keypoints, depending on the task.

3. **Export Annotations**: Convert these annotations into the YOLO `*.txt` file format which Ultralytics supports.

4. **Organize Dataset**: Arrange your dataset into the correct folder structure. You should have `train/` and `val/` top-level directories, and within each, an `images/` and `labels/` subdirectory.

    ```
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    ```

5. **Create a `data.yaml` File**: In your dataset's root directory, create a `data.yaml` file that describes the dataset, classes, and other necessary information.

6. **Optimize Images (Optional)**: If you want to reduce the size of the dataset for more efficient processing, you can optimize the images using the code below. This is not required, but recommended for smaller dataset sizes and faster download speeds.

7. **Zip Dataset**: Compress the entire dataset folder into a zip file.

8. **Document and PR**: Create a documentation page describing your dataset and how it fits into the existing framework. After that, submit a Pull Request (PR). Refer to [Ultralytics Contribution Guidelines](https://docs.ultralytics.com/help/contributing) for more details on how to submit a PR.

### Example Code to Optimize and Zip a Dataset

!!! Example "Optimize and Zip a Dataset"

    === "Python"

    ```python
    from pathlib import Path
    from ultralytics.data.utils import compress_one_image
    from ultralytics.utils.downloads import zip_directory

    # Define dataset directory
    path = Path('path/to/dataset')

    # Optimize images in dataset (optional)
    for f in path.rglob('*.jpg'):
        compress_one_image(f)

    # Zip dataset into 'path/to/dataset.zip'
    zip_directory(path)
    ```

By following these steps, you can contribute a new dataset that integrates well with Ultralytics' existing structure.
