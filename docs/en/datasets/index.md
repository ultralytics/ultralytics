---
comments: true
description: Explore Ultralytics' diverse datasets for vision tasks like detection, segmentation, classification, and more. Enhance your projects with high-quality annotated data.
keywords: Ultralytics, datasets, computer vision, object detection, instance segmentation, pose estimation, image classification, multi-object tracking
---

# Datasets Overview

Ultralytics provides support for various datasets to facilitate computer vision tasks such as detection, [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation), pose estimation, classification, and multi-object tracking. Below is a list of the main Ultralytics datasets, followed by a summary of each computer vision task and the respective datasets.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/YDXKa1EljmU"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics Datasets Overview
</p>

## [Object Detection](detect/index.md)

[Bounding box](https://www.ultralytics.com/glossary/bounding-box) object detection is a computer vision technique that involves detecting and localizing objects in an image by drawing a bounding box around each object.

- [African-wildlife](detect/african-wildlife.md): A dataset featuring images of African wildlife, including buffalo, elephants, rhinos, and zebras.
- [Argoverse](detect/argoverse.md): A dataset containing 3D tracking and motion forecasting data from urban environments with rich annotations.
- [Brain-tumor](detect/brain-tumor.md): A dataset for detecting brain tumors that includes MRI or CT scan images with details on tumor presence, location, and characteristics.
- [COCO](detect/coco.md): Common Objects in Context (COCO) is a large-scale object detection, segmentation, and captioning dataset with 80 object categories.
- [COCO8](detect/coco8.md): A smaller subset of the first 4 images from COCO train and COCO val, suitable for quick tests.
- [COCO8-Grayscale](detect/coco8-grayscale.md): A grayscale version of COCO8 created by converting RGB to grayscale, useful for single-channel model evaluation.
- [COCO8-Multispectral](detect/coco8-multispectral.md): A 10-channel multispectral version of COCO8 created by interpolating RGB wavelengths, useful for spectral-aware model evaluation.
- [COCO128](detect/coco128.md): A smaller subset of the first 128 images from COCO train and COCO val, suitable for tests.
- [Construction-PPE](detect/construction-ppe.md): A dataset of construction site imagery annotated with key safety gear such as helmets, vests, gloves, boots, and goggles, along with labels for missing equipment, supporting the development of AI models for compliance and worker protection.
- [Global Wheat 2020](detect/globalwheat2020.md): A dataset containing images of wheat heads for the Global Wheat Challenge 2020.
- [HomeObjects-3K](detect/homeobjects-3k.md): A dataset of annotated indoor scenes featuring 12 common household items, ideal for developing and testing computer vision models in smart home systems, robotics, and augmented reality.
- [KITTI](detect/kitti.md) New: A well-known autonomous driving dataset featuring stereo, LiDAR, and GPS/IMU inputs, used for 2D object detection in varied road scenes.
- [LVIS](detect/lvis.md): A large-scale object detection, segmentation, and captioning dataset with 1203 object categories.
- [Medical-pills](detect/medical-pills.md): A dataset containing labeled images of medical pills, designed to aid in tasks like pharmaceutical quality control, sorting, and ensuring compliance with industry standards.
- [Objects365](detect/objects365.md): A high-quality, large-scale dataset for object detection with 365 object categories and over 600K annotated images.
- [OpenImagesV7](detect/open-images-v7.md): A comprehensive dataset by Google with 1.7M train images and 42k validation images.
- [RF100](detect/roboflow-100.md): A diverse object detection benchmark with 100 datasets spanning seven imagery domains for comprehensive model evaluation.
- [Signature](detect/signature.md): A dataset featuring images of various documents with annotated signatures, supporting document verification and fraud detection research.
- [SKU-110K](detect/sku-110k.md): A dataset featuring dense object detection in retail environments with over 11K images and 1.7 million bounding boxes.
- [VisDrone](detect/visdrone.md): A dataset containing object detection and multi-object tracking data from drone-captured imagery with over 10K images and video sequences.
- [VOC](detect/voc.md): The Pascal Visual Object Classes (VOC) dataset for object detection and segmentation with 20 object classes and over 11K images.
- [xView](detect/xview.md): A dataset for object detection in overhead imagery with 60 object categories and over 1 million annotated objects.

## [Instance Segmentation](segment/index.md)

Instance segmentation is a computer vision technique that involves identifying and localizing objects in an image at the pixel level. Unlike semantic segmentation which only classifies each pixel, [instance segmentation](https://www.ultralytics.com/glossary/instance-segmentation) distinguishes between different instances of the same class.

- [Carparts-seg](segment/carparts-seg.md): Purpose-built dataset for identifying vehicle parts, catering to design, manufacturing, and research needs. It serves for both object detection and segmentation tasks.
- [COCO](segment/coco.md): A large-scale dataset designed for object detection, segmentation, and captioning tasks with over 200K labeled images.
- [COCO8-seg](segment/coco8-seg.md): A smaller dataset for instance segmentation tasks, containing a subset of 8 COCO images with segmentation annotations.
- [COCO128-seg](segment/coco128-seg.md): A smaller dataset for instance segmentation tasks, containing a subset of 128 COCO images with segmentation annotations.
- [Crack-seg](segment/crack-seg.md): Specifically crafted dataset for detecting cracks on roads and walls, applicable for both object detection and segmentation tasks.
- [Package-seg](segment/package-seg.md): Tailored dataset for identifying packages in warehouses or industrial settings, suitable for both object detection and segmentation applications.

## [Pose Estimation](pose/index.md)

Pose estimation is a technique used to determine the pose of the object relative to the camera or the world coordinate system. This involves identifying key points or joints on objects, particularly humans or animals.

- [COCO](pose/coco.md): A large-scale dataset with human pose annotations designed for pose estimation tasks.
- [COCO8-pose](pose/coco8-pose.md): A smaller dataset for pose estimation tasks, containing a subset of 8 COCO images with human pose annotations.
- [Dog-pose](pose/dog-pose.md): A comprehensive dataset featuring approximately 6,000 images focused on dogs, annotated with 24 keypoints per dog, tailored for pose estimation tasks.
- [Hand-Keypoints](pose/hand-keypoints.md): A concise dataset featuring over 26,000 images centered on human hands, annotated with 21 keypoints per hand, designed for pose estimation tasks.
- [Tiger-pose](pose/tiger-pose.md): A compact dataset consisting of 263 images focused on tigers, annotated with 12 keypoints per tiger for pose estimation tasks.

## [Classification](classify/index.md)

[Image classification](https://www.ultralytics.com/glossary/image-classification) is a computer vision task that involves categorizing an image into one or more predefined classes or categories based on its visual content.

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
- [MNIST160](classify/mnist.md): First 8 images of each MNIST category from the MNIST dataset. Dataset contains 160 images total.

## [Oriented Bounding Boxes (OBB)](obb/index.md)

Oriented Bounding Boxes (OBB) is a method in computer vision for detecting angled objects in images using rotated bounding boxes, often applied to aerial and satellite imagery. Unlike traditional bounding boxes, OBB can better fit objects at various orientations.

- [DOTA-v2](obb/dota-v2.md): A popular OBB aerial imagery dataset with 1.7 million instances and 11,268 images.
- [DOTA8](obb/dota8.md): A smaller subset of the first 8 images from the DOTAv1 split set, 4 for training and 4 for validation, suitable for quick tests.

## [Multi-Object Tracking](track/index.md)

Multi-object tracking is a computer vision technique that involves detecting and tracking multiple objects over time in a video sequence. This task extends object detection by maintaining consistent identities of objects across frames.

- [Argoverse](detect/argoverse.md): A dataset containing 3D tracking and motion forecasting data from urban environments with rich annotations for multi-object tracking tasks.
- [VisDrone](detect/visdrone.md): A dataset containing object detection and multi-object tracking data from drone-captured imagery with over 10K images and video sequences.

## Contribute New Datasets

Contributing a new dataset involves several steps to ensure that it aligns well with the existing infrastructure. Below are the necessary steps:

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yMR7BgwHQ3g?start=427"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Contribute to Ultralytics Datasets
</p>

### Steps to Contribute a New Dataset

1. **Collect Images**: Gather the images that belong to the dataset. These could be collected from various sources, such as public databases or your own collection.
2. **Annotate Images**: Annotate these images with bounding boxes, segments, or keypoints, depending on the task.
3. **Export Annotations**: Convert these annotations into the YOLO `*.txt` file format which Ultralytics supports.
4. **Organize Dataset**: Arrange your dataset into the correct folder structure. You should have `images/` and `labels/` top-level directories, and within each, a `train/` and `val/` subdirectory.

    ```
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    ```

5. **Create a `data.yaml` File**: In your dataset's root directory, create a `data.yaml` file that describes the dataset, classes, and other necessary information.
6. **Optimize Images (Optional)**: If you want to reduce the size of the dataset for more efficient processing, you can optimize the images using the code below. This is not required, but recommended for smaller dataset sizes and faster download speeds.
7. **Zip Dataset**: Compress the entire dataset folder into a zip file.
8. **Document and PR**: Create a documentation page describing your dataset and how it fits into the existing framework. After that, submit a Pull Request (PR). Refer to [Ultralytics Contribution Guidelines](https://docs.ultralytics.com/help/contributing/) for more details on how to submit a PR.

### Example Code to Optimize and Zip a Dataset

!!! example "Optimize and Zip a Dataset"

    === "Python"

       ```python
       from pathlib import Path

       from ultralytics.data.utils import compress_one_image
       from ultralytics.utils.downloads import zip_directory

       # Define dataset directory
       path = Path("path/to/dataset")

       # Optimize images in dataset (optional)
       for f in path.rglob("*.jpg"):
           compress_one_image(f)

       # Zip dataset into 'path/to/dataset.zip'
       zip_directory(path)
       ```

By following these steps, you can contribute a new dataset that integrates well with Ultralytics' existing structure.

## FAQ

### What datasets does Ultralytics support for object detection?

Ultralytics supports a wide variety of datasets for [object detection](https://www.ultralytics.com/glossary/object-detection), including:

- [COCO](detect/coco.md): A large-scale object detection, segmentation, and captioning dataset with 80 object categories.
- [LVIS](detect/lvis.md): An extensive dataset with 1203 object categories, designed for more fine-grained object detection and segmentation.
- [Argoverse](detect/argoverse.md): A dataset containing 3D tracking and motion forecasting data from urban environments with rich annotations.
- [VisDrone](detect/visdrone.md): A dataset with object detection and multi-object tracking data from drone-captured imagery.
- [SKU-110K](detect/sku-110k.md): Featuring dense object detection in retail environments with over 11K images.

These datasets facilitate training robust [Ultralytics YOLO](https://docs.ultralytics.com/models/) models for various object detection applications.

### How do I contribute a new dataset to Ultralytics?

Contributing a new dataset involves several steps:

1. **Collect Images**: Gather images from public databases or personal collections.
2. **Annotate Images**: Apply bounding boxes, segments, or keypoints, depending on the task.
3. **Export Annotations**: Convert annotations into the YOLO `*.txt` format.
4. **Organize Dataset**: Use the folder structure with `train/` and `val/` directories, each containing `images/` and `labels/` subdirectories.
5. **Create a `data.yaml` File**: Include dataset descriptions, classes, and other relevant information.
6. **Optimize Images (Optional)**: Reduce dataset size for efficiency.
7. **Zip Dataset**: Compress the dataset into a zip file.
8. **Document and PR**: Describe your dataset and submit a Pull Request following [Ultralytics Contribution Guidelines](https://docs.ultralytics.com/help/contributing/).

Visit [Contribute New Datasets](#contribute-new-datasets) for a comprehensive guide.

### Why should I use Ultralytics Platform for my dataset?

[Ultralytics Platform](https://platform.ultralytics.com/) offers powerful features for dataset management and analysis, including:

- **Seamless Dataset Management**: Upload, organize, and manage your datasets in one place.
- **Immediate Training Integration**: Use uploaded datasets directly for model training without additional setup.
- **Visualization Tools**: Explore and visualize your dataset images and annotations.
- **Dataset Analysis**: Get insights into your dataset distribution and characteristics.

The platform streamlines the transition from dataset management to model training, making the entire process more efficient. Learn more about [Ultralytics Platform Datasets](https://docs.ultralytics.com/platform/data/).

### What are the unique features of Ultralytics YOLO models for computer vision?

Ultralytics YOLO models provide several unique features for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks:

- **Real-time Performance**: High-speed inference and training capabilities for time-sensitive applications.
- **Versatility**: Support for detection, segmentation, classification, and pose estimation tasks in a unified framework.
- **Pretrained Models**: Access to high-performing, pretrained models for various applications, reducing training time.
- **Extensive Community Support**: Active community and comprehensive documentation for troubleshooting and development.
- **Easy Integration**: Simple API for integrating with existing projects and workflows.

Discover more about YOLO models on the [Ultralytics Models](https://docs.ultralytics.com/models/) page.

### How can I optimize and zip a dataset using Ultralytics tools?

To optimize and zip a dataset using Ultralytics tools, follow this example code:

!!! example "Optimize and Zip a Dataset"

    === "Python"

        ```python
        from pathlib import Path

        from ultralytics.data.utils import compress_one_image
        from ultralytics.utils.downloads import zip_directory

        # Define dataset directory
        path = Path("path/to/dataset")

        # Optimize images in dataset (optional)
        for f in path.rglob("*.jpg"):
            compress_one_image(f)

        # Zip dataset into 'path/to/dataset.zip'
        zip_directory(path)
        ```

This process helps reduce dataset size for more efficient storage and faster download speeds. Learn more on how to [Optimize and Zip a Dataset](#example-code-to-optimize-and-zip-a-dataset).
