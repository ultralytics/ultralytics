---
comments: true
---

# Datasets Overview

Ultralytics provides support for various datasets to facilitate computer vision tasks such as detection, instance segmentation, pose estimation, classification, and multi-object tracking. Below is a list of the main Ultralytics datasets, followed by a summary of each computer vision task and the respective datasets.

- **Argoverse**: A dataset for 3D tracking and motion forecasting with rich annotations.
- **COCO**: A large-scale object detection, segmentation, and captioning dataset.
- **Global Wheat 2020**: A dataset containing images of wheat heads for object detection and localization.
- **ImageNet**: A large-scale dataset for object detection and image classification.
- **Objects365**: A large-scale, high-quality dataset for object detection.
- **SKU-110K**: A dataset for dense object detection in retail environments.
- **VisDrone**: A dataset for object detection and multi-object tracking in drone-captured imagery.
- **VOC**: The Pascal Visual Object Classes (VOC) dataset for object detection and segmentation.
- **xView**: A dataset for object detection in overhead imagery.

## Dataset by Tasks:

1. [Detection](detect/index.md): Bounding box object detection is a computer vision technique that involves detecting and localizing objects in an image by drawing a bounding box around each object. Datasets: `coco.yaml`, `Objects365.yaml`, `GlobalWheat2020.yaml`, `VisDrone.yaml`, `VOC.yaml`, `xView.yaml`.

2. [Instance Segmentation](segment/index.md): Instance segmentation is a computer vision technique that involves identifying and localizing objects in an image at the pixel level. Datasets: `coco8-seg.yaml`, `coco.yaml`.

3. [Pose Estimation](pose/index.md): Pose estimation is a technique used to determine the pose of the object relative to the camera or the world coordinate system. Datasets: `coco8-pose.yaml`, `coco-pose.yaml`.

4. [Classification](classify/index.md): Image classification is a computer vision task that involves categorizing an image into one or more predefined classes or categories based on its visual content. Datasets: `ImageNet.yaml`, `Argoverse.yaml`, `SKU-110K.yaml`.

5. [Multi-Object Tracking](track/index.md): Multi-object tracking is a computer vision technique that involves detecting and tracking multiple objects over time in a video sequence. Datasets: `Argoverse.yaml`, `VisDrone.yaml`.

Additionally, Ultralytics provides smaller, more focused datasets for more specific tasks like sanity checks and Continuous Integration (CI) tests:

- `coco8.yaml`: Contains the first 4 images from COCO train and COCO val, suitable for quick tests.
- `coco8-pose.yaml`: A smaller dataset for pose estimation tasks, containing a subset of 8 COCO images.
- `coco8-seg.yaml`: A smaller dataset for instance segmentation tasks, containing a subset of 8 COCO images.
