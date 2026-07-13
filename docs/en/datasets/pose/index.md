---
comments: true
description: Learn the Ultralytics YOLO format for pose estimation datasets — COCO-Pose, COCO8-Pose, Dog-Pose, Hand Keypoints, Tiger-Pose — and how to add your own.
keywords: pose estimation, Ultralytics, YOLO format, COCO-Pose, COCO8-Pose, Dog-Pose, Hand Keypoints, Tiger-Pose, dataset conversion, keypoints
title: Pose Estimation Datasets
---

# Pose Estimation Datasets Overview

## Supported Dataset Formats

### Ultralytics YOLO format

The dataset label format used for training YOLO pose models is as follows:

1. One text file per image: Each image in the dataset has a corresponding text file with the same name as the image file and the ".txt" extension.
2. One row per object: Each row in the text file corresponds to one object instance in the image.
3. Object information per row: Each row contains the following information about the object instance:
    - Object class index: An integer representing the class of the object (e.g., 0 for person, 1 for car, etc.).
    - Object center coordinates: The x and y coordinates of the center of the object, normalized to be between 0 and 1.
    - Object width and height: The width and height of the object, normalized to be between 0 and 1.
    - Object keypoint coordinates: The keypoints of the object, normalized to be between 0 and 1.

Here is an example of the label format for a pose estimation task:

Format with 2D keypoints

```
<class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
```

Format with keypoint visibility (includes visibility per point)

```
<class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> <pxn> <pyn> <pn-visibility>
```

In this format, `<class-index>` is the index of the class for the object, `<x> <y> <width> <height>` are the normalized coordinates of the [bounding box](https://www.ultralytics.com/glossary/bounding-box), and `<px1> <py1> <px2> <py2> ... <pxn> <pyn>` are the normalized keypoint coordinates. The visibility channel is optional but useful for datasets that annotate occlusion.

### Dataset YAML format

The Ultralytics framework uses a YAML file format to define the dataset and model configuration for training pose estimation models. Here is an example of the YAML format used for defining a pose dataset:

!!! example "ultralytics/cfg/datasets/coco8-pose.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco8-pose.yaml"
    ```

The `train`, `val`, and `test` fields point to the training, validation, and test images. Each accepts a directory, a list of directories, or a `*.txt` file listing one image path per line (paths starting with `./` resolve relative to the `*.txt` file). A `*.txt` file is useful to train on a subset of a directory, skip unlabeled images, or combine images from multiple sources into one split.

!!! example "Image paths as a `*.txt` file"

    === "dataset.yaml"

        ```yaml
        path: datasets/coco8-pose # dataset root
        train: train.txt # a directory, a list e.g. [images/a, images/b], or a *.txt file
        val: val.txt
        names:
          0: person
        ```

    === "train.txt"

        ```text
        ./images/im0.jpg
        ./images/im1.jpg
        /data/shared/im2.jpg
        ```

`names` is a dictionary of class names. The order of the names should match the order of the object class indices in the YOLO dataset files.

(Optional) `flip_idx` maps each keypoint to its mirror image, so horizontal-flip augmentation keeps left and right consistent on symmetric skeletons such as a human body or face. For five facial landmarks indexed as [left eye, right eye, nose, left mouth, right mouth] = [0, 1, 2, 3, 4], `flip_idx` is [1, 0, 2, 4, 3]: the left-right pairs 0-1 and 3-4 swap, and the nose keeps its own index.

(Optional) `kpt_oks_sigmas` sets custom per-keypoint [OKS](https://docs.ultralytics.com/guides/yolo-performance-metrics/) sigmas used during validation, e.g. `[0.26, 0.25, 0.25, ...]`. The list length must equal the number of keypoints `N` from `kpt_shape`, and every value must be positive. When omitted, the COCO 17-keypoint sigmas are used for `kpt_shape: [17, 3]` and a uniform `1/N` otherwise.

## Usage

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=coco8-pose.yaml model=yolo26n-pose.pt epochs=100 imgsz=640
        ```

## Supported Datasets

This section outlines the datasets that are compatible with Ultralytics YOLO format and can be used for training [pose estimation](../../tasks/pose.md) models:

### COCO-Pose

- **Description**: COCO-Pose is a large-scale human pose estimation dataset covering the COCO 2017 images that contain keypoint-annotated people.
- **Label Format**: Same as Ultralytics YOLO format as described above, with keypoints for human poses.
- **Number of Classes**: 1 (person).
- **Keypoints**: 17 keypoint types including nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles, each with a visibility dimension.
- **Usage**: Suitable for training human pose estimation models.
- **Additional Notes**: The dataset builds on the [COCO Keypoints 2017](http://presentations.cocodataset.org/COCO17-Keypoints-Overview.pdf) challenge: 58,945 images annotated with 156,165 people.
- [Read more about COCO-Pose](coco.md)

### COCO8-Pose

- **Description**: [Ultralytics](https://www.ultralytics.com/) COCO8-Pose is a small, but versatile pose estimation dataset composed of the first 8 images of the COCO train 2017 set, 4 for training and 4 for validation.
- **Label Format**: Same as Ultralytics YOLO format as described above, with keypoints for human poses.
- **Number of Classes**: 1 (person).
- **Keypoints**: 17 keypoint types including nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles, each with a visibility dimension.
- **Usage**: Suitable for testing and debugging pose estimation models, or for experimenting with new keypoint-detection approaches.
- **Additional Notes**: COCO8-Pose is ideal for sanity checks and [CI checks](../../help/CI.md).
- [Read more about COCO8-Pose](coco8-pose.md)

### Dog-Pose

- **Description**: The [Ultralytics](https://www.ultralytics.com/) Dog-Pose dataset contains 6,773 training and 1,703 validation images for canine keypoint estimation.
- **Label Format**: Follows the Ultralytics YOLO format, with annotations for multiple keypoints specific to dog anatomy.
- **Number of Classes**: 1 (dog).
- **Keypoints**: 24 keypoints, each with a visibility dimension, tailored to dog poses such as limbs, joints, and head positions.
- **Usage**: Ideal for training models to estimate dog poses in various scenarios, from research to [real-world applications](https://www.ultralytics.com/blog/custom-training-ultralytics-yolo11-for-dog-pose-estimation).
- **Additional Notes**: Images and annotations are sourced from the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).
- [Read more about Dog-Pose](dog-pose.md)

### Hand Keypoints

- **Description**: The [Ultralytics](https://www.ultralytics.com/) Hand Keypoints dataset comprises 26,768 images, with 18,776 allocated for training and 7,992 for validation.
- **Label Format**: Same as the Ultralytics YOLO format described above, but with 21 keypoints for a human hand and a visibility dimension.
- **Number of Classes**: 1 (hand).
- **Keypoints**: 21 keypoints.
- **Usage**: Great for human hand pose estimation and [gesture recognition](https://www.ultralytics.com/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11).
- **Additional Notes**: Keypoint annotations are generated using [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) for consistent labeling.
- [Read more about Hand Keypoints](hand-keypoints.md)

### Tiger-Pose

- **Description**: The [Ultralytics](https://www.ultralytics.com/) Tiger-Pose dataset comprises 263 images sourced from a [YouTube video](https://www.youtube.com/watch?v=MIBAT6BGE6U), with 210 images allocated for training and 53 for validation.
- **Label Format**: Same as Ultralytics YOLO format as described above, with 12 keypoints for animal pose and no visibility dimension.
- **Number of Classes**: 1 (tiger).
- **Keypoints**: 12 keypoints.
- **Usage**: Great for animal pose or any other pose that is not human-based.
- **Additional Notes**: Released under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
- [Read more about Tiger-Pose](tiger-pose.md)

### Adding your own dataset

If you have your own dataset and would like to use it for training pose estimation models with Ultralytics YOLO format, ensure that it follows the format specified above under "Ultralytics YOLO format". Convert your annotations to the required format and specify the paths, number of classes, and class names in the YAML configuration file.

To skip the conversion step entirely, [Ultralytics Platform](https://platform.ultralytics.com/) lets you upload raw images, annotate keypoints in the browser, and train on the resulting dataset directly.

### Conversion Tool

Ultralytics provides a convenient conversion tool to convert labels from the popular [COCO dataset](../detect/coco.md) format to YOLO format:

!!! example

    === "Python"

        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco(labels_dir="path/to/coco/annotations/", use_keypoints=True)
        ```

This conversion tool can be used to convert the COCO dataset or any dataset in the COCO format to the Ultralytics YOLO format. The `use_keypoints` parameter specifies whether to include keypoints (for pose estimation) in the converted labels.

## FAQ

### What is the Ultralytics YOLO format for pose estimation?

The Ultralytics YOLO format for pose estimation datasets involves labeling each image with a corresponding text file. Each row of the text file stores information about an object instance:

- Object class index
- Object center coordinates (normalized x and y)
- Object width and height (normalized)
- Object keypoint coordinates (normalized pxn and pyn)

For 2D poses, keypoints include normalized x and y coordinates. With a visibility dimension, each keypoint also has a visibility flag. For more details, see [Ultralytics YOLO format](#ultralytics-yolo-format).

### How do I use the COCO-Pose dataset with Ultralytics YOLO?

`coco-pose.yaml` ships with the package and downloads the images and labels on first use, so no manual preparation is needed:

```python
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")  # load pretrained model
results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640)
```

For the dataset details see [COCO-Pose](coco.md), and the [Train](../../modes/train.md) page for the full argument list.

### How can I add my own dataset for pose estimation in Ultralytics YOLO?

To add your dataset:

1. Convert your annotations to the Ultralytics YOLO format.
2. Create a YAML configuration file specifying the dataset paths, number of classes, and class names.
3. Use the configuration file to train your model:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n-pose.pt")
    results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
    ```

    For complete steps, check the [Adding your own dataset](#adding-your-own-dataset) section.

### What is the purpose of the dataset YAML file in Ultralytics YOLO?

The dataset YAML file in Ultralytics YOLO defines the dataset and model configuration for training. It specifies paths to training, validation, and test images, keypoint shapes, class names, and other configuration options. This structured format helps streamline dataset management and model training. Here is an example YAML format:

```yaml
--8<-- "ultralytics/cfg/datasets/coco8-pose.yaml"
```

Read more about creating YAML configuration files in [Dataset YAML format](#dataset-yaml-format).

### How can I convert COCO dataset labels to Ultralytics YOLO format for pose estimation?

Ultralytics provides a conversion tool to convert COCO dataset labels to the YOLO format, including keypoint information:

```python
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="path/to/coco/annotations/", use_keypoints=True)
```

This tool helps seamlessly integrate COCO datasets into YOLO projects. For details, refer to the [Conversion Tool](#conversion-tool) section and the [data preprocessing guide](../../guides/preprocessing-annotated-data.md).
