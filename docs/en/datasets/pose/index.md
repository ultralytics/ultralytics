---
comments: true
description: Learn about Ultralytics YOLO format for pose estimation datasets, supported formats, COCO-Pose, COCO8-Pose, Tiger-Pose, and how to add your own dataset.
keywords: pose estimation, Ultralytics, YOLO format, COCO-Pose, COCO8-Pose, Tiger-Pose, dataset conversion, keypoints
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

Here is an example of the label format for pose estimation task:

Format with Dim = 2

```
<class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
```

Format with Dim = 3

```
<class-index> <x> <y> <width> <height> <px1> <py1> <p1-visibility> <px2> <py2> <p2-visibility> <pxn> <pyn> <pn-visibility>
```

In this format, `<class-index>` is the index of the class for the object,`<x> <y> <width> <height>` are coordinates of [bounding box](https://www.ultralytics.com/glossary/bounding-box), and `<px1> <py1> <px2> <py2> ... <pxn> <pyn>` are the pixel coordinates of the keypoints. The coordinates are separated by spaces.

### Dataset YAML format

The Ultralytics framework uses a YAML file format to define the dataset and model configuration for training pose estimation models. Here is an example of the YAML format used for defining a pose dataset:

```yaml
--8<-- "ultralytics/cfg/datasets/coco8-pose.yaml"
```

The `train` and `val` fields specify the paths to the directories containing the training and validation images, respectively.

`names` is a dictionary of class names. The order of the names should match the order of the object class indices in the YOLO dataset files.

(Optional) if the points are symmetric then need flip_idx, like left-right side of human or face. For example if we assume five keypoints of facial landmark: [left eye, right eye, nose, left mouth, right mouth], and the original index is [0, 1, 2, 3, 4], then flip_idx is [1, 0, 2, 4, 3] (just exchange the left-right index, i.e. 0-1 and 3-4, and do not modify others like nose in this example).

## Usage

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo pose train data=coco8-pose.yaml model=yolo11n-pose.pt epochs=100 imgsz=640
        ```

## Supported Datasets

This section outlines the datasets that are compatible with Ultralytics YOLO format and can be used for training [pose estimation](https://docs.ultralytics.com/tasks/pose/) models:

### COCO-Pose

- **Description**: COCO-Pose is a large-scale [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and pose estimation dataset. It is a subset of the popular COCO dataset and focuses on human pose estimation. COCO-Pose includes multiple keypoints for each human instance.
- **Label Format**: Same as Ultralytics YOLO format as described above, with keypoints for human poses.
- **Number of Classes**: 1 (Human).
- **Keypoints**: 17 keypoints including nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles.
- **Usage**: Suitable for training human pose estimation models.
- **Additional Notes**: The dataset is rich and diverse, containing over 200k labeled images.
- [Read more about COCO-Pose](coco.md)

### COCO8-Pose

- **Description**: [Ultralytics](https://www.ultralytics.com/) COCO8-Pose is a small, but versatile pose detection dataset composed of the first 8 images of the COCO train 2017 set, 4 for training and 4 for validation.
- **Label Format**: Same as Ultralytics YOLO format as described above, with keypoints for human poses.
- **Number of Classes**: 1 (Human).
- **Keypoints**: 17 keypoints including nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles.
- **Usage**: Suitable for testing and debugging object detection models, or for experimenting with new detection approaches.
- **Additional Notes**: COCO8-Pose is ideal for sanity checks and [CI checks](https://docs.ultralytics.com/help/CI/).
- [Read more about COCO8-Pose](coco8-pose.md)

### Tiger-Pose

- **Description**: [Ultralytics](https://www.ultralytics.com/) The Tiger Pose dataset comprises 263 images sourced from a [YouTube Video](https://www.youtube.com/watch?v=MIBAT6BGE6U&pp=ygUbVGlnZXIgd2Fsa2luZyByZWZlcmVuY2UubXA0), with 210 images allocated for training and 53 for validation.
- **Label Format**: Same as Ultralytics YOLO format as described above, with 12 keypoints for animal pose and no visible dimension.
- **Number of Classes**: 1 (Tiger).
- **Keypoints**: 12 keypoints.
- **Usage**: Great for animal pose or any other pose that is not human-based.
- [Read more about Tiger-Pose](tiger-pose.md)

### Hand Keypoints

- **Description**: Hand keypoints pose dataset comprises nearly 26K images, with 18776 images allocated for training and 7992 for validation.
- **Label Format**: Same as Ultralytics YOLO format as described above, but with 21 keypoints for human hand and visible dimension.
- **Number of Classes**: 1 (Hand).
- **Keypoints**: 21 keypoints.
- **Usage**: Great for human hand pose estimation and [gesture recognition](https://www.ultralytics.com/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11).
- [Read more about Hand Keypoints](hand-keypoints.md)

### Dog-Pose

- **Description**: The Dog Pose dataset contains approximately 6,000 images, providing a diverse and extensive resource for training and validation of dog pose estimation models.
- **Label Format**: Follows the Ultralytics YOLO format, with annotations for multiple keypoints specific to dog anatomy.
- **Number of Classes**: 1 (Dog).
- **Keypoints**: Includes 24 keypoints tailored to dog poses, such as limbs, joints, and head positions.
- **Usage**: Ideal for training models to estimate dog poses in various scenarios, from research to [real-world applications](https://www.ultralytics.com/blog/custom-training-ultralytics-yolo11-for-dog-pose-estimation).
- [Read more about Dog-Pose](dog-pose.md)

### Adding your own dataset

If you have your own dataset and would like to use it for training pose estimation models with Ultralytics YOLO format, ensure that it follows the format specified above under "Ultralytics YOLO format". Convert your annotations to the required format and specify the paths, number of classes, and class names in the YAML configuration file.

### Conversion Tool

Ultralytics provides a convenient conversion tool to convert labels from the popular [COCO dataset](https://docs.ultralytics.com/datasets/detect/coco/) format to YOLO format:

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

For 2D poses, keypoints include pixel coordinates. For 3D, each keypoint also has a visibility flag. For more details, see [Ultralytics YOLO format](#ultralytics-yolo-format).

### How do I use the COCO-Pose dataset with Ultralytics YOLO?

To use the [COCO-Pose dataset](https://docs.ultralytics.com/datasets/pose/coco/) with Ultralytics YOLO:

1. Download the dataset and prepare your label files in the YOLO format.
2. Create a YAML configuration file specifying paths to training and validation images, keypoint shape, and class names.
3. Use the configuration file for training:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n-pose.pt")  # load pretrained model
    results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640)
    ```

    For more information, visit [COCO-Pose](coco.md) and [train](../../modes/train.md) sections.

### How can I add my own dataset for pose estimation in Ultralytics YOLO?

To add your dataset:

1. Convert your annotations to the Ultralytics YOLO format.
2. Create a YAML configuration file specifying the dataset paths, number of classes, and class names.
3. Use the configuration file to train your model:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n-pose.pt")
    results = model.train(data="your-dataset.yaml", epochs=100, imgsz=640)
    ```

    For complete steps, check the [Adding your own dataset](#adding-your-own-dataset) section.

### What is the purpose of the dataset YAML file in Ultralytics YOLO?

The dataset YAML file in Ultralytics YOLO defines the dataset and model configuration for training. It specifies paths to training, validation, and test images, keypoint shapes, class names, and other configuration options. This structured format helps streamline [dataset management](https://docs.ultralytics.com/datasets/explorer/) and model training. Here is an example YAML format:

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

This tool helps seamlessly integrate COCO datasets into YOLO projects. For details, refer to the [Conversion Tool](#conversion-tool) section and the [data preprocessing guide](https://docs.ultralytics.com/guides/preprocessing_annotated_data/).
