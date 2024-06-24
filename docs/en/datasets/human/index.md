---
comments: true
description: Discover the benefits of using the practical and diverse human8 dataset for object detection model testing. Learn to configure and use it via Ultralytics HUB and YOLOv8.
keywords: Ultralytics, human8 dataset, object detection, model testing, dataset configuration, detection approaches, sanity check, training pipelines, YOLOv8
---

# Human Attribute Estimation Datasets Overview

Training a robust and accurate human attribute estimation model requires a comprehensive dataset and proper configuration. This guide introduces the Ultralytics YOLO model for human attribute estimation, supported dataset formats and usage instructions.


## Supported Human Attribute Estimation Dataset Formats

### Ultralytics YOLO Human Attribute Estimation Format

The YOLO Human Attribute Estimation format follows the standard YOLO format for [detection](../detect/index.md), with additional attributes for humans.
The Ultralytics YOLO format is a dataset configuration format that allows you to define the dataset root directory, the relative paths to training/validation/testing image directories or `*.txt` files containing image paths, and a dictionary of class names. Here is an example:

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco8  # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test:  # test images (optional)

# Classes (1 COCO class)
names:
  0: person
```

Labels for this format should be exported to YOLO format with one `*.txt` file per image. If there are no objects in an image, no `*.txt` file is required. The `*.txt` file should be formatted with one row per human in the following format:
weight (kg), height (cm), biological gender (0: female, 1: male), age (years) and ethnicity (0: asian, 1: white, 2: middle eastern, 3: indian, 4: latino, 5: black).

```bash
class x_center y_center width height p_weight p_height p_gender p_age p_ethnicity
```

Box coordinates must be in **normalized xywh** format (from 0 to 1). If your boxes are in pixels, you should divide `x_center` and `width` by image width, and `y_center` and `height` by image height. For this format, the class number should always be `0` (person). 

#### Human Attribute Annotations

- Weight (kg): The weight of the person is annotated in kilograms. This numeric value is essential for applications requiring precise biometric data.
  
- Height (cm): The height of the person is annotated in centimeters. Accurate height measurements are crucial for many analytical and identification purposes.
  
- Biological Gender: biological gender is annotated using binary classification where 0 represents female, 1 represents male. This straightforward categorization simplifies biological gender identification tasks in various applications.

    !!! Biological Classes

        | value | class   |
        |-------|---------|
        | 0     | Females |
        | 1     | Males   |
  
- Age: Age is annotated as an integer. This numerical value represents the person's age in years and is essential for demographic analysis and age-related studies.
  
- Ethnicity: Ethnicity is categorized into six distinct groups, each represented by an integer:

    !!! Ethnicity Classes
  
        | value | class          |
        |-------|----------------|
        | 0     | Asian          |
        | 1     | White          |
        | 2     | Middle Eastern |
        | 3     | Indian         |
        | 4     | Latino         |
        | 5     | Black          |

These categories help in the study of diverse populations and enable the development of models that are inclusive and non-biased.

When using the Ultralytics YOLO format, organize your training and validation images and labels as shown in the [COCO8 dataset](../detect/coco8.md) example below.
<p align="center"><img width="800" src="https://github.com/IvorZhu331/ultralytics/assets/26833433/a55ec82d-2bb5-40f9-ac5c-f935e7eb9f07" alt="Example dataset directory structure"></p>

## Supported Datasets

Currently, the following datasets with Human Attribute Estimation are supported:

- [COCO8-human](coco8-human.md) - A small, 8-image subset of the full COCO dataset suitable for testing workflows and Continuous Integration (CI) checks of human attribute estimation training in the `ultralytics` repository.
