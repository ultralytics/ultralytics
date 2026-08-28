---
comments: true
description: Tune Ultralytics YOLO data augmentation — HSV, geometric transforms, mosaic, mixup, cutmix, and copy-paste — from Python, the CLI, or a YAML config.
keywords: YOLO data augmentation, computer vision, deep learning, image transformations, model training, Ultralytics YOLO, HSV adjustments, geometric transformations, mosaic augmentation
---

# Data Augmentation using Ultralytics YOLO

<p align="center">
  <img width="100%" src="https://cdn.ul.run/i/d7b5907c9df57a0806398431ef5bb4f3.avif" alt="YOLO data augmentation examples showing original and augmented images for training">
</p>

## Introduction

[Data augmentation](https://www.ultralytics.com/glossary/data-augmentation) is a crucial technique in computer vision that artificially expands your training dataset by applying various transformations to existing images. When training [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models like Ultralytics YOLO, data augmentation helps improve model robustness, reduces overfitting, and enhances generalization to real-world scenarios.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/e-TwqFtay90"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to use Mosaic, MixUp & more Data Augmentations to help Ultralytics YOLO Models generalize better 🚀
</p>

### Why Data Augmentation Matters

Data augmentation serves multiple critical purposes in training computer vision models:

- **Expanded Dataset**: By creating variations of existing images, you can effectively increase your training dataset size without collecting new data.
- **Improved Generalization**: Models learn to recognize objects under various conditions, making them more robust in real-world applications.
- **Reduced Overfitting**: By introducing variability in the training data, models are less likely to memorize specific image characteristics.
- **Enhanced Performance**: Models trained with proper augmentation typically achieve better [accuracy](https://www.ultralytics.com/glossary/accuracy) on validation and test sets.

Ultralytics YOLO's implementation provides a comprehensive suite of augmentation techniques, each serving specific purposes and contributing to model performance in different ways. This guide explores the augmentation settings below, helping you understand when and how to use them effectively in your projects.

### Example Configurations

You can customize each parameter using the Python API, the command line interface (CLI), or a configuration file. Below are examples of how to set up data augmentation in each method.

!!! example "Configuration Examples"

    === "Python"

        ```python
        import albumentations as A

        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")

        # Training with custom augmentation parameters
        model.train(data="coco8.yaml", epochs=100, hsv_h=0.03, hsv_s=0.6, hsv_v=0.5)

        # Training with every configurable augmentation disabled (disabled values omitted for clarity)
        model.train(
            data="coco8.yaml",
            epochs=100,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            translate=0.0,
            scale=0.0,
            fliplr=0.0,
            mosaic=0.0,
            erasing=0.0,
            auto_augment=None,
        )

        # Training with custom Albumentations transforms (Python API only)
        custom_transforms = [
            A.Blur(blur_limit=7, p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.5),
        ]
        model.train(data="coco8.yaml", epochs=100, augmentations=custom_transforms)
        ```

    === "CLI"

        ```bash
        # Training with custom augmentation parameters
        yolo detect train data=coco8.yaml model=yolo26n.pt epochs=100 hsv_h=0.03 hsv_s=0.6 hsv_v=0.5
        ```

!!! note "Zeroing the augmentation arguments does not disable Albumentations"

    The transforms listed on this page are the ones you control through training arguments. Ultralytics also applies a small [Albumentations](../integrations/albumentations.md) set — blur, median blur, grayscale, and CLAHE, each at `p=0.01` — whenever the `albumentations` package is installed, and no argument in `default.yaml` switches it off. Uninstall the package to remove it, or pass your own list to [`augmentations`](#custom-albumentations-transforms-augmentations) to replace it.

#### Using a configuration file

You can define all training parameters, including augmentations, in a YAML configuration file (e.g., `train_custom.yaml`). The `mode` parameter is only required when using the CLI. This new YAML file will then override [the default one](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml) located in the `ultralytics` package.

```yaml
# train_custom.yaml
# 'mode' is required only for CLI usage
mode: train
data: coco8.yaml
model: yolo26n.pt
epochs: 100
hsv_h: 0.03
hsv_s: 0.6
hsv_v: 0.5
```

Then launch the training with the Python API:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a COCO-pretrained YOLO26n model
        model = YOLO("yolo26n.pt")

        # Train the model with custom configuration
        model.train(cfg="train_custom.yaml")
        ```

    === "CLI"

        ```bash
        # Train the model with custom configuration
        yolo detect train model="yolo26n.pt" cfg=train_custom.yaml
        ```

## Color Space Augmentations

### Hue Adjustment (`hsv_h`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ hsv_h }}`
- **Usage**: Shifts image colors while preserving their relationships. The `hsv_h` hyperparameter defines the shift magnitude, with the final adjustment randomly chosen between `-hsv_h` and `hsv_h`. For example, with `hsv_h=0.3`, the shift is randomly selected within `-0.3` to `0.3`. For values above `0.5`, the hue shift wraps around the color wheel, that's why the augmentations look the same between `0.5` and `-0.5`.
- **Purpose**: Particularly useful for outdoor scenarios where lighting conditions can dramatically affect object appearance. For example, a banana might look more yellow under bright sunlight but more greenish indoors.
- **Ultralytics' implementation**: [RandomHSV](../reference/data/augment.md#ultralytics.data.augment.RandomHSV)

|                                                **`-0.5`**                                                 |                                                **`-0.25`**                                                 |                                                     **`0.0`**                                                     |                                                **`0.25`**                                                 |                                                 **`0.5`**                                                 |
| :-------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/4e6d4b589ee58106c9290a156a6dcb52.avif" alt="Hue shift -0.5 augmentation"/> | <img src="https://cdn.ul.run/i/456d42998989b8537b269a63b5b4d164.avif" alt="Hue shift -0.25 augmentation"/> | <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/7c07282fbb39f3a99e80fc98d80809e0.avif" alt="Hue shift 0.25 augmentation"/> | <img src="https://cdn.ul.run/i/4e6d4b589ee58106c9290a156a6dcb52.avif" alt="Hue shift -0.5 augmentation"/> |

### Saturation Adjustment (`hsv_s`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ hsv_s }}`
- **Usage**: Modifies the intensity of colors in the image. The `hsv_s` hyperparameter defines the shift magnitude, with the final adjustment randomly chosen between `-hsv_s` and `hsv_s`. For example, with `hsv_s=0.7`, the intensity is randomly selected within `-0.7` to `0.7`.
- **Purpose**: Helps models handle varying weather conditions and camera settings. For example, a red traffic sign might appear highly vivid on a sunny day but look dull and faded in foggy conditions.
- **Ultralytics' implementation**: [RandomHSV](../reference/data/augment.md#ultralytics.data.augment.RandomHSV)

|                                                      **`-1.0`**                                                      |                                                 **`-0.5`**                                                 |                                                     **`0.0`**                                                     |                                                 **`0.5`**                                                 |                                                    **`1.0`**                                                    |
| :------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/a57a7883eec214ace18bddd857d04f73.avif" alt="Saturation -1.0 grayscale augmentation"/> | <img src="https://cdn.ul.run/i/1d41030424c2d35931517676fa56f826.avif" alt="Saturation -0.5 augmentation"/> | <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/a62eef95a0653641f0c540ad965a419c.avif" alt="Saturation 0.5 augmentation"/> | <img src="https://cdn.ul.run/i/eaf6309941b5bfb2d596fe6fa6c74fad.avif" alt="Saturation 1.0 vivid augmentation"/> |

### Brightness Adjustment (`hsv_v`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ hsv_v }}`
- **Usage**: Changes the brightness of the image. The `hsv_v` hyperparameter defines the shift magnitude, with the final adjustment randomly chosen between `-hsv_v` and `hsv_v`. For example, with `hsv_v=0.4`, the intensity is randomly selected within `-0.4` to `0.4`.
- **Purpose**: Essential for training models that need to perform in different lighting conditions. For example, a red apple might look bright in sunlight but much darker in the shade.
- **Ultralytics' implementation**: [RandomHSV](../reference/data/augment.md#ultralytics.data.augment.RandomHSV)

|                                                   **`-1.0`**                                                    |                                                 **`-0.5`**                                                 |                                                     **`0.0`**                                                     |                                                 **`0.5`**                                                 |                                                    **`1.0`**                                                     |
| :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/295432ad698c8056a93b1efd510266ce.avif" alt="Brightness -1.0 dark augmentation"/> | <img src="https://cdn.ul.run/i/350a43d3bd18912adf4e30a9045fef45.avif" alt="Brightness -0.5 augmentation"/> | <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/02dcfcbd0d88e9411063c8a47c14da8a.avif" alt="Brightness 0.5 augmentation"/> | <img src="https://cdn.ul.run/i/22b9bbb7b945b6d176de083b45d2936e.avif" alt="Brightness 1.0 bright augmentation"/> |

## Geometric Transformations

### Rotation (`degrees`)

- **Range**: `0.0` to `180`
- **Default**: `{{ degrees }}`
- **Usage**: Rotates images randomly within the specified range. The `degrees` hyperparameter defines the rotation angle, with the final adjustment randomly chosen between `-degrees` and `degrees`. For example, with `degrees=10.0`, the rotation is randomly selected within `-10.0` to `10.0`.
- **Purpose**: Crucial for applications where objects can appear at different orientations. For example, in aerial drone imagery, vehicles can be oriented in any direction, requiring models to recognize objects regardless of their rotation.
- **Ultralytics' implementation**: [RandomPerspective](../reference/data/augment.md#ultralytics.data.augment.RandomPerspective)

|                                                    **`-180`**                                                    |                                                    **`-90`**                                                    |                                                     **`0.0`**                                                     |                                                    **`90`**                                                    |                                                    **`180`**                                                    |
| :--------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/7be174ad67e142d6ca59a0f399b12470.avif" alt="Rotation -180 degrees augmentation"/> | <img src="https://cdn.ul.run/i/3cc73eab112138507235ddb7ad335c02.avif" alt="Rotation -90 degrees augmentation"/> | <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/72c28e437aee0f5baa1a5517396761c9.avif" alt="Rotation 90 degrees augmentation"/> | <img src="https://cdn.ul.run/i/7be174ad67e142d6ca59a0f399b12470.avif" alt="Rotation 180 degrees augmentation"/> |

### Translation (`translate`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ translate }}`
- **Usage**: Shifts images horizontally and vertically by a random fraction of the image size. The `translate` hyperparameter defines the shift magnitude, with the final adjustment randomly chosen twice (once for each axis) within the range `-translate` and `translate`. For example, with `translate=0.5`, the translation is randomly selected within `-0.5` to `0.5` on the x-axis, and another independent random value is selected within the same range on the y-axis.
- **Purpose**: Helps models learn to detect partially visible objects and improves robustness to object position. For example, in vehicle damage assessment applications, car parts may appear fully or partially in frame depending on the photographer's position and distance, the translation augmentation will teach the model to recognize these features regardless of their completeness or position.
- **Ultralytics' implementation**: [RandomPerspective](../reference/data/augment.md#ultralytics.data.augment.RandomPerspective)
- **Note**: For simplicity, the translations applied below are the same each time for both `x` and `y` axes. Values `-1.0` and `1.0` are not shown as they would translate the image completely out of the frame.

|                                                    **`-0.5`**                                                     |                                                    **`-0.25`**                                                     |                                                     **`0.0`**                                                     |                                                    **`0.25`**                                                     |                                                    **`0.5`**                                                     |
| :---------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/1f528a069b8f4de3c3f2547f60afcf6f.avif" alt="Translation -0.5 shift augmentation"/> | <img src="https://cdn.ul.run/i/e62ccbb87047bb496cea12a8ebbbb18d.avif" alt="Translation -0.25 shift augmentation"/> | <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/5bbbce7b4f9af2da21d3940e6c2a3621.avif" alt="Translation 0.25 shift augmentation"/> | <img src="https://cdn.ul.run/i/c23cb467efff66b12fc26dc9a51388f0.avif" alt="Translation 0.5 shift augmentation"/> |

### Scale (`scale`)

- **Range**: `0.0` - `1.0` as a float, or an explicit `(min, max)` tuple
- **Default**: `{{ scale }}`
- **Usage**: Resizes images by a random factor within the specified range. As a float, the `scale` hyperparameter defines the scaling gain, with the final factor randomly chosen between `1-scale` and `1+scale`. For example, with `scale=0.5`, the scaling is randomly selected within `0.5` to `1.5`. As a tuple, `scale` sets that range directly, so `scale=(0.5, 2.0)` samples the factor between `0.5` and `2.0`.
- **Purpose**: Enables models to handle objects at different distances and sizes. For example, in autonomous driving applications, vehicles can appear at various distances from the camera, requiring the model to recognize them regardless of their size.
- **Ultralytics' implementation**: [RandomPerspective](../reference/data/augment.md#ultralytics.data.augment.RandomPerspective)
- **Note**:
    - The value `-1.0` is not shown as it would make the image disappear, while `1.0` simply results in a 2x zoom.
    - The values displayed in the table below are the realized scale deltas, not the values you pass to the `scale` hyperparameter.
    - The float form is validated to the `0.0` - `1.0` range, and a value outside it raises a `ValueError`. To sample a factor beyond a 2x zoom, pass the `(min, max)` tuple form instead.
    - The tuple form applies to the geometric tasks only. Classification training derives its own crop range from the float (`1.0 - scale` to `1.0`), so passing a tuple there raises a `TypeError`.

|                                                   **`-0.5`**                                                   |                                                   **`-0.25`**                                                   |                                                     **`0.0`**                                                     |                                                   **`0.25`**                                                   |                                                   **`0.5`**                                                   |
| :------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/ffef00142c674f45025e24e775d67ea3.avif" alt="Scale 0.5x zoom out augmentation"/> | <img src="https://cdn.ul.run/i/95e4e1b9fe2ad3ff0024e5e54d803894.avif" alt="Scale 0.75x zoom out augmentation"/> | <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/f04c2d505a2019ff718506d01c560834.avif" alt="Scale 1.25x zoom in augmentation"/> | <img src="https://cdn.ul.run/i/0ae6c59cb9ceace49986caa9d66d7c31.avif" alt="Scale 1.5x zoom in augmentation"/> |

### Shear (`shear`)

- **Range**: `-180` to `+180`
- **Default**: `{{ shear }}`
- **Usage**: Introduces a geometric transformation that skews the image along both x-axis and y-axis, effectively shifting parts of the image in one direction while maintaining parallel lines. The `shear` hyperparameter defines the shear angle, with the final adjustment randomly chosen between `-shear` and `shear`. For example, with `shear=10.0`, the shear is randomly selected within `-10` to `10` on the x-axis, and another independent random value is selected within the same range on the y-axis.
- **Purpose**: Helps models generalize to variations in viewing angles caused by slight tilts or oblique viewpoints. For instance, in traffic monitoring, objects like cars and road signs may appear slanted due to non-perpendicular camera placements. Applying shear augmentation ensures the model learns to recognize objects despite such skewed distortions.
- **Ultralytics' implementation**: [RandomPerspective](../reference/data/augment.md#ultralytics.data.augment.RandomPerspective)
- **Note**:
    - `shear` values can rapidly distort the image, so it's recommended to start with small values and gradually increase them.
    - Unlike perspective transformations, shear does not introduce depth or vanishing points but instead distorts the shape of objects by changing their angles while keeping opposite sides parallel.

|                                                  **`-10`**                                                   |                                                  **`-5`**                                                   |                                                     **`0.0`**                                                     |                                                  **`5`**                                                   |                                                  **`10`**                                                   |
| :----------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/eff8321f05415a71a022e3288e2ea16d.avif" alt="Shear -10 degrees augmentation"/> | <img src="https://cdn.ul.run/i/531df19c682dd79420f5a7eba332248a.avif" alt="Shear -5 degrees augmentation"/> | <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/84915017881c3a326fc8dd154df45b98.avif" alt="Shear 5 degrees augmentation"/> | <img src="https://cdn.ul.run/i/d077467c832b1a4e1ff54f71559f1414.avif" alt="Shear 10 degrees augmentation"/> |

### Perspective (`perspective`)

- **Range**: `0.0` - `0.001`
- **Default**: `{{ perspective }}`
- **Usage**: Applies a full perspective transformation along both x-axis and y-axis, simulating how objects appear when viewed from different depths or angles. The `perspective` hyperparameter defines the perspective magnitude, with the final adjustment randomly chosen between `-perspective` and `perspective`. For example, with `perspective=0.001`, the perspective is randomly selected within `-0.001` to `0.001` on the x-axis, and another independent random value is selected within the same range on the y-axis.
- **Purpose**: Perspective augmentation is crucial for handling extreme viewpoint changes, especially in scenarios where objects appear foreshortened or distorted due to perspective shifts. For example, in drone-based object detection, buildings, roads, and vehicles can appear stretched or compressed depending on the drone's tilt and altitude. By applying perspective transformations, models learn to recognize objects despite these perspective-induced distortions, improving their robustness in real-world deployments.
- **Ultralytics' implementation**: [RandomPerspective](../reference/data/augment.md#ultralytics.data.augment.RandomPerspective)

|                                                  **`-0.001`**                                                   |                                                  **`-0.0005`**                                                   |                                                     **`0.0`**                                                     |                                                  **`0.0005`**                                                   |                                                  **`0.001`**                                                   |
| :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/e891861028e8d1edf54fdd91bfaaacb8.avif" alt="Perspective -0.001 transformation"/> | <img src="https://cdn.ul.run/i/4a994d1cfba34dffe511575e198b54fb.avif" alt="Perspective -0.0005 transformation"/> | <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/3199478042f22f6016acfe72671834a8.avif" alt="Perspective 0.0005 transformation"/> | <img src="https://cdn.ul.run/i/56fe2b0375154a2601142b0b0a4004ca.avif" alt="Perspective 0.001 transformation"/> |

### Flip Up-Down (`flipud`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ flipud }}`
- **Usage**: Performs a vertical flip by inverting the image along the y-axis. This transformation mirrors the entire image upside-down but preserves all spatial relationships between objects. The flipud hyperparameter defines the probability of applying the transformation, with a value of `flipud=1.0` ensuring that all images are flipped and a value of `flipud=0.0` disabling the transformation entirely. For example, with `flipud=0.5`, each image has a 50% chance of being flipped upside-down.
- **Purpose**: Useful for scenarios where objects can appear upside down. For example, in robotic vision systems, objects on conveyor belts or robotic arms may be picked up and placed in various orientations. Vertical flipping helps the model recognize objects regardless of their top-down positioning.
- **Ultralytics' implementation**: [RandomFlip](../reference/data/augment.md#ultralytics.data.augment.RandomFlip)

|                                                 **`flipud` off**                                                  |                                                 **`flipud` on**                                                  |
| :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/fc3811926069377142fa93fe8480e3c7.avif" alt="Vertical flip augmentation enabled"/> |

### Flip Left-Right (`fliplr`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ fliplr }}`
- **Usage**: Performs a horizontal flip by mirroring the image along the x-axis. This transformation swaps the left and right sides while maintaining spatial consistency, which helps the model generalize to objects appearing in mirrored orientations. The `fliplr` hyperparameter defines the probability of applying the transformation, with a value of `fliplr=1.0` ensuring that all images are flipped and a value of `fliplr=0.0` disabling the transformation entirely. For example, with `fliplr=0.5`, each image has a 50% chance of being flipped left to right.
- **Purpose**: Horizontal flipping is widely used in object detection, pose estimation, and facial recognition to improve robustness against left-right variations. For example, in autonomous driving, vehicles and pedestrians can appear on either side of the road, and horizontal flipping helps the model recognize them equally well in both orientations.
- **Ultralytics' implementation**: [RandomFlip](../reference/data/augment.md#ultralytics.data.augment.RandomFlip)

|                                                 **`fliplr` off**                                                  |                                                  **`fliplr` on**                                                   |
| :---------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/f6df36016df7e0aae5b8f77f5c6a13b8.avif" alt="Horizontal flip augmentation enabled"/> |

### BGR Channel Swap (`bgr`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ bgr }}`
- **Usage**: Swaps the color channels of an image from RGB to BGR, altering the order in which colors are represented. The `bgr` hyperparameter defines the probability of applying the transformation, with `bgr=1.0` ensuring all images undergo the channel swap and `bgr=0.0` disabling it. For example, with `bgr=0.5`, each image has a 50% chance of being converted from RGB to BGR.
- **Purpose**: Increases robustness to different color channel orderings. For example, when training models that must work across various camera systems and imaging libraries where RGB and BGR formats may be inconsistently used, or when deploying models to environments where the input color format might differ from the training data.
- **Ultralytics' implementation**: [Format](../reference/data/augment.md#ultralytics.data.augment.Format)

|                                                   **`bgr` off**                                                   |                                                **`bgr` on**                                                 |
| :---------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/6168c300c712929da91e22f3d924079c.avif" alt="BGR channel swap augmentation"/> |

### Mosaic (`mosaic`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ mosaic }}`
- **Usage**: Combines four training images into one. The `mosaic` hyperparameter defines the probability of applying the transformation, with `mosaic=1.0` ensuring that all images are combined and `mosaic=0.0` disabling the transformation. For example, with `mosaic=0.5`, each image has a 50% chance of being combined with three other images.
- **Purpose**: Highly effective for improving small object detection and context understanding. For example, in wildlife conservation projects where animals may appear at various distances and scales, mosaic augmentation helps the model learn to recognize the same species across different sizes, partial occlusions, and environmental contexts by artificially creating diverse training samples from limited data.
- **Ultralytics' implementation**: [Mosaic](../reference/data/augment.md#ultralytics.data.augment.Mosaic)
- **Note**:
    - Even if the `mosaic` augmentation makes the model more robust, it can also make the training process more challenging.
    - The `mosaic` augmentation can be disabled near the end of training by setting `close_mosaic` to the number of epochs before completion when it should be turned off. For example, if `epochs` is set to `200` and `close_mosaic` is set to `20`, the `mosaic` augmentation will be disabled after `180` epochs. If `close_mosaic` is set to `0`, the `mosaic` augmentation will be enabled for the entire training process.
    - Closing the mosaic also disables `copy_paste`, `mixup`, and `cutmix` at the same epoch. The four are switched off together, so the final epochs train without them while every other augmentation — the geometric transforms, HSV, flips, and Albumentations — keeps running. Note that `copy_paste` in its default `flip` mode works within a single image rather than combining several.
    - The center of the generated mosaic is determined using random values, and can either be inside the image or outside of it.
    - The current implementation of the `mosaic` augmentation combines the current image with 3 others, drawn from a buffer of recently loaded images, or from anywhere in the dataset when `cache='ram'`. Either way they are sampled with replacement, so the same image can appear more than once in a single mosaic.

|                                                 **`mosaic` off**                                                  |                                                  **`mosaic` on**                                                  |
| :---------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/dd4cc47c4b731ee72e22f2fd1ce9e506.avif" alt="Mosaic 4-image augmentation enabled"/> |

### Mixup (`mixup`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ mixup }}`
- **Usage**: Blends two images and their labels with given probability. The `mixup` hyperparameter defines the probability of applying the transformation, with `mixup=1.0` ensuring that all images are mixed and `mixup=0.0` disabling the transformation. For example, with `mixup=0.5`, each image has a 50% chance of being mixed with another image.
- **Purpose**: Improves model robustness and reduces overfitting. For example, in retail product recognition systems, mixup helps the model learn more robust features by blending images of different products, teaching it to identify items even when they're partially visible or obscured by other products on crowded store shelves.
- **Ultralytics' implementation**: [Mixup](../reference/data/augment.md#ultralytics.data.augment.MixUp)
- **Note**:
    - The `mixup` ratio is a random value picked from a `np.random.beta(32.0, 32.0)` beta distribution, meaning each image contributes approximately 50%, with slight variations.

|                                         **First image, `mixup` off**                                         |                                         **Second image, `mixup` off**                                         |                                                  **`mixup` on**                                                   |
| :----------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="First image for MixUp blending"/> | <img src="https://cdn.ul.run/i/306040505880e6ea261470faf700ff63.avif" alt="Second image for MixUp blending"/> | <img src="https://cdn.ul.run/i/e8d84e6f0d3f19a83962655203e76861.avif" alt="MixUp blending augmentation enabled"/> |

### CutMix (`cutmix`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ cutmix }}`
- **Usage**: Cuts a rectangular region from one image and pastes it onto another image with given probability. The `cutmix` hyperparameter defines the probability of applying the transformation, with `cutmix=1.0` ensuring that all images undergo this transformation and `cutmix=0.0` disabling it completely. For example, with `cutmix=0.5`, each image has a 50% chance of having a region replaced with a patch from another image.
- **Purpose**: Enhances model performance by creating realistic occlusion scenarios while maintaining local feature integrity. For example, in autonomous driving systems, cutmix helps the model learn to recognize vehicles or pedestrians even when they're partially occluded by other objects, improving detection accuracy in complex real-world environments with overlapping objects.
- **Ultralytics' implementation**: [CutMix](../reference/data/augment.md#ultralytics.data.augment.CutMix)
- **Note**:
    - The size and position of the cut region is determined randomly for each application.
    - Unlike mixup which blends pixel values globally, `cutmix` maintains the original pixel intensities within the cut regions, preserving local features.
    - A region is pasted into the target image only if it does not overlap with any existing bounding box. Additionally, only the bounding boxes that retain enough of their original area within the pasted region are preserved.
    - This minimum bounding box area threshold cannot be changed with the current implementation. It is `0.1` (10%) for detection labels and `0.01` (1%) once the labels carry segments.

|                                    **First image, `cutmix` off**                                     |                                    **Second image, `cutmix` off**                                     |                                              **`cutmix` on**                                              |
| :--------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/2fe1ab11a1401421a564f53a3555dc9c.avif" alt="First image for CutMix"/> | <img src="https://cdn.ul.run/i/0ac9c52812458ada12c57069d2de79c3.avif" alt="Second image for CutMix"/> | <img src="https://cdn.ul.run/i/5c6b7c226fec2a90a78a2077c5c46045.avif" alt="CutMix augmentation enabled"/> |

## Copy-Paste Augmentations

### Copy-Paste (`copy_paste`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ copy_paste }}`
- **Usage**: Requires polygon labels, so it applies to segment and OBB tasks; this augmentation copies objects within or between images, controlled by the [`copy_paste_mode`](#copy-paste-mode-copy_paste_mode). In `flip` mode, `copy_paste` is the fraction of eligible objects copied: an image with six eligible objects gains three copies at `copy_paste=0.5`. In `mixup` mode, the same value also controls the probability that copy-paste runs. `copy_paste=0.0` disables the transformation.
- **Purpose**: Particularly useful for instance segmentation tasks and rare object classes. For example, in industrial defect detection where certain types of defects appear infrequently, copy-paste augmentation can artificially increase the occurrence of these rare defects by copying them from one image to another, helping the model better learn these underrepresented cases without requiring additional defective samples.
- **Ultralytics' implementation**: [CopyPaste](../reference/data/augment.md#ultralytics.data.augment.CopyPaste)
- **Note**:
    - As shown in the video below, the `copy_paste` augmentation can be used to copy objects from one image to another.
    - Once an object is selected for copying, its IoA is computed against all objects already present in the target image, regardless of `copy_paste_mode`. The object is pasted only if all IoA values are below `0.3` (30%); it is not pasted if any IoA value is `0.3` or higher.
    - The IoA threshold cannot be changed with the current implementation and is set to `0.3` by default.

|                                               **`copy_paste` off**                                                |                                **`copy_paste` on with `copy_paste_mode=flip`**                                |                                                               **Visualize the `copy_paste` process**                                                               |
| :---------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/9fbbd723bae594447ebecb1ca2be49f7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/8206af761dc359036f46085558ae50c0.avif" alt="Copy-paste augmentation enabled"/> | <video src="https://cdn.ul.run/v/b8ef0d0f394eb89c1deed2ec17fe54a2.mp4" autoplay loop muted playsinline aria-label="Copy-paste augmentation animated demo"></video> |

### Copy-Paste Mode (`copy_paste_mode`)

- **Options**: `'flip'`, `'mixup'`
- **Default**: `'{{ copy_paste_mode }}'`
- **Usage**: Determines the method used for [copy-paste](#copy-paste-copy_paste) augmentation. If set to `'flip'`, the objects come from the same image, while `'mixup'` allows objects to be copied from different images.
- **Purpose**: Allows flexibility in how copied objects are integrated into target images.
- **Ultralytics' implementation**: [CopyPaste](../reference/data/augment.md#ultralytics.data.augment.CopyPaste)
- **Note**:
    - The IoA principle is the same for both `copy_paste_mode` options, but the way the objects are copied is different.
    - Depending on the image size, objects may sometimes be copied partially or entirely outside the frame.
    - Depending on the quality of polygon annotations, copied objects may have slight shape variations compared to the originals.

|                                              **Reference image**                                              |                                         **Chosen image for `copy_paste`**                                         |                             **`copy_paste` on with `copy_paste_mode=mixup`**                             |
| :-----------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/306040505880e6ea261470faf700ff63.avif" alt="Second image for MixUp blending"/> | <img src="https://cdn.ul.run/i/9fbbd723bae594447ebecb1ca2be49f7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/0fffa641272ae757fc03cb3141aeed88.avif" alt="Copy-paste with MixUp mode"/> |

## Classification-Specific Augmentations

### Auto Augment (`auto_augment`)

- **Options**: `'randaugment'`, `'autoaugment'`, `'augmix'`, `None`
- **Default**: `'{{ auto_augment }}'`
- **Usage**: Applies automated augmentation policies for classification. The `'randaugment'` option uses RandAugment, `'autoaugment'` uses AutoAugment, and `'augmix'` uses AugMix. Setting to `None` disables automated augmentation.
- **Purpose**: Optimizes augmentation strategies automatically for classification tasks. The differences are the following:
    - **AutoAugment**: This mode applies predefined augmentation policies learned from datasets like ImageNet, CIFAR10, and SVHN. Users can select these existing policies but cannot train new ones within Torchvision. To discover optimal augmentation strategies for specific datasets, external libraries or custom implementations would be necessary. Reference to the [AutoAugment paper](https://arxiv.org/abs/1805.09501).
    - **RandAugment**: Applies a random selection of transformations with uniform magnitude. This approach reduces the need for an extensive search phase, making it more computationally efficient while still enhancing model robustness. Reference to the [RandAugment paper](https://arxiv.org/abs/1909.13719).
    - **AugMix**: AugMix is a data augmentation method that enhances model robustness by creating diverse image variations through random combinations of simple transformations. Reference to the [AugMix paper](https://arxiv.org/abs/1912.02781).
- **Ultralytics' implementation**: [classify_augmentations()](../reference/data/augment.md#ultralytics.data.augment.classify_augmentations)
- **Note**:
    - Essentially, the main difference between the three methods is the way the augmentation policies are defined and applied.
    - You can refer to [this article](https://sebastianraschka.com/blog/2023/data-augmentation-pytorch.html) that compares the three methods in detail.

### Random Erasing (`erasing`)

- **Range**: `0.0` - `1.0`
- **Default**: `{{ erasing }}`
- **Usage**: Randomly erases portions of the image during classification training. The `erasing` hyperparameter defines the probability of applying the transformation, with `erasing=1.0` erasing a region in every image and `erasing=0.0` disabling the transformation. For example, with `erasing=0.5`, each image has a 50% chance of having a portion erased.
- **Purpose**: Helps models learn robust features and prevents over-reliance on specific image regions. For example, in facial recognition systems, random erasing helps models become more robust to partial occlusions like sunglasses, face masks, or other objects that might partially cover facial features. This improves real-world performance by forcing the model to identify individuals using multiple facial characteristics rather than depending solely on distinctive features that might be obscured.
- **Ultralytics' implementation**: [classify_augmentations()](../reference/data/augment.md#ultralytics.data.augment.classify_augmentations)
- **Note**:
    - The `erasing` augmentation comes with a `scale`, `ratio`, and `value` hyperparameters that cannot be changed with the [current implementation](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py). Their default values are `(0.02, 0.33)`, `(0.3, 3.3)`, and `0`, respectively, as stated in the PyTorch [documentation](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomErasing.html).

|                                                 **`erasing` off**                                                 |                                      **`erasing` on (example 1)**                                      |                                      **`erasing` on (example 2)**                                      |                                      **`erasing` on (example 3)**                                      |
| :---------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| <img src="https://cdn.ul.run/i/96850e8fb08eedb22329434dcbbaaaf7.avif" alt="Original image without augmentation"/> | <img src="https://cdn.ul.run/i/fad5c47ddec857b1190dd9b6fc6dc7c4.avif" alt="Random erasing example 1"/> | <img src="https://cdn.ul.run/i/fb9a4dfba52380002aeeeaf42f4f8a91.avif" alt="Random erasing example 2"/> | <img src="https://cdn.ul.run/i/fa288d9de17ef13aa36252f6c4af09d0.avif" alt="Random erasing example 3"/> |

## Advanced Augmentation Features

### Custom Albumentations Transforms (`augmentations`)

- **Type**: `list` of Albumentations transforms
- **Default**: `None`
- **Usage**: Allows you to provide custom [Albumentations](https://albumentations.ai/) transforms for data augmentation using the Python API. This parameter accepts a list of Albumentations transform objects that will be applied during training instead of the default Albumentations transforms.
- **Purpose**: Provides fine-grained control over data augmentation strategies by leveraging the extensive library of Albumentations transforms. This is particularly useful when you need specialized augmentations beyond the built-in YOLO options, such as elastic deformations and grid distortions for medical imaging, transforms tuned for overhead aerial and satellite perspectives, noise and brightness shifts that simulate low-light conditions, or defect-like texture variations for industrial inspection.
- **Ultralytics' implementation**: [Albumentations](../reference/data/augment.md#ultralytics.data.augment.Albumentations)
- **Note**:
    - Building the transform objects requires the Python API. Ultralytics serializes them with `A.to_dict()` when saving a checkpoint, so an already-serialized list round-trips through a YAML configuration file or the CLI, which is what lets `resume` restore them.
    - Custom transforms completely replace the default Albumentations set. Every augmentation configured elsewhere on this page — `mosaic`, `hsv_h`, `degrees`, and the rest — stays active and is applied independently.
    - Be cautious with spatial transforms that change image geometry. Ultralytics adjusts bounding boxes automatically, but some complex transforms may require additional configuration.
    - Albumentations offers 70+ transforms; the [Albumentations documentation](https://albumentations.ai/docs/) lists them all. Adding many transforms, or computationally expensive ones, slows training down, so start with a small set and watch the epoch time.
    - Applies to the `detect`, `segment`, `semantic`, `depth`, `pose`, and `obb` tasks. Classification is excluded, as it uses a separate augmentation pipeline.

The examples below need Albumentations 1.4.22 or newer, and therefore Python 3.9 or newer.

!!! example "Custom Albumentations Example"

    === "Python API"

        ```python
        import albumentations as A

        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")

        # Define custom Albumentations transforms
        custom_transforms = [
            A.Blur(blur_limit=7, p=0.5),
            A.GaussNoise(std_range=(0.0124, 0.0277), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ]

        # Train with custom Albumentations transforms
        model.train(
            data="coco8.yaml",
            epochs=100,
            augmentations=custom_transforms,  # Pass custom transforms
            imgsz=640,
        )
        ```

    === "More Advanced Example"

        ```python
        import albumentations as A

        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")

        # Define advanced custom Albumentations transforms with specific parameters
        advanced_transforms = [
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                    A.GaussianBlur(blur_limit=7, p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.0124, 0.0277), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ],
                p=0.2,
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.2),
        ]

        # Train with advanced custom transforms
        model.train(
            data="coco8.yaml",
            epochs=100,
            augmentations=advanced_transforms,
            imgsz=640,
        )
        ```

## FAQ

### There are too many augmentations to choose from. How do I know which ones to use?

Choosing the right augmentations depends on your specific use case and dataset. Here are a few general guidelines to help you decide:

- In most cases, slight variations in color and brightness are beneficial. The default values for `hsv_h`, `hsv_s`, and `hsv_v` are a solid starting point.
- If the camera's point of view is consistent and won't change once the model is deployed, you can likely skip geometric transformations such as `rotation`, `translation`, `scale`, `shear`, or `perspective`. However, if the camera angle may vary, and you need the model to be more robust, it's better to keep these augmentations.
- Use the `mosaic` augmentation only if having partially occluded objects or multiple objects per image is acceptable and does not change the label value. Alternatively, you can keep `mosaic` active but increase the `close_mosaic` value to disable it earlier in the training process.

In short: keep it simple. Start with a small set of augmentations and gradually add more as needed. The goal is to improve the model's generalization and robustness, not to overcomplicate the training process. Also, make sure the augmentations you apply reflect the same data distribution your model will encounter in production.

### When starting a training, I see an `albumentations: Blur[...]` reference. Does that mean Ultralytics YOLO runs additional augmentation like blurring?

If the `albumentations` package is installed, Ultralytics automatically applies a set of extra image augmentations using it. These augmentations are handled internally and require no additional configuration.

You can find the full list of applied transformations in our [technical documentation](../reference/data/augment.md#ultralytics.data.augment.Albumentations), as well as in our [Albumentations integration guide](../integrations/albumentations.md). Note that only the augmentations with a probability `p` greater than `0` are active. These are purposefully applied at low frequencies to mimic real-world visual artifacts, such as blur or grayscale effects.

You can also provide your own custom Albumentations transforms using the Python API. See the [Advanced Augmentation Features](#advanced-augmentation-features) section for more details.

### When starting a training, I don't see any reference to albumentations. Why?

Check if the `albumentations` package is installed. If not, install it:

```bash
pip install albumentations
```

Once installed, the package should be automatically detected and used by Ultralytics.

### How do I customize my augmentations?

You can customize augmentations by creating a custom dataset class and trainer. For example, you can replace the default Ultralytics classification augmentations with PyTorch's [torchvision.transforms.Resize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html) or other transforms. See the [custom training example](../tasks/classify.md#train) in the classification documentation for implementation details.
