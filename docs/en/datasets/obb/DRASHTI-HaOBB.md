---
comments: true
description: Explore the DRASHTI-HaOBB dataset for oriented object detection in aerial road traffic images, featuring 1.3M Oriented Bounding Boxes across 14 vehicle categories.
keywords: DRASHTI-HaOBB dataset, vehicle detection, aerial images, oriented bounding boxes, OBB, DRASHTI-HaOBB, multiscale detection, Ultralytics
---

# DRASHTI-HaOBB Dataset

[DRASHTI-HaOBB](https://zenodo.org/records/18278989) is a large-scale, country-specific UAV-based [vehicle detection](https://www.ultralytics.com/glossary/object-detection) dataset designed for [oriented object detection](https://docs.ultralytics.com/tasks/obb/) under dense and heterogeneous traffic conditions.

## Key Features

- **Annotations:** Oriented Bounding Box (OBB) and Heading-angle for Vehicles; Flight-height
- **View:** Drone-based Aerial-view (90° downward gimbal)
- **Location:** India
- **No. of Vehicle Classes:** 14
- **Vehicle Classes:** ‘Auto3WCargo’, ‘AutoRicksaw’, ‘Bus’, ‘Container’, ‘Mixer’, ‘MotorCycle’, ‘PickUp’, ‘SUV’, ‘Sedan’, ‘Tanker’, ‘Tipper’, ‘Trailer’, ‘Truck’, ‘Van’
- **No. of total Images:** 27,577 (84.73% are real-world images and 15.27% are augmented images)
- **No. of total (vehicle) Samples:** 1,308,989
- **Dimension of each 4K image:** 3840 x 2160
- **Pre-defined train/val/test split ratio:** The Dataset is organised into a predefined split ratio to preserve equal class distribution

## Applications

DRASHTI-HaOBB provides a standardised [OBB dataset](https://docs.ultralytics.com/datasets/obb/) for training and evaluating UAV-based vehicle detection models under real-world traffic conditions. The annotations include oriented bounding boxes along with vehicle heading angle and drone flight height to support vehicle detection in dense and heterogeneous aerial traffic scenes. Models trained on the DRASHTI-HaOBB dataset efficiently detect vehicles (i.e., classify vehicles and localise OBBs), which could further support computer vision-based applications for road traffic safety analysis, understanding driving patterns, traffic rule enforcement, and broader intelligent transportation system studies.

## Dataset Structure

DRASHTI-HaOBB is organised into a standard format compatible with [OBB object detection](https://docs.ultralytics.com/tasks/obb/) models:

- **Images:** Collection of high-resolution UAV images of dense and heterogeneous road traffic.
- **Text files:** A label file provides details of each vehicle in the image as OBB (_x1, y1, x2, y2, x3, y3, x4, y4_), vehicle class, difficulty level, and heading angle.
- **Train/Val/Test:** All vehicle classes distributed into 60:30:10 split ratio.

## Dataset YAML

A dataset YAML (Yet Another Markup Language) file provides information about the dataset configuration, vehicle classes, and other metadata. A DRASHTI-HaOBB.yaml file available at https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DRASHTI-HaOBB.yaml includes automatic download and annotation conversion (from DRASHTI-HaOBB to YOLO) functionalities:

!!! example "DRASHTI-HaOBB.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/DRASHTI-HaOBB.yaml"
    ```

## Usage

To train a model on the DRASHTI-HaOBB dataset, use the following code snippets. Refer to the [list of arguments](https://docs.ultralytics.com/modes/train/#train-settings) for training an OBB model.

!!! warning

The [DRASHTI-HaOBB dataset](https://zenodo.org/records/18278989) is released under the [Creative Commons Attribution–NonCommercial–NoDerivatives 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) International license. Commercial use is strictly prohibited. Please credit the dataset creators in any academic or research work. Your understanding and respect for the dataset creators' wishes are greatly appreciated!

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a new YOLO26n-OBB model from scratch
        model = YOLO("yolo26n-obb.yaml")

        # Train the model on the DRASHTI-HaOBB dataset
        results = model.train(data="DRASHTI-HaOBB.yaml", epochs=30, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Train a new YOLO26n-OBB model on the DRASHTI-HaOBB dataset
        yolo obb train data=DRASHTI-HaOBB.yaml model=yolo26n-obb.pt epochs=30 imgsz=1024
        ```

## Sample Data and Annotations

A sample of [DRASHTI-HaOBB](https://zenodo.org/records/18278989) dataset, showing dense and heterogeneous traffic conditions, and the need for OBB over an axis-aligned box.

DRASHTI-HaOBB dataset with oriented bounding box annotations
![sample](https://github.com/user-attachments/assets/ede97912-6c83-4d7e-b3bb-a80e37225fd3)

- **DRASHTI-HaOBB vehicle classes**: This snapshot highlights heterogeneity in Indian road traffic.
  <img width="1018" height="313" alt="Samples" src="https://github.com/user-attachments/assets/d4765495-a405-4028-961a-a0ceddefc533" />

## License

The [DRASHTI-HaOBB dataset](https://zenodo.org/records/18278989) is released under the [Creative Commons Attribution–NonCommercial–NoDerivatives 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) International license. Commercial use is strictly prohibited. Please credit the dataset creators in any academic or research work.

## Citations and Acknowledgements

If you use DRASHTI-HaOBB in your work, please cite the relevant research papers:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{bhavsar_2026_18278989,
            author= {Bhavsar, Yagnik and Zaveri, Mazad and Raval, Mehul and Zaveri, Shaheriar and Ahmedabad University},
            title= {DRASHTI-HaOBB: Drone nadiR-view Annotated imageS of veHicles dataseT for India - Heading-angle Oriented Bounding Box: Indian Vehicle                             Oriented-Object-Detection Dataset with 1.3 Million Samples},
            month        = feb,
            year         = 2026,
            publisher    = {Zenodo},
            version      = {1.0.0},
            doi          = {10.5281/zenodo.18278989},
            url          = {https://doi.org/10.5281/zenodo.18278989},}
        ```

We acknowledge the team behind the DRASHTI-HaOBB dataset at Ahmedabad University for their dedicated effort in curating this resource. For a comprehensive description of the dataset and its characteristics, please visit the [official DRASHTI-HaOBB website](https://sites.google.com/ahduni.edu.in/yagnikmbhavsar/dataset).

## FAQ

### What is the DRASHTI-HaOBB dataset and how is it different from other aerial datasets?

DRASHTI-HaOBB is a large-scale UAV-based oriented vehicle detection dataset designed for road traffic analysis under real-world Indian traffic conditions. It consists of 27,577 nadir-view aerial images with nearly 1.3 million vehicle annotations across 14 heterogeneous vehicle categories, each labelled with oriented bounding boxes, vehicle heading information, and UAV flight height. Unlike many existing aerial datasets that focus on axis-aligned boxes, structured traffic, or a limited number of classes, DRASHTI-HaOBB captures dense, heterogeneous traffic common in developing countries. Detailed vehicle classification and nadir-view UAV imagery of the DRASHTI-HaOBB dataset make it suitable for computer vision-based road traffic safety analysis.

Further, UAV flight height can help determine ground sample distance (a mapping from the image coordinate system (pixels) to the real-world coordinate system (meters)) and estimate the vehicle's size and speed. The vehicle's heading angle can help define the stopping distance and the blind spots around a vehicle.

### How is the class imbalance problem addressed?

In a real-world traffic scenario, vehicle samples across 14 classes are always imbalanced; therefore, to mitigate class imbalance, copy-paste augmentation was performed on real-world images to upsample minority classes.

### What are the possible applications of DRASHTI-HaOBB, and to what extent can it generalise to traffic scenarios in other countries?

DRASHTI-HaOBB dataset is well-suited for research on computer vision-based road traffic safety analysis, as discussed in section [applications](#applications). It spans over 14 different vehicle categories: ‘Auto3WCargo’, ‘AutoRicksaw’, ‘Bus’, ‘Container’, ‘Mixer’, ‘MotorCycle’, ‘PickUp’, ‘SUV’, ‘Sedan’, ‘Tanker’, ‘Tipper’, ‘Trailer’, ‘Truck’, ‘Van’. Since many of these vehicle classes are commonly observed across the Indian subcontinent, the dataset can be effectively leveraged to study traffic scenarios in countries exhibiting similar vehicle compositions.

### How can I train a model using the DRASHTI-HaOBB dataset?

To train a model on the DRASHTI-HaOBB dataset, you can use the following example with [Ultralytics YOLO](https://docs.ultralytics.com/tasks/obb/):

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a new YOLO26n-OBB model from scratch
        model = YOLO("yolo26n-obb.yaml")

        # Train the model on the DRASHTI-HaOBB dataset
        results = model.train(data="DRASHTI-HaOBB.yaml", epochs=30, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Train a new YOLO26n-OBB model on the DRASHTI-HaOBB dataset
        yolo obb train data=DRASHTI-HaOBB.yaml model=yolo26n-obb.pt epochs=30 imgsz=1024
        ```

### How can I improve model training accuracy on the DRASHTI-HaOBB dataset?

- DRASHTI-HaOBB has only 15.27% augmented images (generated using copy-paste augmentation on real-world images). Therefore, different augmentation techniques during model training help to further improve model accuracy. For detailed instructions, visit the [augmentation-settings-and-hyperparameters](https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters).
- DRASHTI-HaOBB images are available in 4K resolution (3840 x 2160) and could be further split into smaller resolutions for better training. Here's a Python snippet to split images:

!!! example

    === "Python"

        ```python
        from ultralytics.data.split_dota import split_test, split_trainval

        # split train and val set, with labels.
        split_trainval(
            data_root="path/to/DRASHTI-HaOBB/",
            save_dir="path/to/DRASHTI-HaOBB-split/",
            rates=[0.5, 1.0, 1.5],  # multiscale
            gap=500,
        )
        # split test set, without labels.
        split_test(
            data_root="path/to/DRASHTI-HaOBB/",
            save_dir="path/to/DRASHTI-HaOBB-split/",
            rates=[0.5, 1.0, 1.5],  # multiscale
            gap=500,
        )
        ```

This process improves training efficiency. For detailed instructions, visit the [split DOTA images section](https://docs.ultralytics.com/datasets/obb/dota-v2/#split-dota-images).

### How can I cite the DRASHTI-HaOBB dataset if I use it in my research?

If you use DRASHTI-HaOBB in your work, please cite the relevant research papers:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{bhavsar_2026_18278989,
            author= {Bhavsar, Yagnik and Zaveri, Mazad and Raval, Mehul and Zaveri, Shaheriar and Ahmedabad University},
            title= {DRASHTI-HaOBB: Drone nadiR-view Annotated imageS of veHicles dataseT for India - Heading-angle Oriented Bounding Box: Indian Vehicle                             Oriented-Object-Detection Dataset with 1.3 Million Samples},
            month        = feb,
            year         = 2026,
            publisher    = {Zenodo},
            version      = {1.0.0},
            doi          = {10.5281/zenodo.18278989},
            url          = {https://doi.org/10.5281/zenodo.18278989},}
        ```

### Can I modify, redistribute, or use the DRASHTI-HaOBB dataset for commercial purposes?

No, because [DRASHTI-HaOBB dataset](https://zenodo.org/records/18278989) is released under the [Creative Commons Attribution–NonCommercial–NoDerivatives 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) International license. Please refer to section [License](#License).
