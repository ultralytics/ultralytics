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
- **No. of  total Images:** 27,577  (84.73% are real-world images and 15.27% are augmented images)    
- **No. of  total (vehicle) Samples:** 1,308,989
- **Dimension of each 4K image:** 3840 x 2160
- **Pre-defined train/val/test split ratio:** The Dataset is organised into a predefined split ratio to preserve equal class distribution 

## Applications

DRASHTI-HaOBB provides a standardised [OBB dataset](https://docs.ultralytics.com/datasets/obb/) for training and evaluating UAV-based vehicle detection models under real-world traffic conditions. The annotations include oriented bounding boxes along with vehicle heading angle and drone flight height to support vehicle detection in dense and heterogeneous aerial traffic scenes. The dataset is well-suited for research on road traffic safety analysis, understanding driving patterns, traffic rules enforcement, and broader intelligent transportation system studies.

## Dataset Structure

DRASHTI-HaOBB is organised into a standard format compatible with [OBB object detection](https://docs.ultralytics.com/tasks/obb/) models:

- **Images:** Collection of high-resolution UAV images of dense and heterogeneous road traffic.
- **Text files:** A label file provides details of each vehicle in the image as OBB (_x1, y1, x2, y2, x3, y3, x4, y4_), vehicle class, difficulty level, and heading angle.
- **Train/Val/Test:** All vehicle classes distributed into 60:30:10 split ratio. 

## Dataset YAML

A dataset YAML (Yet Another Markup Language) file provides information about the dataset configuration, vehicle classes, and other metadata. A DRASHTI-HaOBB.yaml file available at https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DRASHTI-HaOBB.yaml includes automatic download and annotation conversion (from DOTA to YOLO) functionalities:

!!! example "DRASHTI-HaOBB.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/DRASHTI-HaOBB.yaml"
    ```
## Usage

To train a model on the DRASHTI-HaOBB dataset, use the following code snippets. Refer to the [list of arguments](https://docs.ultralytics.com/modes/train/) for training an OBB model. 

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
        @article{9560031,
          author={Ding, Jian and Xue, Nan and Xia, Gui-Song and Bai, Xiang and Yang, Wen and Yang, Michael and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges},
          year={2021},
          volume={},
          number={},
          pages={1-1},
          doi={10.1109/TPAMI.2021.3117983}
        }
        ```

A special note of gratitude to the team behind the DRASHTI-HaOBB datasets for their commendable effort in curating this dataset. For an exhaustive understanding of the dataset and its nuances, please visit the [official DRASHTI-HaOBB website](https://captain-whu.github.io/DRASHTI-HaOBB/index.html).

## FAQ

### can we use it for other countries?
### possible applications of this dataset?

### What is the DRASHTI-HaOBB dataset and why is it important for object detection in aerial images?

The [DRASHTI-HaOBB dataset](https://captain-whu.github.io/DRASHTI-HaOBB/index.html) is a specialized dataset focused on object detection in aerial images. It features Oriented Bounding Boxes (OBB), providing annotated images from diverse aerial scenes. DRASHTI-HaOBB's diversity in object orientation, scale, and shape across its 1.7M annotations and 18 categories makes it ideal for developing and evaluating models tailored for aerial imagery analysis, such as those used in surveillance, environmental monitoring, and disaster management.

### How does the DRASHTI-HaOBB dataset handle different scales and orientations in images?

DRASHTI-HaOBB utilizes Oriented Bounding Boxes (OBB) for annotation, which are represented by rotated rectangles encapsulating objects regardless of their orientation. This method ensures that objects, whether small or at different angles, are accurately captured. The dataset's multiscale images, ranging from 800 × 800 to 20,000 × 20,000 pixels, further allow for the detection of both small and large objects effectively. This approach is particularly valuable for aerial imagery where objects appear at various angles and scales.

### How can I train a model using the DRASHTI-HaOBB dataset?

To train a model on the DRASHTI-HaOBB dataset, you can use the following example with [Ultralytics YOLO](https://docs.ultralytics.com/tasks/obb/):

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a new YOLO26n-OBB model from scratch
        model = YOLO("yolo26n-obb.yaml")

        # Train the model on the DRASHTI-HaOBBv1 dataset
        results = model.train(data="DRASHTI-HaOBBv1.yaml", epochs=100, imgsz=1024)
        ```

    === "CLI"

        ```bash
        # Train a new YOLO26n-OBB model on the DRASHTI-HaOBBv1 dataset
        yolo obb train data=DRASHTI-HaOBBv1.yaml model=yolo26n-obb.pt epochs=100 imgsz=1024
        ```

For more details on how to split and preprocess the DRASHTI-HaOBB images, refer to the [split DRASHTI-HaOBB images section](#split-DRASHTI-HaOBB-images).

### What are the differences between DRASHTI-HaOBB-v1.0, DRASHTI-HaOBB-v1.5, and DRASHTI-HaOBB-v2.0?

- **DRASHTI-HaOBB-v1.0**: Includes 15 common categories across 2,806 images with 188,282 instances. The dataset is split into training, validation, and testing sets.
- **DRASHTI-HaOBB-v1.5**: Builds upon DRASHTI-HaOBB-v1.0 by annotating very small instances (less than 10 pixels) and adding a new category, "container crane," totaling 403,318 instances.
- **DRASHTI-HaOBB-v2.0**: Expands further with annotations from Google Earth and GF-2 Satellite, featuring 11,268 images and 1,793,658 instances. It includes new categories like "airport" and "helipad."

For a detailed comparison and additional specifics, check the [dataset versions section](#dataset-versions).

### How can I prepare high-resolution DRASHTI-HaOBB images for training?

DRASHTI-HaOBB images, which can be very large, are split into smaller resolutions for manageable training. Here's a Python snippet to split images:

!!! example

    === "Python"

        ```python
        from ultralytics.data.split_DRASHTI-HaOBB import split_test, split_trainval

        # split train and val set, with labels.
        split_trainval(
            data_root="path/to/DRASHTI-HaOBBv1.0/",
            save_dir="path/to/DRASHTI-HaOBBv1.0-split/",
            rates=[0.5, 1.0, 1.5],  # multiscale
            gap=500,
        )
        # split test set, without labels.
        split_test(
            data_root="path/to/DRASHTI-HaOBBv1.0/",
            save_dir="path/to/DRASHTI-HaOBBv1.0-split/",
            rates=[0.5, 1.0, 1.5],  # multiscale
            gap=500,
        )
        ```

This process facilitates better training efficiency and model performance. For detailed instructions, visit the [split DRASHTI-HaOBB images section](#split-DRASHTI-HaOBB-images).
