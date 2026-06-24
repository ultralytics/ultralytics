---
comments: true
description: Explore the KITTI depth dataset for monocular depth estimation. Learn about its structure, the Eigen evaluation benchmark, usage, and pretrained YOLO26-Depth models.
keywords: KITTI dataset, depth estimation, monocular depth, autonomous driving, LiDAR depth, Eigen split, YOLO26-Depth, outdoor depth, Ultralytics
---

# KITTI Depth Dataset

The [KITTI](https://www.cvlibs.net/datasets/kitti/) dataset is a real-world outdoor [autonomous-driving](https://www.ultralytics.com/glossary/autonomous-vehicles) benchmark captured from a moving vehicle in and around the city of Karlsruhe. For monocular [depth estimation](https://www.ultralytics.com/glossary/depth-estimation), the ground-truth depth is derived from a Velodyne HDL-64 LiDAR scanner and densified using the method of [Uhrig et al. 2017](https://arxiv.org/abs/1708.06500). The resulting depth maps remain sparse, with roughly 16–20% of pixels carrying a valid depth value. KITTI is the only real outdoor long-range source in the YOLO26-Depth pretraining mix and also serves as the KITTI Eigen evaluation benchmark.

## Key Features

- Real outdoor driving scenes with depths spanning up to roughly 80 m, far beyond the typical indoor range.
- Depth ground truth obtained from a Velodyne HDL-64 LiDAR and densified with the [Sparsity Invariant CNNs](https://arxiv.org/abs/1708.06500) approach of Uhrig et al. 2017.
- Sparse supervision: only about 16–20% of pixels per image carry a valid depth value; invalid pixels are masked out of the loss and metrics.
- Stereo image pairs (left `image_02` and right `image_03`) provide additional viewpoints for training.
- Depth values are stored as `.npy` float32 arrays in meters, following the [Ultralytics depth dataset format](index.md).

## Dataset Structure

The KITTI depth data used by Ultralytics is split into two subsets:

1. **Training split**: 60,040 images (left `image_02` and right `image_03`). The 28 KITTI Eigen test drives are excluded from training to keep evaluation fair.
2. **Evaluation split**: the KITTI Eigen test split, 32,378 images (both cameras). Evaluation uses the Garg crop, an 80 m depth cap, and log-least-squares alignment between predictions and ground truth.

The depth range reaches approximately 80 m, and the dataset YAML (`depth-kitti.yaml`) sets `max_depth: 80` accordingly.

## Role in YOLO26-Depth

KITTI supplies the only real outdoor, long-range supervision in the YOLO26-Depth pretraining mix, complementing the predominantly indoor sources. It is also the standard KITTI Eigen benchmark for reporting driving-scene depth accuracy.

KITTI is a key example of why the default depth head is unbounded (`log` mode): a fixed 10 m output ceiling cannot represent 80 m driving scenes. See the [depth task page](../../tasks/depth.md) for details on the head output range and `max_depth` handling.

## Results

KITTI Eigen `delta1` accuracy by model size (higher is better):

| Model         | KITTI Eigen δ1 |
| ------------- | -------------- |
| YOLO26n-Depth | pending (in training) |
| YOLO26s-Depth | 0.882          |
| YOLO26m-Depth | 0.924          |
| YOLO26l-Depth | 0.927          |
| YOLO26x-Depth | 0.939          |

## Dataset YAML

A YAML (Yet Another Markup Language) file is used to define the dataset configuration. It contains information about the dataset's paths, classes, and other relevant information such as the maximum depth.

!!! example "ultralytics/cfg/datasets/depth-kitti.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/depth-kitti.yaml"
    ```

## Usage

To train a YOLO26n-Depth model on the KITTI dataset for 100 [epochs](https://www.ultralytics.com/glossary/epoch), you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained depth model
        model = YOLO("yolo26n-depth.pt")

        # Train the model on KITTI
        results = model.train(data="depth-kitti.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo depth train data=depth-kitti.yaml model=yolo26n-depth.pt epochs=100 imgsz=640
        ```

## Pretrained Models

Pretrained YOLO26-Depth models auto-download from the Ultralytics [v8.4.0 assets release](https://github.com/ultralytics/assets/releases/tag/v8.4.0) when first referenced by name:

- [YOLO26n-Depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-depth.pt)
- [YOLO26s-Depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-depth.pt)
- [YOLO26m-Depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-depth.pt)
- [YOLO26l-Depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-depth.pt)
- [YOLO26x-Depth](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-depth.pt)

## Citations and Acknowledgments

If you use the KITTI dataset in your research or development work, please cite the following papers:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{geiger2013vision,
              title={Vision meets Robotics: The KITTI Dataset},
              author={Geiger, Andreas and Lenz, Philip and Stiller, Christoph and Urtasun, Raquel},
              journal={The International Journal of Robotics Research},
              year={2013},
              publisher={SAGE Publications}
        }

        @inproceedings{uhrig2017sparsity,
              title={Sparsity Invariant CNNs},
              author={Uhrig, Jonas and Schneider, Nick and Schneider, Lukas and Franke, Uwe and Brox, Thomas and Geiger, Andreas},
              booktitle={International Conference on 3D Vision (3DV)},
              year={2017}
        }
        ```

We would like to acknowledge the Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago for creating and maintaining the KITTI dataset, and Uhrig et al. for the depth densification method that makes dense supervision possible.
