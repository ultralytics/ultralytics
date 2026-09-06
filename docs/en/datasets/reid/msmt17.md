---
comments: true
description: Explore the MSMT17 dataset, the largest public person re-identification benchmark with 126,441 images of 4,101 identities from 15 cameras.
keywords: MSMT17, dataset, person re-identification, ReID, benchmark, large-scale, metric learning, YOLO, computer vision
---

# MSMT17 Dataset

The [MSMT17](https://arxiv.org/abs/1711.08565) (Multi-Scene Multi-Time) dataset is the largest publicly available person re-identification benchmark. It was collected across a university campus using 15 cameras (12 outdoor + 3 indoor) over multiple time periods with varying lighting conditions. The dataset contains 126,441 bounding boxes of 4,101 identities, making it significantly more challenging than [Market-1501](market1501.md) and [DukeMTMC-reID](dukemtmc.md).

## Key Features

- **4,101 identities** captured by 15 cameras (12 outdoor + 3 indoor).
- **30,248 training images** from 1,041 identities.
- **11,659 query images** from 3,060 identities for evaluation.
- **82,161 gallery images** for matching, the largest gallery set among standard ReID benchmarks.
- Collected across **4 time periods** (morning, noon, afternoon, evening) with natural lighting variation.
- Multiple scenes including outdoor walkways, indoor corridors, and building entrances.

## Dataset Structure

The MSMT17 (V2) dataset is organized into three main directories:

1. **mask_train_v2/**: 30,248 images of 1,041 identities for training.
2. **mask_query_v2/**: 11,659 query images for evaluation.
3. **mask_gallery_v2/**: 82,161 gallery images for matching.

### Filename Convention

Each image follows the naming pattern:

```
PPPP_CCC_II_XXXXXXXXXXXX.jpg
```

- `PPPP`: Person ID (e.g., `0001`)
- `CCC`: Camera ID (e.g., `015`, 0-indexed)
- `II`: Image index within the sequence
- `XXXXXXXXXXXX`: Timestamp or frame identifier

For example, `0001_015_01_0201130904.jpg` means person `0001` captured by camera `015`.

## Applications

MSMT17 is considered the most challenging standard ReID benchmark and is used for:

- **Large-scale ReID** evaluation with realistic conditions
- **Multi-scene and multi-time** robustness testing
- **State-of-the-art benchmarking** as the hardest standard ReID dataset
- **Pre-training** ReID models before fine-tuning on smaller datasets

## Usage

To train a YOLO ReID model on MSMT17 for 60 epochs with an image size of 448, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-reid.yaml")

        # Train the model
        results = model.train(data="MSMT17.yaml", epochs=60, imgsz=448)
        ```

    === "CLI"

        ```bash
        # Start training from a YAML model config
        yolo reid train data=MSMT17.yaml model=yolo26n-reid.yaml epochs=60 imgsz=448
        ```

## Citations and Acknowledgments

If you use the MSMT17 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{wei2018person,
            title={Person Transfer GAN to Bridge Domain Gap for Person Re-identification},
            author={Wei, Longhui and Zhang, Shiliang and Gao, Wen and Tian, Qi},
            booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            year={2018}
        }
        ```

We acknowledge Longhui Wei and collaborators for creating and maintaining the MSMT17 dataset. For access, visit the [PKU Vision and Media Computing group page](https://www.pkuvmc.com/dataset.html).

## FAQ

### How does MSMT17 compare to other ReID datasets?

| Dataset                      | Identities | Images      | Cameras | Train IDs | Challenge Level |
| ---------------------------- | ---------- | ----------- | ------- | --------- | --------------- |
| [Market-1501](market1501.md) | 1,501      | 32,668      | 6       | 751       | Moderate        |
| [DukeMTMC-reID](dukemtmc.md) | 1,404      | 36,411      | 8       | 702       | Moderate-Hard   |
| **MSMT17**                   | **4,101**  | **126,441** | **15**  | **1,041** | **Hard**        |

MSMT17 is approximately 4x larger than Market-1501 in terms of images and has nearly 3x more identities, with the added challenge of multi-scene and multi-time variation.

### Why is MSMT17 considered more challenging?

MSMT17's difficulty comes from several factors: (1) 15 cameras spanning indoor and outdoor environments produce larger appearance variations, (2) data collected across 4 time periods introduces significant lighting changes, (3) the large gallery set (82K images) makes retrieval harder, and (4) more identities and images increase the chance of confusing look-alikes.

### How do I obtain the MSMT17 dataset?

MSMT17 requires requesting access from the [PKU Vision and Media Computing group](https://www.pkuvmc.com/dataset.html). After approval, download and extract MSMT17_V2 to your datasets directory.
