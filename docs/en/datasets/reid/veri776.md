---
comments: true
description: Explore VeRi-776, a large-scale vehicle re-identification benchmark with ~50,000 images of 776 vehicles captured by 20 cameras in real urban surveillance.
keywords: VeRi-776, vehicle re-identification, vehicle ReID, dataset, cross-camera, surveillance, benchmark, metric learning, YOLO, computer vision
---

# VeRi-776 Dataset

[VeRi-776](https://github.com/JDAI-CV/VeRidataset) is a large-scale vehicle [re-identification](https://www.ultralytics.com/glossary/reid-re-identification) (ReID) benchmark collected from real-world urban surveillance. It contains roughly 50,000 images of 776 vehicles captured by 20 non-overlapping cameras across a 1 km² area, with each vehicle appearing in 2–18 cameras under varied viewpoints, illumination, resolution, and occlusion. VeRi-776 is the standard benchmark for cross-camera vehicle ReID and a useful demonstration that YOLO ReID is **not person-specific** — the same embedding head works on any identity domain.

## Key Features

- **776 vehicle identities** captured by **20 cameras** in a real traffic-surveillance setting.
- **576 training identities** (the `train` split used by the built-in config).
- Separate **query** and **gallery** splits following the standard cross-camera protocol.
- Diverse viewpoints (front, rear, side), lighting, weather, and resolutions.
- Camera IDs encoded in filenames, enabling the standard same-camera exclusion at evaluation time.

## Dataset Structure

VeRi-776 uses the flat-directory ReID layout with three splits relative to the dataset root:

```
VeRi-776/
├── train/      # training images (576 identities)
├── query/      # query images for evaluation
└── gallery/    # gallery images matched against each query
```

### Filename Convention

Each image encodes the vehicle ID and camera ID:

```
PPPP_cCCC_FFFFFFFF_NN.jpg
```

- `PPPP`: Vehicle (identity) ID
- `cCCC`: Camera ID (1-indexed)
- The remaining fields are frame/sequence identifiers.

The built-in `VeRi-776.yaml` parses this with `filename_re: '(\d+)_c(\d+)_\d+_\d+\.(?:jpg|png|bmp)'`, where group 1 is the vehicle ID and group 2 is the camera ID. The evaluator excludes same-identity-same-camera matches, the standard cross-camera ReID protocol.

## Applications

VeRi-776 is widely used for:

- **Cross-camera vehicle ReID** in intelligent transportation and smart-city systems.
- **Vehicle search and tracking** across distributed camera networks.
- **Cross-domain ReID research**, evaluating how person-trained embeddings transfer to vehicles.

In Ultralytics cross-domain benchmarks, fine-tuning the released `yolo26l-reid.pt` checkpoint reaches **0.4656 mAP / 0.8695 Rank-1** on VeRi-776 — far above the 0.0859 / 0.3045 zero-shot result and the 0.2150 / 0.5209 obtained by fine-tuning a generic ImageNet backbone, under one fixed 120-epoch recipe at `imgsz=448`. See the [ReID Beyond Persons guide](../../guides/reid-custom-dataset.md) for the full comparison.

## Usage

To fine-tune a YOLO ReID model on VeRi-776, use the snippets below. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the released ReID checkpoint as a fine-tune seed
        model = YOLO("yolo26l-reid.pt")

        # Fine-tune on VeRi-776
        results = model.train(data="VeRi-776.yaml", epochs=120, imgsz=448)
        ```

    === "CLI"

        ```bash
        # Fine-tune from the released ReID checkpoint
        yolo reid train data=VeRi-776.yaml model=yolo26l-reid.pt epochs=120 imgsz=448
        ```

## Citations and Acknowledgments

If you use the VeRi-776 dataset in your research or development work, please cite the following papers:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{liu2016deep,
            title={A Deep Learning-Based Approach to Progressive Vehicle Re-identification for Urban Surveillance},
            author={Liu, Xinchen and Liu, Wu and Mei, Tao and Ma, Huadong},
            booktitle={European Conference on Computer Vision (ECCV)},
            year={2016}
        }

        @inproceedings{liu2016large,
            title={Large-scale vehicle re-identification in urban surveillance videos},
            author={Liu, Xinchen and Liu, Wu and Ma, Huadong and Fu, Huiyuan},
            booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
            year={2016}
        }
        ```

We acknowledge Xinchen Liu and collaborators for creating the VeRi-776 dataset. For access and the official toolkit, visit the [VeRidataset repository](https://github.com/JDAI-CV/VeRidataset).

## FAQ

### Is YOLO ReID designed only for people?

No. YOLO ReID learns a generic embedding that maps any cropped image to a vector where same-identity crops are close together. VeRi-776 (vehicles), [ATRW](atrw.md) (animals), and [RP2K](rp2k.md) (retail products) all train the same head — see the [ReID Beyond Persons guide](../../guides/reid-custom-dataset.md).

### Do I need camera IDs to train on VeRi-776?

Camera IDs are only used at evaluation time to exclude same-camera matches (the standard cross-camera protocol). The built-in `VeRi-776.yaml` provides the `filename_re` automatically. For camera-less custom datasets, omit `filename_re` and each image is treated as its own camera.

### Should I train from scratch or fine-tune?

Fine-tune from the released `yolo26l-reid.pt` checkpoint. In every domain benchmarked, fine-tuning from the ReID seed beat both zero-shot evaluation and fine-tuning from a generic classification backbone.
