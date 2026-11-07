---
title: Scene Text OBB Dataset
comments: true
creator:
    name: COCO-Text and HierText Teams
    url: https://github.com/google-research-datasets/hiertext
license:
    name: CC BY 4.0 and CC BY-SA 4.0
    url: https://creativecommons.org/licenses/by-sa/4.0/
description: Train Ultralytics YOLO OBB models on 65,325 COCO-Text and HierText images with 1.29 million legible scene-text word annotations.
keywords: scene text dataset, COCO-Text, HierText, OCR, text detection, oriented bounding boxes, OBB, Ultralytics YOLO
---

# Scene Text OBB Dataset

The Scene Text OBB dataset combines [COCO-Text](https://bgshih.github.io/cocotext/) and [HierText](https://github.com/google-research-datasets/hiertext) into a single-class word-detection dataset for [Oriented Bounding Box (OBB)](index.md) training. It contains natural-scene and document images with rotated, curved, densely packed, handwritten, and printed text.

## Key Features

- **65,325 images** and **1,288,991 legible word annotations** after conversion.
- One `text` class with normalized polygon labels compatible with Ultralytics YOLO OBB models.
- COCO-Text contributes varied natural scenes; HierText contributes dense text from natural scenes and documents.
- Automatic source downloads, annotation conversion, and cross-platform image linking through the dataset YAML.
- Source polygons with more than four points are preserved and reduced to minimum-area rectangles by the OBB data pipeline.

## Dataset Structure

The converted dataset has the following splits. Counts include only legible words with a contour area of at least one pixel; images without their source files are excluded.

| Split | COCO-Text images / words | HierText images / words | Combined images / words |
| ----- | -----------------------: | ----------------------: | ----------------------: |
| Train |          43,686 / 65,919 |         8,281 / 862,577 |        51,967 / 928,496 |
| Val   |            4,927 / 7,211 |         1,724 / 178,382 |         6,651 / 185,593 |
| Test  |            5,073 / 7,714 |         1,634 / 167,188 |         6,707 / 174,902 |
| Total |          53,686 / 80,844 |      11,639 / 1,208,147 |      65,325 / 1,288,991 |

COCO-Text uses its official training split. Its official validation images are partitioned deterministically into approximately balanced validation and test subsets using the image ID. HierText retains its official train, validation, and test splits, with `validation` renamed to `val`.

Images with no retained words remain in the dataset with empty label files and serve as background images. The converted directories use this structure:

```text
datasets/
├── coco-text-yolo/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
└── hiertext-yolo/
    ├── train/
    ├── val/
    └── test/
```

Each label row contains class `0` followed by normalized polygon coordinates. Images are hard-linked when the filesystem supports it and copied otherwise.

## Dataset YAML

The [`scene-text-obb.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/scene-text-obb.yaml) configuration references both converted datasets. On first use, it downloads COCO 2017 images, COCO-Text annotations, HierText annotations, and HierText images before converting them.

!!! example "scene-text-obb.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/scene-text-obb.yaml"
    ```

!!! warning "Download size and licensing"

    The sources require approximately 24 GB of downloads. Allow at least 50 GB of free storage during the initial download and extraction, including temporary COCO archives. If hard links are unavailable, copied images require additional storage. Review the [licensing terms](#licensing) before using or redistributing the converted data.

## Usage

Train a YOLO26 OBB model at 640-pixel image size with the following commands. Existing COCO and HierText source files under the Ultralytics `datasets_dir` are reused.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n-obb.pt")
        results = model.train(data="scene-text-obb.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo obb train data=scene-text-obb.yaml model=yolo26n-obb.pt epochs=100 imgsz=640
        ```

## Applications

Use this dataset to train word-level text detectors for document processing, receipt understanding, sign translation, industrial inspection, and OCR pipelines. Detection is only one stage of OCR: pair the detector with a text recognizer when transcription is required.

## Licensing

The combined dataset uses source-specific terms rather than a single unconditional license:

- **COCO-Text annotations**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The underlying [COCO image terms](https://cocodataset.org/#termsofuse) remain applicable.
- **HierText annotations**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Redistributed adaptations require attribution, a change notice, and licensing under the same or a compatible license.
- **HierText images**: Sourced from Open Images. The [Open Images terms](https://storage.googleapis.com/openimages/web/factsfigures.html) list images as CC BY 2.0 and instruct users to verify individual image licenses.

These annotation licenses permit commercial use subject to their conditions, but the dataset is not public domain. Preserve source attribution and review the applicable image terms when redistributing data.

## Citations and Acknowledgments

If you use this dataset in your work, cite both source datasets:

!!! quote "Citation"

    === "BibTeX"

        ```bibtex
        @article{veit2016cocotext,
          title={COCO-Text: Dataset and Benchmark for Text Detection and Recognition in Natural Images},
          author={Veit, Andreas and Matera, Tomas and Neumann, Lukas and Matas, Jiri and Belongie, Serge},
          journal={arXiv preprint arXiv:1601.07140},
          year={2016}
        }

        @inproceedings{long2022towards,
          title={Towards End-to-End Unified Scene Text Detection and Layout Analysis},
          author={Long, Shangbang and Qin, Siyang and Panteleev, Dmitry and Bissacco, Alessandro and Fujii, Yasuhisa and Raptis, Michalis},
          booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          year={2022}
        }
        ```

We thank the COCO-Text and HierText teams for creating and maintaining these datasets.

## FAQ

### Does this dataset include text transcripts for recognition?

No. The conversion produces single-class word-detection polygons for OBB training. Use the original COCO-Text and HierText annotations if your pipeline also requires transcripts.

### Why does the COCO-Text test split differ from the official release?

COCO-Text provides train and validation annotations. This configuration retains the official train split and deterministically divides the official validation images into validation and test subsets. HierText keeps its official splits.

### Can I use an existing COCO 2017 download?

Yes. Place it under the Ultralytics `datasets_dir` in the standard `coco/` layout, including `images/train2017`, `images/val2017`, and the COCO split text files. The download script reuses it instead of downloading COCO again.
