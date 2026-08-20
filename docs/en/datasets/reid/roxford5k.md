---
comments: true
description: Explore rOxford5k (Revisited Oxford5k), an eval-only landmark image-retrieval benchmark, and understand why whole-scene retrieval differs from instance-level ReID.
keywords: rOxford5k, Revisited Oxford5k, Oxford Buildings, landmark retrieval, image retrieval, instance retrieval, eval-only, benchmark, YOLO, computer vision
---

# rOxford5k Dataset

[rOxford5k](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) (Revisited Oxford5k) is a landmark **image-retrieval** benchmark derived from the classic Oxford Buildings dataset. Each query depicts a famous Oxford landmark, and the task is to retrieve other images of the same building from a gallery. In Ultralytics it is included as an **eval-only** dataset: there is no training split, and it is used to probe how an instance-level [re-identification](https://www.ultralytics.com/glossary/reid-re-identification) embedding behaves on whole-scene retrieval.

!!! warning "Whole-scene retrieval is a different task"

    YOLO ReID embeds **cropped instances** (a person, a vehicle, a product), not full scenes. Landmark retrieval matches entire images of buildings, so it is the **wrong tool** for this benchmark — rOxford5k is documented here to make that boundary explicit, not as a recommended training target.

## Key Features

- **Eval-only** — no training split; zero-shot evaluation only.
- Landmark/building identities (identity = landmark group).
- Standard same-identity mAP over landmark groups.
- Useful as a negative/diagnostic benchmark for instance-level embeddings.

!!! note "Protocol"

    The built-in `rOxford5k.yaml` reports standard same-identity mAP over landmark groups rather than the official `easy`/`medium`/`hard` split. `nc` is 12 (11 classic landmarks plus the general `oxford` query group). The `train` field points at `query` only to satisfy the schema; fine-tuning is disabled for this dataset.

## Dataset Structure

```
rOxford5k/
├── query/      # landmark query images (also referenced as the no-op train split)
└── gallery/    # gallery images searched for each query
```

## Applications

rOxford5k is used here for:

- **Diagnostic evaluation** — demonstrating where instance-level ReID embeddings do and do not transfer.
- **Image-retrieval research** comparisons against whole-scene retrieval methods.

In Ultralytics cross-domain benchmarks, the released `yolo26l-reid.pt` checkpoint scores **0.1463 mAP / 0.2571 Rank-1** zero-shot, with no fine-tune cell (training disabled) — quantifying the mismatch between cropped-instance embeddings and whole-scene retrieval. See the [ReID Beyond Persons guide](../../guides/reid-custom-dataset.md).

## Usage

rOxford5k is evaluated zero-shot. For a comprehensive list of available arguments, refer to the model [Validation](../../modes/val.md) page.

!!! example "Evaluate Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the released ReID checkpoint
        model = YOLO("yolo26l-reid.pt")

        # Zero-shot retrieval evaluation (no training)
        metrics = model.val(data="rOxford5k.yaml", imgsz=448)
        ```

    === "CLI"

        ```bash
        # Zero-shot retrieval evaluation
        yolo reid val data=rOxford5k.yaml model=yolo26l-reid.pt imgsz=448
        ```

## Citations and Acknowledgments

If you use rOxford5k in your research or development work, please cite the following papers:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{radenovic2018revisiting,
            title={Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking},
            author={Radenovi{\'c}, Filip and Iscen, Ahmet and Tolias, Giorgos and Avrithis, Yannis and Chum, Ond{\v{r}}ej},
            booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            year={2018}
        }

        @inproceedings{philbin2007object,
            title={Object Retrieval with Large Vocabularies and Fast Spatial Matching},
            author={Philbin, James and Chum, Ondrej and Isard, Michael and Sivic, Josef and Zisserman, Andrew},
            booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            year={2007}
        }
        ```

We acknowledge the Visual Geometry Group (University of Oxford) and the authors of the revisited benchmark. For data and protocol details, visit the [Oxford Buildings page](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/).

## FAQ

### Why is rOxford5k eval-only?

It is included as a diagnostic to show that cropped-instance ReID embeddings do not transfer well to whole-scene landmark retrieval. There is no training split and fine-tuning is intentionally disabled.

### Should I fine-tune YOLO ReID for landmark retrieval?

No. ReID embeds cropped instances, not full scenes — landmark retrieval is a different problem. Use a dedicated image-retrieval method for whole-scene matching. For instance-level identity tasks (people, vehicles, animals, products), see the [ReID Beyond Persons guide](../../guides/reid-custom-dataset.md).
