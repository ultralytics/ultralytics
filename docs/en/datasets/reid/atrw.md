---
comments: true
description: Explore ATRW, the Amur Tiger Re-identification in the Wild benchmark for individual animal ReID, and learn how to train YOLO ReID on folder-per-identity wildlife data.
keywords: ATRW, Amur tiger, animal re-identification, wildlife ReID, individual identification, dataset, folder-per-identity, conservation, YOLO, computer vision
---

# ATRW Dataset

[ATRW](https://lila.science/datasets/atrw) (Amur Tiger Re-identification in the Wild) is a benchmark for **individual animal re-identification**. It provides labeled images of individual Amur tigers captured in zoos and wild-like enclosures with pose, viewpoint, and lighting variation. Distinguishing one tiger from another relies on fine-grained stripe patterns, which makes ATRW a strong test of fine-grained [re-identification](https://www.ultralytics.com/glossary/reid-re-identification) and a practical example of applying YOLO ReID to wildlife conservation.

## Key Features

- **Individual-tiger identity labels** for fine-grained animal ReID.
- **Folder-per-identity layout** — no camera structure, so each image is treated as its own camera.
- Real-world pose, viewpoint, and illumination variation.
- Designed for conservation and wildlife-monitoring applications.

!!! note "Evaluation protocol"

    The built-in `ATRW.yaml` uses an **identity-disjoint holdout**: ~20% of the labeled identities are held out as `query`/`gallery` because the official competition test split withholds identity labels (scored on a competition server). The `nc` value (training identities after the holdout) is set in the YAML — verify it against your prepared data and update if your split differs.

## Dataset Structure

ATRW uses the folder-per-identity ReID layout, where each subdirectory name is the identity label:

```
ATRW/
├── train/          # one folder per tiger identity
│   ├── 0001/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── ...
├── query/          # held-out identities, same folder-per-identity structure
└── gallery/        # gallery searched for each query
```

Because filenames do not encode a camera ID, no `filename_re` is needed — the pipeline treats each image as its own camera and reports standard retrieval mAP.

## Applications

ATRW supports:

- **Wildlife conservation** — non-invasive individual tracking from camera-trap and survey imagery.
- **Fine-grained animal ReID** research on stripe/marking-based identity cues.
- **Cross-domain ReID** evaluation, testing transfer of person-trained embeddings to animals.

In Ultralytics cross-domain benchmarks, the released `yolo26l-reid.pt` checkpoint already achieves strong zero-shot results (**0.6313 mAP / 0.9524 Rank-1**), improving to **0.6855 / 1.0000** after fine-tuning at `imgsz=448`. See the [ReID Beyond Persons guide](../../guides/reid-custom-dataset.md) for the full comparison.

## Usage

To train a YOLO ReID model on ATRW, use the snippets below. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the released ReID checkpoint as a fine-tune seed
        model = YOLO("yolo26l-reid.pt")

        # Fine-tune on ATRW (folder-per-identity, no camera IDs)
        results = model.train(data="ATRW.yaml", epochs=120, imgsz=448)
        ```

    === "CLI"

        ```bash
        # Fine-tune from the released ReID checkpoint
        yolo reid train data=ATRW.yaml model=yolo26l-reid.pt epochs=120 imgsz=448
        ```

## Citations and Acknowledgments

If you use the ATRW dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{li2020atrw,
            title={ATRW: A Benchmark for Amur Tiger Re-identification in the Wild},
            author={Li, Shuyuan and Li, Jianguo and Tang, Hanlin and Qian, Rui and Lin, Weiyao},
            booktitle={Proceedings of the 28th ACM International Conference on Multimedia (ACM MM)},
            year={2020}
        }
        ```

We acknowledge the CVWC2019 organizers and the dataset authors. ATRW is distributed via [LILA BC](https://lila.science/datasets/atrw).

## FAQ

### Why does ATRW have no camera IDs?

Wildlife imagery typically lacks a fixed-camera structure, so ATRW is stored folder-per-identity with no camera labels. The evaluator then treats each image as its own camera and reports standard retrieval mAP — the correct protocol for camera-less data such as most animal and product datasets.

### Can YOLO ReID identify other animals?

Yes. ReID learns a generic per-identity embedding, so any animal dataset with multiple images per individual can train the same head. ATRW (tigers) is one example; the [ReID Beyond Persons guide](../../guides/reid-custom-dataset.md) covers vehicles, animals, and products.

### What does `nc` mean for ATRW?

`nc` is the number of **training** identities (folders under `train/`) after the identity-disjoint eval holdout. It excludes query/gallery identities. Confirm the count against your prepared data and update the YAML if needed.
