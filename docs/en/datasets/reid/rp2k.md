---
comments: true
description: Explore RP2K, a large-scale retail product dataset adapted for product ReID, with open-set and closed-set protocols for SKU-level retrieval with YOLO ReID.
keywords: RP2K, retail product dataset, product ReID, SKU retrieval, open-set, closed-set, fine-grained, instance retrieval, YOLO, computer vision
---

# RP2K Dataset

[RP2K](https://www.pinlandata.com/rp2k_dataset/) is a large-scale retail product dataset (~384,000 images of 2,388 stock-keeping units, or SKUs) photographed on real store shelves. Adapted for [re-identification](https://www.ultralytics.com/glossary/reid-re-identification), each **SKU is an identity** and the task is to match a query product image against a gallery of products — a fine-grained, camera-less retrieval problem that demonstrates YOLO ReID on retail/product data. Ultralytics ships two ready-to-build configs, `rp2k-full-closedset.yaml` and `rp2k-full-openset.yaml`, covering the two standard ReID protocols.

## Key Features

- **~384K images / 2,388 SKUs** of real retail products.
- **Two protocols**: closed-set (query/gallery are held-out images of _seen_ SKUs) and open-set (query/gallery SKUs are _disjoint_ from training).
- **Folder-per-identity** layout — SKU names (often CJK text) are hashed to stable IDs; no `filename_re` needed.
- **Camera-less** — each image is treated as its own camera, so standard retrieval mAP applies.
- Deterministic build: filtered to ≥4 images per SKU, short side ≥32 px, content-deduplicated, capped at 250 images per SKU.

## Protocols

| Config                     | Protocol   | Train identities (`nc`) | Description                                                                             |
| -------------------------- | ---------- | ----------------------- | --------------------------------------------------------------------------------------- |
| `rp2k-full-closedset.yaml` | Closed-set | 2,351                   | Every SKU appears in train + query + gallery; evaluates retrieval on _seen_ identities. |
| `rp2k-full-openset.yaml`   | Open-set   | 1,881                   | 80/20 identity split; query/gallery SKUs are held out and never seen in training.       |

Open-set is the more realistic and challenging setting — it measures whether the embedding generalizes to **new products** it was never trained on.

## Dataset Structure

Both configs use the folder-per-identity ReID layout (one folder per SKU):

```
rp2k_full_openset/            # or rp2k_full_closedset/
├── train/          # training SKUs
│   ├── <sku_id>/
│   │   ├── 0000_....jpg
│   │   └── 0001_....jpg
│   └── ...
├── query/          # query images (1 per eval SKU)
└── gallery/        # gallery images matched against each query
```

## Download and Build

Each config includes an automatic download/build script that runs when the dataset is missing. It downloads the full RP2K archive once (~5.9 GB) from an OpenXLab mirror into a shared `datasets/RP2K_raw/` cache (reused by both the open-set and closed-set configs), then deterministically filters and splits it into the requested protocol. Building requires `Pillow` and network access to the mirror.

!!! note "Shared raw cache"

    The raw extraction is cached in `RP2K_raw/` next to your dataset root, so building both protocols downloads the archive only once.

## Applications

RP2K product ReID supports:

- **Retail shelf monitoring** and automated planogram compliance.
- **SKU-level product search** and inventory matching.
- **Fine-grained, open-set retrieval** research where new identities appear at test time.

In Ultralytics product-ReID experiments, `yolo26-reid` reaches strong open-set retrieval (Rank-1 ≈ 0.96) with a notably flat accuracy-vs-model-size curve. See the [ReID Beyond Persons guide](../../guides/reid-custom-dataset.md) for cross-domain context.

## Usage

To fine-tune a YOLO ReID model on RP2K, use the snippets below. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the released ReID checkpoint as a fine-tune seed
        model = YOLO("yolo26l-reid.pt")

        # Fine-tune on the RP2K open-set protocol
        results = model.train(data="rp2k-full-openset.yaml", imgsz=256)
        ```

    === "CLI"

        ```bash
        # Fine-tune from the released ReID checkpoint (open-set)
        yolo reid train data=rp2k-full-openset.yaml model=yolo26l-reid.pt imgsz=256
        ```

## Citations and Acknowledgments

If you use the RP2K dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{peng2021rp2k,
            title={RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Classification},
            author={Peng, Jingtian and Xiao, Chang and Li, Yifan},
            booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
            year={2021}
        }
        ```

We acknowledge Pinlan Data Technology for creating and releasing RP2K. For dataset details and licensing, visit the [RP2K dataset page](https://www.pinlandata.com/rp2k_dataset/).

## FAQ

### What is the difference between the open-set and closed-set configs?

In **closed-set**, every SKU appears in training and is also evaluated (query/gallery are held-out images of seen SKUs). In **open-set**, the evaluation SKUs are identity-disjoint from training — the model must generalize to products it never saw. Open-set is the more realistic retail scenario.

### Do RP2K filenames need a regex?

No. SKU names (often CJK text) are hashed to stable identity IDs and stored folder-per-identity, so no `filename_re` is required. Each image is treated as its own camera, yielding standard retrieval mAP.

### How large is the download?

The full RP2K archive is ~5.9 GB and is cached in a shared `RP2K_raw/` directory, so building both the open-set and closed-set protocols downloads it only once.
