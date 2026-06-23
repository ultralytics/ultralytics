---
comments: true
description: Train Ultralytics YOLO26 ReID on any identity dataset — vehicles, animals, products, and more. Covers the folder-per-identity dataset layout, zero-shot evaluation, fine-tuning from the ReID checkpoint, the recommended recipe, and how much data you need, with cross-domain benchmark results.
keywords: custom ReID dataset, vehicle ReID, animal ReID, product retrieval, instance retrieval, fine-tuning, transfer learning, YOLO26 ReID, VeRi-776, ATRW, DeepFashion In-Shop, embeddings, mAP, Rank-1, folder-per-identity
---

# Training ReID on Custom Datasets: Vehicles, Animals, Products, and More

YOLO26 ReID is **not person-specific**. The model learns to map each cropped image to an embedding vector such that images of the same identity land close together — and "identity" can be a person, a vehicle, an individual animal, or a product item. Any dataset with **multiple images per identity** can train the same embedding head.

To quantify how well this works, we benchmarked the released `yolo26l-reid.pt` checkpoint across four very different domains, comparing zero-shot evaluation (no training) against fine-tuning from the ReID checkpoint and fine-tuning the same architecture from a generic ImageNet classification backbone. All fine-tune cells share one fixed 120-epoch recipe at `imgsz=448`:

| Domain | Dataset | Zero-shot mAP / Rank-1 | Fine-tune from `yolo26l-reid.pt` | Fine-tune from `yolo26l-cls.pt` (ImageNet) |
| ------ | ------- | ---------------------- | -------------------------------- | ------------------------------------------ |
| **Vehicles** | [VeRi-776](../datasets/reid/veri776.md) — real cross-camera traffic surveillance | 0.0859 / 0.3045 | **0.4656 / 0.8695** | 0.2150 / 0.5209 |
| **Animals** | [ATRW](../datasets/reid/atrw.md) — Amur tiger individual re-identification | 0.6313 / 0.9524 | **0.6855 / 1.0000** | 0.6214 / 1.0000 |
| **Products** | [DeepFashion In-Shop](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) — clothes retrieval by item ID | 0.2486 / 0.4129 | **0.7317 / 0.9145** | 0.5946 / 0.8125 |
| **Landmarks** | [rOxford5k](../datasets/reid/roxford5k.md) — whole-scene building retrieval (eval-only) | 0.1463 / 0.2571 | — | — |

Three takeaways drive the rest of this guide:

1. **Zero-shot transfer varies enormously by domain** — run a zero-shot evaluation first before deciding to train.
2. **Fine-tuning from the ReID checkpoint beats fine-tuning from a generic backbone in every domain**, with the same recipe.
3. **Whole-scene retrieval (landmarks) is the wrong tool** — the model embeds cropped instances, not full scenes.

!!! tip "New to the ReID task?"

    Read the [ReID task page](../tasks/reid.md) first for the conceptual overview (embeddings, PK sampling, triplet loss), and the [ReID fine-tuning guide](reid-finetuning.md) for the person-specific workflow. This guide focuses on **non-person identity datasets**.

## Dataset Layout

For custom datasets, use the **folder-per-identity layout** — Ultralytics auto-detects it. Each subdirectory name is the identity label, exactly like a classification dataset, but with separate `query` and `gallery` evaluation splits:

```text
my_dataset/
├── train/
│   ├── 0001/          # one folder per identity
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   ├── 0002/
│   │   └── ...
│   └── ...
├── query/             # evaluation queries, same folder-per-identity structure
│   ├── 0501/
│   │   └── ...
│   └── ...
└── gallery/           # evaluation gallery, searched for each query
    ├── 0501/
    │   └── ...
    └── ...
```

The minimal dataset YAML:

```yaml
# my_dataset.yaml
path: my_dataset # dataset root dir
train: train # training images (relative to 'path')
val: query # query images for evaluation
gallery: gallery # gallery images compared against each query

nc: 500 # number of TRAIN identities (folders under train/)
```

Key points:

- **`nc` counts training identities only** — the number of identity folders under `train/`, not query/gallery identities.
- **Camera IDs are optional.** Without them, each image is treated as its own camera, which yields standard retrieval mAP — the correct protocol for camera-less data (products, web images, most animal datasets).
- **If your filenames encode camera IDs** (e.g., `0001_c3_000451.jpg` in cross-camera surveillance data), add `filename_re` to the YAML with a regex where group 1 is the identity ID and group 2 is the camera ID. The evaluator then excludes same-identity-same-camera matches, the standard cross-camera ReID protocol:

```yaml
filename_re: '(\d+)_c(\d+)_\d+\.jpg' # group1=id, group2=camera
```

For flat-directory benchmarks (Market-1501-style, no identity folders), see the [ReID Datasets guide](../datasets/reid/index.md).

## Do You Need to Fine-Tune? Run Zero-Shot First

Before training anything, evaluate the released checkpoint directly on your dataset. It takes seconds and tells you which of three situations you are in:

!!! example "Zero-shot evaluation"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26l-reid.pt")  # auto-downloads
        metrics = model.val(data="my_dataset.yaml", imgsz=448)
        print(metrics.results_dict["metrics/mAP"])
        ```

    === "CLI"

        ```bash
        yolo reid val model=yolo26l-reid.pt data=my_dataset.yaml imgsz=448
        ```

Interpret the result against the benchmark table:

- **Cropped subjects with person-like statistics can work out of the box.** Individual tigers hit **Rank-1 0.9524 zero-shot** — distinctive stripe patterns on cropped animals transfer surprisingly well. If your zero-shot numbers are already acceptable, you may not need to train at all.
- **Vehicles and products start low and need fine-tuning.** Zero-shot mAP was 0.0859 on vehicles and 0.2486 on products — usable as a weak baseline, but fine-tuning lifted these to 0.4656 and 0.7317 respectively.
- **Whole-scene retrieval is the wrong tool.** Landmark retrieval on full building photos scored only 0.1463 mAP zero-shot. The model embeds **cropped object instances**, not scenes. If your images contain the object of interest plus background, first run a detector and crop to the object (see [Predict mode](../modes/predict.md)), then feed the crops to ReID.

## Start From the ReID Checkpoint, Not a Generic Backbone

When fine-tuning, initialize from `yolo26l-reid.pt` rather than a generic classification backbone. In the benchmark, fine-tuning from the ReID checkpoint beat fine-tuning the identical architecture from ImageNet weights **in every domain**, with identical recipes:

| Domain | From `yolo26l-reid.pt` (mAP) | From `yolo26l-cls.pt` (mAP) | Advantage |
| ------ | ---------------------------- | --------------------------- | --------- |
| Vehicles | **0.4656** | 0.2150 | 2.2× |
| Products | **0.7317** | 0.5946 | 1.2× |
| Animals | **0.6855** | 0.6214 | 1.1× |

The metric-learning structure the checkpoint learned on person data — embedding geometry shaped by triplet loss and identity supervision — transfers across domains even when the visual content does not. The gap is largest exactly where zero-shot transfer is weakest (vehicles), so "my domain looks nothing like people" is a reason to fine-tune, not a reason to discard the ReID initialization.

!!! example "Fine-tune on a custom domain"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26l-reid.pt")  # the identity head is reshaped to your nc automatically
        model.train(data="my_dataset.yaml", epochs=120, imgsz=448)
        ```

    === "CLI"

        ```bash
        yolo reid train model=yolo26l-reid.pt data=my_dataset.yaml epochs=120 imgsz=448
        ```

## Recommended Recipe

The settings below produced every fine-tuned number in this guide, unchanged across vehicles, animals, and products:

- **Model size**: `yolo26l-reid.pt` for best accuracy; `yolo26n`/`s`/`m`-reid for faster inference on edge devices. See the [task page model table](../tasks/reid.md#models).
- **Image size**: `imgsz=448`, square. Accuracy climbs with resolution up to 448 on these models.
- **Epochs**: ~120 is a good fine-tuning budget; small datasets converge sooner.
- **PK sampling**: the benchmarks used `reid_p=8 reid_k=8` (8 identities × 8 images = batch 64). Keep `reid_k ≥ 4` so the triplet loss has enough positive pairs per identity. See [ReID training arguments](../tasks/reid.md#reid-specific-training-arguments).
- **Evaluation-time boosts**: `reid_tta=True` (flip test-time augmentation) and `reid_reranking=True` (k-reciprocal re-ranking) improve mAP with no retraining — details on the [task page](../tasks/reid.md#val).

After training, `predict()` returns one L2-normalized embedding per image on `results[0].embeddings` — compare embeddings with cosine or L2 distance. For ranked gallery search with a rendered match montage, use the built-in solution:

!!! example "Retrieve and visualize top matches"

    ```python
    from ultralytics.solutions import ReIDVisualizer

    viz = ReIDVisualizer("path/to/best.pt", imgsz=448)
    viz.visualize("query.jpg", "gallery/", k=5)  # writes a ranked comparison montage
    ```

## How Much Data Do You Need?

A data-efficiency probe on the vehicle domain trained the same recipe on 25%, 50%, and 100% of the training **identities**:

| Train identities | Fraction | mAP |
| ---------------- | -------- | ------ |
| 144 | 25% | 0.2859 |
| 288 | 50% | 0.3709 |
| 576 | 100% | 0.4656 |

Accuracy scales smoothly with identity count and was **not saturating at 576 identities** — more identities would likely keep helping. The practical guidance: when collecting data, **prioritize more identities over more images per identity**. Identity diversity is what teaches the embedding to separate fine-grained instances; a handful of images per identity (enough for PK sampling, i.e. ~4+) is sufficient per individual.

## Caveats

These results are a **transfer comparison, not per-domain benchmarks**. All numbers come from one fixed recipe (120 epochs, `imgsz=448`, shared hyperparameters) applied unchanged to every domain — per-domain tuning would raise the absolute numbers, and domain-specialized models from the vehicle-ReID or product-retrieval literature exceed them. The ATRW (tiger) evaluation is small — 42 queries over 21 identities — so its Rank-1 scores are coarse and the perfect 1.0000 after fine-tuning should be read as "saturated this small eval split", not as a solved problem. Treat the table as evidence that the ReID checkpoint transfers and that fine-tuning from it is the right default, not as state-of-the-art claims for any individual domain.

## FAQ

### Can I use YOLO26 ReID for objects other than people?

Yes. The embedding head is identity-agnostic — any dataset with multiple cropped images per identity works: vehicles, individual animals, product items, etc. Lay it out folder-per-identity, write a 5-line YAML, and run zero-shot evaluation first to see whether you even need to fine-tune.

### My images contain the object plus a lot of background. Will ReID work?

Not well — the landmark benchmark (whole scenes, 0.1463 zero-shot mAP) shows the model is designed for cropped instances. Run a detector first, crop each object with [Predict mode](../modes/predict.md), and feed the crops to the ReID model.

### Should I start from `yolo26l-reid.pt` even though my domain has nothing to do with people?

Yes. With identical recipes, the ReID initialization beat a generic ImageNet backbone in all three fine-tuned domains — by 2.2× mAP on vehicles. The transferred asset is the metric-learning embedding structure, not person-specific features.

### My dataset has no camera IDs. Is evaluation still correct?

Yes. Without `filename_re`, each image is treated as its own camera and evaluation reduces to standard retrieval mAP — the correct protocol for camera-less data. Add `filename_re` (group 1 = identity, group 2 = camera) only if your data is genuinely cross-camera and you want the same-camera exclusion protocol.

### Should I collect more identities or more images per identity?

More identities. The vehicle data-efficiency probe scaled from 0.2859 → 0.3709 → 0.4656 mAP as training identities grew from 144 → 288 → 576, with no sign of saturation. Around 4+ images per identity is enough for PK sampling to form triplets.
