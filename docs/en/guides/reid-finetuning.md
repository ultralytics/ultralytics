---
comments: true
description: Learn how to fine-tune the pretrained YOLO26 ReID seeds on your own person re-identification dataset. Covers the general ReID finetune seed vs the Market-1501 champion, dataset YAML format, the Python API workflow, recommended image size, model-size selection, and evaluation with mAP and Rank-1.
keywords: YOLO26 ReID, person re-identification fine-tuning, ReID finetune seed, LUPerson-NL, MSMT17, Market-1501, transfer learning ReID, custom ReID dataset, yolo26l-reid.pt, reid_p, reid_k, triplet loss, PK sampling, mAP, Rank-1, re-ranking
---

# Fine-Tuning the Pretrained ReID Seeds on Your Own Dataset

The `yolo26{n,s,m,l,x}-reid.pt` weights are **general-purpose person re-identification (ReID) finetune seeds**. When you load one with `YOLO("yolo26l-reid.pt")` and call `train()`, the weights auto-download from the [`ultralytics/assets`](https://github.com/ultralytics/assets) GitHub release and serve as a strong starting point for fine-tuning on your own [ReID dataset](../datasets/reid/index.md) — exactly the way `yolo26l.pt` is the starting point for detection [fine-tuning](finetuning-guide.md).

This guide explains what these seeds are, how they differ from the Market-1501 evaluation checkpoints, and how to fine-tune them on a custom person ReID dataset.

!!! tip "New to the ReID task?"

    Read the [Person Re-Identification (ReID) task page](../tasks/reid.md) first for the conceptual overview (embeddings, PK sampling, BNNeck, triplet loss). This guide focuses specifically on **transfer learning from the pretrained seeds**.

## What the ReID Seeds Are

The bare-named `yolo26{size}-reid.pt` weights are **not** randomly initialized and **not** a plain ImageNet or detection backbone. They are produced by a dedicated person-ReID pretraining lineage:

1. **Large-scale self-/weakly-supervised pretraining on [LUPerson-NL](https://github.com/DengpanFu/LUPerson-NL)** — roughly **10.7M person images across ~434K identities** — which teaches the model domain-general person appearance features.
2. **Supervised ReID refinement on MSMT17-scale labeled data** — [MSMT17](../datasets/reid/msmt17.md) is the largest labeled ReID benchmark — which shapes the embedding space for the retrieval objective.

The result is a **domain-general person-ReID seed**: it already understands clothing, body shape, pose, and viewpoint variation, so fine-tuning it on your own dataset converges far better than training the same architecture from scratch. In internal experiments, starting from a ReID-pretrained seed improved final mAP by roughly **+50% relative** versus training from a non-ReID initialization on the same data and schedule.

!!! note "Pretraining provenance"

    The seeds are pretrained on LUPerson-NL and refined on MSMT17-scale labeled ReID data. Treat them as **general person-ReID pretraining**, not as a single-dataset (e.g., Market-1501) model — they are intended to transfer, not to reproduce one benchmark's numbers.

### Seed vs. Market-1501 Champion

There are two families of `-reid` checkpoints, with different purposes. Choose by what you want to do:

| Weight                        | Purpose                                                                                                                      | Use as finetune seed?                                                           |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `yolo26{size}-reid.pt`        | **General finetune seed** — auto-downloaded by `train()`. Domain-general; transfers well to new datasets.                    | **Yes** — this is the default.                                                  |
| `yolo26{size}-reid-market.pt` | **Market-1501 evaluation champion** — for out-of-the-box inference and reproducing published Market-1501 mAP/Rank-1 numbers. | **No** — it is tuned to Market and transfers poorly zero-shot to other domains. |

In short: fine-tune from `yolo26{size}-reid.pt`. Reach for `yolo26{size}-reid-market.pt` only when you want ready-made Market-1501 inference or to reproduce that benchmark, not as a transfer-learning starting point.

## Choosing a Model Size

Unlike detection — where smaller models are a safe default — ReID transfer favors **larger seeds for unknown deployment domains**:

- **New / unknown domain (different cameras, geography, viewpoints):** prefer **`l` or `x`**. Larger seeds are more robust out-of-domain and degrade more gracefully under domain shift.
- **Data close to a known public domain (surveillance-style pedestrian crops):** smaller sizes (`n`/`s`/`m`) can be sufficient and are cheaper to run.

In-domain accuracy tends to top out around **L** (extra capacity at `x` gives little in-domain gain), but `l`/`x` hold up better when the test distribution differs from training. See the [ReID task page](../tasks/reid.md#models) for per-size benchmarks.

## Dataset Format

A ReID dataset is described by a YAML file with `path`, split directories, the training identity count `nc`, and a filename parsing rule. The schema mirrors the built-in [`Market-1501.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Market-1501.yaml) and [`DukeMTMC-reID.yaml`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DukeMTMC-reID.yaml) configs:

```yaml
# my_reid.yaml
path: my_reid # dataset root dir (relative to your Ultralytics datasets directory)
train: bounding_box_train # training images (relative to 'path')
val: query # query images for evaluation
gallery: bounding_box_test # gallery images compared against each query

nc: 751 # number of training identities

# Filename parsing. Built-in presets: 'market1501', 'dukemtmc', 'msmt17'.
# Or a custom regex where group(1)=person ID and (optional) group(2)=camera ID.
filename_re: market1501
cam_0indexed: false # set true if your camera IDs start at 0
```

Key points specific to ReID datasets:

- **`gallery` is required** (detection/classification datasets have no such field). Evaluation compares every `val` query embedding against all `gallery` embeddings.
- **Identity and camera IDs are encoded in the filename**, not in subfolders. Use a built-in `filename_re` preset or a custom regex.
- **Camera ID is optional.** If your filenames carry only a person ID, use a regex with a **single capture group** and the same-camera exclusion step is skipped automatically.

For full details — folder layout, naming conventions per dataset, and custom-regex examples — see the [ReID Datasets guide](../datasets/reid/index.md).

## Fine-Tuning Workflow

Use the **Python API** for ReID training and validation. ReID introduces task-specific config keys (such as `reid_p` and `triplet_weight`) that the CLI only accepts when `task=reid` is explicit in the command (see the [gotcha below](#cli-and-the-reid-only-keys)); the Python API always sets the task for you, so it is the robust path.

!!! example "Fine-tune the general ReID seed"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26l-reid.pt")  # auto-downloads the general ReID finetune seed
    model.train(data="my_reid.yaml", epochs=60, imgsz=448)
    ```

- **`imgsz=448`** is the validated default for these models — Rank-1 climbs as image size increases from 288 up to 448, then saturates/regresses by 512. Use a square input; rectangular inputs are catastrophic on the YOLO backbone. (The `n`/`s` benchmark checkpoints on the task page were trained at 256; 448 is the recommended size for best accuracy on the larger seeds.)
- Loading a `.pt` seed transfers all compatible weights automatically and reshapes the identity-classification head to your dataset's `nc`. No manual weight surgery is required.

### ReID-Specific Training Arguments

These keys exist **only for the `reid` task** and are not part of the general YOLO configuration (they are not in `default.yaml`). The most common are below; most users should keep the defaults. See the [ReID task page](../tasks/reid.md#reid-specific-training-arguments) for the full table and descriptions.

| Argument         | Default | What it controls                                                                                                                       |
| ---------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `reid_p`         | `16`    | **P** — identities per batch (effective batch size = `reid_p × reid_k`).                                                               |
| `reid_k`         | `4`     | **K** — images per identity per batch (needed for triplet positive/negative pairs).                                                    |
| `triplet_weight` | `1.0`   | Weight of the triplet metric-learning loss.                                                                                            |
| `triplet_margin` | `0.3`   | Triplet margin (typical 0.2–0.5).                                                                                                      |
| `ce_weight`      | `1.0`   | Weight of the identity cross-entropy loss.                                                                                             |
| `center_weight`  | `0.0`   | Center loss (disabled by default; try `~0.0005` to enable).                                                                            |
| `focal_gamma`    | `0.0`   | Focal-CE gamma (disabled by default).                                                                                                  |
| `supcon_temp`    | `0.0`   | Supervised-contrastive temperature; when `>0` it replaces triplet (try `~0.07`).                                                       |
| `reid_arcface`   | `False` | Sub-center ArcFace angular-margin identity loss (triplet stays on); sharper separation of near-identical classes such as product SKUs. |

!!! example "Override a ReID training argument"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26l-reid.pt")
    model.train(data="my_reid.yaml", epochs=60, imgsz=448, reid_p=32, reid_k=8)  # batch = 256
    ```

## Evaluating Your Fine-Tuned Model

Validation follows the standard query-gallery protocol and reports **mAP** and **Rank-1 / Rank-5 / Rank-10** accuracy. Two post-processing options can boost accuracy at evaluation time with no retraining:

- **Flip TTA** (`reid_tta=True`): averages embeddings of the image and its horizontal mirror; typically +1–2% mAP.
- **K-reciprocal re-ranking** (`reid_reranking=True`): refines the distance matrix using neighborhood structure; typically +15–17% mAP.

!!! example

    ```python
    from ultralytics import YOLO

    model = YOLO("path/to/best.pt")
    metrics = model.val(reid_tta=True, reid_reranking=True)
    metrics.results_dict["metrics/mAP"]  # mAP
    ```

See the [ReID task page](../tasks/reid.md#val) for the full evaluation reference.

## CLI and the ReID-Only Keys

The ReID-specific keys (`reid_p`, `reid_k`, `triplet_weight`, `triplet_margin`, `ce_weight`, etc.) are accepted on the CLI **only when `task=reid` is explicit in the command**:

```bash
# Works — 'reid' is given explicitly, so the ReID-only keys are recognized
yolo reid train model=yolo26l-reid.pt data=my_reid.yaml reid_p=16 imgsz=448

# Fails — task is inferred from the model file *after* argument parsing,
# so reid_p is rejected as "not a valid YOLO argument"
yolo train model=yolo26l-reid.pt data=my_reid.yaml reid_p=16
```

To avoid this entirely, **use the Python API for ReID train/val** — it always sets the task before the arguments are validated, so every ReID key is accepted:

```python
from ultralytics import YOLO

model = YOLO("yolo26l-reid.pt")
model.train(data="my_reid.yaml", epochs=60, imgsz=448, reid_p=16)
```

## FAQ

### Are the `yolo26-reid.pt` weights trained from scratch?

No. They are a general-purpose person-ReID finetune seed produced by large-scale pretraining on [LUPerson-NL](https://github.com/DengpanFu/LUPerson-NL) (~10.7M images, ~434K identities) followed by supervised refinement on MSMT17-scale labeled ReID data. They are designed to transfer to your own dataset, not to be trained from random initialization.

### Should I fine-tune from `yolo26l-reid.pt` or `yolo26l-reid-market.pt`?

Fine-tune from **`yolo26l-reid.pt`** — the domain-general seed. `yolo26l-reid-market.pt` is the Market-1501 evaluation champion, meant for out-of-the-box inference or reproducing Market-1501 numbers; it is tuned to Market and transfers poorly to other domains.

### Why can't I pass `reid_p` on the command line?

The ReID-only keys are only recognized when `task=reid` is explicit in the command (`yolo reid train ...`). If the task is inferred from the model file (`yolo train model=...-reid.pt ...`), parsing rejects the key. Use the [Python API](#fine-tuning-workflow), which always sets the task first.

### What image size should I fine-tune at?

`imgsz=448` (square) is the validated default for these seeds. Rank-1 improves from 288 up to 448 and then saturates or regresses by 512. Avoid rectangular inputs on the YOLO backbone.

### How much data do I need?

Because the seed already encodes general person appearance, fine-tuning works with far less data than training from scratch. Ensure your dataset has multiple images per identity across different cameras so PK sampling can form meaningful triplet pairs (`reid_k ≥ 4`).
