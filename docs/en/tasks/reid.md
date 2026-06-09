---
comments: true
description: Learn about person re-identification (ReID) using YOLO26. Train, validate, predict, and export ReID models for matching people across camera views.
keywords: YOLO26, person re-identification, ReID, metric learning, Market-1501, DukeMTMC, MSMT17, BNNeck, PK sampling, triplet loss, re-ranking, embedding extraction, train, validate, predict
model_name: yolo26n-reid
---

# Person Re-Identification (ReID)

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/person-reid-overview.avif" alt="YOLO person re-identification matching people across camera views">

Person re-identification (ReID) matches the same individual across different camera views or time instances. Unlike object detection which locates objects, or classification which categorizes images, ReID produces a compact embedding vector for each person image that can be compared against other embeddings to determine identity matches.

The output of a ReID model is a fixed-dimensional embedding vector. Two images of the same person should produce embeddings that are close in distance, while images of different people should produce embeddings that are far apart.

!!! tip

    YOLO26 ReID models use the `-reid` suffix, e.g., `yolo26n-reid.pt`, and support [Market-1501](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/Market-1501.yaml), [DukeMTMC-reID](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/DukeMTMC-reID.yaml), and [MSMT17](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/MSMT17.yaml) datasets.

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/26)

YOLO26 ReID ships **two checkpoint families** per size. The bare `yolo26{n,s,m,l,x}-reid.pt` weights are general-purpose **fine-tuning seeds** (pretrained on [LUPerson-NL](https://github.com/DengpanFu/LUPerson-NL) and refined on MSMT17) — load these to [fine-tune on your own dataset](../guides/reid-finetuning.md). The `yolo26{n,s,m,l,x}-reid-market.pt` weights are the **Market-1501 evaluation champions** for out-of-the-box inference and reproducing the benchmark below. All models use a BNNeck architecture with PK batch sampling and multi-loss training (cross-entropy + triplet **or** supervised-contrastive metric loss, with optional center loss on top).

| Model                                                                                                        | size<br><sup>(pixels) | mAP<br><sup>Market-1501 | Rank-1<br><sup>Market-1501 | mAP<br><sup>+re-ranking | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------------------------------ | --------------------- | ----------------------- | -------------------------- | ----------------------- | ------------------- | ------------------ |
| [YOLO26n-reid-market](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-reid-market.pt) | 448                   | 67.3                    | 86.6                       | 84.2                    | 2.8                 | 2.0                |
| [YOLO26s-reid-market](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-reid-market.pt) | 448                   | 72.9                    | 89.4                       | 87.3                    | 7.5                 | 6.6                |
| [YOLO26m-reid-market](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-reid-market.pt) | 448                   | 73.6                    | 88.5                       | 87.5                    | 12.4                | 20.1               |
| [YOLO26l-reid-market](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-reid-market.pt) | 448                   | 76.8                    | 90.6                       | 88.8                    | 15.3                | 25.2               |
| [YOLO26x-reid-market](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-reid-market.pt) | 448                   | 75.5                    | 90.5                       | 88.3                    | 32.7                | 55.9               |

- **mAP** and **Rank-1** are on [Market-1501](../datasets/reid/market1501.md) using the standard query–gallery protocol at imgsz=448. The **+re-ranking** column applies k-reciprocal re-ranking (`reid_reranking=True`). <br>Reproduce with `yolo reid val model=yolo26l-reid-market.pt data=Market-1501.yaml imgsz=448 device=0`.
- Accuracy rises with model size and **plateaus at L** (x does not beat l). For an unknown deployment domain prefer **l/x** (more robust out-of-domain) — see [model-size selection](../guides/reid-finetuning.md#choosing-a-model-size).
- **Params** and **FLOPs** are measured at imgsz=448. The bare `-reid.pt` seeds share the same architecture (and therefore params/FLOPs) as their `-reid-market` counterparts.

## Train

Train a YOLO26n-reid model on the Market-1501 dataset for 60 epochs at image size 448. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! tip "Fine-tuning on your own dataset"

    To adapt a pretrained ReID seed to a custom person re-identification dataset, see the [ReID Fine-Tuning guide](../guides/reid-finetuning.md). It explains the general `yolo26{size}-reid.pt` finetune seeds (vs. the Market-1501 champion), the dataset YAML format, recommended image size, and model-size selection.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-reid.yaml")  # build a new model from YAML
        model = YOLO("yolo26n-reid.pt")  # load the pretrained ReID seed (recommended for training)
        model = YOLO("yolo26n-reid.yaml").load("yolo26n-reid.pt")  # build from YAML and transfer seed weights

        # Train the model
        results = model.train(data="Market-1501.yaml", epochs=60, imgsz=448)
        ```

    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo reid train data=Market-1501.yaml model=yolo26n-reid.yaml epochs=60 imgsz=448

        # Start training from the pretrained ReID seed
        yolo reid train data=Market-1501.yaml model=yolo26n-reid.pt epochs=60 imgsz=448

        # Build a new model from YAML, transfer seed weights to it and start training
        yolo reid train data=Market-1501.yaml model=yolo26n-reid.yaml pretrained=yolo26n-reid.pt epochs=60 imgsz=448
        ```

### ReID-Specific Training Arguments

These arguments are **only available for the `reid` task** and are not part of the general YOLO configuration. You can pass them via Python (`model.train(reid_p=16)`) or CLI (`yolo reid train reid_p=16 ...`).

#### Batch Sampling

ReID training uses **PK sampling** instead of random batching. Each training batch is built by selecting `P` random person identities and then sampling `K` images for each identity. This guarantees every batch contains multiple images of the same person, which is required for the triplet loss to find meaningful positive/negative pairs.

| Argument  | Type  | Default | Description                                                                                                                                                        |
| --------- | ----- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `reid_p`  | `int` | `16`    | **P** — number of different person identities in each batch. The actual batch size equals `reid_p × reid_k` (e.g., 16 × 4 = 64 images).                           |
| `reid_k`  | `int` | `4`     | **K** — number of images sampled per identity in each batch. Higher values give the triplet loss more same-person pairs to compare, improving hard-negative mining. |

!!! tip

    The effective batch size is `reid_p × reid_k`. For better training, increase `reid_k` first (e.g., `reid_k=8` with `reid_p=32` for batch size 256). Make sure your GPU has enough memory for the resulting batch.

#### Loss Weights

ReID training combines multiple loss functions. The two main losses are **cross-entropy** (CE, for identity classification) and **triplet** (for metric learning). You can optionally enable center loss or supervised contrastive loss. Most users should keep the defaults.

| Argument          | Type    | Default | Description                                                                                                                                                                                                                       |
| ----------------- | ------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ce_weight`       | `float` | `1.0`   | Weight of the **cross-entropy loss**. This loss teaches the model to classify each person's identity during training. Higher values make the model focus more on identity classification.                                          |
| `triplet_weight`  | `float` | `1.0`   | Weight of the **triplet loss**. This loss pulls same-person embeddings closer and pushes different-person embeddings apart. It is the core metric-learning objective.                                                              |
| `triplet_margin`  | `float` | `0.3`   | Margin for the triplet loss. The model learns to keep the distance between different-person embeddings at least this much larger than same-person distances. Typical values: 0.2–0.5.                                             |
| `center_weight`   | `float` | `0.0`   | Weight of the **center loss** (disabled by default). When enabled (> 0), this loss pulls each person's embeddings toward a learned class center, reducing intra-class variation. Try `0.0005` if enabling.                        |
| `center_momentum` | `float` | `0.9`   | How fast the class centers update when center loss is enabled. Value of 0.9 means centers are updated slowly using exponential moving average. Only used when `center_weight > 0`.                                                |
| `focal_gamma`     | `float` | `0.0`   | Focal loss gamma for the cross-entropy component (disabled by default). When > 0, down-weights easy-to-classify samples so the model focuses on hard examples. Try `2.0` if you have many easy identities.                       |
| `supcon_temp`     | `float` | `0.0`   | Temperature for **supervised contrastive loss** (disabled by default). When > 0, replaces the triplet loss with SupCon loss which uses all positive/negative pairs rather than just the hardest. Try `0.07` if enabling.          |

### Dataset format

YOLO ReID dataset format can be found in detail in the [Dataset Guide](../datasets/reid/index.md).

## Val

Validate a trained YOLO26n-reid model on the Market-1501 dataset. The evaluation uses the standard Market-1501 protocol: L2 distance between query and gallery embeddings, excluding same-pid-same-camid matches. Reports mAP and Rank-1/5/10 accuracy.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-reid-market.pt")  # load the Market-1501 champion (reproduces the benchmark)
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.results_dict["metrics/mAP"]  # mAP
        metrics.results_dict["metrics/rank1"]  # Rank-1 accuracy
        ```

    === "CLI"

        ```bash
        yolo reid val model=yolo26n-reid-market.pt  # val the Market-1501 champion
        yolo reid val model=path/to/best.pt  # val custom model
        ```

### Test-Time Augmentation (TTA)

Enable horizontal flip TTA with `reid_tta=True` to average embeddings from original and horizontally-flipped images. This typically improves mAP by 1-2% at the cost of doubling inference time.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("path/to/best.pt")
        metrics = model.val(reid_tta=True)
        ```

    === "CLI"

        ```bash
        yolo reid val model=path/to/best.pt reid_tta=True
        ```

### K-Reciprocal Re-Ranking

Enable [k-reciprocal re-ranking](https://arxiv.org/abs/1701.08398) with `reid_reranking=True` to refine distance rankings using neighborhood structure. This post-processing technique (Zhong et al., CVPR 2017) can improve mAP by **15-17%** with no additional training — it only modifies the distance matrix at evaluation time. Re-ranking increases evaluation time due to the additional pairwise computations.

For best results, combine both TTA and re-ranking:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("path/to/best.pt")
        metrics = model.val(reid_tta=True, reid_reranking=True)
        ```

    === "CLI"

        ```bash
        yolo reid val model=path/to/best.pt reid_tta=True reid_reranking=True
        ```

### ReID Evaluation Arguments

These arguments are **only available for `reid` validation** and improve accuracy without any retraining.

| Argument          | Type   | Default | Description                                                                                                                                                                                                                                                     |
| ----------------- | ------ | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `reid_tta`        | `bool` | `False` | **Test-Time Augmentation**. When enabled, the model processes both the original image and a horizontally-flipped copy, then averages the two embeddings. This makes the embedding more robust and typically adds +1–2% mAP. Trade-off: doubles inference time.   |
| `reid_reranking`  | `bool` | `False` | **K-reciprocal re-ranking**. A post-processing step that refines the distance ranking by checking whether two images are mutual nearest neighbors. Can boost mAP by +15–17% with no retraining. Trade-off: increases evaluation time due to extra computation.   |

## Predict

Use a trained YOLO26n-reid model to extract embedding vectors from person images.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-reid.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model (extract embeddings)
        results = model("path/to/person.jpg")
        ```

    === "CLI"

        ```bash
        yolo reid predict model=yolo26n-reid.pt source='path/to/person.jpg'  # predict with official model
        yolo reid predict model=path/to/best.pt source='path/to/person.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

## Export

Export a YOLO26n-reid model to a different format like ONNX, CoreML, etc.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-reid.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom-trained model

        # Export the model
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-reid.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx   # export custom-trained model
        ```

Available YOLO26-reid export formats are in the table below. You can export to any format using the `format` argument, i.e., `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e., `yolo predict model=yolo26n-reid.onnx`. Usage examples are shown for your model after export completes.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### What is person re-identification (ReID) and how does YOLO26 handle it?

Person re-identification (ReID) recognizes the same person across different camera views or at different times. YOLO26 ReID models produce compact embedding vectors from person images. These embeddings can be compared using distance metrics (e.g., L2 or cosine distance) to determine if two images show the same person. The model is trained with PK batch sampling and a combination of cross-entropy plus either batch-hard triplet **or** supervised-contrastive metric loss (mutually exclusive), with an optional center-loss term on top.

### How do I train a YOLO26 ReID model?

To train a YOLO26 ReID model on Market-1501:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n-reid.yaml")
        results = model.train(data="Market-1501.yaml", epochs=60, imgsz=448)
        ```

    === "CLI"

        ```bash
        yolo reid train data=Market-1501.yaml model=yolo26n-reid.yaml epochs=60 imgsz=448
        ```

Key hyperparameters include `reid_p` (identities per batch), `reid_k` (images per identity), `triplet_margin`, and loss weights. For more details, see the [Configuration](../usage/cfg.md) page.

### What datasets are supported for ReID training?

The following datasets are supported out of the box:

| Dataset | Train Images | IDs | Cameras | Config |
|---------|-------------|-----|---------|--------|
| [Market-1501](../datasets/reid/market1501.md) | 12,936 | 751 | 6 | `Market-1501.yaml` |
| [DukeMTMC-reID](../datasets/reid/dukemtmc.md) | 16,522 | 702 | 8 | `DukeMTMC-reID.yaml` |
| [MSMT17](../datasets/reid/msmt17.md) | 30,248 | 1,041 | 15 | `MSMT17.yaml` |

Custom datasets are also supported via configurable filename regex patterns. See the [ReID Datasets](../datasets/reid/index.md) guide for format details.

### What is PK batch sampling in ReID training?

PK sampling is a batch construction strategy where each batch contains P randomly selected identities, each with K randomly sampled images. This ensures every batch has multiple images per identity, which is essential for computing meaningful triplet losses that require positive pairs (same identity) and negative pairs (different identities) within each batch.

### How can I improve ReID evaluation accuracy without retraining?

Two post-processing techniques are available that improve mAP at evaluation time with no retraining needed:

1. **Flip TTA** (`reid_tta=True`): Averages embeddings from original and horizontally-flipped images. Typically +1-2% mAP.
2. **K-reciprocal re-ranking** (`reid_reranking=True`): Refines distance rankings using neighborhood structure ([Zhong et al., CVPR 2017](https://arxiv.org/abs/1701.08398)). Typically +15-17% mAP.

```bash
yolo reid val model=path/to/best.pt reid_tta=True reid_reranking=True
```

### How does ReID differ from image classification?

While both tasks use backbone networks for feature extraction, they solve different problems:

- **Classification** assigns one of N predefined labels to an image. The model outputs class probabilities.
- **ReID** produces an embedding vector that captures identity-discriminative features. The model must generalize to identities never seen during training, making it an open-set problem. At inference time, embeddings are compared by distance rather than classified.
