---
comments: true
description: Explore the Market-1501 dataset, the most widely used benchmark for person re-identification with 32,668 images of 1,501 identities from 6 cameras.
keywords: Market-1501, dataset, person re-identification, ReID, benchmark, metric learning, YOLO, computer vision
---

# Market-1501 Dataset

The [Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) dataset is the most widely used benchmark for person re-identification (ReID) research. It was collected in front of a supermarket at Tsinghua University using 6 cameras (5 HD + 1 low-resolution). The dataset contains 32,668 bounding boxes of 1,501 identities detected by a DPM detector.

## Key Features

- **1,501 identities** captured by 6 cameras with overlapping fields of view.
- **12,936 training images** from 751 identities.
- **3,368 query images** from 750 identities for evaluation.
- **19,732 gallery images** from 750 identities (plus distractors) for matching.
- Images are detected bounding boxes (not hand-cropped), reflecting realistic conditions.
- Standard evaluation protocol excludes same-person-same-camera matches.

## Dataset Structure

The Market-1501 dataset is organized into three main directories:

1. **bounding_box_train/**: 12,936 images of 751 identities for training.
2. **query/**: 3,368 images used as queries during evaluation.
3. **bounding_box_test/**: 19,732 gallery images used for matching against queries.

### Filename Convention

Each image follows the naming pattern:

```
PPPP_cCsS_XXXXXX_XX.jpg
```

- `PPPP`: 4-digit person ID (0001-1501, with -1 for junk images and 0000 for background)
- `C`: Camera ID (1-6)
- `S`: Sequence number
- `XXXXXX_XX`: Frame number and detection index

## Benchmark Results

Market-1501 results for the published `yolo26{n,s,m,l,x}-reid-market.pt` champions (standard query–gallery protocol, imgsz=448):

| Model | mAP | Rank-1 | Rank-5 | Rank-10 |
|-------|-----|--------|--------|---------|
| YOLO26n-reid-market | 67.3 | 86.6 | 94.8 | 96.9 |
| YOLO26s-reid-market | 72.9 | 89.4 | 96.3 | 97.9 |
| YOLO26m-reid-market | 73.6 | 88.5 | 96.3 | 97.7 |
| YOLO26l-reid-market | 76.8 | 90.6 | 96.6 | 98.0 |
| YOLO26x-reid-market | 75.5 | 90.5 | 96.4 | 97.6 |

[K-reciprocal re-ranking](../../tasks/reid.md) (`reid_reranking=True`) raises mAP to 84.2 / 87.3 / 87.5 / 88.8 / 88.3 respectively. The bare `yolo26{size}-reid.pt` weights are general fine-tuning seeds (not Market-tuned) — see the [ReID Fine-Tuning guide](../../guides/reid-finetuning.md).

## Applications

Market-1501 is the standard benchmark for evaluating person ReID methods, including:

- **Metric learning** approaches (triplet loss, contrastive loss)
- **Cross-camera person tracking** and retrieval
- **Re-ranking** methods for improving retrieval accuracy
- **Domain adaptation** for transferring ReID models across datasets

## Usage

To train a YOLO ReID model on Market-1501 for 60 epochs with an image size of 448, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n-reid.yaml")

        # Train the model
        results = model.train(data="Market-1501.yaml", epochs=60, imgsz=448)
        ```

    === "CLI"

        ```bash
        # Start training from a YAML model config
        yolo reid train data=Market-1501.yaml model=yolo26n-reid.yaml epochs=60 imgsz=448
        ```

## Citations and Acknowledgments

If you use the Market-1501 dataset in your research or development work, please cite the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @inproceedings{zheng2015scalable,
            title={Scalable Person Re-identification: A Benchmark},
            author={Zheng, Liang and Shen, Liyue and Tian, Lu and Wang, Shengjin and Wang, Jingdong and Tian, Qi},
            booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
            year={2015}
        }
        ```

We acknowledge Liang Zheng and collaborators for creating and maintaining the Market-1501 dataset as a foundational resource for the person re-identification research community. For more information, visit the [Market-1501 project page](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html).

## FAQ

### How can I train a YOLO ReID model on Market-1501?

To train a YOLO ReID model on Market-1501 using Ultralytics:

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

For best results, use PK sampling with `reid_p=32` and `reid_k=8` for a batch size of 256, combined with `ce_weight=2.0` for stronger identity classification. See the [Training](../../modes/train.md) page for all available arguments.

### What are the key metrics for Market-1501 evaluation?

Market-1501 evaluation uses two primary metrics:

- **mAP (mean Average Precision)**: The average precision across all queries, measuring overall retrieval quality. Values typically range from 0.5 to 0.95 for state-of-the-art methods.
- **Rank-1 accuracy**: The fraction of queries where the correct identity is the top-ranked gallery result. State-of-the-art methods achieve 0.90+ Rank-1 accuracy.

The evaluation protocol excludes gallery images with the same person ID and camera ID as the query, since same-camera matches are trivially easy.

### How do I set up Market-1501 for training?

1. Download the dataset from the [Market-1501 project page](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html).
2. Extract the archive to your datasets directory.
3. The directory structure should contain `bounding_box_train/`, `query/`, and `bounding_box_test/` folders.
4. Train using `data=Market-1501.yaml` which points to the standard dataset layout.

### What is the evaluation protocol for Market-1501?

The standard single-query protocol works as follows:

1. For each query image, compute its embedding vector.
2. Compare against all gallery image embeddings using L2 or cosine distance.
3. Rank gallery images by ascending distance.
4. Remove gallery images with the same person ID and camera ID as the query.
5. Compute Average Precision (AP) from the ranked list.
6. Average AP across all queries to get mAP.
