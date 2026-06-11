---
comments: true
description: Learn how to structure datasets for YOLO ReID tasks. Detailed folder structure, naming conventions, and usage examples for person re-identification training.
keywords: YOLO, person re-identification, ReID, dataset structure, Market-1501, DukeMTMC, MSMT17, Ultralytics, metric learning, training data
---

# ReID Datasets Overview

## Dataset Structure for YOLO ReID Tasks

For [Ultralytics](https://www.ultralytics.com/) YOLO re-identification tasks, the dataset must follow a specific structure where person images are organized under a root directory with separate splits for training, query, and gallery sets.

### Image Naming Convention

ReID datasets encode person identity (PID) and camera ID (CamID) in filenames. Different datasets use different naming conventions:

| Dataset | Pattern | Example |
|---------|---------|---------|
| [Market-1501](market1501.md) | `PPPP_cCsS_XXXXXX_XX.jpg` | `0001_c1s1_001051_00.jpg` |
| [DukeMTMC-reID](dukemtmc.md) | `PPPP_cC_fXXXXXX.jpg` | `0001_c1_f0046182.jpg` |
| [MSMT17](msmt17.md) | `PPPP_CCC_II_XXXX.jpg` | `0001_015_01_0201130904.jpg` |

Each format encodes person ID and camera ID differently, but the YOLO ReID pipeline handles them automatically via the `filename_re` setting in the dataset YAML config.

### Folder Structure Example

Consider the [Market-1501](market1501.md) dataset as an example. The folder structure should look like this:

```
Market-1501-v15.09.15/
|
|-- bounding_box_train/
|   |-- 0001_c1s1_001051_00.jpg
|   |-- 0001_c1s1_009376_00.jpg
|   |-- 0001_c2s1_001051_00.jpg
|   |-- 0002_c1s1_001051_00.jpg
|   |-- ...
|
|-- query/
|   |-- 0001_c1s1_001051_00.jpg
|   |-- 0002_c4s2_012428_00.jpg
|   |-- ...
|
|-- bounding_box_test/
|   |-- 0001_c1s1_001051_00.jpg
|   |-- 0001_c2s2_065432_00.jpg
|   |-- ...
```

### Dataset YAML Format

A ReID dataset YAML config specifies the root path, split directories, number of training identities, and filename parsing:

```yaml
path: Market-1501-v15.09.15  # dataset root dir
train: bounding_box_train     # training images
val: query                    # query images for evaluation
gallery: bounding_box_test    # gallery images for evaluation

nc: 751  # number of training identities

# Optional: filename parsing (defaults to 'market1501')
# Built-in presets: 'market1501', 'dukemtmc', 'msmt17'
# Or provide a custom regex with group(1)=pid, group(2)=camid
filename_re: market1501
cam_0indexed: false  # set true if camera IDs start at 0
```

!!! note

    Unlike detection or classification datasets, ReID datasets require a `gallery` field that specifies the gallery set used during evaluation. The evaluation protocol compares each query image against all gallery images to compute mAP and Rank-1 metrics.

!!! tip "Camera ID is optional"

    Camera ID (camid) is only needed for the standard Market-1501 evaluation protocol, which excludes same-person-same-camera matches from evaluation. If your custom dataset doesn't have camera information in the filenames, simply use a regex with **one capture group** (person ID only) and the pipeline will work correctly — the same-camera exclusion step is automatically skipped.

    ```yaml
    # Example: custom dataset with PID-only filenames like "0001_001.jpg"
    filename_re: '(\d+)_\d+\.(?i:jpg|png|bmp)'  # single-quoted: no YAML escape interpretation; (?i:...) matches .JPG too
    ```

## Usage

To train a YOLO ReID model on a dataset, you can use the following code snippets. For a comprehensive list of available arguments, refer to the model [Training](../../modes/train.md) page.

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

## Supported Datasets

| Dataset | Images | IDs | Cameras | Difficulty |
|---------|--------|-----|---------|------------|
| [Market-1501](market1501.md) | 32,668 | 1,501 | 6 | Moderate |
| [DukeMTMC-reID](dukemtmc.md) | 36,411 | 1,404 | 8 | Moderate-Hard |
| [MSMT17](msmt17.md) | 126,441 | 4,101 | 15 | Hard |

### Benchmark Results

YOLO26 ReID results (imgsz=448, standard query–gallery protocol). Market-1501 numbers are the published `yolo26{size}-reid-market.pt` champions; DukeMTMC-reID numbers are the `yolo26{size}-reid.pt` seed fine-tuned on Duke:

| Model size | Market-1501 (`-reid-market.pt`)<br>mAP / R-1 | DukeMTMC-reID (`-reid.pt` + Duke FT)<br>mAP / R-1 |
|------------|----------------------------------------------|----------------------------------------------------|
| YOLO26n | 67.3 / 86.6 | 48.0 / 69.8 |
| YOLO26s | 72.9 / 89.4 | — |
| YOLO26m | 73.6 / 88.5 | — |
| YOLO26l | 76.8 / 90.6 | 57.0 / 75.8 |
| YOLO26x | 75.5 / 90.5 | — |

[K-reciprocal re-ranking](../../tasks/reid.md) (`reid_reranking=True`) adds a further +12–20 mAP. The bare `yolo26{size}-reid.pt` weights are general fine-tuning seeds — to adapt them to your own data see the [ReID Fine-Tuning guide](../../guides/reid-finetuning.md).

## FAQ

### What is the evaluation protocol for ReID datasets?

ReID evaluation follows a query-gallery protocol. Each query image is compared against all gallery images by computing embedding distances. For each query, gallery images are ranked by distance, and standard retrieval metrics are computed:

- **mAP (mean Average Precision)**: Measures overall retrieval quality across all queries.
- **Rank-1**: The fraction of queries where the correct identity appears as the top-ranked result.

Same-person-same-camera matches are excluded from evaluation, as they are trivially easy to match.

### How does ReID dataset structure differ from classification datasets?

Classification datasets organize images into class subdirectories (e.g., `cat/`, `dog/`). ReID datasets instead encode identity information in the filename (e.g., `0001_c1s1_001051_00.jpg`) and keep all images in a flat directory. This is because ReID also needs camera ID information for the evaluation protocol.

### Can I use custom ReID datasets with YOLO?

Yes. Create a YAML config file with `path`, `train`, `val`, `gallery`, and `nc` fields pointing to your dataset. Use one of the built-in filename presets (`market1501`, `dukemtmc`, `msmt17`) via `filename_re`, or provide a custom regex where group(1) captures the person ID. Camera ID (group 2) is **optional** — if your regex only has one capture group, the pipeline works without camera information. If you do include camera IDs, set `cam_0indexed: true` if they start at 0.
