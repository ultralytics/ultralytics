---
title: K-Fold Cross Validation for YOLO Datasets
comments: true
description: Build K-Fold splits for a YOLO detection dataset with scikit-learn and pandas, train Ultralytics YOLO26 on every fold, and aggregate the results into one score.
keywords: K-Fold cross validation, YOLO, Ultralytics, scikit-learn, pandas, dataset split, model evaluation, object detection, fold balance, data leakage
---

# How to Run K-Fold Cross Validation with Ultralytics YOLO

K-Fold cross validation splits a dataset into `k` equally sized folds and trains `k` models, each holding out a different fold for validation. Every image is therefore validated exactly once and trained on in the other `k - 1` folds. Averaging the `k` results gives a far more stable estimate of model quality than a single train/val split, because it no longer depends on which images happened to land in the validation set.

This guide builds `k=5` folds for a [YOLO detection dataset](../datasets/detect/index.md) using scikit-learn and pandas, writes one dataset YAML per fold, trains [Ultralytics YOLO26](../models/yolo26.md) on each of them, and aggregates the results. Cross validation pays off most when a dataset is small, noisy, or class-imbalanced; for large, diverse datasets a single well-constructed train/val/test split gives the same answer for a fifth of the compute.

<p align="center">
  <img width="800" src="https://cdn.ul.run/i/457d0a77dc06d7204322ec056248c4b5.avif" alt="K-fold cross validation data splitting">
</p>

## Setup

This walkthrough uses the [African Wildlife](../datasets/detect/african-wildlife.md) dataset, which downloads automatically and is already in [YOLO detection format](../datasets/detect/index.md). Substitute your own by pointing `dataset_path` at it, provided it uses the standard `images/<split>` and `labels/<split>` directory layout — the collection block below walks those directories rather than reading the `train` and `val` entries of your data YAML, so a dataset defined by path lists or non-standard directories needs those globs adjusted first.

| Class Label | Instance Count |
| :---------- | -------------: |
| buffalo     |            488 |
| elephant    |            649 |
| rhino       |            484 |
| zebra       |            689 |
| **Total**   |      **2,310** |

The dataset ships 1,504 images split into `train`, `val` and `test`. This guide folds the 1,277 `train` and `val` images and **leaves `test` untouched** — cross validation replaces the train/val boundary, not the held-out set. The counts above are for the folded pool, not the whole dataset. Keeping a split out of the folds is what lets you evaluate once at the end on data that took no part in choosing anything.

Install Ultralytics and the two helper libraries this guide uses:

```bash
pip install -U ultralytics scikit-learn pandas
```

Then download the dataset once, so the rest of the guide can read it from disk:

```python
from ultralytics.data.utils import check_det_dataset

dataset_yaml = "african-wildlife.yaml"  # swap in your own; every later step reads this variable
data = check_det_dataset(dataset_yaml)  # downloads on first use
print(data["path"])
```

!!! note "Choosing `k`"

    This guide uses `k=5`, which holds out 20% of the data per fold. Smaller datasets benefit from more folds, at proportionally more training time — `k` folds means `k` full training runs.

## Building the Class-Count Matrix

The split has to be computed over something. Each image is represented by a row counting how many instances of each class its label file contains, which turns the dataset into a table `scikit-learn` can split.

Start by collecting the label and image files, and pairing them:

```python
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS

dataset_path = Path(data["path"]).resolve()  # check_det_dataset leaves an existing relative path relative
pool = ("train", "val")  # the shipped test split stays out of the folds

images = sorted(
    p for s in pool for p in (dataset_path / "images" / s).rglob("*.*") if p.suffix[1:].lower() in IMG_FORMATS
)
labels = sorted(p for s in pool for p in (dataset_path / "labels" / s).rglob("*.txt"))

# key by path relative to images/ and labels/, so the same basename in two splits stays distinct
img_by_key = {p.relative_to(dataset_path / "images").with_suffix(""): p for p in images}
lbl_by_key = {p.relative_to(dataset_path / "labels").with_suffix(""): p for p in labels}

assert len(img_by_key) == len(images), "two images share one stem, i.e. foo.jpg and foo.png; they map to one label file"
orphans = sorted(set(lbl_by_key) - set(img_by_key))
assert not orphans, f"label files with no matching image: {orphans[:3]}"
backgrounds = set(img_by_key) - set(lbl_by_key)
print(f"{len(images)} images, {len(labels)} labels, {len(backgrounds)} background images")
```

```text
1277 images, 1277 labels, 0 background images
```

!!! warning "Three details this block gets right on purpose"

    - **Scope the label glob to `labels/`.** Ultralytics datasets nest labels as `labels/train/`, `labels/val/`, `labels/test/`, and a pattern that expects them one level up returns nothing at all. A glob rooted at the dataset directory instead re-reads its own output on a second run.
    - **Use `IMG_FORMATS` rather than a hand-written extension list.** Ultralytics accepts 13 image formats, and on Linux and macOS `Path.rglob` matches extensions case-sensitively regardless of the filesystem — so a hardcoded `[".jpg", ".jpeg", ".png"]` silently drops the three `.JPG` files in this dataset, and every `.webp` or `.tif` in yours.
    - **Pair by key, never by position, and make the key the path under `images`/`labels` rather than the bare filename.** The two lists come from different globs, so any dropped or extra file shifts one relative to the other and every later pair is wrong — a failure that trains cleanly and reports meaningless metrics. The bare stem is not enough either: Ultralytics maps `images/train/0001.jpg` to `labels/train/0001.txt`, so the same basename is legal in two splits, and keying on `0001` would silently keep one of them while the assertion still passed.
    - **An image with no label file is a background, not an error.** Ultralytics reads a missing label file as an empty label array, so the checks above reject only two things: an *orphan label* — a `.txt` with no image — and two images sharing one stem, such as `foo.jpg` and `foo.png`. The second is rejected rather than resolved because `img2label_paths` maps both to the same `foo.txt`, so the package cannot tell them apart either; dropping one silently would be worse than stopping.

Now count the instances per class:

```python
from collections import Counter

import pandas as pd

classes = data["names"]  # check_det_dataset normalizes names to {int: str}, so the keys are the class IDs
cls_idx = sorted(classes)

labels_df = pd.DataFrame(0.0, index=sorted(img_by_key), columns=cls_idx)
for key, label_file in lbl_by_key.items():  # images with no label file keep their all-zero row
    counter = Counter()
    for line in label_file.read_text().splitlines():
        if line.strip():  # skip blank lines; split() also handles tabs and leading whitespace
            counter[int(line.split()[0])] += 1
    for cls, n in counter.items():
        labels_df.loc[key, cls] = n

print(labels_df.sum().rename(classes))
```

```text
buffalo     488.0
elephant    649.0
rhino       484.0
zebra       689.0
dtype: float64
```

An image with no objects produces an all-zero row, which is correct — a background image is legitimate training data, not missing data. These totals cover the whole pooled set; the next section drops duplicates from it, so treat them as a starting point rather than as the composition of the folds.

## Watch for Duplicate and Near-Duplicate Images

Cross validation assumes the folds are independent. Duplicated or near-duplicated images break that assumption: a copy in training and its twin in validation inflates that fold's metrics, and no amount of averaging removes the bias.

This is not a hypothetical for the example dataset. Hashing the folded pool finds **25 groups of byte-identical images covering 50 files**, none of them detectable from the filenames. Deduplicate before splitting:

```python
import hashlib
from collections import defaultdict

by_hash = defaultdict(list)
for key, image in img_by_key.items():
    by_hash[hashlib.md5(image.read_bytes()).hexdigest()].append(key)
duplicates = {h: keys for h, keys in by_hash.items() if len(keys) > 1}
print(f"{len(duplicates)} duplicate groups covering {sum(len(k) for k in duplicates.values())} images")

# keep one image per hash group and rebuild the matrix, so no copy can straddle a fold
drop = {k for keys in duplicates.values() for k in sorted(keys)[1:]}
img_by_key = {k: v for k, v in img_by_key.items() if k not in drop}
labels_df = labels_df.drop(index=sorted(drop))
print(f"dropped {len(drop)} duplicate images, {len(labels_df)} remain")
print(labels_df.sum().rename(classes))  # the folded pool's real composition; every class total moved
```

```text
25 duplicate groups covering 50 images
dropped 25 duplicate images, 1252 remain
buffalo     478.0
elephant    645.0
rhino       469.0
zebra       676.0
dtype: float64
```

Dropping one image per group is the blunt option shown above and it is enough here. Keeping them together with `GroupKFold`, keyed on the hash, preserves the data instead — worth it when duplicates are numerous. The same reasoning applies to frames from one video, photographs of one subject, and augmented copies of one source image: group them, or the folds are not independent. Note that byte-identical hashing does not catch re-encoded or resized near-duplicates; a perceptual hash does.

## Splitting into K Folds

`KFold` shuffles the rows and partitions them into `k` groups. `random_state` is what makes the split reproducible; the global `random` module plays no part in it.

```python
from sklearn.model_selection import KFold

ksplit = 5
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)
kfolds = list(kf.split(labels_df))

folds = [f"fold_{n}" for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=labels_df.index, columns=folds, dtype=object)
for i, (train, val) in enumerate(kfolds, start=1):
    folds_df.loc[labels_df.index[train], f"fold_{i}"] = "train"
    folds_df.loc[labels_df.index[val], f"fold_{i}"] = "val"

for fold in folds:
    print(f"{fold}: train={(folds_df[fold] == 'train').sum()} val={(folds_df[fold] == 'val').sum()}")
```

```text
fold_1: train=1001 val=251
fold_2: train=1001 val=251
fold_3: train=1002 val=250
fold_4: train=1002 val=250
fold_5: train=1002 val=250
```

!!! warning "Assign with `.loc[rows, column]`, not `df[column].loc[rows]`"

    The second form is chained assignment. Under pandas copy-on-write — the default from pandas 3.0 — it writes to a temporary copy and silently leaves the DataFrame untouched, so every fold column stays `NaN`. The failure surfaces much later as `TypeError: unsupported operand type(s) for /: 'PosixPath' and 'float'` when a fold name is used to build a path, with nothing pointing back at the assignment.

## Checking the Fold Balance

`KFold` is uniformly random over rows. It does **not** stratify, so it does not guarantee that every class is evenly represented in every fold — it only guarantees that each image is validated once. Check the result rather than assuming it:

```python
fold_distrb = pd.DataFrame(index=folds, columns=cls_idx, dtype=float)
for n, (train, val) in enumerate(kfolds, start=1):
    ratio = labels_df.iloc[val].sum() / (labels_df.iloc[train].sum() + 1e-7)
    fold_distrb.loc[f"fold_{n}"] = ratio
fold_distrb.columns = [classes[i] for i in cls_idx]
print(fold_distrb.round(3))
```

```text
        buffalo  elephant  rhino  zebra
fold_1    0.216     0.243  0.237  0.328
fold_2    0.285     0.287  0.303  0.172
fold_3    0.258     0.238  0.212  0.229
fold_4    0.219     0.255  0.285  0.222
fold_5    0.275     0.229  0.218  0.313
```

Each cell is the ratio of validation instances to training instances for that class. With `k` folds the expected value is `1 / (k - 1)`, so `0.25` here. Every cell above sits within about a third of that, which is fine — with 484 instances in the rarest class, plain `KFold` spreads them adequately.

The check matters when it fails. On a dataset with a class of only a handful of instances, some folds end up with **zero** validation instances of it, per-class AP is undefined there, and the average across folds is silently computed over a shifting set of classes. If you see a `0.000` cell, stratify the split on each image's rarest present class:

```python
from sklearn.model_selection import StratifiedKFold

freq = labels_df.sum()
y = labels_df.apply(
    lambda row: min((c for c in cls_idx if row[c] > 0), key=lambda c: freq[c]) if row.sum() else -1,
    axis=1,
)
kfolds = list(StratifiedKFold(ksplit, shuffle=True, random_state=20).split(labels_df, y))

# rebuild folds_df from the new kfolds -- every later step reads folds_df, not kfolds
folds_df = pd.DataFrame(index=labels_df.index, columns=folds, dtype=object)
for i, (train, val) in enumerate(kfolds, start=1):
    folds_df.loc[labels_df.index[train], f"fold_{i}"] = "train"
    folds_df.loc[labels_df.index[val], f"fold_{i}"] = "val"
```

Rebuilding `folds_df` is the part that is easy to miss: every step after this reads `folds_df`, never `kfolds`, so swapping the splitter alone leaves the original unstratified folds in place with nothing to show for it.

Check the stratum sizes before trusting the result, because `StratifiedKFold` degrades quietly rather than refusing:

```python
print(y.value_counts().min(), "smallest stratum vs", ksplit, "folds")
```

A stratum smaller than `ksplit` produces a `UserWarning` and folds in which that class is simply absent from validation — measured on scikit-learn 1.7.2, a stratum of 1 leaves 4 of 5 folds with no validation instances of it, a stratum of 2 leaves 3, and only a stratum of `ksplit` or more is clean. That is the failure this recipe was meant to fix, so when the smallest stratum is under `ksplit`, lower `k` or group the rare classes instead of stratifying on them. The background stratum (`-1`) counts too.

This is also a proxy, not true multi-label stratification: an image containing several classes contributes to only one stratum. `StratifiedKFold` cannot take a multi-label target directly — it raises `ValueError: Supported target types are: ('binary', 'multiclass')` — so exact multi-label balance needs the `iterative-stratification` package.

## Creating One Dataset YAML per Fold

A fold is just a list of which images train and which validate. Ultralytics dataset YAMLs accept a `.txt` file listing image paths wherever they accept a directory, so a fold needs two text files and a YAML — no image is copied anywhere:

```python
import yaml

save_path = dataset_path.parent / f"{ksplit}-fold"
save_path.mkdir(parents=True, exist_ok=True)

ds_yamls = []
for fold in folds:
    for split in ("train", "val"):
        keys = folds_df.index[folds_df[fold] == split]
        (save_path / f"{fold}_{split}.txt").write_text("\n".join(str(img_by_key[k]) for k in keys) + "\n")
    fold_yaml = save_path / f"{fold}.yaml"
    fold_yaml.write_text(
        yaml.safe_dump(
            {
                "path": save_path.as_posix(),
                "train": f"{fold}_train.txt",
                "val": f"{fold}_val.txt",
                "names": classes,
            }
        )
    )
    ds_yamls.append(fold_yaml)
```

Note that `save_path` sits **beside** the dataset, not inside it. A fold directory written into the dataset root gets picked up by the label glob on the next run.

!!! tip "Why not copy the images into fold directories?"

    Copying works and gives you self-contained fold directories, but it duplicates the folded pool `k` times. Measured on the 1,252 images left after deduplication, plus their labels, at `k=5`:

    | Approach   | Disk    | Files  |
    | ---------- | ------- | ------ |
    | Copy files | ~430 MB | 12,520 |
    | Text lists | 402 KB  | 15     |

    Both produce identical folds and identical training scans. If you do need real directories — to ship a fold elsewhere, or to hand it to a tool that cannot read a path list — copy by iterating the stem-keyed dictionaries rather than zipping two lists:

    ```python
    import shutil

    for key, image in img_by_key.items():
        for fold, split in folds_df.loc[key].items():
            for src, sub, suffix in [(image, "images", image.suffix)] + (
                [(lbl_by_key[key], "labels", ".txt")] if key in lbl_by_key else []
            ):
                dst = save_path / fold / split / sub / key.parent / f"{key.name}{suffix}"
                dst.parent.mkdir(parents=True, exist_ok=True)  # key keeps the split subdirectory
                shutil.copy(src, dst)
    ```

    This only rearranges files. The `ds_yamls` built above still point at the text lists, so to train from the copied tree you would also write one YAML per fold whose `train` and `val` name the new `fold_N/train` and `fold_N/val` directories. The destination reuses the same relative `key`, for the same reason the pairing does: flattening to the bare filename lets `train/0001.jpg` and `val/0001.jpg` land on top of each other, and `shutil.copy` overwrites without a word. The suffix is appended to `key.name` rather than set with `with_suffix`, which would turn `foo.bar.jpg` into `foo.jpg` and collide all over again.

    `shutil.copy` overwrites an existing destination silently, so a re-run is not protected in either approach — delete `save_path` first.

## Training on Every Fold

Build a fresh model for each fold so no weights carry over, and keep the result object:

```python
from ultralytics import YOLO

results = {}
for k, fold_yaml in enumerate(ds_yamls):
    model = YOLO("yolo26n.pt")  # fresh weights per fold
    results[k] = model.train(data=str(fold_yaml), epochs=100, batch=16, project="kfold_demo", name=f"fold_{k + 1}")
```

Each run reports a clean scan, which is the checkpoint confirming images and labels were paired correctly:

```text
train: Scanning .../african-wildlife/labels/train... 1001 images, 0 backgrounds, 0 corrupt
val: Scanning .../african-wildlife/labels/train... 251 images, 0 backgrounds, 0 corrupt
```

The counts match the fold sizes printed earlier. A background is legitimate training data, so the number to compare against is not zero but the background count you measured before splitting — African Wildlife labels every image, so that count was 0 and anything above it here means images and labels came apart; go back to the orphan-label check. The directory in the scan line is just wherever the first label file happened to sit, so both lines naming the same split is expected and does not mean the fold is wrong.

!!! note "Budget for `k` full training runs"

    `epochs=100` across five folds is five complete trainings. Reduce `epochs`, or start with `k=3`, when you are still iterating. On CPU, use a small `imgsz` and expect this to take hours.

## Aggregating the Results

This is the step that turns `k` runs into one number. `model.train()` returns a `DetMetrics` object per fold; the spread across folds tells you how sensitive your model is to the split:

```python
summary = pd.DataFrame(
    {
        k + 1: {
            "mAP50-95": r.box.map,
            "mAP50": r.box.map50,
            "precision": r.box.mp,
            "recall": r.box.mr,
            "fitness": r.fitness,
        }
        for k, r in results.items()
    }
).T
summary.index.name = "fold"
print(summary.round(4))
print(summary.agg(["mean", "std"]).round(4))
```

Report the **mean** as your headline number and the **standard deviation** as its uncertainty. A large spread means the metric was never stable enough to compare two models on a single split. Note that `fitness` is a property on the returned object while `box.fitness` is a method — printing the latter without `()` gives a `<bound method>` repr rather than a value.

!!! tip "Just need a single train/val split?"

    If cross validation is more than your project needs, [`autosplit`](../reference/data/split.md) writes a one-shot split in a single call. It handles image extensions correctly, but it writes bare `.txt` files without a dataset YAML, and its ratios are sampling probabilities rather than exact proportions:

    ```python
    from ultralytics.data.split import autosplit

    autosplit(path="path/to/images", weights=(0.8, 0.2, 0.0), annotated_only=True)
    ```

## Conclusion

K-Fold cross validation gives you `k` independent estimates of model quality instead of one, which is what makes results comparable on small or imbalanced datasets where a single validation split is mostly noise. Average the per-fold `mAP50-95` for your headline number and report the standard deviation alongside it, then read the [YOLO performance metrics guide](yolo-performance-metrics.md) to interpret what the spread is telling you. Once your baseline is stable enough to compare against, move on to [hyperparameter tuning](hyperparameter-tuning.md).

## FAQ

### What is K-Fold Cross Validation and why is it useful in object detection?

K-Fold cross validation divides a dataset into `k` folds and trains `k` models, each validating on a different fold, so every image is validated exactly once. For object detection this matters because a single validation split can easily over- or under-represent a rare class or a difficult scene, making one model look better than another for reasons that have nothing to do with the model. Averaging across folds removes that dependence. Start with the [K-Fold split](#splitting-into-k-folds) section for the implementation.

### How do I combine the results from all K folds into one number?

Average the per-fold metric you care about. `results[k].box.map` holds `mAP50-95` for fold `k`, so `sum(r.box.map for r in results.values()) / len(results)` is the cross-validated mAP, and the standard deviation across folds tells you how sensitive the model is to the split. See [Aggregating the Results](#aggregating-the-results).

### How much disk space does K-Fold cross validation need?

None beyond the dataset itself, if you define each fold as a `.txt` list of image paths. Copying images into per-fold directories instead multiplies the folded pool by `k` — for the 1,252 deduplicated images and their labels here, 86 MB, that is about 430 MB and 12,520 files at `k=5`, against 402 KB and 15 files for the list approach.

### Should I use K-Fold cross validation or a single train/val split?

Use K-Fold when the dataset is small, noisy, or class-imbalanced enough that a single validation split gives an unstable estimate. For large, diverse datasets a well-constructed train/val/test split reaches the same conclusion for a fraction of the compute, since K-Fold costs `k` complete training runs.

### How should I design folds for segmentation, classification, pose, or OBB?

The workflow here targets the YOLO detection format, but it adapts to every YOLO task — the task changes how you compose the folds, not whether cross validation helps:

| Task       | Fold design                                                                                                                                                                |
| :--------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `detect`   | Split at the image level, balancing object and class distributions across folds. Keep related images (same patient, video sequence, camera, or site) within a single fold. |
| `segment`  | Use the same image-level strategy as detection, additionally preserving mask and class coverage in every fold.                                                             |
| `classify` | Prefer stratified folds so class frequencies stay balanced between training and validation.                                                                                |
| `pose`     | Split by subject or sequence so the same person or animal never appears on both sides of a fold.                                                                           |
| `obb`      | Split at the image level, keeping tiles or crops from the same scene together — especially important for aerial imagery.                                                   |

### Can I use K-Fold Cross Validation with my own dataset?

Yes, as long as the annotations are in [YOLO detection format](../datasets/detect/index.md) and laid out as `images/<split>` and `labels/<split>`, which is what the collection block globs. Point `dataset_path` at your dataset and read `classes` from your own data YAML. Images with no label file are fine — Ultralytics reads them as backgrounds and they get an all-zero row. The check in [Building the Class-Count Matrix](#building-the-class-count-matrix) rejects only the reverse case, a label file with no image, which is the failure worth catching early.

### Do I still need a separate test set?

Yes, and this guide keeps one: the dataset's shipped `test` split is excluded from the folds by the `pool = ("train", "val")` line. Cross validation measures how well a training recipe generalizes, but once you start choosing between recipes on the cross-validated score, that score has informed your decisions.

Use the held-out split once, at the end, and note that neither the fold models nor the fold YAMLs are the right thing to point at it: each fold model saw only four fifths of the pool, and a `fold_*.yaml` defines no `test` entry, so `model.val(split="test")` against one raises `FileNotFoundError: ... 'test:' is not defined`. The original dataset YAML is not it either — its `train` entry is `images/train` alone, so training against it never touches the `val` pool the folds were built from.

Retrain the settled recipe on the whole folded pool with one more path list, then evaluate against the original YAML. This continues the walkthrough above and reuses its `dataset_path`, `img_by_key` and `classes`:

```python
final_dir = dataset_path.parent / "final"
final_dir.mkdir(parents=True, exist_ok=True)
(final_dir / "pool.txt").write_text("\n".join(str(p) for p in img_by_key.values()) + "\n")
(final_dir / "final.yaml").write_text(
    yaml.safe_dump(
        {
            "path": final_dir.as_posix(),
            "train": "pool.txt",
            "val": "pool.txt",  # training requires a val entry; see the warning below
            "names": classes,
        }
    )
)

final = YOLO("yolo26n.pt")
final.train(data=str(final_dir / "final.yaml"), epochs=100, batch=16, patience=0)  # patience=0 disables early stopping

fitted = YOLO(final.trainer.last)  # last.pt, so no checkpoint was chosen using training labels
print(fitted.val(data=dataset_yaml, split="test").box.map)  # the number to report
```

Two details carry the correctness here. Passing `data=dataset_yaml` to `val()` redirects it away from `final.yaml`, which has no `test` entry, to the original dataset — and your own dataset's YAML needs a `test` entry for this step to mean anything. And the evaluation uses `last.pt` rather than the model `train()` leaves loaded: with `pool.txt` on both sides, every epoch validates on its own training images, so `best.pt` is picked on labels the model has already seen — `Model.train()` reloads exactly that checkpoint. Taking the final epoch's weights instead keeps the test score free of any selection. `patience=0` is there for the same reason: left at its default, early stopping would decide when to stop from metrics measured on the training images.
