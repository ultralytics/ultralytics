---
description: Learn how to split datasets into train, validation, and test subsets using Ultralytics utilities for efficient data preparation.
keywords: dataset splitting, autosplit dataset, training dataset preparation, validation set creation, Ultralytics data tools
---

# Reference for `ultralytics/data/split.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`split_classify_dataset`](#ultralytics.data.split.split_classify_dataset)
        - [`autosplit`](#ultralytics.data.split.autosplit)


## Function `ultralytics.data.split.split_classify_dataset` {#ultralytics.data.split.split\_classify\_dataset}

```python
def split_classify_dataset(source_dir: str | Path, train_ratio: float = 0.8) -> Path
```

Split classification dataset into train and val directories in a new directory.

Creates a new directory '{source_dir}_split' with train/val subdirectories, preserving the original class structure
with an 80/20 split by default.

Directory structure:
    Before:
        caltech/
        ├── class1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── class2/
        │   ├── img1.jpg
        │   └── ...
        └── ...

    After:
        caltech_split/
        ├── train/
        │   ├── class1/
        │   │   ├── img1.jpg
        │   │   └── ...
        │   ├── class2/
        │   │   ├── img1.jpg
        │   │   └── ...
        │   └── ...
        └── val/
            ├── class1/
            │   ├── img2.jpg
            │   └── ...
            ├── class2/
            │   └── ...
            └── ...

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `source_dir` | `str | Path` | Path to classification dataset root directory. | *required* |
| `train_ratio` | `float` | Ratio for train split, between 0 and 1. | `0.8` |

**Returns**

| Type | Description |
| --- | --- |
| `Path` | Path to the created split directory. |

**Examples**

```python
Split dataset with default 80/20 ratio
>>> split_classify_dataset("path/to/caltech")

Split with custom ratio
>>> split_classify_dataset("path/to/caltech", 0.75)
```

<details>
<summary>Source code in <code>ultralytics/data/split.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split.py#L13-L95"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def split_classify_dataset(source_dir: str | Path, train_ratio: float = 0.8) -> Path:
    """Split classification dataset into train and val directories in a new directory.

    Creates a new directory '{source_dir}_split' with train/val subdirectories, preserving the original class structure
    with an 80/20 split by default.

    Directory structure:
        Before:
            caltech/
            ├── class1/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            ├── class2/
            │   ├── img1.jpg
            │   └── ...
            └── ...

        After:
            caltech_split/
            ├── train/
            │   ├── class1/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   ├── class2/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   └── ...
            └── val/
                ├── class1/
                │   ├── img2.jpg
                │   └── ...
                ├── class2/
                │   └── ...
                └── ...

    Args:
        source_dir (str | Path): Path to classification dataset root directory.
        train_ratio (float): Ratio for train split, between 0 and 1.

    Returns:
        (Path): Path to the created split directory.

    Examples:
        Split dataset with default 80/20 ratio
        >>> split_classify_dataset("path/to/caltech")

        Split with custom ratio
        >>> split_classify_dataset("path/to/caltech", 0.75)
    """
    source_path = Path(source_dir)
    split_path = Path(f"{source_path}_split")
    train_path, val_path = split_path / "train", split_path / "val"

    # Create directory structure
    split_path.mkdir(exist_ok=True)
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    # Process class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    total_images = sum(len(list(d.glob("*.*"))) for d in class_dirs)
    stats = f"{len(class_dirs)} classes, {total_images} images"
    LOGGER.info(f"Splitting {source_path} ({stats}) into {train_ratio:.0%} train, {1 - train_ratio:.0%} val...")

    for class_dir in class_dirs:
        # Create class directories
        (train_path / class_dir.name).mkdir(exist_ok=True)
        (val_path / class_dir.name).mkdir(exist_ok=True)

        # Split and copy files
        image_files = list(class_dir.glob("*.*"))
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)

        for img in image_files[:split_idx]:
            shutil.copy2(img, train_path / class_dir.name / img.name)

        for img in image_files[split_idx:]:
            shutil.copy2(img, val_path / class_dir.name / img.name)

    LOGGER.info(f"Split complete in {split_path} ✅")
    return split_path
```
</details>


<br><br><hr><br>

## Function `ultralytics.data.split.autosplit` {#ultralytics.data.split.autosplit}

```python
def autosplit(
    path: Path = DATASETS_DIR / "coco8/images",
    weights: tuple[float, float, float] = (0.9, 0.1, 0.0),
    annotated_only: bool = False,
) -> None
```

Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt

files.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `path` | `Path` | Path to images directory. | `DATASETS_DIR / "coco8/images"` |
| `weights` | `tuple[float, float, float]` | Train, validation, and test split fractions. | `(0.9, 0.1, 0.0)` |
| `annotated_only` | `bool` | If True, only images with an associated txt file are used. | `False` |

**Examples**

```python
Split images with default weights
>>> from ultralytics.data.split import autosplit
>>> autosplit()

Split with custom weights and annotated images only
>>> autosplit(path="path/to/images", weights=(0.8, 0.15, 0.05), annotated_only=True)
```

<details>
<summary>Source code in <code>ultralytics/data/split.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split.py#L98-L134"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def autosplit(
    path: Path = DATASETS_DIR / "coco8/images",
    weights: tuple[float, float, float] = (0.9, 0.1, 0.0),
    annotated_only: bool = False,
) -> None:
    """Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt
    files.

    Args:
        path (Path): Path to images directory.
        weights (tuple[float, float, float]): Train, validation, and test split fractions.
        annotated_only (bool): If True, only images with an associated txt file are used.

    Examples:
        Split images with default weights
        >>> from ultralytics.data.split import autosplit
        >>> autosplit()

        Split with custom weights and annotated images only
        >>> autosplit(path="path/to/images", weights=(0.8, 0.15, 0.05), annotated_only=True)
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], "a", encoding="utf-8") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # add image to txt file
```
</details>

<br><br>
