---
description: Learn how to utilize the ultralytics.data.split_dota module to process and split DOTA datasets efficiently. Explore detailed functions and examples.
keywords: Ultralytics, DOTA dataset, data splitting, YOLO, Python, bbox_iof, load_yolo_dota, get_windows, crop_and_save
---

# Reference for `ultralytics/data/split_dota.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split_dota.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split_dota.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`bbox_iof`](#ultralytics.data.split_dota.bbox_iof)
        - [`load_yolo_dota`](#ultralytics.data.split_dota.load_yolo_dota)
        - [`get_windows`](#ultralytics.data.split_dota.get_windows)
        - [`get_window_obj`](#ultralytics.data.split_dota.get_window_obj)
        - [`crop_and_save`](#ultralytics.data.split_dota.crop_and_save)
        - [`split_images_and_labels`](#ultralytics.data.split_dota.split_images_and_labels)
        - [`split_trainval`](#ultralytics.data.split_dota.split_trainval)
        - [`split_test`](#ultralytics.data.split_dota.split_test)


## Function `ultralytics.data.split_dota.bbox_iof` {#ultralytics.data.split\_dota.bbox\_iof}

```python
def bbox_iof(polygon1: np.ndarray, bbox2: np.ndarray, eps: float = 1e-6) -> np.ndarray
```

Calculate Intersection over Foreground (IoF) between polygons and bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `polygon1` | `np.ndarray` | Polygon coordinates with shape (N, 8). | *required* |
| `bbox2` | `np.ndarray` | Bounding boxes with shape (M, 4). | *required* |
| `eps` | `float, optional` | Small value to prevent division by zero. | `1e-6` |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray` | IoF scores with shape (N, M). |

!!! note "Notes"

    Polygon format: [x1, y1, x2, y2, x3, y3, x4, y4].
    Bounding box format: [x_min, y_min, x_max, y_max].

<details>
<summary>Source code in <code>ultralytics/data/split_dota.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split_dota.py#L20-L63"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def bbox_iof(polygon1: np.ndarray, bbox2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Calculate Intersection over Foreground (IoF) between polygons and bounding boxes.

    Args:
        polygon1 (np.ndarray): Polygon coordinates with shape (N, 8).
        bbox2 (np.ndarray): Bounding boxes with shape (M, 4).
        eps (float, optional): Small value to prevent division by zero.

    Returns:
        (np.ndarray): IoF scores with shape (N, M).

    Notes:
        Polygon format: [x1, y1, x2, y2, x3, y3, x4, y4].
        Bounding box format: [x_min, y_min, x_max, y_max].
    """
    check_requirements("shapely>=2.0.0")
    from shapely.geometry import Polygon

    polygon1 = polygon1.reshape(-1, 4, 2)
    lt_point = np.min(polygon1, axis=-2)  # left-top
    rb_point = np.max(polygon1, axis=-2)  # right-bottom
    bbox1 = np.concatenate([lt_point, rb_point], axis=-1)

    lt = np.maximum(bbox1[:, None, :2], bbox2[..., :2])
    rb = np.minimum(bbox1[:, None, 2:], bbox2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    left, top, right, bottom = (bbox2[..., i] for i in range(4))
    polygon2 = np.stack([left, top, right, top, right, bottom, left, bottom], axis=-1).reshape(-1, 4, 2)

    sg_polys1 = [Polygon(p) for p in polygon1]
    sg_polys2 = [Polygon(p) for p in polygon2]
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs
```
</details>


<br><br><hr><br>

## Function `ultralytics.data.split_dota.load_yolo_dota` {#ultralytics.data.split\_dota.load\_yolo\_dota}

```python
def load_yolo_dota(data_root: str, split: str = "train") -> list[dict[str, Any]]
```

Load DOTA dataset annotations and image information.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `data_root` | `str` | Data root directory. | *required* |
| `split` | `str, optional` | The split data set, could be 'train' or 'val'. | `"train"` |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, Any]]` | List of annotation dictionaries containing image information. |

!!! note "Notes"

    The directory structure assumed for the DOTA dataset:
        - data_root
            - images
                - train
                - val
            - labels
                - train
                - val

<details>
<summary>Source code in <code>ultralytics/data/split_dota.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split_dota.py#L66-L98"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def load_yolo_dota(data_root: str, split: str = "train") -> list[dict[str, Any]]:
    """Load DOTA dataset annotations and image information.

    Args:
        data_root (str): Data root directory.
        split (str, optional): The split data set, could be 'train' or 'val'.

    Returns:
        (list[dict[str, Any]]): List of annotation dictionaries containing image information.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    assert split in {"train", "val"}, f"Split must be 'train' or 'val', not {split}."
    im_dir = Path(data_root) / "images" / split
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(Path(data_root) / "images" / split / "*"))
    lb_files = img2label_paths(im_files)
    annos = []
    for im_file, lb_file in zip(im_files, lb_files):
        w, h = exif_size(Image.open(im_file))
        with open(lb_file, encoding="utf-8") as f:
            lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
            lb = np.array(lb, dtype=np.float32)
        annos.append(dict(ori_size=(h, w), label=lb, filepath=im_file))
    return annos
```
</details>


<br><br><hr><br>

## Function `ultralytics.data.split_dota.get_windows` {#ultralytics.data.split\_dota.get\_windows}

```python
def get_windows(
    im_size: tuple[int, int],
    crop_sizes: tuple[int, ...] = (1024,),
    gaps: tuple[int, ...] = (200,),
    im_rate_thr: float = 0.6,
    eps: float = 0.01,
) -> np.ndarray
```

Get the coordinates of sliding windows for image cropping.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im_size` | `tuple[int, int]` | Original image size, (H, W). | *required* |
| `crop_sizes` | `tuple[int, ...], optional` | Crop size of windows. | `(1024,)` |
| `gaps` | `tuple[int, ...], optional` | Gap between crops. | `(200,)` |
| `im_rate_thr` | `float, optional` | Threshold for the ratio of image area within a window to the total window area. | `0.6` |
| `eps` | `float, optional` | Epsilon value for math operations. | `0.01` |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray` | Array of window coordinates of shape (N, 4) where each row is [x_start, y_start, x_stop, y_stop]. |

<details>
<summary>Source code in <code>ultralytics/data/split_dota.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split_dota.py#L101-L150"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_windows(
    im_size: tuple[int, int],
    crop_sizes: tuple[int, ...] = (1024,),
    gaps: tuple[int, ...] = (200,),
    im_rate_thr: float = 0.6,
    eps: float = 0.01,
) -> np.ndarray:
    """Get the coordinates of sliding windows for image cropping.

    Args:
        im_size (tuple[int, int]): Original image size, (H, W).
        crop_sizes (tuple[int, ...], optional): Crop size of windows.
        gaps (tuple[int, ...], optional): Gap between crops.
        im_rate_thr (float, optional): Threshold for the ratio of image area within a window to the total window area.
        eps (float, optional): Epsilon value for math operations.

    Returns:
        (np.ndarray): Array of window coordinates of shape (N, 4) where each row is [x_start, y_start, x_stop, y_stop].
    """
    h, w = im_size
    windows = []
    for crop_size, gap in zip(crop_sizes, gaps):
        assert crop_size > gap, f"invalid crop_size gap pair [{crop_size} {gap}]"
        step = crop_size - gap

        xn = 1 if w <= crop_size else ceil((w - crop_size) / step + 1)
        xs = [step * i for i in range(xn)]
        if len(xs) > 1 and xs[-1] + crop_size > w:
            xs[-1] = w - crop_size

        yn = 1 if h <= crop_size else ceil((h - crop_size) / step + 1)
        ys = [step * i for i in range(yn)]
        if len(ys) > 1 and ys[-1] + crop_size > h:
            ys[-1] = h - crop_size

        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)
        stop = start + crop_size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    im_in_wins = windows.copy()
    im_in_wins[:, 0::2] = np.clip(im_in_wins[:, 0::2], 0, w)
    im_in_wins[:, 1::2] = np.clip(im_in_wins[:, 1::2], 0, h)
    im_areas = (im_in_wins[:, 2] - im_in_wins[:, 0]) * (im_in_wins[:, 3] - im_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[:, 1])
    im_rates = im_areas / win_areas
    if not (im_rates > im_rate_thr).any():
        max_rate = im_rates.max()
        im_rates[abs(im_rates - max_rate) < eps] = 1
    return windows[im_rates > im_rate_thr]
```
</details>


<br><br><hr><br>

## Function `ultralytics.data.split_dota.get_window_obj` {#ultralytics.data.split\_dota.get\_window\_obj}

```python
def get_window_obj(anno: dict[str, Any], windows: np.ndarray, iof_thr: float = 0.7) -> list[np.ndarray]
```

Get objects for each window based on IoF threshold.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `anno` | `dict[str, Any]` |  | *required* |
| `windows` | `np.ndarray` |  | *required* |
| `iof_thr` | `float` |  | `0.7` |

<details>
<summary>Source code in <code>ultralytics/data/split_dota.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split_dota.py#L153-L164"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_window_obj(anno: dict[str, Any], windows: np.ndarray, iof_thr: float = 0.7) -> list[np.ndarray]:
    """Get objects for each window based on IoF threshold."""
    h, w = anno["ori_size"]
    label = anno["label"]
    if len(label):
        label[:, 1::2] *= w
        label[:, 2::2] *= h
        iofs = bbox_iof(label[:, 1:], windows)
        # Unnormalized and misaligned coordinates
        return [(label[iofs[:, i] >= iof_thr]) for i in range(len(windows))]  # window_anns
    else:
        return [np.zeros((0, 9), dtype=np.float32) for _ in range(len(windows))]  # window_anns
```
</details>


<br><br><hr><br>

## Function `ultralytics.data.split_dota.crop_and_save` {#ultralytics.data.split\_dota.crop\_and\_save}

```python
def crop_and_save(
    anno: dict[str, Any],
    windows: np.ndarray,
    window_objs: list[np.ndarray],
    im_dir: str,
    lb_dir: str,
    allow_background_images: bool = True,
) -> None
```

Crop images and save new labels for each window.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `anno` | `dict[str, Any]` | Annotation dict, including 'filepath', 'label', 'ori_size' as its keys. | *required* |
| `windows` | `np.ndarray` | Array of windows coordinates with shape (N, 4). | *required* |
| `window_objs` | `list[np.ndarray]` | A list of labels inside each window. | *required* |
| `im_dir` | `str` | The output directory path of images. | *required* |
| `lb_dir` | `str` | The output directory path of labels. | *required* |
| `allow_background_images` | `bool, optional` | Whether to include background images without labels. | `True` |

!!! note "Notes"

    The directory structure assumed for the DOTA dataset:
        - data_root
            - images
                - train
                - val
            - labels
                - train
                - val

<details>
<summary>Source code in <code>ultralytics/data/split_dota.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split_dota.py#L167-L215"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def crop_and_save(
    anno: dict[str, Any],
    windows: np.ndarray,
    window_objs: list[np.ndarray],
    im_dir: str,
    lb_dir: str,
    allow_background_images: bool = True,
) -> None:
    """Crop images and save new labels for each window.

    Args:
        anno (dict[str, Any]): Annotation dict, including 'filepath', 'label', 'ori_size' as its keys.
        windows (np.ndarray): Array of windows coordinates with shape (N, 4).
        window_objs (list[np.ndarray]): A list of labels inside each window.
        im_dir (str): The output directory path of images.
        lb_dir (str): The output directory path of labels.
        allow_background_images (bool, optional): Whether to include background images without labels.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    im = cv2.imread(anno["filepath"])
    name = Path(anno["filepath"]).stem
    for i, window in enumerate(windows):
        x_start, y_start, x_stop, y_stop = window.tolist()
        new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
        patch_im = im[y_start:y_stop, x_start:x_stop]
        ph, pw = patch_im.shape[:2]

        label = window_objs[i]
        if len(label) or allow_background_images:
            cv2.imwrite(str(Path(im_dir) / f"{new_name}.jpg"), patch_im)
        if len(label):
            label[:, 1::2] -= x_start
            label[:, 2::2] -= y_start
            label[:, 1::2] /= pw
            label[:, 2::2] /= ph

            with open(Path(lb_dir) / f"{new_name}.txt", "w", encoding="utf-8") as f:
                for lb in label:
                    formatted_coords = [f"{coord:.6g}" for coord in lb[1:]]
                    f.write(f"{int(lb[0])} {' '.join(formatted_coords)}\n")
```
</details>


<br><br><hr><br>

## Function `ultralytics.data.split_dota.split_images_and_labels` {#ultralytics.data.split\_dota.split\_images\_and\_labels}

```python
def split_images_and_labels(
    data_root: str,
    save_dir: str,
    split: str = "train",
    crop_sizes: tuple[int, ...] = (1024,),
    gaps: tuple[int, ...] = (200,),
) -> None
```

Split both images and labels for a given dataset split.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `data_root` | `str` | Root directory of the dataset. | *required* |
| `save_dir` | `str` | Directory to save the split dataset. | *required* |
| `split` | `str, optional` | The split data set, could be 'train' or 'val'. | `"train"` |
| `crop_sizes` | `tuple[int, ...], optional` | Tuple of crop sizes. | `(1024,)` |
| `gaps` | `tuple[int, ...], optional` | Tuple of gaps between crops. | `(200,)` |

!!! note "Notes"

    The directory structure assumed for the DOTA dataset:
        - data_root
            - images
                - split
            - labels
                - split
    and the output directory structure is:
        - save_dir
            - images
                - split
            - labels
                - split

<details>
<summary>Source code in <code>ultralytics/data/split_dota.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split_dota.py#L218-L257"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def split_images_and_labels(
    data_root: str,
    save_dir: str,
    split: str = "train",
    crop_sizes: tuple[int, ...] = (1024,),
    gaps: tuple[int, ...] = (200,),
) -> None:
    """Split both images and labels for a given dataset split.

    Args:
        data_root (str): Root directory of the dataset.
        save_dir (str): Directory to save the split dataset.
        split (str, optional): The split data set, could be 'train' or 'val'.
        crop_sizes (tuple[int, ...], optional): Tuple of crop sizes.
        gaps (tuple[int, ...], optional): Tuple of gaps between crops.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - split
                - labels
                    - split
        and the output directory structure is:
            - save_dir
                - images
                    - split
                - labels
                    - split
    """
    im_dir = Path(save_dir) / "images" / split
    im_dir.mkdir(parents=True, exist_ok=True)
    lb_dir = Path(save_dir) / "labels" / split
    lb_dir.mkdir(parents=True, exist_ok=True)

    annos = load_yolo_dota(data_root, split=split)
    for anno in TQDM(annos, total=len(annos), desc=split):
        windows = get_windows(anno["ori_size"], crop_sizes, gaps)
        window_objs = get_window_obj(anno, windows)
        crop_and_save(anno, windows, window_objs, str(im_dir), str(lb_dir))
```
</details>


<br><br><hr><br>

## Function `ultralytics.data.split_dota.split_trainval` {#ultralytics.data.split\_dota.split\_trainval}

```python
def split_trainval(
    data_root: str, save_dir: str, crop_size: int = 1024, gap: int = 200, rates: tuple[float, ...] = (1.0,)
) -> None
```

Split train and val sets of DOTA dataset with multiple scaling rates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `data_root` | `str` | Root directory of the dataset. | *required* |
| `save_dir` | `str` | Directory to save the split dataset. | *required* |
| `crop_size` | `int, optional` | Base crop size. | `1024` |
| `gap` | `int, optional` | Base gap between crops. | `200` |
| `rates` | `tuple[float, ...], optional` | Scaling rates for crop_size and gap. | `(1.0,)` |

!!! note "Notes"

    The directory structure assumed for the DOTA dataset:
        - data_root
            - images
                - train
                - val
            - labels
                - train
                - val
    and the output directory structure is:
        - save_dir
            - images
                - train
                - val
            - labels
                - train
                - val

<details>
<summary>Source code in <code>ultralytics/data/split_dota.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split_dota.py#L260-L295"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def split_trainval(
    data_root: str, save_dir: str, crop_size: int = 1024, gap: int = 200, rates: tuple[float, ...] = (1.0,)
) -> None:
    """Split train and val sets of DOTA dataset with multiple scaling rates.

    Args:
        data_root (str): Root directory of the dataset.
        save_dir (str): Directory to save the split dataset.
        crop_size (int, optional): Base crop size.
        gap (int, optional): Base gap between crops.
        rates (tuple[float, ...], optional): Scaling rates for crop_size and gap.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
        and the output directory structure is:
            - save_dir
                - images
                    - train
                    - val
                - labels
                    - train
                    - val
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    for split in {"train", "val"}:
        split_images_and_labels(data_root, save_dir, split, crop_sizes, gaps)
```
</details>


<br><br><hr><br>

## Function `ultralytics.data.split_dota.split_test` {#ultralytics.data.split\_dota.split\_test}

```python
def split_test(
    data_root: str, save_dir: str, crop_size: int = 1024, gap: int = 200, rates: tuple[float, ...] = (1.0,)
) -> None
```

Split test set of DOTA dataset, labels are not included within this set.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `data_root` | `str` | Root directory of the dataset. | *required* |
| `save_dir` | `str` | Directory to save the split dataset. | *required* |
| `crop_size` | `int, optional` | Base crop size. | `1024` |
| `gap` | `int, optional` | Base gap between crops. | `200` |
| `rates` | `tuple[float, ...], optional` | Scaling rates for crop_size and gap. | `(1.0,)` |

!!! note "Notes"

    The directory structure assumed for the DOTA dataset:
        - data_root
            - images
                - test
    and the output directory structure is:
        - save_dir
            - images
                - test

<details>
<summary>Source code in <code>ultralytics/data/split_dota.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/split_dota.py#L298-L339"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def split_test(
    data_root: str, save_dir: str, crop_size: int = 1024, gap: int = 200, rates: tuple[float, ...] = (1.0,)
) -> None:
    """Split test set of DOTA dataset, labels are not included within this set.

    Args:
        data_root (str): Root directory of the dataset.
        save_dir (str): Directory to save the split dataset.
        crop_size (int, optional): Base crop size.
        gap (int, optional): Base gap between crops.
        rates (tuple[float, ...], optional): Scaling rates for crop_size and gap.

    Notes:
        The directory structure assumed for the DOTA dataset:
            - data_root
                - images
                    - test
        and the output directory structure is:
            - save_dir
                - images
                    - test
    """
    crop_sizes, gaps = [], []
    for r in rates:
        crop_sizes.append(int(crop_size / r))
        gaps.append(int(gap / r))
    save_dir = Path(save_dir) / "images" / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    im_dir = Path(data_root) / "images" / "test"
    assert im_dir.exists(), f"Can't find {im_dir}, please check your data root."
    im_files = glob(str(im_dir / "*"))
    for im_file in TQDM(im_files, total=len(im_files), desc="test"):
        w, h = exif_size(Image.open(im_file))
        windows = get_windows((h, w), crop_sizes=crop_sizes, gaps=gaps)
        im = cv2.imread(im_file)
        name = Path(im_file).stem
        for window in windows:
            x_start, y_start, x_stop, y_stop = window.tolist()
            new_name = f"{name}__{x_stop - x_start}__{x_start}___{y_start}"
            patch_im = im[y_start:y_stop, x_start:x_stop]
            cv2.imwrite(str(save_dir / f"{new_name}.jpg"), patch_im)
```
</details>

<br><br>
