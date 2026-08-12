---
title: utils.metrics API Reference
description: Explore detailed metrics and utility functions for model validation and performance analysis with Ultralytics' metrics module.
keywords: Ultralytics, metrics, model validation, performance analysis, IoU, confusion matrix
---

# Reference for `ultralytics/utils/metrics.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`ConfusionMatrix`](#ultralytics.utils.metrics.ConfusionMatrix)
        - [`Metric`](#ultralytics.utils.metrics.Metric)
        - [`DetMetrics`](#ultralytics.utils.metrics.DetMetrics)
        - [`SegmentMetrics`](#ultralytics.utils.metrics.SegmentMetrics)
        - [`PoseMetrics`](#ultralytics.utils.metrics.PoseMetrics)
        - [`ClassifyMetrics`](#ultralytics.utils.metrics.ClassifyMetrics)
        - [`OBBMetrics`](#ultralytics.utils.metrics.OBBMetrics)
        - [`SemanticMetrics`](#ultralytics.utils.metrics.SemanticMetrics)
        - [`DepthMetrics`](#ultralytics.utils.metrics.DepthMetrics)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`Metric.ap50`](#ultralytics.utils.metrics.Metric.ap50)
        - [`Metric.ap`](#ultralytics.utils.metrics.Metric.ap)
        - [`Metric.mp`](#ultralytics.utils.metrics.Metric.mp)
        - [`Metric.mr`](#ultralytics.utils.metrics.Metric.mr)
        - [`Metric.map50`](#ultralytics.utils.metrics.Metric.map50)
        - [`Metric.map75`](#ultralytics.utils.metrics.Metric.map75)
        - [`Metric.map`](#ultralytics.utils.metrics.Metric.map)
        - [`Metric.maps`](#ultralytics.utils.metrics.Metric.maps)
        - [`Metric.curves`](#ultralytics.utils.metrics.Metric.curves)
        - [`Metric.curves_results`](#ultralytics.utils.metrics.Metric.curves_results)
        - [`DetMetrics.keys`](#ultralytics.utils.metrics.DetMetrics.keys)
        - [`DetMetrics.maps`](#ultralytics.utils.metrics.DetMetrics.maps)
        - [`DetMetrics.fitness`](#ultralytics.utils.metrics.DetMetrics.fitness)
        - [`DetMetrics.ap_class_index`](#ultralytics.utils.metrics.DetMetrics.ap_class_index)
        - [`DetMetrics.results_dict`](#ultralytics.utils.metrics.DetMetrics.results_dict)
        - [`DetMetrics.curves`](#ultralytics.utils.metrics.DetMetrics.curves)
        - [`DetMetrics.curves_results`](#ultralytics.utils.metrics.DetMetrics.curves_results)
        - [`SegmentMetrics.keys`](#ultralytics.utils.metrics.SegmentMetrics.keys)
        - [`SegmentMetrics.maps`](#ultralytics.utils.metrics.SegmentMetrics.maps)
        - [`SegmentMetrics.fitness`](#ultralytics.utils.metrics.SegmentMetrics.fitness)
        - [`SegmentMetrics.curves`](#ultralytics.utils.metrics.SegmentMetrics.curves)
        - [`SegmentMetrics.curves_results`](#ultralytics.utils.metrics.SegmentMetrics.curves_results)
        - [`PoseMetrics.keys`](#ultralytics.utils.metrics.PoseMetrics.keys)
        - [`PoseMetrics.maps`](#ultralytics.utils.metrics.PoseMetrics.maps)
        - [`PoseMetrics.fitness`](#ultralytics.utils.metrics.PoseMetrics.fitness)
        - [`PoseMetrics.curves`](#ultralytics.utils.metrics.PoseMetrics.curves)
        - [`PoseMetrics.curves_results`](#ultralytics.utils.metrics.PoseMetrics.curves_results)
        - [`ClassifyMetrics.fitness`](#ultralytics.utils.metrics.ClassifyMetrics.fitness)
        - [`ClassifyMetrics.results_dict`](#ultralytics.utils.metrics.ClassifyMetrics.results_dict)
        - [`ClassifyMetrics.keys`](#ultralytics.utils.metrics.ClassifyMetrics.keys)
        - [`ClassifyMetrics.curves`](#ultralytics.utils.metrics.ClassifyMetrics.curves)
        - [`ClassifyMetrics.curves_results`](#ultralytics.utils.metrics.ClassifyMetrics.curves_results)
        - [`SemanticMetrics.miou`](#ultralytics.utils.metrics.SemanticMetrics.miou)
        - [`SemanticMetrics.pixel_accuracy`](#ultralytics.utils.metrics.SemanticMetrics.pixel_accuracy)
        - [`SemanticMetrics.per_class_iou`](#ultralytics.utils.metrics.SemanticMetrics.per_class_iou)
        - [`SemanticMetrics.per_class_pixel_accuracy`](#ultralytics.utils.metrics.SemanticMetrics.per_class_pixel_accuracy)
        - [`SemanticMetrics.fitness`](#ultralytics.utils.metrics.SemanticMetrics.fitness)
        - [`SemanticMetrics.keys`](#ultralytics.utils.metrics.SemanticMetrics.keys)
        - [`SemanticMetrics.ap_class_index`](#ultralytics.utils.metrics.SemanticMetrics.ap_class_index)
        - [`SemanticMetrics.results_dict`](#ultralytics.utils.metrics.SemanticMetrics.results_dict)
        - [`SemanticMetrics.curves`](#ultralytics.utils.metrics.SemanticMetrics.curves)
        - [`SemanticMetrics.curves_results`](#ultralytics.utils.metrics.SemanticMetrics.curves_results)
        - [`DepthMetrics.keys`](#ultralytics.utils.metrics.DepthMetrics.keys)
        - [`DepthMetrics.delta1`](#ultralytics.utils.metrics.DepthMetrics.delta1)
        - [`DepthMetrics.delta2`](#ultralytics.utils.metrics.DepthMetrics.delta2)
        - [`DepthMetrics.delta3`](#ultralytics.utils.metrics.DepthMetrics.delta3)
        - [`DepthMetrics.abs_rel`](#ultralytics.utils.metrics.DepthMetrics.abs_rel)
        - [`DepthMetrics.rmse`](#ultralytics.utils.metrics.DepthMetrics.rmse)
        - [`DepthMetrics.silog`](#ultralytics.utils.metrics.DepthMetrics.silog)
        - [`DepthMetrics.fitness`](#ultralytics.utils.metrics.DepthMetrics.fitness)
        - [`DepthMetrics.results_dict`](#ultralytics.utils.metrics.DepthMetrics.results_dict)
        - [`DepthMetrics.curves`](#ultralytics.utils.metrics.DepthMetrics.curves)
        - [`DepthMetrics.curves_results`](#ultralytics.utils.metrics.DepthMetrics.curves_results)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`ConfusionMatrix._append_matches`](#ultralytics.utils.metrics.ConfusionMatrix._append_matches)
        - [`ConfusionMatrix.process_cls_preds`](#ultralytics.utils.metrics.ConfusionMatrix.process_cls_preds)
        - [`ConfusionMatrix.process_batch`](#ultralytics.utils.metrics.ConfusionMatrix.process_batch)
        - [`ConfusionMatrix.tp_fp`](#ultralytics.utils.metrics.ConfusionMatrix.tp_fp)
        - [`ConfusionMatrix.plot_matches`](#ultralytics.utils.metrics.ConfusionMatrix.plot_matches)
        - [`ConfusionMatrix.plot`](#ultralytics.utils.metrics.ConfusionMatrix.plot)
        - [`ConfusionMatrix.print`](#ultralytics.utils.metrics.ConfusionMatrix.print)
        - [`ConfusionMatrix.summary`](#ultralytics.utils.metrics.ConfusionMatrix.summary)
        - [`Metric.mean_results`](#ultralytics.utils.metrics.Metric.mean_results)
        - [`Metric.class_result`](#ultralytics.utils.metrics.Metric.class_result)
        - [`Metric.fitness`](#ultralytics.utils.metrics.Metric.fitness)
        - [`Metric.update`](#ultralytics.utils.metrics.Metric.update)
        - [`Metric.clear_image_metrics`](#ultralytics.utils.metrics.Metric.clear_image_metrics)
        - [`Metric.update_image_metrics`](#ultralytics.utils.metrics.Metric.update_image_metrics)
        - [`DetMetrics.update_stats`](#ultralytics.utils.metrics.DetMetrics.update_stats)
        - [`DetMetrics.process`](#ultralytics.utils.metrics.DetMetrics.process)
        - [`DetMetrics.clear_stats`](#ultralytics.utils.metrics.DetMetrics.clear_stats)
        - [`DetMetrics.clear_image_metrics`](#ultralytics.utils.metrics.DetMetrics.clear_image_metrics)
        - [`DetMetrics.mean_results`](#ultralytics.utils.metrics.DetMetrics.mean_results)
        - [`DetMetrics.class_result`](#ultralytics.utils.metrics.DetMetrics.class_result)
        - [`DetMetrics.summary`](#ultralytics.utils.metrics.DetMetrics.summary)
        - [`SegmentMetrics.update_stats`](#ultralytics.utils.metrics.SegmentMetrics.update_stats)
        - [`SegmentMetrics.clear_image_metrics`](#ultralytics.utils.metrics.SegmentMetrics.clear_image_metrics)
        - [`SegmentMetrics.process`](#ultralytics.utils.metrics.SegmentMetrics.process)
        - [`SegmentMetrics.mean_results`](#ultralytics.utils.metrics.SegmentMetrics.mean_results)
        - [`SegmentMetrics.class_result`](#ultralytics.utils.metrics.SegmentMetrics.class_result)
        - [`SegmentMetrics.summary`](#ultralytics.utils.metrics.SegmentMetrics.summary)
        - [`PoseMetrics.update_stats`](#ultralytics.utils.metrics.PoseMetrics.update_stats)
        - [`PoseMetrics.clear_image_metrics`](#ultralytics.utils.metrics.PoseMetrics.clear_image_metrics)
        - [`PoseMetrics.process`](#ultralytics.utils.metrics.PoseMetrics.process)
        - [`PoseMetrics.mean_results`](#ultralytics.utils.metrics.PoseMetrics.mean_results)
        - [`PoseMetrics.class_result`](#ultralytics.utils.metrics.PoseMetrics.class_result)
        - [`PoseMetrics.summary`](#ultralytics.utils.metrics.PoseMetrics.summary)
        - [`ClassifyMetrics.process`](#ultralytics.utils.metrics.ClassifyMetrics.process)
        - [`ClassifyMetrics.summary`](#ultralytics.utils.metrics.ClassifyMetrics.summary)
        - [`SemanticMetrics.update_stats`](#ultralytics.utils.metrics.SemanticMetrics.update_stats)
        - [`SemanticMetrics.process`](#ultralytics.utils.metrics.SemanticMetrics.process)
        - [`SemanticMetrics.clear_stats`](#ultralytics.utils.metrics.SemanticMetrics.clear_stats)
        - [`SemanticMetrics._plot_iou_bars`](#ultralytics.utils.metrics.SemanticMetrics._plot_iou_bars)
        - [`SemanticMetrics.mean_results`](#ultralytics.utils.metrics.SemanticMetrics.mean_results)
        - [`SemanticMetrics.class_result`](#ultralytics.utils.metrics.SemanticMetrics.class_result)
        - [`SemanticMetrics.summary`](#ultralytics.utils.metrics.SemanticMetrics.summary)
        - [`DepthMetrics.update_stats`](#ultralytics.utils.metrics.DepthMetrics.update_stats)
        - [`DepthMetrics.process`](#ultralytics.utils.metrics.DepthMetrics.process)
        - [`DepthMetrics.clear_stats`](#ultralytics.utils.metrics.DepthMetrics.clear_stats)
        - [`DepthMetrics.mean_results`](#ultralytics.utils.metrics.DepthMetrics.mean_results)
        - [`DepthMetrics.summary`](#ultralytics.utils.metrics.DepthMetrics.summary)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`bbox_ioa`](#ultralytics.utils.metrics.bbox_ioa)
        - [`box_iou`](#ultralytics.utils.metrics.box_iou)
        - [`bbox_iou`](#ultralytics.utils.metrics.bbox_iou)
        - [`mask_iou`](#ultralytics.utils.metrics.mask_iou)
        - [`kpt_iou`](#ultralytics.utils.metrics.kpt_iou)
        - [`_get_covariance_matrix`](#ultralytics.utils.metrics._get_covariance_matrix)
        - [`probiou`](#ultralytics.utils.metrics.probiou)
        - [`batch_probiou`](#ultralytics.utils.metrics.batch_probiou)
        - [`smooth_bce`](#ultralytics.utils.metrics.smooth_bce)
        - [`smooth`](#ultralytics.utils.metrics.smooth)
        - [`plot_pr_curve`](#ultralytics.utils.metrics.plot_pr_curve)
        - [`plot_mc_curve`](#ultralytics.utils.metrics.plot_mc_curve)
        - [`compute_ap`](#ultralytics.utils.metrics.compute_ap)
        - [`ap_per_class`](#ultralytics.utils.metrics.ap_per_class)


## Class `ultralytics.utils.metrics.ConfusionMatrix` {#ultralytics.utils.metrics.ConfusionMatrix}

```python
ConfusionMatrix(names: dict[int, str] | None = None, task: str = "detect", save_matches: bool = False)
```

**Bases:** `DataExportMixin`

A class for calculating and updating a confusion matrix for object detection and classification tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `names` | `dict[int, str], optional` | Names of classes, used as labels on the plot. | `None` |
| `task` | `str, optional` | Type of task, one of 'detect', 'classify', 'semantic', or 'obb'. | `"detect"` |
| `save_matches` | `bool, optional` | Save the indices of GTs, TPs, FPs, FNs for visualization. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `task` | `str` | The type of task, one of 'detect', 'classify', 'semantic', or 'obb'. |
| `matrix` | `np.ndarray` | The confusion matrix, with dimensions depending on the task. |
| `nc` | `int` | The number of classes. |
| `names` | `dict[int, str]` | The names of the classes, used as labels on the plot. |
| `matches` | `dict \| None` | Contains the indices of ground truths and predictions categorized into TP, FP and FN. |

**Methods**

| Name | Description |
| --- | --- |
| [`_append_matches`](#ultralytics.utils.metrics.ConfusionMatrix._append_matches) | Append the matches to TP, FP, FN or GT list for the last batch. |
| [`plot`](#ultralytics.utils.metrics.ConfusionMatrix.plot) | Plot the confusion matrix using matplotlib and save it to a file. |
| [`plot_matches`](#ultralytics.utils.metrics.ConfusionMatrix.plot_matches) | Plot grid of GT, TP, FP, FN for each image. |
| [`print`](#ultralytics.utils.metrics.ConfusionMatrix.print) | Print the confusion matrix to the console. |
| [`process_batch`](#ultralytics.utils.metrics.ConfusionMatrix.process_batch) | Update confusion matrix for object detection task. |
| [`process_cls_preds`](#ultralytics.utils.metrics.ConfusionMatrix.process_cls_preds) | Update confusion matrix for classification task. |
| [`summary`](#ultralytics.utils.metrics.ConfusionMatrix.summary) | Generate a summarized representation of the confusion matrix as a list of dictionaries, with optional |
| [`tp_fp`](#ultralytics.utils.metrics.ConfusionMatrix.tp_fp) | Return true positives and false positives. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L336-L656">View on GitHub</a>
```python
class ConfusionMatrix(DataExportMixin):
    """A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, one of 'detect', 'classify', 'semantic', or 'obb'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        names (dict[int, str]): The names of the classes, used as labels on the plot.
        matches (dict | None): Contains the indices of ground truths and predictions categorized into TP, FP and FN.
    """

    def __init__(self, names: dict[int, str] | None = None, task: str = "detect", save_matches: bool = False):
        """Initialize a ConfusionMatrix instance.

        Args:
            names (dict[int, str], optional): Names of classes, used as labels on the plot.
            task (str, optional): Type of task, one of 'detect', 'classify', 'semantic', or 'obb'.
            save_matches (bool, optional): Save the indices of GTs, TPs, FPs, FNs for visualization.
        """
        names = names if names is not None else {}
        self.task = task
        self.nc = len(names)  # number of classes
        self.matrix = (
            np.zeros((self.nc, self.nc))
            if self.task in {"classify", "semantic"}
            else np.zeros((self.nc + 1, self.nc + 1))
        )
        self.names = names  # name of classes
        self.matches = {} if save_matches else None
```
</details>

<br>

### Method `ultralytics.utils.metrics.ConfusionMatrix._append_matches` {#ultralytics.utils.metrics.ConfusionMatrix.\_append\_matches}

```python
def _append_matches(self, mtype: str, batch: dict[str, Any], idx: int) -> None
```

Append the matches to TP, FP, FN or GT list for the last batch.

This method updates the matches dictionary by appending specific batch data to the appropriate match type (True Positive, False Positive, or False Negative).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `mtype` | `str` | Match type identifier ('TP', 'FP', 'FN' or 'GT'). | *required* |
| `batch` | `dict[str, Any]` | Batch data containing detection results with keys like 'bboxes', 'cls', 'conf', 'keypoints', 'masks'. | *required* |
| `idx` | `int` | Index of the specific detection to append from the batch. | *required* |

!!! note "Notes"

    For masks, handles both overlap and non-overlap cases. When masks.max() > 1.0, it indicates
    overlap_mask=True with shape (1, H, W), otherwise uses direct indexing.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L366-L389">View on GitHub</a>
```python
def _append_matches(self, mtype: str, batch: dict[str, Any], idx: int) -> None:
    """Append the matches to TP, FP, FN or GT list for the last batch.

    This method updates the matches dictionary by appending specific batch data to the appropriate match type (True
    Positive, False Positive, or False Negative).

    Args:
        mtype (str): Match type identifier ('TP', 'FP', 'FN' or 'GT').
        batch (dict[str, Any]): Batch data containing detection results with keys like 'bboxes', 'cls', 'conf',
            'keypoints', 'masks'.
        idx (int): Index of the specific detection to append from the batch.

    Notes:
        For masks, handles both overlap and non-overlap cases. When masks.max() > 1.0, it indicates
        overlap_mask=True with shape (1, H, W), otherwise uses direct indexing.
    """
    if self.matches is None:
        return
    for k, v in batch.items():
        if k in {"bboxes", "cls", "conf", "keypoints"}:
            self.matches[mtype][k] += v[[idx]]
        elif k == "masks":
            # NOTE: masks.max() > 1.0 means overlap_mask=True with (1, H, W) shape
            self.matches[mtype][k] += [v[0] == idx + 1] if v.max() > 1.0 else [v[idx]]
```
</details>

<br>

### Method `ultralytics.utils.metrics.ConfusionMatrix.plot` {#ultralytics.utils.metrics.ConfusionMatrix.plot}

```python
def plot(self, normalize: bool = True, save_dir: str = "", on_plot=None)
```

Plot the confusion matrix using matplotlib and save it to a file.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool, optional` | Whether to normalize the confusion matrix. | `True` |
| `save_dir` | `str, optional` | Directory where the plot will be saved. | `""` |
| `on_plot` | `callable, optional` | An optional callback to pass plots path and data when they are rendered. | `None` |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L539-L610">View on GitHub</a>
```python
@TryExcept(msg="ConfusionMatrix plot failure")
@plt_settings()
def plot(self, normalize: bool = True, save_dir: str = "", on_plot=None):
    """Plot the confusion matrix using matplotlib and save it to a file.

    Args:
        normalize (bool, optional): Whether to normalize the confusion matrix.
        save_dir (str, optional): Directory where the plot will be saved.
        on_plot (callable, optional): An optional callback to pass plots path and data when they are rendered.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    names, n = list(self.names.values()), self.nc
    if self.nc >= 100:  # downsample for large class count
        k = max(2, self.nc // 60)  # step size for downsampling, always > 1
        keep_idx = slice(None, None, k)  # create slice instead of array
        names = names[keep_idx]  # slice class names
        array = array[keep_idx, :][:, keep_idx]  # slice matrix rows and cols
        n = (self.nc + k - 1) // k  # number of retained classes
    nc = n if self.task in {"classify", "semantic"} else n + 1  # adjust for background if needed
    ticklabels = "auto"
    if 0 < nc < 99:
        ticklabels = names if self.task in {"classify", "semantic"} else [*names, "background"]
    xy_ticks = np.arange(len(ticklabels)) if ticklabels != "auto" else np.arange(nc)
    tick_fontsize = max(6, 15 - 0.1 * nc)  # Minimum size is 6
    label_fontsize = max(6, 12 - 0.1 * nc)
    title_fontsize = max(6, 12 - 0.1 * nc)
    btm = max(0.1, 0.25 - 0.001 * nc)  # Minimum value is 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
        im = ax.imshow(array, cmap="Blues", vmin=0.0, interpolation="none")
        ax.xaxis.set_label_position("bottom")
        if nc < 30:  # Add score for each cell of confusion matrix
            color_threshold = 0.45 * (1 if normalize else np.nanmax(array))  # text color threshold
            for i, row in enumerate(array[:nc]):
                for j, val in enumerate(row[:nc]):
                    val = array[i, j]
                    if np.isnan(val):
                        continue
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}" if normalize else f"{int(val)}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if val > color_threshold else "black",
                    )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    title = "Confusion Matrix" + " Normalized" * normalize
    ax.set_xlabel("True", fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel("Predicted", fontsize=label_fontsize, labelpad=10)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.set_xticks(xy_ticks)
    ax.set_yticks(xy_ticks)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
    if ticklabels != "auto":
        ax.set_xticklabels(ticklabels, fontsize=tick_fontsize, rotation=90, ha="center")
        ax.set_yticklabels(ticklabels, fontsize=tick_fontsize)
    for s in ("left", "right", "bottom", "top", "outline"):
        if s != "outline":
            ax.spines[s].set_visible(False)  # Confusion matrix plot don't have outline
        cbar.ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=btm)  # Adjust layout to ensure equal margins
    plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
    fig.savefig(plot_fname, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(plot_fname, {"type": "confusion_matrix", "matrix": self.matrix.tolist()})
```
</details>

<br>

### Method `ultralytics.utils.metrics.ConfusionMatrix.plot_matches` {#ultralytics.utils.metrics.ConfusionMatrix.plot\_matches}

```python
def plot_matches(
    self, img: torch.Tensor, im_file: str, save_dir: Path, show_labels: bool = True, show_conf: bool = True
) -> None
```

Plot grid of GT, TP, FP, FN for each image.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img` | `torch.Tensor` | Image to plot onto. | *required* |
| `im_file` | `str` | Image filename to save visualizations. | *required* |
| `save_dir` | `Path` | Location to save the visualizations to. | *required* |
| `show_labels` | `bool` | Whether to display class labels in the visualization. | `True` |
| `show_conf` | `bool` | Whether to display confidence values in the visualization. | `True` |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L494-L535">View on GitHub</a>
```python
def plot_matches(
    self, img: torch.Tensor, im_file: str, save_dir: Path, show_labels: bool = True, show_conf: bool = True
) -> None:
    """Plot grid of GT, TP, FP, FN for each image.

    Args:
        img (torch.Tensor): Image to plot onto.
        im_file (str): Image filename to save visualizations.
        save_dir (Path): Location to save the visualizations to.
        show_labels (bool): Whether to display class labels in the visualization.
        show_conf (bool): Whether to display confidence values in the visualization.
    """
    if not self.matches:
        return
    from .ops import xyxy2xywh
    from .plotting import plot_images

    # Create batch of 4 (GT, TP, FP, FN)
    labels = defaultdict(list)
    for i, mtype in enumerate(["GT", "FP", "TP", "FN"]):
        mbatch = self.matches[mtype]
        if "conf" not in mbatch:
            mbatch["conf"] = torch.tensor([1.0] * len(mbatch["bboxes"]), device=img.device)
        mbatch["batch_idx"] = torch.ones(len(mbatch["bboxes"]), device=img.device) * i
        for k in mbatch:
            labels[k] += mbatch[k]

    labels = {k: torch.stack(v, 0) if len(v) else torch.empty(0) for k, v in labels.items()}
    if self.task != "obb" and labels["bboxes"].shape[0]:
        labels["bboxes"] = xyxy2xywh(labels["bboxes"])
    (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    plot_images(
        labels,
        img.repeat(4, 1, 1, 1),
        paths=["Ground Truth", "False Positives", "True Positives", "False Negatives"],
        fname=save_dir / "visualizations" / Path(im_file).name,
        names=self.names,
        max_subplots=4,
        conf_thres=0.001,
        show_labels=show_labels,
        show_conf=show_conf,
    )
```
</details>

<br>

### Method `ultralytics.utils.metrics.ConfusionMatrix.print` {#ultralytics.utils.metrics.ConfusionMatrix.print}

```python
def print(self)
```

Print the confusion matrix to the console.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L612-L615">View on GitHub</a>
```python
def print(self):
    """Print the confusion matrix to the console."""
    for i in range(self.matrix.shape[0]):
        LOGGER.info(" ".join(map(str, self.matrix[i])))
```
</details>

<br>

### Method `ultralytics.utils.metrics.ConfusionMatrix.process_batch` {#ultralytics.utils.metrics.ConfusionMatrix.process\_batch}

```python
def process_batch(
    self,
    detections: dict[str, torch.Tensor],
    batch: dict[str, Any],
    conf: float = 0.25,
    iou_thres: float = 0.45,
) -> None
```

Update confusion matrix for object detection task.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `detections` | `dict[str, torch.Tensor]` | Dictionary containing detected bounding boxes and their associated information. Should contain 'cls', 'conf', and 'bboxes' keys, where 'bboxes' can be Array[N, 4] for regular boxes or Array[N, 5] for OBB with angle. | *required* |
| `batch` | `dict[str, Any]` | Batch dictionary containing ground truth data with 'bboxes' (Array[M, 4]\| Array[M, 5]) and 'cls' (Array[M]) keys, where M is the number of ground truth objects. | *required* |
| `conf` | `float, optional` | Confidence threshold for detections. | `0.25` |
| `iou_thres` | `float, optional` | IoU threshold for matching detections to ground truth. | `0.45` |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L402-L480">View on GitHub</a>
```python
def process_batch(
    self,
    detections: dict[str, torch.Tensor],
    batch: dict[str, Any],
    conf: float = 0.25,
    iou_thres: float = 0.45,
) -> None:
    """Update confusion matrix for object detection task.

    Args:
        detections (dict[str, torch.Tensor]): Dictionary containing detected bounding boxes and their associated
            information. Should contain 'cls', 'conf', and 'bboxes' keys, where 'bboxes' can be Array[N, 4] for
            regular boxes or Array[N, 5] for OBB with angle.
        batch (dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' (Array[M, 4]| Array[M,
            5]) and 'cls' (Array[M]) keys, where M is the number of ground truth objects.
        conf (float, optional): Confidence threshold for detections.
        iou_thres (float, optional): IoU threshold for matching detections to ground truth.
    """
    gt_cls, gt_bboxes = batch["cls"], batch["bboxes"]
    if self.matches is not None:  # only if visualization is enabled
        self.matches = {k: defaultdict(list) for k in ("TP", "FP", "FN", "GT")}
        for i in range(gt_cls.shape[0]):
            self._append_matches("GT", batch, i)  # store GT
    is_obb = gt_bboxes.shape[1] == 5  # check if boxes contains angle for OBB
    no_pred = detections["cls"].shape[0] == 0
    if gt_cls.shape[0] == 0:  # Check if labels is empty
        if not no_pred:
            detections = {k: detections[k][detections["conf"] > conf] for k in detections}
            detection_classes = detections["cls"].int().tolist()
            for i, dc in enumerate(detection_classes):
                self.matrix[dc, self.nc] += 1  # FP
                self._append_matches("FP", detections, i)
        return
    if no_pred:
        gt_classes = gt_cls.int().tolist()
        for i, gc in enumerate(gt_classes):
            self.matrix[self.nc, gc] += 1  # FN
            self._append_matches("FN", batch, i)
        return

    detections = {k: detections[k][detections["conf"] > conf] for k in detections}
    gt_classes = gt_cls.int().tolist()
    detection_classes = detections["cls"].int().tolist()
    bboxes = detections["bboxes"]
    iou = batch_probiou(gt_bboxes, bboxes) if is_obb else box_iou(gt_bboxes, bboxes)

    x = torch.where(iou > iou_thres)
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    else:
        matches = np.zeros((0, 3))

    m0, m1, _ = matches.transpose().astype(int)
    # matches is deduplicated on both columns, so each gt and each detection appears at most once
    gt_match = np.full(len(gt_classes), -1)
    gt_match[m0] = m1
    matched_det = set(m1.tolist())
    for i, gc in enumerate(gt_classes):
        if (di := gt_match[i].item()) >= 0:
            dc = detection_classes[di]
            self.matrix[dc, gc] += 1  # TP if class is correct else both an FP and an FN
            if dc == gc:
                self._append_matches("TP", detections, di)
            else:
                self._append_matches("FP", detections, di)
                self._append_matches("FN", batch, i)
        else:
            self.matrix[self.nc, gc] += 1  # FN
            self._append_matches("FN", batch, i)

    for i, dc in enumerate(detection_classes):
        if i not in matched_det:
            self.matrix[dc, self.nc] += 1  # FP
            self._append_matches("FP", detections, i)
```
</details>

<br>

### Method `ultralytics.utils.metrics.ConfusionMatrix.process_cls_preds` {#ultralytics.utils.metrics.ConfusionMatrix.process\_cls\_preds}

```python
def process_cls_preds(self, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> None
```

Update confusion matrix for classification task.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `list[torch.Tensor]` | Predicted class labels. | *required* |
| `targets` | `list[torch.Tensor]` | Ground truth class labels. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L391-L400">View on GitHub</a>
```python
def process_cls_preds(self, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
    """Update confusion matrix for classification task.

    Args:
        preds (list[torch.Tensor]): Predicted class labels.
        targets (list[torch.Tensor]): Ground truth class labels.
    """
    preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
    for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
        self.matrix[p][t] += 1
```
</details>

<br>

### Method `ultralytics.utils.metrics.ConfusionMatrix.summary` {#ultralytics.utils.metrics.ConfusionMatrix.summary}

```python
def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, float]]
```

Generate a summarized representation of the confusion matrix as a list of dictionaries, with optional

normalization. This is useful for exporting the matrix to various formats such as CSV, XML, HTML, JSON, or SQL.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool` | Whether to normalize the confusion matrix values. | `False` |
| `decimals` | `int` | Number of decimal places to round the output values to. | `5` |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, float]]` | A list of dictionaries, each representing one predicted class with corresponding values for all actual classes. |

**Examples**

```python
>>> results = model.val(data="coco8.yaml", plots=True)
>>> cm_dict = results.confusion_matrix.summary(normalize=True, decimals=5)
>>> print(cm_dict)
```

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L617-L656">View on GitHub</a>
```python
def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, float]]:
    """Generate a summarized representation of the confusion matrix as a list of dictionaries, with optional
    normalization. This is useful for exporting the matrix to various formats such as CSV, XML, HTML, JSON,
    or SQL.

    Args:
        normalize (bool): Whether to normalize the confusion matrix values.
        decimals (int): Number of decimal places to round the output values to.

    Returns:
        (list[dict[str, float]]): A list of dictionaries, each representing one predicted class with corresponding
            values for all actual classes.

    Examples:
        >>> results = model.val(data="coco8.yaml", plots=True)
        >>> cm_dict = results.confusion_matrix.summary(normalize=True, decimals=5)
        >>> print(cm_dict)
    """
    import re

    names = (
        list(self.names.values())
        if self.task in {"classify", "semantic"}
        else [*list(self.names.values()), "background"]
    )
    clean_names, seen = [], set()
    for name in names:
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        original_clean = clean_name
        counter = 1
        while clean_name.lower() in seen:
            clean_name = f"{original_clean}_{counter}"
            counter += 1
        seen.add(clean_name.lower())
        clean_names.append(clean_name)
    array = (self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)).round(decimals)
    return [
        dict({"Predicted": clean_names[i]}, **{clean_names[j]: array[i, j] for j in range(len(clean_names))})
        for i in range(len(clean_names))
    ]
```
</details>

<br>

### Method `ultralytics.utils.metrics.ConfusionMatrix.tp_fp` {#ultralytics.utils.metrics.ConfusionMatrix.tp\_fp}

```python
def tp_fp(self) -> tuple[np.ndarray, np.ndarray]
```

Return true positives and false positives.

**Returns**

| Type | Description |
| --- | --- |
| `tp (np.ndarray)` | True positives. |
| `fp (np.ndarray)` | False positives. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L482-L492">View on GitHub</a>
```python
def tp_fp(self) -> tuple[np.ndarray, np.ndarray]:
    """Return true positives and false positives.

    Returns:
        tp (np.ndarray): True positives.
        fp (np.ndarray): False positives.
    """
    tp = self.matrix.diagonal()  # true positives
    fp = self.matrix.sum(1) - tp  # false positives
    # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
    return (tp, fp) if self.task in {"classify", "semantic"} else (tp[:-1], fp[:-1])  # remove background row/col
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.metrics.Metric` {#ultralytics.utils.metrics.Metric}

```python
Metric()
```

**Bases:** `SimpleClass`

Class for computing evaluation metrics for Ultralytics YOLO models.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `p` | `list` | Precision for each class. Shape: (nc,). |
| `r` | `list` | Recall for each class. Shape: (nc,). |
| `f1` | `list` | F1 score for each class. Shape: (nc,). |
| `all_ap` | `list` | AP scores for all classes and all IoU thresholds. Shape: (nc, 10). |
| `ap_class_index` | `list` | Index of class for each AP score. Shape: (nc,). |
| `nc` | `int` | Number of classes. |

**Methods**

| Name | Description |
| --- | --- |
| [`ap50`](#ultralytics.utils.metrics.Metric.ap50) | Return the Average Precision (AP) at an IoU threshold of 0.5 for all classes. |
| [`ap`](#ultralytics.utils.metrics.Metric.ap) | Return the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes. |
| [`mp`](#ultralytics.utils.metrics.Metric.mp) | Return the Mean Precision of all classes. |
| [`mr`](#ultralytics.utils.metrics.Metric.mr) | Return the Mean Recall of all classes. |
| [`map50`](#ultralytics.utils.metrics.Metric.map50) | Return the mean Average Precision (mAP) at an IoU threshold of 0.5. |
| [`map75`](#ultralytics.utils.metrics.Metric.map75) | Return the mean Average Precision (mAP) at an IoU threshold of 0.75. |
| [`map`](#ultralytics.utils.metrics.Metric.map) | Return the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05. |
| [`maps`](#ultralytics.utils.metrics.Metric.maps) | Return mAP of each class. |
| [`curves`](#ultralytics.utils.metrics.Metric.curves) | Return a list of curves for accessing specific metrics curves. |
| [`curves_results`](#ultralytics.utils.metrics.Metric.curves_results) | Return a list of curves results for accessing specific metrics curves. |
| [`class_result`](#ultralytics.utils.metrics.Metric.class_result) | Return class-aware result, p[i], r[i], ap50[i], ap[i]. |
| [`clear_image_metrics`](#ultralytics.utils.metrics.Metric.clear_image_metrics) | Clear stored per-image metrics from the current validation run. |
| [`fitness`](#ultralytics.utils.metrics.Metric.fitness) | Return model fitness as a weighted combination of metrics. |
| [`mean_results`](#ultralytics.utils.metrics.Metric.mean_results) | Return mean of results, mp, mr, map50, map. |
| [`update`](#ultralytics.utils.metrics.Metric.update) | Update the evaluation metrics with a new set of results. |
| [`update_image_metrics`](#ultralytics.utils.metrics.Metric.update_image_metrics) | Update per-image precision, recall, F1, TP, FP, and FN at IoU threshold 0.5. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L890-L1092">View on GitHub</a>
```python
class Metric(SimpleClass):
    """Class for computing evaluation metrics for Ultralytics YOLO models.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50: AP at IoU threshold of 0.5 for all classes.
        ap: AP at IoU thresholds from 0.5 to 0.95 for all classes.
        mp: Mean precision of all classes.
        mr: Mean recall of all classes.
        map50: Mean AP at IoU threshold of 0.5 for all classes.
        map75: Mean AP at IoU threshold of 0.75 for all classes.
        map: Mean AP at IoU thresholds from 0.5 to 0.95 for all classes.
        mean_results: Mean of results, returns mp, mr, map50, map.
        class_result: Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps: mAP of each class.
        fitness: Model fitness as a weighted combination of metrics.
        update: Update metric attributes with new evaluation results.
        curves: Provides a list of curves for accessing specific metrics like precision, recall, F1, etc.
        curves_results: Provide a list of results for accessing specific metrics like precision, recall, F1, etc.
    """

    def __init__(self) -> None:
        """Initialize a Metric instance for computing evaluation metrics for the YOLO model."""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0
        self.image_metrics = {}
```
</details>

<br>

### Property `ultralytics.utils.metrics.Metric.ap50` {#ultralytics.utils.metrics.Metric.ap50}

```python
def ap50(self) -> np.ndarray | list
```

Return the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray \| list` | Array of shape (nc,) with AP50 values per class, or an empty list if not available. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L929-L935">View on GitHub</a>
```python
@property
def ap50(self) -> np.ndarray | list:
    """Return the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

    Returns:
        (np.ndarray | list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
    """
    return self.all_ap[:, 0] if len(self.all_ap) else []
```
</details>

<br>

### Property `ultralytics.utils.metrics.Metric.ap` {#ultralytics.utils.metrics.Metric.ap}

```python
def ap(self) -> np.ndarray | list
```

Return the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray \| list` | Array of shape (nc,) with AP50-95 values per class, or an empty list if not available. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L938-L944">View on GitHub</a>
```python
@property
def ap(self) -> np.ndarray | list:
    """Return the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

    Returns:
        (np.ndarray | list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
    """
    return self.all_ap.mean(1) if len(self.all_ap) else []
```
</details>

<br>

### Property `ultralytics.utils.metrics.Metric.mp` {#ultralytics.utils.metrics.Metric.mp}

```python
def mp(self) -> float
```

Return the Mean Precision of all classes.

**Returns**

| Type | Description |
| --- | --- |
| `float` | The mean precision of all classes. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L947-L953">View on GitHub</a>
```python
@property
def mp(self) -> float:
    """Return the Mean Precision of all classes.

    Returns:
        (float): The mean precision of all classes.
    """
    return self.p.mean() if len(self.p) else 0.0
```
</details>

<br>

### Property `ultralytics.utils.metrics.Metric.mr` {#ultralytics.utils.metrics.Metric.mr}

```python
def mr(self) -> float
```

Return the Mean Recall of all classes.

**Returns**

| Type | Description |
| --- | --- |
| `float` | The mean recall of all classes. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L956-L962">View on GitHub</a>
```python
@property
def mr(self) -> float:
    """Return the Mean Recall of all classes.

    Returns:
        (float): The mean recall of all classes.
    """
    return self.r.mean() if len(self.r) else 0.0
```
</details>

<br>

### Property `ultralytics.utils.metrics.Metric.map50` {#ultralytics.utils.metrics.Metric.map50}

```python
def map50(self) -> float
```

Return the mean Average Precision (mAP) at an IoU threshold of 0.5.

**Returns**

| Type | Description |
| --- | --- |
| `float` | The mAP at an IoU threshold of 0.5. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L965-L971">View on GitHub</a>
```python
@property
def map50(self) -> float:
    """Return the mean Average Precision (mAP) at an IoU threshold of 0.5.

    Returns:
        (float): The mAP at an IoU threshold of 0.5.
    """
    return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0
```
</details>

<br>

### Property `ultralytics.utils.metrics.Metric.map75` {#ultralytics.utils.metrics.Metric.map75}

```python
def map75(self) -> float
```

Return the mean Average Precision (mAP) at an IoU threshold of 0.75.

**Returns**

| Type | Description |
| --- | --- |
| `float` | The mAP at an IoU threshold of 0.75. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L974-L980">View on GitHub</a>
```python
@property
def map75(self) -> float:
    """Return the mean Average Precision (mAP) at an IoU threshold of 0.75.

    Returns:
        (float): The mAP at an IoU threshold of 0.75.
    """
    return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0
```
</details>

<br>

### Property `ultralytics.utils.metrics.Metric.map` {#ultralytics.utils.metrics.Metric.map}

```python
def map(self) -> float
```

Return the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

**Returns**

| Type | Description |
| --- | --- |
| `float` | The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L983-L989">View on GitHub</a>
```python
@property
def map(self) -> float:
    """Return the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

    Returns:
        (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
    """
    return self.all_ap.mean() if len(self.all_ap) else 0.0
```
</details>

<br>

### Property `ultralytics.utils.metrics.Metric.maps` {#ultralytics.utils.metrics.Metric.maps}

```python
def maps(self) -> np.ndarray
```

Return mAP of each class.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1000-L1005">View on GitHub</a>
```python
@property
def maps(self) -> np.ndarray:
    """Return mAP of each class."""
    maps = np.zeros(self.nc) + self.map
    for i, c in enumerate(self.ap_class_index):
        maps[c] = self.ap[i]
    return maps
```
</details>

<br>

### Property `ultralytics.utils.metrics.Metric.curves` {#ultralytics.utils.metrics.Metric.curves}

```python
def curves(self) -> list
```

Return a list of curves for accessing specific metrics curves.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1046-L1048">View on GitHub</a>
```python
@property
def curves(self) -> list:
    """Return a list of curves for accessing specific metrics curves."""
    return []
```
</details>

<br>

### Property `ultralytics.utils.metrics.Metric.curves_results` {#ultralytics.utils.metrics.Metric.curves\_results}

```python
def curves_results(self) -> list[list]
```

Return a list of curves results for accessing specific metrics curves.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1051-L1058">View on GitHub</a>
```python
@property
def curves_results(self) -> list[list]:
    """Return a list of curves results for accessing specific metrics curves."""
    return [
        [self.px, self.prec_values, "Recall", "Precision"],
        [self.px, self.f1_curve, "Confidence", "F1"],
        [self.px, self.p_curve, "Confidence", "Precision"],
        [self.px, self.r_curve, "Confidence", "Recall"],
    ]
```
</details>

<br>

### Method `ultralytics.utils.metrics.Metric.class_result` {#ultralytics.utils.metrics.Metric.class\_result}

```python
def class_result(self, i: int) -> tuple[float, float, float, float]
```

Return class-aware result, p[i], r[i], ap50[i], ap[i].

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `i` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L995-L997">View on GitHub</a>
```python
def class_result(self, i: int) -> tuple[float, float, float, float]:
    """Return class-aware result, p[i], r[i], ap50[i], ap[i]."""
    return self.p[i], self.r[i], self.ap50[i], self.ap[i]
```
</details>

<br>

### Method `ultralytics.utils.metrics.Metric.clear_image_metrics` {#ultralytics.utils.metrics.Metric.clear\_image\_metrics}

```python
def clear_image_metrics(self) -> None
```

Clear stored per-image metrics from the current validation run.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1041-L1043">View on GitHub</a>
```python
def clear_image_metrics(self) -> None:
    """Clear stored per-image metrics from the current validation run."""
    self.image_metrics.clear()
```
</details>

<br>

### Method `ultralytics.utils.metrics.Metric.fitness` {#ultralytics.utils.metrics.Metric.fitness}

```python
def fitness(self) -> float
```

Return model fitness as a weighted combination of metrics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1007-L1010">View on GitHub</a>
```python
def fitness(self) -> float:
    """Return model fitness as a weighted combination of metrics."""
    w = [0.0, 0.0, 0.0, 1.0]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())
```
</details>

<br>

### Method `ultralytics.utils.metrics.Metric.mean_results` {#ultralytics.utils.metrics.Metric.mean\_results}

```python
def mean_results(self) -> list[float]
```

Return mean of results, mp, mr, map50, map.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L991-L993">View on GitHub</a>
```python
def mean_results(self) -> list[float]:
    """Return mean of results, mp, mr, map50, map."""
    return [self.mp, self.mr, self.map50, self.map]
```
</details>

<br>

### Method `ultralytics.utils.metrics.Metric.update` {#ultralytics.utils.metrics.Metric.update}

```python
def update(self, results: tuple)
```

Update the evaluation metrics with a new set of results.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `results` | `tuple` | A tuple containing evaluation metrics:<br>- p (list): Precision for each class.<br>- r (list): Recall for each class.<br>- f1 (list): F1 score for each class.<br>- all_ap (list): AP scores for all classes and all IoU thresholds.<br>- ap_class_index (list): Index of class for each AP score.<br>- p_curve (list): Precision curve for each class.<br>- r_curve (list): Recall curve for each class.<br>- f1_curve (list): F1 curve for each class.<br>- px (list): X values for the curves.<br>- prec_values (list): Precision values for each class. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1012-L1039">View on GitHub</a>
```python
def update(self, results: tuple):
    """Update the evaluation metrics with a new set of results.

    Args:
        results (tuple): A tuple containing evaluation metrics:
            - p (list): Precision for each class.
            - r (list): Recall for each class.
            - f1 (list): F1 score for each class.
            - all_ap (list): AP scores for all classes and all IoU thresholds.
            - ap_class_index (list): Index of class for each AP score.
            - p_curve (list): Precision curve for each class.
            - r_curve (list): Recall curve for each class.
            - f1_curve (list): F1 curve for each class.
            - px (list): X values for the curves.
            - prec_values (list): Precision values for each class.
    """
    (
        self.p,
        self.r,
        self.f1,
        self.all_ap,
        self.ap_class_index,
        self.p_curve,
        self.r_curve,
        self.f1_curve,
        self.px,
        self.prec_values,
    ) = results
```
</details>

<br>

### Method `ultralytics.utils.metrics.Metric.update_image_metrics` {#ultralytics.utils.metrics.Metric.update\_image\_metrics}

```python
def update_image_metrics(self, tp: np.ndarray, target_cls: np.ndarray, pred_cls: np.ndarray, im_name: str) -> None
```

Update per-image precision, recall, F1, TP, FP, and FN at IoU threshold 0.5.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tp` | `np.ndarray` | True positive array of shape (num_preds, num_iou_thresholds), where the first column (IoU >= 0.5) is used. | *required* |
| `target_cls` | `np.ndarray` | Ground truth class labels for the image. | *required* |
| `pred_cls` | `np.ndarray` | Predicted class labels for the image. | *required* |
| `im_name` | `str` | The image filename used as the per-image key. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1060-L1092">View on GitHub</a>
```python
def update_image_metrics(self, tp: np.ndarray, target_cls: np.ndarray, pred_cls: np.ndarray, im_name: str) -> None:
    """Update per-image precision, recall, F1, TP, FP, and FN at IoU threshold 0.5.

    Args:
        tp (np.ndarray): True positive array of shape (num_preds, num_iou_thresholds), where the first column (IoU
            >= 0.5) is used.
        target_cls (np.ndarray): Ground truth class labels for the image.
        pred_cls (np.ndarray): Predicted class labels for the image.
        im_name (str): The image filename used as the per-image key.
    """
    # Use the default IoU=0.5 column to match the validator's image-level matching policy.
    tp = int(tp[:, 0].sum())
    num_preds = pred_cls.shape[0]
    num_targets = target_cls.shape[0]
    fp = num_preds - tp
    fn = num_targets - tp
    if num_preds == 0 and num_targets == 0:
        # Empty-GT image with no predictions is a trivially correct call, so report a perfect score rather than
        # zeroing out P/R/F1 by the standard 0/0 fallback below.
        precision = recall = f1 = 1.0
    else:
        precision = tp / num_preds if num_preds else 0.0
        recall = tp / num_targets if num_targets else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom else 0.0
    self.image_metrics[im_name] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.metrics.DetMetrics` {#ultralytics.utils.metrics.DetMetrics}

```python
DetMetrics(names: dict[int, str] | None = None)
```

**Bases:** `SimpleClass`, `DataExportMixin`

Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `names` | `dict[int, str], optional` | Dictionary of class names. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `names` | `dict[int, str]` | A dictionary of class names. |
| `box` | `Metric` | An instance of the Metric class for storing detection results. |
| `speed` | `dict[str, float]` | A dictionary for storing execution times of different parts of the detection process. |
| `stats` | `dict[str, list]` | A dictionary containing lists for true positives, confidence scores, predicted classes, target classes, and target images. |
| `nt_per_class` |  | Number of targets per class. |
| `nt_per_image` |  | Number of targets per image. |

**Methods**

| Name | Description |
| --- | --- |
| [`keys`](#ultralytics.utils.metrics.DetMetrics.keys) | Return a list of keys for accessing specific metrics. |
| [`maps`](#ultralytics.utils.metrics.DetMetrics.maps) | Return mean Average Precision (mAP) scores per class. |
| [`fitness`](#ultralytics.utils.metrics.DetMetrics.fitness) | Return the fitness of box object. |
| [`ap_class_index`](#ultralytics.utils.metrics.DetMetrics.ap_class_index) | Return the average precision index per class. |
| [`results_dict`](#ultralytics.utils.metrics.DetMetrics.results_dict) | Return dictionary of computed performance metrics and statistics. |
| [`curves`](#ultralytics.utils.metrics.DetMetrics.curves) | Return a list of curves for accessing specific metrics curves. |
| [`curves_results`](#ultralytics.utils.metrics.DetMetrics.curves_results) | Return a list of computed performance metrics and statistics. |
| [`class_result`](#ultralytics.utils.metrics.DetMetrics.class_result) | Return the result of evaluating the performance of an object detection model on a specific class. |
| [`clear_image_metrics`](#ultralytics.utils.metrics.DetMetrics.clear_image_metrics) | Clear stored per-image metrics. |
| [`clear_stats`](#ultralytics.utils.metrics.DetMetrics.clear_stats) | Clear the stored statistics. |
| [`mean_results`](#ultralytics.utils.metrics.DetMetrics.mean_results) | Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95. |
| [`process`](#ultralytics.utils.metrics.DetMetrics.process) | Process predicted results for object detection and update metrics. |
| [`summary`](#ultralytics.utils.metrics.DetMetrics.summary) | Generate a summarized representation of per-class detection metrics as a list of dictionaries. Includes |
| [`update_stats`](#ultralytics.utils.metrics.DetMetrics.update_stats) | Update statistics by appending new values to existing stat collections. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1095-L1264">View on GitHub</a>
```python
class DetMetrics(SimpleClass, DataExportMixin):
    """Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP).

    Attributes:
        names (dict[int, str]): A dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    Methods:
        update_stats: Update statistics by appending new values to existing stat collections.
        process: Process predicted results for object detection and update metrics.
        clear_stats: Clear the stored statistics.
        keys: Return a list of keys for accessing specific metrics.
        mean_results: Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95.
        class_result: Return the result of evaluating the performance of an object detection model on a specific class.
        maps: Return mean Average Precision (mAP) scores per class.
        fitness: Return the fitness of box object.
        ap_class_index: Return the average precision index per class.
        results_dict: Return dictionary of computed performance metrics and statistics.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Return a list of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class detection metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        """Initialize a DetMetrics instance with class names.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        self.names = names if names is not None else {}
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
        self.nt_per_class = None
        self.nt_per_image = None
```
</details>

<br>

### Property `ultralytics.utils.metrics.DetMetrics.keys` {#ultralytics.utils.metrics.DetMetrics.keys}

```python
def keys(self) -> list[str]
```

Return a list of keys for accessing specific metrics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1188-L1190">View on GitHub</a>
```python
@property
def keys(self) -> list[str]:
    """Return a list of keys for accessing specific metrics."""
    return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
```
</details>

<br>

### Property `ultralytics.utils.metrics.DetMetrics.maps` {#ultralytics.utils.metrics.DetMetrics.maps}

```python
def maps(self) -> np.ndarray
```

Return mean Average Precision (mAP) scores per class.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1201-L1203">View on GitHub</a>
```python
@property
def maps(self) -> np.ndarray:
    """Return mean Average Precision (mAP) scores per class."""
    return self.box.maps
```
</details>

<br>

### Property `ultralytics.utils.metrics.DetMetrics.fitness` {#ultralytics.utils.metrics.DetMetrics.fitness}

```python
def fitness(self) -> float
```

Return the fitness of box object.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1206-L1208">View on GitHub</a>
```python
@property
def fitness(self) -> float:
    """Return the fitness of box object."""
    return self.box.fitness()
```
</details>

<br>

### Property `ultralytics.utils.metrics.DetMetrics.ap_class_index` {#ultralytics.utils.metrics.DetMetrics.ap\_class\_index}

```python
def ap_class_index(self) -> list
```

Return the average precision index per class.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1211-L1213">View on GitHub</a>
```python
@property
def ap_class_index(self) -> list:
    """Return the average precision index per class."""
    return self.box.ap_class_index
```
</details>

<br>

### Property `ultralytics.utils.metrics.DetMetrics.results_dict` {#ultralytics.utils.metrics.DetMetrics.results\_dict}

```python
def results_dict(self) -> dict[str, float]
```

Return dictionary of computed performance metrics and statistics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1216-L1220">View on GitHub</a>
```python
@property
def results_dict(self) -> dict[str, float]:
    """Return dictionary of computed performance metrics and statistics."""
    keys = [*self.keys, "fitness"]
    values = ((float(x) if hasattr(x, "item") else x) for x in ([*self.mean_results(), self.fitness]))
    return dict(zip(keys, values))
```
</details>

<br>

### Property `ultralytics.utils.metrics.DetMetrics.curves` {#ultralytics.utils.metrics.DetMetrics.curves}

```python
def curves(self) -> list[str]
```

Return a list of curves for accessing specific metrics curves.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1223-L1225">View on GitHub</a>
```python
@property
def curves(self) -> list[str]:
    """Return a list of curves for accessing specific metrics curves."""
    return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]
```
</details>

<br>

### Property `ultralytics.utils.metrics.DetMetrics.curves_results` {#ultralytics.utils.metrics.DetMetrics.curves\_results}

```python
def curves_results(self) -> list[list]
```

Return a list of computed performance metrics and statistics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1228-L1230">View on GitHub</a>
```python
@property
def curves_results(self) -> list[list]:
    """Return a list of computed performance metrics and statistics."""
    return self.box.curves_results
```
</details>

<br>

### Method `ultralytics.utils.metrics.DetMetrics.class_result` {#ultralytics.utils.metrics.DetMetrics.class\_result}

```python
def class_result(self, i: int) -> tuple[float, float, float, float]
```

Return the result of evaluating the performance of an object detection model on a specific class.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `i` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1196-L1198">View on GitHub</a>
```python
def class_result(self, i: int) -> tuple[float, float, float, float]:
    """Return the result of evaluating the performance of an object detection model on a specific class."""
    return self.box.class_result(i)
```
</details>

<br>

### Method `ultralytics.utils.metrics.DetMetrics.clear_image_metrics` {#ultralytics.utils.metrics.DetMetrics.clear\_image\_metrics}

```python
def clear_image_metrics(self) -> None
```

Clear stored per-image metrics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1183-L1185">View on GitHub</a>
```python
def clear_image_metrics(self) -> None:
    """Clear stored per-image metrics."""
    self.box.clear_image_metrics()
```
</details>

<br>

### Method `ultralytics.utils.metrics.DetMetrics.clear_stats` {#ultralytics.utils.metrics.DetMetrics.clear\_stats}

```python
def clear_stats(self)
```

Clear the stored statistics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1178-L1181">View on GitHub</a>
```python
def clear_stats(self):
    """Clear the stored statistics."""
    for v in self.stats.values():
        v.clear()
```
</details>

<br>

### Method `ultralytics.utils.metrics.DetMetrics.mean_results` {#ultralytics.utils.metrics.DetMetrics.mean\_results}

```python
def mean_results(self) -> list[float]
```

Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1192-L1194">View on GitHub</a>
```python
def mean_results(self) -> list[float]:
    """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
    return self.box.mean_results()
```
</details>

<br>

### Method `ultralytics.utils.metrics.DetMetrics.process` {#ultralytics.utils.metrics.DetMetrics.process}

```python
def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]
```

Process predicted results for object detection and update metrics.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `save_dir` | `Path` | Directory to save plots. Defaults to Path("."). | `Path(".")` |
| `plot` | `bool` | Whether to plot precision-recall curves. Defaults to False. | `False` |
| `on_plot` | `callable, optional` | Function to call after plots are generated. Defaults to None. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, np.ndarray]` | Dictionary containing concatenated statistics arrays. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1147-L1176">View on GitHub</a>
```python
def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
    """Process predicted results for object detection and update metrics.

    Args:
        save_dir (Path): Directory to save plots. Defaults to Path(".").
        plot (bool): Whether to plot precision-recall curves. Defaults to False.
        on_plot (callable, optional): Function to call after plots are generated. Defaults to None.

    Returns:
        (dict[str, np.ndarray]): Dictionary containing concatenated statistics arrays.
    """
    stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}  # to numpy
    if not stats:
        return stats
    results = ap_per_class(
        stats["tp"],
        stats["conf"],
        stats["pred_cls"],
        stats["target_cls"],
        plot=plot,
        save_dir=save_dir,
        names=self.names,
        on_plot=on_plot,
        prefix="Box",
    )[2:]
    self.box.nc = len(self.names)
    self.box.update(results)
    self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=len(self.names))
    self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=len(self.names))
    return stats
```
</details>

<br>

### Method `ultralytics.utils.metrics.DetMetrics.summary` {#ultralytics.utils.metrics.DetMetrics.summary}

```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]
```

Generate a summarized representation of per-class detection metrics as a list of dictionaries. Includes

shared scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool` | For Detect metrics, everything is normalized by default [0-1]. | `True` |
| `decimals` | `int` | Number of decimal places to round the metrics values to. | `5` |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, Any]]` | A list of dictionaries, each representing one class with corresponding metric values. |

**Examples**

```python
>>> results = model.val(data="coco8.yaml")
>>> detection_summary = results.summary()
>>> print(detection_summary)
```

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1232-L1264">View on GitHub</a>
```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
    """Generate a summarized representation of per-class detection metrics as a list of dictionaries. Includes
    shared scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class.

    Args:
        normalize (bool): For Detect metrics, everything is normalized by default [0-1].
        decimals (int): Number of decimal places to round the metrics values to.

    Returns:
        (list[dict[str, Any]]): A list of dictionaries, each representing one class with corresponding metric
            values.

    Examples:
       >>> results = model.val(data="coco8.yaml")
       >>> detection_summary = results.summary()
       >>> print(detection_summary)
    """
    per_class = {
        "Box-P": self.box.p,
        "Box-R": self.box.r,
        "Box-F1": self.box.f1,
    }
    return [
        {
            "Class": self.names[self.ap_class_index[i]],
            "Images": self.nt_per_image[self.ap_class_index[i]],
            "Instances": self.nt_per_class[self.ap_class_index[i]],
            **{k: round(v[i], decimals) for k, v in per_class.items()},
            "mAP50": round(self.class_result(i)[2], decimals),
            "mAP50-95": round(self.class_result(i)[3], decimals),
        }
        for i in range(len(per_class["Box-P"]))
    ]
```
</details>

<br>

### Method `ultralytics.utils.metrics.DetMetrics.update_stats` {#ultralytics.utils.metrics.DetMetrics.update\_stats}

```python
def update_stats(self, stat: dict[str, Any]) -> None
```

Update statistics by appending new values to existing stat collections.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stat` | `dict[str, Any]` | Dictionary containing new statistical values to append. Keys should match existing keys in self.stats. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1136-L1145">View on GitHub</a>
```python
def update_stats(self, stat: dict[str, Any]) -> None:
    """Update statistics by appending new values to existing stat collections.

    Args:
        stat (dict[str, Any]): Dictionary containing new statistical values to append. Keys should match existing
            keys in self.stats.
    """
    for k in self.stats:
        self.stats[k].append(stat[k])
    self.box.update_image_metrics(stat["tp"], stat["target_cls"], stat["pred_cls"], stat["im_name"])
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.metrics.SegmentMetrics` {#ultralytics.utils.metrics.SegmentMetrics}

```python
SegmentMetrics(names: dict[int, str] | None = None)
```

**Bases:** `DetMetrics`

Calculate and aggregate detection and segmentation metrics over a given set of classes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `names` | `dict[int, str], optional` | Dictionary of class names. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `names` | `dict[int, str]` | Dictionary of class names. |
| `box` | `Metric` | An instance of the Metric class for storing detection results. |
| `seg` | `Metric` | An instance of the Metric class to calculate mask segmentation metrics. |
| `speed` | `dict[str, float]` | A dictionary for storing execution times of different parts of the detection process. |
| `stats` | `dict[str, list]` | A dictionary containing lists for true positives, confidence scores, predicted classes, target classes, and target images. |
| `nt_per_class` |  | Number of targets per class. |
| `nt_per_image` |  | Number of targets per image. |

**Methods**

| Name | Description |
| --- | --- |
| [`keys`](#ultralytics.utils.metrics.SegmentMetrics.keys) | Return a list of keys for accessing metrics. |
| [`maps`](#ultralytics.utils.metrics.SegmentMetrics.maps) | Return mAP scores for object detection and segmentation models. |
| [`fitness`](#ultralytics.utils.metrics.SegmentMetrics.fitness) | Return the fitness score for both segmentation and bounding box models. |
| [`curves`](#ultralytics.utils.metrics.SegmentMetrics.curves) | Return a list of curves for accessing specific metrics curves. |
| [`curves_results`](#ultralytics.utils.metrics.SegmentMetrics.curves_results) | Return a list of computed performance metrics and statistics. |
| [`class_result`](#ultralytics.utils.metrics.SegmentMetrics.class_result) | Return classification results for a specified class index. |
| [`clear_image_metrics`](#ultralytics.utils.metrics.SegmentMetrics.clear_image_metrics) | Clear stored per-image metrics. |
| [`mean_results`](#ultralytics.utils.metrics.SegmentMetrics.mean_results) | Return the mean metrics for bounding box and segmentation results. |
| [`process`](#ultralytics.utils.metrics.SegmentMetrics.process) | Process the detection and segmentation metrics over the given set of predictions. |
| [`summary`](#ultralytics.utils.metrics.SegmentMetrics.summary) | Generate a summarized representation of per-class segmentation metrics as a list of dictionaries. Includes |
| [`update_stats`](#ultralytics.utils.metrics.SegmentMetrics.update_stats) | Update statistics by appending new values to existing stat collections. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1267-L1415">View on GitHub</a>
```python
class SegmentMetrics(DetMetrics):
    """Calculate and aggregate detection and segmentation metrics over a given set of classes.

    Attributes:
        names (dict[int, str]): Dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        seg (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    Methods:
        process: Process the detection and segmentation metrics over the given set of predictions.
        keys: Return a list of keys for accessing metrics.
        mean_results: Return the mean metrics for bounding box and segmentation results.
        class_result: Return classification results for a specified class index.
        maps: Return mAP scores for object detection and segmentation models.
        fitness: Return the fitness score for both segmentation and bounding box models.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Provide a list of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class segmentation metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        """Initialize a SegmentMetrics instance with class names.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        DetMetrics.__init__(self, names)
        self.seg = Metric()
        self.stats["tp_m"] = []  # add additional stats for masks
```
</details>

<br>

### Property `ultralytics.utils.metrics.SegmentMetrics.keys` {#ultralytics.utils.metrics.SegmentMetrics.keys}

```python
def keys(self) -> list[str]
```

Return a list of keys for accessing metrics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1345-L1353">View on GitHub</a>
```python
@property
def keys(self) -> list[str]:
    """Return a list of keys for accessing metrics."""
    return [
        *DetMetrics.keys.fget(self),
        "metrics/precision(M)",
        "metrics/recall(M)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
    ]
```
</details>

<br>

### Property `ultralytics.utils.metrics.SegmentMetrics.maps` {#ultralytics.utils.metrics.SegmentMetrics.maps}

```python
def maps(self) -> np.ndarray
```

Return mAP scores for object detection and segmentation models.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1364-L1366">View on GitHub</a>
```python
@property
def maps(self) -> np.ndarray:
    """Return mAP scores for object detection and segmentation models."""
    return DetMetrics.maps.fget(self) + self.seg.maps
```
</details>

<br>

### Property `ultralytics.utils.metrics.SegmentMetrics.fitness` {#ultralytics.utils.metrics.SegmentMetrics.fitness}

```python
def fitness(self) -> float
```

Return the fitness score for both segmentation and bounding box models.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1369-L1371">View on GitHub</a>
```python
@property
def fitness(self) -> float:
    """Return the fitness score for both segmentation and bounding box models."""
    return self.seg.fitness() + DetMetrics.fitness.fget(self)
```
</details>

<br>

### Property `ultralytics.utils.metrics.SegmentMetrics.curves` {#ultralytics.utils.metrics.SegmentMetrics.curves}

```python
def curves(self) -> list[str]
```

Return a list of curves for accessing specific metrics curves.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1374-L1382">View on GitHub</a>
```python
@property
def curves(self) -> list[str]:
    """Return a list of curves for accessing specific metrics curves."""
    return [
        *DetMetrics.curves.fget(self),
        "Precision-Recall(M)",
        "F1-Confidence(M)",
        "Precision-Confidence(M)",
        "Recall-Confidence(M)",
    ]
```
</details>

<br>

### Property `ultralytics.utils.metrics.SegmentMetrics.curves_results` {#ultralytics.utils.metrics.SegmentMetrics.curves\_results}

```python
def curves_results(self) -> list[list]
```

Return a list of computed performance metrics and statistics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1385-L1387">View on GitHub</a>
```python
@property
def curves_results(self) -> list[list]:
    """Return a list of computed performance metrics and statistics."""
    return DetMetrics.curves_results.fget(self) + self.seg.curves_results
```
</details>

<br>

### Method `ultralytics.utils.metrics.SegmentMetrics.class_result` {#ultralytics.utils.metrics.SegmentMetrics.class\_result}

```python
def class_result(self, i: int) -> list[float]
```

Return classification results for a specified class index.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `i` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1359-L1361">View on GitHub</a>
```python
def class_result(self, i: int) -> list[float]:
    """Return classification results for a specified class index."""
    return DetMetrics.class_result(self, i) + self.seg.class_result(i)
```
</details>

<br>

### Method `ultralytics.utils.metrics.SegmentMetrics.clear_image_metrics` {#ultralytics.utils.metrics.SegmentMetrics.clear\_image\_metrics}

```python
def clear_image_metrics(self) -> None
```

Clear stored per-image metrics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1312-L1315">View on GitHub</a>
```python
def clear_image_metrics(self) -> None:
    """Clear stored per-image metrics."""
    super().clear_image_metrics()
    self.seg.clear_image_metrics()
```
</details>

<br>

### Method `ultralytics.utils.metrics.SegmentMetrics.mean_results` {#ultralytics.utils.metrics.SegmentMetrics.mean\_results}

```python
def mean_results(self) -> list[float]
```

Return the mean metrics for bounding box and segmentation results.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1355-L1357">View on GitHub</a>
```python
def mean_results(self) -> list[float]:
    """Return the mean metrics for bounding box and segmentation results."""
    return DetMetrics.mean_results(self) + self.seg.mean_results()
```
</details>

<br>

### Method `ultralytics.utils.metrics.SegmentMetrics.process` {#ultralytics.utils.metrics.SegmentMetrics.process}

```python
def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]
```

Process the detection and segmentation metrics over the given set of predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `save_dir` | `Path` | Directory to save plots. Defaults to Path("."). | `Path(".")` |
| `plot` | `bool` | Whether to plot precision-recall curves. Defaults to False. | `False` |
| `on_plot` | `callable, optional` | Function to call after plots are generated. Defaults to None. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, np.ndarray]` | Dictionary containing concatenated statistics arrays. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1317-L1342">View on GitHub</a>
```python
def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
    """Process the detection and segmentation metrics over the given set of predictions.

    Args:
        save_dir (Path): Directory to save plots. Defaults to Path(".").
        plot (bool): Whether to plot precision-recall curves. Defaults to False.
        on_plot (callable, optional): Function to call after plots are generated. Defaults to None.

    Returns:
        (dict[str, np.ndarray]): Dictionary containing concatenated statistics arrays.
    """
    stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)  # process box stats
    results_mask = ap_per_class(
        stats["tp_m"],
        stats["conf"],
        stats["pred_cls"],
        stats["target_cls"],
        plot=plot,
        on_plot=on_plot,
        save_dir=save_dir,
        names=self.names,
        prefix="Mask",
    )[2:]
    self.seg.nc = len(self.names)
    self.seg.update(results_mask)
    return stats
```
</details>

<br>

### Method `ultralytics.utils.metrics.SegmentMetrics.summary` {#ultralytics.utils.metrics.SegmentMetrics.summary}

```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]
```

Generate a summarized representation of per-class segmentation metrics as a list of dictionaries. Includes

both box and mask scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool` | For Segment metrics, everything is normalized by default [0-1]. | `True` |
| `decimals` | `int` | Number of decimal places to round the metrics values to. | `5` |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, Any]]` | A list of dictionaries, each representing one class with corresponding metric values. |

**Examples**

```python
>>> results = model.val(data="coco8-seg.yaml")
>>> seg_summary = results.summary(decimals=4)
>>> print(seg_summary)
```

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1389-L1415">View on GitHub</a>
```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
    """Generate a summarized representation of per-class segmentation metrics as a list of dictionaries. Includes
    both box and mask scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for
    each class.

    Args:
        normalize (bool): For Segment metrics, everything is normalized by default [0-1].
        decimals (int): Number of decimal places to round the metrics values to.

    Returns:
        (list[dict[str, Any]]): A list of dictionaries, each representing one class with corresponding metric
            values.

    Examples:
        >>> results = model.val(data="coco8-seg.yaml")
        >>> seg_summary = results.summary(decimals=4)
        >>> print(seg_summary)
    """
    per_class = {
        "Mask-P": self.seg.p,
        "Mask-R": self.seg.r,
        "Mask-F1": self.seg.f1,
    }
    summary = DetMetrics.summary(self, normalize, decimals)  # get box summary
    for i, s in enumerate(summary):
        s.update({**{k: round(v[i], decimals) for k, v in per_class.items()}})
    return summary
```
</details>

<br>

### Method `ultralytics.utils.metrics.SegmentMetrics.update_stats` {#ultralytics.utils.metrics.SegmentMetrics.update\_stats}

```python
def update_stats(self, stat: dict[str, Any]) -> None
```

Update statistics by appending new values to existing stat collections.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stat` | `dict[str, Any]` | Dictionary containing new statistical values to append. Keys should match existing keys in self.stats. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1302-L1310">View on GitHub</a>
```python
def update_stats(self, stat: dict[str, Any]) -> None:
    """Update statistics by appending new values to existing stat collections.

    Args:
        stat (dict[str, Any]): Dictionary containing new statistical values to append. Keys should match existing
            keys in self.stats.
    """
    super().update_stats(stat)  # update box stats
    self.seg.update_image_metrics(stat["tp_m"], stat["target_cls"], stat["pred_cls"], stat["im_name"])
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.metrics.PoseMetrics` {#ultralytics.utils.metrics.PoseMetrics}

```python
PoseMetrics(names: dict[int, str] | None = None)
```

**Bases:** `DetMetrics`

Calculate and aggregate detection and pose metrics over a given set of classes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `names` | `dict[int, str], optional` | Dictionary of class names. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `names` | `dict[int, str]` | Dictionary of class names. |
| `pose` | `Metric` | An instance of the Metric class to calculate pose metrics. |
| `box` | `Metric` | An instance of the Metric class for storing detection results. |
| `speed` | `dict[str, float]` | A dictionary for storing execution times of different parts of the detection process. |
| `stats` | `dict[str, list]` | A dictionary containing lists for true positives, confidence scores, predicted classes, target classes, and target images. |
| `nt_per_class` |  | Number of targets per class. |
| `nt_per_image` |  | Number of targets per image. |

**Methods**

| Name | Description |
| --- | --- |
| [`keys`](#ultralytics.utils.metrics.PoseMetrics.keys) | Return a list of evaluation metric keys. |
| [`maps`](#ultralytics.utils.metrics.PoseMetrics.maps) | Return the mean average precision (mAP) per class for both box and pose detections. |
| [`fitness`](#ultralytics.utils.metrics.PoseMetrics.fitness) | Return combined fitness score for pose and box detection. |
| [`curves`](#ultralytics.utils.metrics.PoseMetrics.curves) | Return a list of curves for accessing specific metrics curves. |
| [`curves_results`](#ultralytics.utils.metrics.PoseMetrics.curves_results) | Return a list of computed performance metrics and statistics. |
| [`class_result`](#ultralytics.utils.metrics.PoseMetrics.class_result) | Return the class-wise detection results for a specific class i. |
| [`clear_image_metrics`](#ultralytics.utils.metrics.PoseMetrics.clear_image_metrics) | Clear stored per-image metrics. |
| [`mean_results`](#ultralytics.utils.metrics.PoseMetrics.mean_results) | Return the mean results of box and pose. |
| [`process`](#ultralytics.utils.metrics.PoseMetrics.process) | Process the detection and pose metrics over the given set of predictions. |
| [`summary`](#ultralytics.utils.metrics.PoseMetrics.summary) | Generate a summarized representation of per-class pose metrics as a list of dictionaries. Includes both box |
| [`update_stats`](#ultralytics.utils.metrics.PoseMetrics.update_stats) | Update statistics by appending new values to existing stat collections. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1418-L1565">View on GitHub</a>
```python
class PoseMetrics(DetMetrics):
    """Calculate and aggregate detection and pose metrics over a given set of classes.

    Attributes:
        names (dict[int, str]): Dictionary of class names.
        pose (Metric): An instance of the Metric class to calculate pose metrics.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    Methods:
        process: Process the detection and pose metrics over the given set of predictions.
        keys: Return a list of keys for accessing metrics.
        mean_results: Return the mean results of box and pose.
        class_result: Return the class-wise detection results for a specific class i.
        maps: Return the mean average precision (mAP) per class for both box and pose detections.
        fitness: Return combined fitness score for pose and box detection.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Provide a list of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class pose metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        """Initialize the PoseMetrics class with class names.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        super().__init__(names)
        self.pose = Metric()
        self.stats["tp_p"] = []  # add additional stats for pose
```
</details>

<br>

### Property `ultralytics.utils.metrics.PoseMetrics.keys` {#ultralytics.utils.metrics.PoseMetrics.keys}

```python
def keys(self) -> list[str]
```

Return a list of evaluation metric keys.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1496-L1504">View on GitHub</a>
```python
@property
def keys(self) -> list[str]:
    """Return a list of evaluation metric keys."""
    return [
        *DetMetrics.keys.fget(self),
        "metrics/precision(P)",
        "metrics/recall(P)",
        "metrics/mAP50(P)",
        "metrics/mAP50-95(P)",
    ]
```
</details>

<br>

### Property `ultralytics.utils.metrics.PoseMetrics.maps` {#ultralytics.utils.metrics.PoseMetrics.maps}

```python
def maps(self) -> np.ndarray
```

Return the mean average precision (mAP) per class for both box and pose detections.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1515-L1517">View on GitHub</a>
```python
@property
def maps(self) -> np.ndarray:
    """Return the mean average precision (mAP) per class for both box and pose detections."""
    return DetMetrics.maps.fget(self) + self.pose.maps
```
</details>

<br>

### Property `ultralytics.utils.metrics.PoseMetrics.fitness` {#ultralytics.utils.metrics.PoseMetrics.fitness}

```python
def fitness(self) -> float
```

Return combined fitness score for pose and box detection.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1520-L1522">View on GitHub</a>
```python
@property
def fitness(self) -> float:
    """Return combined fitness score for pose and box detection."""
    return self.pose.fitness() + DetMetrics.fitness.fget(self)
```
</details>

<br>

### Property `ultralytics.utils.metrics.PoseMetrics.curves` {#ultralytics.utils.metrics.PoseMetrics.curves}

```python
def curves(self) -> list[str]
```

Return a list of curves for accessing specific metrics curves.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1525-L1533">View on GitHub</a>
```python
@property
def curves(self) -> list[str]:
    """Return a list of curves for accessing specific metrics curves."""
    return [
        *DetMetrics.curves.fget(self),
        "Precision-Recall(P)",
        "F1-Confidence(P)",
        "Precision-Confidence(P)",
        "Recall-Confidence(P)",
    ]
```
</details>

<br>

### Property `ultralytics.utils.metrics.PoseMetrics.curves_results` {#ultralytics.utils.metrics.PoseMetrics.curves\_results}

```python
def curves_results(self) -> list[list]
```

Return a list of computed performance metrics and statistics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1536-L1538">View on GitHub</a>
```python
@property
def curves_results(self) -> list[list]:
    """Return a list of computed performance metrics and statistics."""
    return DetMetrics.curves_results.fget(self) + self.pose.curves_results
```
</details>

<br>

### Method `ultralytics.utils.metrics.PoseMetrics.class_result` {#ultralytics.utils.metrics.PoseMetrics.class\_result}

```python
def class_result(self, i: int) -> list[float]
```

Return the class-wise detection results for a specific class i.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `i` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1510-L1512">View on GitHub</a>
```python
def class_result(self, i: int) -> list[float]:
    """Return the class-wise detection results for a specific class i."""
    return DetMetrics.class_result(self, i) + self.pose.class_result(i)
```
</details>

<br>

### Method `ultralytics.utils.metrics.PoseMetrics.clear_image_metrics` {#ultralytics.utils.metrics.PoseMetrics.clear\_image\_metrics}

```python
def clear_image_metrics(self) -> None
```

Clear stored per-image metrics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1463-L1466">View on GitHub</a>
```python
def clear_image_metrics(self) -> None:
    """Clear stored per-image metrics."""
    super().clear_image_metrics()
    self.pose.clear_image_metrics()
```
</details>

<br>

### Method `ultralytics.utils.metrics.PoseMetrics.mean_results` {#ultralytics.utils.metrics.PoseMetrics.mean\_results}

```python
def mean_results(self) -> list[float]
```

Return the mean results of box and pose.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1506-L1508">View on GitHub</a>
```python
def mean_results(self) -> list[float]:
    """Return the mean results of box and pose."""
    return DetMetrics.mean_results(self) + self.pose.mean_results()
```
</details>

<br>

### Method `ultralytics.utils.metrics.PoseMetrics.process` {#ultralytics.utils.metrics.PoseMetrics.process}

```python
def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]
```

Process the detection and pose metrics over the given set of predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `save_dir` | `Path` | Directory to save plots. Defaults to Path("."). | `Path(".")` |
| `plot` | `bool` | Whether to plot precision-recall curves. Defaults to False. | `False` |
| `on_plot` | `callable, optional` | Function to call after plots are generated. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, np.ndarray]` | Dictionary containing concatenated statistics arrays. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1468-L1493">View on GitHub</a>
```python
def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
    """Process the detection and pose metrics over the given set of predictions.

    Args:
        save_dir (Path): Directory to save plots. Defaults to Path(".").
        plot (bool): Whether to plot precision-recall curves. Defaults to False.
        on_plot (callable, optional): Function to call after plots are generated.

    Returns:
        (dict[str, np.ndarray]): Dictionary containing concatenated statistics arrays.
    """
    stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)  # process box stats
    results_pose = ap_per_class(
        stats["tp_p"],
        stats["conf"],
        stats["pred_cls"],
        stats["target_cls"],
        plot=plot,
        on_plot=on_plot,
        save_dir=save_dir,
        names=self.names,
        prefix="Pose",
    )[2:]
    self.pose.nc = len(self.names)
    self.pose.update(results_pose)
    return stats
```
</details>

<br>

### Method `ultralytics.utils.metrics.PoseMetrics.summary` {#ultralytics.utils.metrics.PoseMetrics.summary}

```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]
```

Generate a summarized representation of per-class pose metrics as a list of dictionaries. Includes both box

and pose scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool` | For Pose metrics, everything is normalized by default [0-1]. | `True` |
| `decimals` | `int` | Number of decimal places to round the metrics values to. | `5` |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, Any]]` | A list of dictionaries, each representing one class with corresponding metric values. |

**Examples**

```python
>>> results = model.val(data="coco8-pose.yaml")
>>> pose_summary = results.summary(decimals=4)
>>> print(pose_summary)
```

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1540-L1565">View on GitHub</a>
```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
    """Generate a summarized representation of per-class pose metrics as a list of dictionaries. Includes both box
    and pose scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class.

    Args:
        normalize (bool): For Pose metrics, everything is normalized by default [0-1].
        decimals (int): Number of decimal places to round the metrics values to.

    Returns:
        (list[dict[str, Any]]): A list of dictionaries, each representing one class with corresponding metric
            values.

    Examples:
        >>> results = model.val(data="coco8-pose.yaml")
        >>> pose_summary = results.summary(decimals=4)
        >>> print(pose_summary)
    """
    per_class = {
        "Pose-P": self.pose.p,
        "Pose-R": self.pose.r,
        "Pose-F1": self.pose.f1,
    }
    summary = DetMetrics.summary(self, normalize, decimals)  # get box summary
    for i, s in enumerate(summary):
        s.update({**{k: round(v[i], decimals) for k, v in per_class.items()}})
    return summary
```
</details>

<br>

### Method `ultralytics.utils.metrics.PoseMetrics.update_stats` {#ultralytics.utils.metrics.PoseMetrics.update\_stats}

```python
def update_stats(self, stat: dict[str, Any]) -> None
```

Update statistics by appending new values to existing stat collections.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `stat` | `dict[str, Any]` | Dictionary containing new statistical values to append. Keys should match existing keys in self.stats. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1453-L1461">View on GitHub</a>
```python
def update_stats(self, stat: dict[str, Any]) -> None:
    """Update statistics by appending new values to existing stat collections.

    Args:
        stat (dict[str, Any]): Dictionary containing new statistical values to append. Keys should match existing
            keys in self.stats.
    """
    super().update_stats(stat)  # update box stats
    self.pose.update_image_metrics(stat["tp_p"], stat["target_cls"], stat["pred_cls"], stat["im_name"])
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.metrics.ClassifyMetrics` {#ultralytics.utils.metrics.ClassifyMetrics}

```python
ClassifyMetrics()
```

**Bases:** `SimpleClass`, `DataExportMixin`

Class for computing classification metrics including top-1 and top-5 accuracy.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `top1` | `float` | The top-1 accuracy. |
| `top5` | `float` | The top-5 accuracy. |
| `speed` | `dict[str, float]` | A dictionary containing the time taken for each step in the pipeline. |

**Methods**

| Name | Description |
| --- | --- |
| [`fitness`](#ultralytics.utils.metrics.ClassifyMetrics.fitness) | Return mean of top-1 and top-5 accuracies as fitness score. |
| [`results_dict`](#ultralytics.utils.metrics.ClassifyMetrics.results_dict) | Return a dictionary with model's performance metrics and fitness score. |
| [`keys`](#ultralytics.utils.metrics.ClassifyMetrics.keys) | Return a list of keys for the results_dict property. |
| [`curves`](#ultralytics.utils.metrics.ClassifyMetrics.curves) | Return a list of curves for accessing specific metrics curves. |
| [`curves_results`](#ultralytics.utils.metrics.ClassifyMetrics.curves_results) | Return a list of curves results for accessing specific metrics curves. |
| [`process`](#ultralytics.utils.metrics.ClassifyMetrics.process) | Process target classes and predicted classes to compute metrics. |
| [`summary`](#ultralytics.utils.metrics.ClassifyMetrics.summary) | Generate a single-row summary of classification metrics (Top-1 and Top-5 accuracy). |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1568-L1644">View on GitHub</a>
```python
class ClassifyMetrics(SimpleClass, DataExportMixin):
    """Class for computing classification metrics including top-1 and top-5 accuracy.

    Attributes:
        top1 (float): The top-1 accuracy.
        top5 (float): The top-5 accuracy.
        speed (dict[str, float]): A dictionary containing the time taken for each step in the pipeline.

    Methods:
        process: Process target classes and predicted classes to compute metrics.
        fitness: Return mean of top-1 and top-5 accuracies as fitness score.
        results_dict: Return a dictionary with model's performance metrics and fitness score.
        keys: Return a list of keys for the results_dict property.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Provide a list of computed performance metrics and statistics.
        summary: Generate a single-row summary of classification metrics (Top-1 and Top-5 accuracy).
    """

    def __init__(self) -> None:
        """Initialize a ClassifyMetrics instance."""
        self.top1 = 0
        self.top5 = 0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
```
</details>

<br>

### Property `ultralytics.utils.metrics.ClassifyMetrics.fitness` {#ultralytics.utils.metrics.ClassifyMetrics.fitness}

```python
def fitness(self) -> float
```

Return mean of top-1 and top-5 accuracies as fitness score.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1605-L1607">View on GitHub</a>
```python
@property
def fitness(self) -> float:
    """Return mean of top-1 and top-5 accuracies as fitness score."""
    return (self.top1 + self.top5) / 2
```
</details>

<br>

### Property `ultralytics.utils.metrics.ClassifyMetrics.results_dict` {#ultralytics.utils.metrics.ClassifyMetrics.results\_dict}

```python
def results_dict(self) -> dict[str, float]
```

Return a dictionary with model's performance metrics and fitness score.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1610-L1612">View on GitHub</a>
```python
@property
def results_dict(self) -> dict[str, float]:
    """Return a dictionary with model's performance metrics and fitness score."""
    return dict(zip([*self.keys, "fitness"], [self.top1, self.top5, self.fitness]))
```
</details>

<br>

### Property `ultralytics.utils.metrics.ClassifyMetrics.keys` {#ultralytics.utils.metrics.ClassifyMetrics.keys}

```python
def keys(self) -> list[str]
```

Return a list of keys for the results_dict property.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1615-L1617">View on GitHub</a>
```python
@property
def keys(self) -> list[str]:
    """Return a list of keys for the results_dict property."""
    return ["metrics/accuracy_top1", "metrics/accuracy_top5"]
```
</details>

<br>

### Property `ultralytics.utils.metrics.ClassifyMetrics.curves` {#ultralytics.utils.metrics.ClassifyMetrics.curves}

```python
def curves(self) -> list
```

Return a list of curves for accessing specific metrics curves.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1620-L1622">View on GitHub</a>
```python
@property
def curves(self) -> list:
    """Return a list of curves for accessing specific metrics curves."""
    return []
```
</details>

<br>

### Property `ultralytics.utils.metrics.ClassifyMetrics.curves_results` {#ultralytics.utils.metrics.ClassifyMetrics.curves\_results}

```python
def curves_results(self) -> list
```

Return a list of curves results for accessing specific metrics curves.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1625-L1627">View on GitHub</a>
```python
@property
def curves_results(self) -> list:
    """Return a list of curves results for accessing specific metrics curves."""
    return []
```
</details>

<br>

### Method `ultralytics.utils.metrics.ClassifyMetrics.process` {#ultralytics.utils.metrics.ClassifyMetrics.process}

```python
def process(self, targets: torch.Tensor, pred: torch.Tensor)
```

Process target classes and predicted classes to compute metrics.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `targets` | `torch.Tensor` | Target classes. | *required* |
| `pred` | `torch.Tensor` | Predicted classes. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1592-L1602">View on GitHub</a>
```python
def process(self, targets: torch.Tensor, pred: torch.Tensor):
    """Process target classes and predicted classes to compute metrics.

    Args:
        targets (torch.Tensor): Target classes.
        pred (torch.Tensor): Predicted classes.
    """
    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    self.top1, self.top5 = acc.mean(0).tolist()
```
</details>

<br>

### Method `ultralytics.utils.metrics.ClassifyMetrics.summary` {#ultralytics.utils.metrics.ClassifyMetrics.summary}

```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, float]]
```

Generate a single-row summary of classification metrics (Top-1 and Top-5 accuracy).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool` | For Classify metrics, everything is normalized by default [0-1]. | `True` |
| `decimals` | `int` | Number of decimal places to round the metrics values to. | `5` |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict[str, float]]` | A list with one dictionary containing Top-1 and Top-5 classification accuracy. |

**Examples**

```python
>>> results = model.val(data="imagenet10")
>>> classify_summary = results.summary(decimals=4)
>>> print(classify_summary)
```

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1629-L1644">View on GitHub</a>
```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, float]]:
    """Generate a single-row summary of classification metrics (Top-1 and Top-5 accuracy).

    Args:
        normalize (bool): For Classify metrics, everything is normalized by default [0-1].
        decimals (int): Number of decimal places to round the metrics values to.

    Returns:
        (list[dict[str, float]]): A list with one dictionary containing Top-1 and Top-5 classification accuracy.

    Examples:
        >>> results = model.val(data="imagenet10")
        >>> classify_summary = results.summary(decimals=4)
        >>> print(classify_summary)
    """
    return [{"top1_acc": round(self.top1, decimals), "top5_acc": round(self.top5, decimals)}]
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.metrics.OBBMetrics` {#ultralytics.utils.metrics.OBBMetrics}

```python
OBBMetrics(names: dict[int, str] | None = None)
```

**Bases:** `DetMetrics`

Metrics for evaluating oriented bounding box (OBB) detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `names` | `dict[int, str], optional` | Dictionary of class names. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `names` | `dict[int, str]` | Dictionary of class names. |
| `box` | `Metric` | An instance of the Metric class for storing detection results. |
| `speed` | `dict[str, float]` | A dictionary for storing execution times of different parts of the detection process. |
| `stats` | `dict[str, list]` | A dictionary containing lists for true positives, confidence scores, predicted classes, target classes, and target images. |
| `nt_per_class` |  | Number of targets per class. |
| `nt_per_image` |  | Number of targets per image. |

**References**

- https://arxiv.org/pdf/2106.06072.pdf

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1647-L1669">View on GitHub</a>
```python
class OBBMetrics(DetMetrics):
    """Metrics for evaluating oriented bounding box (OBB) detection.

    Attributes:
        names (dict[int, str]): Dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes,
            target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    References:
        https://arxiv.org/pdf/2106.06072.pdf
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        """Initialize an OBBMetrics instance with class names.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        DetMetrics.__init__(self, names)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.metrics.SemanticMetrics` {#ultralytics.utils.metrics.SemanticMetrics}

```python
SemanticMetrics(names: dict[int, str] | None = None)
```

**Bases:** `SimpleClass`, `DataExportMixin`

Metrics for semantic segmentation, including mIoU, pixel accuracy, and per-class IoU.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `names` | `dict, optional` | Dictionary mapping class indices to names. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `names` | `dict` | Class names mapping. |
| `nc` | `int` | Number of classes. |
| `cm_nc` | `int` | Confusion matrix side length (2 for binary segmentation, else nc). |
| `device` | `torch.device \| None` | Device used for confusion matrix accumulation. |
| `matrix` | `torch.Tensor \| None` | Accumulated confusion matrix of shape (cm_nc, cm_nc). |
| `speed` | `dict` | Processing speed statistics. |
| `nt_per_image` | `np.ndarray` | Number of images containing each class. |
| `nt_per_class` | `np.ndarray` | Number of pixels per class. |
| `_miou` | `float` | Cached mean IoU. |
| `_pixel_accuracy` | `float` | Cached pixel accuracy. |
| `_per_class_iou` | `np.ndarray` | Cached per-class IoU values. |
| `_per_class_pixel_acc` | `np.ndarray` | Cached per-class pixel accuracy. |

**Methods**

| Name | Description |
| --- | --- |
| [`miou`](#ultralytics.utils.metrics.SemanticMetrics.miou) | Return mean IoU (foreground IoU only for binary segmentation). |
| [`pixel_accuracy`](#ultralytics.utils.metrics.SemanticMetrics.pixel_accuracy) | Return overall pixel accuracy. |
| [`per_class_iou`](#ultralytics.utils.metrics.SemanticMetrics.per_class_iou) | Return per-class IoU values (foreground IoU only for binary segmentation). |
| [`per_class_pixel_accuracy`](#ultralytics.utils.metrics.SemanticMetrics.per_class_pixel_accuracy) | Return per-class pixel accuracy (diagonal / row sum for each class). |
| [`fitness`](#ultralytics.utils.metrics.SemanticMetrics.fitness) | Return model fitness as mean IoU. |
| [`keys`](#ultralytics.utils.metrics.SemanticMetrics.keys) | Return metric keys for logging. |
| [`ap_class_index`](#ultralytics.utils.metrics.SemanticMetrics.ap_class_index) | Return the indices of classes present in the ground truth for per-class reporting. |
| [`results_dict`](#ultralytics.utils.metrics.SemanticMetrics.results_dict) | Return results dictionary. |
| [`curves`](#ultralytics.utils.metrics.SemanticMetrics.curves) | Return an empty list because semantic segmentation has no PR curves. |
| [`curves_results`](#ultralytics.utils.metrics.SemanticMetrics.curves_results) | Return empty list (no PR curve results). |
| [`_plot_iou_bars`](#ultralytics.utils.metrics.SemanticMetrics._plot_iou_bars) | Plot per-class IoU bar chart. |
| [`class_result`](#ultralytics.utils.metrics.SemanticMetrics.class_result) | Return the result of evaluating the performance on a specific class. |
| [`clear_stats`](#ultralytics.utils.metrics.SemanticMetrics.clear_stats) | Clear accumulated statistics. |
| [`mean_results`](#ultralytics.utils.metrics.SemanticMetrics.mean_results) | Return mean results for logging. |
| [`process`](#ultralytics.utils.metrics.SemanticMetrics.process) | Compute final metrics from accumulated confusion matrix. |
| [`summary`](#ultralytics.utils.metrics.SemanticMetrics.summary) | Generate a per-class summary of semantic segmentation metrics, with global mIoU and pixel accuracy on each |
| [`update_stats`](#ultralytics.utils.metrics.SemanticMetrics.update_stats) | Accumulate confusion matrix from predictions and targets. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1672-L1889">View on GitHub</a>
```python
class SemanticMetrics(SimpleClass, DataExportMixin):
    """Metrics for semantic segmentation, including mIoU, pixel accuracy, and per-class IoU.

    Attributes:
        names (dict): Class names mapping.
        nc (int): Number of classes.
        cm_nc (int): Confusion matrix side length (2 for binary segmentation, else nc).
        device (torch.device | None): Device used for confusion matrix accumulation.
        matrix (torch.Tensor | None): Accumulated confusion matrix of shape (cm_nc, cm_nc).
        speed (dict): Processing speed statistics.
        nt_per_image (np.ndarray): Number of images containing each class.
        nt_per_class (np.ndarray): Number of pixels per class.
        _miou (float): Cached mean IoU.
        _pixel_accuracy (float): Cached pixel accuracy.
        _per_class_iou (np.ndarray): Cached per-class IoU values.
        _per_class_pixel_acc (np.ndarray): Cached per-class pixel accuracy.
    """

    def __init__(self, names: dict[int, str] | None = None) -> None:
        """Initialize semantic segmentation metrics.

        Args:
            names (dict, optional): Dictionary mapping class indices to names.
        """
        self.names = names or {}
        self.nc = len(self.names)
        self.cm_nc = 2 if self.nc == 1 else self.nc
        self.matrix = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.nt_per_image = np.zeros(self.nc, dtype=np.int32)
        self._miou = 0.0
        self._pixel_accuracy = 0.0
        self._per_class_iou = np.zeros(self.nc, dtype=np.float32)
        self._per_class_pixel_acc = np.zeros(self.nc, dtype=np.float32)
        self.nt_per_class = np.zeros(self.nc, dtype=np.int32)
```
</details>

<br>

### Property `ultralytics.utils.metrics.SemanticMetrics.miou` {#ultralytics.utils.metrics.SemanticMetrics.miou}

```python
def miou(self)
```

Return mean IoU (foreground IoU only for binary segmentation).

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1804-L1806">View on GitHub</a>
```python
@property
def miou(self):
    """Return mean IoU (foreground IoU only for binary segmentation)."""
    return self._miou
```
</details>

<br>

### Property `ultralytics.utils.metrics.SemanticMetrics.pixel_accuracy` {#ultralytics.utils.metrics.SemanticMetrics.pixel\_accuracy}

```python
def pixel_accuracy(self)
```

Return overall pixel accuracy.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1809-L1811">View on GitHub</a>
```python
@property
def pixel_accuracy(self):
    """Return overall pixel accuracy."""
    return self._pixel_accuracy
```
</details>

<br>

### Property `ultralytics.utils.metrics.SemanticMetrics.per_class_iou` {#ultralytics.utils.metrics.SemanticMetrics.per\_class\_iou}

```python
def per_class_iou(self)
```

Return per-class IoU values (foreground IoU only for binary segmentation).

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1814-L1816">View on GitHub</a>
```python
@property
def per_class_iou(self):
    """Return per-class IoU values (foreground IoU only for binary segmentation)."""
    return self._per_class_iou
```
</details>

<br>

### Property `ultralytics.utils.metrics.SemanticMetrics.per_class_pixel_accuracy` {#ultralytics.utils.metrics.SemanticMetrics.per\_class\_pixel\_accuracy}

```python
def per_class_pixel_accuracy(self)
```

Return per-class pixel accuracy (diagonal / row sum for each class).

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1819-L1821">View on GitHub</a>
```python
@property
def per_class_pixel_accuracy(self):
    """Return per-class pixel accuracy (diagonal / row sum for each class)."""
    return self._per_class_pixel_acc
```
</details>

<br>

### Property `ultralytics.utils.metrics.SemanticMetrics.fitness` {#ultralytics.utils.metrics.SemanticMetrics.fitness}

```python
def fitness(self)
```

Return model fitness as mean IoU.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1824-L1826">View on GitHub</a>
```python
@property
def fitness(self):
    """Return model fitness as mean IoU."""
    return self.miou
```
</details>

<br>

### Property `ultralytics.utils.metrics.SemanticMetrics.keys` {#ultralytics.utils.metrics.SemanticMetrics.keys}

```python
def keys(self)
```

Return metric keys for logging.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1829-L1831">View on GitHub</a>
```python
@property
def keys(self):
    """Return metric keys for logging."""
    return ["metrics/mIoU", "metrics/pixel_acc"]
```
</details>

<br>

### Property `ultralytics.utils.metrics.SemanticMetrics.ap_class_index` {#ultralytics.utils.metrics.SemanticMetrics.ap\_class\_index}

```python
def ap_class_index(self)
```

Return the indices of classes present in the ground truth for per-class reporting.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1845-L1847">View on GitHub</a>
```python
@property
def ap_class_index(self):
    """Return the indices of classes present in the ground truth for per-class reporting."""
    return [i for i in range(self.nc) if self.nt_per_class[i] > 0]
```
</details>

<br>

### Property `ultralytics.utils.metrics.SemanticMetrics.results_dict` {#ultralytics.utils.metrics.SemanticMetrics.results\_dict}

```python
def results_dict(self)
```

Return results dictionary.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1850-L1852">View on GitHub</a>
```python
@property
def results_dict(self):
    """Return results dictionary."""
    return dict(zip([*self.keys, "fitness"], [*self.mean_results(), self.fitness]))
```
</details>

<br>

### Property `ultralytics.utils.metrics.SemanticMetrics.curves` {#ultralytics.utils.metrics.SemanticMetrics.curves}

```python
def curves(self)
```

Return an empty list because semantic segmentation has no PR curves.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1855-L1857">View on GitHub</a>
```python
@property
def curves(self):
    """Return an empty list because semantic segmentation has no PR curves."""
    return []
```
</details>

<br>

### Property `ultralytics.utils.metrics.SemanticMetrics.curves_results` {#ultralytics.utils.metrics.SemanticMetrics.curves\_results}

```python
def curves_results(self)
```

Return empty list (no PR curve results).

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1860-L1862">View on GitHub</a>
```python
@property
def curves_results(self):
    """Return empty list (no PR curve results)."""
    return []
```
</details>

<br>

### Method `ultralytics.utils.metrics.SemanticMetrics._plot_iou_bars` {#ultralytics.utils.metrics.SemanticMetrics.\_plot\_iou\_bars}

```python
def _plot_iou_bars(self, save_dir, on_plot)
```

Plot per-class IoU bar chart.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `save_dir` | `Path \| str` | Directory to save the plot. | *required* |
| `on_plot` | `callable, optional` | Function to call after plot is saved. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1774-L1801">View on GitHub</a>
```python
@plt_settings()
def _plot_iou_bars(self, save_dir, on_plot):
    """Plot per-class IoU bar chart.

    Args:
        save_dir (Path | str): Directory to save the plot.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)
    names = list(self.names.values()) if self.names else [str(i) for i in range(self.nc)]
    x = np.arange(self.nc)
    bars = ax.bar(x, self._per_class_iou, color=[[c / 255.0 for c in colors(i, False)] for i in range(self.nc)])
    ax.set_xlabel("Class")
    ax.set_ylabel("IoU")
    ax.set_title("Per-Class IoU")
    ax.set_ylim(0, 1)
    if 0 < len(names) < 30:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    fname = Path(save_dir) / "iou_bar_chart.png"
    plt.savefig(fname, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(fname)
```
</details>

<br>

### Method `ultralytics.utils.metrics.SemanticMetrics.class_result` {#ultralytics.utils.metrics.SemanticMetrics.class\_result}

```python
def class_result(self, i: int) -> list[float]
```

Return the result of evaluating the performance on a specific class.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `i` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1837-L1842">View on GitHub</a>
```python
def class_result(self, i: int) -> list[float]:
    """Return the result of evaluating the performance on a specific class."""
    if self._per_class_iou is None or len(self._per_class_iou) == 0:
        return [0.0, 0.0]
    c = self.ap_class_index[i]
    return [float(self._per_class_iou[c]), float(self._per_class_pixel_acc[c])]
```
</details>

<br>

### Method `ultralytics.utils.metrics.SemanticMetrics.clear_stats` {#ultralytics.utils.metrics.SemanticMetrics.clear\_stats}

```python
def clear_stats(self)
```

Clear accumulated statistics.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1768-L1771">View on GitHub</a>
```python
def clear_stats(self):
    """Clear accumulated statistics."""
    self.matrix = None
    self.nt_per_image.fill(0)
```
</details>

<br>

### Method `ultralytics.utils.metrics.SemanticMetrics.mean_results` {#ultralytics.utils.metrics.SemanticMetrics.mean\_results}

```python
def mean_results(self)
```

Return mean results for logging.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1833-L1835">View on GitHub</a>
```python
def mean_results(self):
    """Return mean results for logging."""
    return [self.miou, self.pixel_accuracy]
```
</details>

<br>

### Method `ultralytics.utils.metrics.SemanticMetrics.process` {#ultralytics.utils.metrics.SemanticMetrics.process}

```python
def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot: callable | None = None) -> None
```

Compute final metrics from accumulated confusion matrix.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `save_dir` | `Path` | Directory to save plots. Defaults to Path('.'). | `Path(".")` |
| `plot` | `bool` | Whether to plot IoU bars and confusion matrix. Defaults to False. | `False` |
| `on_plot` | `callable, optional` | Function to call after plots are generated. Defaults to None. | `None` |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1732-L1766">View on GitHub</a>
```python
def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot: callable | None = None) -> None:
    """Compute final metrics from accumulated confusion matrix.

    Args:
        save_dir (Path): Directory to save plots. Defaults to Path('.').
        plot (bool): Whether to plot IoU bars and confusion matrix. Defaults to False.
        on_plot (callable, optional): Function to call after plots are generated. Defaults to None.
    """
    if self.matrix is None:
        return

    intersection = torch.diagonal(self.matrix)
    union = self.matrix.sum(1) + self.matrix.sum(0) - intersection
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(intersection, dtype=torch.float32))
    row_sum = self.matrix.sum(1)
    pa = intersection / (row_sum + 1e-10)

    if self.nc == 1:
        self._miou = float(iou[1].item())
        self._per_class_iou = iou[1:].cpu().numpy()
        self._per_class_pixel_acc = pa[1:].cpu().numpy()
        self.nt_per_class = np.array([row_sum[1].item()], dtype=np.int32)
    else:
        # Average IoU only over classes present in the ground truth; classes with no GT pixels (absent
        # from the val set or removed by the `classes` filter) are excluded.
        present = row_sum > 0
        self._miou = float(iou[present].mean().item()) if present.any() else 0.0
        self._per_class_iou = iou.cpu().numpy()
        self._per_class_pixel_acc = pa.cpu().numpy()
        self.nt_per_class = row_sum[: self.nc].cpu().numpy().astype(np.int32)

    self._pixel_accuracy = float((intersection.sum() / (self.matrix.sum() + 1e-10)).item())

    if plot:
        self._plot_iou_bars(save_dir, on_plot)
```
</details>

<br>

### Method `ultralytics.utils.metrics.SemanticMetrics.summary` {#ultralytics.utils.metrics.SemanticMetrics.summary}

```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict]
```

Generate a per-class summary of semantic segmentation metrics, with global mIoU and pixel accuracy on each

row.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool` | For semantic metrics, values are already in [0, 1]. | `True` |
| `decimals` | `int` | Number of decimal places to round the metric values to. | `5` |

**Returns**

| Type | Description |
| --- | --- |
| `list[dict]` | A list of dictionaries, one per class, with per-class IoU and shared scalars. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1864-L1889">View on GitHub</a>
```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict]:
    """Generate a per-class summary of semantic segmentation metrics, with global mIoU and pixel accuracy on each
    row.

    Args:
        normalize (bool): For semantic metrics, values are already in [0, 1].
        decimals (int): Number of decimal places to round the metric values to.

    Returns:
        (list[dict]): A list of dictionaries, one per class, with per-class IoU and shared scalars.
    """
    miou = round(self.miou, decimals)
    pixel_acc = round(self.pixel_accuracy, decimals)
    per_class = self.per_class_iou
    names = self.names or {i: str(i) for i in range(len(per_class))}
    return [
        {
            "Class": names.get(c, str(c)),
            "Images": int(self.nt_per_image[c]),
            "Pixels": int(self.nt_per_class[c]),
            "IoU": round(float(per_class[c]), decimals),
            "mIoU": miou,
            "pixel_acc": pixel_acc,
        }
        for c in self.ap_class_index
    ]
```
</details>

<br>

### Method `ultralytics.utils.metrics.SemanticMetrics.update_stats` {#ultralytics.utils.metrics.SemanticMetrics.update\_stats}

```python
def update_stats(self, preds: torch.Tensor, targets: torch.Tensor) -> None
```

Accumulate confusion matrix from predictions and targets.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Predicted class IDs [B, H, W]. | *required* |
| `targets` | `torch.Tensor` | Ground truth class IDs [B, H, W]. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1708-L1730">View on GitHub</a>
```python
def update_stats(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
    """Accumulate confusion matrix from predictions and targets.

    Args:
        preds (torch.Tensor): Predicted class IDs [B, H, W].
        targets (torch.Tensor): Ground truth class IDs [B, H, W].
    """
    if self.matrix is None:
        self.matrix = torch.zeros((self.cm_nc, self.cm_nc), device=preds.device, dtype=torch.float32)

    valid = (targets != 255) & (preds >= 0) & (preds < self.cm_nc) & (targets >= 0) & (targets < self.cm_nc)
    hist = torch.bincount(self.cm_nc * targets[valid] + preds[valid], minlength=self.cm_nc**2).reshape(
        self.cm_nc, self.cm_nc
    )
    self.matrix += hist.to(self.matrix.dtype)

    present = torch.zeros((targets.shape[0], self.cm_nc), dtype=torch.bool, device=targets.device)
    batch_idx = torch.arange(targets.shape[0], device=targets.device).view(-1, 1, 1).expand_as(targets)
    present[batch_idx[valid], targets[valid].long()] = True
    if self.nc == 1:
        self.nt_per_image[0] += int(present[:, 1].sum())
    else:
        self.nt_per_image += present[:, : self.nc].sum(0).cpu().numpy()
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.metrics.DepthMetrics` {#ultralytics.utils.metrics.DepthMetrics}

```python
DepthMetrics(min_depth: float = 0.001, max_depth: float = 100.0, align: str = "median")
```

**Bases:** `SimpleClass`, `DataExportMixin`

Monocular depth estimation metrics: delta1-3, abs_rel, rmse, silog.

Metrics are finalized per image, then averaged across the val set so every image weighs equally regardless of its valid-pixel count, matching the per-sample averaging used by Depth Anything V2 and Monodepth2. Images with fewer than 10 valid ground-truth pixels are skipped entirely, the same floor Depth Anything V2 applies to its own valid mask. Non-finite predictions are scored at a depth bound rather than hiding the image from the average. Per-image results are accumulated in float64 on CPU, so DDP reduction is still a plain sum-then-all_reduce. Following the standard Eigen evaluation protocol, pixels with gt outside (min_depth, max_depth) are excluded and predictions are clamped into that range.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `min_depth` | `float` | Minimum valid depth in meters; pixels with gt <= min_depth are ignored. | `0.001` |
| `max_depth` | `float` | Maximum valid depth in meters; pixels with gt >= max_depth are ignored and predictions are clamped to it. | `100.0` |
| `align` | `str` | Per-image scale alignment before scoring, following the Depth Anything eval protocol. "median" rescales each prediction by median(gt)/median(pred) so affine-invariant (scale-ambiguous) outputs are comparable to metric GT; "none" disables alignment and scores predictions in their raw output scale. | `"median"` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `min_depth` | `float` | Minimum valid depth in meters. |
| `max_depth` | `float` | Maximum valid depth in meters. |

**Methods**

| Name | Description |
| --- | --- |
| [`keys`](#ultralytics.utils.metrics.DepthMetrics.keys) | Metric keys for logging. |
| [`delta1`](#ultralytics.utils.metrics.DepthMetrics.delta1) | Mean per-image fraction of pixels with max(p/g, g/p) < 1.25. |
| [`delta2`](#ultralytics.utils.metrics.DepthMetrics.delta2) | Mean per-image fraction of pixels with max(p/g, g/p) < 1.25**2. |
| [`delta3`](#ultralytics.utils.metrics.DepthMetrics.delta3) | Mean per-image fraction of pixels with max(p/g, g/p) < 1.25**3. |
| [`abs_rel`](#ultralytics.utils.metrics.DepthMetrics.abs_rel) | Mean per-image absolute relative error. |
| [`rmse`](#ultralytics.utils.metrics.DepthMetrics.rmse) | Mean per-image root mean squared error (meters). |
| [`silog`](#ultralytics.utils.metrics.DepthMetrics.silog) | Mean per-image scale-invariant logarithmic error (x100). |
| [`fitness`](#ultralytics.utils.metrics.DepthMetrics.fitness) | Fitness = delta1 (higher is better). |
| [`results_dict`](#ultralytics.utils.metrics.DepthMetrics.results_dict) | Results dict including fitness. |
| [`curves`](#ultralytics.utils.metrics.DepthMetrics.curves) | No PR curves for depth. |
| [`curves_results`](#ultralytics.utils.metrics.DepthMetrics.curves_results) | No PR curve results for depth. |
| [`clear_stats`](#ultralytics.utils.metrics.DepthMetrics.clear_stats) | Reset accumulators. |
| [`mean_results`](#ultralytics.utils.metrics.DepthMetrics.mean_results) | Return metric values in `keys` order. |
| [`process`](#ultralytics.utils.metrics.DepthMetrics.process) | Finalize metrics by averaging the accumulated per-image results. |
| [`summary`](#ultralytics.utils.metrics.DepthMetrics.summary) | Single-row summary of global depth metrics. |
| [`update_stats`](#ultralytics.utils.metrics.DepthMetrics.update_stats) | Accumulate per-image metrics, with per-image scale alignment. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1892-L2066">View on GitHub</a>
```python
class DepthMetrics(SimpleClass, DataExportMixin):
    """Monocular depth estimation metrics: delta1-3, abs_rel, rmse, silog.

    Metrics are finalized per image, then averaged across the val set so every image weighs equally regardless of its
    valid-pixel count, matching the per-sample averaging used by Depth Anything V2 and Monodepth2. Images with fewer
    than 10 valid ground-truth pixels are skipped entirely, the same floor Depth Anything V2 applies to its own valid
    mask. Non-finite predictions are scored at a depth bound rather than hiding the image from the average. Per-image
    results are accumulated in float64 on CPU, so DDP reduction is still a plain sum-then-all_reduce. Following the
    standard Eigen evaluation protocol, pixels with gt outside (min_depth, max_depth) are excluded and predictions are
    clamped into that range.

    Attributes:
        min_depth (float): Minimum valid depth in meters.
        max_depth (float): Maximum valid depth in meters.
    """

    def __init__(
        self,
        min_depth: float = 0.001,
        max_depth: float = 100.0,
        align: str = "median",
    ) -> None:
        """Initialize depth metric accumulators.

        Args:
            min_depth (float): Minimum valid depth in meters; pixels with gt <= min_depth are ignored.
            max_depth (float): Maximum valid depth in meters; pixels with gt >= max_depth are ignored and predictions
                are clamped to it.
            align (str): Per-image scale alignment before scoring, following the Depth Anything eval protocol. "median"
                rescales each prediction by median(gt)/median(pred) so affine-invariant (scale-ambiguous) outputs are
                comparable to metric GT; "none" disables alignment and scores predictions in their raw output scale.
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.align = align
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self._totals = None
        self._count = 0.0
        self._results = {}
```
</details>

<br>

### Property `ultralytics.utils.metrics.DepthMetrics.keys` {#ultralytics.utils.metrics.DepthMetrics.keys}

```python
def keys(self) -> list[str]
```

Metric keys for logging.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1999-L2008">View on GitHub</a>
```python
@property
def keys(self) -> list[str]:
    """Metric keys for logging."""
    return [
        "metrics/delta1",
        "metrics/delta2",
        "metrics/delta3",
        "metrics/abs_rel",
        "metrics/rmse",
        "metrics/silog",
    ]
```
</details>

<br>

### Property `ultralytics.utils.metrics.DepthMetrics.delta1` {#ultralytics.utils.metrics.DepthMetrics.delta1}

```python
def delta1(self) -> float
```

Mean per-image fraction of pixels with max(p/g, g/p) < 1.25.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2015-L2017">View on GitHub</a>
```python
@property
def delta1(self) -> float:
    """Mean per-image fraction of pixels with max(p/g, g/p) < 1.25."""
    return self._results.get("metrics/delta1", 0.0)
```
</details>

<br>

### Property `ultralytics.utils.metrics.DepthMetrics.delta2` {#ultralytics.utils.metrics.DepthMetrics.delta2}

```python
def delta2(self) -> float
```

Mean per-image fraction of pixels with max(p/g, g/p) < 1.25**2.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2020-L2022">View on GitHub</a>
```python
@property
def delta2(self) -> float:
    """Mean per-image fraction of pixels with max(p/g, g/p) < 1.25**2."""
    return self._results.get("metrics/delta2", 0.0)
```
</details>

<br>

### Property `ultralytics.utils.metrics.DepthMetrics.delta3` {#ultralytics.utils.metrics.DepthMetrics.delta3}

```python
def delta3(self) -> float
```

Mean per-image fraction of pixels with max(p/g, g/p) < 1.25**3.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2025-L2027">View on GitHub</a>
```python
@property
def delta3(self) -> float:
    """Mean per-image fraction of pixels with max(p/g, g/p) < 1.25**3."""
    return self._results.get("metrics/delta3", 0.0)
```
</details>

<br>

### Property `ultralytics.utils.metrics.DepthMetrics.abs_rel` {#ultralytics.utils.metrics.DepthMetrics.abs\_rel}

```python
def abs_rel(self) -> float
```

Mean per-image absolute relative error.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2030-L2032">View on GitHub</a>
```python
@property
def abs_rel(self) -> float:
    """Mean per-image absolute relative error."""
    return self._results.get("metrics/abs_rel", 0.0)
```
</details>

<br>

### Property `ultralytics.utils.metrics.DepthMetrics.rmse` {#ultralytics.utils.metrics.DepthMetrics.rmse}

```python
def rmse(self) -> float
```

Mean per-image root mean squared error (meters).

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2035-L2037">View on GitHub</a>
```python
@property
def rmse(self) -> float:
    """Mean per-image root mean squared error (meters)."""
    return self._results.get("metrics/rmse", 0.0)
```
</details>

<br>

### Property `ultralytics.utils.metrics.DepthMetrics.silog` {#ultralytics.utils.metrics.DepthMetrics.silog}

```python
def silog(self) -> float
```

Mean per-image scale-invariant logarithmic error (x100).

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2040-L2042">View on GitHub</a>
```python
@property
def silog(self) -> float:
    """Mean per-image scale-invariant logarithmic error (x100)."""
    return self._results.get("metrics/silog", 0.0)
```
</details>

<br>

### Property `ultralytics.utils.metrics.DepthMetrics.fitness` {#ultralytics.utils.metrics.DepthMetrics.fitness}

```python
def fitness(self) -> float
```

Fitness = delta1 (higher is better).

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2045-L2047">View on GitHub</a>
```python
@property
def fitness(self) -> float:
    """Fitness = delta1 (higher is better)."""
    return self._results.get("metrics/delta1", 0.0)
```
</details>

<br>

### Property `ultralytics.utils.metrics.DepthMetrics.results_dict` {#ultralytics.utils.metrics.DepthMetrics.results\_dict}

```python
def results_dict(self) -> dict[str, float]
```

Results dict including fitness.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2050-L2052">View on GitHub</a>
```python
@property
def results_dict(self) -> dict[str, float]:
    """Results dict including fitness."""
    return dict(zip([*self.keys, "fitness"], [*self.mean_results(), self.fitness]))
```
</details>

<br>

### Property `ultralytics.utils.metrics.DepthMetrics.curves` {#ultralytics.utils.metrics.DepthMetrics.curves}

```python
def curves(self) -> list
```

No PR curves for depth.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2055-L2057">View on GitHub</a>
```python
@property
def curves(self) -> list:
    """No PR curves for depth."""
    return []
```
</details>

<br>

### Property `ultralytics.utils.metrics.DepthMetrics.curves_results` {#ultralytics.utils.metrics.DepthMetrics.curves\_results}

```python
def curves_results(self) -> list
```

No PR curve results for depth.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2060-L2062">View on GitHub</a>
```python
@property
def curves_results(self) -> list:
    """No PR curve results for depth."""
    return []
```
</details>

<br>

### Method `ultralytics.utils.metrics.DepthMetrics.clear_stats` {#ultralytics.utils.metrics.DepthMetrics.clear\_stats}

```python
def clear_stats(self) -> None
```

Reset accumulators.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1992-L1996">View on GitHub</a>
```python
def clear_stats(self) -> None:
    """Reset accumulators."""
    self._totals = None
    self._count = 0.0
    self._results = {}
```
</details>

<br>

### Method `ultralytics.utils.metrics.DepthMetrics.mean_results` {#ultralytics.utils.metrics.DepthMetrics.mean\_results}

```python
def mean_results(self) -> list[float]
```

Return metric values in `keys` order.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2010-L2012">View on GitHub</a>
```python
def mean_results(self) -> list[float]:
    """Return metric values in `keys` order."""
    return [self._results.get(k, 0.0) for k in self.keys]
```
</details>

<br>

### Method `ultralytics.utils.metrics.DepthMetrics.process` {#ultralytics.utils.metrics.DepthMetrics.process}

```python
def process(self, *args, **kwargs) -> None
```

Finalize metrics by averaging the accumulated per-image results.

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1977-L1990">View on GitHub</a>
```python
def process(self, *args, **kwargs) -> None:
    """Finalize metrics by averaging the accumulated per-image results."""
    if self._totals is None or self._count == 0:
        self._results = dict.fromkeys(self.keys, 0.0)
        return
    d1, d2, d3, abs_rel, rmse, silog = (float(x) for x in self._totals / self._count)
    self._results = {
        "metrics/delta1": d1,
        "metrics/delta2": d2,
        "metrics/delta3": d3,
        "metrics/abs_rel": abs_rel,
        "metrics/rmse": rmse,
        "metrics/silog": silog,
    }
```
</details>

<br>

### Method `ultralytics.utils.metrics.DepthMetrics.summary` {#ultralytics.utils.metrics.DepthMetrics.summary}

```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict]
```

Single-row summary of global depth metrics.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `normalize` | `bool` |  | `True` |
| `decimals` | `int` |  | `5` |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L2064-L2066">View on GitHub</a>
```python
def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict]:
    """Single-row summary of global depth metrics."""
    return [{k.split("/")[-1]: round(v, decimals) for k, v in self._results.items()}]
```
</details>

<br>

### Method `ultralytics.utils.metrics.DepthMetrics.update_stats` {#ultralytics.utils.metrics.DepthMetrics.update\_stats}

```python
def update_stats(self, preds: torch.Tensor, targets: torch.Tensor) -> None
```

Accumulate per-image metrics, with per-image scale alignment.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Predicted depth (B,1,H,W) or (B,H,W). | *required* |
| `targets` | `torch.Tensor` | Ground-truth depth in meters, same shape. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L1932-L1975">View on GitHub</a>
```python
def update_stats(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
    """Accumulate per-image metrics, with per-image scale alignment.

    Args:
        preds (torch.Tensor): Predicted depth (B,1,H,W) or (B,H,W).
        targets (torch.Tensor): Ground-truth depth in meters, same shape.
    """
    p = preds.squeeze(1) if preds.ndim == 4 else preds
    g = targets.squeeze(1) if targets.ndim == 4 else targets
    if p.ndim == 2:  # single image (H,W) -> (1,H,W) so alignment is always per-image
        p, g = p[None], g[None]
    for pi, gi in zip(p, g):
        # Eigen protocol: score only pixels with gt inside (min_depth, max_depth)
        mask = (gi > self.min_depth) & (gi < self.max_depth)
        if int(mask.sum()) < 10:  # Depth Anything V2 floor: aligning the median of a few pixels is meaningless
            continue
        pv = pi[mask].float()
        gv = gi[mask].float()
        if self.align == "median":
            finite = torch.isfinite(pv)
            if finite.any():
                scale = torch.median(gv[finite]) / torch.median(pv[finite].clamp_min(self.min_depth))
                pv = pv * scale
        pv = torch.nan_to_num(pv, nan=self.max_depth, posinf=self.max_depth, neginf=self.min_depth).clamp(
            self.min_depth, self.max_depth
        )
        thresh = torch.maximum(pv / gv, gv / pv)
        log_diff = torch.log(pv) - torch.log(gv)
        # λ=1 variance form (ZoeDepth/KITTI), finalized per image so silog also averages per-sample
        silog = (log_diff.pow(2).mean() - log_diff.mean().pow(2)).clamp_min(0.0).sqrt() * 100
        image_metrics = torch.stack(
            [
                (thresh < 1.25).float().mean(),
                (thresh < 1.25**2).float().mean(),
                (thresh < 1.25**3).float().mean(),
                (torch.abs(pv - gv) / gv).mean(),
                ((pv - gv) ** 2).mean().sqrt(),
                silog,
            ]
        )
        if self._totals is None:
            self._totals = torch.zeros(6, dtype=torch.float64)
        self._totals += image_metrics.cpu().double()  # float64 on CPU; MPS tensors cannot be float64
        self._count += 1.0
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.bbox_ioa` {#ultralytics.utils.metrics.bbox\_ioa}

```python
def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray
```

Calculate the intersection over box2 area given box1 and box2.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `box1` | `np.ndarray` | A numpy array of shape (N, 4) representing N bounding boxes in x1y1x2y2 format. | *required* |
| `box2` | `np.ndarray` | A numpy array of shape (M, 4) representing M bounding boxes in x1y1x2y2 format. | *required* |
| `iou` | `bool, optional` | Calculate the standard IoU if True else return inter_area/box2_area. | `False` |
| `eps` | `float, optional` | A small value to avoid division by zero. | `1e-7` |

**Returns**

| Type | Description |
| --- | --- |
| `np.ndarray` | A numpy array of shape (N, M) representing the intersection over box2 area. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L51-L79">View on GitHub</a>
```python
def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    """Calculate the intersection over box2 area given box1 and box2.

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes in x1y1x2y2 format.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes in x1y1x2y2 format.
        iou (bool, optional): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (np.ndarray): A numpy array of shape (N, M) representing the intersection over box2 area.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.box_iou` {#ultralytics.utils.metrics.box\_iou}

```python
def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor
```

Calculate intersection-over-union (IoU) of boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `box1` | `torch.Tensor` | A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format. | *required* |
| `box2` | `torch.Tensor` | A tensor of shape (M, 4) representing M bounding boxes in (x1, y1, x2, y2) format. | *required* |
| `eps` | `float, optional` | A small value to avoid division by zero. | `1e-7` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | An NxM tensor containing the pairwise IoU values for every element in box1 and box2. |

**References**

- https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L82-L102">View on GitHub</a>
```python
def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate intersection-over-union (IoU) of boxes.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes in (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.

    References:
        https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.bbox_iou` {#ultralytics.utils.metrics.bbox\_iou}

```python
def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor
```

Calculate the Intersection over Union (IoU) between bounding boxes.

This function supports various shapes for `box1` and `box2` as long as the last dimension is 4. For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4). Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`, or (x1, y1, x2, y2) if `xywh=False`.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `box1` | `torch.Tensor` | A tensor representing one or more bounding boxes, with the last dimension being 4. | *required* |
| `box2` | `torch.Tensor` | A tensor representing one or more bounding boxes, with the last dimension being 4. | *required* |
| `xywh` | `bool, optional` | If True, input boxes are in (x, y, w, h) format. If False, input boxes are in (x1, y1, x2, y2) format. | `True` |
| `GIoU` | `bool, optional` | If True, calculate Generalized IoU. | `False` |
| `DIoU` | `bool, optional` | If True, calculate Distance IoU. | `False` |
| `CIoU` | `bool, optional` | If True, calculate Complete IoU. | `False` |
| `eps` | `float, optional` | A small value to avoid division by zero. | `1e-7` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | IoU, GIoU, DIoU, or CIoU values depending on the specified flags. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L105-L171">View on GitHub</a>
```python
def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4. For instance, you
    may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4). Internally, the code will split the last
    dimension into (x, y, w, h) if `xywh=True`, or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in (x1, y1,
            x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU.
        DIoU (bool, optional): If True, calculate Distance IoU.
        CIoU (bool, optional): If True, calculate Complete IoU.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.mask_iou` {#ultralytics.utils.metrics.mask\_iou}

```python
def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor
```

Calculate masks IoU.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `mask1` | `torch.Tensor` | A tensor of shape (N, n) where N is the number of ground truth objects and n is the product of image width and height. | *required* |
| `mask2` | `torch.Tensor` | A tensor of shape (M, n) where M is the number of predicted objects and n is the product of image width and height. | *required* |
| `eps` | `float, optional` | A small value to avoid division by zero. | `1e-7` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | A tensor of shape (N, M) representing masks IoU. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L174-L189">View on GitHub</a>
```python
def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
            product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the product
            of image width and height.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.kpt_iou` {#ultralytics.utils.metrics.kpt\_iou}

```python
def kpt_iou(
    kpt1: torch.Tensor, kpt2: torch.Tensor, area: torch.Tensor, sigma: list[float], eps: float = 1e-7
) -> torch.Tensor
```

Calculate Object Keypoint Similarity (OKS).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `kpt1` | `torch.Tensor` | A tensor of shape (N, 17, 3) representing ground truth keypoints. | *required* |
| `kpt2` | `torch.Tensor` | A tensor of shape (M, 17, 3) representing predicted keypoints. | *required* |
| `area` | `torch.Tensor` | A tensor of shape (N,) representing areas from ground truth. | *required* |
| `sigma` | `list[float]` | A list containing 17 values representing keypoint scales. | *required* |
| `eps` | `float, optional` | A small value to avoid division by zero. | `1e-7` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | A tensor of shape (N, M) representing keypoint similarities. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L192-L212">View on GitHub</a>
```python
def kpt_iou(
    kpt1: torch.Tensor, kpt2: torch.Tensor, area: torch.Tensor, sigma: list[float], eps: float = 1e-7
) -> torch.Tensor:
    """Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list[float]): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics._get_covariance_matrix` {#ultralytics.utils.metrics.\_get\_covariance\_matrix}

```python
def _get_covariance_matrix(boxes: torch.Tensor, floor: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Generate covariance matrix from oriented bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `boxes` | `torch.Tensor` | A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format. | *required* |
| `floor` | `float, optional` | Small value added to width/height to bound gradients for sub-stride boxes. | `0.0` |

**Returns**

| Type | Description |
| --- | --- |
| `tuple[torch.Tensor, torch.Tensor, torch.Tensor]` | Covariance matrix components (a, b, c) where the covariance matrix is [[a, c], [c, b]], each of shape (N, 1). |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L215-L233">View on GitHub</a>
```python
def _get_covariance_matrix(boxes: torch.Tensor, floor: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate covariance matrix from oriented bounding boxes.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.
        floor (float, optional): Small value added to width/height to bound gradients for sub-stride boxes.

    Returns:
        (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Covariance matrix components (a, b, c) where the covariance
            matrix is [[a, c], [c, b]], each of shape (N, 1).
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12 + floor, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.probiou` {#ultralytics.utils.metrics.probiou}

```python
def probiou(
    obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7, floor: float = 0.0
) -> torch.Tensor
```

Calculate probabilistic IoU between oriented bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `obb1` | `torch.Tensor` | Ground truth OBBs, shape (N, 5), format xywhr. | *required* |
| `obb2` | `torch.Tensor` | Predicted OBBs, shape (N, 5), format xywhr. | *required* |
| `CIoU` | `bool, optional` | If True, calculate CIoU. | `False` |
| `eps` | `float, optional` | Small value to avoid division by zero. | `1e-7` |
| `floor` | `float, optional` | Small value passed to `_get_covariance_matrix` to bound gradients for sub-stride boxes. | `0.0` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | OBB similarities, shape (N,). |

!!! note "Notes"

    OBB format: [center_x, center_y, width, height, rotation_angle].

**References**

- https://arxiv.org/pdf/2106.06072v1.pdf

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L236-L281">View on GitHub</a>
```python
def probiou(
    obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7, floor: float = 0.0
) -> torch.Tensor:
    """Calculate probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        CIoU (bool, optional): If True, calculate CIoU.
        eps (float, optional): Small value to avoid division by zero.
        floor (float, optional): Small value passed to `_get_covariance_matrix` to bound gradients for sub-stride boxes.

    Returns:
        (torch.Tensor): OBB similarities, shape (N,).

    Notes:
        OBB format: [center_x, center_y, width, height, rotation_angle].

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1, floor)
    a2, b2, c2 = _get_covariance_matrix(obb2, floor)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.batch_probiou` {#ultralytics.utils.metrics.batch\_probiou}

```python
def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor
```

Calculate the probabilistic IoU between oriented bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `obb1` | `torch.Tensor \| np.ndarray` | A tensor of shape (N, 5) representing ground truth obbs, with xywhr format. | *required* |
| `obb2` | `torch.Tensor \| np.ndarray` | A tensor of shape (M, 5) representing predicted obbs, with xywhr format. | *required* |
| `eps` | `float, optional` | A small value to avoid division by zero. | `1e-7` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | A tensor of shape (N, M) representing obb similarities. |

**References**

- https://arxiv.org/pdf/2106.06072v1.pdf

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L284-L317">View on GitHub</a>
```python
def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.smooth_bce` {#ultralytics.utils.metrics.smooth\_bce}

```python
def smooth_bce(eps: float = 0.1) -> tuple[float, float]
```

Compute smoothed positive and negative Binary Cross-Entropy targets.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `eps` | `float, optional` | The epsilon value for label smoothing. | `0.1` |

**Returns**

| Type | Description |
| --- | --- |
| `pos (float)` | Positive label smoothing BCE target. |
| `neg (float)` | Negative label smoothing BCE target. |

**References**

- https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L320-L333">View on GitHub</a>
```python
def smooth_bce(eps: float = 0.1) -> tuple[float, float]:
    """Compute smoothed positive and negative Binary Cross-Entropy targets.

    Args:
        eps (float, optional): The epsilon value for label smoothing.

    Returns:
        pos (float): Positive label smoothing BCE target.
        neg (float): Negative label smoothing BCE target.

    References:
        https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    return 1.0 - 0.5 * eps, 0.5 * eps
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.smooth` {#ultralytics.utils.metrics.smooth}

```python
def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray
```

Box filter of fraction f.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `y` | `np.ndarray` |  | *required* |
| `f` | `float` |  | `0.05` |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L659-L664">View on GitHub</a>
```python
def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.plot_pr_curve` {#ultralytics.utils.metrics.plot\_pr\_curve}

```python
def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] | None = None,
    on_plot=None,
)
```

Plot precision-recall curve.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `px` | `np.ndarray` | X values for the PR curve. | *required* |
| `py` | `np.ndarray` | Y values for the PR curve. | *required* |
| `ap` | `np.ndarray` | Average precision values. | *required* |
| `save_dir` | `Path, optional` | Path to save the plot. | `Path("pr_curve.png")` |
| `names` | `dict[int, str], optional` | Dictionary mapping class indices to class names. | `None` |
| `on_plot` | `callable, optional` | Function to call after plot is saved. | `None` |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L668-L710">View on GitHub</a>
```python
@plt_settings()
def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] | None = None,
    on_plot=None,
):
    """Plot precision-recall curve.

    Args:
        px (np.ndarray): X values for the PR curve.
        py (np.ndarray): Y values for the PR curve.
        ap (np.ndarray): Average precision values.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    names = names if names is not None else {}
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="gray")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        # Pass PR curve data for interactive plotting (class names stored at model level)
        # Transpose py to match other curves: y[class][point] format
        on_plot(save_dir, {"type": "pr_curve", "x": px.tolist(), "y": py.T.tolist(), "ap": ap.tolist()})
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.plot_mc_curve` {#ultralytics.utils.metrics.plot\_mc\_curve}

```python
def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] | None = None,
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    on_plot=None,
)
```

Plot metric-confidence curve.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `px` | `np.ndarray` | X values for the metric-confidence curve. | *required* |
| `py` | `np.ndarray` | Y values for the metric-confidence curve. | *required* |
| `save_dir` | `Path, optional` | Path to save the plot. | `Path("mc_curve.png")` |
| `names` | `dict[int, str], optional` | Dictionary mapping class indices to class names. | `None` |
| `xlabel` | `str, optional` | X-axis label. | `"Confidence"` |
| `ylabel` | `str, optional` | Y-axis label. | `"Metric"` |
| `on_plot` | `callable, optional` | Function to call after plot is saved. | `None` |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L714-L757">View on GitHub</a>
```python
@plt_settings()
def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] | None = None,
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    on_plot=None,
):
    """Plot metric-confidence curve.

    Args:
        px (np.ndarray): X values for the metric-confidence curve.
        py (np.ndarray): Y values for the metric-confidence curve.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    names = names if names is not None else {}
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="gray")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.1)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        # Pass metric-confidence curve data for interactive plotting (class names stored at model level)
        on_plot(save_dir, {"type": f"{ylabel.lower()}_curve", "x": px.tolist(), "y": py.tolist()})
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.compute_ap` {#ultralytics.utils.metrics.compute\_ap}

```python
def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]
```

Compute the average precision (AP) given the recall and precision curves.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `recall` | `list[float]` | The recall curve. | *required* |
| `precision` | `list[float]` | The precision curve. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `ap (float)` | Average precision. |
| `mpre (np.ndarray)` | Precision envelope curve. |
| `mrec (np.ndarray)` | Modified recall curve with sentinel values added at the beginning and end. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L760-L789">View on GitHub</a>
```python
def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list[float]): The recall curve.
        precision (list[float]): The precision curve.

    Returns:
        ap (float): Average precision.
        mpre (np.ndarray): Precision envelope curve.
        mrec (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        func = np.trapezoid if checks.check_version(np.__version__, ">=2.0") else np.trapz  # np.trapz deprecated
        ap = func(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.metrics.ap_per_class` {#ultralytics.utils.metrics.ap\_per\_class}

```python
def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] | None = None,
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple
```

Compute the average precision per class for object detection evaluation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tp` | `np.ndarray` | Binary array indicating whether the detection is correct (True) or not (False). | *required* |
| `conf` | `np.ndarray` | Array of confidence scores of the detections. | *required* |
| `pred_cls` | `np.ndarray` | Array of predicted classes of the detections. | *required* |
| `target_cls` | `np.ndarray` | Array of true classes of the targets. | *required* |
| `plot` | `bool, optional` | Whether to plot PR curves or not. | `False` |
| `on_plot` | `callable, optional` | A callback to pass plots path and data when they are rendered. | `None` |
| `save_dir` | `Path, optional` | Directory to save the PR curves. | `Path()` |
| `names` | `dict[int, str], optional` | Dictionary of class names to plot PR curves. | `None` |
| `eps` | `float, optional` | A small value to avoid division by zero. | `1e-16` |
| `prefix` | `str, optional` | A prefix string for saving the plot files. | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `tp (np.ndarray)` | True positive counts at threshold given by max F1 metric for each class. |
| `fp (np.ndarray)` | False positive counts at threshold given by max F1 metric for each class. |
| `p (np.ndarray)` | Precision values at threshold given by max F1 metric for each class. |
| `r (np.ndarray)` | Recall values at threshold given by max F1 metric for each class. |
| `f1 (np.ndarray)` | F1-score values at threshold given by max F1 metric for each class. |
| `ap (np.ndarray)` | Average precision for each class at different IoU thresholds. |
| `unique_classes (np.ndarray)` | An array of unique classes that have data. |
| `p_curve (np.ndarray)` | Precision curves for each class. |
| `r_curve (np.ndarray)` | Recall curves for each class. |
| `f1_curve (np.ndarray)` | F1-score curves for each class. |
| `x (np.ndarray)` | X-axis values for the curves. |
| `prec_values (np.ndarray)` | Precision values at mAP@0.5 for each class. |

<details>
<summary>Source code in <code>ultralytics/utils/metrics.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L792-L887">View on GitHub</a>
```python
def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] | None = None,
    eps: float = 1e-16,
    prefix: str = "",
) -> tuple:
    """Compute the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the targets.
        plot (bool, optional): Whether to plot PR curves or not.
        on_plot (callable, optional): A callback to pass plots path and data when they are rendered.
        save_dir (Path, optional): Directory to save the PR curves.
        names (dict[int, str], optional): Dictionary of class names to plot PR curves.
        eps (float, optional): A small value to avoid division by zero.
        prefix (str, optional): A prefix string for saving the plot files.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class.
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class.
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class.
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
        p_curve (np.ndarray): Precision curves for each class.
        r_curve (np.ndarray): Recall curves for each class.
        f1_curve (np.ndarray): F1-score curves for each class.
        x (np.ndarray): X-axis values for the curves.
        prec_values (np.ndarray): Precision values at mAP@0.5 for each class.
    """
    names = names if names is not None else {}
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            prec_values.append(np.zeros_like(x))  # keep one row per class, aligned with `ap` and `names`
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}  # dict: only classes that have data
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values
```
</details>

<br><br>
