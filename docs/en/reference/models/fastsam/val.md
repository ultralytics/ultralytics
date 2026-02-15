---
description: Discover FastSAM Validator for segmentation in Ultralytics YOLO. Learn how to validate with custom metrics and avoid common errors. Contribute on GitHub.
keywords: FastSAM Validator, Ultralytics, YOLO, segmentation, validation, metrics, GitHub, contribute, documentation
---

# Reference for `ultralytics/models/fastsam/val.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/val.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/val.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`FastSAMValidator`](#ultralytics.models.fastsam.val.FastSAMValidator)


## Class `ultralytics.models.fastsam.val.FastSAMValidator` {#ultralytics.models.fastsam.val.FastSAMValidator}

```python
FastSAMValidator(self, dataloader = None, save_dir = None, args = None, _callbacks = None)
```

**Bases:** `SegmentationValidator`

Custom validation class for FastSAM (Segment Anything Model) segmentation in the Ultralytics YOLO framework.

Extends the SegmentationValidator class, customizing the validation process specifically for FastSAM. This class sets the task to 'segment' and uses the SegmentMetrics for evaluation. Additionally, plotting features are disabled to avoid errors during validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dataloader` | `torch.utils.data.DataLoader, optional` | DataLoader to be used for validation. | `None` |
| `save_dir` | `Path, optional` | Directory to save results. | `None` |
| `args` | `SimpleNamespace, optional` | Configuration for the validator. | `None` |
| `_callbacks` | `list, optional` | List of callback functions to be invoked during validation. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `dataloader` | `torch.utils.data.DataLoader` | The data loader object used for validation. |
| `save_dir` | `Path` | The directory where validation results will be saved. |
| `args` | `SimpleNamespace` | Additional arguments for customization of the validation process. |
| `_callbacks` | `list` | List of callback functions to be invoked during validation. |
| `metrics` | `SegmentMetrics` | Segmentation metrics calculator for evaluation. |

!!! note "Notes"

    Plots for ConfusionMatrix and other related metrics are disabled in this class to avoid errors.

<details>
<summary>Source code in <code>ultralytics/models/fastsam/val.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/fastsam/val.py#L6-L38"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class FastSAMValidator(SegmentationValidator):
    """Custom validation class for FastSAM (Segment Anything Model) segmentation in the Ultralytics YOLO framework.

    Extends the SegmentationValidator class, customizing the validation process specifically for FastSAM. This class
    sets the task to 'segment' and uses the SegmentMetrics for evaluation. Additionally, plotting features are disabled
    to avoid errors during validation.

    Attributes:
        dataloader (torch.utils.data.DataLoader): The data loader object used for validation.
        save_dir (Path): The directory where validation results will be saved.
        args (SimpleNamespace): Additional arguments for customization of the validation process.
        _callbacks (list): List of callback functions to be invoked during validation.
        metrics (SegmentMetrics): Segmentation metrics calculator for evaluation.

    Methods:
        __init__: Initialize the FastSAMValidator with custom settings for FastSAM.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize the FastSAMValidator class, setting the task to 'segment' and metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            args (SimpleNamespace, optional): Configuration for the validator.
            _callbacks (list, optional): List of callback functions to be invoked during validation.

        Notes:
            Plots for ConfusionMatrix and other related metrics are disabled in this class to avoid errors.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "segment"
        self.args.plots = False  # disable ConfusionMatrix and other plots to avoid errors
```
</details>

<br><br>
