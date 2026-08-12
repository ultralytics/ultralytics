---
title: utils.loss API Reference
description: Explore detailed descriptions and implementations of various loss functions used in Ultralytics models, including Varifocal Loss, Focal Loss, Bbox Loss, and more.
keywords: Ultralytics, loss functions, Varifocal Loss, Focal Loss, Bbox Loss, Rotated Bbox Loss, Keypoint Loss, YOLO, model training, documentation
---

# Reference for `ultralytics/utils/loss.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`VarifocalLoss`](#ultralytics.utils.loss.VarifocalLoss)
        - [`FocalLoss`](#ultralytics.utils.loss.FocalLoss)
        - [`DFLoss`](#ultralytics.utils.loss.DFLoss)
        - [`BboxLoss`](#ultralytics.utils.loss.BboxLoss)
        - [`RLELoss`](#ultralytics.utils.loss.RLELoss)
        - [`RotatedBboxLoss`](#ultralytics.utils.loss.RotatedBboxLoss)
        - [`MultiChannelDiceLoss`](#ultralytics.utils.loss.MultiChannelDiceLoss)
        - [`BCEDiceLoss`](#ultralytics.utils.loss.BCEDiceLoss)
        - [`KeypointLoss`](#ultralytics.utils.loss.KeypointLoss)
        - [`v8DetectionLoss`](#ultralytics.utils.loss.v8DetectionLoss)
        - [`v8SegmentationLoss`](#ultralytics.utils.loss.v8SegmentationLoss)
        - [`v8PoseLoss`](#ultralytics.utils.loss.v8PoseLoss)
        - [`PoseLoss26`](#ultralytics.utils.loss.PoseLoss26)
        - [`v8ClassificationLoss`](#ultralytics.utils.loss.v8ClassificationLoss)
        - [`v8OBBLoss`](#ultralytics.utils.loss.v8OBBLoss)
        - [`DepthLoss26`](#ultralytics.utils.loss.DepthLoss26)
        - [`E2EDetectLoss`](#ultralytics.utils.loss.E2EDetectLoss)
        - [`E2ELoss`](#ultralytics.utils.loss.E2ELoss)
        - [`TVPDetectLoss`](#ultralytics.utils.loss.TVPDetectLoss)
        - [`TVPSegmentLoss`](#ultralytics.utils.loss.TVPSegmentLoss)
        - [`SemanticSegmentationLoss`](#ultralytics.utils.loss.SemanticSegmentationLoss)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`VarifocalLoss.forward`](#ultralytics.utils.loss.VarifocalLoss.forward)
        - [`FocalLoss.forward`](#ultralytics.utils.loss.FocalLoss.forward)
        - [`DFLoss.__call__`](#ultralytics.utils.loss.DFLoss.__call__)
        - [`BboxLoss.forward`](#ultralytics.utils.loss.BboxLoss.forward)
        - [`RLELoss.forward`](#ultralytics.utils.loss.RLELoss.forward)
        - [`RotatedBboxLoss.forward`](#ultralytics.utils.loss.RotatedBboxLoss.forward)
        - [`MultiChannelDiceLoss.forward`](#ultralytics.utils.loss.MultiChannelDiceLoss.forward)
        - [`BCEDiceLoss.forward`](#ultralytics.utils.loss.BCEDiceLoss.forward)
        - [`KeypointLoss.forward`](#ultralytics.utils.loss.KeypointLoss.forward)
        - [`v8DetectionLoss.preprocess`](#ultralytics.utils.loss.v8DetectionLoss.preprocess)
        - [`v8DetectionLoss.bbox_decode`](#ultralytics.utils.loss.v8DetectionLoss.bbox_decode)
        - [`v8DetectionLoss.get_assigned_targets_and_loss`](#ultralytics.utils.loss.v8DetectionLoss.get_assigned_targets_and_loss)
        - [`v8DetectionLoss.parse_output`](#ultralytics.utils.loss.v8DetectionLoss.parse_output)
        - [`v8DetectionLoss.__call__`](#ultralytics.utils.loss.v8DetectionLoss.__call__)
        - [`v8DetectionLoss.loss`](#ultralytics.utils.loss.v8DetectionLoss.loss)
        - [`v8SegmentationLoss.loss`](#ultralytics.utils.loss.v8SegmentationLoss.loss)
        - [`v8SegmentationLoss.single_mask_loss`](#ultralytics.utils.loss.v8SegmentationLoss.single_mask_loss)
        - [`v8SegmentationLoss.calculate_segmentation_loss`](#ultralytics.utils.loss.v8SegmentationLoss.calculate_segmentation_loss)
        - [`v8PoseLoss.loss`](#ultralytics.utils.loss.v8PoseLoss.loss)
        - [`v8PoseLoss.kpts_decode`](#ultralytics.utils.loss.v8PoseLoss.kpts_decode)
        - [`v8PoseLoss._select_target_keypoints`](#ultralytics.utils.loss.v8PoseLoss._select_target_keypoints)
        - [`v8PoseLoss.calculate_keypoints_loss`](#ultralytics.utils.loss.v8PoseLoss.calculate_keypoints_loss)
        - [`PoseLoss26.loss`](#ultralytics.utils.loss.PoseLoss26.loss)
        - [`PoseLoss26.kpts_decode`](#ultralytics.utils.loss.PoseLoss26.kpts_decode)
        - [`PoseLoss26.calculate_rle_loss`](#ultralytics.utils.loss.PoseLoss26.calculate_rle_loss)
        - [`PoseLoss26.calculate_keypoints_loss`](#ultralytics.utils.loss.PoseLoss26.calculate_keypoints_loss)
        - [`v8ClassificationLoss.__call__`](#ultralytics.utils.loss.v8ClassificationLoss.__call__)
        - [`v8OBBLoss.preprocess`](#ultralytics.utils.loss.v8OBBLoss.preprocess)
        - [`v8OBBLoss.loss`](#ultralytics.utils.loss.v8OBBLoss.loss)
        - [`v8OBBLoss.bbox_decode`](#ultralytics.utils.loss.v8OBBLoss.bbox_decode)
        - [`v8OBBLoss.calculate_angle_loss`](#ultralytics.utils.loss.v8OBBLoss.calculate_angle_loss)
        - [`DepthLoss26._grad_l1`](#ultralytics.utils.loss.DepthLoss26._grad_l1)
        - [`DepthLoss26.__call__`](#ultralytics.utils.loss.DepthLoss26.__call__)
        - [`E2EDetectLoss.__call__`](#ultralytics.utils.loss.E2EDetectLoss.__call__)
        - [`E2ELoss.__call__`](#ultralytics.utils.loss.E2ELoss.__call__)
        - [`E2ELoss.update`](#ultralytics.utils.loss.E2ELoss.update)
        - [`E2ELoss.decay`](#ultralytics.utils.loss.E2ELoss.decay)
        - [`TVPDetectLoss.parse_output`](#ultralytics.utils.loss.TVPDetectLoss.parse_output)
        - [`TVPDetectLoss.__call__`](#ultralytics.utils.loss.TVPDetectLoss.__call__)
        - [`TVPDetectLoss.loss`](#ultralytics.utils.loss.TVPDetectLoss.loss)
        - [`TVPDetectLoss._get_vp_features`](#ultralytics.utils.loss.TVPDetectLoss._get_vp_features)
        - [`TVPSegmentLoss.__call__`](#ultralytics.utils.loss.TVPSegmentLoss.__call__)
        - [`TVPSegmentLoss.loss`](#ultralytics.utils.loss.TVPSegmentLoss.loss)
        - [`SemanticSegmentationLoss._resize_masks`](#ultralytics.utils.loss.SemanticSegmentationLoss._resize_masks)
        - [`SemanticSegmentationLoss._ce_loss`](#ultralytics.utils.loss.SemanticSegmentationLoss._ce_loss)
        - [`SemanticSegmentationLoss._dice_loss`](#ultralytics.utils.loss.SemanticSegmentationLoss._dice_loss)
        - [`SemanticSegmentationLoss._binary_dice_loss`](#ultralytics.utils.loss.SemanticSegmentationLoss._binary_dice_loss)
        - [`SemanticSegmentationLoss.forward`](#ultralytics.utils.loss.SemanticSegmentationLoss.forward)


## Class `ultralytics.utils.loss.VarifocalLoss` {#ultralytics.utils.loss.VarifocalLoss}

```python
VarifocalLoss(gamma: float = 2.0, alpha: float = 0.75)
```

**Bases:** `nn.Module`

Varifocal loss by Zhang et al.

Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on hard-to-classify examples and balancing positive/negative samples.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `gamma` | `float` |  | `2.0` |
| `alpha` | `float` |  | `0.75` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `gamma` | `float` | The focusing parameter that controls how much the loss focuses on hard-to-classify examples. |
| `alpha` | `float` | The balancing factor used to address class imbalance. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.utils.loss.VarifocalLoss.forward) | Compute varifocal loss between predictions and ground truth. |

**References**

- https://arxiv.org/abs/2008.13367

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L22-L51">View on GitHub</a>
```python
class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
```
</details>

<br>

### Method `ultralytics.utils.loss.VarifocalLoss.forward` {#ultralytics.utils.loss.VarifocalLoss.forward}

```python
def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor
```

Compute varifocal loss between predictions and ground truth.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_score` | `torch.Tensor` |  | *required* |
| `gt_score` | `torch.Tensor` |  | *required* |
| `label` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L42-L51">View on GitHub</a>
```python
def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Compute varifocal loss between predictions and ground truth."""
    weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
    with autocast(enabled=False, device=pred_score.device.type):
        loss = (
            (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
            .mean(1)
            .sum()
        )
    return loss
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.FocalLoss` {#ultralytics.utils.loss.FocalLoss}

```python
FocalLoss(gamma: float = 1.5, alpha: float = 0.25)
```

**Bases:** `nn.Module`

Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on hard negatives during training.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `gamma` | `float` |  | `1.5` |
| `alpha` | `float` |  | `0.25` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `gamma` | `float` | The focusing parameter that controls how much the loss focuses on hard-to-classify examples. |
| `alpha` | `torch.Tensor` | The balancing factor used to address class imbalance. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.utils.loss.FocalLoss.forward) | Calculate focal loss with modulating factors for class imbalance. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L54-L86">View on GitHub</a>
```python
class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)
```
</details>

<br>

### Method `ultralytics.utils.loss.FocalLoss.forward` {#ultralytics.utils.loss.FocalLoss.forward}

```python
def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor
```

Calculate focal loss with modulating factors for class imbalance.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred` | `torch.Tensor` |  | *required* |
| `label` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L71-L86">View on GitHub</a>
```python
def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Calculate focal loss with modulating factors for class imbalance."""
    loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
    # p_t = torch.exp(-loss)
    # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

    # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
    pred_prob = pred.sigmoid()  # prob from logits
    p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
    modulating_factor = (1.0 - p_t) ** self.gamma
    loss *= modulating_factor
    if (self.alpha > 0).any():
        self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
        alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        loss *= alpha_factor
    return loss.mean(1).sum()
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.DFLoss` {#ultralytics.utils.loss.DFLoss}

```python
DFLoss(reg_max: int = 16)
```

**Bases:** `nn.Module`

Criterion class for computing Distribution Focal Loss (DFL).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `reg_max` | `int` |  | `16` |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.loss.DFLoss.__call__) | Return sum of left and right DFL losses from https://arxiv.org/abs/2006.04388. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L89-L108">View on GitHub</a>
```python
class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max
```
</details>

<br>

### Method `ultralytics.utils.loss.DFLoss.__call__` {#ultralytics.utils.loss.DFLoss.\_\_call\_\_}

```python
def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor
```

Return sum of left and right DFL losses from https://arxiv.org/abs/2006.04388.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_dist` | `torch.Tensor` |  | *required* |
| `target` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L97-L108">View on GitHub</a>
```python
def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return sum of left and right DFL losses from https://arxiv.org/abs/2006.04388."""
    target = target.clamp_(0, self.reg_max - 1 - 0.01)
    tl = target.long()  # target left
    tr = tl + 1  # target right
    wl = tr - target  # weight left
    wr = 1 - wl  # weight right
    # Compute log_softmax once, then two gathers; cross_entropy(x, t) = -log_softmax(x).gather(t)
    logp = F.log_softmax(pred_dist, dim=1)
    return -(
        logp.gather(1, tl.view(-1, 1)).view(tl.shape) * wl + logp.gather(1, tr.view(-1, 1)).view(tl.shape) * wr
    ).mean(-1, keepdim=True)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.BboxLoss` {#ultralytics.utils.loss.BboxLoss}

```python
BboxLoss(reg_max: int = 16)
```

**Bases:** `nn.Module`

Criterion class for computing training losses for bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `reg_max` | `int` |  | `16` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.utils.loss.BboxLoss.forward) | Compute IoU and DFL losses for bounding boxes. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L111-L155">View on GitHub</a>
```python
class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
```
</details>

<br>

### Method `ultralytics.utils.loss.BboxLoss.forward` {#ultralytics.utils.loss.BboxLoss.forward}

```python
def forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
    imgsz: torch.Tensor,
    stride: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]
```

Compute IoU and DFL losses for bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_dist` | `torch.Tensor` |  | *required* |
| `pred_bboxes` | `torch.Tensor` |  | *required* |
| `anchor_points` | `torch.Tensor` |  | *required* |
| `target_bboxes` | `torch.Tensor` |  | *required* |
| `target_scores` | `torch.Tensor` |  | *required* |
| `target_scores_sum` | `torch.Tensor` |  | *required* |
| `fg_mask` | `torch.Tensor` |  | *required* |
| `imgsz` | `torch.Tensor` |  | *required* |
| `stride` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L119-L155">View on GitHub</a>
```python
def forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
    imgsz: torch.Tensor,
    stride: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute IoU and DFL losses for bounding boxes."""
    weight = target_scores[fg_mask].sum(-1, keepdim=True)
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

    # DFL loss
    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        target_ltrb = bbox2dist(anchor_points, target_bboxes)
        # normalize ltrb by image size
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss_dfl = (
            F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_iou, loss_dfl
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.RLELoss` {#ultralytics.utils.loss.RLELoss}

```python
RLELoss(use_target_weight: bool = True, size_average: bool = True, residual: bool = True)
```

**Bases:** `nn.Module`

Residual Log-Likelihood Estimation Loss.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `use_target_weight` | `bool` | Whether to use target weights for loss calculation. | `True` |
| `size_average` | `bool` | Whether to average the loss over elements. | `True` |
| `residual` | `bool` | Whether to include residual log-likelihood term. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `size_average` | `bool` | Option to average the loss by the batch_size. |
| `use_target_weight` | `bool` | Option to use weighted loss. |
| `residual` | `bool` | Option to add L1 loss and let the flow learn the residual error distribution. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.utils.loss.RLELoss.forward) | Args: |

**References**

- https://arxiv.org/abs/2107.11291
- https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L158-L209">View on GitHub</a>
```python
class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Attributes:
        size_average (bool): Option to average the loss by the batch_size.
        use_target_weight (bool): Option to use weighted loss.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
```
</details>

<br>

### Method `ultralytics.utils.loss.RLELoss.forward` {#ultralytics.utils.loss.RLELoss.forward}

```python
def forward(
    self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
) -> torch.Tensor
```

Args:

sigma (torch.Tensor): Output sigma, shape (N, D).
    log_phi (torch.Tensor): Output log_phi, shape (N).
    error (torch.Tensor): Error, shape (N, D).
    target_weight (torch.Tensor): Weights across different joint types, shape (N).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `sigma` | `torch.Tensor` |  | *required* |
| `log_phi` | `torch.Tensor` |  | *required* |
| `error` | `torch.Tensor` |  | *required* |
| `target_weight` | `torch.Tensor` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L184-L209">View on GitHub</a>
```python
def forward(
    self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
) -> torch.Tensor:
    """
    Args:
        sigma (torch.Tensor): Output sigma, shape (N, D).
        log_phi (torch.Tensor): Output log_phi, shape (N).
        error (torch.Tensor): Error, shape (N, D).
        target_weight (torch.Tensor): Weights across different joint types, shape (N).
    """
    log_sigma = torch.log(sigma)
    loss = log_sigma - log_phi.unsqueeze(1)

    if self.residual:
        loss += torch.log(sigma * 2) + torch.abs(error)

    if self.use_target_weight:
        assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
        if target_weight.dim() == 1:
            target_weight = target_weight.unsqueeze(1)
        loss *= target_weight

    if self.size_average:
        loss /= len(loss)

    return loss.sum()
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.RotatedBboxLoss` {#ultralytics.utils.loss.RotatedBboxLoss}

```python
RotatedBboxLoss(reg_max: int)
```

**Bases:** `BboxLoss`

Criterion class for computing training losses for rotated bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `reg_max` | `int` |  | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.utils.loss.RotatedBboxLoss.forward) | Compute IoU and DFL losses for rotated bounding boxes. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L212-L258">View on GitHub</a>
```python
class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    floor = 0.01

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)
```
</details>

<br>

### Method `ultralytics.utils.loss.RotatedBboxLoss.forward` {#ultralytics.utils.loss.RotatedBboxLoss.forward}

```python
def forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
    imgsz: torch.Tensor,
    stride: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]
```

Compute IoU and DFL losses for rotated bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_dist` | `torch.Tensor` |  | *required* |
| `pred_bboxes` | `torch.Tensor` |  | *required* |
| `anchor_points` | `torch.Tensor` |  | *required* |
| `target_bboxes` | `torch.Tensor` |  | *required* |
| `target_scores` | `torch.Tensor` |  | *required* |
| `target_scores_sum` | `torch.Tensor` |  | *required* |
| `fg_mask` | `torch.Tensor` |  | *required* |
| `imgsz` | `torch.Tensor` |  | *required* |
| `stride` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L221-L258">View on GitHub</a>
```python
def forward(
    self,
    pred_dist: torch.Tensor,
    pred_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_bboxes: torch.Tensor,
    target_scores: torch.Tensor,
    target_scores_sum: torch.Tensor,
    fg_mask: torch.Tensor,
    imgsz: torch.Tensor,
    stride: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute IoU and DFL losses for rotated bounding boxes."""
    weight = target_scores[fg_mask].sum(-1, keepdim=True)
    iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], floor=self.floor)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

    # DFL loss
    if self.dfl_loss:
        target_ltrb = rbox2dist(
            target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
        )
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss_dfl = (
            F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_iou, loss_dfl
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.MultiChannelDiceLoss` {#ultralytics.utils.loss.MultiChannelDiceLoss}

```python
MultiChannelDiceLoss(smooth: float = 1e-6, reduction: str = "mean")
```

**Bases:** `nn.Module`

Criterion class for computing multi-channel Dice losses.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `smooth` | `float` | Smoothing factor to avoid division by zero. | `1e-6` |
| `reduction` | `str` | Reduction method ('mean', 'sum', or 'none'). | `"mean"` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.utils.loss.MultiChannelDiceLoss.forward) | Calculate multi-channel Dice loss between predictions and targets. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L261-L291">View on GitHub</a>
```python
class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
```
</details>

<br>

### Method `ultralytics.utils.loss.MultiChannelDiceLoss.forward` {#ultralytics.utils.loss.MultiChannelDiceLoss.forward}

```python
def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor
```

Calculate multi-channel Dice loss between predictions and targets.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred` | `torch.Tensor` |  | *required* |
| `target` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L275-L291">View on GitHub</a>
```python
def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate multi-channel Dice loss between predictions and targets."""
    assert pred.size() == target.size(), "the size of predict and target must be equal."

    pred = pred.sigmoid()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
    dice_loss = 1.0 - dice
    dice_loss = dice_loss.mean(dim=1)

    if self.reduction == "mean":
        return dice_loss.mean()
    elif self.reduction == "sum":
        return dice_loss.sum()
    else:
        return dice_loss
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.BCEDiceLoss` {#ultralytics.utils.loss.BCEDiceLoss}

```python
BCEDiceLoss(weight_bce: float = 0.5, weight_dice: float = 0.5)
```

**Bases:** `nn.Module`

Criterion class for computing combined BCE and Dice losses.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight_bce` | `float` | Weight factor for BCE loss component. | `0.5` |
| `weight_dice` | `float` | Weight factor for Dice loss component. | `0.5` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.utils.loss.BCEDiceLoss.forward) | Calculate combined BCE and Dice loss between predictions and targets. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L294-L315">View on GitHub</a>
```python
class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)
```
</details>

<br>

### Method `ultralytics.utils.loss.BCEDiceLoss.forward` {#ultralytics.utils.loss.BCEDiceLoss.forward}

```python
def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor
```

Calculate combined BCE and Dice loss between predictions and targets.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred` | `torch.Tensor` |  | *required* |
| `target` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L310-L315">View on GitHub</a>
```python
def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate combined BCE and Dice loss between predictions and targets."""
    _, _, mask_h, mask_w = pred.shape
    if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
        target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
    return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.KeypointLoss` {#ultralytics.utils.loss.KeypointLoss}

```python
KeypointLoss(sigmas: torch.Tensor)
```

**Bases:** `nn.Module`

Criterion class for computing keypoint losses.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `sigmas` | `torch.Tensor` |  | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.utils.loss.KeypointLoss.forward) | Calculate keypoint loss factor and Euclidean distance loss for keypoints. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L318-L334">View on GitHub</a>
```python
class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas
```
</details>

<br>

### Method `ultralytics.utils.loss.KeypointLoss.forward` {#ultralytics.utils.loss.KeypointLoss.forward}

```python
def forward(
    self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
) -> torch.Tensor
```

Calculate keypoint loss factor and Euclidean distance loss for keypoints.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_kpts` | `torch.Tensor` |  | *required* |
| `gt_kpts` | `torch.Tensor` |  | *required* |
| `kpt_mask` | `torch.Tensor` |  | *required* |
| `area` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L326-L334">View on GitHub</a>
```python
def forward(
    self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
) -> torch.Tensor:
    """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
    d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
    kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
    # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
    e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
    return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.v8DetectionLoss` {#ultralytics.utils.loss.v8DetectionLoss}

```python
v8DetectionLoss(model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None)
```

Criterion class for computing training losses for YOLOv8 object detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |
| `tal_topk` | `int` |  | `10` |
| `tal_topk2` | `int \| None` |  | `None` |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.loss.v8DetectionLoss.__call__) | Calculate the sum of the loss for box, cls and dfl multiplied by batch size. |
| [`bbox_decode`](#ultralytics.utils.loss.v8DetectionLoss.bbox_decode) | Decode predicted object bounding box coordinates from anchor points and distribution. |
| [`get_assigned_targets_and_loss`](#ultralytics.utils.loss.v8DetectionLoss.get_assigned_targets_and_loss) | Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and |
| [`loss`](#ultralytics.utils.loss.v8DetectionLoss.loss) | Calculate detection loss using assigned targets. |
| [`parse_output`](#ultralytics.utils.loss.v8DetectionLoss.parse_output) | Parse model predictions to extract features. |
| [`preprocess`](#ultralytics.utils.loss.v8DetectionLoss.preprocess) | Preprocess targets by converting to tensor format and scaling coordinates. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L337-L486">View on GitHub</a>
```python
class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.loss_names = "box_loss", "cls_loss", "dfl_loss" if self.use_dfl else "l1_loss"

        # Class weights for handling imbalanced datasets
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
```
</details>

<br>

### Method `ultralytics.utils.loss.v8DetectionLoss.__call__` {#ultralytics.utils.loss.v8DetectionLoss.\_\_call\_\_}

```python
def __call__(
    self,
    preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate the sum of the loss for box, cls and dfl multiplied by batch size.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor] \| tuple[torch.Tensor, dict[str, torch.Tensor]]` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L472-L478">View on GitHub</a>
```python
def __call__(
    self,
    preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
    return self.loss(self.parse_output(preds), batch)
```
</details>

<br>

### Method `ultralytics.utils.loss.v8DetectionLoss.bbox_decode` {#ultralytics.utils.loss.v8DetectionLoss.bbox\_decode}

```python
def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor
```

Decode predicted object bounding box coordinates from anchor points and distribution.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `anchor_points` | `torch.Tensor` |  | *required* |
| `pred_dist` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L393-L400">View on GitHub</a>
```python
def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
    """Decode predicted object bounding box coordinates from anchor points and distribution."""
    if self.use_dfl:
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
    return dist2bbox(pred_dist, anchor_points, xywh=False)
```
</details>

<br>

### Method `ultralytics.utils.loss.v8DetectionLoss.get_assigned_targets_and_loss` {#ultralytics.utils.loss.v8DetectionLoss.get\_assigned\_targets\_and\_loss}

```python
def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple
```

Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and

target indices.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` |  | *required* |
| `batch` | `dict[str, Any]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L402-L464">View on GitHub</a>
```python
def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
    target indices.
    """
    loss = torch.zeros(3, device=self.device)  # box, cls, dfl
    pred_distri, pred_scores = (
        preds["boxes"].permute(0, 2, 1).contiguous(),
        preds["scores"].permute(0, 2, 1).contiguous(),
    )
    anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

    # Targets
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    # Pboxes
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

    _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)

    # Cls loss with optional class weighting
    bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
    if self.class_weights is not None:
        bce_loss *= self.class_weights
    loss[1] = bce_loss.sum() / target_scores_sum  # BCE

    # Bbox loss
    if fg_mask.sum():
        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )

    loss[0] *= self.hyp.box  # box gain
    loss[1] *= self.hyp.cls  # cls gain
    loss[2] *= self.hyp.dfl  # dfl gain
    return (
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
        loss,
        dict(zip(self.loss_names, loss.detach())),
    )  # loss(box, cls, dfl)
```
</details>

<br>

### Method `ultralytics.utils.loss.v8DetectionLoss.loss` {#ultralytics.utils.loss.v8DetectionLoss.loss}

```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate detection loss using assigned targets.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L480-L486">View on GitHub</a>
```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate detection loss using assigned targets."""
    batch_size = preds["boxes"].shape[0]
    loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
    return loss * batch_size, loss_detach
```
</details>

<br>

### Method `ultralytics.utils.loss.v8DetectionLoss.parse_output` {#ultralytics.utils.loss.v8DetectionLoss.parse\_output}

```python
def parse_output(self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]) -> torch.Tensor
```

Parse model predictions to extract features.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor] \| tuple[torch.Tensor, dict[str, torch.Tensor]]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L466-L470">View on GitHub</a>
```python
def parse_output(
    self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
) -> torch.Tensor:
    """Parse model predictions to extract features."""
    return preds[1] if isinstance(preds, tuple) else preds
```
</details>

<br>

### Method `ultralytics.utils.loss.v8DetectionLoss.preprocess` {#ultralytics.utils.loss.v8DetectionLoss.preprocess}

```python
def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor
```

Preprocess targets by converting to tensor format and scaling coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `targets` | `torch.Tensor` |  | *required* |
| `batch_size` | `int` |  | *required* |
| `scale_tensor` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L375-L391">View on GitHub</a>
```python
def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
    """Preprocess targets by converting to tensor format and scaling coordinates."""
    nl, ne = targets.shape
    if nl == 0:
        out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
    else:
        batch_idx = targets[:, 0].long()  # image index
        _, counts = batch_idx.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
        out[batch_idx, within_idx] = targets[:, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
    return out
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.v8SegmentationLoss` {#ultralytics.utils.loss.v8SegmentationLoss}

```python
v8SegmentationLoss(model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None)
```

**Bases:** `v8DetectionLoss`

Criterion class for computing training losses for YOLOv8 segmentation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |
| `tal_topk` | `int` |  | `10` |
| `tal_topk2` | `int \| None` |  | `None` |

**Methods**

| Name | Description |
| --- | --- |
| [`calculate_segmentation_loss`](#ultralytics.utils.loss.v8SegmentationLoss.calculate_segmentation_loss) | Calculate the loss for instance segmentation. |
| [`loss`](#ultralytics.utils.loss.v8SegmentationLoss.loss) | Calculate and return the combined loss for detection and segmentation. |
| [`single_mask_loss`](#ultralytics.utils.loss.v8SegmentationLoss.single_mask_loss) | Compute the instance segmentation loss for a single image. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L489-L648">View on GitHub</a>
```python
class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.loss_names = ("box_loss", "seg_loss", *self.loss_names[1:], "sem_loss")
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)
```
</details>

<br>

### Method `ultralytics.utils.loss.v8SegmentationLoss.calculate_segmentation_loss` {#ultralytics.utils.loss.v8SegmentationLoss.calculate\_segmentation\_loss}

```python
def calculate_segmentation_loss(
    self,
    fg_mask: torch.Tensor,
    masks: torch.Tensor,
    target_gt_idx: torch.Tensor,
    target_bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    proto: torch.Tensor,
    pred_masks: torch.Tensor,
    imgsz: torch.Tensor,
) -> torch.Tensor
```

Calculate the loss for instance segmentation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `fg_mask` | `torch.Tensor` | A binary tensor of shape (BS, N_anchors) indicating which anchors are positive. | *required* |
| `masks` | `torch.Tensor` | Ground truth masks, shape (BS, H, W) if `overlap` else (N_instances_in_batch, H, W). | *required* |
| `target_gt_idx` | `torch.Tensor` | Indexes of ground truth objects for each anchor of shape (BS, N_anchors). | *required* |
| `target_bboxes` | `torch.Tensor` | Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4). | *required* |
| `batch_idx` | `torch.Tensor` | Batch indices of shape (N_labels_in_batch, 1). | *required* |
| `proto` | `torch.Tensor` | Prototype masks of shape (BS, 32, H, W). | *required* |
| `pred_masks` | `torch.Tensor` | Predicted masks for each anchor of shape (BS, N_anchors, 32). | *required* |
| `imgsz` | `torch.Tensor` | Size of the input image as a tensor of shape (2), i.e., (H, W). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | The calculated loss for instance segmentation. |

!!! note "Notes"

    The batch loss can be computed for improved speed at higher memory usage.
    For example, pred_mask can be computed as follows:
        pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L587-L648">View on GitHub</a>
```python
def calculate_segmentation_loss(
    self,
    fg_mask: torch.Tensor,
    masks: torch.Tensor,
    target_gt_idx: torch.Tensor,
    target_bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    proto: torch.Tensor,
    pred_masks: torch.Tensor,
    imgsz: torch.Tensor,
) -> torch.Tensor:
    """Calculate the loss for instance segmentation.

    Args:
        fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
        masks (torch.Tensor): Ground truth masks, shape (BS, H, W) if `overlap` else (N_instances_in_batch, H, W).
        target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
        target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
        batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
        proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
        pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
        imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

    Returns:
        (torch.Tensor): The calculated loss for instance segmentation.

    Notes:
        The batch loss can be computed for improved speed at higher memory usage.
        For example, pred_mask can be computed as follows:
            pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
    """
    _, _, mask_h, mask_w = proto.shape
    loss = 0

    # Normalize to 0-1
    target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

    # Areas of target bboxes
    marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

    # Normalize to mask size
    mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

    for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea)):
        fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i = single_i
        if fg_mask_i.any():
            mask_idx = target_gt_idx_i[fg_mask_i]
            if self.overlap:
                gt_mask = masks[i] == (mask_idx + 1).view(-1, 1, 1)
                gt_mask = gt_mask.float()
            else:
                gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

            loss += self.single_mask_loss(
                gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
            )

        # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

    return loss / fg_mask.sum()
```
</details>

<br>

### Method `ultralytics.utils.loss.v8SegmentationLoss.loss` {#ultralytics.utils.loss.v8SegmentationLoss.loss}

```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate and return the combined loss for detection and segmentation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L501-L561">View on GitHub</a>
```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate and return the combined loss for detection and segmentation."""
    pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
    loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semantic
    if isinstance(proto, tuple) and len(proto) == 2:
        proto, pred_semantic = proto
    else:
        pred_semantic = None
    (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
    # NOTE: re-assign index for consistency for now. Need to be removed in the future.
    loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

    batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
    if fg_mask.sum():
        # Masks loss
        masks = batch["masks"].to(self.device).float()
        if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
            # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
            proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
        )
        loss[1] = self.calculate_segmentation_loss(
            fg_mask,
            masks,
            target_gt_idx,
            target_bboxes,
            batch["batch_idx"].view(-1, 1),
            proto,
            pred_masks,
            imgsz,
        )
        if pred_semantic is not None:
            sem_idx = batch["sem_masks"].to(self.device).long().unsqueeze(1)  # Nx1xHxW
            if self.overlap:
                present = masks != 0  # NxHxW
            else:
                batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                present = torch.zeros(batch_size, *masks.shape[-2:], dtype=torch.bool, device=self.device)
                for i in range(batch_size):
                    instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                    if len(instance_mask_i):
                        present[i] = instance_mask_i.sum(dim=0) != 0
            # One-hot targets zeroed at uncovered pixels, without F.one_hot's int64 NxHxWxC intermediate
            sem_masks = torch.zeros(sem_idx.shape[0], self.nc, *sem_idx.shape[2:], device=self.device)
            sem_masks.scatter_(1, sem_idx, present.unsqueeze(1).float())  # NxCxHxW

            loss[4] = self.bcedice_loss(pred_semantic, sem_masks)
            loss[4] *= self.hyp.box  # seg gain

    # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
    else:
        loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
        if pred_semantic is not None:
            loss[4] += (pred_semantic * 0).sum()

    loss[1] *= self.hyp.box  # seg gain
    return loss * batch_size, dict(zip(self.loss_names, loss.detach()))  # loss(box, seg, cls, dfl, semantic)
```
</details>

<br>

### Method `ultralytics.utils.loss.v8SegmentationLoss.single_mask_loss` {#ultralytics.utils.loss.v8SegmentationLoss.single\_mask\_loss}

```python
def single_mask_loss(
    gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
) -> torch.Tensor
```

Compute the instance segmentation loss for a single image.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `gt_mask` | `torch.Tensor` | Ground truth mask of shape (N, H, W), where N is the number of objects. | *required* |
| `pred` | `torch.Tensor` | Predicted mask coefficients of shape (N, 32). | *required* |
| `proto` | `torch.Tensor` | Prototype masks of shape (32, H, W). | *required* |
| `xyxy` | `torch.Tensor` | Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4). | *required* |
| `area` | `torch.Tensor` | Area of each ground truth bounding box of shape (N,). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | The calculated mask loss for a single image. |

!!! note "Notes"

    The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
    predicted masks from the prototype masks and predicted mask coefficients.

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L564-L585">View on GitHub</a>
```python
@staticmethod
def single_mask_loss(
    gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
) -> torch.Tensor:
    """Compute the instance segmentation loss for a single image.

    Args:
        gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
        pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
        proto (torch.Tensor): Prototype masks of shape (32, H, W).
        xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
        area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

    Returns:
        (torch.Tensor): The calculated mask loss for a single image.

    Notes:
        The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
        predicted masks from the prototype masks and predicted mask coefficients.
    """
    pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
    loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
    return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.v8PoseLoss` {#ultralytics.utils.loss.v8PoseLoss}

```python
v8PoseLoss(model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int = 10)
```

**Bases:** `v8DetectionLoss`

Criterion class for computing training losses for YOLOv8 pose estimation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |
| `tal_topk` | `int` |  | `10` |
| `tal_topk2` | `int` |  | `10` |

**Methods**

| Name | Description |
| --- | --- |
| [`_select_target_keypoints`](#ultralytics.utils.loss.v8PoseLoss._select_target_keypoints) | Select target keypoints for each anchor based on batch index and target ground truth index. |
| [`calculate_keypoints_loss`](#ultralytics.utils.loss.v8PoseLoss.calculate_keypoints_loss) | Calculate the keypoints loss for the model. |
| [`kpts_decode`](#ultralytics.utils.loss.v8PoseLoss.kpts_decode) | Decode predicted keypoints to image coordinates. |
| [`loss`](#ultralytics.utils.loss.v8PoseLoss.loss) | Calculate the total loss and detach it for pose estimation. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L651-L817">View on GitHub</a>
```python
class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.loss_names = ("box_loss", "pose_loss", "kobj_loss", *self.loss_names[1:])
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints

        sigmas = getattr(model, "kpt_oks_sigmas", None)
        if sigmas is None:
            sigmas = (
                torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
            )
        else:
            sigmas = torch.as_tensor(sigmas, device=self.device, dtype=torch.float32).flatten()
            if len(sigmas) != nkpt or not torch.all(sigmas > 0):
                raise ValueError(f"'kpt_oks_sigmas' must be {nkpt} positive values, got {sigmas.tolist()}")

        self.keypoint_loss = KeypointLoss(sigmas=sigmas)
```
</details>

<br>

### Method `ultralytics.utils.loss.v8PoseLoss._select_target_keypoints` {#ultralytics.utils.loss.v8PoseLoss.\_select\_target\_keypoints}

```python
def _select_target_keypoints(
    self,
    keypoints: torch.Tensor,
    batch_idx: torch.Tensor,
    target_gt_idx: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor
```

Select target keypoints for each anchor based on batch index and target ground truth index.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `keypoints` | `torch.Tensor` | Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim). | *required* |
| `batch_idx` | `torch.Tensor` | Batch index tensor for keypoints, shape (N_kpts_in_batch, 1). | *required* |
| `target_gt_idx` | `torch.Tensor` | Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors). | *required* |
| `masks` | `torch.Tensor` | Binary mask tensor indicating object presence, shape (BS, N_anchors). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim). |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L723-L768">View on GitHub</a>
```python
def _select_target_keypoints(
    self,
    keypoints: torch.Tensor,
    batch_idx: torch.Tensor,
    target_gt_idx: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Select target keypoints for each anchor based on batch index and target ground truth index.

    Args:
        keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
        batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
        target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
        masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

    Returns:
        (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
    """
    batch_idx = batch_idx.flatten()
    batch_size = len(masks)

    # Find the maximum number of keypoints in a single image
    max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

    # Create a tensor to hold batched keypoints
    batched_keypoints = torch.zeros(
        (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
    )

    # Vectorized fill: compute within-batch position for each keypoint using cumulative offsets
    batch_idx_long = batch_idx.long()
    offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=keypoints.device)
    offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
    offsets = offsets.cumsum(0)
    within_idx = torch.arange(len(batch_idx), device=keypoints.device) - offsets[batch_idx_long]
    batched_keypoints[batch_idx_long, within_idx] = keypoints

    # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
    target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

    # Use target_gt_idx_expanded to select keypoints from batched_keypoints
    selected_keypoints = batched_keypoints.gather(
        1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
    )

    return selected_keypoints
```
</details>

<br>

### Method `ultralytics.utils.loss.v8PoseLoss.calculate_keypoints_loss` {#ultralytics.utils.loss.v8PoseLoss.calculate\_keypoints\_loss}

```python
def calculate_keypoints_loss(
    self,
    masks: torch.Tensor,
    target_gt_idx: torch.Tensor,
    keypoints: torch.Tensor,
    batch_idx: torch.Tensor,
    stride_tensor: torch.Tensor,
    target_bboxes: torch.Tensor,
    pred_kpts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]
```

Calculate the keypoints loss for the model.

This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is a binary classification loss that classifies whether a keypoint is present or not.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `masks` | `torch.Tensor` | Binary mask tensor indicating object presence, shape (BS, N_anchors). | *required* |
| `target_gt_idx` | `torch.Tensor` | Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors). | *required* |
| `keypoints` | `torch.Tensor` | Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim). | *required* |
| `batch_idx` | `torch.Tensor` | Batch index tensor for keypoints, shape (N_kpts_in_batch, 1). | *required* |
| `stride_tensor` | `torch.Tensor` | Stride tensor for anchors, shape (N_anchors, 1). | *required* |
| `target_bboxes` | `torch.Tensor` | Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4). | *required* |
| `pred_kpts` | `torch.Tensor` | Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `kpts_loss (torch.Tensor)` | The keypoints loss. |
| `kpts_obj_loss (torch.Tensor)` | The keypoints object loss. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L770-L817">View on GitHub</a>
```python
def calculate_keypoints_loss(
    self,
    masks: torch.Tensor,
    target_gt_idx: torch.Tensor,
    keypoints: torch.Tensor,
    batch_idx: torch.Tensor,
    stride_tensor: torch.Tensor,
    target_bboxes: torch.Tensor,
    pred_kpts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the keypoints loss for the model.

    This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
    based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
    a binary classification loss that classifies whether a keypoint is present or not.

    Args:
        masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
        target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
        keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
        batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
        stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
        target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
        pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

    Returns:
        kpts_loss (torch.Tensor): The keypoints loss.
        kpts_obj_loss (torch.Tensor): The keypoints object loss.
    """
    # Select target keypoints using helper method
    selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

    kpts_loss = 0
    kpts_obj_loss = 0

    if masks.any():
        target_bboxes /= stride_tensor
        gt_kpt = selected_keypoints[masks]
        gt_kpt[..., :2] /= stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks][:, None, None]
        area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
        pred_kpt = pred_kpts[masks]
        kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
        kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

        if pred_kpt.shape[-1] == 3:
            kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

    return kpts_loss, kpts_obj_loss
```
</details>

<br>

### Method `ultralytics.utils.loss.v8PoseLoss.kpts_decode` {#ultralytics.utils.loss.v8PoseLoss.kpts\_decode}

```python
def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor
```

Decode predicted keypoints to image coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `anchor_points` | `torch.Tensor` |  | *required* |
| `pred_kpts` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L715-L721">View on GitHub</a>
```python
@staticmethod
def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
    """Decode predicted keypoints to image coordinates."""
    y = pred_kpts.clone()
    y[..., :2] *= 2.0
    y[..., 0] += anchor_points[:, [0]] - 0.5
    y[..., 1] += anchor_points[:, [1]] - 0.5
    return y
```
</details>

<br>

### Method `ultralytics.utils.loss.v8PoseLoss.loss` {#ultralytics.utils.loss.v8PoseLoss.loss}

```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate the total loss and detach it for pose estimation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L675-L712">View on GitHub</a>
```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate the total loss and detach it for pose estimation."""
    pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
    loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
    (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
        self.get_assigned_targets_and_loss(preds, batch)
    )
    # NOTE: re-assign index for consistency for now. Need to be removed in the future.
    loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

    batch_size = pred_kpts.shape[0]
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

    # Pboxes
    pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

    # Keypoint loss
    if fg_mask.sum():
        keypoints = batch["keypoints"].to(self.device).float().clone()
        keypoints[..., 0] *= imgsz[1]
        keypoints[..., 1] *= imgsz[0]

        loss[1], loss[2] = self.calculate_keypoints_loss(
            fg_mask,
            target_gt_idx,
            keypoints,
            batch["batch_idx"].view(-1, 1),
            stride_tensor,
            target_bboxes,
            pred_kpts,
        )

    loss[1] *= self.hyp.pose  # pose gain
    loss[2] *= self.hyp.kobj  # kobj gain

    return loss * batch_size, dict(zip(self.loss_names, loss.detach()))  # loss(box, pose, kobj, cls, dfl)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.PoseLoss26` {#ultralytics.utils.loss.PoseLoss26}

```python
PoseLoss26(model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None)
```

**Bases:** `v8PoseLoss`

Criterion class for computing training losses for YOLO26 pose estimation with RLE loss support.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |
| `tal_topk` | `int` |  | `10` |
| `tal_topk2` | `int \| None` |  | `None` |

**Methods**

| Name | Description |
| --- | --- |
| [`calculate_keypoints_loss`](#ultralytics.utils.loss.PoseLoss26.calculate_keypoints_loss) | Calculate the keypoints loss for the model. |
| [`calculate_rle_loss`](#ultralytics.utils.loss.PoseLoss26.calculate_rle_loss) | Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints. |
| [`kpts_decode`](#ultralytics.utils.loss.PoseLoss26.kpts_decode) | Decode predicted keypoints to image coordinates. |
| [`loss`](#ultralytics.utils.loss.PoseLoss26.loss) | Calculate the total loss and detach it for pose estimation. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L820-L995">View on GitHub</a>
```python
class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLO26 pose estimation with RLE loss support."""

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.loss_names += ("rle_loss",)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )
```
</details>

<br>

### Method `ultralytics.utils.loss.PoseLoss26.calculate_keypoints_loss` {#ultralytics.utils.loss.PoseLoss26.calculate\_keypoints\_loss}

```python
def calculate_keypoints_loss(
    self,
    masks: torch.Tensor,
    target_gt_idx: torch.Tensor,
    keypoints: torch.Tensor,
    batch_idx: torch.Tensor,
    stride_tensor: torch.Tensor,
    target_bboxes: torch.Tensor,
    pred_kpts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Calculate the keypoints loss for the model.

This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is a binary classification loss that classifies whether a keypoint is present or not.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `masks` | `torch.Tensor` | Binary mask tensor indicating object presence, shape (BS, N_anchors). | *required* |
| `target_gt_idx` | `torch.Tensor` | Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors). | *required* |
| `keypoints` | `torch.Tensor` | Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim). | *required* |
| `batch_idx` | `torch.Tensor` | Batch index tensor for keypoints, shape (N_kpts_in_batch, 1). | *required* |
| `stride_tensor` | `torch.Tensor` | Stride tensor for anchors, shape (N_anchors, 1). | *required* |
| `target_bboxes` | `torch.Tensor` | Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4). | *required* |
| `pred_kpts` | `torch.Tensor` | Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `kpts_loss (torch.Tensor)` | The keypoints loss. |
| `kpts_obj_loss (torch.Tensor)` | The keypoints object loss. |
| `rle_loss (torch.Tensor)` | The RLE loss. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L943-L995">View on GitHub</a>
```python
def calculate_keypoints_loss(
    self,
    masks: torch.Tensor,
    target_gt_idx: torch.Tensor,
    keypoints: torch.Tensor,
    batch_idx: torch.Tensor,
    stride_tensor: torch.Tensor,
    target_bboxes: torch.Tensor,
    pred_kpts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the keypoints loss for the model.

    This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
    based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
    a binary classification loss that classifies whether a keypoint is present or not.

    Args:
        masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
        target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
        keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
        batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
        stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
        target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
        pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

    Returns:
        kpts_loss (torch.Tensor): The keypoints loss.
        kpts_obj_loss (torch.Tensor): The keypoints object loss.
        rle_loss (torch.Tensor): The RLE loss.
    """
    # Select target keypoints using inherited helper method
    selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

    kpts_loss = 0
    kpts_obj_loss = 0
    rle_loss = 0

    if masks.any():
        target_bboxes /= stride_tensor
        gt_kpt = selected_keypoints[masks]
        gt_kpt[..., :2] /= stride_tensor.view(1, -1).expand(masks.shape[0], -1)[masks][:, None, None]
        area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
        pred_kpt = pred_kpts[masks]
        kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
        kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

        if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
            rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
            rle_loss = rle_loss.clamp(min=0)
        if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
            kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

    return kpts_loss, kpts_obj_loss, rle_loss
```
</details>

<br>

### Method `ultralytics.utils.loss.PoseLoss26.calculate_rle_loss` {#ultralytics.utils.loss.PoseLoss26.calculate\_rle\_loss}

```python
def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor
```

Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_kpt` | `torch.Tensor` | Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4. | *required* |
| `gt_kpt` | `torch.Tensor` | Ground truth keypoints, shape (N, num_keypoints, kpts_dim). | *required* |
| `kpt_mask` | `torch.Tensor` | Mask for valid keypoints, shape (N, num_keypoints). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | The RLE loss. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L901-L941">View on GitHub</a>
```python
def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
    """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

    Args:
        pred_kpt (torch.Tensor): Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4.
        gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, num_keypoints, kpts_dim).
        kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

    Returns:
        (torch.Tensor): The RLE loss.
    """
    if not kpt_mask.any():
        return pred_kpt[..., :0].sum()

    pred_kpt_visible = pred_kpt[kpt_mask]
    gt_kpt_visible = gt_kpt[kpt_mask]
    pred_coords = pred_kpt_visible[:, 0:2]
    pred_sigma = pred_kpt_visible[:, -2:]
    gt_coords = gt_kpt_visible[:, 0:2]

    target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
    target_weights = target_weights[kpt_mask]

    pred_sigma = pred_sigma.sigmoid()
    error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)
    if not error.numel():
        return pred_kpt[..., :0].sum()

    # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
    valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
    if not valid_mask.any():
        return pred_kpt[..., :0].sum()

    error = error[valid_mask]
    error = error.clamp(-100, 100)  # Prevent numerical instability
    pred_sigma = pred_sigma[valid_mask]
    target_weights = target_weights[valid_mask]

    log_phi = self.flow_model.log_prob(error)

    return self.rle_loss(pred_sigma, log_phi, error, target_weights)
```
</details>

<br>

### Method `ultralytics.utils.loss.PoseLoss26.kpts_decode` {#ultralytics.utils.loss.PoseLoss26.kpts\_decode}

```python
def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor
```

Decode predicted keypoints to image coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `anchor_points` | `torch.Tensor` |  | *required* |
| `pred_kpts` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L894-L899">View on GitHub</a>
```python
@staticmethod
def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
    """Decode predicted keypoints to image coordinates."""
    y = pred_kpts.clone()
    y[..., 0] += anchor_points[:, [0]]
    y[..., 1] += anchor_points[:, [1]]
    return y
```
</details>

<br>

### Method `ultralytics.utils.loss.PoseLoss26.loss` {#ultralytics.utils.loss.PoseLoss26.loss}

```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate the total loss and detach it for pose estimation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L839-L891">View on GitHub</a>
```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate the total loss and detach it for pose estimation."""
    pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
    loss = torch.zeros(
        6 if self.rle_loss else 5, device=self.device
    )  # box, kpt_location, kpt_visibility, cls, dfl[, rle]
    (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
        self.get_assigned_targets_and_loss(preds, batch)
    )
    # NOTE: re-assign index for consistency for now. Need to be removed in the future.
    loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

    batch_size = pred_kpts.shape[0]
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

    pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

    if self.rle_loss and preds.get("kpts_sigma", None) is not None:
        pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
        pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
        pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

    pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

    # Keypoint loss
    if fg_mask.sum():
        keypoints = batch["keypoints"].to(self.device).float().clone()
        keypoints[..., 0] *= imgsz[1]
        keypoints[..., 1] *= imgsz[0]

        keypoints_loss = self.calculate_keypoints_loss(
            fg_mask,
            target_gt_idx,
            keypoints,
            batch["batch_idx"].view(-1, 1),
            stride_tensor,
            target_bboxes,
            pred_kpts,
        )
        loss[1] = keypoints_loss[0]
        loss[2] = keypoints_loss[1]
        if self.rle_loss is not None:
            loss[5] = keypoints_loss[2]

    loss[1] *= self.hyp.pose  # pose gain
    loss[2] *= self.hyp.kobj  # kobj gain
    if self.rle_loss is not None:
        loss[5] *= self.hyp.rle  # rle gain

    # loss(box, kpt_location, kpt_visibility, cls, dfl[, rle])
    return loss * batch_size, dict(zip(self.loss_names, loss.detach()))
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.v8ClassificationLoss` {#ultralytics.utils.loss.v8ClassificationLoss}

```python
v8ClassificationLoss()
```

Criterion class for computing training losses for classification.

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.loss.v8ClassificationLoss.__call__) | Compute the classification loss between predictions and true labels. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L998-L1005">View on GitHub</a>
```python
class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""
```
</details>

<br>

### Method `ultralytics.utils.loss.v8ClassificationLoss.__call__` {#ultralytics.utils.loss.v8ClassificationLoss.\_\_call\_\_}

```python
def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Compute the classification loss between predictions and true labels.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `Any` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1001-L1005">View on GitHub</a>
```python
def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the classification loss between predictions and true labels."""
    preds = preds[1] if isinstance(preds, (list, tuple)) else preds
    loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
    return loss, {"loss": loss.detach()}
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.v8OBBLoss` {#ultralytics.utils.loss.v8OBBLoss}

```python
v8OBBLoss(model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None)
```

**Bases:** `v8DetectionLoss`

Calculates losses for object detection, classification, and box distribution in rotated YOLO models.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |
| `tal_topk` |  |  | `10` |
| `tal_topk2` | `int \| None` |  | `None` |

**Methods**

| Name | Description |
| --- | --- |
| [`bbox_decode`](#ultralytics.utils.loss.v8OBBLoss.bbox_decode) | Decode predicted object bounding box coordinates from anchor points and distribution. |
| [`calculate_angle_loss`](#ultralytics.utils.loss.v8OBBLoss.calculate_angle_loss) | Calculate oriented angle loss. |
| [`loss`](#ultralytics.utils.loss.v8OBBLoss.loss) | Calculate and return the loss for oriented bounding box detection. |
| [`preprocess`](#ultralytics.utils.loss.v8OBBLoss.preprocess) | Preprocess targets for oriented bounding box detection. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1008-L1176">View on GitHub</a>
```python
class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.loss_names = (*self.loss_names, "angle_loss")
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)
```
</details>

<br>

### Method `ultralytics.utils.loss.v8OBBLoss.bbox_decode` {#ultralytics.utils.loss.v8OBBLoss.bbox\_decode}

```python
def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor) -> torch.Tensor
```

Decode predicted object bounding box coordinates from anchor points and distribution.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `anchor_points` | `torch.Tensor` | Anchor points, (h*w, 2). | *required* |
| `pred_dist` | `torch.Tensor` | Predicted rotated distance, (bs, h*w, 4). | *required* |
| `pred_angle` | `torch.Tensor` | Predicted angle, (bs, h*w, 1). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Predicted rotated bounding boxes with angles, (bs, h*w, 5). |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1129-L1145">View on GitHub</a>
```python
def bbox_decode(
    self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
) -> torch.Tensor:
    """Decode predicted object bounding box coordinates from anchor points and distribution.

    Args:
        anchor_points (torch.Tensor): Anchor points, (h*w, 2).
        pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
    """
    if self.use_dfl:
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
    return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)
```
</details>

<br>

### Method `ultralytics.utils.loss.v8OBBLoss.calculate_angle_loss` {#ultralytics.utils.loss.v8OBBLoss.calculate\_angle\_loss}

```python
def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3)
```

Calculate oriented angle loss.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_bboxes` | `torch.Tensor` | Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta). | *required* |
| `target_bboxes` | `torch.Tensor` | Target bounding boxes with shape [N, 5] (x, y, w, h, theta). | *required* |
| `fg_mask` | `torch.Tensor` | Foreground mask indicating valid predictions. | *required* |
| `weight` | `torch.Tensor` | Loss weights for each prediction. | *required* |
| `target_scores_sum` | `torch.Tensor` | Sum of target scores for normalization. | *required* |
| `lambda_val` | `int` | Controls the sensitivity to aspect ratio. | `3` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | The calculated angle loss. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1147-L1176">View on GitHub</a>
```python
def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
    """Calculate oriented angle loss.

    Args:
        pred_bboxes (torch.Tensor): Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).
        target_bboxes (torch.Tensor): Target bounding boxes with shape [N, 5] (x, y, w, h, theta).
        fg_mask (torch.Tensor): Foreground mask indicating valid predictions.
        weight (torch.Tensor): Loss weights for each prediction.
        target_scores_sum (torch.Tensor): Sum of target scores for normalization.
        lambda_val (int): Controls the sensitivity to aspect ratio.

    Returns:
        (torch.Tensor): The calculated angle loss.
    """
    w_gt = target_bboxes[..., 2]
    h_gt = target_bboxes[..., 3]
    pred_theta = pred_bboxes[..., 4]
    target_theta = target_bboxes[..., 4]

    log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
    scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

    delta_theta = pred_theta - target_theta
    delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
    ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

    ang_loss = scale_weight[fg_mask] * ang_loss
    ang_loss = ang_loss * weight

    return ang_loss.sum() / target_scores_sum
```
</details>

<br>

### Method `ultralytics.utils.loss.v8OBBLoss.loss` {#ultralytics.utils.loss.v8OBBLoss.loss}

```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate and return the loss for oriented bounding box detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1043-L1127">View on GitHub</a>
```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate and return the loss for oriented bounding box detection."""
    loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
    pred_distri, pred_scores, pred_angle = (
        preds["boxes"].permute(0, 2, 1).contiguous(),
        preds["scores"].permute(0, 2, 1).contiguous(),
        preds["angle"].permute(0, 2, 1).contiguous(),
    )
    anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
    batch_size = pred_angle.shape[0]  # batch size

    dtype = pred_scores.dtype
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

    # targets
    try:
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
        rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
        targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    except RuntimeError as e:
        raise TypeError(
            "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
            "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
            "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
            "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
            "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
        ) from e

    # Pboxes
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xywhr, (b, h*w, 5)

    bboxes_for_assigner = pred_bboxes.clone().detach()
    # Only the first four elements need to be scaled
    bboxes_for_assigner[..., :4] *= stride_tensor
    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
        pred_scores.detach().sigmoid(),
        bboxes_for_assigner.type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)

    # Cls loss
    # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
    bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # BCE
    if self.class_weights is not None:
        bce_loss *= self.class_weights
    loss[1] = bce_loss.sum() / target_scores_sum

    # Bbox loss
    if fg_mask.sum():
        target_bboxes[..., :4] /= stride_tensor
        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )
        weight = target_scores[fg_mask].sum(-1)
        loss[3] = self.calculate_angle_loss(
            pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
        )  # angle loss
    else:
        loss[0] += (pred_angle * 0).sum()

    loss[0] *= self.hyp.box  # box gain
    loss[1] *= self.hyp.cls  # cls gain
    loss[2] *= self.hyp.dfl  # dfl gain
    loss[3] *= self.hyp.angle  # angle gain

    return loss * batch_size, dict(zip(self.loss_names, loss.detach()))  # loss(box, cls, dfl, angle)
```
</details>

<br>

### Method `ultralytics.utils.loss.v8OBBLoss.preprocess` {#ultralytics.utils.loss.v8OBBLoss.preprocess}

```python
def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor
```

Preprocess targets for oriented bounding box detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `targets` | `torch.Tensor` |  | *required* |
| `batch_size` | `int` |  | *required* |
| `scale_tensor` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1025-L1041">View on GitHub</a>
```python
def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
    """Preprocess targets for oriented bounding box detection."""
    if targets.shape[0] == 0:
        out = torch.zeros(batch_size, 0, 6, device=self.device)
    else:
        batch_idx = targets[:, 0].long()  # image index
        _, counts = batch_idx.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
        packed_targets = targets[:, 1:].clone()
        packed_targets[:, 1:5].mul_(scale_tensor)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
        out[batch_idx, within_idx] = packed_targets
    return out
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.DepthLoss26` {#ultralytics.utils.loss.DepthLoss26}

```python
DepthLoss26(model: torch.nn.Module)
```

Criterion class for computing training losses for YOLO depth estimation.

Uses scale-invariant log loss (SILog) + gradient-matching loss, following the Depth Anything approach. SILog handles scale ambiguity while gradient loss preserves edges.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.loss.DepthLoss26.__call__) | Calculate depth estimation loss. |
| [`_grad_l1`](#ultralytics.utils.loss.DepthLoss26._grad_l1) | L1 between predicted and GT log-depth spatial gradients (dx, dy), gated by the valid mask. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1179-L1270">View on GitHub</a>
```python
class DepthLoss26:
    """Criterion class for computing training losses for YOLO depth estimation.

    Uses scale-invariant log loss (SILog) + gradient-matching loss, following the Depth Anything approach. SILog handles
    scale ambiguity while gradient loss preserves edges.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize DepthLoss26."""
        device = next(model.parameters()).device
        self.device = device
        h = model.args  # hyperparameters
        self.silog_weight = h.dlog
        self.grad_weight = h.dgrad
        self.silog_lambda = h.dlam  # 1.0 = scale-invariant, 0.0 = log-RMSE
        self.grad_scales = 4
        self.loss_names = "dlog_loss", "dgrad_loss"
```
</details>

<br>

### Method `ultralytics.utils.loss.DepthLoss26.__call__` {#ultralytics.utils.loss.DepthLoss26.\_\_call\_\_}

```python
def __call__(
    self, preds: dict[str, torch.Tensor] | torch.Tensor, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate depth estimation loss.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict \| torch.Tensor` | Dict with "depth" key or raw tensor of (B, 1, H, W) predicted depth. | *required* |
| `batch` | `dict` | Dict with "depth" key holding (B, H, W) ground truth depth in meters. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `loss_sum (torch.Tensor)` | Total loss scaled by batch size. |
| `loss_items (dict[str, torch.Tensor])` | Detached silog/gradient losses keyed by loss_names. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1210-L1270">View on GitHub</a>
```python
def __call__(
    self, preds: dict[str, torch.Tensor] | torch.Tensor, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate depth estimation loss.

    Args:
        preds (dict | torch.Tensor): Dict with "depth" key or raw tensor of (B, 1, H, W) predicted depth.
        batch (dict): Dict with "depth" key holding (B, H, W) ground truth depth in meters.

    Returns:
        loss_sum (torch.Tensor): Total loss scaled by batch size.
        loss_items (dict[str, torch.Tensor]): Detached silog/gradient losses keyed by loss_names.
    """
    loss = torch.zeros(2, device=self.device)
    pred_depth = preds["depth"] if isinstance(preds, dict) else preds
    gt_depth = batch["depth"].to(self.device)

    if gt_depth.ndim == 3:
        gt_depth = gt_depth.unsqueeze(1)

    if gt_depth.shape[-2:] != pred_depth.shape[-2:]:
        pred_depth = F.interpolate(pred_depth, size=gt_depth.shape[-2:], mode="bilinear", align_corners=True)

    valid = gt_depth > 0.001
    if valid.sum() < 10:
        # Keep the result attached so BaseTrainer's unconditional backward() works.
        return pred_depth.sum() * 0.0, dict(zip(self.loss_names, loss.detach()))

    pred_valid = pred_depth[valid]
    gt_valid = gt_depth[valid]

    pred_valid = pred_valid.clamp(min=0.001)

    log_diff = torch.log(pred_valid) - torch.log(gt_valid)
    # Centered variance form: non-negative by construction and fp16-stable near convergence.
    m = log_diff.mean()
    silog = torch.sqrt(((log_diff - m) ** 2).mean() + (1.0 - self.silog_lambda) * m**2 + 1e-6)
    loss[0] = silog * self.silog_weight

    # Multi-scale gradient-matching loss.
    pred_log = torch.log(pred_depth.clamp(min=0.001))
    gt_log = torch.log(gt_depth.clamp(min=0.001))
    valid_f = valid.float()
    grad_loss = self._grad_l1(pred_log, gt_log, valid_f)
    for _ in range(1, max(self.grad_scales, 1)):
        if pred_log.shape[-1] < 4 or pred_log.shape[-2] < 4:
            break
        vp = F.avg_pool2d(valid_f, 2)
        occupied = vp > 0
        # Continue per image while its occupied cells are mostly full: contiguous padding is, LiDAR scatter is not
        keep = (vp.sum(dim=(1, 2, 3)) > 0.7 * occupied.sum(dim=(1, 2, 3))).view(-1, 1, 1, 1)
        if not keep.any():
            break
        denom = vp.clamp(min=1e-6)
        pred_log = F.avg_pool2d(pred_log * valid_f, 2) / denom
        gt_log = F.avg_pool2d(gt_log * valid_f, 2) / denom
        valid_f = occupied.float() * keep  # zeroed images cannot re-enter deeper levels
        grad_loss = grad_loss + self._grad_l1(pred_log, gt_log, valid_f)
    loss[1] = grad_loss * self.grad_weight

    return loss * pred_depth.shape[0], dict(zip(self.loss_names, loss.detach()))
```
</details>

<br>

### Method `ultralytics.utils.loss.DepthLoss26._grad_l1` {#ultralytics.utils.loss.DepthLoss26.\_grad\_l1}

```python
def _grad_l1(pred_log: torch.Tensor, gt_log: torch.Tensor, valid_f: torch.Tensor) -> torch.Tensor
```

L1 between predicted and GT log-depth spatial gradients (dx, dy), gated by the valid mask.

Each gradient is zeroed unless both contributing pixels are valid, so edges are only matched where GT is defined.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_log` | `torch.Tensor` |  | *required* |
| `gt_log` | `torch.Tensor` |  | *required* |
| `valid_f` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1198-L1208">View on GitHub</a>
```python
@staticmethod
def _grad_l1(pred_log: torch.Tensor, gt_log: torch.Tensor, valid_f: torch.Tensor) -> torch.Tensor:
    """L1 between predicted and GT log-depth spatial gradients (dx, dy), gated by the valid mask.

    Each gradient is zeroed unless both contributing pixels are valid, so edges are only
    matched where GT is defined.
    """
    pred_dx = (pred_log[:, :, :, 1:] - pred_log[:, :, :, :-1]) * valid_f[:, :, :, 1:] * valid_f[:, :, :, :-1]
    gt_dx = (gt_log[:, :, :, 1:] - gt_log[:, :, :, :-1]) * valid_f[:, :, :, 1:] * valid_f[:, :, :, :-1]
    pred_dy = (pred_log[:, :, 1:, :] - pred_log[:, :, :-1, :]) * valid_f[:, :, 1:, :] * valid_f[:, :, :-1, :]
    gt_dy = (gt_log[:, :, 1:, :] - gt_log[:, :, :-1, :]) * valid_f[:, :, 1:, :] * valid_f[:, :, :-1, :]
    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.E2EDetectLoss` {#ultralytics.utils.loss.E2EDetectLoss}

```python
E2EDetectLoss(model: torch.nn.Module)
```

Criterion class for computing training losses for end-to-end detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.loss.E2EDetectLoss.__call__) | Calculate the sum of the loss for box, cls and dfl multiplied by batch size. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1273-L1290">View on GitHub</a>
```python
class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model: torch.nn.Module):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)
```
</details>

<br>

### Method `ultralytics.utils.loss.E2EDetectLoss.__call__` {#ultralytics.utils.loss.E2EDetectLoss.\_\_call\_\_}

```python
def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate the sum of the loss for box, cls and dfl multiplied by batch size.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `Any` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1281-L1290">View on GitHub</a>
```python
def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
    preds = preds[1] if isinstance(preds, tuple) else preds
    one2many = preds["one2many"]
    loss_one2many = self.one2many(one2many, batch)
    one2one = preds["one2one"]
    loss_one2one = self.one2one(one2one, batch)
    return loss_one2many[0] + loss_one2one[0], {
        k: loss_one2many[1][k] + loss_one2one[1][k] for k in loss_one2many[1]
    }
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.E2ELoss` {#ultralytics.utils.loss.E2ELoss}

```python
E2ELoss(model: torch.nn.Module, loss_fn=v8DetectionLoss)
```

Criterion class for computing training losses for end-to-end detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |
| `loss_fn` |  |  | `v8DetectionLoss` |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.loss.E2ELoss.__call__) | Calculate the sum of the loss for box, cls and dfl multiplied by batch size. |
| [`decay`](#ultralytics.utils.loss.E2ELoss.decay) | Calculate the decayed weight for one-to-many loss based on the current update step. |
| [`update`](#ultralytics.utils.loss.E2ELoss.update) | Update the weights for one-to-many and one-to-one losses based on the decay schedule. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1293-L1325">View on GitHub</a>
```python
class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model: torch.nn.Module, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # init gain
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # final gain
        self.final_o2m = 0.1
```
</details>

<br>

### Method `ultralytics.utils.loss.E2ELoss.__call__` {#ultralytics.utils.loss.E2ELoss.\_\_call\_\_}

```python
def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate the sum of the loss for box, cls and dfl multiplied by batch size.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `Any` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1309-L1315">View on GitHub</a>
```python
def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
    preds = self.one2many.parse_output(preds)
    one2many, one2one = preds["one2many"], preds["one2one"]
    loss_one2many = self.one2many.loss(one2many, batch)
    loss_one2one = self.one2one.loss(one2one, batch)
    return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]
```
</details>

<br>

### Method `ultralytics.utils.loss.E2ELoss.decay` {#ultralytics.utils.loss.E2ELoss.decay}

```python
def decay(self, x) -> float
```

Calculate the decayed weight for one-to-many loss based on the current update step.

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1323-L1325">View on GitHub</a>
```python
def decay(self, x) -> float:
    """Calculate the decayed weight for one-to-many loss based on the current update step."""
    return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m
```
</details>

<br>

### Method `ultralytics.utils.loss.E2ELoss.update` {#ultralytics.utils.loss.E2ELoss.update}

```python
def update(self) -> None
```

Update the weights for one-to-many and one-to-one losses based on the decay schedule.

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1317-L1321">View on GitHub</a>
```python
def update(self) -> None:
    """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
    self.updates += 1
    self.o2m = self.decay(self.updates)
    self.o2o = max(self.total - self.o2m, 0)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.TVPDetectLoss` {#ultralytics.utils.loss.TVPDetectLoss}

```python
TVPDetectLoss(model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None)
```

Criterion class for computing training losses for text-visual prompt detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |
| `tal_topk` |  |  | `10` |
| `tal_topk2` | `int \| None` |  | `None` |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.loss.TVPDetectLoss.__call__) | Calculate the loss for text-visual prompt detection. |
| [`_get_vp_features`](#ultralytics.utils.loss.TVPDetectLoss._get_vp_features) | Extract visual-prompt features from the model output. |
| [`loss`](#ultralytics.utils.loss.TVPDetectLoss.loss) | Calculate the loss for text-visual prompt detection. |
| [`parse_output`](#ultralytics.utils.loss.TVPDetectLoss.parse_output) | Parse model predictions to extract features. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1328-L1369">View on GitHub</a>
```python
class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        self.loss_names = tuple(k[:-5] for k in self.vp_criterion.loss_names)  # strip "_loss" suffix
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max
```
</details>

<br>

### Method `ultralytics.utils.loss.TVPDetectLoss.__call__` {#ultralytics.utils.loss.TVPDetectLoss.\_\_call\_\_}

```python
def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate the loss for text-visual prompt detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `Any` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1345-L1347">View on GitHub</a>
```python
def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate the loss for text-visual prompt detection."""
    return self.loss(self.parse_output(preds), batch)
```
</details>

<br>

### Method `ultralytics.utils.loss.TVPDetectLoss._get_vp_features` {#ultralytics.utils.loss.TVPDetectLoss.\_get\_vp\_features}

```python
def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]
```

Extract visual-prompt features from the model output.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1361-L1369">View on GitHub</a>
```python
def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Extract visual-prompt features from the model output."""
    scores = preds["scores"]
    vnc = scores.shape[1]

    self.vp_criterion.nc = vnc
    self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
    self.vp_criterion.assigner.num_classes = vnc
    return scores
```
</details>

<br>

### Method `ultralytics.utils.loss.TVPDetectLoss.loss` {#ultralytics.utils.loss.TVPDetectLoss.loss}

```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate the loss for text-visual prompt detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `dict[str, torch.Tensor]` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1349-L1359">View on GitHub</a>
```python
def loss(
    self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate the loss for text-visual prompt detection."""
    if self.ori_nc == preds["scores"].shape[1]:
        loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
        return loss, dict(zip(self.loss_names, loss.detach()))

    preds["scores"] = self._get_vp_features(preds)
    vp_loss = self.vp_criterion(preds, batch)
    return vp_loss[0][1], dict(zip(self.loss_names, vp_loss[1].values()))
```
</details>

<br>

### Method `ultralytics.utils.loss.TVPDetectLoss.parse_output` {#ultralytics.utils.loss.TVPDetectLoss.parse\_output}

```python
def parse_output(self, preds) -> dict[str, torch.Tensor]
```

Parse model predictions to extract features.

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1341-L1343">View on GitHub</a>
```python
def parse_output(self, preds) -> dict[str, torch.Tensor]:
    """Parse model predictions to extract features."""
    return self.vp_criterion.parse_output(preds)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.TVPSegmentLoss` {#ultralytics.utils.loss.TVPSegmentLoss}

```python
TVPSegmentLoss(model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None)
```

**Bases:** `TVPDetectLoss`

Criterion class for computing training losses for text-visual prompt segmentation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` |  | *required* |
| `tal_topk` |  |  | `10` |
| `tal_topk2` | `int \| None` |  | `None` |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.loss.TVPSegmentLoss.__call__) | Calculate the loss for text-visual prompt segmentation. |
| [`loss`](#ultralytics.utils.loss.TVPSegmentLoss.loss) | Calculate the loss for text-visual prompt segmentation. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1372-L1396">View on GitHub</a>
```python
class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model: torch.nn.Module, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk, tal_topk2)
        self.loss_names = tuple(k[:-5] for k in self.vp_criterion.loss_names if k != "sem_loss")  # strip "_loss"
        self.hyp = self.vp_criterion.hyp
```
</details>

<br>

### Method `ultralytics.utils.loss.TVPSegmentLoss.__call__` {#ultralytics.utils.loss.TVPSegmentLoss.\_\_call\_\_}

```python
def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate the loss for text-visual prompt segmentation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `Any` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1382-L1384">View on GitHub</a>
```python
def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate the loss for text-visual prompt segmentation."""
    return self.loss(self.parse_output(preds), batch)
```
</details>

<br>

### Method `ultralytics.utils.loss.TVPSegmentLoss.loss` {#ultralytics.utils.loss.TVPSegmentLoss.loss}

```python
def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Calculate the loss for text-visual prompt segmentation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `Any` |  | *required* |
| `batch` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1386-L1396">View on GitHub</a>
```python
def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calculate the loss for text-visual prompt segmentation."""
    if self.ori_nc == preds["scores"].shape[1]:
        loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
        return loss, dict(zip(self.loss_names, loss.detach()))

    preds["scores"] = self._get_vp_features(preds)
    vp_loss = self.vp_criterion(preds, batch)
    cls_loss = vp_loss[0][2]
    # zip drops the trailing "sem_loss" item to match the logged columns
    return cls_loss, dict(zip(self.loss_names, vp_loss[1].values()))
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.loss.SemanticSegmentationLoss` {#ultralytics.utils.loss.SemanticSegmentationLoss}

```python
SemanticSegmentationLoss(model: torch.nn.Module)
```

**Bases:** `nn.Module`

Loss function for semantic segmentation using cross-entropy and Dice terms.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` | Model containing the SemanticSegment head. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `nc` | `int` | Number of semantic classes. |
| `ce` | `nn.CrossEntropyLoss` | Cross-entropy loss with ignore_index=255. |

**Methods**

| Name | Description |
| --- | --- |
| [`_binary_dice_loss`](#ultralytics.utils.loss.SemanticSegmentationLoss._binary_dice_loss) | Compute Dice loss for single-class (binary) segmentation. |
| [`_ce_loss`](#ultralytics.utils.loss.SemanticSegmentationLoss._ce_loss) | Compute cross-entropy on flattened pixels to avoid the CUDA nll_loss2d path. |
| [`_dice_loss`](#ultralytics.utils.loss.SemanticSegmentationLoss._dice_loss) | Compute Dice loss excluding ignore pixels. |
| [`_resize_masks`](#ultralytics.utils.loss.SemanticSegmentationLoss._resize_masks) | Resize masks to match prediction spatial dimensions. |
| [`forward`](#ultralytics.utils.loss.SemanticSegmentationLoss.forward) | Compute semantic segmentation loss with optional auxiliary loss. |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1399-L1515">View on GitHub</a>
```python
class SemanticSegmentationLoss(nn.Module):
    """Loss function for semantic segmentation using cross-entropy and Dice terms.

    Attributes:
        nc (int): Number of semantic classes.
        ce (nn.CrossEntropyLoss): Cross-entropy loss with ignore_index=255.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize semantic segmentation loss.

        Args:
            model (torch.nn.Module): Model containing the SemanticSegment head.
        """
        super().__init__()
        m = model.model[-1]
        self.nc = m.nc
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        data_name = Path(str(getattr(model.args, "data", "") or "")).stem.lower()
        self.use_cityscapes_weight = data_name in {"cityscapes", "cityscapes8"} and self.nc == len(CITYSCAPES_WEIGHT)
        weight = getattr(model, "class_weights", None)  # cls_pw frequency weights, else hardcoded Cityscapes
        if weight is None and self.use_cityscapes_weight:
            weight = torch.from_numpy(CITYSCAPES_WEIGHT)
        weight = None if weight is None else weight.to(device=self.device, dtype=self.dtype)
        if self.nc == 1:
            self.ce = nn.BCEWithLogitsLoss(reduction="sum")  # binary: class weighting intentionally unsupported
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=255, reduction="sum").to(device=self.device, dtype=self.dtype)
            if weight is not None:
                # Non-persistent: weight is a deterministic constant, no need to serialize into ckpt state_dict.
                self.ce.register_buffer("weight", weight, persistent=False)
```
</details>

<br>

### Method `ultralytics.utils.loss.SemanticSegmentationLoss._binary_dice_loss` {#ultralytics.utils.loss.SemanticSegmentationLoss.\_binary\_dice\_loss}

```python
def _binary_dice_loss(self, preds, masks, valid)
```

Compute Dice loss for single-class (binary) segmentation.

Pixels with value 255 are excluded from Dice terms to match BCE valid-pixel filtering.

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1468-L1478">View on GitHub</a>
```python
def _binary_dice_loss(self, preds, masks, valid):
    """Compute Dice loss for single-class (binary) segmentation.

    Pixels with value 255 are excluded from Dice terms to match BCE valid-pixel filtering.
    """
    valid = valid.reshape_as(masks).float()
    pred_soft = preds.squeeze(1).sigmoid()
    target = (masks == 1).float()
    intersection = (pred_soft * target * valid).sum()
    cardinality = ((pred_soft + target) * valid).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)
```
</details>

<br>

### Method `ultralytics.utils.loss.SemanticSegmentationLoss._ce_loss` {#ultralytics.utils.loss.SemanticSegmentationLoss.\_ce\_loss}

```python
def _ce_loss(self, preds, masks, valid)
```

Compute cross-entropy on flattened pixels to avoid the CUDA nll_loss2d path.

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1440-L1451">View on GitHub</a>
```python
def _ce_loss(self, preds, masks, valid):
    """Compute cross-entropy on flattened pixels to avoid the CUDA nll_loss2d path."""
    flat = masks.reshape(-1)
    if self.nc == 1:
        logits = preds.reshape(-1)[valid]
        target = flat[valid].float()
        denominator = valid.sum()
    else:
        logits = preds.permute(0, 2, 3, 1).reshape(-1, self.nc)
        target = flat.long()
        denominator = valid.sum() if self.ce.weight is None else self.ce.weight[target[valid]].sum()
    return self.ce(logits, target) / denominator.clamp_min(1)
```
</details>

<br>

### Method `ultralytics.utils.loss.SemanticSegmentationLoss._dice_loss` {#ultralytics.utils.loss.SemanticSegmentationLoss.\_dice\_loss}

```python
def _dice_loss(self, preds, masks, valid)
```

Compute Dice loss excluding ignore pixels.

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1453-L1466">View on GitHub</a>
```python
def _dice_loss(self, preds, masks, valid):
    """Compute Dice loss excluding ignore pixels."""
    if self.nc == 1:
        return self._binary_dice_loss(preds, masks, valid)
    flat_target = masks.reshape(-1)
    pred_soft = F.softmax(preds, dim=1)
    target = flat_target[valid].long()
    flat_pred = pred_soft.float().permute(0, 2, 3, 1).reshape(-1, self.nc)[valid]
    intersection = torch.zeros(self.nc, device=preds.device, dtype=torch.float32)
    intersection.scatter_add_(0, target, flat_pred.gather(1, target[:, None]).squeeze(1))
    pred_sum = flat_pred.sum(dim=0)
    target_sum = torch.bincount(target, minlength=self.nc).to(device=preds.device, dtype=torch.float32)
    cardinality = pred_sum + target_sum
    return (1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)).mean()
```
</details>

<br>

### Method `ultralytics.utils.loss.SemanticSegmentationLoss._resize_masks` {#ultralytics.utils.loss.SemanticSegmentationLoss.\_resize\_masks}

```python
def _resize_masks(self, masks, target_shape)
```

Resize masks to match prediction spatial dimensions.

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1432-L1438">View on GitHub</a>
```python
def _resize_masks(self, masks, target_shape):
    """Resize masks to match prediction spatial dimensions."""
    if masks.shape[1:] != target_shape:
        return (
            F.interpolate(masks.float().unsqueeze(1), size=target_shape, mode="nearest").squeeze(1).to(torch.int32)
        )
    return masks
```
</details>

<br>

### Method `ultralytics.utils.loss.SemanticSegmentationLoss.forward` {#ultralytics.utils.loss.SemanticSegmentationLoss.forward}

```python
def forward(self, preds, batch)
```

Compute semantic segmentation loss with optional auxiliary loss.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor \| tuple` | Main logits [B, nc, H', W'], or (main, aux) tuple. | *required* |
| `batch` | `dict` | Batch dict with 'semantic_mask' [B, H, W] containing class IDs (255=ignore). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `tuple[torch.Tensor, dict[str, torch.Tensor]]` | Total loss * batch_size and a dict of detached loss items (ce_loss, dice_loss, aux_loss). |

<details>
<summary>Source code in <code>ultralytics/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py#L1480-L1515">View on GitHub</a>
```python
def forward(self, preds, batch):
    """Compute semantic segmentation loss with optional auxiliary loss.

    Args:
        preds (torch.Tensor | tuple): Main logits [B, nc, H', W'], or (main, aux) tuple.
        batch (dict): Batch dict with 'semantic_mask' [B, H, W] containing class IDs (255=ignore).

    Returns:
        (tuple[torch.Tensor, dict[str, torch.Tensor]]): Total loss * batch_size and a dict of detached loss items
            (ce_loss, dice_loss, aux_loss).
    """
    # Unpack auxiliary logits when present.
    aux_logits = None
    if isinstance(preds, tuple):
        preds, aux_logits = preds

    masks = batch["semantic_mask"].to(preds.device)
    valid = masks.reshape(-1) != 255
    if preds.shape[2:] != masks.shape[1:]:
        preds = F.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)

    # Main cross-entropy and Dice loss.
    ce_loss = self._ce_loss(preds, masks, valid)
    dice_loss = self._dice_loss(preds, masks, valid)
    total = ce_loss + dice_loss

    # Auxiliary cross-entropy loss. Match ce_loss dtype so adding to total succeeds under AMP.
    aux_loss = torch.tensor(0.0, device=preds.device, dtype=ce_loss.dtype)
    if aux_logits is not None:
        if aux_logits.shape[2:] != masks.shape[1:]:
            aux_logits = F.interpolate(aux_logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
        aux_loss = self._ce_loss(aux_logits, masks, valid) * 0.4
        total += aux_loss

    loss_items = {"ce_loss": ce_loss.detach(), "dice_loss": dice_loss.detach(), "aux_loss": aux_loss.detach()}
    return total * preds.shape[0], loss_items
```
</details>

<br><br>
