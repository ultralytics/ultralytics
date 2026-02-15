---
description: Explore detailed implementations of loss functions for DETR and RT-DETR models in Ultralytics.
keywords: ultralytics, YOLO, DETR, RT-DETR, loss functions, object detection, deep learning, focal loss, varifocal loss, Hungarian matcher
---

# Reference for `ultralytics/models/utils/loss.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DETRLoss`](#ultralytics.models.utils.loss.DETRLoss)
        - [`RTDETRDetectionLoss`](#ultralytics.models.utils.loss.RTDETRDetectionLoss)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DETRLoss._get_loss_class`](#ultralytics.models.utils.loss.DETRLoss._get_loss_class)
        - [`DETRLoss._get_loss_bbox`](#ultralytics.models.utils.loss.DETRLoss._get_loss_bbox)
        - [`DETRLoss._get_loss_aux`](#ultralytics.models.utils.loss.DETRLoss._get_loss_aux)
        - [`DETRLoss._get_index`](#ultralytics.models.utils.loss.DETRLoss._get_index)
        - [`DETRLoss._get_assigned_bboxes`](#ultralytics.models.utils.loss.DETRLoss._get_assigned_bboxes)
        - [`DETRLoss._get_loss`](#ultralytics.models.utils.loss.DETRLoss._get_loss)
        - [`DETRLoss.forward`](#ultralytics.models.utils.loss.DETRLoss.forward)
        - [`RTDETRDetectionLoss.forward`](#ultralytics.models.utils.loss.RTDETRDetectionLoss.forward)
        - [`RTDETRDetectionLoss.get_dn_match_indices`](#ultralytics.models.utils.loss.RTDETRDetectionLoss.get_dn_match_indices)


## Class `ultralytics.models.utils.loss.DETRLoss` {#ultralytics.models.utils.loss.DETRLoss}

```python
def __init__(
    self,
    nc: int = 80,
    loss_gain: dict[str, float] | None = None,
    aux_loss: bool = True,
    use_fl: bool = True,
    use_vfl: bool = False,
    use_uni_match: bool = False,
    uni_match_ind: int = 0,
    gamma: float = 1.5,
    alpha: float = 0.25,
)
```

**Bases:** `nn.Module`

DETR (DEtection TRansformer) Loss class for calculating various loss components.

This class computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary losses for the DETR object detection model.

Uses default loss_gain if not provided. Initializes HungarianMatcher with preset cost gains. Supports auxiliary losses and various loss types.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `loss_gain` | `dict[str, float], optional` | Coefficients for different loss components. | `None` |
| `aux_loss` | `bool` | Whether to use auxiliary losses from each decoder layer. | `True` |
| `use_fl` | `bool` | Whether to use FocalLoss. | `True` |
| `use_vfl` | `bool` | Whether to use VarifocalLoss. | `False` |
| `use_uni_match` | `bool` | Whether to use fixed layer for auxiliary branch label assignment. | `False` |
| `uni_match_ind` | `int` | Index of fixed layer for uni_match. | `0` |
| `gamma` | `float` | The focusing parameter that controls how much the loss focuses on hard-to-classify examples. | `1.5` |
| `alpha` | `float` | The balancing factor used to address class imbalance. | `0.25` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `nc` | `int` | Number of classes. |
| `loss_gain` | `dict[str, float]` | Coefficients for different loss components. |
| `aux_loss` | `bool` | Whether to compute auxiliary losses. |
| `use_fl` | `bool` | Whether to use FocalLoss. |
| `use_vfl` | `bool` | Whether to use VarifocalLoss. |
| `use_uni_match` | `bool` | Whether to use a fixed layer for auxiliary branch label assignment. |
| `uni_match_ind` | `int` | Index of fixed layer to use if use_uni_match is True. |
| `matcher` | `HungarianMatcher` | Object to compute matching cost and indices. |
| `fl` | `FocalLoss | None` | Focal Loss object if use_fl is True, otherwise None. |
| `vfl` | `VarifocalLoss | None` | Varifocal Loss object if use_vfl is True, otherwise None. |
| `device` | `torch.device` | Device on which tensors are stored. |

**Methods**

| Name | Description |
| --- | --- |
| [`_get_assigned_bboxes`](#ultralytics.models.utils.loss.DETRLoss._get_assigned_bboxes) | Assign predicted bounding boxes to ground truth bounding boxes based on match indices. |
| [`_get_index`](#ultralytics.models.utils.loss.DETRLoss._get_index) | Extract batch indices, source indices, and destination indices from match indices. |
| [`_get_loss`](#ultralytics.models.utils.loss.DETRLoss._get_loss) | Calculate losses for a single prediction layer. |
| [`_get_loss_aux`](#ultralytics.models.utils.loss.DETRLoss._get_loss_aux) | Get auxiliary losses for intermediate decoder layers. |
| [`_get_loss_bbox`](#ultralytics.models.utils.loss.DETRLoss._get_loss_bbox) | Compute bounding box and GIoU losses for predicted and ground truth bounding boxes. |
| [`_get_loss_class`](#ultralytics.models.utils.loss.DETRLoss._get_loss_class) | Compute classification loss based on predictions, target values, and ground truth scores. |
| [`forward`](#ultralytics.models.utils.loss.DETRLoss.forward) | Calculate loss for predicted bounding boxes and scores. |

<details>
<summary>Source code in <code>ultralytics/models/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py#L17-L390"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class DETRLoss(nn.Module):
    """DETR (DEtection TRansformer) Loss class for calculating various loss components.

    This class computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary losses for the DETR
    object detection model.

    Attributes:
        nc (int): Number of classes.
        loss_gain (dict[str, float]): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Whether to use FocalLoss.
        use_vfl (bool): Whether to use VarifocalLoss.
        use_uni_match (bool): Whether to use a fixed layer for auxiliary branch label assignment.
        uni_match_ind (int): Index of fixed layer to use if use_uni_match is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss | None): Focal Loss object if use_fl is True, otherwise None.
        vfl (VarifocalLoss | None): Varifocal Loss object if use_vfl is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
    ):
        """Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with preset cost gains. Supports auxiliary
        losses and various loss types.

        Args:
            nc (int): Number of classes.
            loss_gain (dict[str, float], optional): Coefficients for different loss components.
            aux_loss (bool): Whether to use auxiliary losses from each decoder layer.
            use_fl (bool): Whether to use FocalLoss.
            use_vfl (bool): Whether to use VarifocalLoss.
            use_uni_match (bool): Whether to use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
            gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
            alpha (float): The balancing factor used to address class imbalance.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None
```
</details>

<br>

### Method `ultralytics.models.utils.loss.DETRLoss._get_assigned_bboxes` {#ultralytics.models.utils.loss.DETRLoss.\_get\_assigned\_bboxes}

```python
def _get_assigned_bboxes(
    self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, match_indices: list[tuple]
) -> tuple[torch.Tensor, torch.Tensor]
```

Assign predicted bounding boxes to ground truth bounding boxes based on match indices.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_bboxes` | `torch.Tensor` | Predicted bounding boxes. | *required* |
| `gt_bboxes` | `torch.Tensor` | Ground truth bounding boxes. | *required* |
| `match_indices` | `list[tuple]` | List of tuples containing matched indices. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `pred_assigned (torch.Tensor)` | Assigned predicted bounding boxes. |
| `gt_assigned (torch.Tensor)` | Assigned ground truth bounding boxes. |

<details>
<summary>Source code in <code>ultralytics/models/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py#L273-L299"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_assigned_bboxes(
    self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, match_indices: list[tuple]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign predicted bounding boxes to ground truth bounding boxes based on match indices.

    Args:
        pred_bboxes (torch.Tensor): Predicted bounding boxes.
        gt_bboxes (torch.Tensor): Ground truth bounding boxes.
        match_indices (list[tuple]): List of tuples containing matched indices.

    Returns:
        pred_assigned (torch.Tensor): Assigned predicted bounding boxes.
        gt_assigned (torch.Tensor): Assigned ground truth bounding boxes.
    """
    pred_assigned = torch.cat(
        [
            t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
            for t, (i, _) in zip(pred_bboxes, match_indices)
        ]
    )
    gt_assigned = torch.cat(
        [
            t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
            for t, (_, j) in zip(gt_bboxes, match_indices)
        ]
    )
    return pred_assigned, gt_assigned
```
</details>

<br>

### Method `ultralytics.models.utils.loss.DETRLoss._get_index` {#ultralytics.models.utils.loss.DETRLoss.\_get\_index}

```python
def _get_index(match_indices: list[tuple]) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
```

Extract batch indices, source indices, and destination indices from match indices.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `match_indices` | `list[tuple]` | List of tuples containing matched indices. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `batch_idx (tuple[torch.Tensor, torch.Tensor])` | Tuple containing (batch_idx, src_idx). |
| `dst_idx (torch.Tensor)` | Destination indices. |

<details>
<summary>Source code in <code>ultralytics/models/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py#L258-L271"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _get_index(match_indices: list[tuple]) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Extract batch indices, source indices, and destination indices from match indices.

    Args:
        match_indices (list[tuple]): List of tuples containing matched indices.

    Returns:
        batch_idx (tuple[torch.Tensor, torch.Tensor]): Tuple containing (batch_idx, src_idx).
        dst_idx (torch.Tensor): Destination indices.
    """
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
    src_idx = torch.cat([src for (src, _) in match_indices])
    dst_idx = torch.cat([dst for (_, dst) in match_indices])
    return (batch_idx, src_idx), dst_idx
```
</details>

<br>

### Method `ultralytics.models.utils.loss.DETRLoss._get_loss` {#ultralytics.models.utils.loss.DETRLoss.\_get\_loss}

```python
def _get_loss(
    self,
    pred_bboxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_cls: torch.Tensor,
    gt_groups: list[int],
    masks: torch.Tensor | None = None,
    gt_mask: torch.Tensor | None = None,
    postfix: str = "",
    match_indices: list[tuple] | None = None,
) -> dict[str, torch.Tensor]
```

Calculate losses for a single prediction layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_bboxes` | `torch.Tensor` | Predicted bounding boxes. | *required* |
| `pred_scores` | `torch.Tensor` | Predicted class scores. | *required* |
| `gt_bboxes` | `torch.Tensor` | Ground truth bounding boxes. | *required* |
| `gt_cls` | `torch.Tensor` | Ground truth classes. | *required* |
| `gt_groups` | `list[int]` | Number of ground truths per image. | *required* |
| `masks` | `torch.Tensor, optional` | Predicted masks if using segmentation. | `None` |
| `gt_mask` | `torch.Tensor, optional` | Ground truth masks if using segmentation. | `None` |
| `postfix` | `str, optional` | String to append to loss names. | `""` |
| `match_indices` | `list[tuple], optional` | Pre-computed matching indices. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, torch.Tensor]` | Dictionary of losses. |

<details>
<summary>Source code in <code>ultralytics/models/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py#L301-L349"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_loss(
    self,
    pred_bboxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_cls: torch.Tensor,
    gt_groups: list[int],
    masks: torch.Tensor | None = None,
    gt_mask: torch.Tensor | None = None,
    postfix: str = "",
    match_indices: list[tuple] | None = None,
) -> dict[str, torch.Tensor]:
    """Calculate losses for a single prediction layer.

    Args:
        pred_bboxes (torch.Tensor): Predicted bounding boxes.
        pred_scores (torch.Tensor): Predicted class scores.
        gt_bboxes (torch.Tensor): Ground truth bounding boxes.
        gt_cls (torch.Tensor): Ground truth classes.
        gt_groups (list[int]): Number of ground truths per image.
        masks (torch.Tensor, optional): Predicted masks if using segmentation.
        gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.
        postfix (str, optional): String to append to loss names.
        match_indices (list[tuple], optional): Pre-computed matching indices.

    Returns:
        (dict[str, torch.Tensor]): Dictionary of losses.
    """
    if match_indices is None:
        match_indices = self.matcher(
            pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
        )

    idx, gt_idx = self._get_index(match_indices)
    pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

    bs, nq = pred_scores.shape[:2]
    targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
    targets[idx] = gt_cls[gt_idx]

    gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
    if len(gt_bboxes):
        gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

    return {
        **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
        **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
        # **(self._get_loss_mask(masks, gt_mask, match_indices, postfix) if masks is not None and gt_mask is not None else {})
    }
```
</details>

<br>

### Method `ultralytics.models.utils.loss.DETRLoss._get_loss_aux` {#ultralytics.models.utils.loss.DETRLoss.\_get\_loss\_aux}

```python
def _get_loss_aux(
    self,
    pred_bboxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_cls: torch.Tensor,
    gt_groups: list[int],
    match_indices: list[tuple] | None = None,
    postfix: str = "",
    masks: torch.Tensor | None = None,
    gt_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]
```

Get auxiliary losses for intermediate decoder layers.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_bboxes` | `torch.Tensor` | Predicted bounding boxes from auxiliary layers. | *required* |
| `pred_scores` | `torch.Tensor` | Predicted scores from auxiliary layers. | *required* |
| `gt_bboxes` | `torch.Tensor` | Ground truth bounding boxes. | *required* |
| `gt_cls` | `torch.Tensor` | Ground truth classes. | *required* |
| `gt_groups` | `list[int]` | Number of ground truths per image. | *required* |
| `match_indices` | `list[tuple], optional` | Pre-computed matching indices. | `None` |
| `postfix` | `str, optional` | String to append to loss names. | `""` |
| `masks` | `torch.Tensor, optional` | Predicted masks if using segmentation. | `None` |
| `gt_mask` | `torch.Tensor, optional` | Ground truth masks if using segmentation. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, torch.Tensor]` | Dictionary of auxiliary losses. |

<details>
<summary>Source code in <code>ultralytics/models/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py#L186-L255"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_loss_aux(
    self,
    pred_bboxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_cls: torch.Tensor,
    gt_groups: list[int],
    match_indices: list[tuple] | None = None,
    postfix: str = "",
    masks: torch.Tensor | None = None,
    gt_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Get auxiliary losses for intermediate decoder layers.

    Args:
        pred_bboxes (torch.Tensor): Predicted bounding boxes from auxiliary layers.
        pred_scores (torch.Tensor): Predicted scores from auxiliary layers.
        gt_bboxes (torch.Tensor): Ground truth bounding boxes.
        gt_cls (torch.Tensor): Ground truth classes.
        gt_groups (list[int]): Number of ground truths per image.
        match_indices (list[tuple], optional): Pre-computed matching indices.
        postfix (str, optional): String to append to loss names.
        masks (torch.Tensor, optional): Predicted masks if using segmentation.
        gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.

    Returns:
        (dict[str, torch.Tensor]): Dictionary of auxiliary losses.
    """
    # NOTE: loss class, bbox, giou, mask, dice
    loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
    if match_indices is None and self.use_uni_match:
        match_indices = self.matcher(
            pred_bboxes[self.uni_match_ind],
            pred_scores[self.uni_match_ind],
            gt_bboxes,
            gt_cls,
            gt_groups,
            masks=masks[self.uni_match_ind] if masks is not None else None,
            gt_mask=gt_mask,
        )
    for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
        aux_masks = masks[i] if masks is not None else None
        loss_ = self._get_loss(
            aux_bboxes,
            aux_scores,
            gt_bboxes,
            gt_cls,
            gt_groups,
            masks=aux_masks,
            gt_mask=gt_mask,
            postfix=postfix,
            match_indices=match_indices,
        )
        loss[0] += loss_[f"loss_class{postfix}"]
        loss[1] += loss_[f"loss_bbox{postfix}"]
        loss[2] += loss_[f"loss_giou{postfix}"]
        # if masks is not None and gt_mask is not None:
        #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
        #     loss[3] += loss_[f'loss_mask{postfix}']
        #     loss[4] += loss_[f'loss_dice{postfix}']

    loss = {
        f"loss_class_aux{postfix}": loss[0],
        f"loss_bbox_aux{postfix}": loss[1],
        f"loss_giou_aux{postfix}": loss[2],
    }
    # if masks is not None and gt_mask is not None:
    #     loss[f'loss_mask_aux{postfix}'] = loss[3]
    #     loss[f'loss_dice_aux{postfix}'] = loss[4]
    return loss
```
</details>

<br>

### Method `ultralytics.models.utils.loss.DETRLoss._get_loss_bbox` {#ultralytics.models.utils.loss.DETRLoss.\_get\_loss\_bbox}

```python
def _get_loss_bbox(
    self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = ""
) -> dict[str, torch.Tensor]
```

Compute bounding box and GIoU losses for predicted and ground truth bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_bboxes` | `torch.Tensor` | Predicted bounding boxes with shape (N, 4). | *required* |
| `gt_bboxes` | `torch.Tensor` | Ground truth bounding boxes with shape (N, 4). | *required* |
| `postfix` | `str, optional` | String to append to the loss names for identification in multi-loss scenarios. | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, torch.Tensor]` | Dictionary containing: |

!!! note "Notes"

    If no ground truth boxes are provided (empty list), zero-valued tensors are returned for both losses.

<details>
<summary>Source code in <code>ultralytics/models/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py#L121-L153"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_loss_bbox(
    self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = ""
) -> dict[str, torch.Tensor]:
    """Compute bounding box and GIoU losses for predicted and ground truth bounding boxes.

    Args:
        pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (N, 4).
        gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (N, 4).
        postfix (str, optional): String to append to the loss names for identification in multi-loss scenarios.

    Returns:
        (dict[str, torch.Tensor]): Dictionary containing:
            - loss_bbox{postfix}: L1 loss between predicted and ground truth boxes, scaled by the bbox loss gain.
            - loss_giou{postfix}: GIoU loss between predicted and ground truth boxes, scaled by the giou loss gain.

    Notes:
        If no ground truth boxes are provided (empty list), zero-valued tensors are returned for both losses.
    """
    # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
    name_bbox = f"loss_bbox{postfix}"
    name_giou = f"loss_giou{postfix}"

    loss = {}
    if len(gt_bboxes) == 0:
        loss[name_bbox] = torch.tensor(0.0, device=self.device)
        loss[name_giou] = torch.tensor(0.0, device=self.device)
        return loss

    loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
    loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
    loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
    loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
    return {k: v.squeeze() for k, v in loss.items()}
```
</details>

<br>

### Method `ultralytics.models.utils.loss.DETRLoss._get_loss_class` {#ultralytics.models.utils.loss.DETRLoss.\_get\_loss\_class}

```python
def _get_loss_class(
    self, pred_scores: torch.Tensor, targets: torch.Tensor, gt_scores: torch.Tensor, num_gts: int, postfix: str = ""
) -> dict[str, torch.Tensor]
```

Compute classification loss based on predictions, target values, and ground truth scores.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_scores` | `torch.Tensor` | Predicted class scores with shape (B, N, C). | *required* |
| `targets` | `torch.Tensor` | Target class indices with shape (B, N). | *required* |
| `gt_scores` | `torch.Tensor` | Ground truth confidence scores with shape (B, N). | *required* |
| `num_gts` | `int` | Number of ground truth objects. | *required* |
| `postfix` | `str, optional` | String to append to the loss name for identification in multi-loss scenarios. | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, torch.Tensor]` | Dictionary containing classification loss value. |

!!! note "Notes"

    The function supports different classification loss types:
    - Varifocal Loss (if self.vfl is not None and num_gts > 0)
    - Focal Loss (if self.fl is not None)
    - BCE Loss (default fallback)

<details>
<summary>Source code in <code>ultralytics/models/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py#L80-L119"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_loss_class(
    self, pred_scores: torch.Tensor, targets: torch.Tensor, gt_scores: torch.Tensor, num_gts: int, postfix: str = ""
) -> dict[str, torch.Tensor]:
    """Compute classification loss based on predictions, target values, and ground truth scores.

    Args:
        pred_scores (torch.Tensor): Predicted class scores with shape (B, N, C).
        targets (torch.Tensor): Target class indices with shape (B, N).
        gt_scores (torch.Tensor): Ground truth confidence scores with shape (B, N).
        num_gts (int): Number of ground truth objects.
        postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

    Returns:
        (dict[str, torch.Tensor]): Dictionary containing classification loss value.

    Notes:
        The function supports different classification loss types:
        - Varifocal Loss (if self.vfl is not None and num_gts > 0)
        - Focal Loss (if self.fl is not None)
        - BCE Loss (default fallback)
    """
    # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
    name_class = f"loss_class{postfix}"
    bs, nq = pred_scores.shape[:2]
    # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
    one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
    one_hot.scatter_(2, targets.unsqueeze(-1), 1)
    one_hot = one_hot[..., :-1]
    gt_scores = gt_scores.view(bs, nq, 1) * one_hot

    if self.fl:
        if num_gts and self.vfl:
            loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
        else:
            loss_cls = self.fl(pred_scores, one_hot.float())
        loss_cls /= max(num_gts, 1) / nq
    else:
        loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

    return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}
```
</details>

<br>

### Method `ultralytics.models.utils.loss.DETRLoss.forward` {#ultralytics.models.utils.loss.DETRLoss.forward}

```python
def forward(
    self,
    pred_bboxes: torch.Tensor,
    pred_scores: torch.Tensor,
    batch: dict[str, Any],
    postfix: str = "",
    **kwargs: Any,
) -> dict[str, torch.Tensor]
```

Calculate loss for predicted bounding boxes and scores.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pred_bboxes` | `torch.Tensor` | Predicted bounding boxes, shape (L, B, N, 4). | *required* |
| `pred_scores` | `torch.Tensor` | Predicted class scores, shape (L, B, N, C). | *required* |
| `batch` | `dict[str, Any]` | Batch information containing cls, bboxes, and gt_groups. | *required* |
| `postfix` | `str, optional` | Postfix for loss names. | `""` |
| `**kwargs` | `Any` | Additional arguments, may include 'match_indices'. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, torch.Tensor]` | Computed losses, including main and auxiliary (if enabled). |

!!! note "Notes"

    Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
    self.aux_loss is True.

<details>
<summary>Source code in <code>ultralytics/models/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py#L351-L390"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(
    self,
    pred_bboxes: torch.Tensor,
    pred_scores: torch.Tensor,
    batch: dict[str, Any],
    postfix: str = "",
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """Calculate loss for predicted bounding boxes and scores.

    Args:
        pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (L, B, N, 4).
        pred_scores (torch.Tensor): Predicted class scores, shape (L, B, N, C).
        batch (dict[str, Any]): Batch information containing cls, bboxes, and gt_groups.
        postfix (str, optional): Postfix for loss names.
        **kwargs (Any): Additional arguments, may include 'match_indices'.

    Returns:
        (dict[str, torch.Tensor]): Computed losses, including main and auxiliary (if enabled).

    Notes:
        Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
        self.aux_loss is True.
    """
    self.device = pred_bboxes.device
    match_indices = kwargs.get("match_indices", None)
    gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

    total_loss = self._get_loss(
        pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
    )

    if self.aux_loss:
        total_loss.update(
            self._get_loss_aux(
                pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
            )
        )

    return total_loss
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.utils.loss.RTDETRDetectionLoss` {#ultralytics.models.utils.loss.RTDETRDetectionLoss}

```python
RTDETRDetectionLoss()
```

**Bases:** `DETRLoss`

Real-Time DEtection TRansformer (RT-DETR) Detection Loss class that extends the DETRLoss.

This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as an additional denoising training loss when provided with denoising metadata.

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.utils.loss.RTDETRDetectionLoss.forward) | Forward pass to compute detection loss with optional denoising loss. |
| [`get_dn_match_indices`](#ultralytics.models.utils.loss.RTDETRDetectionLoss.get_dn_match_indices) | Get match indices for denoising. |

<details>
<summary>Source code in <code>ultralytics/models/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py#L393-L466"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RTDETRDetectionLoss(DETRLoss):
```
</details>

<br>

### Method `ultralytics.models.utils.loss.RTDETRDetectionLoss.forward` {#ultralytics.models.utils.loss.RTDETRDetectionLoss.forward}

```python
def forward(
    self,
    preds: tuple[torch.Tensor, torch.Tensor],
    batch: dict[str, Any],
    dn_bboxes: torch.Tensor | None = None,
    dn_scores: torch.Tensor | None = None,
    dn_meta: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]
```

Forward pass to compute detection loss with optional denoising loss.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `tuple[torch.Tensor, torch.Tensor]` | Tuple containing predicted bounding boxes and scores. | *required* |
| `batch` | `dict[str, Any]` | Batch data containing ground truth information. | *required* |
| `dn_bboxes` | `torch.Tensor, optional` | Denoising bounding boxes. | `None` |
| `dn_scores` | `torch.Tensor, optional` | Denoising scores. | `None` |
| `dn_meta` | `dict[str, Any], optional` | Metadata for denoising. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `dict[str, torch.Tensor]` | Dictionary containing total loss and denoising loss if applicable. |

<details>
<summary>Source code in <code>ultralytics/models/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py#L400-L438"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(
    self,
    preds: tuple[torch.Tensor, torch.Tensor],
    batch: dict[str, Any],
    dn_bboxes: torch.Tensor | None = None,
    dn_scores: torch.Tensor | None = None,
    dn_meta: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Forward pass to compute detection loss with optional denoising loss.

    Args:
        preds (tuple[torch.Tensor, torch.Tensor]): Tuple containing predicted bounding boxes and scores.
        batch (dict[str, Any]): Batch data containing ground truth information.
        dn_bboxes (torch.Tensor, optional): Denoising bounding boxes.
        dn_scores (torch.Tensor, optional): Denoising scores.
        dn_meta (dict[str, Any], optional): Metadata for denoising.

    Returns:
        (dict[str, torch.Tensor]): Dictionary containing total loss and denoising loss if applicable.
    """
    pred_bboxes, pred_scores = preds
    total_loss = super().forward(pred_bboxes, pred_scores, batch)

    # Check for denoising metadata to compute denoising training loss
    if dn_meta is not None:
        dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
        assert len(batch["gt_groups"]) == len(dn_pos_idx)

        # Get the match indices for denoising
        match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

        # Compute the denoising training loss
        dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
        total_loss.update(dn_loss)
    else:
        # If no denoising metadata is provided, set denoising loss to zero
        total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

    return total_loss
```
</details>

<br>

### Method `ultralytics.models.utils.loss.RTDETRDetectionLoss.get_dn_match_indices` {#ultralytics.models.utils.loss.RTDETRDetectionLoss.get\_dn\_match\_indices}

```python
def get_dn_match_indices(
    dn_pos_idx: list[torch.Tensor], dn_num_group: int, gt_groups: list[int]
) -> list[tuple[torch.Tensor, torch.Tensor]]
```

Get match indices for denoising.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dn_pos_idx` | `list[torch.Tensor]` | List of tensors containing positive indices for denoising. | *required* |
| `dn_num_group` | `int` | Number of denoising groups. | *required* |
| `gt_groups` | `list[int]` | List of integers representing number of ground truths per image. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[tuple[torch.Tensor, torch.Tensor]]` | List of tuples containing matched indices for denoising. |

<details>
<summary>Source code in <code>ultralytics/models/utils/loss.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/utils/loss.py#L441-L466"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def get_dn_match_indices(
    dn_pos_idx: list[torch.Tensor], dn_num_group: int, gt_groups: list[int]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Get match indices for denoising.

    Args:
        dn_pos_idx (list[torch.Tensor]): List of tensors containing positive indices for denoising.
        dn_num_group (int): Number of denoising groups.
        gt_groups (list[int]): List of integers representing number of ground truths per image.

    Returns:
        (list[tuple[torch.Tensor, torch.Tensor]]): List of tuples containing matched indices for denoising.
    """
    dn_match_indices = []
    idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
    for i, num_gt in enumerate(gt_groups):
        if num_gt > 0:
            gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
            gt_idx = gt_idx.repeat(dn_num_group)
            assert len(dn_pos_idx[i]) == len(gt_idx), (
                f"Expected the same length, but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
            )
            dn_match_indices.append((dn_pos_idx[i], gt_idx))
        else:
            dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
    return dn_match_indices
```
</details>

<br><br>
