---
title: models.sam.sam3.necks API Reference
description: Explore the ultralytics.models.sam.sam3.necks module for SAM3 neck components that connect vision backbones to downstream heads.
keywords: Ultralytics, SAM3, SAM, neck, backbone, ViTDet, segmentation, Python
---

# Reference for `ultralytics/models/sam/sam3/necks.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/necks.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/necks.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`Sam3DualViTDetNeck`](#ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`Sam3DualViTDetNeck.forward`](#ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.forward)
        - [`Sam3DualViTDetNeck.sam_forward_feature_levels`](#ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.sam_forward_feature_levels)
        - [`Sam3DualViTDetNeck.set_imgsz`](#ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.set_imgsz)


## Class `ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck` {#ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck}

```python
Sam3DualViTDetNeck(
    trunk: nn.Module,
    position_encoding: nn.Module,
    d_model: int,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    add_sam2_neck: bool = False,
)
```

**Bases:** `nn.Module`

A neck that implements a simple FPN as in ViTDet, with support for dual necks (for SAM3 and SAM2).

Supports a "dual neck" setting with two identical necks (for SAM3 and SAM2) that have different weights.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trunk` | `nn.Module` | The backbone. | *required* |
| `position_encoding` | `nn.Module` | The positional encoding to use. | *required* |
| `d_model` | `int` | The dimension of the model. | *required* |
| `scale_factors` | `tuple` | Scale factors for each FPN level. | `(4.0, 2.0, 1.0, 0.5)` |
| `add_sam2_neck` | `bool` | Whether to add a second neck for SAM2. | `False` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.forward) | Get feature maps and positional encodings from the neck. |
| [`sam_forward_feature_levels`](#ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.sam_forward_feature_levels) | Run neck convolutions and compute positional encodings for each feature level. |
| [`set_imgsz`](#ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.set_imgsz) | Set the image size for the trunk backbone. |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/necks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/necks.py#L15-L132">View on GitHub</a>
```python
class Sam3DualViTDetNeck(nn.Module):
    """A neck that implements a simple FPN as in ViTDet, with support for dual necks (for SAM3 and SAM2)."""

    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ):
        """SimpleFPN neck a la ViTDet, very lightly adapted from detectron2.

        Supports a "dual neck" setting with two identical necks (for SAM3 and SAM2) that have different weights.

        Args:
            trunk (nn.Module): The backbone.
            position_encoding (nn.Module): The positional encoding to use.
            d_model (int): The dimension of the model.
            scale_factors (tuple): Scale factors for each FPN level.
            add_sam2_neck (bool): Whether to add a second neck for SAM2.
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            # Assumes sam2 neck is just a clone of the original neck
            self.sam2_convs = deepcopy(self.convs)
```
</details>

<br>

### Method `ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.forward` {#ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.forward}

```python
def forward(
    self, tensor_list: list[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor] | None]
```

Get feature maps and positional encodings from the neck.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tensor_list` | `list[torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/necks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/necks.py#L106-L116">View on GitHub</a>
```python
def forward(
    self, tensor_list: list[torch.Tensor]
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor] | None, list[torch.Tensor] | None]:
    """Get feature maps and positional encodings from the neck."""
    xs = self.trunk(tensor_list)
    x = xs[-1]  # simpleFPN
    sam3_out, sam3_pos = self.sam_forward_feature_levels(x, self.convs)
    if self.sam2_convs is None:
        return sam3_out, sam3_pos, None, None
    sam2_out, sam2_pos = self.sam_forward_feature_levels(x, self.sam2_convs)
    return sam3_out, sam3_pos, sam2_out, sam2_pos
```
</details>

<br>

### Method `ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.sam_forward_feature_levels` {#ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.sam\_forward\_feature\_levels}

```python
def sam_forward_feature_levels(
    self, x: torch.Tensor, convs: nn.ModuleList
) -> tuple[list[torch.Tensor], list[torch.Tensor]]
```

Run neck convolutions and compute positional encodings for each feature level.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |
| `convs` | `nn.ModuleList` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/necks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/necks.py#L118-L127">View on GitHub</a>
```python
def sam_forward_feature_levels(
    self, x: torch.Tensor, convs: nn.ModuleList
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run neck convolutions and compute positional encodings for each feature level."""
    outs, poss = [], []
    for conv in convs:
        feat = conv(x)
        outs.append(feat)
        poss.append(self.position_encoding(feat).to(feat.dtype))
    return outs, poss
```
</details>

<br>

### Method `ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.set_imgsz` {#ultralytics.models.sam.sam3.necks.Sam3DualViTDetNeck.set\_imgsz}

```python
def set_imgsz(self, imgsz: list[int] | None = None)
```

Set the image size for the trunk backbone.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `imgsz` | `list[int] \| None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/necks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/necks.py#L129-L132">View on GitHub</a>
```python
def set_imgsz(self, imgsz: list[int] | None = None):
    """Set the image size for the trunk backbone."""
    imgsz = imgsz if imgsz is not None else [1008, 1008]
    self.trunk.set_imgsz(imgsz)
```
</details>

<br><br>
