---
title: models.sam.sam3.vl_combiner API Reference
description: Explore the ultralytics.models.sam.sam3.vl_combiner module for combining vision and language features in SAM3.
keywords: Ultralytics, SAM3, vision-language, backbone, feature fusion, transformer, deep learning, Python
---

# Reference for `ultralytics/models/sam/sam3/vl_combiner.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`SAM3VLBackbone`](#ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`SAM3VLBackbone.forward`](#ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward)
        - [`SAM3VLBackbone.forward_image`](#ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward_image)
        - [`SAM3VLBackbone.forward_image_sam2`](#ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward_image_sam2)
        - [`SAM3VLBackbone.forward_text`](#ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward_text)
        - [`SAM3VLBackbone.set_imgsz`](#ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.set_imgsz)


## Class `ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone` {#ultralytics.models.sam.sam3.vl\_combiner.SAM3VLBackbone}

```python
SAM3VLBackbone(
    visual: Sam3DualViTDetNeck,
    text,
    compile_visual: bool = False,
    act_ckpt_whole_vision_backbone: bool = False,
    act_ckpt_whole_language_backbone: bool = False,
    scalp=0,
)
```

**Bases:** `nn.Module`

This backbone combines a vision backbone and a language backbone without fusion. As such it is more of a

convenience wrapper to handle the two backbones together.

It adds support for activation checkpointing and compilation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `visual` | `Sam3DualViTDetNeck` | The vision backbone to use. | *required* |
| `text` | `nn.Module` | The text encoder to use. | *required* |
| `compile_visual` | `bool` | Whether to `torch.compile` the vision backbone. | `False` |
| `act_ckpt_whole_vision_backbone` | `bool` | Whether to checkpoint activations for the whole vision backbone. | `False` |
| `act_ckpt_whole_language_backbone` | `bool` | Whether to checkpoint activations for the whole language backbone. | `False` |
| `scalp` | `int` | Number of trailing (lowest-resolution) feature levels to drop from the backbone output. | `0` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward) | Forward pass of the backbone combiner. |
| [`forward_image`](#ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward_image) | Forward pass of the vision backbone and get both SAM3 and SAM2 features. |
| [`forward_image_sam2`](#ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward_image_sam2) | Forward pass of the vision backbone to get SAM2 features only. |
| [`forward_text`](#ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward_text) | Forward pass of the text encoder. |
| [`set_imgsz`](#ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.set_imgsz) | Set the image size for the vision backbone. |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/vl_combiner.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L18-L165">View on GitHub</a>
```python
class SAM3VLBackbone(nn.Module):
    """This backbone combines a vision backbone and a language backbone without fusion. As such it is more of a
    convenience wrapper to handle the two backbones together.

    It adds support for activation checkpointing and compilation.
    """

    def __init__(
        self,
        visual: Sam3DualViTDetNeck,
        text,
        compile_visual: bool = False,
        act_ckpt_whole_vision_backbone: bool = False,
        act_ckpt_whole_language_backbone: bool = False,
        scalp=0,
    ):
        """Initialize the backbone combiner.

        Args:
            visual (Sam3DualViTDetNeck): The vision backbone to use.
            text (nn.Module): The text encoder to use.
            compile_visual (bool): Whether to `torch.compile` the vision backbone.
            act_ckpt_whole_vision_backbone (bool): Whether to checkpoint activations for the whole vision backbone.
            act_ckpt_whole_language_backbone (bool): Whether to checkpoint activations for the whole language backbone.
            scalp (int): Number of trailing (lowest-resolution) feature levels to drop from the backbone output.
        """
        super().__init__()
        self.vision_backbone: Sam3DualViTDetNeck = torch.compile(visual) if compile_visual else visual
        self.language_backbone = text
        self.scalp = scalp
        # allow running activation checkpointing on the entire vision and language backbones
        self.act_ckpt_whole_vision_backbone = act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = act_ckpt_whole_language_backbone
```
</details>

<br>

### Method `ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward` {#ultralytics.models.sam.sam3.vl\_combiner.SAM3VLBackbone.forward}

```python
def forward(
    self,
    samples: torch.Tensor,
    captions: list[str],
    input_boxes: torch.Tensor = None,
    additional_text: list[str] | None = None,
)
```

Forward pass of the backbone combiner.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `samples` | `torch.Tensor` | The input images. | *required* |
| `captions` | `list[str]` | The input captions. | *required* |
| `input_boxes` | `torch.Tensor, optional` | When the text contains box place-holders, the tensor containing their spatial features. | `None` |
| `additional_text` | `list[str], optional` | Extra text (different from the captions) to encode in the same backbone forward pass. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Output dictionary with the following keys: `vision_features` (the output of the vision backbone), `language_features` (the output of the language backbone), `language_mask` (the attention mask of the language backbone), `vision_pos_enc` (the positional encoding of the vision backbone) and, when `additional_text` is provided, `additional_text_features` and `additional_text_mask` (the language backbone output and attention mask for the additional text). |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/vl_combiner.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L52-L78">View on GitHub</a>
```python
def forward(
    self,
    samples: torch.Tensor,
    captions: list[str],
    input_boxes: torch.Tensor = None,
    additional_text: list[str] | None = None,
):
    """Forward pass of the backbone combiner.

    Args:
        samples (torch.Tensor): The input images.
        captions (list[str]): The input captions.
        input_boxes (torch.Tensor, optional): When the text contains box place-holders, the tensor containing their
            spatial features.
        additional_text (list[str], optional): Extra text (different from the captions) to encode in the same
            backbone forward pass.

    Returns:
        (dict): Output dictionary with the following keys: `vision_features` (the output of the vision backbone),
            `language_features` (the output of the language backbone), `language_mask` (the attention mask of the
            language backbone), `vision_pos_enc` (the positional encoding of the vision
            backbone) and, when `additional_text` is provided, `additional_text_features` and
            `additional_text_mask` (the language backbone output and attention mask for the additional text).
    """
    output = self.forward_image(samples)
    output.update(self.forward_text(captions, input_boxes, additional_text))
    return output
```
</details>

<br>

### Method `ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward_image` {#ultralytics.models.sam.sam3.vl\_combiner.SAM3VLBackbone.forward\_image}

```python
def forward_image(self, samples: torch.Tensor)
```

Forward pass of the vision backbone and get both SAM3 and SAM2 features.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `samples` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/vl_combiner.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L80-L112">View on GitHub</a>
```python
def forward_image(self, samples: torch.Tensor):
    """Forward pass of the vision backbone and get both SAM3 and SAM2 features."""
    # Forward through backbone
    sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(samples)
    if self.scalp > 0:
        # Discard the lowest resolution features
        sam3_features, sam3_pos = (
            sam3_features[: -self.scalp],
            sam3_pos[: -self.scalp],
        )
        if sam2_features is not None and sam2_pos is not None:
            sam2_features, sam2_pos = (
                sam2_features[: -self.scalp],
                sam2_pos[: -self.scalp],
            )

    sam2_output = None

    if sam2_features is not None and sam2_pos is not None:
        sam2_src = sam2_features[-1]
        sam2_output = {
            "vision_features": sam2_src,
            "vision_pos_enc": sam2_pos,
            "backbone_fpn": sam2_features,
        }

    sam3_src = sam3_features[-1]
    return {
        "vision_features": sam3_src,
        "vision_pos_enc": sam3_pos,
        "backbone_fpn": sam3_features,
        "sam2_backbone_out": sam2_output,
    }
```
</details>

<br>

### Method `ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward_image_sam2` {#ultralytics.models.sam.sam3.vl\_combiner.SAM3VLBackbone.forward\_image\_sam2}

```python
def forward_image_sam2(self, samples: torch.Tensor)
```

Forward pass of the vision backbone to get SAM2 features only.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `samples` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/vl_combiner.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L114-L133">View on GitHub</a>
```python
def forward_image_sam2(self, samples: torch.Tensor):
    """Forward pass of the vision backbone to get SAM2 features only."""
    xs = self.vision_backbone.trunk(samples)
    x = xs[-1]  # simpleFPN

    assert self.vision_backbone.sam2_convs is not None, "SAM2 neck is not available."
    sam2_features, sam2_pos = self.vision_backbone.sam_forward_feature_levels(x, self.vision_backbone.sam2_convs)

    if self.scalp > 0:
        # Discard the lowest resolution features
        sam2_features, sam2_pos = (
            sam2_features[: -self.scalp],
            sam2_pos[: -self.scalp],
        )

    return {
        "vision_features": sam2_features[-1],
        "vision_pos_enc": sam2_pos,
        "backbone_fpn": sam2_features,
    }
```
</details>

<br>

### Method `ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.forward_text` {#ultralytics.models.sam.sam3.vl\_combiner.SAM3VLBackbone.forward\_text}

```python
def forward_text(self, captions, input_boxes=None, additional_text=None)
```

Forward pass of the text encoder.

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/vl_combiner.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L135-L160">View on GitHub</a>
```python
def forward_text(self, captions, input_boxes=None, additional_text=None):
    """Forward pass of the text encoder."""
    output = {}

    # Forward through text_encoder
    text_to_encode = copy(captions)
    if additional_text is not None:
        # if there are additional_text, we piggy-back them into this forward.
        # They'll be used later for output alignment
        text_to_encode += additional_text

    with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
        text_attention_mask, text_memory, text_embeds = self.language_backbone(text_to_encode, input_boxes)

    if additional_text is not None:
        output["additional_text_features"] = text_memory[:, -len(additional_text) :]
        output["additional_text_mask"] = text_attention_mask[-len(additional_text) :]

    text_memory = text_memory[:, : len(captions)]
    text_attention_mask = text_attention_mask[: len(captions)]
    text_embeds = text_embeds[:, : len(captions)]
    output["language_features"] = text_memory
    output["language_mask"] = text_attention_mask
    output["language_embeds"] = text_embeds  # Text embeddings before forward to the encoder

    return output
```
</details>

<br>

### Method `ultralytics.models.sam.sam3.vl_combiner.SAM3VLBackbone.set_imgsz` {#ultralytics.models.sam.sam3.vl\_combiner.SAM3VLBackbone.set\_imgsz}

```python
def set_imgsz(self, imgsz: list[int] | None = None)
```

Set the image size for the vision backbone.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `imgsz` | `list[int] \| None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/vl_combiner.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L162-L165">View on GitHub</a>
```python
def set_imgsz(self, imgsz: list[int] | None = None):
    """Set the image size for the vision backbone."""
    imgsz = imgsz if imgsz is not None else [1008, 1008]
    self.vision_backbone.set_imgsz(imgsz)
```
</details>

<br><br>
