---
description: Explore the ultralytics.models.sam.sam3.vl_combiner module for combining vision and language features in SAM3.
keywords: Ultralytics, SAM3, vision-language, backbone, feature fusion, transformer, deep learning, Python
---

# Reference for `ultralytics/models/sam/sam3/vl_combiner.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

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
def __init__(
    self,
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

:param visual: The vision backbone to use :param text: The text encoder to use

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `visual` | `Sam3DualViTDetNeck` |  | *required* |
| `text` |  |  | *required* |
| `compile_visual` | `bool` |  | `False` |
| `act_ckpt_whole_vision_backbone` | `bool` |  | `False` |
| `act_ckpt_whole_language_backbone` | `bool` |  | `False` |
| `scalp` |  |  | `0` |

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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L18-L160"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

        :param visual: The vision backbone to use
        :param text: The text encoder to use
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

:param samples: The input images
:param captions: The input captions
:param input_boxes: If the text contains place-holders for boxes, this
    parameter contains the tensor containing their spatial features
:param additional_text: This can be used to encode some additional text
    (different from the captions) in the same forward of the backbone
:return: Output dictionary with the following keys:
    - vision_features: The output of the vision backbone
    - language_features: The output of the language backbone
    - language_mask: The attention mask of the language backbone
    - vision_pos_enc: The positional encoding of the vision backbone
    - (optional) additional_text_features: The output of the language
        backbone for the additional text
    - (optional) additional_text_mask: The attention mask of the
        language backbone for the additional text

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `samples` | `torch.Tensor` |  | *required* |
| `captions` | `list[str]` |  | *required* |
| `input_boxes` | `torch.Tensor` |  | `None` |
| `additional_text` | `list[str] | None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/vl_combiner.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L47-L74"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(
    self,
    samples: torch.Tensor,
    captions: list[str],
    input_boxes: torch.Tensor = None,
    additional_text: list[str] | None = None,
):
    """Forward pass of the backbone combiner.

    :param samples: The input images
    :param captions: The input captions
    :param input_boxes: If the text contains place-holders for boxes, this
        parameter contains the tensor containing their spatial features
    :param additional_text: This can be used to encode some additional text
        (different from the captions) in the same forward of the backbone
    :return: Output dictionary with the following keys:
        - vision_features: The output of the vision backbone
        - language_features: The output of the language backbone
        - language_mask: The attention mask of the language backbone
        - vision_pos_enc: The positional encoding of the vision backbone
        - (optional) additional_text_features: The output of the language
            backbone for the additional text
        - (optional) additional_text_mask: The attention mask of the
            language backbone for the additional text
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L76-L108"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L110-L129"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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
def forward_text(self, captions, input_boxes = None, additional_text = None)
```

Forward pass of the text encoder.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `captions` |  |  | *required* |
| `input_boxes` |  |  | `None` |
| `additional_text` |  |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/vl_combiner.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L131-L156"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
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
def set_imgsz(self, imgsz: list[int] = [1008, 1008])
```

Set the image size for the vision backbone.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `imgsz` | `list[int]` |  | `[1008, 1008]` |

<details>
<summary>Source code in <code>ultralytics/models/sam/sam3/vl_combiner.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/sam3/vl_combiner.py#L158-L160"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def set_imgsz(self, imgsz: list[int] = [1008, 1008]):
    """Set the image size for the vision backbone."""
    self.vision_backbone.set_imgsz(imgsz)
```
</details>

<br><br>
