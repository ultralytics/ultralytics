---
description: Discover detailed instructions for building various Segment Anything Model (SAM) and Segment Anything Model 2 (SAM 2) architectures with Ultralytics, including SAM ViT and Mobile-SAM.
keywords: Ultralytics, SAM model, Segment Anything Model, SAM 2 model, Segment Anything Model 2, SAM ViT, Mobile-SAM, model building, deep learning, AI
---

# Reference for `ultralytics/models/sam/build.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`_load_checkpoint`](#ultralytics.models.sam.build._load_checkpoint)
        - [`build_sam_vit_h`](#ultralytics.models.sam.build.build_sam_vit_h)
        - [`build_sam_vit_l`](#ultralytics.models.sam.build.build_sam_vit_l)
        - [`build_sam_vit_b`](#ultralytics.models.sam.build.build_sam_vit_b)
        - [`build_mobile_sam`](#ultralytics.models.sam.build.build_mobile_sam)
        - [`build_sam2_t`](#ultralytics.models.sam.build.build_sam2_t)
        - [`build_sam2_s`](#ultralytics.models.sam.build.build_sam2_s)
        - [`build_sam2_b`](#ultralytics.models.sam.build.build_sam2_b)
        - [`build_sam2_l`](#ultralytics.models.sam.build.build_sam2_l)
        - [`_build_sam`](#ultralytics.models.sam.build._build_sam)
        - [`_build_sam2`](#ultralytics.models.sam.build._build_sam2)
        - [`build_sam`](#ultralytics.models.sam.build.build_sam)


## Function `ultralytics.models.sam.build._load_checkpoint` {#ultralytics.models.sam.build.\_load\_checkpoint}

```python
def _load_checkpoint(model, checkpoint)
```

Load checkpoint into model from file path.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` |  |  | *required* |
| `checkpoint` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L24-L36"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _load_checkpoint(model, checkpoint):
    """Load checkpoint into model from file path."""
    if checkpoint is None:
        return model

    checkpoint = attempt_download_asset(checkpoint)
    with open(checkpoint, "rb") as f:
        state_dict = torch_load(f)
    # Handle nested "model" key
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    return model
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.build.build_sam_vit_h` {#ultralytics.models.sam.build.build\_sam\_vit\_h}

```python
def build_sam_vit_h(checkpoint = None)
```

Build and return a Segment Anything Model (SAM) h-size model with specified encoder parameters.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `checkpoint` |  |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L39-L47"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_sam_vit_h(checkpoint=None):
    """Build and return a Segment Anything Model (SAM) h-size model with specified encoder parameters."""
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.build.build_sam_vit_l` {#ultralytics.models.sam.build.build\_sam\_vit\_l}

```python
def build_sam_vit_l(checkpoint = None)
```

Build and return a Segment Anything Model (SAM) l-size model with specified encoder parameters.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `checkpoint` |  |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L50-L58"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_sam_vit_l(checkpoint=None):
    """Build and return a Segment Anything Model (SAM) l-size model with specified encoder parameters."""
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.build.build_sam_vit_b` {#ultralytics.models.sam.build.build\_sam\_vit\_b}

```python
def build_sam_vit_b(checkpoint = None)
```

Build and return a Segment Anything Model (SAM) b-size model with specified encoder parameters.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `checkpoint` |  |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L61-L69"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_sam_vit_b(checkpoint=None):
    """Build and return a Segment Anything Model (SAM) b-size model with specified encoder parameters."""
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.build.build_mobile_sam` {#ultralytics.models.sam.build.build\_mobile\_sam}

```python
def build_mobile_sam(checkpoint = None)
```

Build and return a Mobile Segment Anything Model (Mobile-SAM) for efficient image segmentation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `checkpoint` |  |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L72-L81"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_mobile_sam(checkpoint=None):
    """Build and return a Mobile Segment Anything Model (Mobile-SAM) for efficient image segmentation."""
    return _build_sam(
        encoder_embed_dim=[64, 128, 160, 320],
        encoder_depth=[2, 2, 6, 2],
        encoder_num_heads=[2, 4, 5, 10],
        encoder_global_attn_indexes=None,
        mobile_sam=True,
        checkpoint=checkpoint,
    )
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.build.build_sam2_t` {#ultralytics.models.sam.build.build\_sam2\_t}

```python
def build_sam2_t(checkpoint = None)
```

Build and return a Segment Anything Model 2 (SAM2) tiny-size model with specified architecture parameters.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `checkpoint` |  |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L84-L94"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_sam2_t(checkpoint=None):
    """Build and return a Segment Anything Model 2 (SAM2) tiny-size model with specified architecture parameters."""
    return _build_sam2(
        encoder_embed_dim=96,
        encoder_stages=[1, 2, 7, 2],
        encoder_num_heads=1,
        encoder_global_att_blocks=[5, 7, 9],
        encoder_window_spec=[8, 4, 14, 7],
        encoder_backbone_channel_list=[768, 384, 192, 96],
        checkpoint=checkpoint,
    )
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.build.build_sam2_s` {#ultralytics.models.sam.build.build\_sam2\_s}

```python
def build_sam2_s(checkpoint = None)
```

Build and return a small-size Segment Anything Model 2 (SAM2) with specified architecture parameters.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `checkpoint` |  |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L97-L107"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_sam2_s(checkpoint=None):
    """Build and return a small-size Segment Anything Model 2 (SAM2) with specified architecture parameters."""
    return _build_sam2(
        encoder_embed_dim=96,
        encoder_stages=[1, 2, 11, 2],
        encoder_num_heads=1,
        encoder_global_att_blocks=[7, 10, 13],
        encoder_window_spec=[8, 4, 14, 7],
        encoder_backbone_channel_list=[768, 384, 192, 96],
        checkpoint=checkpoint,
    )
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.build.build_sam2_b` {#ultralytics.models.sam.build.build\_sam2\_b}

```python
def build_sam2_b(checkpoint = None)
```

Build and return a Segment Anything Model 2 (SAM2) base-size model with specified architecture parameters.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `checkpoint` |  |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L110-L121"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_sam2_b(checkpoint=None):
    """Build and return a Segment Anything Model 2 (SAM2) base-size model with specified architecture parameters."""
    return _build_sam2(
        encoder_embed_dim=112,
        encoder_stages=[2, 3, 16, 3],
        encoder_num_heads=2,
        encoder_global_att_blocks=[12, 16, 20],
        encoder_window_spec=[8, 4, 14, 7],
        encoder_window_spatial_size=[14, 14],
        encoder_backbone_channel_list=[896, 448, 224, 112],
        checkpoint=checkpoint,
    )
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.build.build_sam2_l` {#ultralytics.models.sam.build.build\_sam2\_l}

```python
def build_sam2_l(checkpoint = None)
```

Build and return a large-size Segment Anything Model 2 (SAM2) with specified architecture parameters.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `checkpoint` |  |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L124-L134"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_sam2_l(checkpoint=None):
    """Build and return a large-size Segment Anything Model 2 (SAM2) with specified architecture parameters."""
    return _build_sam2(
        encoder_embed_dim=144,
        encoder_stages=[2, 6, 36, 4],
        encoder_num_heads=2,
        encoder_global_att_blocks=[23, 33, 43],
        encoder_window_spec=[8, 4, 16, 8],
        encoder_backbone_channel_list=[1152, 576, 288, 144],
        checkpoint=checkpoint,
    )
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.build._build_sam` {#ultralytics.models.sam.build.\_build\_sam}

```python
def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    mobile_sam=False,
)
```

Build a Segment Anything Model (SAM) with specified encoder parameters.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `encoder_embed_dim` | `int | list[int]` | Embedding dimension for the encoder. | *required* |
| `encoder_depth` | `int | list[int]` | Depth of the encoder. | *required* |
| `encoder_num_heads` | `int | list[int]` | Number of attention heads in the encoder. | *required* |
| `encoder_global_attn_indexes` | `list[int] | None` | Indexes for global attention in the encoder. | *required* |
| `checkpoint` | `str | None, optional` | Path to the model checkpoint file. | `None` |
| `mobile_sam` | `bool, optional` | Whether to build a Mobile-SAM model. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `SAMModel` | A Segment Anything Model instance with the specified architecture. |

**Examples**

```python
>>> sam = _build_sam(768, 12, 12, [2, 5, 8, 11])
>>> sam = _build_sam([64, 128, 160, 320], [2, 2, 6, 2], [2, 4, 5, 10], None, mobile_sam=True)
```

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L137-L225"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    mobile_sam=False,
):
    """Build a Segment Anything Model (SAM) with specified encoder parameters.

    Args:
        encoder_embed_dim (int | list[int]): Embedding dimension for the encoder.
        encoder_depth (int | list[int]): Depth of the encoder.
        encoder_num_heads (int | list[int]): Number of attention heads in the encoder.
        encoder_global_attn_indexes (list[int] | None): Indexes for global attention in the encoder.
        checkpoint (str | None, optional): Path to the model checkpoint file.
        mobile_sam (bool, optional): Whether to build a Mobile-SAM model.

    Returns:
        (SAMModel): A Segment Anything Model instance with the specified architecture.

    Examples:
        >>> sam = _build_sam(768, 12, 12, [2, 5, 8, 11])
        >>> sam = _build_sam([64, 128, 160, 320], [2, 2, 6, 2], [2, 4, 5, 10], None, mobile_sam=True)
    """
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_encoder = (
        TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=encoder_embed_dim,
            depths=encoder_depth,
            num_heads=encoder_num_heads,
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        )
        if mobile_sam
        else ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    )
    sam = SAMModel(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if checkpoint is not None:
        sam = _load_checkpoint(sam, checkpoint)
    sam.eval()
    return sam
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.build._build_sam2` {#ultralytics.models.sam.build.\_build\_sam2}

```python
def _build_sam2(
    encoder_embed_dim=1280,
    encoder_stages=(2, 6, 36, 4),
    encoder_num_heads=2,
    encoder_global_att_blocks=(7, 15, 23, 31),
    encoder_backbone_channel_list=(1152, 576, 288, 144),
    encoder_window_spatial_size=(7, 7),
    encoder_window_spec=(8, 4, 16, 8),
    checkpoint=None,
)
```

Build and return a Segment Anything Model 2 (SAM2) with specified architecture parameters.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `encoder_embed_dim` | `int, optional` | Embedding dimension for the encoder. | `1280` |
| `encoder_stages` | `list[int], optional` | Number of blocks in each stage of the encoder. | `(2, 6, 36, 4)` |
| `encoder_num_heads` | `int, optional` | Number of attention heads in the encoder. | `2` |
| `encoder_global_att_blocks` | `list[int], optional` | Indices of global attention blocks in the encoder. | `(7, 15, 23, 31)` |
| `encoder_backbone_channel_list` | `list[int], optional` | Channel dimensions for each level of the encoder backbone. | `(1152, 576, 288, 144)` |
| `encoder_window_spatial_size` | `list[int], optional` | Spatial size of the window for position embeddings. | `(7, 7)` |
| `encoder_window_spec` | `list[int], optional` | Window specifications for each stage of the encoder. | `(8, 4, 16, 8)` |
| `checkpoint` | `str | None, optional` | Path to the checkpoint file for loading pre-trained weights. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `SAM2Model` | A configured and initialized SAM2 model. |

**Examples**

```python
>>> sam2_model = _build_sam2(encoder_embed_dim=96, encoder_stages=[1, 2, 7, 2])
>>> sam2_model.eval()
```

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L228-L316"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _build_sam2(
    encoder_embed_dim=1280,
    encoder_stages=(2, 6, 36, 4),
    encoder_num_heads=2,
    encoder_global_att_blocks=(7, 15, 23, 31),
    encoder_backbone_channel_list=(1152, 576, 288, 144),
    encoder_window_spatial_size=(7, 7),
    encoder_window_spec=(8, 4, 16, 8),
    checkpoint=None,
):
    """Build and return a Segment Anything Model 2 (SAM2) with specified architecture parameters.

    Args:
        encoder_embed_dim (int, optional): Embedding dimension for the encoder.
        encoder_stages (list[int], optional): Number of blocks in each stage of the encoder.
        encoder_num_heads (int, optional): Number of attention heads in the encoder.
        encoder_global_att_blocks (list[int], optional): Indices of global attention blocks in the encoder.
        encoder_backbone_channel_list (list[int], optional): Channel dimensions for each level of the encoder backbone.
        encoder_window_spatial_size (list[int], optional): Spatial size of the window for position embeddings.
        encoder_window_spec (list[int], optional): Window specifications for each stage of the encoder.
        checkpoint (str | None, optional): Path to the checkpoint file for loading pre-trained weights.

    Returns:
        (SAM2Model): A configured and initialized SAM2 model.

    Examples:
        >>> sam2_model = _build_sam2(encoder_embed_dim=96, encoder_stages=[1, 2, 7, 2])
        >>> sam2_model.eval()
    """
    image_encoder = ImageEncoder(
        trunk=Hiera(
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            stages=encoder_stages,
            global_att_blocks=encoder_global_att_blocks,
            window_pos_embed_bkg_spatial_size=encoder_window_spatial_size,
            window_spec=encoder_window_spec,
        ),
        neck=FpnNeck(
            d_model=256,
            backbone_channel_list=encoder_backbone_channel_list,
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        ),
        scalp=1,
    )
    memory_attention = MemoryAttention(d_model=256, pos_enc_at_input=True, num_layers=4, layer=MemoryAttentionLayer())
    memory_encoder = MemoryEncoder(out_dim=64)

    is_sam2_1 = checkpoint is not None and "sam2.1" in checkpoint
    sam2 = SAM2Model(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=7,
        image_size=1024,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        compile_image_encoder=False,
        no_obj_embed_spatial=is_sam2_1,
        proj_tpos_enc_in_obj_ptrs=is_sam2_1,
        use_signed_tpos_enc_to_obj_ptrs=is_sam2_1,
        sam_mask_decoder_extra_args=dict(
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
        ),
    )

    if checkpoint is not None:
        sam2 = _load_checkpoint(sam2, checkpoint)
    sam2.eval()
    return sam2
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.build.build_sam` {#ultralytics.models.sam.build.build\_sam}

```python
def build_sam(ckpt = "sam_b.pt")
```

Build and return a Segment Anything Model (SAM) based on the provided checkpoint.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `ckpt` | `str | Path, optional` | Path to the checkpoint file or name of a pre-defined SAM model. | `"sam_b.pt"` |

**Returns**

| Type | Description |
| --- | --- |
| `SAMModel | SAM2Model` | A configured and initialized SAM or SAM2 model instance. |

**Examples**

```python
>>> sam_model = build_sam("sam_b.pt")
>>> sam_model = build_sam("path/to/custom_checkpoint.pt")
```

!!! note "Notes"

    Supported pre-defined models include:
    - SAM: 'sam_h.pt', 'sam_l.pt', 'sam_b.pt', 'mobile_sam.pt'
    - SAM2: 'sam2_t.pt', 'sam2_s.pt', 'sam2_b.pt', 'sam2_l.pt'

**Raises**

| Type | Description |
| --- | --- |
| `FileNotFoundError` | If the provided checkpoint is not a supported SAM model. |

<details>
<summary>Source code in <code>ultralytics/models/sam/build.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/build.py#L335-L365"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_sam(ckpt="sam_b.pt"):
    """Build and return a Segment Anything Model (SAM) based on the provided checkpoint.

    Args:
        ckpt (str | Path, optional): Path to the checkpoint file or name of a pre-defined SAM model.

    Returns:
        (SAMModel | SAM2Model): A configured and initialized SAM or SAM2 model instance.

    Raises:
        FileNotFoundError: If the provided checkpoint is not a supported SAM model.

    Examples:
        >>> sam_model = build_sam("sam_b.pt")
        >>> sam_model = build_sam("path/to/custom_checkpoint.pt")

    Notes:
        Supported pre-defined models include:
        - SAM: 'sam_h.pt', 'sam_l.pt', 'sam_b.pt', 'mobile_sam.pt'
        - SAM2: 'sam2_t.pt', 'sam2_s.pt', 'sam2_b.pt', 'sam2_l.pt'
    """
    model_builder = None
    ckpt = str(ckpt)  # to allow Path ckpt types
    for k in sam_model_map.keys():
        if ckpt.endswith(k):
            model_builder = sam_model_map.get(k)

    if not model_builder:
        raise FileNotFoundError(f"{ckpt} is not a supported SAM model. Available models are: \n {sam_model_map.keys()}")

    return model_builder(ckpt)
```
</details>

<br><br>
