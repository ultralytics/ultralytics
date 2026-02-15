---
description: Explore detailed documentation of various SAM and SAM 2 modules such as MaskDownSampler, CXBlock, and more, available in Ultralytics' repository.
keywords: Ultralytics, SAM encoder, SAM 2 encoder, DropPath, MaskDownSampler, CXBlock, Fuser, TwoWayTransformer, TwoWayAttentionBlock, RoPEAttention, MultiScaleAttention, MultiScaleBlock, PositionEmbeddingSine, do_pool
---

# Reference for `ultralytics/models/sam/modules/blocks.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DropPath`](#ultralytics.models.sam.modules.blocks.DropPath)
        - [`MaskDownSampler`](#ultralytics.models.sam.modules.blocks.MaskDownSampler)
        - [`CXBlock`](#ultralytics.models.sam.modules.blocks.CXBlock)
        - [`Fuser`](#ultralytics.models.sam.modules.blocks.Fuser)
        - [`SAM2TwoWayAttentionBlock`](#ultralytics.models.sam.modules.blocks.SAM2TwoWayAttentionBlock)
        - [`SAM2TwoWayTransformer`](#ultralytics.models.sam.modules.blocks.SAM2TwoWayTransformer)
        - [`RoPEAttention`](#ultralytics.models.sam.modules.blocks.RoPEAttention)
        - [`MultiScaleAttention`](#ultralytics.models.sam.modules.blocks.MultiScaleAttention)
        - [`MultiScaleBlock`](#ultralytics.models.sam.modules.blocks.MultiScaleBlock)
        - [`PositionEmbeddingSine`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine)
        - [`PositionEmbeddingRandom`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom)
        - [`Block`](#ultralytics.models.sam.modules.blocks.Block)
        - [`REAttention`](#ultralytics.models.sam.modules.blocks.REAttention)
        - [`PatchEmbed`](#ultralytics.models.sam.modules.blocks.PatchEmbed)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DropPath.forward`](#ultralytics.models.sam.modules.blocks.DropPath.forward)
        - [`MaskDownSampler.forward`](#ultralytics.models.sam.modules.blocks.MaskDownSampler.forward)
        - [`CXBlock.forward`](#ultralytics.models.sam.modules.blocks.CXBlock.forward)
        - [`Fuser.forward`](#ultralytics.models.sam.modules.blocks.Fuser.forward)
        - [`RoPEAttention.forward`](#ultralytics.models.sam.modules.blocks.RoPEAttention.forward)
        - [`MultiScaleAttention.forward`](#ultralytics.models.sam.modules.blocks.MultiScaleAttention.forward)
        - [`MultiScaleBlock.forward`](#ultralytics.models.sam.modules.blocks.MultiScaleBlock.forward)
        - [`PositionEmbeddingSine._encode_xy`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine._encode_xy)
        - [`PositionEmbeddingSine.encode_boxes`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.encode_boxes)
        - [`PositionEmbeddingSine.encode_points`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.encode_points)
        - [`PositionEmbeddingSine.forward`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.forward)
        - [`PositionEmbeddingRandom._pe_encoding`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom._pe_encoding)
        - [`PositionEmbeddingRandom.forward`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom.forward)
        - [`PositionEmbeddingRandom.forward_with_coords`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom.forward_with_coords)
        - [`Block.forward`](#ultralytics.models.sam.modules.blocks.Block.forward)
        - [`REAttention.forward`](#ultralytics.models.sam.modules.blocks.REAttention.forward)
        - [`PatchEmbed.forward`](#ultralytics.models.sam.modules.blocks.PatchEmbed.forward)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`do_pool`](#ultralytics.models.sam.modules.blocks.do_pool)


## Class `ultralytics.models.sam.modules.blocks.DropPath` {#ultralytics.models.sam.modules.blocks.DropPath}

```python
DropPath(self, drop_prob: float = 0.0, scale_by_keep: bool = True)
```

**Bases:** `nn.Module`

Implements stochastic depth regularization for neural networks during training.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `drop_prob` | `float` |  | `0.0` |
| `scale_by_keep` | `bool` |  | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `drop_prob` | `float` | Probability of dropping a path during training. |
| `scale_by_keep` | `bool` | Whether to scale the output by the keep probability. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.blocks.DropPath.forward) | Apply stochastic depth to input tensor during training, with optional scaling. |

**Examples**

```python
>>> drop_path = DropPath(drop_prob=0.2, scale_by_keep=True)
>>> x = torch.randn(32, 64, 224, 224)
>>> output = drop_path(x)
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L19-L50"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class DropPath(nn.Module):
    """Implements stochastic depth regularization for neural networks during training.

    Attributes:
        drop_prob (float): Probability of dropping a path during training.
        scale_by_keep (bool): Whether to scale the output by the keep probability.

    Methods:
        forward: Applies stochastic depth to input tensor during training, with optional scaling.

    Examples:
        >>> drop_path = DropPath(drop_prob=0.2, scale_by_keep=True)
        >>> x = torch.randn(32, 64, 224, 224)
        >>> output = drop_path(x)
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """Initialize DropPath module for stochastic depth regularization during training."""
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.DropPath.forward` {#ultralytics.models.sam.modules.blocks.DropPath.forward}

```python
def forward(self, x: Tensor) -> Tensor
```

Apply stochastic depth to input tensor during training, with optional scaling.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L41-L50"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: Tensor) -> Tensor:
    """Apply stochastic depth to input tensor during training, with optional scaling."""
    if self.drop_prob == 0.0 or not self.training:
        return x
    keep_prob = 1 - self.drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and self.scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.MaskDownSampler` {#ultralytics.models.sam.modules.blocks.MaskDownSampler}

```python
def __init__(
    self,
    embed_dim: int = 256,
    kernel_size: int = 4,
    stride: int = 4,
    padding: int = 0,
    total_stride: int = 16,
    activation: type[nn.Module] = nn.GELU,
    interpol_size: tuple[int, int] | None = None,
)
```

**Bases:** `nn.Module`

A mask downsampling and embedding module for efficient processing of input masks.

This class implements a mask downsampler that progressively reduces the spatial dimensions of input masks while expanding their channel dimensions using convolutional layers, layer normalization, and activation functions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `embed_dim` | `int` |  | `256` |
| `kernel_size` | `int` |  | `4` |
| `stride` | `int` |  | `4` |
| `padding` | `int` |  | `0` |
| `total_stride` | `int` |  | `16` |
| `activation` | `type[nn.Module]` |  | `nn.GELU` |
| `interpol_size` | `tuple[int, int] | None` |  | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `encoder` | `nn.Sequential` | A sequential container of convolutional layers, layer normalization, and activation<br>    functions for downsampling and embedding masks. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.blocks.MaskDownSampler.forward) | Downsample and encode input mask to embed_dim channels using convolutional layers and LayerNorm2d. |

**Examples**

```python
>>> mask_downsampler = MaskDownSampler(embed_dim=256, kernel_size=4, stride=4, padding=0, total_stride=16)
>>> input_mask = torch.randn(1, 1, 256, 256)
>>> output = mask_downsampler(input_mask)
>>> print(output.shape)
torch.Size([1, 256, 16, 16])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L53-L124"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class MaskDownSampler(nn.Module):
    """A mask downsampling and embedding module for efficient processing of input masks.

    This class implements a mask downsampler that progressively reduces the spatial dimensions of input masks while
    expanding their channel dimensions using convolutional layers, layer normalization, and activation functions.

    Attributes:
        encoder (nn.Sequential): A sequential container of convolutional layers, layer normalization, and activation
            functions for downsampling and embedding masks.

    Methods:
        forward: Downsamples and encodes input mask to embed_dim channels.

    Examples:
        >>> mask_downsampler = MaskDownSampler(embed_dim=256, kernel_size=4, stride=4, padding=0, total_stride=16)
        >>> input_mask = torch.randn(1, 1, 256, 256)
        >>> output = mask_downsampler(input_mask)
        >>> print(output.shape)
        torch.Size([1, 256, 16, 16])
    """

    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        total_stride: int = 16,
        activation: type[nn.Module] = nn.GELU,
        interpol_size: tuple[int, int] | None = None,
    ):
        """Initialize a mask downsampler module for progressive downsampling and channel expansion."""
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.append(
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append(activation())
            mask_in_chans = mask_out_chans

        self.encoder.append(nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1))
        self.interpol_size = interpol_size
        if self.interpol_size is not None:
            assert isinstance(self.interpol_size, (list, tuple)), (
                f"Unsupported type {type(self.interpol_size)}. Should be a list or tuple."
            )
            self.interpol_size = list(interpol_size)
            assert len(self.interpol_size) == 2
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.MaskDownSampler.forward` {#ultralytics.models.sam.modules.blocks.MaskDownSampler.forward}

```python
def forward(self, x: Tensor) -> Tensor
```

Downsample and encode input mask to embed_dim channels using convolutional layers and LayerNorm2d.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L114-L124"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: Tensor) -> Tensor:
    """Downsample and encode input mask to embed_dim channels using convolutional layers and LayerNorm2d."""
    if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
        x = F.interpolate(
            x.float(),
            size=self.interpol_size,
            align_corners=False,
            mode="bilinear",
            antialias=True,
        ).to(x.dtype)
    return self.encoder(x)
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.CXBlock` {#ultralytics.models.sam.modules.blocks.CXBlock}

```python
def __init__(
    self,
    dim: int,
    kernel_size: int = 7,
    padding: int = 3,
    drop_path: float = 0.0,
    layer_scale_init_value: float = 1e-6,
    use_dwconv: bool = True,
)
```

**Bases:** `nn.Module`

ConvNeXt Block for efficient feature extraction in convolutional neural networks.

This block implements a modified version of the ConvNeXt architecture, offering improved performance and flexibility in feature extraction.

This block implements a modified version of the ConvNeXt architecture, offering improved performance and flexibility in feature extraction.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dim` | `int` | Number of input channels. | *required* |
| `kernel_size` | `int` | Size of the convolutional kernel. | `7` |
| `padding` | `int` | Padding size for the convolution. | `3` |
| `drop_path` | `float` | Stochastic depth rate. | `0.0` |
| `layer_scale_init_value` | `float` | Initial value for Layer Scale. | `1e-6` |
| `use_dwconv` | `bool` | Whether to use depthwise convolution. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `dwconv` | `nn.Conv2d` | Depthwise or standard 2D convolution layer. |
| `norm` | `LayerNorm2d` | Layer normalization applied to channels. |
| `pwconv1` | `nn.Linear` | First pointwise convolution implemented as a linear layer. |
| `act` | `nn.GELU` | GELU activation function. |
| `pwconv2` | `nn.Linear` | Second pointwise convolution implemented as a linear layer. |
| `gamma` | `nn.Parameter | None` | Learnable scale parameter for layer scaling. |
| `drop_path` | `nn.Module` | DropPath layer for stochastic depth regularization. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.blocks.CXBlock.forward) | Apply ConvNeXt block operations to input tensor, including convolutions and residual connection. |

**Examples**

```python
>>> import torch
>>> x = torch.randn(1, 64, 56, 56)
>>> block = CXBlock(dim=64, kernel_size=7, padding=3)
>>> output = block(x)
>>> print(output.shape)
torch.Size([1, 64, 56, 56])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L127-L209"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class CXBlock(nn.Module):
    """ConvNeXt Block for efficient feature extraction in convolutional neural networks.

    This block implements a modified version of the ConvNeXt architecture, offering improved performance and flexibility
    in feature extraction.

    Attributes:
        dwconv (nn.Conv2d): Depthwise or standard 2D convolution layer.
        norm (LayerNorm2d): Layer normalization applied to channels.
        pwconv1 (nn.Linear): First pointwise convolution implemented as a linear layer.
        act (nn.GELU): GELU activation function.
        pwconv2 (nn.Linear): Second pointwise convolution implemented as a linear layer.
        gamma (nn.Parameter | None): Learnable scale parameter for layer scaling.
        drop_path (nn.Module): DropPath layer for stochastic depth regularization.

    Methods:
        forward: Processes the input tensor through the ConvNeXt block.

    Examples:
        >>> import torch
        >>> x = torch.randn(1, 64, 56, 56)
        >>> block = CXBlock(dim=64, kernel_size=7, padding=3)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        padding: int = 3,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_dwconv: bool = True,
    ):
        """Initialize a ConvNeXt Block for efficient feature extraction in convolutional neural networks.

        This block implements a modified version of the ConvNeXt architecture, offering improved performance and
        flexibility in feature extraction.

        Args:
            dim (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding size for the convolution.
            drop_path (float): Stochastic depth rate.
            layer_scale_init_value (float): Initial value for Layer Scale.
            use_dwconv (bool): Whether to use depthwise convolution.
        """
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.CXBlock.forward` {#ultralytics.models.sam.modules.blocks.CXBlock.forward}

```python
def forward(self, x: Tensor) -> Tensor
```

Apply ConvNeXt block operations to input tensor, including convolutions and residual connection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L195-L209"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: Tensor) -> Tensor:
    """Apply ConvNeXt block operations to input tensor, including convolutions and residual connection."""
    input = x
    x = self.dwconv(x)
    x = self.norm(x)
    x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    x = self.pwconv1(x)
    x = self.act(x)
    x = self.pwconv2(x)
    if self.gamma is not None:
        x = self.gamma * x
    x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    x = input + self.drop_path(x)
    return x
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.Fuser` {#ultralytics.models.sam.modules.blocks.Fuser}

```python
Fuser(self, layer: nn.Module, num_layers: int, dim: int | None = None, input_projection: bool = False)
```

**Bases:** `nn.Module`

A module for fusing features through multiple layers of a neural network.

This class applies a series of identical layers to an input tensor, optionally projecting the input first.

This module creates a sequence of identical layers and optionally applies an input projection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `layer` | `nn.Module` | The layer to be replicated in the fuser. | *required* |
| `num_layers` | `int` | The number of times to replicate the layer. | *required* |
| `dim` | `int | None` | The dimension for input projection, if used. | `None` |
| `input_projection` | `bool` | Whether to use input projection. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `proj` | `nn.Module` | An optional input projection layer. Identity if no projection is needed. |
| `layers` | `nn.ModuleList` | A list of identical layers to be applied sequentially. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.blocks.Fuser.forward) | Apply a series of layers to the input tensor, optionally projecting it first. |

**Examples**

```python
>>> layer = CXBlock(dim=256)
>>> fuser = Fuser(layer, num_layers=3, dim=256, input_projection=True)
>>> x = torch.randn(1, 256, 32, 32)
>>> output = fuser(x)
>>> print(output.shape)
torch.Size([1, 256, 32, 32])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L212-L257"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Fuser(nn.Module):
    """A module for fusing features through multiple layers of a neural network.

    This class applies a series of identical layers to an input tensor, optionally projecting the input first.

    Attributes:
        proj (nn.Module): An optional input projection layer. Identity if no projection is needed.
        layers (nn.ModuleList): A list of identical layers to be applied sequentially.

    Methods:
        forward: Applies the fuser to an input tensor.

    Examples:
        >>> layer = CXBlock(dim=256)
        >>> fuser = Fuser(layer, num_layers=3, dim=256, input_projection=True)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = fuser(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, layer: nn.Module, num_layers: int, dim: int | None = None, input_projection: bool = False):
        """Initialize the Fuser module for feature fusion through multiple layers.

        This module creates a sequence of identical layers and optionally applies an input projection.

        Args:
            layer (nn.Module): The layer to be replicated in the fuser.
            num_layers (int): The number of times to replicate the layer.
            dim (int | None): The dimension for input projection, if used.
            input_projection (bool): Whether to use input projection.
        """
        super().__init__()
        self.proj = nn.Identity()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

        if input_projection:
            assert dim is not None
            self.proj = nn.Conv2d(dim, dim, kernel_size=1)
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.Fuser.forward` {#ultralytics.models.sam.modules.blocks.Fuser.forward}

```python
def forward(self, x: Tensor) -> Tensor
```

Apply a series of layers to the input tensor, optionally projecting it first.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L252-L257"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: Tensor) -> Tensor:
    """Apply a series of layers to the input tensor, optionally projecting it first."""
    x = self.proj(x)
    for layer in self.layers:
        x = layer(x)
    return x
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.SAM2TwoWayAttentionBlock` {#ultralytics.models.sam.modules.blocks.SAM2TwoWayAttentionBlock}

```python
def __init__(
    self,
    embedding_dim: int,
    num_heads: int,
    mlp_dim: int = 2048,
    activation: type[nn.Module] = nn.ReLU,
    attention_downsample_rate: int = 2,
    skip_first_layer_pe: bool = False,
) -> None
```

**Bases:** `TwoWayAttentionBlock`

A two-way attention block for performing self-attention and cross-attention in both directions.

This block extends the TwoWayAttentionBlock and consists of four main components: self-attention on sparse inputs, cross-attention from sparse to dense inputs, an MLP block on sparse inputs, and cross-attention from dense to sparse inputs.

This block extends the TwoWayAttentionBlock and consists of four main components: self-attention on sparse inputs, cross-attention from sparse to dense inputs, an MLP block on sparse inputs, and cross-attention from dense to sparse inputs.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `embedding_dim` | `int` | The channel dimension of the embeddings. | *required* |
| `num_heads` | `int` | The number of heads in the attention layers. | *required* |
| `mlp_dim` | `int` | The hidden dimension of the MLP block. | `2048` |
| `activation` | `type[nn.Module]` | The activation function of the MLP block. | `nn.ReLU` |
| `attention_downsample_rate` | `int` | The downsample rate for attention computations. | `2` |
| `skip_first_layer_pe` | `bool` | Whether to skip the positional encoding in the first layer. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `self_attn` | `Attention` | Self-attention layer for queries. |
| `norm1` | `nn.LayerNorm` | Layer normalization after the first attention block. |
| `cross_attn_token_to_image` | `Attention` | Cross-attention layer from queries to keys. |
| `norm2` | `nn.LayerNorm` | Layer normalization after the second attention block. |
| `mlp` | `MLP` | MLP block for transforming query embeddings. |
| `norm3` | `nn.LayerNorm` | Layer normalization after the MLP block. |
| `norm4` | `nn.LayerNorm` | Layer normalization after the third attention block. |
| `cross_attn_image_to_token` | `Attention` | Cross-attention layer from keys to queries. |
| `skip_first_layer_pe` | `bool` | Flag to skip positional encoding in the first layer. |

**Examples**

```python
>>> block = SAM2TwoWayAttentionBlock(embedding_dim=256, num_heads=8)
>>> sparse_input = torch.randn(1, 100, 256)
>>> dense_input = torch.randn(1, 256, 16, 16)
>>> sparse_output, dense_output = block(sparse_input, dense_input)
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L260-L312"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SAM2TwoWayAttentionBlock(TwoWayAttentionBlock):
    """A two-way attention block for performing self-attention and cross-attention in both directions.

    This block extends the TwoWayAttentionBlock and consists of four main components: self-attention on sparse inputs,
    cross-attention from sparse to dense inputs, an MLP block on sparse inputs, and cross-attention from dense to sparse
    inputs.

    Attributes:
        self_attn (Attention): Self-attention layer for queries.
        norm1 (nn.LayerNorm): Layer normalization after the first attention block.
        cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization after the second attention block.
        mlp (MLP): MLP block for transforming query embeddings.
        norm3 (nn.LayerNorm): Layer normalization after the MLP block.
        norm4 (nn.LayerNorm): Layer normalization after the third attention block.
        cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
        skip_first_layer_pe (bool): Flag to skip positional encoding in the first layer.

    Methods:
        forward: Processes input through the attention blocks and MLP.

    Examples:
        >>> block = SAM2TwoWayAttentionBlock(embedding_dim=256, num_heads=8)
        >>> sparse_input = torch.randn(1, 100, 256)
        >>> dense_input = torch.randn(1, 256, 16, 16)
        >>> sparse_output, dense_output = block(sparse_input, dense_input)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """Initialize a SAM2TwoWayAttentionBlock for performing self-attention and cross-attention in two directions.

        This block extends the TwoWayAttentionBlock and consists of four main components: self-attention on sparse
        inputs, cross-attention from sparse to dense inputs, an MLP block on sparse inputs, and cross-attention from
        dense to sparse inputs.

        Args:
            embedding_dim (int): The channel dimension of the embeddings.
            num_heads (int): The number of heads in the attention layers.
            mlp_dim (int): The hidden dimension of the MLP block.
            activation (type[nn.Module]): The activation function of the MLP block.
            attention_downsample_rate (int): The downsample rate for attention computations.
            skip_first_layer_pe (bool): Whether to skip the positional encoding in the first layer.
        """
        super().__init__(embedding_dim, num_heads, mlp_dim, activation, attention_downsample_rate, skip_first_layer_pe)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, act=activation)
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.SAM2TwoWayTransformer` {#ultralytics.models.sam.modules.blocks.SAM2TwoWayTransformer}

```python
def __init__(
    self,
    depth: int,
    embedding_dim: int,
    num_heads: int,
    mlp_dim: int,
    activation: type[nn.Module] = nn.ReLU,
    attention_downsample_rate: int = 2,
) -> None
```

**Bases:** `TwoWayTransformer`

A Two-Way Transformer module for simultaneous attention to image and query points.

This class extends the TwoWayTransformer, implementing a specialized transformer decoder that attends to an input image using queries with supplied positional embeddings. It is particularly useful for tasks like object detection, image segmentation, and point cloud processing.

This transformer decoder attends to an input image using queries with supplied positional embeddings. It is designed for tasks like object detection, image segmentation, and point cloud processing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `depth` | `int` | Number of layers in the transformer. | *required* |
| `embedding_dim` | `int` | Channel dimension for the input embeddings. | *required* |
| `num_heads` | `int` | Number of heads for multihead attention. Must divide embedding_dim. | *required* |
| `mlp_dim` | `int` | Channel dimension internal to the MLP block. | *required* |
| `activation` | `type[nn.Module]` | Activation function to use in the MLP block. | `nn.ReLU` |
| `attention_downsample_rate` | `int` | Downsampling rate for attention computations. | `2` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `depth` | `int` | Number of layers in the transformer. |
| `embedding_dim` | `int` | Channel dimension for input embeddings. |
| `num_heads` | `int` | Number of heads for multihead attention. |
| `mlp_dim` | `int` | Internal channel dimension for the MLP block. |
| `layers` | `nn.ModuleList` | List of SAM2TwoWayAttentionBlock layers comprising the transformer. |
| `final_attn_token_to_image` | `Attention` | Final attention layer from queries to image. |
| `norm_final_attn` | `nn.LayerNorm` | Layer normalization applied to final queries. |

**Examples**

```python
>>> transformer = SAM2TwoWayTransformer(depth=5, embedding_dim=256, num_heads=8, mlp_dim=2048)
>>> image_embedding = torch.randn(1, 256, 64, 64)
>>> query_embedding = torch.randn(1, 100, 256)
>>> output = transformer(image_embedding, query_embedding)
>>> print(output[0].shape, output[1].shape)
torch.Size([1, 100, 256]) torch.Size([1, 256, 64, 64])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L315-L377"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SAM2TwoWayTransformer(TwoWayTransformer):
    """A Two-Way Transformer module for simultaneous attention to image and query points.

    This class extends the TwoWayTransformer, implementing a specialized transformer decoder that attends to an input
    image using queries with supplied positional embeddings. It is particularly useful for tasks like object detection,
    image segmentation, and point cloud processing.

    Attributes:
        depth (int): Number of layers in the transformer.
        embedding_dim (int): Channel dimension for input embeddings.
        num_heads (int): Number of heads for multihead attention.
        mlp_dim (int): Internal channel dimension for the MLP block.
        layers (nn.ModuleList): List of SAM2TwoWayAttentionBlock layers comprising the transformer.
        final_attn_token_to_image (Attention): Final attention layer from queries to image.
        norm_final_attn (nn.LayerNorm): Layer normalization applied to final queries.

    Methods:
        forward: Processes input image embeddings and query embeddings through the transformer.

    Examples:
        >>> transformer = SAM2TwoWayTransformer(depth=5, embedding_dim=256, num_heads=8, mlp_dim=2048)
        >>> image_embedding = torch.randn(1, 256, 64, 64)
        >>> query_embedding = torch.randn(1, 100, 256)
        >>> output = transformer(image_embedding, query_embedding)
        >>> print(output[0].shape, output[1].shape)
        torch.Size([1, 100, 256]) torch.Size([1, 256, 64, 64])
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """Initialize a SAM2TwoWayTransformer instance.

        This transformer decoder attends to an input image using queries with supplied positional embeddings. It is
        designed for tasks like object detection, image segmentation, and point cloud processing.

        Args:
            depth (int): Number of layers in the transformer.
            embedding_dim (int): Channel dimension for the input embeddings.
            num_heads (int): Number of heads for multihead attention. Must divide embedding_dim.
            mlp_dim (int): Channel dimension internal to the MLP block.
            activation (type[nn.Module]): Activation function to use in the MLP block.
            attention_downsample_rate (int): Downsampling rate for attention computations.
        """
        super().__init__(depth, embedding_dim, num_heads, mlp_dim, activation, attention_downsample_rate)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                SAM2TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.RoPEAttention` {#ultralytics.models.sam.modules.blocks.RoPEAttention}

```python
def __init__(
    self,
    *args,
    rope_theta: float = 10000.0,
    rope_k_repeat: bool = False,
    feat_sizes: tuple[int, int] = (32, 32),  # [w, h] for stride 16 feats at 512 resolution
    **kwargs,
)
```

**Bases:** `Attention`

Implements rotary position encoding for attention mechanisms in transformer architectures.

This class extends the base Attention class by incorporating Rotary Position Encoding (RoPE) to enhance the positional awareness of the attention mechanism.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `*args` |  |  | *required* |
| `rope_theta` | `float` |  | `10000.0` |
| `rope_k_repeat` | `bool` |  | `False` |
| `feat_sizes` | `tuple[int, int]` |  | `(32, 32)` |
| `**kwargs` |  |  | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `compute_cis` | `Callable` | Function to compute axial complex numbers for rotary encoding. |
| `freqs_cis` | `torch.Tensor` | Precomputed frequency tensor for rotary encoding. |
| `rope_k_repeat` | `bool` | Flag to repeat query RoPE to match key length for cross-attention to memories. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.blocks.RoPEAttention.forward) | Apply rotary position encoding and compute attention between query, key, and value tensors. |

**Examples**

```python
>>> rope_attn = RoPEAttention(embedding_dim=256, num_heads=8, rope_theta=10000.0, feat_sizes=(32, 32))
>>> q = torch.randn(1, 1024, 256)
>>> k = torch.randn(1, 1024, 256)
>>> v = torch.randn(1, 1024, 256)
>>> output = rope_attn(q, k, v)
>>> print(output.shape)
torch.Size([1, 1024, 256])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L380-L453"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RoPEAttention(Attention):
    """Implements rotary position encoding for attention mechanisms in transformer architectures.

    This class extends the base Attention class by incorporating Rotary Position Encoding (RoPE) to enhance the
    positional awareness of the attention mechanism.

    Attributes:
        compute_cis (Callable): Function to compute axial complex numbers for rotary encoding.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for rotary encoding.
        rope_k_repeat (bool): Flag to repeat query RoPE to match key length for cross-attention to memories.

    Methods:
        forward: Applies rotary position encoding and computes attention between query, key, and value tensors.

    Examples:
        >>> rope_attn = RoPEAttention(embedding_dim=256, num_heads=8, rope_theta=10000.0, feat_sizes=(32, 32))
        >>> q = torch.randn(1, 1024, 256)
        >>> k = torch.randn(1, 1024, 256)
        >>> v = torch.randn(1, 1024, 256)
        >>> output = rope_attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 1024, 256])
    """

    def __init__(
        self,
        *args,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, int] = (32, 32),  # [w, h] for stride 16 feats at 512 resolution
        **kwargs,
    ):
        """Initialize RoPEAttention with rotary position encoding for enhanced positional awareness."""
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat  # repeat q rope to match k length, needed for cross-attention to memories
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.RoPEAttention.forward` {#ultralytics.models.sam.modules.blocks.RoPEAttention.forward}

```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0) -> torch.Tensor
```

Apply rotary position encoding and compute attention between query, key, and value tensors.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `q` | `torch.Tensor` |  | *required* |
| `k` | `torch.Tensor` |  | *required* |
| `v` | `torch.Tensor` |  | *required* |
| `num_k_exclude_rope` | `int` |  | `0` |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L420-L453"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0) -> torch.Tensor:
    """Apply rotary position encoding and compute attention between query, key, and value tensors."""
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    # Separate into heads
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)

    # Apply rotary position encoding
    w = h = math.sqrt(q.shape[-2])
    self.freqs_cis = self.freqs_cis.to(q.device)
    if self.freqs_cis.shape[0] != q.shape[-2]:
        self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
    if q.shape[-2] != k.shape[-2]:
        assert self.rope_k_repeat

    num_k_rope = k.size(-2) - num_k_exclude_rope
    q, k[:, :, :num_k_rope] = apply_rotary_enc(
        q,
        k[:, :, :num_k_rope],
        freqs_cis=self.freqs_cis,
        repeat_freqs_k=self.rope_k_repeat,
    )

    # Attention
    out = F.scaled_dot_product_attention(q, k, v)

    out = self._recombine_heads(out)
    out = self.out_proj(out)

    return out
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.MultiScaleAttention` {#ultralytics.models.sam.modules.blocks.MultiScaleAttention}

```python
MultiScaleAttention(self, dim: int, dim_out: int, num_heads: int, q_pool: nn.Module = None)
```

**Bases:** `nn.Module`

Implements multiscale self-attention with optional query pooling for efficient feature extraction.

This class provides a flexible implementation of multiscale attention, allowing for optional downsampling of query features through pooling. It's designed to enhance the model's ability to capture multiscale information in visual tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dim` | `int` |  | *required* |
| `dim_out` | `int` |  | *required* |
| `num_heads` | `int` |  | *required* |
| `q_pool` | `nn.Module` |  | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `dim` | `int` | Input dimension of the feature map. |
| `dim_out` | `int` | Output dimension of the attention module. |
| `num_heads` | `int` | Number of attention heads. |
| `scale` | `float` | Scaling factor for dot-product attention. |
| `q_pool` | `nn.Module | None` | Optional pooling module for query features. |
| `qkv` | `nn.Linear` | Linear projection for query, key, and value. |
| `proj` | `nn.Linear` | Output projection. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.blocks.MultiScaleAttention.forward) | Apply multiscale attention with optional query pooling to extract multiscale features. |

**Examples**

```python
>>> import torch
>>> from torch import nn
>>> x = torch.randn(1, 64, 64, 256)
>>> msa = MultiScaleAttention(dim=256, dim_out=256, num_heads=8)
>>> output = msa(x)
>>> print(output.shape)
torch.Size([1, 64, 64, 256])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L471-L547"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class MultiScaleAttention(nn.Module):
    """Implements multiscale self-attention with optional query pooling for efficient feature extraction.

    This class provides a flexible implementation of multiscale attention, allowing for optional downsampling of query
    features through pooling. It's designed to enhance the model's ability to capture multiscale information in visual
    tasks.

    Attributes:
        dim (int): Input dimension of the feature map.
        dim_out (int): Output dimension of the attention module.
        num_heads (int): Number of attention heads.
        scale (float): Scaling factor for dot-product attention.
        q_pool (nn.Module | None): Optional pooling module for query features.
        qkv (nn.Linear): Linear projection for query, key, and value.
        proj (nn.Linear): Output projection.

    Methods:
        forward: Applies multiscale attention to the input tensor.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> x = torch.randn(1, 64, 64, 256)
        >>> msa = MultiScaleAttention(dim=256, dim_out=256, num_heads=8)
        >>> output = msa(x)
        >>> print(output.shape)
        torch.Size([1, 64, 64, 256])
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        """Initialize multiscale attention with optional query pooling for efficient feature extraction."""
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.MultiScaleAttention.forward` {#ultralytics.models.sam.modules.blocks.MultiScaleAttention.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply multiscale attention with optional query pooling to extract multiscale features.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L521-L547"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply multiscale attention with optional query pooling to extract multiscale features."""
    B, H, W, _ = x.shape
    # qkv with shape (B, H * W, 3, nHead, C)
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
    # q, k, v with shape (B, H * W, nheads, C)
    q, k, v = torch.unbind(qkv, 2)

    # Q pooling (for downsample at stage changes)
    if self.q_pool:
        q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
        H, W = q.shape[1:3]  # downsampled shape
        q = q.reshape(B, H * W, self.num_heads, -1)

    # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
    x = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    )
    # Transpose back
    x = x.transpose(1, 2)
    x = x.reshape(B, H, W, -1)

    x = self.proj(x)

    return x
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.MultiScaleBlock` {#ultralytics.models.sam.modules.blocks.MultiScaleBlock}

```python
def __init__(
    self,
    dim: int,
    dim_out: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    drop_path: float = 0.0,
    norm_layer: nn.Module | str = "LayerNorm",
    q_stride: tuple[int, int] | None = None,
    act_layer: type[nn.Module] = nn.GELU,
    window_size: int = 0,
)
```

**Bases:** `nn.Module`

A multiscale attention block with window partitioning and query pooling for efficient vision transformers.

This class implements a multiscale attention mechanism with optional window partitioning and downsampling, designed for use in vision transformer architectures.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dim` | `int` |  | *required* |
| `dim_out` | `int` |  | *required* |
| `num_heads` | `int` |  | *required* |
| `mlp_ratio` | `float` |  | `4.0` |
| `drop_path` | `float` |  | `0.0` |
| `norm_layer` | `nn.Module | str` |  | `"LayerNorm"` |
| `q_stride` | `tuple[int, int] | None` |  | `None` |
| `act_layer` | `type[nn.Module]` |  | `nn.GELU` |
| `window_size` | `int` |  | `0` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `dim` | `int` | Input dimension of the block. |
| `dim_out` | `int` | Output dimension of the block. |
| `norm1` | `nn.Module` | First normalization layer. |
| `window_size` | `int` | Size of the window for partitioning. |
| `pool` | `nn.Module | None` | Pooling layer for query downsampling. |
| `q_stride` | `tuple[int, int] | None` | Stride for query pooling. |
| `attn` | `MultiScaleAttention` | Multi-scale attention module. |
| `drop_path` | `nn.Module` | Drop path layer for regularization. |
| `norm2` | `nn.Module` | Second normalization layer. |
| `mlp` | `MLP` | Multi-layer perceptron module. |
| `proj` | `nn.Linear | None` | Projection layer for dimension mismatch. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.blocks.MultiScaleBlock.forward) | Process input through multiscale attention and MLP, with optional windowing and downsampling. |

**Examples**

```python
>>> block = MultiScaleBlock(dim=256, dim_out=512, num_heads=8, window_size=7)
>>> x = torch.randn(1, 56, 56, 256)
>>> output = block(x)
>>> print(output.shape)
torch.Size([1, 28, 28, 512])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L550-L661"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class MultiScaleBlock(nn.Module):
    """A multiscale attention block with window partitioning and query pooling for efficient vision transformers.

    This class implements a multiscale attention mechanism with optional window partitioning and downsampling, designed
    for use in vision transformer architectures.

    Attributes:
        dim (int): Input dimension of the block.
        dim_out (int): Output dimension of the block.
        norm1 (nn.Module): First normalization layer.
        window_size (int): Size of the window for partitioning.
        pool (nn.Module | None): Pooling layer for query downsampling.
        q_stride (tuple[int, int] | None): Stride for query pooling.
        attn (MultiScaleAttention): Multi-scale attention module.
        drop_path (nn.Module): Drop path layer for regularization.
        norm2 (nn.Module): Second normalization layer.
        mlp (MLP): Multi-layer perceptron module.
        proj (nn.Linear | None): Projection layer for dimension mismatch.

    Methods:
        forward: Processes input tensor through the multiscale block.

    Examples:
        >>> block = MultiScaleBlock(dim=256, dim_out=512, num_heads=8, window_size=7)
        >>> x = torch.randn(1, 56, 56, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 28, 28, 512])
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module | str = "LayerNorm",
        q_stride: tuple[int, int] | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        window_size: int = 0,
    ):
        """Initialize a multiscale attention block with window partitioning and optional query pooling."""
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            act=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.MultiScaleBlock.forward` {#ultralytics.models.sam.modules.blocks.MultiScaleBlock.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Process input through multiscale attention and MLP, with optional windowing and downsampling.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L628-L661"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Process input through multiscale attention and MLP, with optional windowing and downsampling."""
    shortcut = x  # B, H, W, C
    x = self.norm1(x)

    # Skip connection
    if self.dim != self.dim_out:
        shortcut = do_pool(self.proj(x), self.pool)

    # Window partition
    window_size = self.window_size
    if window_size > 0:
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, window_size)

    # Window Attention + Q Pooling (if stage change)
    x = self.attn(x)
    if self.q_stride:
        # Shapes have changed due to Q pooling
        window_size = self.window_size // self.q_stride[0]
        H, W = shortcut.shape[1:3]

        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        pad_hw = (H + pad_h, W + pad_w)

    # Reverse window partition
    if self.window_size > 0:
        x = window_unpartition(x, window_size, pad_hw, (H, W))

    x = shortcut + self.drop_path(x)
    # MLP
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.PositionEmbeddingSine` {#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine}

```python
def __init__(
    self,
    num_pos_feats: int,
    temperature: int = 10000,
    normalize: bool = True,
    scale: float | None = None,
)
```

**Bases:** `nn.Module`

A module for generating sinusoidal positional embeddings for 2D inputs like images.

This class implements sinusoidal position encoding for 2D spatial positions, which can be used in transformer-based models for computer vision tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `num_pos_feats` | `int` |  | *required* |
| `temperature` | `int` |  | `10000` |
| `normalize` | `bool` |  | `True` |
| `scale` | `float | None` |  | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `num_pos_feats` | `int` | Number of positional features (half of the embedding dimension). |
| `temperature` | `int` | Temperature parameter for the sinusoidal functions. |
| `normalize` | `bool` | Whether to normalize the positional embeddings. |
| `scale` | `float` | Scaling factor for the embeddings when normalize is True. |
| `cache` | `dict` | Cache for storing precomputed embeddings. |

**Methods**

| Name | Description |
| --- | --- |
| [`_encode_xy`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine._encode_xy) | Encode 2D positions using sine/cosine functions for transformer positional embeddings. |
| [`encode_boxes`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.encode_boxes) | Encode box coordinates and dimensions into positional embeddings for detection. |
| [`encode_points`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.encode_points) | Encode 2D points with sinusoidal embeddings and append labels. |
| [`forward`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.forward) | Generate sinusoidal position embeddings for 2D inputs like images. |

**Examples**

```python
>>> pos_emb = PositionEmbeddingSine(num_pos_feats=128)
>>> x = torch.randn(1, 3, 224, 224)
>>> embeddings = pos_emb(x)
>>> print(embeddings.shape)
torch.Size([1, 256, 224, 224])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L664-L775"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class PositionEmbeddingSine(nn.Module):
    """A module for generating sinusoidal positional embeddings for 2D inputs like images.

    This class implements sinusoidal position encoding for 2D spatial positions, which can be used in transformer-based
    models for computer vision tasks.

    Attributes:
        num_pos_feats (int): Number of positional features (half of the embedding dimension).
        temperature (int): Temperature parameter for the sinusoidal functions.
        normalize (bool): Whether to normalize the positional embeddings.
        scale (float): Scaling factor for the embeddings when normalize is True.
        cache (dict): Cache for storing precomputed embeddings.

    Methods:
        _encode_xy: Encodes 2D positions using sine and cosine functions.
        encode_boxes: Encodes box coordinates and dimensions into positional embeddings.
        encode_points: Encodes 2D point coordinates with sinusoidal positional embeddings.
        forward: Generates sinusoidal position embeddings for 2D inputs.

    Examples:
        >>> pos_emb = PositionEmbeddingSine(num_pos_feats=128)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> embeddings = pos_emb(x)
        >>> print(embeddings.shape)
        torch.Size([1, 256, 224, 224])
    """

    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float | None = None,
    ):
        """Initialize sinusoidal position embeddings for 2D image inputs."""
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.PositionEmbeddingSine._encode_xy` {#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.\_encode\_xy}

```python
def _encode_xy(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

Encode 2D positions using sine/cosine functions for transformer positional embeddings.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |
| `y` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L712-L725"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _encode_xy(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode 2D positions using sine/cosine functions for transformer positional embeddings."""
    assert len(x) == len(y) and x.ndim == y.ndim == 1
    x_embed = x * self.scale
    y_embed = y * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
    return pos_x, pos_y
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.encode_boxes` {#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.encode\_boxes}

```python
def encode_boxes(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor
```

Encode box coordinates and dimensions into positional embeddings for detection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |
| `y` | `torch.Tensor` |  | *required* |
| `w` | `torch.Tensor` |  | *required* |
| `h` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L728-L731"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@torch.no_grad()
def encode_boxes(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Encode box coordinates and dimensions into positional embeddings for detection."""
    pos_x, pos_y = self._encode_xy(x, y)
    return torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.encode_points` {#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.encode\_points}

```python
def encode_points(self, x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor
```

Encode 2D points with sinusoidal embeddings and append labels.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |
| `y` | `torch.Tensor` |  | *required* |
| `labels` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L736-L742"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@torch.no_grad()
def encode_points(self, x: torch.Tensor, y: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Encode 2D points with sinusoidal embeddings and append labels."""
    (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
    assert bx == by and nx == ny and bx == bl and nx == nl
    pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
    pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
    return torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.forward` {#ultralytics.models.sam.modules.blocks.PositionEmbeddingSine.forward}

```python
def forward(self, x: torch.Tensor) -> Tensor
```

Generate sinusoidal position embeddings for 2D inputs like images.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L745-L775"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@torch.no_grad()
def forward(self, x: torch.Tensor) -> Tensor:
    """Generate sinusoidal position embeddings for 2D inputs like images."""
    cache_key = (x.shape[-2], x.shape[-1])
    if cache_key in self.cache:
        return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
    y_embed = (
        torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
        .view(1, -1, 1)
        .repeat(x.shape[0], 1, x.shape[-1])
    )
    x_embed = (
        torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
        .view(1, 1, -1)
        .repeat(x.shape[0], x.shape[-2], 1)
    )

    if self.normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    self.cache[cache_key] = pos[0]
    return pos
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom` {#ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom}

```python
PositionEmbeddingRandom(self, num_pos_feats: int = 64, scale: float | None = None) -> None
```

**Bases:** `nn.Module`

Positional encoding using random spatial frequencies.

This class generates positional embeddings for input coordinates using random spatial frequencies. It is particularly useful for transformer-based models that require position information.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `num_pos_feats` | `int` |  | `64` |
| `scale` | `float | None` |  | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `positional_encoding_gaussian_matrix` | `torch.Tensor` | A buffer containing random values for encoding. |

**Methods**

| Name | Description |
| --- | --- |
| [`_pe_encoding`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom._pe_encoding) | Encode normalized [0,1] coordinates using random spatial frequencies. |
| [`forward`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom.forward) | Generate positional encoding for a grid using random spatial frequencies. |
| [`forward_with_coords`](#ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom.forward_with_coords) | Positionally encode input coordinates, normalizing them to [0,1] based on the given image size. |

**Examples**

```python
>>> pe = PositionEmbeddingRandom(num_pos_feats=64)
>>> size = (32, 32)
>>> encoding = pe(size)
>>> print(encoding.shape)
torch.Size([128, 32, 32])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L778-L841"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies.

    This class generates positional embeddings for input coordinates using random spatial frequencies. It is
    particularly useful for transformer-based models that require position information.

    Attributes:
        positional_encoding_gaussian_matrix (torch.Tensor): A buffer containing random values for encoding.

    Methods:
        _pe_encoding: Positionally encodes points that are normalized to [0,1].
        forward: Generates positional encoding for a grid of the specified size.
        forward_with_coords: Positionally encodes points that are not normalized to [0,1].

    Examples:
        >>> pe = PositionEmbeddingRandom(num_pos_feats=64)
        >>> size = (32, 32)
        >>> encoding = pe(size)
        >>> print(encoding.shape)
        torch.Size([128, 32, 32])
    """

    def __init__(self, num_pos_feats: int = 64, scale: float | None = None) -> None:
        """Initialize random spatial frequency position embedding for transformers."""
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

        # Set non-deterministic for forward() error 'cumsum_cuda_kernel does not have a deterministic implementation'
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom._pe_encoding` {#ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom.\_pe\_encoding}

```python
def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor
```

Encode normalized [0,1] coordinates using random spatial frequencies.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `coords` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L811-L818"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
    """Encode normalized [0,1] coordinates using random spatial frequencies."""
    # Assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
    coords = 2 * coords - 1
    coords = coords @ self.positional_encoding_gaussian_matrix
    coords = 2 * np.pi * coords
    # Outputs d_1 x ... x d_n x C shape
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom.forward` {#ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom.forward}

```python
def forward(self, size: tuple[int, int]) -> torch.Tensor
```

Generate positional encoding for a grid using random spatial frequencies.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `size` | `tuple[int, int]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L820-L834"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, size: tuple[int, int]) -> torch.Tensor:
    """Generate positional encoding for a grid using random spatial frequencies."""
    h, w = size
    grid = torch.ones(
        (h, w),
        device=self.positional_encoding_gaussian_matrix.device,
        dtype=self.positional_encoding_gaussian_matrix.dtype,
    )
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    y_embed = y_embed / h
    x_embed = x_embed / w

    pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
    return pe.permute(2, 0, 1)  # C x H x W
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom.forward_with_coords` {#ultralytics.models.sam.modules.blocks.PositionEmbeddingRandom.forward\_with\_coords}

```python
def forward_with_coords(self, coords_input: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor
```

Positionally encode input coordinates, normalizing them to [0,1] based on the given image size.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `coords_input` | `torch.Tensor` |  | *required* |
| `image_size` | `tuple[int, int]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L836-L841"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward_with_coords(self, coords_input: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Positionally encode input coordinates, normalizing them to [0,1] based on the given image size."""
    coords = coords_input.clone()
    coords[:, :, 0] = coords[:, :, 0] / image_size[1]
    coords[:, :, 1] = coords[:, :, 1] / image_size[0]
    return self._pe_encoding(coords)  # B x N x C
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.Block` {#ultralytics.models.sam.modules.blocks.Block}

```python
def __init__(
    self,
    dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = True,
    norm_layer: type[nn.Module] = nn.LayerNorm,
    act_layer: type[nn.Module] = nn.GELU,
    use_rel_pos: bool = False,
    rel_pos_zero_init: bool = True,
    window_size: int = 0,
    input_size: tuple[int, int] | None = None,
) -> None
```

**Bases:** `nn.Module`

Transformer block with support for window attention and residual propagation.

This class implements a transformer block that can use either global or windowed self-attention, followed by a feed-forward network. It supports relative positional embeddings and is designed for use in vision transformer architectures.

This constructor sets up a transformer block that can use either global or windowed self-attention, followed by a feed-forward network. It supports relative positional embeddings and is designed for use in vision transformer architectures.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dim` | `int` | Number of input channels. | *required* |
| `num_heads` | `int` | Number of attention heads in the self-attention layer. | *required* |
| `mlp_ratio` | `float` | Ratio of mlp hidden dimension to embedding dimension. | `4.0` |
| `qkv_bias` | `bool` | If True, adds a learnable bias to query, key, value projections. | `True` |
| `norm_layer` | `type[nn.Module]` | Type of normalization layer to use. | `nn.LayerNorm` |
| `act_layer` | `type[nn.Module]` | Type of activation function to use in the MLP block. | `nn.GELU` |
| `use_rel_pos` | `bool` | If True, uses relative positional embeddings in attention. | `False` |
| `rel_pos_zero_init` | `bool` | If True, initializes relative positional parameters to zero. | `True` |
| `window_size` | `int` | Size of attention window. If 0, uses global attention. | `0` |
| `input_size` | `tuple[int, int] | None` | Input resolution for calculating relative positional parameter size. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `norm1` | `nn.Module` | First normalization layer. |
| `attn` | `REAttention` | Self-attention layer with optional relative positional encoding. |
| `norm2` | `nn.Module` | Second normalization layer. |
| `mlp` | `MLPBlock` | Multi-layer perceptron block. |
| `window_size` | `int` | Size of attention window. If 0, global attention is used. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.blocks.Block.forward) | Process input through transformer block with optional windowed self-attention and residual connection. |

**Examples**

```python
>>> import torch
>>> block = Block(dim=256, num_heads=8, window_size=7)
>>> x = torch.randn(1, 56, 56, 256)
>>> output = block(x)
>>> print(output.shape)
torch.Size([1, 56, 56, 256])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L844-L932"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Block(nn.Module):
    """Transformer block with support for window attention and residual propagation.

    This class implements a transformer block that can use either global or windowed self-attention, followed by a
    feed-forward network. It supports relative positional embeddings and is designed for use in vision transformer
    architectures.

    Attributes:
        norm1 (nn.Module): First normalization layer.
        attn (REAttention): Self-attention layer with optional relative positional encoding.
        norm2 (nn.Module): Second normalization layer.
        mlp (MLPBlock): Multi-layer perceptron block.
        window_size (int): Size of attention window. If 0, global attention is used.

    Methods:
        forward: Processes input through the transformer block.

    Examples:
        >>> import torch
        >>> block = Block(dim=256, num_heads=8, window_size=7)
        >>> x = torch.randn(1, 56, 56, 256)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 56, 56, 256])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """Initialize a transformer block with optional window attention and relative positional embeddings.

        This constructor sets up a transformer block that can use either global or windowed self-attention, followed by
        a feed-forward network. It supports relative positional embeddings and is designed for use in vision transformer
        architectures.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in the self-attention layer.
            mlp_ratio (float): Ratio of mlp hidden dimension to embedding dimension.
            qkv_bias (bool): If True, adds a learnable bias to query, key, value projections.
            norm_layer (type[nn.Module]): Type of normalization layer to use.
            act_layer (type[nn.Module]): Type of activation function to use in the MLP block.
            use_rel_pos (bool): If True, uses relative positional embeddings in attention.
            rel_pos_zero_init (bool): If True, initializes relative positional parameters to zero.
            window_size (int): Size of attention window. If 0, uses global attention.
            input_size (tuple[int, int] | None): Input resolution for calculating relative positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = REAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.Block.forward` {#ultralytics.models.sam.modules.blocks.Block.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Process input through transformer block with optional windowed self-attention and residual connection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L917-L932"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Process input through transformer block with optional windowed self-attention and residual connection."""
    shortcut = x
    x = self.norm1(x)
    # Window partition
    if self.window_size > 0:
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, self.window_size)

    x = self.attn(x)
    # Reverse window partition
    if self.window_size > 0:
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))

    x = shortcut + x
    return x + self.mlp(self.norm2(x))
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.REAttention` {#ultralytics.models.sam.modules.blocks.REAttention}

```python
def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    qkv_bias: bool = True,
    use_rel_pos: bool = False,
    rel_pos_zero_init: bool = True,
    input_size: tuple[int, int] | None = None,
) -> None
```

**Bases:** `nn.Module`

Relative Position Attention module for efficient self-attention in transformer architectures.

This class implements a multi-head attention mechanism with relative positional embeddings, designed for use in vision transformer models.

This module implements multi-head attention with optional relative positional encodings, designed specifically for vision tasks in transformer models.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dim` | `int` | Number of input channels. | *required* |
| `num_heads` | `int` | Number of attention heads. | `8` |
| `qkv_bias` | `bool` | If True, adds a learnable bias to query, key, value projections. | `True` |
| `use_rel_pos` | `bool` | If True, uses relative positional encodings. | `False` |
| `rel_pos_zero_init` | `bool` | If True, initializes relative positional parameters to zero. | `True` |
| `input_size` | `tuple[int, int] | None` | Input resolution for calculating relative positional parameter size.<br>    Required if use_rel_pos is True. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `num_heads` | `int` | Number of attention heads. |
| `scale` | `float` | Scaling factor for attention computation. |
| `qkv` | `nn.Linear` | Linear projection for query, key, and value. |
| `proj` | `nn.Linear` | Output projection layer. |
| `use_rel_pos` | `bool` | Whether to use relative positional embeddings. |
| `rel_pos_h` | `nn.Parameter` | Relative positional embeddings for height dimension. |
| `rel_pos_w` | `nn.Parameter` | Relative positional embeddings for width dimension. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.blocks.REAttention.forward) | Apply multi-head attention with optional relative positional encoding to input tensor. |

**Examples**

```python
>>> attention = REAttention(dim=256, num_heads=8, input_size=(32, 32))
>>> x = torch.randn(1, 32, 32, 256)
>>> output = attention(x)
>>> print(output.shape)
torch.Size([1, 32, 32, 256])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L935-L1014"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class REAttention(nn.Module):
    """Relative Position Attention module for efficient self-attention in transformer architectures.

    This class implements a multi-head attention mechanism with relative positional embeddings, designed for use in
    vision transformer models.

    Attributes:
        num_heads (int): Number of attention heads.
        scale (float): Scaling factor for attention computation.
        qkv (nn.Linear): Linear projection for query, key, and value.
        proj (nn.Linear): Output projection layer.
        use_rel_pos (bool): Whether to use relative positional embeddings.
        rel_pos_h (nn.Parameter): Relative positional embeddings for height dimension.
        rel_pos_w (nn.Parameter): Relative positional embeddings for width dimension.

    Methods:
        forward: Applies multi-head attention with optional relative positional encoding to input tensor.

    Examples:
        >>> attention = REAttention(dim=256, num_heads=8, input_size=(32, 32))
        >>> x = torch.randn(1, 32, 32, 256)
        >>> output = attention(x)
        >>> print(output.shape)
        torch.Size([1, 32, 32, 256])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        """Initialize a Relative Position Attention module for transformer-based architectures.

        This module implements multi-head attention with optional relative positional encodings, designed specifically
        for vision tasks in transformer models.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): If True, adds a learnable bias to query, key, value projections.
            use_rel_pos (bool): If True, uses relative positional encodings.
            rel_pos_zero_init (bool): If True, initializes relative positional parameters to zero.
            input_size (tuple[int, int] | None): Input resolution for calculating relative positional parameter size.
                Required if use_rel_pos is True.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            # Initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.REAttention.forward` {#ultralytics.models.sam.modules.blocks.REAttention.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply multi-head attention with optional relative positional encoding to input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L999-L1014"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply multi-head attention with optional relative positional encoding to input tensor."""
    B, H, W, _ = x.shape
    # qkv with shape (3, B, nHead, H * W, C)
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    # q, k, v with shape (B * nHead, H * W, C)
    q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

    attn = (q * self.scale) @ k.transpose(-2, -1)

    if self.use_rel_pos:
        attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

    attn = attn.softmax(dim=-1)
    x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    return self.proj(x)
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.blocks.PatchEmbed` {#ultralytics.models.sam.modules.blocks.PatchEmbed}

```python
def __init__(
    self,
    kernel_size: tuple[int, int] = (16, 16),
    stride: tuple[int, int] = (16, 16),
    padding: tuple[int, int] = (0, 0),
    in_chans: int = 3,
    embed_dim: int = 768,
    bias: bool = True,
) -> None
```

**Bases:** `nn.Module`

Image to Patch Embedding module for vision transformer architectures.

This module converts an input image into a sequence of patch embeddings using a convolutional layer. It is commonly used as the first layer in vision transformer architectures to transform image data into a suitable format for subsequent transformer blocks.

This module is typically used as the first layer in vision transformer architectures to transform image data into a suitable format for subsequent transformer blocks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `kernel_size` | `tuple[int, int]` | Size of the convolutional kernel for patch extraction. | `(16, 16)` |
| `stride` | `tuple[int, int]` | Stride of the convolutional operation. | `(16, 16)` |
| `padding` | `tuple[int, int]` | Padding applied to the input before convolution. | `(0, 0)` |
| `in_chans` | `int` | Number of input image channels. | `3` |
| `embed_dim` | `int` | Dimensionality of the output patch embeddings. | `768` |
| `bias` | `bool` | Whether to include a bias term in the convolutional layer. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `proj` | `nn.Conv2d` | Convolutional layer for projecting image patches to embeddings. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.blocks.PatchEmbed.forward) | Compute patch embedding by applying convolution and transposing resulting tensor. |

**Examples**

```python
>>> patch_embed = PatchEmbed(kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768)
>>> x = torch.randn(1, 3, 224, 224)
>>> output = patch_embed(x)
>>> print(output.shape)
torch.Size([1, 768, 14, 14])
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L1017-L1066"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class PatchEmbed(nn.Module):
    """Image to Patch Embedding module for vision transformer architectures.

    This module converts an input image into a sequence of patch embeddings using a convolutional layer. It is commonly
    used as the first layer in vision transformer architectures to transform image data into a suitable format for
    subsequent transformer blocks.

    Attributes:
        proj (nn.Conv2d): Convolutional layer for projecting image patches to embeddings.

    Methods:
        forward: Applies patch embedding to the input tensor.

    Examples:
        >>> patch_embed = PatchEmbed(kernel_size=(16, 16), stride=(16, 16), in_chans=3, embed_dim=768)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = patch_embed(x)
        >>> print(output.shape)
        torch.Size([1, 768, 14, 14])
    """

    def __init__(
        self,
        kernel_size: tuple[int, int] = (16, 16),
        stride: tuple[int, int] = (16, 16),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        """Initialize the PatchEmbed module for converting image patches to embeddings.

        This module is typically used as the first layer in vision transformer architectures to transform image data
        into a suitable format for subsequent transformer blocks.

        Args:
            kernel_size (tuple[int, int]): Size of the convolutional kernel for patch extraction.
            stride (tuple[int, int]): Stride of the convolutional operation.
            padding (tuple[int, int]): Padding applied to the input before convolution.
            in_chans (int): Number of input image channels.
            embed_dim (int): Dimensionality of the output patch embeddings.
            bias (bool): Whether to include a bias term in the convolutional layer.
        """
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
```
</details>

<br>

### Method `ultralytics.models.sam.modules.blocks.PatchEmbed.forward` {#ultralytics.models.sam.modules.blocks.PatchEmbed.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Compute patch embedding by applying convolution and transposing resulting tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L1064-L1066"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute patch embedding by applying convolution and transposing resulting tensor."""
    return self.proj(x).permute(0, 2, 3, 1)  # B C H W -> B H W C
```
</details>


<br><br><hr><br>

## Function `ultralytics.models.sam.modules.blocks.do_pool` {#ultralytics.models.sam.modules.blocks.do\_pool}

```python
def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor
```

Apply pooling and optional normalization to a tensor, handling spatial dimension permutations.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |
| `pool` | `nn.Module` |  | *required* |
| `norm` | `nn.Module` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/blocks.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/blocks.py#L456-L468"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    """Apply pooling and optional normalization to a tensor, handling spatial dimension permutations."""
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x
```
</details>

<br><br>
