---
description: Explore the TwoWayTransformer module in Ultralytics, designed for simultaneous attention to image and query points. Ideal for object detection and segmentation tasks.
keywords: Ultralytics, TwoWayTransformer, module, deep learning, transformer, object detection, image segmentation, attention mechanism, neural networks
---

# Reference for `ultralytics/models/sam/modules/transformer.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/transformer.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/transformer.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`TwoWayTransformer`](#ultralytics.models.sam.modules.transformer.TwoWayTransformer)
        - [`TwoWayAttentionBlock`](#ultralytics.models.sam.modules.transformer.TwoWayAttentionBlock)
        - [`Attention`](#ultralytics.models.sam.modules.transformer.Attention)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`TwoWayTransformer.forward`](#ultralytics.models.sam.modules.transformer.TwoWayTransformer.forward)
        - [`TwoWayAttentionBlock.forward`](#ultralytics.models.sam.modules.transformer.TwoWayAttentionBlock.forward)
        - [`Attention._separate_heads`](#ultralytics.models.sam.modules.transformer.Attention._separate_heads)
        - [`Attention._recombine_heads`](#ultralytics.models.sam.modules.transformer.Attention._recombine_heads)
        - [`Attention.forward`](#ultralytics.models.sam.modules.transformer.Attention.forward)


## Class `ultralytics.models.sam.modules.transformer.TwoWayTransformer` {#ultralytics.models.sam.modules.transformer.TwoWayTransformer}

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

**Bases:** `nn.Module`

A Two-Way Transformer module for simultaneous attention to image and query points.

This class implements a specialized transformer decoder that attends to an input image using queries with supplied positional embeddings. It's useful for tasks like object detection, image segmentation, and point cloud processing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `depth` | `int` | Number of layers in the transformer. | *required* |
| `embedding_dim` | `int` | Channel dimension for input embeddings. | *required* |
| `num_heads` | `int` | Number of heads for multihead attention. Must divide embedding_dim. | *required* |
| `mlp_dim` | `int` | Internal channel dimension for the MLP block. | *required* |
| `activation` | `type[nn.Module], optional` | Activation function to use in the MLP block. | `nn.ReLU` |
| `attention_downsample_rate` | `int, optional` | Downsampling rate for attention mechanism. | `2` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `depth` | `int` | Number of layers in the transformer. |
| `embedding_dim` | `int` | Channel dimension for input embeddings. |
| `num_heads` | `int` | Number of heads for multihead attention. |
| `mlp_dim` | `int` | Internal channel dimension for the MLP block. |
| `layers` | `nn.ModuleList` | List of TwoWayAttentionBlock layers composing the transformer. |
| `final_attn_token_to_image` | `Attention` | Final attention layer from queries to image. |
| `norm_final_attn` | `nn.LayerNorm` | Layer normalization applied to final queries. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.transformer.TwoWayTransformer.forward) | Process image and point embeddings through the Two-Way Transformer. |

**Examples**

```python
>>> transformer = TwoWayTransformer(depth=6, embedding_dim=256, num_heads=8, mlp_dim=2048)
>>> image_embedding = torch.randn(1, 256, 32, 32)
>>> image_pe = torch.randn(1, 256, 32, 32)
>>> point_embedding = torch.randn(1, 100, 256)
>>> output_queries, output_image = transformer(image_embedding, image_pe, point_embedding)
>>> print(output_queries.shape, output_image.shape)
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/transformer.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/transformer.py#L13-L122"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class TwoWayTransformer(nn.Module):
    """A Two-Way Transformer module for simultaneous attention to image and query points.

    This class implements a specialized transformer decoder that attends to an input image using queries with supplied
    positional embeddings. It's useful for tasks like object detection, image segmentation, and point cloud processing.

    Attributes:
        depth (int): Number of layers in the transformer.
        embedding_dim (int): Channel dimension for input embeddings.
        num_heads (int): Number of heads for multihead attention.
        mlp_dim (int): Internal channel dimension for the MLP block.
        layers (nn.ModuleList): List of TwoWayAttentionBlock layers composing the transformer.
        final_attn_token_to_image (Attention): Final attention layer from queries to image.
        norm_final_attn (nn.LayerNorm): Layer normalization applied to final queries.

    Methods:
        forward: Process image and point embeddings through the transformer.

    Examples:
        >>> transformer = TwoWayTransformer(depth=6, embedding_dim=256, num_heads=8, mlp_dim=2048)
        >>> image_embedding = torch.randn(1, 256, 32, 32)
        >>> image_pe = torch.randn(1, 256, 32, 32)
        >>> point_embedding = torch.randn(1, 100, 256)
        >>> output_queries, output_image = transformer(image_embedding, image_pe, point_embedding)
        >>> print(output_queries.shape, output_image.shape)
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
        """Initialize a Two-Way Transformer for simultaneous attention to image and query points.

        Args:
            depth (int): Number of layers in the transformer.
            embedding_dim (int): Channel dimension for input embeddings.
            num_heads (int): Number of heads for multihead attention. Must divide embedding_dim.
            mlp_dim (int): Internal channel dimension for the MLP block.
            activation (type[nn.Module], optional): Activation function to use in the MLP block.
            attention_downsample_rate (int, optional): Downsampling rate for attention mechanism.
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
```
</details>

<br>

### Method `ultralytics.models.sam.modules.transformer.TwoWayTransformer.forward` {#ultralytics.models.sam.modules.transformer.TwoWayTransformer.forward}

```python
def forward(
    self,
    image_embedding: torch.Tensor,
    image_pe: torch.Tensor,
    point_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]
```

Process image and point embeddings through the Two-Way Transformer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `image_embedding` | `torch.Tensor` | Image to attend to, with shape (B, embedding_dim, H, W). | *required* |
| `image_pe` | `torch.Tensor` | Positional encoding to add to the image, with same shape as image_embedding. | *required* |
| `point_embedding` | `torch.Tensor` | Embedding to add to query points, with shape (B, N_points, embedding_dim). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `queries (torch.Tensor)` | Processed point embeddings with shape (B, N_points, embedding_dim). |
| `keys (torch.Tensor)` | Processed image embeddings with shape (B, H*W, embedding_dim). |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/transformer.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/transformer.py#L81-L122"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(
    self,
    image_embedding: torch.Tensor,
    image_pe: torch.Tensor,
    point_embedding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Process image and point embeddings through the Two-Way Transformer.

    Args:
        image_embedding (torch.Tensor): Image to attend to, with shape (B, embedding_dim, H, W).
        image_pe (torch.Tensor): Positional encoding to add to the image, with same shape as image_embedding.
        point_embedding (torch.Tensor): Embedding to add to query points, with shape (B, N_points, embedding_dim).

    Returns:
        queries (torch.Tensor): Processed point embeddings with shape (B, N_points, embedding_dim).
        keys (torch.Tensor): Processed image embeddings with shape (B, H*W, embedding_dim).
    """
    # BxCxHxW -> BxHWxC == B x N_image_tokens x C
    image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
    image_pe = image_pe.flatten(2).permute(0, 2, 1)

    # Prepare queries
    queries = point_embedding
    keys = image_embedding

    # Apply transformer blocks and final layernorm
    for layer in self.layers:
        queries, keys = layer(
            queries=queries,
            keys=keys,
            query_pe=point_embedding,
            key_pe=image_pe,
        )

    # Apply the final attention layer from the points to the image
    q = queries + point_embedding
    k = keys + image_pe
    attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
    queries = queries + attn_out
    queries = self.norm_final_attn(queries)

    return queries, keys
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.transformer.TwoWayAttentionBlock` {#ultralytics.models.sam.modules.transformer.TwoWayAttentionBlock}

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

**Bases:** `nn.Module`

A two-way attention block for simultaneous attention to image and query points.

This class implements a specialized transformer block with four main layers: self-attention on sparse inputs, cross-attention of sparse inputs to dense inputs, MLP block on sparse inputs, and cross-attention of dense inputs to sparse inputs.

This block implements a specialized transformer layer with four main components: self-attention on sparse inputs, cross-attention of sparse inputs to dense inputs, MLP block on sparse inputs, and cross-attention of dense inputs to sparse inputs.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `embedding_dim` | `int` | Channel dimension of the embeddings. | *required* |
| `num_heads` | `int` | Number of attention heads in the attention layers. | *required* |
| `mlp_dim` | `int, optional` | Hidden dimension of the MLP block. | `2048` |
| `activation` | `type[nn.Module], optional` | Activation function for the MLP block. | `nn.ReLU` |
| `attention_downsample_rate` | `int, optional` | Downsampling rate for the attention mechanism. | `2` |
| `skip_first_layer_pe` | `bool, optional` | Whether to skip positional encoding in the first layer. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `self_attn` | `Attention` | Self-attention layer for queries. |
| `norm1` | `nn.LayerNorm` | Layer normalization after self-attention. |
| `cross_attn_token_to_image` | `Attention` | Cross-attention layer from queries to keys. |
| `norm2` | `nn.LayerNorm` | Layer normalization after token-to-image attention. |
| `mlp` | `MLPBlock` | MLP block for transforming query embeddings. |
| `norm3` | `nn.LayerNorm` | Layer normalization after MLP block. |
| `norm4` | `nn.LayerNorm` | Layer normalization after image-to-token attention. |
| `cross_attn_image_to_token` | `Attention` | Cross-attention layer from keys to queries. |
| `skip_first_layer_pe` | `bool` | Whether to skip positional encoding in the first layer. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.models.sam.modules.transformer.TwoWayAttentionBlock.forward) | Apply two-way attention to process query and key embeddings in a transformer block. |

**Examples**

```python
>>> embedding_dim, num_heads = 256, 8
>>> block = TwoWayAttentionBlock(embedding_dim, num_heads)
>>> queries = torch.randn(1, 100, embedding_dim)
>>> keys = torch.randn(1, 1000, embedding_dim)
>>> query_pe = torch.randn(1, 100, embedding_dim)
>>> key_pe = torch.randn(1, 1000, embedding_dim)
>>> processed_queries, processed_keys = block(queries, keys, query_pe, key_pe)
```

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/transformer.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/transformer.py#L125-L237"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class TwoWayAttentionBlock(nn.Module):
    """A two-way attention block for simultaneous attention to image and query points.

    This class implements a specialized transformer block with four main layers: self-attention on sparse inputs,
    cross-attention of sparse inputs to dense inputs, MLP block on sparse inputs, and cross-attention of dense inputs to
    sparse inputs.

    Attributes:
        self_attn (Attention): Self-attention layer for queries.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        cross_attn_token_to_image (Attention): Cross-attention layer from queries to keys.
        norm2 (nn.LayerNorm): Layer normalization after token-to-image attention.
        mlp (MLPBlock): MLP block for transforming query embeddings.
        norm3 (nn.LayerNorm): Layer normalization after MLP block.
        norm4 (nn.LayerNorm): Layer normalization after image-to-token attention.
        cross_attn_image_to_token (Attention): Cross-attention layer from keys to queries.
        skip_first_layer_pe (bool): Whether to skip positional encoding in the first layer.

    Methods:
        forward: Apply self-attention and cross-attention to queries and keys.

    Examples:
        >>> embedding_dim, num_heads = 256, 8
        >>> block = TwoWayAttentionBlock(embedding_dim, num_heads)
        >>> queries = torch.randn(1, 100, embedding_dim)
        >>> keys = torch.randn(1, 1000, embedding_dim)
        >>> query_pe = torch.randn(1, 100, embedding_dim)
        >>> key_pe = torch.randn(1, 1000, embedding_dim)
        >>> processed_queries, processed_keys = block(queries, keys, query_pe, key_pe)
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
        """Initialize a TwoWayAttentionBlock for simultaneous attention to image and query points.

        This block implements a specialized transformer layer with four main components: self-attention on sparse
        inputs, cross-attention of sparse inputs to dense inputs, MLP block on sparse inputs, and cross-attention of
        dense inputs to sparse inputs.

        Args:
            embedding_dim (int): Channel dimension of the embeddings.
            num_heads (int): Number of attention heads in the attention layers.
            mlp_dim (int, optional): Hidden dimension of the MLP block.
            activation (type[nn.Module], optional): Activation function for the MLP block.
            attention_downsample_rate (int, optional): Downsampling rate for the attention mechanism.
            skip_first_layer_pe (bool, optional): Whether to skip positional encoding in the first layer.
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe
```
</details>

<br>

### Method `ultralytics.models.sam.modules.transformer.TwoWayAttentionBlock.forward` {#ultralytics.models.sam.modules.transformer.TwoWayAttentionBlock.forward}

```python
def forward(
    self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]
```

Apply two-way attention to process query and key embeddings in a transformer block.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `queries` | `torch.Tensor` | Query embeddings with shape (B, N_queries, embedding_dim). | *required* |
| `keys` | `torch.Tensor` | Key embeddings with shape (B, N_keys, embedding_dim). | *required* |
| `query_pe` | `torch.Tensor` | Positional encodings for queries with same shape as queries. | *required* |
| `key_pe` | `torch.Tensor` | Positional encodings for keys with same shape as keys. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `queries (torch.Tensor)` | Processed query embeddings with shape (B, N_queries, embedding_dim). |
| `keys (torch.Tensor)` | Processed key embeddings with shape (B, N_keys, embedding_dim). |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/transformer.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/transformer.py#L194-L237"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(
    self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply two-way attention to process query and key embeddings in a transformer block.

    Args:
        queries (torch.Tensor): Query embeddings with shape (B, N_queries, embedding_dim).
        keys (torch.Tensor): Key embeddings with shape (B, N_keys, embedding_dim).
        query_pe (torch.Tensor): Positional encodings for queries with same shape as queries.
        key_pe (torch.Tensor): Positional encodings for keys with same shape as keys.

    Returns:
        queries (torch.Tensor): Processed query embeddings with shape (B, N_queries, embedding_dim).
        keys (torch.Tensor): Processed key embeddings with shape (B, N_keys, embedding_dim).
    """
    # Self attention block
    if self.skip_first_layer_pe:
        queries = self.self_attn(q=queries, k=queries, v=queries)
    else:
        q = queries + query_pe
        attn_out = self.self_attn(q=q, k=q, v=queries)
        queries = queries + attn_out
    queries = self.norm1(queries)

    # Cross attention block, tokens attending to image embedding
    q = queries + query_pe
    k = keys + key_pe
    attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
    queries = queries + attn_out
    queries = self.norm2(queries)

    # MLP block
    mlp_out = self.mlp(queries)
    queries = queries + mlp_out
    queries = self.norm3(queries)

    # Cross attention block, image embedding attending to tokens
    q = queries + query_pe
    k = keys + key_pe
    attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
    keys = keys + attn_out
    keys = self.norm4(keys)

    return queries, keys
```
</details>


<br><br><hr><br>

## Class `ultralytics.models.sam.modules.transformer.Attention` {#ultralytics.models.sam.modules.transformer.Attention}

```python
Attention(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1, kv_in_dim: int | None = None) -> None
```

**Bases:** `nn.Module`

An attention layer with downscaling capability for embedding size after projection.

This class implements a multi-head attention mechanism with the option to downsample the internal dimension of queries, keys, and values.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `embedding_dim` | `int` | Dimensionality of input embeddings. | *required* |
| `num_heads` | `int` | Number of attention heads. | *required* |
| `downsample_rate` | `int, optional` | Factor by which internal dimensions are downsampled. | `1` |
| `kv_in_dim` | `int | None, optional` | Dimensionality of key and value inputs. If None, uses embedding_dim. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `embedding_dim` | `int` | Dimensionality of input embeddings. |
| `kv_in_dim` | `int` | Dimensionality of key and value inputs. |
| `internal_dim` | `int` | Internal dimension after downsampling. |
| `num_heads` | `int` | Number of attention heads. |
| `q_proj` | `nn.Linear` | Linear projection for queries. |
| `k_proj` | `nn.Linear` | Linear projection for keys. |
| `v_proj` | `nn.Linear` | Linear projection for values. |
| `out_proj` | `nn.Linear` | Linear projection for output. |

**Methods**

| Name | Description |
| --- | --- |
| [`_recombine_heads`](#ultralytics.models.sam.modules.transformer.Attention._recombine_heads) | Recombine separated attention heads into a single tensor. |
| [`_separate_heads`](#ultralytics.models.sam.modules.transformer.Attention._separate_heads) | Separate the input tensor into the specified number of attention heads. |
| [`forward`](#ultralytics.models.sam.modules.transformer.Attention.forward) | Apply multi-head attention to query, key, and value tensors with optional downsampling. |

**Examples**

```python
>>> attn = Attention(embedding_dim=256, num_heads=8, downsample_rate=2)
>>> q = torch.randn(1, 100, 256)
>>> k = v = torch.randn(1, 50, 256)
>>> output = attn(q, k, v)
>>> print(output.shape)
torch.Size([1, 100, 256])
```

**Raises**

| Type | Description |
| --- | --- |
| `AssertionError` | If num_heads does not evenly divide the internal dim (embedding_dim / downsample_rate). |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/transformer.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/transformer.py#L240-L344"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Attention(nn.Module):
    """An attention layer with downscaling capability for embedding size after projection.

    This class implements a multi-head attention mechanism with the option to downsample the internal dimension of
    queries, keys, and values.

    Attributes:
        embedding_dim (int): Dimensionality of input embeddings.
        kv_in_dim (int): Dimensionality of key and value inputs.
        internal_dim (int): Internal dimension after downsampling.
        num_heads (int): Number of attention heads.
        q_proj (nn.Linear): Linear projection for queries.
        k_proj (nn.Linear): Linear projection for keys.
        v_proj (nn.Linear): Linear projection for values.
        out_proj (nn.Linear): Linear projection for output.

    Methods:
        _separate_heads: Separate input tensor into attention heads.
        _recombine_heads: Recombine separated attention heads.
        forward: Compute attention output for given query, key, and value tensors.

    Examples:
        >>> attn = Attention(embedding_dim=256, num_heads=8, downsample_rate=2)
        >>> q = torch.randn(1, 100, 256)
        >>> k = v = torch.randn(1, 50, 256)
        >>> output = attn(q, k, v)
        >>> print(output.shape)
        torch.Size([1, 100, 256])
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        kv_in_dim: int | None = None,
    ) -> None:
        """Initialize the Attention module with specified dimensions and settings.

        Args:
            embedding_dim (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads.
            downsample_rate (int, optional): Factor by which internal dimensions are downsampled.
            kv_in_dim (int | None, optional): Dimensionality of key and value inputs. If None, uses embedding_dim.

        Raises:
            AssertionError: If num_heads does not evenly divide the internal dim (embedding_dim / downsample_rate).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
```
</details>

<br>

### Method `ultralytics.models.sam.modules.transformer.Attention._recombine_heads` {#ultralytics.models.sam.modules.transformer.Attention.\_recombine\_heads}

```python
def _recombine_heads(x: Tensor) -> Tensor
```

Recombine separated attention heads into a single tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/transformer.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/transformer.py#L308-L312"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _recombine_heads(x: Tensor) -> Tensor:
    """Recombine separated attention heads into a single tensor."""
    b, n_heads, n_tokens, c_per_head = x.shape
    x = x.transpose(1, 2)
    return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C
```
</details>

<br>

### Method `ultralytics.models.sam.modules.transformer.Attention._separate_heads` {#ultralytics.models.sam.modules.transformer.Attention.\_separate\_heads}

```python
def _separate_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor
```

Separate the input tensor into the specified number of attention heads.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |
| `num_heads` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/transformer.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/transformer.py#L301-L305"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _separate_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Separate the input tensor into the specified number of attention heads."""
    b, n, c = x.shape
    x = x.reshape(b, n, num_heads, c // num_heads)
    return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head
```
</details>

<br>

### Method `ultralytics.models.sam.modules.transformer.Attention.forward` {#ultralytics.models.sam.modules.transformer.Attention.forward}

```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor
```

Apply multi-head attention to query, key, and value tensors with optional downsampling.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `q` | `torch.Tensor` | Query tensor with shape (B, N_q, embedding_dim). | *required* |
| `k` | `torch.Tensor` | Key tensor with shape (B, N_k, kv_in_dim). | *required* |
| `v` | `torch.Tensor` | Value tensor with shape (B, N_k, kv_in_dim). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after attention with shape (B, N_q, embedding_dim). |

<details>
<summary>Source code in <code>ultralytics/models/sam/modules/transformer.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/modules/transformer.py#L314-L344"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply multi-head attention to query, key, and value tensors with optional downsampling.

    Args:
        q (torch.Tensor): Query tensor with shape (B, N_q, embedding_dim).
        k (torch.Tensor): Key tensor with shape (B, N_k, kv_in_dim).
        v (torch.Tensor): Value tensor with shape (B, N_k, kv_in_dim).

    Returns:
        (torch.Tensor): Output tensor after attention with shape (B, N_q, embedding_dim).
    """
    # Input projections
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    # Separate into heads
    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)

    # Attention
    _, _, _, c_per_head = q.shape
    attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
    attn = attn / math.sqrt(c_per_head)
    attn = torch.softmax(attn, dim=-1)

    # Get output
    out = attn @ v
    out = self._recombine_heads(out)
    return self.out_proj(out)
```
</details>

<br><br>
