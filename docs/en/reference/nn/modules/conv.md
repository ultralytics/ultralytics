---
description: Explore detailed documentation on convolution modules like Conv, LightConv, GhostConv, and more used in Ultralytics models.
keywords: Ultralytics, convolution modules, Conv, LightConv, GhostConv, YOLO, deep learning, neural networks
---

# Reference for `ultralytics/nn/modules/conv.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`Conv`](#ultralytics.nn.modules.conv.Conv)
        - [`Conv2`](#ultralytics.nn.modules.conv.Conv2)
        - [`LightConv`](#ultralytics.nn.modules.conv.LightConv)
        - [`DWConv`](#ultralytics.nn.modules.conv.DWConv)
        - [`DWConvTranspose2d`](#ultralytics.nn.modules.conv.DWConvTranspose2d)
        - [`ConvTranspose`](#ultralytics.nn.modules.conv.ConvTranspose)
        - [`Focus`](#ultralytics.nn.modules.conv.Focus)
        - [`GhostConv`](#ultralytics.nn.modules.conv.GhostConv)
        - [`RepConv`](#ultralytics.nn.modules.conv.RepConv)
        - [`ChannelAttention`](#ultralytics.nn.modules.conv.ChannelAttention)
        - [`SpatialAttention`](#ultralytics.nn.modules.conv.SpatialAttention)
        - [`CBAM`](#ultralytics.nn.modules.conv.CBAM)
        - [`Concat`](#ultralytics.nn.modules.conv.Concat)
        - [`Index`](#ultralytics.nn.modules.conv.Index)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`Conv.forward`](#ultralytics.nn.modules.conv.Conv.forward)
        - [`Conv.forward_fuse`](#ultralytics.nn.modules.conv.Conv.forward_fuse)
        - [`Conv2.forward`](#ultralytics.nn.modules.conv.Conv2.forward)
        - [`Conv2.forward_fuse`](#ultralytics.nn.modules.conv.Conv2.forward_fuse)
        - [`Conv2.fuse_convs`](#ultralytics.nn.modules.conv.Conv2.fuse_convs)
        - [`LightConv.forward`](#ultralytics.nn.modules.conv.LightConv.forward)
        - [`ConvTranspose.forward`](#ultralytics.nn.modules.conv.ConvTranspose.forward)
        - [`ConvTranspose.forward_fuse`](#ultralytics.nn.modules.conv.ConvTranspose.forward_fuse)
        - [`Focus.forward`](#ultralytics.nn.modules.conv.Focus.forward)
        - [`GhostConv.forward`](#ultralytics.nn.modules.conv.GhostConv.forward)
        - [`RepConv.forward_fuse`](#ultralytics.nn.modules.conv.RepConv.forward_fuse)
        - [`RepConv.forward`](#ultralytics.nn.modules.conv.RepConv.forward)
        - [`RepConv.get_equivalent_kernel_bias`](#ultralytics.nn.modules.conv.RepConv.get_equivalent_kernel_bias)
        - [`RepConv._pad_1x1_to_3x3_tensor`](#ultralytics.nn.modules.conv.RepConv._pad_1x1_to_3x3_tensor)
        - [`RepConv._fuse_bn_tensor`](#ultralytics.nn.modules.conv.RepConv._fuse_bn_tensor)
        - [`RepConv.fuse_convs`](#ultralytics.nn.modules.conv.RepConv.fuse_convs)
        - [`ChannelAttention.forward`](#ultralytics.nn.modules.conv.ChannelAttention.forward)
        - [`SpatialAttention.forward`](#ultralytics.nn.modules.conv.SpatialAttention.forward)
        - [`CBAM.forward`](#ultralytics.nn.modules.conv.CBAM.forward)
        - [`Concat.forward`](#ultralytics.nn.modules.conv.Concat.forward)
        - [`Index.forward`](#ultralytics.nn.modules.conv.Index.forward)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`autopad`](#ultralytics.nn.modules.conv.autopad)


## Class `ultralytics.nn.modules.conv.Conv` {#ultralytics.nn.modules.conv.Conv}

```python
Conv(self, c1, c2, k = 1, s = 1, p = None, g = 1, d = 1, act = True)
```

**Bases:** `nn.Module`

Standard convolution module with batch normalization and activation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `k` | `int` | Kernel size. | `1` |
| `s` | `int` | Stride. | `1` |
| `p` | `int, optional` | Padding. | `None` |
| `g` | `int` | Groups. | `1` |
| `d` | `int` | Dilation. | `1` |
| `act` | `bool | nn.Module` | Activation function. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `conv` | `nn.Conv2d` | Convolutional layer. |
| `bn` | `nn.BatchNorm2d` | Batch normalization layer. |
| `act` | `nn.Module` | Activation function layer. |
| `default_act` | `nn.Module` | Default activation function (SiLU). |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.conv.Conv.forward) | Apply convolution, batch normalization and activation to input tensor. |
| [`forward_fuse`](#ultralytics.nn.modules.conv.Conv.forward_fuse) | Apply convolution and activation without batch normalization. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L39-L89"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.Conv.forward` {#ultralytics.nn.modules.conv.Conv.forward}

```python
def forward(self, x)
```

Apply convolution, batch normalization and activation to input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L69-L78"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x):
    """Apply convolution, batch normalization and activation to input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor.
    """
    return self.act(self.bn(self.conv(x)))
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.Conv.forward_fuse` {#ultralytics.nn.modules.conv.Conv.forward\_fuse}

```python
def forward_fuse(self, x)
```

Apply convolution and activation without batch normalization.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L80-L89"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward_fuse(self, x):
    """Apply convolution and activation without batch normalization.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor.
    """
    return self.act(self.conv(x))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.Conv2` {#ultralytics.nn.modules.conv.Conv2}

```python
Conv2(self, c1, c2, k = 3, s = 1, p = None, g = 1, d = 1, act = True)
```

**Bases:** `Conv`

Simplified RepConv module with Conv fusing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `k` | `int` | Kernel size. | `3` |
| `s` | `int` | Stride. | `1` |
| `p` | `int, optional` | Padding. | `None` |
| `g` | `int` | Groups. | `1` |
| `d` | `int` | Dilation. | `1` |
| `act` | `bool | nn.Module` | Activation function. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `conv` | `nn.Conv2d` | Main 3x3 convolutional layer. |
| `cv2` | `nn.Conv2d` | Additional 1x1 convolutional layer. |
| `bn` | `nn.BatchNorm2d` | Batch normalization layer. |
| `act` | `nn.Module` | Activation function layer. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.conv.Conv2.forward) | Apply convolution, batch normalization and activation to input tensor. |
| [`forward_fuse`](#ultralytics.nn.modules.conv.Conv2.forward_fuse) | Apply fused convolution, batch normalization and activation to input tensor. |
| [`fuse_convs`](#ultralytics.nn.modules.conv.Conv2.fuse_convs) | Fuse parallel convolutions. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L92-L147"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.Conv2.forward` {#ultralytics.nn.modules.conv.Conv2.forward}

```python
def forward(self, x)
```

Apply convolution, batch normalization and activation to input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L118-L127"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x):
    """Apply convolution, batch normalization and activation to input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor.
    """
    return self.act(self.bn(self.conv(x) + self.cv2(x)))
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.Conv2.forward_fuse` {#ultralytics.nn.modules.conv.Conv2.forward\_fuse}

```python
def forward_fuse(self, x)
```

Apply fused convolution, batch normalization and activation to input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L129-L138"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward_fuse(self, x):
    """Apply fused convolution, batch normalization and activation to input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor.
    """
    return self.act(self.bn(self.conv(x)))
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.Conv2.fuse_convs` {#ultralytics.nn.modules.conv.Conv2.fuse\_convs}

```python
def fuse_convs(self)
```

Fuse parallel convolutions.

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L140-L147"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse_convs(self):
    """Fuse parallel convolutions."""
    w = torch.zeros_like(self.conv.weight.data)
    i = [x // 2 for x in w.shape[2:]]
    w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
    self.conv.weight.data += w
    self.__delattr__("cv2")
    self.forward = self.forward_fuse
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.LightConv` {#ultralytics.nn.modules.conv.LightConv}

```python
LightConv(self, c1, c2, k = 1, act = nn.ReLU())
```

**Bases:** `nn.Module`

Light convolution module with 1x1 and depthwise convolutions.

This implementation is based on the PaddleDetection HGNetV2 backbone.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `k` | `int` | Kernel size for depthwise convolution. | `1` |
| `act` | `nn.Module` | Activation function. | `nn.ReLU()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `conv1` | `Conv` | 1x1 convolution layer. |
| `conv2` | `DWConv` | Depthwise convolution layer. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.conv.LightConv.forward) | Apply 2 convolutions to input tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L150-L182"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.LightConv.forward` {#ultralytics.nn.modules.conv.LightConv.forward}

```python
def forward(self, x)
```

Apply 2 convolutions to input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L173-L182"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x):
    """Apply 2 convolutions to input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor.
    """
    return self.conv2(self.conv1(x))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.DWConv` {#ultralytics.nn.modules.conv.DWConv}

```python
DWConv(self, c1, c2, k = 1, s = 1, d = 1, act = True)
```

**Bases:** `Conv`

Depth-wise convolution module.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `k` | `int` | Kernel size. | `1` |
| `s` | `int` | Stride. | `1` |
| `d` | `int` | Dilation. | `1` |
| `act` | `bool | nn.Module` | Activation function. | `True` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L185-L199"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.DWConvTranspose2d` {#ultralytics.nn.modules.conv.DWConvTranspose2d}

```python
DWConvTranspose2d(self, c1, c2, k = 1, s = 1, p1 = 0, p2 = 0)
```

**Bases:** `nn.ConvTranspose2d`

Depth-wise transpose convolution module.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `k` | `int` | Kernel size. | `1` |
| `s` | `int` | Stride. | `1` |
| `p1` | `int` | Padding. | `0` |
| `p2` | `int` | Output padding. | `0` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L202-L216"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.ConvTranspose` {#ultralytics.nn.modules.conv.ConvTranspose}

```python
ConvTranspose(self, c1, c2, k = 2, s = 2, p = 0, bn = True, act = True)
```

**Bases:** `nn.Module`

Convolution transpose module with optional batch normalization and activation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `k` | `int` | Kernel size. | `2` |
| `s` | `int` | Stride. | `2` |
| `p` | `int` | Padding. | `0` |
| `bn` | `bool` | Use batch normalization. | `True` |
| `act` | `bool | nn.Module` | Activation function. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `conv_transpose` | `nn.ConvTranspose2d` | Transposed convolution layer. |
| `bn` | `nn.BatchNorm2d | nn.Identity` | Batch normalization layer. |
| `act` | `nn.Module` | Activation function layer. |
| `default_act` | `nn.Module` | Default activation function (SiLU). |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.conv.ConvTranspose.forward) | Apply transposed convolution, batch normalization and activation to input. |
| [`forward_fuse`](#ultralytics.nn.modules.conv.ConvTranspose.forward_fuse) | Apply convolution transpose and activation to input. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L219-L268"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.ConvTranspose.forward` {#ultralytics.nn.modules.conv.ConvTranspose.forward}

```python
def forward(self, x)
```

Apply transposed convolution, batch normalization and activation to input.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L248-L257"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x):
    """Apply transposed convolution, batch normalization and activation to input.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor.
    """
    return self.act(self.bn(self.conv_transpose(x)))
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.ConvTranspose.forward_fuse` {#ultralytics.nn.modules.conv.ConvTranspose.forward\_fuse}

```python
def forward_fuse(self, x)
```

Apply convolution transpose and activation to input.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L259-L268"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward_fuse(self, x):
    """Apply convolution transpose and activation to input.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor.
    """
    return self.act(self.conv_transpose(x))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.Focus` {#ultralytics.nn.modules.conv.Focus}

```python
Focus(self, c1, c2, k = 1, s = 1, p = None, g = 1, act = True)
```

**Bases:** `nn.Module`

Focus module for concentrating feature information.

Slices input tensor into 4 parts and concatenates them in the channel dimension.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `k` | `int` | Kernel size. | `1` |
| `s` | `int` | Stride. | `1` |
| `p` | `int, optional` | Padding. | `None` |
| `g` | `int` | Groups. | `1` |
| `act` | `bool | nn.Module` | Activation function. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `conv` | `Conv` | Convolution layer. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.conv.Focus.forward) | Apply Focus operation and convolution to input tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L271-L307"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.Focus.forward` {#ultralytics.nn.modules.conv.Focus.forward}

```python
def forward(self, x)
```

Apply Focus operation and convolution to input tensor.

Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L296-L307"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x):
    """Apply Focus operation and convolution to input tensor.

    Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor.
    """
    return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.GhostConv` {#ultralytics.nn.modules.conv.GhostConv}

```python
GhostConv(self, c1, c2, k = 1, s = 1, g = 1, act = True)
```

**Bases:** `nn.Module`

Ghost Convolution module.

Generates more features with fewer parameters by using cheap operations.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `k` | `int` | Kernel size. | `1` |
| `s` | `int` | Stride. | `1` |
| `g` | `int` | Groups. | `1` |
| `act` | `bool | nn.Module` | Activation function. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `cv1` | `Conv` | Primary convolution. |
| `cv2` | `Conv` | Cheap operation convolution. |
| `References` |  |  |
| `https` |  | //github.com/huawei-noah/Efficient-AI-Backbones |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.conv.GhostConv.forward) | Apply Ghost Convolution to input tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L311-L350"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.GhostConv.forward` {#ultralytics.nn.modules.conv.GhostConv.forward}

```python
def forward(self, x)
```

Apply Ghost Convolution to input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor with concatenated features. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L340-L350"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x):
    """Apply Ghost Convolution to input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor with concatenated features.
    """
    y = self.cv1(x)
    return torch.cat((y, self.cv2(y)), 1)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.RepConv` {#ultralytics.nn.modules.conv.RepConv}

```python
RepConv(self, c1, c2, k = 3, s = 1, p = 1, g = 1, d = 1, act = True, bn = False, deploy = False)
```

**Bases:** `nn.Module`

RepConv module with training and deploy modes.

This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `k` | `int` | Kernel size. | `3` |
| `s` | `int` | Stride. | `1` |
| `p` | `int` | Padding. | `1` |
| `g` | `int` | Groups. | `1` |
| `d` | `int` | Dilation. | `1` |
| `act` | `bool | nn.Module` | Activation function. | `True` |
| `bn` | `bool` | Use batch normalization for identity branch. | `False` |
| `deploy` | `bool` | Deploy mode for inference. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `conv1` | `Conv` | 3x3 convolution. |
| `conv2` | `Conv` | 1x1 convolution. |
| `bn` | `nn.BatchNorm2d, optional` | Batch normalization for identity branch. |
| `act` | `nn.Module` | Activation function. |
| `default_act` | `nn.Module` | Default activation function (SiLU). |
| `References` |  |  |
| `https` |  | //github.com/DingXiaoH/RepVGG/blob/main/repvgg.py |

**Methods**

| Name | Description |
| --- | --- |
| [`_fuse_bn_tensor`](#ultralytics.nn.modules.conv.RepConv._fuse_bn_tensor) | Fuse batch normalization with convolution weights. |
| [`_pad_1x1_to_3x3_tensor`](#ultralytics.nn.modules.conv.RepConv._pad_1x1_to_3x3_tensor) | Pad a 1x1 kernel to 3x3 size. |
| [`forward`](#ultralytics.nn.modules.conv.RepConv.forward) | Forward pass for training mode. |
| [`forward_fuse`](#ultralytics.nn.modules.conv.RepConv.forward_fuse) | Forward pass for deploy mode. |
| [`fuse_convs`](#ultralytics.nn.modules.conv.RepConv.fuse_convs) | Fuse convolutions for inference by creating a single equivalent convolution. |
| [`get_equivalent_kernel_bias`](#ultralytics.nn.modules.conv.RepConv.get_equivalent_kernel_bias) | Calculate equivalent kernel and bias by fusing convolutions. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L353-L509"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.RepConv._fuse_bn_tensor` {#ultralytics.nn.modules.conv.RepConv.\_fuse\_bn\_tensor}

```python
def _fuse_bn_tensor(self, branch)
```

Fuse batch normalization with convolution weights.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `branch` | `Conv | nn.BatchNorm2d | None` | Branch to fuse. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `kernel (torch.Tensor)` | Fused kernel. |
| `bias (torch.Tensor)` | Fused bias. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L447-L481"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _fuse_bn_tensor(self, branch):
    """Fuse batch normalization with convolution weights.

    Args:
        branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

    Returns:
        kernel (torch.Tensor): Fused kernel.
        bias (torch.Tensor): Fused bias.
    """
    if branch is None:
        return 0, 0
    if isinstance(branch, Conv):
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
    elif isinstance(branch, nn.BatchNorm2d):
        if not hasattr(self, "id_tensor"):
            input_dim = self.c1 // self.g
            kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.c1):
                kernel_value[i, i % input_dim, 1, 1] = 1
            self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
        kernel = self.id_tensor
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.RepConv._pad_1x1_to_3x3_tensor` {#ultralytics.nn.modules.conv.RepConv.\_pad\_1x1\_to\_3x3\_tensor}

```python
def _pad_1x1_to_3x3_tensor(kernel1x1)
```

Pad a 1x1 kernel to 3x3 size.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `kernel1x1` | `torch.Tensor` | 1x1 convolution kernel. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Padded 3x3 kernel. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L433-L445"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _pad_1x1_to_3x3_tensor(kernel1x1):
    """Pad a 1x1 kernel to 3x3 size.

    Args:
        kernel1x1 (torch.Tensor): 1x1 convolution kernel.

    Returns:
        (torch.Tensor): Padded 3x3 kernel.
    """
    if kernel1x1 is None:
        return 0
    else:
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.RepConv.forward` {#ultralytics.nn.modules.conv.RepConv.forward}

```python
def forward(self, x)
```

Forward pass for training mode.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L408-L418"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x):
    """Forward pass for training mode.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor.
    """
    id_out = 0 if self.bn is None else self.bn(x)
    return self.act(self.conv1(x) + self.conv2(x) + id_out)
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.RepConv.forward_fuse` {#ultralytics.nn.modules.conv.RepConv.forward\_fuse}

```python
def forward_fuse(self, x)
```

Forward pass for deploy mode.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L397-L406"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward_fuse(self, x):
    """Forward pass for deploy mode.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor.
    """
    return self.act(self.conv(x))
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.RepConv.fuse_convs` {#ultralytics.nn.modules.conv.RepConv.fuse\_convs}

```python
def fuse_convs(self)
```

Fuse convolutions for inference by creating a single equivalent convolution.

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L483-L509"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse_convs(self):
    """Fuse convolutions for inference by creating a single equivalent convolution."""
    if hasattr(self, "conv"):
        return
    kernel, bias = self.get_equivalent_kernel_bias()
    self.conv = nn.Conv2d(
        in_channels=self.conv1.conv.in_channels,
        out_channels=self.conv1.conv.out_channels,
        kernel_size=self.conv1.conv.kernel_size,
        stride=self.conv1.conv.stride,
        padding=self.conv1.conv.padding,
        dilation=self.conv1.conv.dilation,
        groups=self.conv1.conv.groups,
        bias=True,
    ).requires_grad_(False)
    self.conv.weight.data = kernel
    self.conv.bias.data = bias
    for para in self.parameters():
        para.detach_()
    self.__delattr__("conv1")
    self.__delattr__("conv2")
    if hasattr(self, "nm"):
        self.__delattr__("nm")
    if hasattr(self, "bn"):
        self.__delattr__("bn")
    if hasattr(self, "id_tensor"):
        self.__delattr__("id_tensor")
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.RepConv.get_equivalent_kernel_bias` {#ultralytics.nn.modules.conv.RepConv.get\_equivalent\_kernel\_bias}

```python
def get_equivalent_kernel_bias(self)
```

Calculate equivalent kernel and bias by fusing convolutions.

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Equivalent kernel |
| `torch.Tensor` | Equivalent bias |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L420-L430"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_equivalent_kernel_bias(self):
    """Calculate equivalent kernel and bias by fusing convolutions.

    Returns:
        (torch.Tensor): Equivalent kernel
        (torch.Tensor): Equivalent bias
    """
    kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
    kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
    kernelid, biasid = self._fuse_bn_tensor(self.bn)
    return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.ChannelAttention` {#ultralytics.nn.modules.conv.ChannelAttention}

```python
ChannelAttention(self, channels: int) -> None
```

**Bases:** `nn.Module`

Channel-attention module for feature recalibration.

Applies attention weights to channels based on global average pooling.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `channels` | `int` | Number of input channels. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `pool` | `nn.AdaptiveAvgPool2d` | Global average pooling. |
| `fc` | `nn.Conv2d` | Fully connected layer implemented as 1x1 convolution. |
| `act` | `nn.Sigmoid` | Sigmoid activation for attention weights. |
| `References` |  |  |
| `https` |  | //github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.conv.ChannelAttention.forward) | Apply channel attention to input tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L512-L546"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.ChannelAttention.forward` {#ultralytics.nn.modules.conv.ChannelAttention.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply channel attention to input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Channel-attended output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L537-L546"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply channel attention to input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Channel-attended output tensor.
    """
    return x * self.act(self.fc(self.pool(x)))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.SpatialAttention` {#ultralytics.nn.modules.conv.SpatialAttention}

```python
SpatialAttention(self, kernel_size = 7)
```

**Bases:** `nn.Module`

Spatial-attention module for feature recalibration.

Applies attention weights to spatial dimensions based on channel statistics.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `kernel_size` | `int` | Size of the convolutional kernel (3 or 7). | `7` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `cv1` | `nn.Conv2d` | Convolution layer for spatial attention. |
| `act` | `nn.Sigmoid` | Sigmoid activation for attention weights. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.conv.SpatialAttention.forward) | Apply spatial attention to input tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L549-L580"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.SpatialAttention.forward` {#ultralytics.nn.modules.conv.SpatialAttention.forward}

```python
def forward(self, x)
```

Apply spatial attention to input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Spatial-attended output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L571-L580"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x):
    """Apply spatial attention to input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Spatial-attended output tensor.
    """
    return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.CBAM` {#ultralytics.nn.modules.conv.CBAM}

```python
CBAM(self, c1, kernel_size = 7)
```

**Bases:** `nn.Module`

Convolutional Block Attention Module.

Combines channel and spatial attention mechanisms for comprehensive feature refinement.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `kernel_size` | `int` | Size of the convolutional kernel for spatial attention. | `7` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `channel_attention` | `ChannelAttention` | Channel attention module. |
| `spatial_attention` | `SpatialAttention` | Spatial attention module. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.conv.CBAM.forward) | Apply channel and spatial attention sequentially to input tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L583-L613"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.CBAM.forward` {#ultralytics.nn.modules.conv.CBAM.forward}

```python
def forward(self, x)
```

Apply channel and spatial attention sequentially to input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Attended output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L604-L613"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x):
    """Apply channel and spatial attention sequentially to input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Attended output tensor.
    """
    return self.spatial_attention(self.channel_attention(x))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.Concat` {#ultralytics.nn.modules.conv.Concat}

```python
Concat(self, dimension = 1)
```

**Bases:** `nn.Module`

Concatenate a list of tensors along specified dimension.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dimension` | `int` | Dimension along which to concatenate tensors. | `1` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `d` | `int` | Dimension along which to concatenate tensors. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.conv.Concat.forward) | Concatenate input tensors along specified dimension. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L616-L641"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.Concat.forward` {#ultralytics.nn.modules.conv.Concat.forward}

```python
def forward(self, x: list[torch.Tensor])
```

Concatenate input tensors along specified dimension.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` | List of input tensors. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Concatenated tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L632-L641"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: list[torch.Tensor]):
    """Concatenate input tensors along specified dimension.

    Args:
        x (list[torch.Tensor]): List of input tensors.

    Returns:
        (torch.Tensor): Concatenated tensor.
    """
    return torch.cat(x, self.d)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.conv.Index` {#ultralytics.nn.modules.conv.Index}

```python
Index(self, index = 0)
```

**Bases:** `nn.Module`

Returns a particular index of the input.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `index` | `int` | Index to select from input. | `0` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `int` | Index to select from input. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.conv.Index.forward) | Select and return a particular index from input. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L644-L669"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index
```
</details>

<br>

### Method `ultralytics.nn.modules.conv.Index.forward` {#ultralytics.nn.modules.conv.Index.forward}

```python
def forward(self, x: list[torch.Tensor])
```

Select and return a particular index from input.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` | List of input tensors. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Selected tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L660-L669"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: list[torch.Tensor]):
    """Select and return a particular index from input.

    Args:
        x (list[torch.Tensor]): List of input tensors.

    Returns:
        (torch.Tensor): Selected tensor.
    """
    return x[self.index]
```
</details>


<br><br><hr><br>

## Function `ultralytics.nn.modules.conv.autopad` {#ultralytics.nn.modules.conv.autopad}

```python
def autopad(k, p = None, d = 1)
```

Pad to 'same' shape outputs.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `k` |  |  | *required* |
| `p` |  |  | `None` |
| `d` |  |  | `1` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/conv.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py#L30-L36"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
```
</details>

<br><br>
