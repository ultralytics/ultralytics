---
description: Explore detailed documentation of block modules in Ultralytics, available for deep learning tasks. Contribute and improve the codebase.
keywords: Ultralytics, YOLO, neural networks, block modules, DFL, Proto, HGStem, HGBlock, SPP, SPPF, C1, C2, C2f, C3, C3x, RepC3, C3TR, C3Ghost, GhostBottleneck, Bottleneck, BottleneckCSP, ResNetBlock, MaxSigmoidAttnBlock, ImagePoolingAttn, ContrastiveHead, RepBottleneck, RepCSP, RepNCSPELAN4, ADown, SPPELAN, Silence, CBLinear, CBFuse
---

# Reference for `ultralytics/nn/modules/block.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`DFL`](#ultralytics.nn.modules.block.DFL)
        - [`Proto`](#ultralytics.nn.modules.block.Proto)
        - [`HGStem`](#ultralytics.nn.modules.block.HGStem)
        - [`HGBlock`](#ultralytics.nn.modules.block.HGBlock)
        - [`SPP`](#ultralytics.nn.modules.block.SPP)
        - [`SPPF`](#ultralytics.nn.modules.block.SPPF)
        - [`C1`](#ultralytics.nn.modules.block.C1)
        - [`C2`](#ultralytics.nn.modules.block.C2)
        - [`C2f`](#ultralytics.nn.modules.block.C2f)
        - [`C3`](#ultralytics.nn.modules.block.C3)
        - [`C3x`](#ultralytics.nn.modules.block.C3x)
        - [`RepC3`](#ultralytics.nn.modules.block.RepC3)
        - [`C3TR`](#ultralytics.nn.modules.block.C3TR)
        - [`C3Ghost`](#ultralytics.nn.modules.block.C3Ghost)
        - [`GhostBottleneck`](#ultralytics.nn.modules.block.GhostBottleneck)
        - [`Bottleneck`](#ultralytics.nn.modules.block.Bottleneck)
        - [`BottleneckCSP`](#ultralytics.nn.modules.block.BottleneckCSP)
        - [`ResNetBlock`](#ultralytics.nn.modules.block.ResNetBlock)
        - [`ResNetLayer`](#ultralytics.nn.modules.block.ResNetLayer)
        - [`MaxSigmoidAttnBlock`](#ultralytics.nn.modules.block.MaxSigmoidAttnBlock)
        - [`C2fAttn`](#ultralytics.nn.modules.block.C2fAttn)
        - [`ImagePoolingAttn`](#ultralytics.nn.modules.block.ImagePoolingAttn)
        - [`ContrastiveHead`](#ultralytics.nn.modules.block.ContrastiveHead)
        - [`BNContrastiveHead`](#ultralytics.nn.modules.block.BNContrastiveHead)
        - [`RepBottleneck`](#ultralytics.nn.modules.block.RepBottleneck)
        - [`RepCSP`](#ultralytics.nn.modules.block.RepCSP)
        - [`RepNCSPELAN4`](#ultralytics.nn.modules.block.RepNCSPELAN4)
        - [`ELAN1`](#ultralytics.nn.modules.block.ELAN1)
        - [`AConv`](#ultralytics.nn.modules.block.AConv)
        - [`ADown`](#ultralytics.nn.modules.block.ADown)
        - [`SPPELAN`](#ultralytics.nn.modules.block.SPPELAN)
        - [`CBLinear`](#ultralytics.nn.modules.block.CBLinear)
        - [`CBFuse`](#ultralytics.nn.modules.block.CBFuse)
        - [`C3f`](#ultralytics.nn.modules.block.C3f)
        - [`C3k2`](#ultralytics.nn.modules.block.C3k2)
        - [`C3k`](#ultralytics.nn.modules.block.C3k)
        - [`RepVGGDW`](#ultralytics.nn.modules.block.RepVGGDW)
        - [`CIB`](#ultralytics.nn.modules.block.CIB)
        - [`C2fCIB`](#ultralytics.nn.modules.block.C2fCIB)
        - [`Attention`](#ultralytics.nn.modules.block.Attention)
        - [`PSABlock`](#ultralytics.nn.modules.block.PSABlock)
        - [`PSA`](#ultralytics.nn.modules.block.PSA)
        - [`C2PSA`](#ultralytics.nn.modules.block.C2PSA)
        - [`C2fPSA`](#ultralytics.nn.modules.block.C2fPSA)
        - [`SCDown`](#ultralytics.nn.modules.block.SCDown)
        - [`TorchVision`](#ultralytics.nn.modules.block.TorchVision)
        - [`AAttn`](#ultralytics.nn.modules.block.AAttn)
        - [`ABlock`](#ultralytics.nn.modules.block.ABlock)
        - [`A2C2f`](#ultralytics.nn.modules.block.A2C2f)
        - [`SwiGLUFFN`](#ultralytics.nn.modules.block.SwiGLUFFN)
        - [`Residual`](#ultralytics.nn.modules.block.Residual)
        - [`SAVPE`](#ultralytics.nn.modules.block.SAVPE)
        - [`Proto26`](#ultralytics.nn.modules.block.Proto26)
        - [`RealNVP`](#ultralytics.nn.modules.block.RealNVP)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`RealNVP.prior`](#ultralytics.nn.modules.block.RealNVP.prior)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`DFL.forward`](#ultralytics.nn.modules.block.DFL.forward)
        - [`Proto.forward`](#ultralytics.nn.modules.block.Proto.forward)
        - [`HGStem.forward`](#ultralytics.nn.modules.block.HGStem.forward)
        - [`HGBlock.forward`](#ultralytics.nn.modules.block.HGBlock.forward)
        - [`SPP.forward`](#ultralytics.nn.modules.block.SPP.forward)
        - [`SPPF.forward`](#ultralytics.nn.modules.block.SPPF.forward)
        - [`C1.forward`](#ultralytics.nn.modules.block.C1.forward)
        - [`C2.forward`](#ultralytics.nn.modules.block.C2.forward)
        - [`C2f.forward`](#ultralytics.nn.modules.block.C2f.forward)
        - [`C2f.forward_split`](#ultralytics.nn.modules.block.C2f.forward_split)
        - [`C3.forward`](#ultralytics.nn.modules.block.C3.forward)
        - [`RepC3.forward`](#ultralytics.nn.modules.block.RepC3.forward)
        - [`GhostBottleneck.forward`](#ultralytics.nn.modules.block.GhostBottleneck.forward)
        - [`Bottleneck.forward`](#ultralytics.nn.modules.block.Bottleneck.forward)
        - [`BottleneckCSP.forward`](#ultralytics.nn.modules.block.BottleneckCSP.forward)
        - [`ResNetBlock.forward`](#ultralytics.nn.modules.block.ResNetBlock.forward)
        - [`ResNetLayer.forward`](#ultralytics.nn.modules.block.ResNetLayer.forward)
        - [`MaxSigmoidAttnBlock.forward`](#ultralytics.nn.modules.block.MaxSigmoidAttnBlock.forward)
        - [`C2fAttn.forward`](#ultralytics.nn.modules.block.C2fAttn.forward)
        - [`C2fAttn.forward_split`](#ultralytics.nn.modules.block.C2fAttn.forward_split)
        - [`ImagePoolingAttn.forward`](#ultralytics.nn.modules.block.ImagePoolingAttn.forward)
        - [`ContrastiveHead.forward`](#ultralytics.nn.modules.block.ContrastiveHead.forward)
        - [`BNContrastiveHead.fuse`](#ultralytics.nn.modules.block.BNContrastiveHead.fuse)
        - [`BNContrastiveHead.forward_fuse`](#ultralytics.nn.modules.block.BNContrastiveHead.forward_fuse)
        - [`BNContrastiveHead.forward`](#ultralytics.nn.modules.block.BNContrastiveHead.forward)
        - [`RepNCSPELAN4.forward`](#ultralytics.nn.modules.block.RepNCSPELAN4.forward)
        - [`RepNCSPELAN4.forward_split`](#ultralytics.nn.modules.block.RepNCSPELAN4.forward_split)
        - [`AConv.forward`](#ultralytics.nn.modules.block.AConv.forward)
        - [`ADown.forward`](#ultralytics.nn.modules.block.ADown.forward)
        - [`SPPELAN.forward`](#ultralytics.nn.modules.block.SPPELAN.forward)
        - [`CBLinear.forward`](#ultralytics.nn.modules.block.CBLinear.forward)
        - [`CBFuse.forward`](#ultralytics.nn.modules.block.CBFuse.forward)
        - [`C3f.forward`](#ultralytics.nn.modules.block.C3f.forward)
        - [`RepVGGDW.forward`](#ultralytics.nn.modules.block.RepVGGDW.forward)
        - [`RepVGGDW.forward_fuse`](#ultralytics.nn.modules.block.RepVGGDW.forward_fuse)
        - [`RepVGGDW.fuse`](#ultralytics.nn.modules.block.RepVGGDW.fuse)
        - [`CIB.forward`](#ultralytics.nn.modules.block.CIB.forward)
        - [`Attention.forward`](#ultralytics.nn.modules.block.Attention.forward)
        - [`PSABlock.forward`](#ultralytics.nn.modules.block.PSABlock.forward)
        - [`PSA.forward`](#ultralytics.nn.modules.block.PSA.forward)
        - [`C2PSA.forward`](#ultralytics.nn.modules.block.C2PSA.forward)
        - [`SCDown.forward`](#ultralytics.nn.modules.block.SCDown.forward)
        - [`TorchVision.forward`](#ultralytics.nn.modules.block.TorchVision.forward)
        - [`AAttn.forward`](#ultralytics.nn.modules.block.AAttn.forward)
        - [`ABlock._init_weights`](#ultralytics.nn.modules.block.ABlock._init_weights)
        - [`ABlock.forward`](#ultralytics.nn.modules.block.ABlock.forward)
        - [`A2C2f.forward`](#ultralytics.nn.modules.block.A2C2f.forward)
        - [`SwiGLUFFN.forward`](#ultralytics.nn.modules.block.SwiGLUFFN.forward)
        - [`Residual.forward`](#ultralytics.nn.modules.block.Residual.forward)
        - [`SAVPE.forward`](#ultralytics.nn.modules.block.SAVPE.forward)
        - [`Proto26.forward`](#ultralytics.nn.modules.block.Proto26.forward)
        - [`Proto26.fuse`](#ultralytics.nn.modules.block.Proto26.fuse)
        - [`RealNVP.nets`](#ultralytics.nn.modules.block.RealNVP.nets)
        - [`RealNVP.nett`](#ultralytics.nn.modules.block.RealNVP.nett)
        - [`RealNVP.init_weights`](#ultralytics.nn.modules.block.RealNVP.init_weights)
        - [`RealNVP.backward_p`](#ultralytics.nn.modules.block.RealNVP.backward_p)
        - [`RealNVP.log_prob`](#ultralytics.nn.modules.block.RealNVP.log_prob)


## Class `ultralytics.nn.modules.block.DFL` {#ultralytics.nn.modules.block.DFL}

```python
DFL(self, c1: int = 16)
```

**Bases:** `nn.Module`

Integral module of Distribution Focal Loss (DFL).

Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | `16` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.DFL.forward) | Apply the DFL module to input tensor and return transformed output. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L58-L79"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
```
</details>

<br>

### Method `ultralytics.nn.modules.block.DFL.forward` {#ultralytics.nn.modules.block.DFL.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply the DFL module to input tensor and return transformed output.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L76-L79"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the DFL module to input tensor and return transformed output."""
    b, _, a = x.shape  # batch, channels, anchors
    return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.Proto` {#ultralytics.nn.modules.block.Proto}

```python
Proto(self, c1: int, c_: int = 256, c2: int = 32)
```

**Bases:** `nn.Module`

Ultralytics YOLO models mask Proto module for segmentation models.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c_` | `int` | Intermediate channels. | `256` |
| `c2` | `int` | Output channels (number of protos). | `32` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.Proto.forward) | Perform a forward pass through layers using an upsampled input image. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L83-L102"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.Proto.forward` {#ultralytics.nn.modules.block.Proto.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Perform a forward pass through layers using an upsampled input image.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L100-L102"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Perform a forward pass through layers using an upsampled input image."""
    return self.cv3(self.cv2(self.upsample(self.cv1(x))))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.HGStem` {#ultralytics.nn.modules.block.HGStem}

```python
HGStem(self, c1: int, cm: int, c2: int)
```

**Bases:** `nn.Module`

StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `cm` | `int` | Middle channels. | *required* |
| `c2` | `int` | Output channels. | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.HGStem.forward) | Forward pass of a PPHGNetV2 backbone layer. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L105-L138"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.HGStem.forward` {#ultralytics.nn.modules.block.HGStem.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass of a PPHGNetV2 backbone layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L127-L138"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of a PPHGNetV2 backbone layer."""
    x = self.stem1(x)
    x = F.pad(x, [0, 1, 0, 1])
    x2 = self.stem2a(x)
    x2 = F.pad(x2, [0, 1, 0, 1])
    x2 = self.stem2b(x2)
    x1 = self.pool(x)
    x = torch.cat([x1, x2], dim=1)
    x = self.stem3(x)
    x = self.stem4(x)
    return x
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.HGBlock` {#ultralytics.nn.modules.block.HGBlock}

```python
def __init__(
    self,
    c1: int,
    cm: int,
    c2: int,
    k: int = 3,
    n: int = 6,
    lightconv: bool = False,
    shortcut: bool = False,
    act: nn.Module = nn.ReLU(),
)
```

**Bases:** `nn.Module`

HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `cm` | `int` | Middle channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `k` | `int` | Kernel size. | `3` |
| `n` | `int` | Number of LightConv or Conv blocks. | `6` |
| `lightconv` | `bool` | Whether to use LightConv. | `False` |
| `shortcut` | `bool` | Whether to use shortcut connection. | `False` |
| `act` | `nn.Module` | Activation function. | `nn.ReLU()` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.HGBlock.forward) | Forward pass of a PPHGNetV2 backbone layer. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L141-L182"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2
```
</details>

<br>

### Method `ultralytics.nn.modules.block.HGBlock.forward` {#ultralytics.nn.modules.block.HGBlock.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass of a PPHGNetV2 backbone layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L177-L182"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of a PPHGNetV2 backbone layer."""
    y = [x]
    y.extend(m(y[-1]) for m in self.m)
    y = self.ec(self.sc(torch.cat(y, 1)))
    return y + x if self.add else y
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.SPP` {#ultralytics.nn.modules.block.SPP}

```python
SPP(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13))
```

**Bases:** `nn.Module`

Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `k` | `tuple` | Kernel sizes for max pooling. | `(5, 9, 13)` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.SPP.forward) | Forward pass of the SPP layer, performing spatial pyramid pooling. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L185-L205"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
```
</details>

<br>

### Method `ultralytics.nn.modules.block.SPP.forward` {#ultralytics.nn.modules.block.SPP.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass of the SPP layer, performing spatial pyramid pooling.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L202-L205"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the SPP layer, performing spatial pyramid pooling."""
    x = self.cv1(x)
    return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.SPPF` {#ultralytics.nn.modules.block.SPPF}

```python
SPPF(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False)
```

**Bases:** `nn.Module`

Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `k` | `int` | Kernel size. | `5` |
| `n` | `int` | Number of pooling iterations. | `3` |
| `shortcut` | `bool` | Whether to use shortcut connection. | `False` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.SPPF.forward) | Apply sequential pooling operations to input and return concatenated feature maps. |

!!! note "Notes"

    This module is equivalent to SPP(k=(5, 9, 13)).

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L208-L237"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2
```
</details>

<br>

### Method `ultralytics.nn.modules.block.SPPF.forward` {#ultralytics.nn.modules.block.SPPF.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply sequential pooling operations to input and return concatenated feature maps.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L232-L237"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply sequential pooling operations to input and return concatenated feature maps."""
    y = [self.cv1(x)]
    y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
    y = self.cv2(torch.cat(y, 1))
    return y + x if getattr(self, "add", False) else y
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C1` {#ultralytics.nn.modules.block.C1}

```python
C1(self, c1: int, c2: int, n: int = 1)
```

**Bases:** `nn.Module`

CSP Bottleneck with 1 convolution.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of convolutions. | `1` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.C1.forward) | Apply convolution and residual connection to input tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L240-L258"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.C1.forward` {#ultralytics.nn.modules.block.C1.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply convolution and residual connection to input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L255-L258"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply convolution and residual connection to input tensor."""
    y = self.cv1(x)
    return self.m(y) + y
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C2` {#ultralytics.nn.modules.block.C2}

```python
C2(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5)
```

**Bases:** `nn.Module`

CSP Bottleneck with 2 convolutions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of Bottleneck blocks. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `True` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.C2.forward) | Forward pass through the CSP bottleneck with 2 convolutions. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L261-L285"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.C2.forward` {#ultralytics.nn.modules.block.C2.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through the CSP bottleneck with 2 convolutions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L282-L285"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the CSP bottleneck with 2 convolutions."""
    a, b = self.cv1(x).chunk(2, 1)
    return self.cv2(torch.cat((self.m(a), b), 1))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C2f` {#ultralytics.nn.modules.block.C2f}

```python
C2f(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5)
```

**Bases:** `nn.Module`

Faster Implementation of CSP Bottleneck with 2 convolutions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of Bottleneck blocks. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `False` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.C2f.forward) | Forward pass through C2f layer. |
| [`forward_split`](#ultralytics.nn.modules.block.C2f.forward_split) | Forward pass using split() instead of chunk(). |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L288-L319"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.C2f.forward` {#ultralytics.nn.modules.block.C2f.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through C2f layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L308-L312"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through C2f layer."""
    y = list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.C2f.forward_split` {#ultralytics.nn.modules.block.C2f.forward\_split}

```python
def forward_split(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass using split() instead of chunk().

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L314-L319"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward_split(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass using split() instead of chunk()."""
    y = self.cv1(x).split((self.c, self.c), 1)
    y = [y[0], y[1]]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C3` {#ultralytics.nn.modules.block.C3}

```python
C3(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5)
```

**Bases:** `nn.Module`

CSP Bottleneck with 3 convolutions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of Bottleneck blocks. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `True` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.C3.forward) | Forward pass through the CSP bottleneck with 3 convolutions. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L322-L345"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.C3.forward` {#ultralytics.nn.modules.block.C3.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through the CSP bottleneck with 3 convolutions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L343-L345"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the CSP bottleneck with 3 convolutions."""
    return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C3x` {#ultralytics.nn.modules.block.C3x}

```python
C3x(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5)
```

**Bases:** `C3`

C3 module with cross-convolutions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of Bottleneck blocks. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `True` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L348-L364"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.RepC3` {#ultralytics.nn.modules.block.RepC3}

```python
RepC3(self, c1: int, c2: int, n: int = 3, e: float = 1.0)
```

**Bases:** `nn.Module`

Rep C3.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of RepConv blocks. | `3` |
| `e` | `float` | Expansion ratio. | `1.0` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.RepC3.forward) | Forward pass of RepC3 module. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L367-L388"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize RepC3 module with RepConv blocks.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()
```
</details>

<br>

### Method `ultralytics.nn.modules.block.RepC3.forward` {#ultralytics.nn.modules.block.RepC3.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass of RepC3 module.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L386-L388"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of RepC3 module."""
    return self.cv3(self.m(self.cv1(x)) + self.cv2(x))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C3TR` {#ultralytics.nn.modules.block.C3TR}

```python
C3TR(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5)
```

**Bases:** `C3`

C3 module with TransformerBlock().

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of Transformer blocks. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `True` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L391-L407"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C3Ghost` {#ultralytics.nn.modules.block.C3Ghost}

```python
C3Ghost(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5)
```

**Bases:** `C3`

C3 module with GhostBottleneck().

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of Ghost bottleneck blocks. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `True` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L410-L426"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.GhostBottleneck` {#ultralytics.nn.modules.block.GhostBottleneck}

```python
GhostBottleneck(self, c1: int, c2: int, k: int = 3, s: int = 1)
```

**Bases:** `nn.Module`

Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `k` | `int` | Kernel size. | `3` |
| `s` | `int` | Stride. | `1` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.GhostBottleneck.forward) | Apply skip connection and addition to input tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L429-L454"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )
```
</details>

<br>

### Method `ultralytics.nn.modules.block.GhostBottleneck.forward` {#ultralytics.nn.modules.block.GhostBottleneck.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply skip connection and addition to input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L452-L454"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply skip connection and addition to input tensor."""
    return self.conv(x) + self.shortcut(x)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.Bottleneck` {#ultralytics.nn.modules.block.Bottleneck}

```python
Bottleneck(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5)
```

**Bases:** `nn.Module`

Standard bottleneck.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `shortcut` | `bool` | Whether to use shortcut connection. | `True` |
| `g` | `int` | Groups for convolutions. | `1` |
| `k` | `tuple` | Kernel sizes for convolutions. | `(3, 3)` |
| `e` | `float` | Expansion ratio. | `0.5` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.Bottleneck.forward) | Apply bottleneck with optional shortcut connection. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L457-L481"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
```
</details>

<br>

### Method `ultralytics.nn.modules.block.Bottleneck.forward` {#ultralytics.nn.modules.block.Bottleneck.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply bottleneck with optional shortcut connection.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L479-L481"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply bottleneck with optional shortcut connection."""
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.BottleneckCSP` {#ultralytics.nn.modules.block.BottleneckCSP}

```python
BottleneckCSP(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5)
```

**Bases:** `nn.Module`

CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of Bottleneck blocks. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `True` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.BottleneckCSP.forward) | Apply CSP bottleneck with 4 convolutions. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L484-L512"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.BottleneckCSP.forward` {#ultralytics.nn.modules.block.BottleneckCSP.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply CSP bottleneck with 4 convolutions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L508-L512"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply CSP bottleneck with 4 convolutions."""
    y1 = self.cv3(self.m(self.cv1(x)))
    y2 = self.cv2(x)
    return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.ResNetBlock` {#ultralytics.nn.modules.block.ResNetBlock}

```python
ResNetBlock(self, c1: int, c2: int, s: int = 1, e: int = 4)
```

**Bases:** `nn.Module`

ResNet block with standard convolution layers.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `s` | `int` | Stride. | `1` |
| `e` | `int` | Expansion ratio. | `4` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.ResNetBlock.forward) | Forward pass through the ResNet block. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L515-L536"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()
```
</details>

<br>

### Method `ultralytics.nn.modules.block.ResNetBlock.forward` {#ultralytics.nn.modules.block.ResNetBlock.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through the ResNet block.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L534-L536"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the ResNet block."""
    return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.ResNetLayer` {#ultralytics.nn.modules.block.ResNetLayer}

```python
ResNetLayer(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4)
```

**Bases:** `nn.Module`

ResNet layer with multiple ResNet blocks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `s` | `int` | Stride. | `1` |
| `is_first` | `bool` | Whether this is the first layer. | `False` |
| `n` | `int` | Number of ResNet blocks. | `1` |
| `e` | `int` | Expansion ratio. | `4` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.ResNetLayer.forward) | Forward pass through the ResNet layer. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L539-L567"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.ResNetLayer.forward` {#ultralytics.nn.modules.block.ResNetLayer.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through the ResNet layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L565-L567"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the ResNet layer."""
    return self.layer(x)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.MaxSigmoidAttnBlock` {#ultralytics.nn.modules.block.MaxSigmoidAttnBlock}

```python
MaxSigmoidAttnBlock(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False)
```

**Bases:** `nn.Module`

Max Sigmoid attention block.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `nh` | `int` | Number of heads. | `1` |
| `ec` | `int` | Embedding channels. | `128` |
| `gc` | `int` | Guide channels. | `512` |
| `scale` | `bool` | Whether to use learnable scale parameter. | `False` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.MaxSigmoidAttnBlock.forward) | Forward pass of MaxSigmoidAttnBlock. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L570-L619"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0
```
</details>

<br>

### Method `ultralytics.nn.modules.block.MaxSigmoidAttnBlock.forward` {#ultralytics.nn.modules.block.MaxSigmoidAttnBlock.forward}

```python
def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor
```

Forward pass of MaxSigmoidAttnBlock.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |
| `guide` | `torch.Tensor` | Guide tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after attention. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L593-L619"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
    """Forward pass of MaxSigmoidAttnBlock.

    Args:
        x (torch.Tensor): Input tensor.
        guide (torch.Tensor): Guide tensor.

    Returns:
        (torch.Tensor): Output tensor after attention.
    """
    bs, _, h, w = x.shape

    guide = self.gl(guide)
    guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
    embed = self.ec(x) if self.ec is not None else x
    embed = embed.view(bs, self.nh, self.hc, h, w)

    aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
    aw = aw.max(dim=-1)[0]
    aw = aw / (self.hc**0.5)
    aw = aw + self.bias[None, :, None, None]
    aw = aw.sigmoid() * self.scale

    x = self.proj_conv(x)
    x = x.view(bs, self.nh, -1, h, w)
    x = x * aw.unsqueeze(2)
    return x.view(bs, -1, h, w)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C2fAttn` {#ultralytics.nn.modules.block.C2fAttn}

```python
def __init__(
    self,
    c1: int,
    c2: int,
    n: int = 1,
    ec: int = 128,
    nh: int = 1,
    gc: int = 512,
    shortcut: bool = False,
    g: int = 1,
    e: float = 0.5,
)
```

**Bases:** `nn.Module`

C2f module with an additional attn module.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of Bottleneck blocks. | `1` |
| `ec` | `int` | Embedding channels for attention. | `128` |
| `nh` | `int` | Number of heads for attention. | `1` |
| `gc` | `int` | Guide channels for attention. | `512` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `False` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.C2fAttn.forward) | Forward pass through C2f layer with attention. |
| [`forward_split`](#ultralytics.nn.modules.block.C2fAttn.forward_split) | Forward pass using split() instead of chunk(). |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L622-L685"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.C2fAttn.forward` {#ultralytics.nn.modules.block.C2fAttn.forward}

```python
def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor
```

Forward pass through C2f layer with attention.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |
| `guide` | `torch.Tensor` | Guide tensor for attention. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after processing. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L657-L670"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
    """Forward pass through C2f layer with attention.

    Args:
        x (torch.Tensor): Input tensor.
        guide (torch.Tensor): Guide tensor for attention.

    Returns:
        (torch.Tensor): Output tensor after processing.
    """
    y = list(self.cv1(x).chunk(2, 1))
    y.extend(m(y[-1]) for m in self.m)
    y.append(self.attn(y[-1], guide))
    return self.cv2(torch.cat(y, 1))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.C2fAttn.forward_split` {#ultralytics.nn.modules.block.C2fAttn.forward\_split}

```python
def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor
```

Forward pass using split() instead of chunk().

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |
| `guide` | `torch.Tensor` | Guide tensor for attention. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after processing. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L672-L685"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
    """Forward pass using split() instead of chunk().

    Args:
        x (torch.Tensor): Input tensor.
        guide (torch.Tensor): Guide tensor for attention.

    Returns:
        (torch.Tensor): Output tensor after processing.
    """
    y = list(self.cv1(x).split((self.c, self.c), 1))
    y.extend(m(y[-1]) for m in self.m)
    y.append(self.attn(y[-1], guide))
    return self.cv2(torch.cat(y, 1))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.ImagePoolingAttn` {#ultralytics.nn.modules.block.ImagePoolingAttn}

```python
def __init__(
    self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
)
```

**Bases:** `nn.Module`

ImagePoolingAttn: Enhance the text embeddings with image-aware information.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `ec` | `int` | Embedding channels. | `256` |
| `ch` | `tuple` | Channel dimensions for feature maps. | `()` |
| `ct` | `int` | Channel dimension for text embeddings. | `512` |
| `nh` | `int` | Number of attention heads. | `8` |
| `k` | `int` | Kernel size for pooling. | `3` |
| `scale` | `bool` | Whether to use learnable scale parameter. | `False` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.ImagePoolingAttn.forward) | Forward pass of ImagePoolingAttn. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L688-L750"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k
```
</details>

<br>

### Method `ultralytics.nn.modules.block.ImagePoolingAttn.forward` {#ultralytics.nn.modules.block.ImagePoolingAttn.forward}

```python
def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor
```

Forward pass of ImagePoolingAttn.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` | List of input feature maps. | *required* |
| `text` | `torch.Tensor` | Text embeddings. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Enhanced text embeddings. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L720-L750"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
    """Forward pass of ImagePoolingAttn.

    Args:
        x (list[torch.Tensor]): List of input feature maps.
        text (torch.Tensor): Text embeddings.

    Returns:
        (torch.Tensor): Enhanced text embeddings.
    """
    bs = x[0].shape[0]
    assert len(x) == self.nf
    num_patches = self.k**2
    x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
    x = torch.cat(x, dim=-1).transpose(1, 2)
    q = self.query(text)
    k = self.key(x)
    v = self.value(x)

    # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
    q = q.reshape(bs, -1, self.nh, self.hc)
    k = k.reshape(bs, -1, self.nh, self.hc)
    v = v.reshape(bs, -1, self.nh, self.hc)

    aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
    aw = aw / (self.hc**0.5)
    aw = F.softmax(aw, dim=-1)

    x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
    x = self.proj(x.reshape(bs, -1, self.ec))
    return x * self.scale + text
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.ContrastiveHead` {#ultralytics.nn.modules.block.ContrastiveHead}

```python
ContrastiveHead(self)
```

**Bases:** `nn.Module`

Implements contrastive learning head for region-text similarity in vision-language models.

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.ContrastiveHead.forward) | Forward function of contrastive learning. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L753-L776"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())
```
</details>

<br>

### Method `ultralytics.nn.modules.block.ContrastiveHead.forward` {#ultralytics.nn.modules.block.ContrastiveHead.forward}

```python
def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor
```

Forward function of contrastive learning.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Image features. | *required* |
| `w` | `torch.Tensor` | Text features. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Similarity scores. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L763-L776"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Forward function of contrastive learning.

    Args:
        x (torch.Tensor): Image features.
        w (torch.Tensor): Text features.

    Returns:
        (torch.Tensor): Similarity scores.
    """
    x = F.normalize(x, dim=1, p=2)
    w = F.normalize(w, dim=-1, p=2)
    x = torch.einsum("bchw,bkc->bkhw", x, w)
    return x * self.logit_scale.exp() + self.bias
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.BNContrastiveHead` {#ultralytics.nn.modules.block.BNContrastiveHead}

```python
BNContrastiveHead(self, embed_dims: int)
```

**Bases:** `nn.Module`

Batch Norm Contrastive Head using batch norm instead of l2-normalization.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `embed_dims` | `int` | Embed dimensions of text and image features. | *required* |
| `embed_dims` | `int` | Embedding dimensions for features. | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.BNContrastiveHead.forward) | Forward function of contrastive learning with batch normalization. |
| [`forward_fuse`](#ultralytics.nn.modules.block.BNContrastiveHead.forward_fuse) | Passes image features through unchanged after fusing. |
| [`fuse`](#ultralytics.nn.modules.block.BNContrastiveHead.fuse) | Fuse the batch normalization layer in the BNContrastiveHead module. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L779-L825"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.BNContrastiveHead.forward` {#ultralytics.nn.modules.block.BNContrastiveHead.forward}

```python
def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor
```

Forward function of contrastive learning with batch normalization.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Image features. | *required* |
| `w` | `torch.Tensor` | Text features. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Similarity scores. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L811-L825"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Forward function of contrastive learning with batch normalization.

    Args:
        x (torch.Tensor): Image features.
        w (torch.Tensor): Text features.

    Returns:
        (torch.Tensor): Similarity scores.
    """
    x = self.norm(x)
    w = F.normalize(w, dim=-1, p=2)

    x = torch.einsum("bchw,bkc->bkhw", x, w)
    return x * self.logit_scale.exp() + self.bias
```
</details>

<br>

### Method `ultralytics.nn.modules.block.BNContrastiveHead.forward_fuse` {#ultralytics.nn.modules.block.BNContrastiveHead.forward\_fuse}

```python
def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor
```

Passes image features through unchanged after fusing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |
| `w` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L807-L809"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Passes image features through unchanged after fusing."""
    return x
```
</details>

<br>

### Method `ultralytics.nn.modules.block.BNContrastiveHead.fuse` {#ultralytics.nn.modules.block.BNContrastiveHead.fuse}

```python
def fuse(self)
```

Fuse the batch normalization layer in the BNContrastiveHead module.

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L799-L804"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse(self):
    """Fuse the batch normalization layer in the BNContrastiveHead module."""
    del self.norm
    del self.bias
    del self.logit_scale
    self.forward = self.forward_fuse
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.RepBottleneck` {#ultralytics.nn.modules.block.RepBottleneck}

```python
RepBottleneck(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5)
```

**Bases:** `Bottleneck`

Rep bottleneck.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `shortcut` | `bool` | Whether to use shortcut connection. | `True` |
| `g` | `int` | Groups for convolutions. | `1` |
| `k` | `tuple` | Kernel sizes for convolutions. | `(3, 3)` |
| `e` | `float` | Expansion ratio. | `0.5` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L828-L846"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.RepCSP` {#ultralytics.nn.modules.block.RepCSP}

```python
RepCSP(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5)
```

**Bases:** `C3`

Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of RepBottleneck blocks. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `True` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L849-L865"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.RepNCSPELAN4` {#ultralytics.nn.modules.block.RepNCSPELAN4}

```python
RepNCSPELAN4(self, c1: int, c2: int, c3: int, c4: int, n: int = 1)
```

**Bases:** `nn.Module`

CSP-ELAN.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `c3` | `int` | Intermediate channels. | *required* |
| `c4` | `int` | Intermediate channels for RepCSP. | *required* |
| `n` | `int` | Number of RepCSP blocks. | `1` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.RepNCSPELAN4.forward) | Forward pass through RepNCSPELAN4 layer. |
| [`forward_split`](#ultralytics.nn.modules.block.RepNCSPELAN4.forward_split) | Forward pass using split() instead of chunk(). |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L868-L898"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.RepNCSPELAN4.forward` {#ultralytics.nn.modules.block.RepNCSPELAN4.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through RepNCSPELAN4 layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L888-L892"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through RepNCSPELAN4 layer."""
    y = list(self.cv1(x).chunk(2, 1))
    y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
    return self.cv4(torch.cat(y, 1))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.RepNCSPELAN4.forward_split` {#ultralytics.nn.modules.block.RepNCSPELAN4.forward\_split}

```python
def forward_split(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass using split() instead of chunk().

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L894-L898"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward_split(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass using split() instead of chunk()."""
    y = list(self.cv1(x).split((self.c, self.c), 1))
    y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
    return self.cv4(torch.cat(y, 1))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.ELAN1` {#ultralytics.nn.modules.block.ELAN1}

```python
ELAN1(self, c1: int, c2: int, c3: int, c4: int)
```

**Bases:** `RepNCSPELAN4`

ELAN1 module with 4 convolutions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `c3` | `int` | Intermediate channels. | *required* |
| `c4` | `int` | Intermediate channels for convolutions. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L901-L918"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.AConv` {#ultralytics.nn.modules.block.AConv}

```python
AConv(self, c1: int, c2: int)
```

**Bases:** `nn.Module`

AConv.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.AConv.forward) | Forward pass through AConv layer. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L921-L937"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.AConv.forward` {#ultralytics.nn.modules.block.AConv.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through AConv layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L934-L937"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through AConv layer."""
    x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
    return self.cv1(x)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.ADown` {#ultralytics.nn.modules.block.ADown}

```python
ADown(self, c1: int, c2: int)
```

**Bases:** `nn.Module`

ADown.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.ADown.forward) | Forward pass through ADown layer. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L940-L962"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.ADown.forward` {#ultralytics.nn.modules.block.ADown.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through ADown layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L955-L962"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through ADown layer."""
    x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
    x1, x2 = x.chunk(2, 1)
    x1 = self.cv1(x1)
    x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
    x2 = self.cv2(x2)
    return torch.cat((x1, x2), 1)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.SPPELAN` {#ultralytics.nn.modules.block.SPPELAN}

```python
SPPELAN(self, c1: int, c2: int, c3: int, k: int = 5)
```

**Bases:** `nn.Module`

SPP-ELAN.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `c3` | `int` | Intermediate channels. | *required* |
| `k` | `int` | Kernel size for max pooling. | `5` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.SPPELAN.forward) | Forward pass through SPPELAN layer. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L965-L989"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.SPPELAN.forward` {#ultralytics.nn.modules.block.SPPELAN.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through SPPELAN layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L985-L989"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through SPPELAN layer."""
    y = [self.cv1(x)]
    y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
    return self.cv5(torch.cat(y, 1))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.CBLinear` {#ultralytics.nn.modules.block.CBLinear}

```python
CBLinear(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1)
```

**Bases:** `nn.Module`

CBLinear.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2s` | `list[int]` | List of output channel sizes. | *required* |
| `k` | `int` | Kernel size. | `1` |
| `s` | `int` | Stride. | `1` |
| `p` | `int | None` | Padding. | `None` |
| `g` | `int` | Groups. | `1` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.CBLinear.forward) | Forward pass through CBLinear layer. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L992-L1012"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.CBLinear.forward` {#ultralytics.nn.modules.block.CBLinear.forward}

```python
def forward(self, x: torch.Tensor) -> list[torch.Tensor]
```

Forward pass through CBLinear layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1010-L1012"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    """Forward pass through CBLinear layer."""
    return self.conv(x).split(self.c2s, dim=1)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.CBFuse` {#ultralytics.nn.modules.block.CBFuse}

```python
CBFuse(self, idx: list[int])
```

**Bases:** `nn.Module`

CBFuse.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `idx` | `list[int]` | Indices for feature selection. | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.CBFuse.forward) | Forward pass through CBFuse layer. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1015-L1038"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx
```
</details>

<br>

### Method `ultralytics.nn.modules.block.CBFuse.forward` {#ultralytics.nn.modules.block.CBFuse.forward}

```python
def forward(self, xs: list[torch.Tensor]) -> torch.Tensor
```

Forward pass through CBFuse layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `xs` | `list[torch.Tensor]` | List of input tensors. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Fused output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1027-L1038"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
    """Forward pass through CBFuse layer.

    Args:
        xs (list[torch.Tensor]): List of input tensors.

    Returns:
        (torch.Tensor): Fused output tensor.
    """
    target_size = xs[-1].shape[2:]
    res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
    return torch.sum(torch.stack(res + xs[-1:]), dim=0)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C3f` {#ultralytics.nn.modules.block.C3f}

```python
C3f(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5)
```

**Bases:** `nn.Module`

Faster Implementation of CSP Bottleneck with 3 convolutions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of Bottleneck blocks. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `False` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.C3f.forward) | Forward pass through C3f layer. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1041-L1066"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with three convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.C3f.forward` {#ultralytics.nn.modules.block.C3f.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through C3f layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1062-L1066"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through C3f layer."""
    y = [self.cv2(x), self.cv1(x)]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv3(torch.cat(y, 1))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C3k2` {#ultralytics.nn.modules.block.C3k2}

```python
def __init__(
    self,
    c1: int,
    c2: int,
    n: int = 1,
    c3k: bool = False,
    e: float = 0.5,
    attn: bool = False,
    g: int = 1,
    shortcut: bool = True,
)
```

**Bases:** `C2f`

Faster Implementation of CSP Bottleneck with 2 convolutions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of blocks. | `1` |
| `c3k` | `bool` | Whether to use C3k blocks. | `False` |
| `e` | `float` | Expansion ratio. | `0.5` |
| `attn` | `bool` | Whether to use attention blocks. | `False` |
| `g` | `int` | Groups for convolutions. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `True` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1069-L1106"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C3k` {#ultralytics.nn.modules.block.C3k}

```python
C3k(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3)
```

**Bases:** `C3`

C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of Bottleneck blocks. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `True` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |
| `k` | `int` | Kernel size. | `3` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1109-L1127"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.RepVGGDW` {#ultralytics.nn.modules.block.RepVGGDW}

```python
RepVGGDW(self, ed: int) -> None
```

**Bases:** `torch.nn.Module`

RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `ed` | `int` | Input and output channels. | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.RepVGGDW.forward) | Perform a forward pass of the RepVGGDW block. |
| [`forward_fuse`](#ultralytics.nn.modules.block.RepVGGDW.forward_fuse) | Perform a forward pass of the fused RepVGGDW block. |
| [`fuse`](#ultralytics.nn.modules.block.RepVGGDW.fuse) | Fuse the convolutional layers in the RepVGGDW block. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1130-L1192"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()
```
</details>

<br>

### Method `ultralytics.nn.modules.block.RepVGGDW.forward` {#ultralytics.nn.modules.block.RepVGGDW.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Perform a forward pass of the RepVGGDW block.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after applying the depth-wise convolution. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1145-L1154"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Perform a forward pass of the RepVGGDW block.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor after applying the depth-wise convolution.
    """
    return self.act(self.conv(x) + self.conv1(x))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.RepVGGDW.forward_fuse` {#ultralytics.nn.modules.block.RepVGGDW.forward\_fuse}

```python
def forward_fuse(self, x: torch.Tensor) -> torch.Tensor
```

Perform a forward pass of the fused RepVGGDW block.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after applying the depth-wise convolution. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1156-L1165"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
    """Perform a forward pass of the fused RepVGGDW block.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor after applying the depth-wise convolution.
    """
    return self.act(self.conv(x))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.RepVGGDW.fuse` {#ultralytics.nn.modules.block.RepVGGDW.fuse}

```python
def fuse(self)
```

Fuse the convolutional layers in the RepVGGDW block.

This method fuses the convolutional layers and updates the weights and biases accordingly.

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1168-L1192"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@torch.no_grad()
def fuse(self):
    """Fuse the convolutional layers in the RepVGGDW block.

    This method fuses the convolutional layers and updates the weights and biases accordingly.
    """
    if not hasattr(self, "conv1"):
        return  # already fused
    conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
    conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

    conv_w = conv.weight
    conv_b = conv.bias
    conv1_w = conv1.weight
    conv1_b = conv1.bias

    conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

    final_conv_w = conv_w + conv1_w
    final_conv_b = conv_b + conv1_b

    conv.weight.data.copy_(final_conv_w)
    conv.bias.data.copy_(final_conv_b)

    self.conv = conv
    del self.conv1
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.CIB` {#ultralytics.nn.modules.block.CIB}

```python
CIB(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False)
```

**Bases:** `nn.Module`

Compact Inverted Block (CIB) module.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `shortcut` | `bool, optional` | Whether to add a shortcut connection. Defaults to True. | `True` |
| `e` | `float, optional` | Scaling factor for the hidden channels. Defaults to 0.5. | `0.5` |
| `lk` | `bool, optional` | Whether to use RepVGGDW for the third convolutional layer. Defaults to False. | `False` |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `shortcut` | `bool` | Whether to use shortcut connection. | `True` |
| `e` | `float` | Expansion ratio. | `0.5` |
| `lk` | `bool` | Whether to use RepVGGDW. | `False` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.CIB.forward) | Forward pass of the CIB module. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1195-L1237"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2
```
</details>

<br>

### Method `ultralytics.nn.modules.block.CIB.forward` {#ultralytics.nn.modules.block.CIB.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass of the CIB module.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1228-L1237"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the CIB module.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor.
    """
    return x + self.cv1(x) if self.add else self.cv1(x)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C2fCIB` {#ultralytics.nn.modules.block.C2fCIB}

```python
C2fCIB(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5)
```

**Bases:** `C2f`

C2fCIB class represents a convolutional block with C2f and CIB modules.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `n` | `int, optional` | Number of CIB modules to stack. Defaults to 1. | `1` |
| `shortcut` | `bool, optional` | Whether to use shortcut connection. Defaults to False. | `False` |
| `lk` | `bool, optional` | Whether to use large kernel. Defaults to False. | `False` |
| `g` | `int, optional` | Number of groups for grouped convolution. Defaults to 1. | `1` |
| `e` | `float, optional` | Expansion ratio for CIB modules. Defaults to 0.5. | `0.5` |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of CIB modules. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connection. | `False` |
| `lk` | `bool` | Whether to use large kernel. | `False` |
| `g` | `int` | Groups for convolutions. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1240-L1268"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.Attention` {#ultralytics.nn.modules.block.Attention}

```python
Attention(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5)
```

**Bases:** `nn.Module`

Attention module that performs self-attention on the input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dim` | `int` | The input tensor dimension. | *required* |
| `num_heads` | `int` | The number of attention heads. | `8` |
| `attn_ratio` | `float` | The ratio of the attention key dimension to the head dimension. | `0.5` |
| `dim` | `int` | Input dimension. | *required* |
| `num_heads` | `int` | Number of attention heads. | `8` |
| `attn_ratio` | `float` | Attention ratio for key dimension. | `0.5` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `num_heads` | `int` | The number of attention heads. |
| `head_dim` | `int` | The dimension of each attention head. |
| `key_dim` | `int` | The dimension of the attention key. |
| `scale` | `float` | The scaling factor for the attention scores. |
| `qkv` | `Conv` | Convolutional layer for computing the query, key, and value. |
| `proj` | `Conv` | Convolutional layer for projecting the attended values. |
| `pe` | `Conv` | Convolutional layer for positional encoding. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.Attention.forward) | Forward pass of the Attention module. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1271-L1328"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.Attention.forward` {#ultralytics.nn.modules.block.Attention.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass of the Attention module.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | The input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | The output tensor after self-attention. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1308-L1328"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the Attention module.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        (torch.Tensor): The output tensor after self-attention.
    """
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x)
    q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2
    )

    attn = (q.transpose(-2, -1) @ k) * self.scale
    attn = attn.softmax(dim=-1)
    x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
    x = self.proj(x)
    return x
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.PSABlock` {#ultralytics.nn.modules.block.PSABlock}

```python
PSABlock(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None
```

**Bases:** `nn.Module`

PSABlock class implementing a Position-Sensitive Attention block for neural networks.

This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers with optional shortcut connections.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c` | `int` | Input and output channels. | *required* |
| `attn_ratio` | `float` | Attention ratio for key dimension. | `0.5` |
| `num_heads` | `int` | Number of attention heads. | `4` |
| `shortcut` | `bool` | Whether to use shortcut connections. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `attn` | `Attention` | Multi-head attention module. |
| `ffn` | `nn.Sequential` | Feed-forward neural network module. |
| `add` | `bool` | Flag indicating whether to add shortcut connections. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.PSABlock.forward) | Execute a forward pass through PSABlock. |

**Examples**

```python
Create a PSABlock and perform a forward pass
>>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
>>> input_tensor = torch.randn(1, 128, 32, 32)
>>> output_tensor = psablock(input_tensor)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1331-L1378"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut
```
</details>

<br>

### Method `ultralytics.nn.modules.block.PSABlock.forward` {#ultralytics.nn.modules.block.PSABlock.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Execute a forward pass through PSABlock.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after attention and feed-forward processing. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1367-L1378"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Execute a forward pass through PSABlock.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor after attention and feed-forward processing.
    """
    x = x + self.attn(x) if self.add else self.attn(x)
    x = x + self.ffn(x) if self.add else self.ffn(x)
    return x
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.PSA` {#ultralytics.nn.modules.block.PSA}

```python
PSA(self, c1: int, c2: int, e: float = 0.5)
```

**Bases:** `nn.Module`

PSA class for implementing Position-Sensitive Attention in neural networks.

This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to input tensors, enhancing feature extraction and processing capabilities.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `e` | `float` | Expansion ratio. | `0.5` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `c` | `int` | Number of hidden channels after applying the initial convolution. |
| `cv1` | `Conv` | 1x1 convolution layer to reduce the number of input channels to 2*c. |
| `cv2` | `Conv` | 1x1 convolution layer to reduce the number of output channels to c1. |
| `attn` | `Attention` | Attention module for position-sensitive attention. |
| `ffn` | `nn.Sequential` | Feed-forward network for further processing. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.PSA.forward) | Execute forward pass in PSA module. |

**Examples**

```python
Create a PSA module and apply it to an input tensor
>>> psa = PSA(c1=128, c2=128, e=0.5)
>>> input_tensor = torch.randn(1, 128, 64, 64)
>>> output_tensor = psa.forward(input_tensor)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1381-L1433"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.PSA.forward` {#ultralytics.nn.modules.block.PSA.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Execute forward pass in PSA module.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after attention and feed-forward processing. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1421-L1433"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Execute forward pass in PSA module.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor after attention and feed-forward processing.
    """
    a, b = self.cv1(x).split((self.c, self.c), dim=1)
    b = b + self.attn(b)
    b = b + self.ffn(b)
    return self.cv2(torch.cat((a, b), 1))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C2PSA` {#ultralytics.nn.modules.block.C2PSA}

```python
C2PSA(self, c1: int, c2: int, n: int = 1, e: float = 0.5)
```

**Bases:** `nn.Module`

C2PSA module with attention mechanism for enhanced feature extraction and processing.

This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of PSABlock modules. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `c` | `int` | Number of hidden channels. |
| `cv1` | `Conv` | 1x1 convolution layer to reduce the number of input channels to 2*c. |
| `cv2` | `Conv` | 1x1 convolution layer to reduce the number of output channels to c1. |
| `m` | `nn.Sequential` | Sequential container of PSABlock modules for attention and feed-forward operations. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.C2PSA.forward) | Process the input tensor through a series of PSA blocks. |

**Examples**

```python
>>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
>>> input_tensor = torch.randn(1, 256, 64, 64)
>>> output_tensor = c2psa(input_tensor)
```

!!! note "Notes"

    This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1436-L1488"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.C2PSA.forward` {#ultralytics.nn.modules.block.C2PSA.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Process the input tensor through a series of PSA blocks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after processing. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1477-L1488"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Process the input tensor through a series of PSA blocks.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor after processing.
    """
    a, b = self.cv1(x).split((self.c, self.c), dim=1)
    b = self.m(b)
    return self.cv2(torch.cat((a, b), 1))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.C2fPSA` {#ultralytics.nn.modules.block.C2fPSA}

```python
C2fPSA(self, c1: int, c2: int, n: int = 1, e: float = 0.5)
```

**Bases:** `C2f`

C2fPSA module with enhanced feature extraction using PSA blocks.

This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `n` | `int` | Number of PSABlock modules. | `1` |
| `e` | `float` | Expansion ratio. | `0.5` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `c` | `int` | Number of hidden channels. |
| `cv1` | `Conv` | 1x1 convolution layer to reduce the number of input channels to 2*c. |
| `cv2` | `Conv` | 1x1 convolution layer to reduce the number of output channels to c2. |
| `m` | `nn.ModuleList` | List of PSABlock modules for feature extraction. |

**Examples**

```python
>>> import torch
>>> from ultralytics.nn.modules.block import C2fPSA
>>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
>>> x = torch.randn(1, 64, 128, 128)
>>> output = model(x)
>>> print(output.shape)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1491-L1527"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
        m (nn.ModuleList): List of PSABlock modules for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.SCDown` {#ultralytics.nn.modules.block.SCDown}

```python
SCDown(self, c1: int, c2: int, k: int, s: int)
```

**Bases:** `nn.Module`

SCDown module for downsampling with separable convolutions.

This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Input channels. | *required* |
| `c2` | `int` | Output channels. | *required* |
| `k` | `int` | Kernel size. | *required* |
| `s` | `int` | Stride. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `cv1` | `Conv` | Pointwise convolution layer that reduces the number of channels. |
| `cv2` | `Conv` | Depthwise convolution layer that performs spatial downsampling. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.SCDown.forward) | Apply convolution and downsampling to the input tensor. |

**Examples**

```python
>>> import torch
>>> from ultralytics.nn.modules.block import SCDown
>>> model = SCDown(c1=64, c2=128, k=3, s=2)
>>> x = torch.randn(1, 64, 128, 128)
>>> y = model(x)
>>> print(y.shape)
torch.Size([1, 128, 64, 64])
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1530-L1575"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.SCDown.forward` {#ultralytics.nn.modules.block.SCDown.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply convolution and downsampling to the input tensor.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Downsampled output tensor. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1566-L1575"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply convolution and downsampling to the input tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Downsampled output tensor.
    """
    return self.cv2(self.cv1(x))
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.TorchVision` {#ultralytics.nn.modules.block.TorchVision}

```python
TorchVision(self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False)
```

**Bases:** `nn.Module`

TorchVision module to allow loading any torchvision model.

This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `str` | Name of the torchvision model to load. | *required* |
| `weights` | `str, optional` | Pre-trained weights to load. Default is "DEFAULT". | `"DEFAULT"` |
| `unwrap` | `bool, optional` | Unwraps the model to a sequential containing all but the last `truncate` layers. | `True` |
| `truncate` | `int, optional` | Number of layers to truncate from the end if `unwrap` is True. Default is 2. | `2` |
| `split` | `bool, optional` | Returns output from intermediate child modules as list. Default is False. | `False` |
| `weights` | `str` | Pre-trained weights to load. | `"DEFAULT"` |
| `unwrap` | `bool` | Whether to unwrap the model. | `True` |
| `truncate` | `int` | Number of layers to truncate. | `2` |
| `split` | `bool` | Whether to split the output. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `m` | `nn.Module` | The loaded torchvision model, possibly truncated and unwrapped. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.TorchVision.forward) | Forward pass through the model. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1578-L1638"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()
```
</details>

<br>

### Method `ultralytics.nn.modules.block.TorchVision.forward` {#ultralytics.nn.modules.block.TorchVision.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through the model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor | list[torch.Tensor]` | Output tensor or list of tensors. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1624-L1638"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the model.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
    """
    if self.split:
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
    else:
        y = self.m(x)
    return y
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.AAttn` {#ultralytics.nn.modules.block.AAttn}

```python
AAttn(self, dim: int, num_heads: int, area: int = 1)
```

**Bases:** `nn.Module`

Area-attention module for YOLO models, providing efficient attention mechanisms.

This module implements an area-based attention mechanism that processes input features in a spatially-aware manner, making it particularly effective for object detection tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dim` | `int` | Number of hidden channels. | *required* |
| `num_heads` | `int` | Number of heads into which the attention mechanism is divided. | *required* |
| `area` | `int` | Number of areas the feature map is divided into. | `1` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `area` | `int` | Number of areas the feature map is divided into. |
| `num_heads` | `int` | Number of heads into which the attention mechanism is divided. |
| `head_dim` | `int` | Dimension of each attention head. |
| `qkv` | `Conv` | Convolution layer for computing query, key and value tensors. |
| `proj` | `Conv` | Projection convolution layer. |
| `pe` | `Conv` | Position encoding convolution layer. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.AAttn.forward) | Process the input tensor through the area-attention. |

**Examples**

```python
>>> attn = AAttn(dim=256, num_heads=8, area=4)
>>> x = torch.randn(1, 256, 32, 32)
>>> output = attn(x)
>>> print(output.shape)
torch.Size([1, 256, 32, 32])
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1641-L1721"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided into.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.AAttn.forward` {#ultralytics.nn.modules.block.AAttn.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Process the input tensor through the area-attention.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after area-attention. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1685-L1721"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Process the input tensor through the area-attention.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor after area-attention.
    """
    B, C, H, W = x.shape
    N = H * W

    qkv = self.qkv(x).flatten(2).transpose(1, 2)
    if self.area > 1:
        qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
        B, N, _ = qkv.shape
    q, k, v = (
        qkv.view(B, N, self.num_heads, self.head_dim * 3)
        .permute(0, 2, 3, 1)
        .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
    )
    attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
    attn = attn.softmax(dim=-1)
    x = v @ attn.transpose(-2, -1)
    x = x.permute(0, 3, 1, 2)
    v = v.permute(0, 3, 1, 2)

    if self.area > 1:
        x = x.reshape(B // self.area, N * self.area, C)
        v = v.reshape(B // self.area, N * self.area, C)
        B, N, _ = x.shape

    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    x = x + self.pe(v)
    return self.proj(x)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.ABlock` {#ultralytics.nn.modules.block.ABlock}

```python
ABlock(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1)
```

**Bases:** `nn.Module`

Area-attention block module for efficient feature extraction in YOLO models.

This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps. It uses a novel area-based attention approach that is more efficient than traditional self-attention while maintaining effectiveness.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `dim` | `int` | Number of input channels. | *required* |
| `num_heads` | `int` | Number of heads into which the attention mechanism is divided. | *required* |
| `mlp_ratio` | `float` | Expansion ratio for MLP hidden dimension. | `1.2` |
| `area` | `int` | Number of areas the feature map is divided into. | `1` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `attn` | `AAttn` | Area-attention module for processing spatial features. |
| `mlp` | `nn.Sequential` | Multi-layer perceptron for feature transformation. |

**Methods**

| Name | Description |
| --- | --- |
| [`_init_weights`](#ultralytics.nn.modules.block.ABlock._init_weights) | Initialize weights using a truncated normal distribution. |
| [`forward`](#ultralytics.nn.modules.block.ABlock.forward) | Forward pass through ABlock. |

**Examples**

```python
>>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
>>> x = torch.randn(1, 256, 32, 32)
>>> output = block(x)
>>> print(output.shape)
torch.Size([1, 256, 32, 32])
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1724-L1786"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.ABlock._init_weights` {#ultralytics.nn.modules.block.ABlock.\_init\_weights}

```python
def _init_weights(m: nn.Module)
```

Initialize weights using a truncated normal distribution.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `m` | `nn.Module` | Module to initialize. | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1765-L1774"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def _init_weights(m: nn.Module):
    """Initialize weights using a truncated normal distribution.

    Args:
        m (nn.Module): Module to initialize.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.ABlock.forward` {#ultralytics.nn.modules.block.ABlock.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through ABlock.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after area-attention and feed-forward processing. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1776-L1786"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through ABlock.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor after area-attention and feed-forward processing.
    """
    x = x + self.attn(x)
    return x + self.mlp(x)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.A2C2f` {#ultralytics.nn.modules.block.A2C2f}

```python
def __init__(
    self,
    c1: int,
    c2: int,
    n: int = 1,
    a2: bool = True,
    area: int = 1,
    residual: bool = False,
    mlp_ratio: float = 2.0,
    e: float = 0.5,
    g: int = 1,
    shortcut: bool = True,
)
```

**Bases:** `nn.Module`

Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature processing. It supports both area-attention and standard convolution modes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output channels. | *required* |
| `n` | `int` | Number of ABlock or C3k modules to stack. | `1` |
| `a2` | `bool` | Whether to use area attention blocks. If False, uses C3k blocks instead. | `True` |
| `area` | `int` | Number of areas the feature map is divided into. | `1` |
| `residual` | `bool` | Whether to use residual connections with learnable gamma parameter. | `False` |
| `mlp_ratio` | `float` | Expansion ratio for MLP hidden dimension. | `2.0` |
| `e` | `float` | Channel expansion ratio for hidden channels. | `0.5` |
| `g` | `int` | Number of groups for grouped convolutions. | `1` |
| `shortcut` | `bool` | Whether to use shortcut connections in C3k blocks. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `cv1` | `Conv` | Initial 1x1 convolution layer that reduces input channels to hidden channels. |
| `cv2` | `Conv` | Final 1x1 convolution layer that processes concatenated features. |
| `gamma` | `nn.Parameter | None` | Learnable parameter for residual scaling when using area attention. |
| `m` | `nn.ModuleList` | List of either ABlock or C3k modules for feature processing. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.A2C2f.forward) | Forward pass through A2C2f layer. |

**Examples**

```python
>>> m = A2C2f(512, 512, n=1, a2=True, area=1)
>>> x = torch.randn(1, 512, 32, 32)
>>> output = m(x)
>>> print(output.shape)
torch.Size([1, 512, 32, 32])
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1789-L1868"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided into.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )
```
</details>

<br>

### Method `ultralytics.nn.modules.block.A2C2f.forward` {#ultralytics.nn.modules.block.A2C2f.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Forward pass through A2C2f layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after processing. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1854-L1868"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through A2C2f layer.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        (torch.Tensor): Output tensor after processing.
    """
    y = [self.cv1(x)]
    y.extend(m(y[-1]) for m in self.m)
    y = self.cv2(torch.cat(y, 1))
    if self.gamma is not None:
        return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
    return y
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.SwiGLUFFN` {#ultralytics.nn.modules.block.SwiGLUFFN}

```python
SwiGLUFFN(self, gc: int, ec: int, e: int = 4) -> None
```

**Bases:** `nn.Module`

SwiGLU Feed-Forward Network for transformer-based architectures.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `gc` | `int` | Guide channels. | *required* |
| `ec` | `int` | Embedding channels. | *required* |
| `e` | `int` | Expansion factor. | `4` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.SwiGLUFFN.forward) | Apply SwiGLU transformation to input features. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1871-L1891"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.SwiGLUFFN.forward` {#ultralytics.nn.modules.block.SwiGLUFFN.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply SwiGLU transformation to input features.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1886-L1891"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU transformation to input features."""
    x12 = self.w12(x)
    x1, x2 = x12.chunk(2, dim=-1)
    hidden = F.silu(x1) * x2
    return self.w3(hidden)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.Residual` {#ultralytics.nn.modules.block.Residual}

```python
Residual(self, m: nn.Module) -> None
```

**Bases:** `nn.Module`

Residual connection wrapper for neural network modules.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `m` | `nn.Module` | Module to wrap with residual connection. | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.Residual.forward) | Apply residual connection to input features. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1894-L1912"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.Residual.forward` {#ultralytics.nn.modules.block.Residual.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply residual connection to input features.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1910-L1912"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply residual connection to input features."""
    return x + self.m(x)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.SAVPE` {#ultralytics.nn.modules.block.SAVPE}

```python
SAVPE(self, ch: list[int], c3: int, embed: int)
```

**Bases:** `nn.Module`

Spatial-Aware Visual Prompt Embedding module for feature enhancement.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `ch` | `list[int]` | List of input channel dimensions. | *required* |
| `c3` | `int` | Intermediate channels. | *required* |
| `embed` | `int` | Embedding dimension. | *required* |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.SAVPE.forward) | Process input features and visual prompts to generate enhanced embeddings. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1915-L1971"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.SAVPE.forward` {#ultralytics.nn.modules.block.SAVPE.forward}

```python
def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor
```

Process input features and visual prompts to generate enhanced embeddings.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |
| `vp` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1945-L1971"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
    """Process input features and visual prompts to generate enhanced embeddings."""
    y = [self.cv2[i](xi) for i, xi in enumerate(x)]
    y = self.cv4(torch.cat(y, dim=1))

    x = [self.cv1[i](xi) for i, xi in enumerate(x)]
    x = self.cv3(torch.cat(x, dim=1))

    B, C, H, W = x.shape

    Q = vp.shape[1]

    x = x.view(B, C, -1)

    y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
    vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

    y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

    y = y.reshape(B, Q, self.c, -1)
    vp = vp.reshape(B, Q, 1, -1)

    score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
    score = F.softmax(score, dim=-1).to(y.dtype)
    aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

    return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.Proto26` {#ultralytics.nn.modules.block.Proto26}

```python
Proto26(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80)
```

**Bases:** `Proto`

Ultralytics YOLO26 models mask Proto module for segmentation models.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `ch` | `tuple` | Tuple of channel sizes from backbone feature maps. | `()` |
| `c_` | `int` | Intermediate channels. | `256` |
| `c2` | `int` | Output channels (number of protos). | `32` |
| `nc` | `int` | Number of classes for semantic segmentation. | `80` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.block.Proto26.forward) | Perform a forward pass by fusing multi-scale feature maps and generating proto masks. |
| [`fuse`](#ultralytics.nn.modules.block.Proto26.fuse) | Fuse the model for inference by removing the semantic segmentation head. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1974-L2006"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))
```
</details>

<br>

### Method `ultralytics.nn.modules.block.Proto26.forward` {#ultralytics.nn.modules.block.Proto26.forward}

```python
def forward(self, x: torch.Tensor, return_semseg: bool = True) -> torch.Tensor
```

Perform a forward pass by fusing multi-scale feature maps and generating proto masks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |
| `return_semseg` | `bool` |  | `True` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L1991-L2002"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor, return_semseg: bool = True) -> torch.Tensor:
    """Perform a forward pass by fusing multi-scale feature maps and generating proto masks."""
    feat = x[0]
    for i, f in enumerate(self.feat_refine):
        up_feat = f(x[i + 1])
        up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
        feat = feat + up_feat
    p = super().forward(self.feat_fuse(feat))
    if self.training and return_semseg:
        semseg = self.semseg(feat)
        return (p, semseg)
    return p
```
</details>

<br>

### Method `ultralytics.nn.modules.block.Proto26.fuse` {#ultralytics.nn.modules.block.Proto26.fuse}

```python
def fuse(self)
```

Fuse the model for inference by removing the semantic segmentation head.

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L2004-L2006"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse(self):
    """Fuse the model for inference by removing the semantic segmentation head."""
    self.semseg = None
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.block.RealNVP` {#ultralytics.nn.modules.block.RealNVP}

```python
RealNVP(self)
```

**Bases:** `nn.Module`

RealNVP: a flow-based generative model.

References:
    https://arxiv.org/abs/1605.08803
    https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py

**Methods**

| Name | Description |
| --- | --- |
| [`prior`](#ultralytics.nn.modules.block.RealNVP.prior) | The prior distribution. |
| [`backward_p`](#ultralytics.nn.modules.block.RealNVP.backward_p) | Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian |
| [`init_weights`](#ultralytics.nn.modules.block.RealNVP.init_weights) | Initialize model weights. |
| [`log_prob`](#ultralytics.nn.modules.block.RealNVP.log_prob) | Calculate the log probability of given sample in data space. |
| [`nets`](#ultralytics.nn.modules.block.RealNVP.nets) | Get the scale model in a single invertible mapping. |
| [`nett`](#ultralytics.nn.modules.block.RealNVP.nett) | Get the translation model in a single invertible mapping. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L2009-L2067"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()
```
</details>

<br>

### Property `ultralytics.nn.modules.block.RealNVP.prior` {#ultralytics.nn.modules.block.RealNVP.prior}

```python
def prior(self)
```

The prior distribution.

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L2028-L2030"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def prior(self):
    """The prior distribution."""
    return torch.distributions.MultivariateNormal(self.loc, self.cov)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.RealNVP.backward_p` {#ultralytics.nn.modules.block.RealNVP.backward\_p}

```python
def backward_p(self, x)
```

Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian

matrix.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L2049-L2060"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def backward_p(self, x):
    """Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian
    matrix.
    """
    log_det_jacob, z = x.new_zeros(x.shape[0]), x
    for i in reversed(range(len(self.t))):
        z_ = self.mask[i] * z
        s = self.s[i](z_) * (1 - self.mask[i])
        t = self.t[i](z_) * (1 - self.mask[i])
        z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
        log_det_jacob -= s.sum(dim=1)
    return z, log_det_jacob
```
</details>

<br>

### Method `ultralytics.nn.modules.block.RealNVP.init_weights` {#ultralytics.nn.modules.block.RealNVP.init\_weights}

```python
def init_weights(self)
```

Initialize model weights.

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L2043-L2047"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_weights(self):
    """Initialize model weights."""
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.01)
```
</details>

<br>

### Method `ultralytics.nn.modules.block.RealNVP.log_prob` {#ultralytics.nn.modules.block.RealNVP.log\_prob}

```python
def log_prob(self, x)
```

Calculate the log probability of given sample in data space.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L2062-L2067"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def log_prob(self, x):
    """Calculate the log probability of given sample in data space."""
    if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
        self.float()
    z, log_det = self.backward_p(x)
    return self.prior.log_prob(z) + log_det
```
</details>

<br>

### Method `ultralytics.nn.modules.block.RealNVP.nets` {#ultralytics.nn.modules.block.RealNVP.nets}

```python
def nets()
```

Get the scale model in a single invertible mapping.

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L2018-L2020"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def nets():
    """Get the scale model in a single invertible mapping."""
    return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())
```
</details>

<br>

### Method `ultralytics.nn.modules.block.RealNVP.nett` {#ultralytics.nn.modules.block.RealNVP.nett}

```python
def nett()
```

Get the translation model in a single invertible mapping.

<details>
<summary>Source code in <code>ultralytics/nn/modules/block.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py#L2023-L2025"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def nett():
    """Get the translation model in a single invertible mapping."""
    return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
```
</details>

<br><br>
