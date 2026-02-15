---
description: Explore activation functions in Ultralytics, including the Unified activation function and other custom implementations for neural networks.
keywords: ultralytics, activation functions, neural networks, Unified activation, AGLU, SiLU, ReLU, PyTorch, deep learning, custom activations
---

# Reference for `ultralytics/nn/modules/activation.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/activation.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/activation.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`AGLU`](#ultralytics.nn.modules.activation.AGLU)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`AGLU.forward`](#ultralytics.nn.modules.activation.AGLU.forward)


## Class `ultralytics.nn.modules.activation.AGLU` {#ultralytics.nn.modules.activation.AGLU}

```python
AGLU(self, device = None, dtype = None) -> None
```

**Bases:** `nn.Module`

Unified activation function module from AGLU.

This class implements a parameterized activation function with learnable parameters lambda and kappa, based on the AGLU (Adaptive Gated Linear Unit) approach.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `device` |  |  | `None` |
| `dtype` |  |  | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `act` | `nn.Softplus` | Softplus activation function with negative beta. |
| `lambd` | `nn.Parameter` | Learnable lambda parameter initialized with uniform distribution. |
| `kappa` | `nn.Parameter` | Learnable kappa parameter initialized with uniform distribution. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.activation.AGLU.forward) | Apply the Adaptive Gated Linear Unit (AGLU) activation function. |

**Examples**

```python
>>> import torch
    >>> m = AGLU()
    >>> input = torch.randn(2)
    >>> output = m(input)
    >>> print(output.shape)
    torch.Size([2])

References:
    https://github.com/kostas1515/AGLU
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/activation.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/activation.py#L8-L54"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class AGLU(nn.Module):
    """Unified activation function module from AGLU.

    This class implements a parameterized activation function with learnable parameters lambda and kappa, based on the
    AGLU (Adaptive Gated Linear Unit) approach.

    Attributes:
        act (nn.Softplus): Softplus activation function with negative beta.
        lambd (nn.Parameter): Learnable lambda parameter initialized with uniform distribution.
        kappa (nn.Parameter): Learnable kappa parameter initialized with uniform distribution.

    Methods:
        forward: Compute the forward pass of the Unified activation function.

    Examples:
        >>> import torch
        >>> m = AGLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
        >>> print(output.shape)
        torch.Size([2])

    References:
        https://github.com/kostas1515/AGLU
    """

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function with learnable parameters."""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda parameter
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa parameter
```
</details>

<br>

### Method `ultralytics.nn.modules.activation.AGLU.forward` {#ultralytics.nn.modules.activation.AGLU.forward}

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply the Adaptive Gated Linear Unit (AGLU) activation function.

This forward method implements the AGLU activation function with learnable parameters lambda and kappa. The function applies a transformation that adaptively combines linear and non-linear components.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` | Input tensor to apply the activation function to. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Output tensor after applying the AGLU activation function, with the same shape as the input. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/activation.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/activation.py#L41-L54"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the Adaptive Gated Linear Unit (AGLU) activation function.

    This forward method implements the AGLU activation function with learnable parameters lambda and kappa. The
    function applies a transformation that adaptively combines linear and non-linear components.

    Args:
        x (torch.Tensor): Input tensor to apply the activation function to.

    Returns:
        (torch.Tensor): Output tensor after applying the AGLU activation function, with the same shape as the input.
    """
    lam = torch.clamp(self.lambd, min=0.0001)  # Clamp lambda to avoid division by zero
    return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))
```
</details>

<br><br>
