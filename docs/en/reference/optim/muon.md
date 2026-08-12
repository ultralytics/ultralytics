---
title: optim.muon API Reference
description: Explore Ultralytics Muon optimizer with Newton-Schulz orthogonalization for neural network training. Includes MuSGD hybrid optimizer and momentum-based updates.
keywords: Muon optimizer, MuSGD, Newton-Schulz iteration, orthogonalization, momentum optimizer, neural network training, PyTorch optimizer, Ultralytics optimization
---

# Reference for `ultralytics/optim/muon.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/optim/muon.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/optim/muon.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`MuSGD`](#ultralytics.optim.muon.MuSGD)
        - [`Muon`](#ultralytics.optim.muon.Muon)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`MuSGD.step`](#ultralytics.optim.muon.MuSGD.step)
        - [`Muon.step`](#ultralytics.optim.muon.Muon.step)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`zeropower_via_newtonschulz5`](#ultralytics.optim.muon.zeropower_via_newtonschulz5)
        - [`muon_update`](#ultralytics.optim.muon.muon_update)


## Class `ultralytics.optim.muon.MuSGD` {#ultralytics.optim.muon.MuSGD}

```python
MuSGD(
    params,
    lr: float = 1e-3,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    use_muon: bool = False,
    muon: float = 0.5,
    sgd: float = 0.5,
)
```

**Bases:** `optim.Optimizer`

Hybrid optimizer combining Muon and SGD updates for neural network training.

This optimizer implements a combination of Muon (a momentum-based optimizer with orthogonalization via Newton-Schulz iterations) and standard SGD with momentum. It allows different parameter groups to use either the hybrid Muon+SGD approach or pure SGD.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `params` | `Iterable` | Parameters to optimize or dicts defining parameter groups. | *required* |
| `muon` | `float, optional` | Weight factor for Muon updates in hybrid mode. Default: 0.5. | `0.5` |
| `sgd` | `float, optional` | Weight factor for SGD updates in hybrid mode. Default: 0.5. | `0.5` |
| `params` | `Iterable` | Iterable of parameters to optimize or dicts defining parameter groups. | *required* |
| `lr` | `float` | Learning rate. | `1e-3` |
| `momentum` | `float` | Momentum factor for SGD. | `0.0` |
| `weight_decay` | `float` | Weight decay (L2 penalty). | `0.0` |
| `nesterov` | `bool` | Whether to use Nesterov momentum. | `False` |
| `use_muon` | `bool` | Whether to enable Muon updates. | `False` |
| `muon` | `float` | Scaling factor for Muon component. | `0.5` |
| `sgd` | `float` | Scaling factor for SGD component. | `0.5` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `muon` | `float` | Scaling factor applied to Muon learning rate. |
| `sgd` | `float` | Scaling factor applied to SGD learning rate in hybrid mode. |

**Methods**

| Name | Description |
| --- | --- |
| [`step`](#ultralytics.optim.muon.MuSGD.step) | Perform a single optimization step. |

**Examples**

```python
>>> param_groups = [
...     {
...         "params": model.conv_params,
...         "lr": 0.02,
...         "use_muon": True,
...         "momentum": 0.95,
...         "nesterov": True,
...         "weight_decay": 0.01,
...     },
...     {
...         "params": model.other_params,
...         "lr": 0.01,
...         "use_muon": False,
...         "momentum": 0.9,
...         "nesterov": False,
...         "weight_decay": 0,
...     },
... ]
>>> optimizer = MuSGD(param_groups, muon=0.5, sgd=0.5)
>>> loss = model(data)
>>> loss.backward()
>>> optimizer.step()
```

!!! note "Notes"

    - Parameter groups with 'use_muon': True will receive both Muon and SGD updates.
    - Parameter groups with 'use_muon': False will receive only SGD updates.
    - The Muon update uses orthogonalization which works best for 2D+ parameter tensors.

<details>
<summary>Source code in <code>ultralytics/optim/muon.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/optim/muon.py#L116-L251">View on GitHub</a>
```python
class MuSGD(optim.Optimizer):
    """Hybrid optimizer combining Muon and SGD updates for neural network training.

    This optimizer implements a combination of Muon (a momentum-based optimizer with orthogonalization via Newton-Schulz
    iterations) and standard SGD with momentum. It allows different parameter groups to use either the hybrid Muon+SGD
    approach or pure SGD.

    Args:
        params (Iterable): Parameters to optimize or dicts defining parameter groups.
        muon (float, optional): Weight factor for Muon updates in hybrid mode. Default: 0.5.
        sgd (float, optional): Weight factor for SGD updates in hybrid mode. Default: 0.5.

    Attributes:
        muon (float): Scaling factor applied to Muon learning rate.
        sgd (float): Scaling factor applied to SGD learning rate in hybrid mode.

    Examples:
        >>> param_groups = [
        ...     {
        ...         "params": model.conv_params,
        ...         "lr": 0.02,
        ...         "use_muon": True,
        ...         "momentum": 0.95,
        ...         "nesterov": True,
        ...         "weight_decay": 0.01,
        ...     },
        ...     {
        ...         "params": model.other_params,
        ...         "lr": 0.01,
        ...         "use_muon": False,
        ...         "momentum": 0.9,
        ...         "nesterov": False,
        ...         "weight_decay": 0,
        ...     },
        ... ]
        >>> optimizer = MuSGD(param_groups, muon=0.5, sgd=0.5)
        >>> loss = model(data)
        >>> loss.backward()
        >>> optimizer.step()

    Notes:
        - Parameter groups with 'use_muon': True will receive both Muon and SGD updates.
        - Parameter groups with 'use_muon': False will receive only SGD updates.
        - The Muon update uses orthogonalization which works best for 2D+ parameter tensors.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
    ):
        """Initialize MuSGD optimizer with hybrid Muon and SGD capabilities.

        Args:
            params (Iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            momentum (float): Momentum factor for SGD.
            weight_decay (float): Weight decay (L2 penalty).
            nesterov (bool): Whether to use Nesterov momentum.
            use_muon (bool): Whether to enable Muon updates.
            muon (float): Scaling factor for Muon component.
            sgd (float): Scaling factor for SGD component.
        """
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "use_muon": use_muon,
        }
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd
```
</details>

<br>

### Method `ultralytics.optim.muon.MuSGD.step` {#ultralytics.optim.muon.MuSGD.step}

```python
def step(self, closure=None)
```

Perform a single optimization step.

Applies either hybrid Muon+SGD updates or pure SGD updates depending on the 'use_muon' flag in each parameter group. For Muon-enabled groups, parameters receive both an orthogonalized Muon update and a standard SGD momentum update.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `closure` | `Callable, optional` | A closure that reevaluates the model and returns the loss. Default: None. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor \| None` | The loss value if closure is provided, otherwise None. |

!!! note "Notes"

    - Parameters with None gradients are skipped.
    - Muon updates use Newton-Schulz orthogonalization and work best on 2D+ tensors.
    - Weight decay is applied only to the SGD component in hybrid mode.

<details>
<summary>Source code in <code>ultralytics/optim/muon.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/optim/muon.py#L197-L251">View on GitHub</a>
```python
@torch.no_grad()
def step(self, closure=None):
    """Perform a single optimization step.

    Applies either hybrid Muon+SGD updates or pure SGD updates depending on the
    'use_muon' flag in each parameter group. For Muon-enabled groups, parameters
    receive both an orthogonalized Muon update and a standard SGD momentum update.

    Args:
        closure (Callable, optional): A closure that reevaluates the model
            and returns the loss. Default: None.

    Returns:
        (torch.Tensor | None): The loss value if closure is provided, otherwise None.

    Notes:
        - Parameters with None gradients are skipped.
        - Muon updates use Newton-Schulz orthogonalization and work best on 2D+ tensors.
        - Weight decay is applied only to the SGD component in hybrid mode.
    """
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            continue
        lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]
        for p in params:
            if len(self.state[p]) == 0:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                if group["use_muon"]:
                    self.state[p]["momentum_buffer_SGD"] = torch.zeros_like(p)
        if group["use_muon"]:
            updates = muon_update(
                [p.grad for p in params],
                [self.state[p]["momentum_buffer"] for p in params],
                beta=momentum,
                nesterov=nesterov,
            )
            torch._foreach_add_(params, updates, alpha=-(lr * self.muon))
            buffers = [self.state[p]["momentum_buffer_SGD"] for p in params]
            lr *= self.sgd
        else:
            buffers = [self.state[p]["momentum_buffer"] for p in params]
        # SGD update
        grads = [p.grad for p in params]
        if group["weight_decay"] != 0:
            grads = torch._foreach_add(grads, params, alpha=group["weight_decay"])
        torch._foreach_mul_(buffers, momentum)
        torch._foreach_add_(buffers, grads)
        updates = torch._foreach_add(grads, buffers, alpha=momentum) if nesterov else buffers
        torch._foreach_add_(params, updates, alpha=-lr)
    return loss
```
</details>


<br><br><hr><br>

## Class `ultralytics.optim.muon.Muon` {#ultralytics.optim.muon.Muon}

```python
Muon(params, lr: float = 0.02, weight_decay: float = 0, momentum: float = 0.95)
```

**Bases:** `optim.Optimizer`

Muon optimizer for usage in non-distributed settings.

This optimizer implements the Muon algorithm, which combines momentum-based updates with orthogonalization via Newton-Schulz iterations. It applies weight decay and learning rate scaling to parameter updates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `params` | `iterable` | Iterable of parameters to optimize or dicts defining parameter groups. | *required* |
| `lr` | `float, optional` | Learning rate. Default: 0.02. | `0.02` |
| `weight_decay` | `float, optional` | Weight decay (L2 penalty) coefficient. Default: 0. | `0` |
| `momentum` | `float, optional` | Momentum coefficient for exponential moving average. Default: 0.95. | `0.95` |
| `params` | `Iterable` | Iterable of parameters to optimize or dicts defining parameter groups. | *required* |
| `lr` | `float` | Learning rate. | `0.02` |
| `weight_decay` | `float` | Weight decay factor applied multiplicatively. | `0` |
| `momentum` | `float` | Momentum factor for gradient accumulation. | `0.95` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `param_groups` | `list` | List of parameter groups with their optimization settings. |
| `state` | `dict` | Dictionary containing optimizer state for each parameter. |

**Methods**

| Name | Description |
| --- | --- |
| [`step`](#ultralytics.optim.muon.Muon.step) | Perform a single optimization step. |

**Examples**

```python
>>> model = YourModel()
>>> optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.01, momentum=0.95)
>>> loss = model(data)
>>> loss.backward()
>>> optimizer.step()
```

!!! note "Notes"

    - Designed for non-distributed training environments.
    - Uses Muon updates with orthogonalization for all parameters.
    - Weight decay is applied multiplicatively before parameter update.
    - Parameters with None gradients are assigned zero gradients for synchronization.

<details>
<summary>Source code in <code>ultralytics/optim/muon.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/optim/muon.py#L254-L341">View on GitHub</a>
```python
class Muon(optim.Optimizer):
    """Muon optimizer for usage in non-distributed settings.

    This optimizer implements the Muon algorithm, which combines momentum-based updates with orthogonalization via
    Newton-Schulz iterations. It applies weight decay and learning rate scaling to parameter updates.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Default: 0.02.
        weight_decay (float, optional): Weight decay (L2 penalty) coefficient. Default: 0.
        momentum (float, optional): Momentum coefficient for exponential moving average. Default: 0.95.

    Attributes:
        param_groups (list): List of parameter groups with their optimization settings.
        state (dict): Dictionary containing optimizer state for each parameter.

    Examples:
        >>> model = YourModel()
        >>> optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.01, momentum=0.95)
        >>> loss = model(data)
        >>> loss.backward()
        >>> optimizer.step()

    Notes:
        - Designed for non-distributed training environments.
        - Uses Muon updates with orthogonalization for all parameters.
        - Weight decay is applied multiplicatively before parameter update.
        - Parameters with None gradients are assigned zero gradients for synchronization.
    """

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0, momentum: float = 0.95):
        """Initialize Muon optimizer with orthogonalization-based updates.

        Args:
            params (Iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            weight_decay (float): Weight decay factor applied multiplicatively.
            momentum (float): Momentum factor for gradient accumulation.
        """
        defaults = {"lr": lr, "weight_decay": weight_decay, "momentum": momentum}
        super().__init__(params, defaults)
```
</details>

<br>

### Method `ultralytics.optim.muon.Muon.step` {#ultralytics.optim.muon.Muon.step}

```python
def step(self, closure=None)
```

Perform a single optimization step.

Applies Muon updates to all parameters, incorporating momentum and orthogonalization. Weight decay is applied multiplicatively before the parameter update.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `closure` | `Callable[[], torch.Tensor] \| None, optional` | A closure that reevaluates the model and returns the loss. Default: None. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor \| None` | The loss value if closure is provided, otherwise None. |

**Examples**

```python
>>> optimizer = Muon(model.parameters())
>>> loss = model(inputs)
>>> loss.backward()
>>> optimizer.step()
```

!!! note "Notes"

    - Parameters with None gradients are assigned zero gradients for synchronization.
    - Weight decay is applied as: p *= (1 - lr * weight_decay).
    - Muon update uses Newton-Schulz orthogonalization and works best on 2D+ tensors.

<details>
<summary>Source code in <code>ultralytics/optim/muon.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/optim/muon.py#L297-L341">View on GitHub</a>
```python
@torch.no_grad()
def step(self, closure=None):
    """Perform a single optimization step.

    Applies Muon updates to all parameters, incorporating momentum and orthogonalization.
    Weight decay is applied multiplicatively before the parameter update.

    Args:
        closure (Callable[[], torch.Tensor] | None, optional): A closure that reevaluates the model
            and returns the loss. Default: None.

    Returns:
        (torch.Tensor | None): The loss value if closure is provided, otherwise None.

    Examples:
        >>> optimizer = Muon(model.parameters())
        >>> loss = model(inputs)
        >>> loss.backward()
        >>> optimizer.step()

    Notes:
        - Parameters with None gradients are assigned zero gradients for synchronization.
        - Weight decay is applied as: p *= (1 - lr * weight_decay).
        - Muon update uses Newton-Schulz orthogonalization and works best on 2D+ tensors.
    """
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        params = group["params"]
        if not params:
            continue
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)  # Force synchronization
            if len(self.state[p]) == 0:
                self.state[p]["momentum_buffer"] = torch.zeros_like(p)
        updates = muon_update(
            [p.grad for p in params], [self.state[p]["momentum_buffer"] for p in params], beta=group["momentum"]
        )
        torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
        torch._foreach_add_(params, updates, alpha=-group["lr"])

    return loss
```
</details>


<br><br><hr><br>

## Function `ultralytics.optim.muon.zeropower_via_newtonschulz5` {#ultralytics.optim.muon.zeropower\_via\_newtonschulz5}

```python
def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor
```

Compute the zeroth power / orthogonalization of matrix G using Newton-Schulz iteration.

This function implements a quintic Newton-Schulz iteration to compute an approximate orthogonalization of the input matrix G. The iteration coefficients are optimized to maximize convergence slope at zero, producing a result similar to UV^T from SVD, where USV^T = G, but with relaxed convergence guarantees that empirically work well for optimization purposes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `G` | `torch.Tensor` | Input 2D matrix or 3D batch of matrices to orthogonalize. | *required* |
| `eps` | `float, optional` | Small epsilon value added to norm for numerical stability. Default: 1e-7. | `1e-7` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Orthogonalized matrix/matrices with same shape as input G. |

**Examples**

```python
>>> G = torch.randn(128, 64)
>>> G_ortho = zeropower_via_newtonschulz5(G)
>>> print(G_ortho.shape)
torch.Size([128, 64])
```

!!! note "Notes"

    - Uses bfloat16 precision for computation.
    - Performs exactly 5 Newton-Schulz iteration steps with fixed coefficients.
    - Automatically transposes for efficiency when rows > columns.
    - Output approximates US'V^T where S' has diagonal entries ~ Uniform(0.5, 1.5).
    - Does not produce exact UV^T but works well empirically for neural network optimization.

<details>
<summary>Source code in <code>ultralytics/optim/muon.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/optim/muon.py#L9-L49">View on GitHub</a>
```python
def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute the zeroth power / orthogonalization of matrix G using Newton-Schulz iteration.

    This function implements a quintic Newton-Schulz iteration to compute an approximate orthogonalization of the input
    matrix G. The iteration coefficients are optimized to maximize convergence slope at zero, producing a result similar
    to UV^T from SVD, where USV^T = G, but with relaxed convergence guarantees that empirically work well for
    optimization purposes.

    Args:
        G (torch.Tensor): Input 2D matrix or 3D batch of matrices to orthogonalize.
        eps (float, optional): Small epsilon value added to norm for numerical stability. Default: 1e-7.

    Returns:
        (torch.Tensor): Orthogonalized matrix/matrices with same shape as input G.

    Examples:
        >>> G = torch.randn(128, 64)
        >>> G_ortho = zeropower_via_newtonschulz5(G)
        >>> print(G_ortho.shape)
        torch.Size([128, 64])

    Notes:
        - Uses bfloat16 precision for computation.
        - Performs exactly 5 Newton-Schulz iteration steps with fixed coefficients.
        - Automatically transposes for efficiency when rows > columns.
        - Output approximates US'V^T where S' has diagonal entries ~ Uniform(0.5, 1.5).
        - Does not produce exact UV^T but works well empirically for neural network optimization.
    """
    assert G.ndim in {2, 3}
    X = G.reshape(-1, G.size(-2), G.size(-1)).bfloat16()
    X /= X.norm(dim=(-2, -1), keepdim=True) + eps  # ensure top singular value <= 1
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        A = X @ X.transpose(-2, -1)
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)  # b * A + c * A @ A
        X = torch.baddbmm(X, B, X, beta=a)  # a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X.reshape(G.shape)
```
</details>


<br><br><hr><br>

## Function `ultralytics.optim.muon.muon_update` {#ultralytics.optim.muon.muon\_update}

```python
def muon_update(
    grad: torch.Tensor | list[torch.Tensor],
    momentum: torch.Tensor | list[torch.Tensor],
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor | list[torch.Tensor]
```

Compute Muon optimizer updates with momentum and orthogonalization.

This function applies momentum to the gradients, optionally uses Nesterov acceleration, and then orthogonalizes the updates using Newton-Schulz iterations. Matrices with the same row count are zero-padded and orthogonalized in a single batched call, and momentum math uses fused foreach ops, avoiding per-parameter kernel launch overhead. Higher-rank tensors are reshaped before orthogonalization, and each update is scaled based on parameter dimensions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `grad` | `torch.Tensor \| list[torch.Tensor]` | Gradient tensor(s) to update. Each must have at least two dimensions. | *required* |
| `momentum` | `torch.Tensor \| list[torch.Tensor]` | Momentum buffer tensor(s), modified in-place. | *required* |
| `beta` | `float, optional` | Momentum coefficient for exponential moving average. Default: 0.95. | `0.95` |
| `nesterov` | `bool, optional` | Whether to use Nesterov momentum acceleration. Default: True. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor \| list[torch.Tensor]` | Orthogonalized update tensor(s), each with the gradient's shape and dtype. |

**Examples**

```python
>>> grad = torch.randn(64, 128)
>>> momentum = torch.zeros_like(grad)
>>> update = muon_update(grad, momentum, beta=0.95, nesterov=True)
>>> print(update.shape)
torch.Size([64, 128])
```

!!! note "Notes"

    - Momentum buffers are updated in-place: momentum = beta * momentum + (1-beta) * grad.
    - With Nesterov: update = beta * momentum + (1-beta) * grad.
    - Without Nesterov: update = momentum.
    - Tensors with more than 2 dimensions are reshaped to 2D with the first dimension preserved.
    - Final updates are scaled by sqrt(max(1, dim[-2] / dim[-1])) to account for parameter dimensions.

<details>
<summary>Source code in <code>ultralytics/optim/muon.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/optim/muon.py#L52-L113">View on GitHub</a>
```python
def muon_update(
    grad: torch.Tensor | list[torch.Tensor],
    momentum: torch.Tensor | list[torch.Tensor],
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor | list[torch.Tensor]:
    """Compute Muon optimizer updates with momentum and orthogonalization.

    This function applies momentum to the gradients, optionally uses Nesterov acceleration, and then orthogonalizes the
    updates using Newton-Schulz iterations. Matrices with the same row count are zero-padded and orthogonalized in a
    single batched call, and momentum math uses fused foreach ops, avoiding per-parameter kernel launch overhead.
    Higher-rank tensors are reshaped before orthogonalization, and each update is scaled based on parameter dimensions.

    Args:
        grad (torch.Tensor | list[torch.Tensor]): Gradient tensor(s) to update. Each must have at least two dimensions.
        momentum (torch.Tensor | list[torch.Tensor]): Momentum buffer tensor(s), modified in-place.
        beta (float, optional): Momentum coefficient for exponential moving average. Default: 0.95.
        nesterov (bool, optional): Whether to use Nesterov momentum acceleration. Default: True.

    Returns:
        (torch.Tensor | list[torch.Tensor]): Orthogonalized update tensor(s), each with the gradient's shape and dtype.

    Examples:
        >>> grad = torch.randn(64, 128)
        >>> momentum = torch.zeros_like(grad)
        >>> update = muon_update(grad, momentum, beta=0.95, nesterov=True)
        >>> print(update.shape)
        torch.Size([64, 128])

    Notes:
        - Momentum buffers are updated in-place: momentum = beta * momentum + (1-beta) * grad.
        - With Nesterov: update = beta * momentum + (1-beta) * grad.
        - Without Nesterov: update = momentum.
        - Tensors with more than 2 dimensions are reshaped to 2D with the first dimension preserved.
        - Final updates are scaled by sqrt(max(1, dim[-2] / dim[-1])) to account for parameter dimensions.
    """
    single = isinstance(grad, torch.Tensor)
    grads, momentums = ([grad], [momentum]) if single else (grad, momentum)
    torch._foreach_mul_(momentums, beta)
    torch._foreach_add_(momentums, grads, alpha=1 - beta)
    if nesterov:
        updates = list(torch._foreach_mul(momentums, beta))
        torch._foreach_add_(updates, grads, alpha=1 - beta)
    else:
        updates = list(momentums)
    buckets = {}  # group matrices transposed to rows <= cols by (rows, scale) for batched orthogonalization
    for i, u in enumerate(updates):
        m = u.view(len(u), -1) if u.ndim > 2 else u
        transpose = m.size(0) > m.size(1)
        if transpose:
            m = m.transpose(0, 1)
        scale = max(1, grads[i].size(-2) / grads[i].size(-1)) ** 0.5
        buckets.setdefault((m.size(0), scale, m.device, m.dtype), []).append((i, m, transpose))
    for (_, scale, _, _), items in buckets.items():
        n = max(m.size(1) for _, m, _ in items)
        # zero-pad columns so different shapes share one batched call (zeros stay zero through Newton-Schulz)
        X = torch.stack([torch.nn.functional.pad(m, (0, n - m.size(1))) for _, m, _ in items])
        X = zeropower_via_newtonschulz5(X).to(grads[items[0][0]].dtype).mul_(scale)
        for j, (i, m, transpose) in enumerate(items):
            x = X[j, :, : m.size(1)]
            updates[i] = (x.T if transpose else x).reshape(grads[i].shape)
    return updates[0] if single else updates
```
</details>

<br><br>
