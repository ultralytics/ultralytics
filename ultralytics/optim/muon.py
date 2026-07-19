# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import optim


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
    for _ in range(5):  # num_steps fixed at 5
        A = X @ X.transpose(-2, -1)
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)  # b * A + c * A @ A
        X = torch.baddbmm(X, B, X, beta=a)  # a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.transpose(-2, -1)
    return X.reshape(G.shape)


def muon_update(
    grads: list[torch.Tensor], momentums: list[torch.Tensor], beta: float = 0.95, nesterov: bool = True
) -> list[torch.Tensor]:
    """Compute Muon optimizer updates with momentum and orthogonalization.

    This function applies momentum to the gradients, optionally uses Nesterov acceleration, and then orthogonalizes the
    updates using Newton-Schulz iterations. Matrices with the same row count are zero-padded and orthogonalized in a
    single batched call, and momentum math uses fused foreach ops, avoiding per-parameter kernel launch overhead.
    Convolutional filters (4D tensors) are reshaped before orthogonalization, and each update is scaled based on
    parameter dimensions.

    Args:
        grads (list[torch.Tensor]): Gradient tensors to update. Each can be 2D or 4D (for conv filters).
        momentums (list[torch.Tensor]): Momentum buffer tensors, modified in-place.
        beta (float, optional): Momentum coefficient for exponential moving average. Default: 0.95.
        nesterov (bool, optional): Whether to use Nesterov momentum acceleration. Default: True.

    Returns:
        (list[torch.Tensor]): Orthogonalized update tensors in gradient dtype, each with same shape as its gradient.

    Examples:
        >>> grads = [torch.randn(64, 128), torch.randn(64, 128)]
        >>> momentums = [torch.zeros_like(g) for g in grads]
        >>> updates = muon_update(grads, momentums, beta=0.95, nesterov=True)
        >>> print(updates[0].shape)
        torch.Size([64, 128])

    Notes:
        - Momentum buffers are updated in-place: momentum = beta * momentum + (1-beta) * grad.
        - With Nesterov: update = beta * momentum + (1-beta) * grad.
        - Without Nesterov: update = momentum.
        - 4D tensors (conv filters) are reshaped to 2D as (out_channels, in_channels*height*width) for orthogonalization.
        - Final updates are scaled by sqrt(max(1, dim[-2] / dim[-1])) to account for parameter dimensions.
    """
    torch._foreach_mul_(momentums, beta)
    torch._foreach_add_(momentums, grads, alpha=1 - beta)
    if nesterov:
        updates = list(torch._foreach_mul(momentums, beta))
        torch._foreach_add_(updates, grads, alpha=1 - beta)
    else:
        updates = list(momentums)
    buckets = {}  # group matrices transposed to rows <= cols by (rows, scale) for batched orthogonalization
    for i, u in enumerate(updates):
        m = u.view(len(u), -1) if u.ndim == 4 else u  # for the case of conv filters
        transpose = m.size(0) > m.size(1)
        if transpose:
            m = m.T
        scale = max(1, grads[i].size(-2) / grads[i].size(-1)) ** 0.5
        buckets.setdefault((m.size(0), scale, m.device, m.dtype), []).append((i, m, transpose))
    for (_, scale, _, _), items in buckets.items():
        n = max(m.size(1) for _, m, _ in items)
        # zero-pad columns so different shapes share one batched call (zeros stay zero through Newton-Schulz)
        X = torch.stack([torch.nn.functional.pad(m, (0, n - m.size(1))) for _, m, _ in items]).bfloat16()
        X = zeropower_via_newtonschulz5(X).to(grads[items[0][0]].dtype).mul_(scale)
        for j, (i, m, transpose) in enumerate(items):
            x = X[j, :, : m.size(1)]
            updates[i] = (x.T if transpose else x).reshape(grads[i].shape)
    return updates


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
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_muon=use_muon,
        )
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

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
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

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
