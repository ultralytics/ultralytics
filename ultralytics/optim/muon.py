# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import optim
from torch.optim.adamw import adamw as adamw_update


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


def muon_update(
    grad: torch.Tensor | list[torch.Tensor],
    momentum: torch.Tensor | list[torch.Tensor],
    beta: float = 0.95,
    nesterov: bool = True,
    conv_scale: bool = True,
    rms: float = 0.0,
    tau: float = 0.0,
) -> torch.Tensor | list[torch.Tensor]:
    """Compute Muon optimizer updates with momentum and orthogonalization.

    This function applies momentum to the gradients, optionally uses Nesterov acceleration, and then orthogonalizes the
    updates using Newton-Schulz iterations. Matrices with the same row count are zero-padded and orthogonalized in a
    single batched call, and momentum math uses fused foreach ops, avoiding per-parameter kernel launch overhead.
    Convolutional filters (4D tensors) are reshaped before orthogonalization, and each update is scaled based on
    parameter dimensions.

    Args:
        grad (torch.Tensor | list[torch.Tensor]): Gradient tensor(s) to update. Each can be 2D or 4D.
        momentum (torch.Tensor | list[torch.Tensor]): Momentum buffer tensor(s), modified in-place.
        beta (float, optional): Momentum coefficient for exponential moving average. Default: 0.95.
        nesterov (bool, optional): Whether to use Nesterov momentum acceleration. Default: True.
        conv_scale (bool, optional): Take the scale from the reshaped 2D matrix, so conv filters scale by sqrt(max(1,
            out / (in * kh * kw))). False takes it from the raw tensor's last two dims, which leaves every
            conv filter unscaled. Default: True.
        rms (float, optional): Target update RMS. Non-zero replaces the scale above with rms * sqrt(max(rows, cols)),
            which an orthogonalized matrix turns into a constant RMS of `rms` regardless of shape, matching Adam's
            update magnitude so both can share one learning rate. Ignores conv_scale. Default: 0.0.
        tau (float, optional): MiMuon threshold on the Frobenius norm of the momentum. Matrices below it skip
            orthogonalization and keep the raw momentum as their update. Zero orthogonalizes every matrix, which is
            plain Muon. Default: 0.0.

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
        - 4D tensors (conv filters) are reshaped to 2D as (out_channels, in_channels*height*width) for orthogonalization.
        - Final updates are scaled by sqrt(max(1, rows / cols)), taken from that 2D matrix when conv_scale.
        - With rms, they are scaled by rms * sqrt(max(rows, cols)) from that 2D matrix instead, fixing their RMS.
        - With tau, matrices whose momentum has a Frobenius norm below it keep that momentum unscaled as their update,
          which is the MiMuon rule (https://arxiv.org/abs/2605.19619) and skips their Newton-Schulz call.
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
    # MiMuon: orthogonalize only where the momentum is large enough, one device sync for the whole list
    small = torch.stack(torch._foreach_norm(updates)).lt(tau).tolist() if tau else None
    buckets = {}  # group matrices transposed to rows <= cols by (rows,) for batched orthogonalization
    for i, u in enumerate(updates):
        if small and small[i]:  # below tau, so the update stays the raw momentum
            continue
        m = u.view(len(u), -1) if u.ndim == 4 else u
        if rms:
            scale = rms * max(m.size(-2), m.size(-1)) ** 0.5  # constant update RMS, matching Adam
        else:
            s = m if conv_scale else grads[i]  # 2D matrix (out, in * kh * kw) for conv filters, or the raw kernel
            scale = max(1, s.size(-2) / s.size(-1)) ** 0.5
        transpose = m.size(0) > m.size(1)
        if transpose:
            m = m.T
        buckets.setdefault((m.size(0), m.device, m.dtype), []).append((i, m, transpose, scale))
    for items in buckets.values():
        n = max(m.size(1) for _, m, _, _ in items)
        # zero-pad columns so different shapes share one batched call (zeros stay zero through Newton-Schulz)
        X = torch.stack([torch.nn.functional.pad(m, (0, n - m.size(1))) for _, m, _, _ in items])
        X = zeropower_via_newtonschulz5(X).to(grads[items[0][0]].dtype)
        for j, (i, m, transpose, scale) in enumerate(items):
            x = X[j, :, : m.size(1)] * scale
            updates[i] = (x.T if transpose else x).reshape(grads[i].shape)
    return updates[0] if single else updates


class MuSGD(optim.Optimizer):
    """Hybrid optimizer combining Muon and SGD updates for neural network training.

    This optimizer implements a combination of Muon (a momentum-based optimizer with orthogonalization via Newton-Schulz
    iterations) and standard SGD with momentum. It allows different parameter groups to use either the hybrid Muon+SGD
    approach or pure SGD.

    Args:
        params (Iterable): Parameters to optimize or dicts defining parameter groups.
        muon (float, optional): Weight factor for Muon updates in hybrid mode. Default: 0.5.
        sgd (float, optional): Weight factor for the SGD component on the Muon groups. Default: 0.5.
        adamw (float, optional): Learning rate factor for AdamW, which replaces SGD as the auxiliary optimizer when
            non-zero. Default: 0.0.
        muon_aux (bool, optional): Whether the Muon groups also take the auxiliary update. Default: True.
        conv_scale (bool, optional): Scale conv Muon updates by their reshaped 2D matrix shape. Default: True.
        tau (float, optional): MiMuon threshold below which a matrix keeps its raw momentum instead of the
            orthogonalized direction. Default: 0.0.

    Attributes:
        muon (float): Scaling factor applied to Muon learning rate.
        sgd (float): Scaling factor applied to SGD learning rate in hybrid mode.
        adamw (float): Scaling factor applied to the AdamW learning rate, or 0 when the auxiliary optimizer is SGD.
        muon_aux (bool): Whether the Muon groups take the auxiliary update on top of the Muon one.
        conv_scale (bool): Whether conv filter updates are scaled by their reshaped 2D matrix shape.
        tau (float): Frobenius norm threshold gating orthogonalization, 0 to orthogonalize every matrix.

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
        - Parameter groups with 'use_muon': True receive a Muon update plus the auxiliary one, or Muon alone when
          muon_aux is False.
        - Parameter groups with 'use_muon': False receive only the auxiliary update.
        - The auxiliary update is SGD, or AdamW when adamw > 0.
        - The Muon update uses orthogonalization which works best for 2D+ parameter tensors.
        - With tau > 0 the Muon update follows MiMuon and drops to plain momentum on matrices whose momentum norm
          falls below tau.
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
        adamw: float = 0.0,
        muon_aux: bool = True,
        conv_scale: bool = True,
        tau: float = 0.0,
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
            sgd (float): Scaling factor for the SGD component on the Muon groups.
            adamw (float): Learning rate factor for AdamW. Non-zero makes AdamW the auxiliary optimizer in place of SGD,
                for the non-Muon groups and for the Muon groups' auxiliary update.
            muon_aux (bool): Whether the Muon groups also take the auxiliary update. False leaves them on Muon alone,
                which then carries a decoupled weight decay instead of the L2 the auxiliary update would apply.
            conv_scale (bool): Take the Muon update scale from the reshaped 2D matrix, scaling conv filters by
                sqrt(max(1, out / (in * kh * kw))) instead of leaving them unscaled.
            tau (float): MiMuon threshold on the momentum Frobenius norm. Matrices below it take a momentum SGD step in
                place of the orthogonalized one, on the same learning rate. Zero keeps every matrix on Muon.
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
        self.adamw = adamw
        self.muon_aux = muon_aux
        self.conv_scale = conv_scale
        self.tau = tau

    def _adamw_step(self, group: dict, params: list, lr: float, beta1: float, beta2: float = 0.999, eps: float = 1e-8):
        """Apply an AdamW update to a non-Muon parameter group, through torch's own AdamW kernels.

        Args:
            group (dict): The parameter group, read for its weight decay, which AdamW decouples.
            params (list[torch.Tensor]): Parameters of the group that have gradients.
            lr (float): Learning rate, already scaled by self.adamw.
            beta1 (float): First moment coefficient, taken from the group's momentum so warmup applies to it.
            beta2 (float): Second moment coefficient.
            eps (float): Term added to the denominator for numerical stability.
        """
        state = [self.state[p] for p in params]
        adamw_update(
            params,
            [p.grad for p in params],
            [s["exp_avg"] for s in state],
            [s["exp_avg_sq"] for s in state],
            [],  # max_exp_avg_sqs, unused without amsgrad
            [s["step"] for s in state],
            foreach=True,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=group["weight_decay"],
            eps=eps,
            maximize=False,
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Muon-enabled groups receive an orthogonalized Muon update, followed by the auxiliary
        update unless self.muon_aux is False. Every other group receives the auxiliary update
        alone. That auxiliary update is SGD with momentum, or AdamW when self.adamw is non-zero.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss. Default: None.

        Returns:
            (torch.Tensor | None): The loss value if closure is provided, otherwise None.

        Notes:
            - Parameters with None gradients are skipped.
            - Muon updates use Newton-Schulz orthogonalization and work best on 2D+ tensors.
            - Weight decay rides on the auxiliary update, as L2 for SGD and decoupled for AdamW. Muon groups that
              skip the auxiliary update take a decoupled decay of their own.
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
            adam = bool(self.adamw)  # the auxiliary optimizer is AdamW rather than SGD
            aux = self.muon_aux or not group["use_muon"]  # this group takes the auxiliary update
            for p in params:
                if len(self.state[p]) == 0:
                    if group["use_muon"]:
                        self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                    if not aux:
                        continue
                    if adam:  # AdamW's own state names, which the fp16 checkpoint conversion already knows to skip
                        self.state[p]["exp_avg"] = torch.zeros_like(p)
                        self.state[p]["exp_avg_sq"] = torch.zeros_like(p)
                        self.state[p]["step"] = torch.zeros((), dtype=torch.float32)
                    else:
                        self.state[p]["momentum_buffer_SGD" if group["use_muon"] else "momentum_buffer"] = (
                            torch.zeros_like(p)
                        )
            if group["use_muon"]:
                updates = muon_update(
                    [p.grad for p in params],
                    [self.state[p]["momentum_buffer"] for p in params],
                    beta=momentum,
                    nesterov=nesterov,
                    conv_scale=self.conv_scale,
                    rms=0.2 if adam else 0.0,  # share one lr with the AdamW groups
                    tau=self.tau,
                )
                if not aux and group["weight_decay"] != 0:  # no aux update to carry the L2, so decouple the decay
                    torch._foreach_mul_(params, 1 - lr * self.muon * group["weight_decay"])
                torch._foreach_add_(params, updates, alpha=-(lr * self.muon))
                if not aux:
                    continue
                if not adam:
                    lr *= self.sgd
            if adam:
                self._adamw_step(group, params, lr * self.adamw, momentum)
                continue
            buffers = [self.state[p]["momentum_buffer_SGD" if group["use_muon"] else "momentum_buffer"] for p in params]
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
