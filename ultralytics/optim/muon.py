from torch import optim
import torch
import math


def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for a, b, c in [  # num_steps fixed at 5
        # original params
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        # option 1
        # (4.0848, -6.8946, 2.9270),
        # (3.9505, -6.3029, 2.6377),
        # (3.7418, -5.5913, 2.3037),
        # (2.8769, -3.1427, 1.2046),
        # (2.8366, -3.0525, 1.2012),
        # option 2
        # (4.01, -9.22, 5.80),
        # (3.49, -6.38, 3.23),
        # (3.34, -6.21, 3.20),
        # (3.64, -7.48, 4.43),
        # (3.46, -5.35, 2.85),
        # # option 3
        # (7.2086, -15.5131, 9.0178),
        # (3.9623, -2.5813, 0.4542),
        # (3.9466, -2.5765, 0.4544),
        # (3.8991, -2.5671, 0.4566),
        # (3.7186, -2.5308, 0.4653),
        # (3.1390, -2.3073, 0.4733),
        # (2.1715, -1.5246, 0.3885),
        # (1.8648, -1.2224, 0.3577),
    ]:
        # for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    # update = grad.lerp_(momentum, beta) if nesterov else momentum
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class MuonWithSGD(optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, dict())

    def adjust_lr(self, lr, param_shape):
        """ Adjust learning rate based on parameter shape."""
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Muon
            if group["use_muon"]:
                # generate weight updates in distributed fashion
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["momentum_buffer_SGD"] = torch.zeros_like(p)
                    # state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
                    # update = (
                    #     grad.lerp_(state["momentum_buffer"], group["momentum"])
                    #     if group["nesterov"]
                    #     else group["momentum"]
                    # )

                    # Muon update
                    # state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                    # update = (
                    #     grad.add(state["momentum_buffer"], alpha=group["momentum"])
                    #     if group["nesterov"]
                    #     else state["momentum_buffer"]
                    # )
                    # # sgd_update = update.clone()
                    # if update.ndim == 4:  # for the case of conv filters
                    #     update = update.view(len(update), -1)
                    # update = zeropower_via_newtonschulz5(update)
                    # update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
                    update = muon_update(
                        grad, state["momentum_buffer"], beta=group["momentum"], nesterov=group["nesterov"]
                    )
                    # TODO
                    lr = group["lr"] / 10
                    lr = self.adjust_lr(lr, p.shape)
                    # p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-lr)

                    # SGD update
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state["momentum_buffer_SGD"].mul_(group["momentum"]).add_(grad)
                    sgd_update = (
                        grad.add(state["momentum_buffer_SGD"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer_SGD"]
                    )
                    p.add_(sgd_update, alpha=-group["lr"])
            else:  # SGD
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    grad = p.grad
                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                    update = (
                        grad.add(state["momentum_buffer"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer"]
                    )
                    p.add_(update, alpha=-group["lr"])
        return loss


class Muon(optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss
