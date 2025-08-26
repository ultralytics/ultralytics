from torch import optim
from torch.optim.sgd import sgd
import torch


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
        # (3.4445, -4.7750, 2.0315),
        # (3.4445, -4.7750, 2.0315),
        # (3.4445, -4.7750, 2.0315),
        # (3.4445, -4.7750, 2.0315),
        # (3.4445, -4.7750, 2.0315),
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
        (7.2086, -15.5131, 9.0178),
        (3.9623, -2.5813, 0.4542),
        (3.9466, -2.5765, 0.4544),
        (3.8991, -2.5671, 0.4566),
        (3.7186, -2.5308, 0.4653),
        (3.1390, -2.3073, 0.4733),
        (2.1715, -1.5246, 0.3885),
        (1.8648, -1.2224, 0.3577),
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
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class MuonWithSGD(optim.Optimizer):
    def __init__(
        self,
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
        ns_steps=5,
        nesterov=True,
        muon_params=None,
        sgd_params=None,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, nesterov=nesterov)
        params = list(muon_params)
        sgd_params = list(sgd_params) if sgd_params is not None else []
        params.extend(sgd_params)
        super(MuonWithSGD, self).__init__(params, defaults)
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            # assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in sgd_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

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
            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # generate weight updates in distributed fashion
            for p in params:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

            # SGD
            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            grads = [p.grad for p in group["params"] if not self.state[p]["use_muon"] and p.grad is not None]
            momentum_buffer_list = [
                self.state[p].get("momentum_buffer")
                for p in group["params"]
                if not self.state[p]["use_muon"] and group["momentum"] != 0
            ]
            sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                # dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=False,
                # foreach=group["foreach"],
                # fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

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
