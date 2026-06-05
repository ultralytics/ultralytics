# ultralytics/nn/modules/dg.py
# Ultralytics 🚀 AGPL-3.0 License
"""Domain-generalization building blocks for ReID training (train-only; no inference cost).

MixStyle (ICLR'21) mixes per-sample feature-style statistics across domains; GradientReversal +
DomainHead implement DANN domain-adversarial feature alignment. All are inert at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Identity forward; gradient is negated and scaled by ``lambd`` on backward."""
    return _GradReverse.apply(x, lambd)


class GradientReversal(nn.Module):
    """Module wrapper around :func:`grad_reverse` with a settable ``lambd``."""

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return grad_reverse(x, self.lambd)


class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR 2021): mix per-sample channel-wise feature statistics.

    Train-only. With probability ``p``, replaces each sample's (mean,std) over spatial dims with a
    Beta-interpolation toward another sample's statistics. ``domain``-aware: when domain ids are
    given, each sample is paired with one from a DIFFERENT domain (falls back to a random
    permutation if no cross-domain partner exists). No-op in eval or when p==0. Zero parameters.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def _perm_cross_domain(self, domain: torch.Tensor) -> torch.Tensor:
        b = domain.shape[0]
        perm = torch.arange(b, device=domain.device)
        for i in range(b):
            diff = (domain != domain[i]).nonzero(as_tuple=True)[0]
            if len(diff):
                perm[i] = diff[torch.randint(len(diff), (1,), device=domain.device)]
        return perm

    def forward(self, x: torch.Tensor, domain: torch.Tensor | None = None) -> torch.Tensor:
        if not self.training or self.p <= 0.0 or torch.rand(1).item() > self.p:
            return x
        b = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig
        if domain is not None and domain.numel() == b:
            perm = self._perm_cross_domain(domain.to(x.device))
        else:
            perm = torch.randperm(b, device=x.device)
        lam = self.beta.sample((b, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix


class DomainHead(nn.Module):
    """Small MLP domain classifier for DANN (train-only). 2-layer with dropout."""

    def __init__(self, in_dim: int, num_domains: int, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
