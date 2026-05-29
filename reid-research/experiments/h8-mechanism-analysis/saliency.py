"""Integrated Gradients on an arbitrary tensor target.

For h8 we integrate gradients of `cosine_sim(emb(query), emb(true_match))`
w.r.t. the query's P5 feature-map activations, with a zero-feature baseline.
50 Riemann steps per spec.

This module is the math kernel; the per-model plumbing (registering the P5 hook,
running the forward pass with a swapped feature tensor) lives in `extract.py`.
"""

from __future__ import annotations

from typing import Callable

import torch


def integrated_gradients(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    baseline: torch.Tensor,
    steps: int = 50,
) -> torch.Tensor:
    """Integrated Gradients attribution of scalar f(x) to each element of x.

    Args:
        f: function that maps tensor of shape x.shape -> scalar tensor.
        x: input tensor; gradient flow target.
        baseline: same shape as x; zeros are typical for activation-space IG.
        steps: number of Riemann steps (50 per h8 spec).

    Returns:
        Attribution tensor, same shape as x.

    Raises:
        ValueError: if f returns NaN on any interpolant (caller should skip-and-log).
    """
    if x.shape != baseline.shape:
        raise ValueError(f"shape mismatch: x={tuple(x.shape)} baseline={tuple(baseline.shape)}")

    grads = torch.zeros_like(x, dtype=torch.float32)
    alphas = torch.linspace(0.0, 1.0, steps, device=x.device, dtype=x.dtype)

    for alpha in alphas:
        interp = baseline + alpha * (x - baseline)
        interp = interp.detach().requires_grad_(True)
        y = f(interp)
        if torch.isnan(y).any():
            raise ValueError("NaN encountered in IG forward pass — skip this query")
        (g,) = torch.autograd.grad(y.sum(), interp, retain_graph=False, create_graph=False)
        grads = grads + g.detach().float()

    avg_grads = grads / steps
    return ((x - baseline).float() * avg_grads).detach()
