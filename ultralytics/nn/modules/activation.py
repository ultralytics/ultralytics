# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Activation modules."""

import torch
import torch.nn as nn


class AGLU(nn.Module):
    """
    Unified activation function module from https://github.com/kostas1515/AGLU.

    This class implements a parameterized activation function with learnable parameters lambda and kappa.

    Attributes:
        act (nn.Softplus): Softplus activation function with negative beta.
        lambd (nn.Parameter): Learnable lambda parameter initialized with uniform distribution.
        kappa (nn.Parameter): Learnable kappa parameter initialized with uniform distribution.
    """

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function with learnable parameters."""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda parameter
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the Unified activation function."""
        lam = torch.clamp(self.lambd, min=0.0001)  # Clamp lambda to avoid division by zero
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))


class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, padding=1, groups=c1, bias=False)
        self.bn = nn.Identity()

    def forward(self, x):
        return torch.max(x, self.conv(x))


# class FReLU(nn.Module):
#     def __init__(self, c1, k=3):  # ch_in, kernel
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
#         self.bn = nn.BatchNorm2d(c1)
#
#     def forward(self, x):
#         return torch.max(x, self.bn(self.conv(x)))
