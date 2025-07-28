# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Activation modules."""

import torch
import torch.nn as nn


class AGLU(nn.Module):
    """
    Unified activation function module from AGLU.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Adaptive Gated Linear Unit (AGLU) activation function.

        This forward method implements the AGLU activation function with learnable parameters lambda and kappa.
        The function applies a transformation that adaptively combines linear and non-linear components.

        Args:
            x (torch.Tensor): Input tensor to apply the activation function to.

        Returns:
            (torch.Tensor): Output tensor after applying the AGLU activation function, with the same shape as the input.
        """
        lam = torch.clamp(self.lambd, min=0.0001)  # Clamp lambda to avoid division by zero
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))
