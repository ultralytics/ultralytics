# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Utility functions for fusing text prompt embeddings (TPE) and visual prompt embeddings (VPE).

This module provides various methods to combine text and visual embeddings for enhanced prompting in YOLO-E models,
supporting backward compatibility while enabling new fusion strategies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PromptEmbeddingFusion:
    """
    Utility class for fusing text prompt embeddings (TPE) and visual prompt embeddings (VPE).

    Supports multiple fusion strategies:
    - 'concat': Simple concatenation along the sequence dimension
    - 'sum': Element-wise addition (requires same dimensions)
    - 'attention': Cross-attention based fusion

    Args:
        method (str): Fusion method - 'concat', 'sum', or 'attention'
        embed_dim (int): Embedding dimension, required for attention-based fusion

    Examples:
        >>> fusion = PromptEmbeddingFusion("concat")
        >>> tpe = torch.randn(1, 10, 512)  # Text prompt embeddings
        >>> vpe = torch.randn(1, 5, 512)  # Visual prompt embeddings
        >>> fused = fusion(tpe, vpe)
        >>> print(fused.shape)  # torch.Size([1, 15, 512])

        >>> fusion = PromptEmbeddingFusion("attention", embed_dim=512)
        >>> fused = fusion(tpe, vpe)
        >>> print(fused.shape)  # torch.Size([1, 10, 512])
    """

    def __init__(self, method: str = "concat", embed_dim: int = 512):
        """Initialize the fusion utility."""
        self.method = method.lower()
        self.embed_dim = embed_dim

        if self.method not in ["concat", "sum", "attention"]:
            raise ValueError(f"Unsupported fusion method: {method}. Supported: 'concat', 'sum', 'attention'")

        # Initialize attention module for attention-based fusion
        if self.method == "attention":
            self.attention = CrossAttentionFusion(embed_dim)

    def __call__(self, tpe: torch.Tensor | None, vpe: torch.Tensor | None) -> torch.Tensor:
        """
        Fuse text and visual prompt embeddings.

        Args:
            tpe (torch.Tensor, optional): Text prompt embeddings with shape (B, N_t, D)
            vpe (torch.Tensor, optional): Visual prompt embeddings with shape (B, N_v, D)

        Returns:
            torch.Tensor: Fused embeddings
        """
        return self.fuse(tpe, vpe)

    def fuse(self, tpe: torch.Tensor | None, vpe: torch.Tensor | None) -> torch.Tensor:
        """
        Fuse text and visual prompt embeddings using the specified method.

        Args:
            tpe (torch.Tensor, optional): Text prompt embeddings with shape (B, N_t, D)
            vpe (torch.Tensor, optional): Visual prompt embeddings with shape (B, N_v, D)

        Returns:
            torch.Tensor: Fused embeddings

        Raises:
            ValueError: If no embeddings provided or incompatible shapes for sum fusion
        """
        # Handle case where only one type of embedding is provided
        if tpe is None and vpe is None:
            raise ValueError("At least one of TPE or VPE must be provided")
        elif tpe is None:
            return vpe
        elif vpe is None:
            return tpe

        # Validate input shapes
        if tpe.ndim != 3 or vpe.ndim != 3:
            raise ValueError("TPE and VPE must be 3D tensors with shape (B, N, D)")
        if tpe.shape[0] != vpe.shape[0]:
            raise ValueError("TPE and VPE must have the same batch size")
        if tpe.shape[2] != vpe.shape[2]:
            raise ValueError("TPE and VPE must have the same embedding dimension")

        # Apply fusion method
        if self.method == "concat":
            return self._concat_fusion(tpe, vpe)
        elif self.method == "sum":
            return self._sum_fusion(tpe, vpe)
        elif self.method == "attention":
            return self._attention_fusion(tpe, vpe)

    def _concat_fusion(self, tpe: torch.Tensor, vpe: torch.Tensor) -> torch.Tensor:
        """Concatenate TPE and VPE along sequence dimension."""
        return torch.cat([tpe, vpe], dim=1)

    def _sum_fusion(self, tpe: torch.Tensor, vpe: torch.Tensor) -> torch.Tensor:
        """Element-wise addition of TPE and VPE (requires same sequence length)."""
        if tpe.shape[1] != vpe.shape[1]:
            raise ValueError(
                f"For sum fusion, TPE and VPE must have same sequence length. "
                f"Got TPE: {tpe.shape[1]}, VPE: {vpe.shape[1]}"
            )
        return tpe + vpe

    def _attention_fusion(self, tpe: torch.Tensor, vpe: torch.Tensor) -> torch.Tensor:
        """Cross-attention based fusion using TPE as queries and VPE as keys/values."""
        return self.attention(tpe, vpe)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module for fusing text and visual prompt embeddings.

    Uses text embeddings as queries and visual embeddings as keys and values,
    producing attended text embeddings that incorporate visual information.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        """Initialize cross-attention fusion module."""
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Layer normalization for residual connection
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, tpe: torch.Tensor, vpe: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention fusion.

        Args:
            tpe (torch.Tensor): Text prompt embeddings (B, N_t, D) - used as queries
            vpe (torch.Tensor): Visual prompt embeddings (B, N_v, D) - used as keys/values

        Returns:
            torch.Tensor: Attended text embeddings (B, N_t, D)
        """
        # Cross-attention: TPE as queries, VPE as keys and values
        attended_tpe, _ = self.multihead_attn(query=tpe, key=vpe, value=vpe)

        # Residual connection with layer norm
        fused = self.layer_norm(tpe + attended_tpe)

        return fused


def fuse_prompt_embeddings(
    tpe: torch.Tensor | None, vpe: torch.Tensor | None, method: str = "concat", embed_dim: int = 512
) -> torch.Tensor:
    """
    Convenience function to fuse text and visual prompt embeddings.

    Args:
        tpe (torch.Tensor, optional): Text prompt embeddings with shape (B, N_t, D)
        vpe (torch.Tensor, optional): Visual prompt embeddings with shape (B, N_v, D)
        method (str): Fusion method - 'concat', 'sum', or 'attention'
        embed_dim (int): Embedding dimension, required for attention-based fusion

    Returns:
        torch.Tensor: Fused embeddings

    Examples:
        >>> tpe = torch.randn(1, 10, 512)
        >>> vpe = torch.randn(1, 5, 512)
        >>> fused = fuse_prompt_embeddings(tpe, vpe, method="concat")
        >>> print(fused.shape)  # torch.Size([1, 15, 512])
    """
    fusion = PromptEmbeddingFusion(method=method, embed_dim=embed_dim)
    return fusion.fuse(tpe, vpe)
