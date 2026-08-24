# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tensor ops that model heads adapt per export format."""

from __future__ import annotations

import torch


def index_dtype(fmt: str | None) -> torch.dtype:
    """Return the index dtype an export format accepts.

    Args:
        fmt (str | None): Export format, or None outside export.

    Returns:
        (torch.dtype): int32 for LiteRT, whose GPU delegates reject int64 index math, int64 otherwise.
    """
    return torch.int32 if fmt == "litert" else torch.int64


def topk_groups(fmt: str | None, export: bool, dynamic: bool) -> int:
    """Return the group count for a grouped top-k.

    Args:
        fmt (str | None): Export format, or None outside export.
        export (bool): Whether the head is running in export mode.
        dynamic (bool): Whether the export uses dynamic axes.

    Returns:
        (int): 8 for static TensorRT exports, where a single large top-k is slow, 1 otherwise.
    """
    return 8 if export and fmt == "engine" and not dynamic else 1


def gather(x: torch.Tensor, index: torch.Tensor, fmt: str | None = None) -> torch.Tensor:
    """Select index rows of x along dim 1.

    Args:
        x (torch.Tensor): Source tensor with shape (batch, n) or (batch, n, channels).
        index (torch.Tensor): Indices into dim 1 with shape (batch, k).
        fmt (str | None): Export format, or None outside export.

    Returns:
        (torch.Tensor): Selected rows with shape (batch, k) or (batch, k, channels).
    """
    if fmt == "litert":  # gather lowers to gather_nd, which GPU delegates do not implement
        b, n = x.shape[:2]
        offset = torch.arange(b, device=x.device, dtype=index.dtype)[..., None] * n
        return x.flatten(0, 1).index_select(0, (index + offset).flatten()).view(b, index.shape[1], *x.shape[2:])
    if fmt == "coreml":  # MIL types int64 gather indices as fp32 and then rejects them
        return x[torch.arange(x.shape[0])[..., None], index]
    return x.gather(1, index if x.ndim == 2 else index[..., None].expand(-1, -1, x.shape[-1]))
