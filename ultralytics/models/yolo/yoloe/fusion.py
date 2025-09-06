# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _aggregate_vpe_by_map(vpe: torch.Tensor, fuse_map: Dict[int, List[int]], num_text: int) -> torch.Tensor:
    """
    Aggregate visual prompt embeddings by a provided mapping from text class id -> vpe indices.

    Args:
        vpe: (B, Nv, D) visual prompt embeddings
        fuse_map: dict mapping text class idx -> list of visual indices to aggregate
        num_text: number of text classes

    Returns:
        (B, num_text, D) aggregated visual embeddings aligned with text class indices
    """
    B, Nv, D = vpe.shape
    device = vpe.device
    out = torch.zeros(B, num_text, D, device=device, dtype=vpe.dtype)
    for t_idx in range(num_text):
        idxs = fuse_map.get(t_idx, [])
        if len(idxs) == 0:
            continue
        sel = vpe[:, idxs, :]  # (B, k, D)
        out[:, t_idx, :] = sel.mean(dim=1)
    return out


def fuse_tpe_vpe(
    tpe: Optional[torch.Tensor],
    vpe: Optional[torch.Tensor],
    *,
    mode: str = "concat",
    alpha: float = 0.5,
    fuse_map: Optional[Dict[int, List[int]]] = None,
) -> torch.Tensor:
    """
    Fuse text and visual prompt embeddings.

    Args:
        tpe: (B, Nt, D) or None
        vpe: (B, Nv, D) or None
        mode: "concat" | "sum"
        alpha: weight for "sum" mode; fused = alpha*tpe + (1-alpha)*vpe_agg
        fuse_map: mapping from text class index -> list of vpe indices to aggregate.

    Returns:
        embed: (B, Nc, D) fused class embedding bank
    """
    if tpe is None and vpe is None:
        raise ValueError("Both tpe and vpe are None; nothing to fuse.")

    if mode == "concat" or tpe is None or vpe is None:
        # If any is None, fallback to the other; concat is identity when one side missing
        parts = [x for x in (tpe, vpe) if x is not None]
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    # Now both tpe and vpe exist
    assert tpe.ndim == 3 and vpe.ndim == 3, "tpe and vpe must be (B, N, D)"
    B, Nt, D = tpe.shape
    Bv, Nv, Dv = vpe.shape
    assert B == Bv and D == Dv, "Batch size and embedding dim must match for tpe/vpe"

    if fuse_map is not None:
        vpe_agg = _aggregate_vpe_by_map(vpe, fuse_map=fuse_map, num_text=Nt)  # (B, Nt, D)
    else:
        # 1:1 assumption if no map provided
        if Nv != Nt:
            raise ValueError(
                f"fuse_map not provided and Nv ({Nv}) != Nt ({Nt}); "
                "either provide fuse_map or ensure equal counts."
            )
        vpe_agg = vpe

    if mode == "sum":
        # Weighted sum, normalize to unit length for stability
        fused = alpha * tpe + (1.0 - alpha) * vpe_agg
        fused = F.normalize(fused, dim=-1)
        return fused

    raise ValueError(f"Unknown fuse mode: {mode}")
