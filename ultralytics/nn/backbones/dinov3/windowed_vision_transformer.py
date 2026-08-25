"""Windowed attention variant of DINOv3 Vision Transformer.

Intermediate layers run local (windowed) self-attention while layers listed in
``global_block_indexes`` run global self-attention. Because only the token
routing changes — not the layer weights — pretrained DINOv3 checkpoints load
without modification.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor

from .vision_transformer import DinoVisionTransformer


class WindowedDinoVisionTransformer(DinoVisionTransformer):
    """DINOv3 ViT with spatial window partitioning for selected layers.

    Tokens are partitioned into ``num_windows x num_windows`` spatial windows
    after patch embedding.  All layers run windowed (local) attention **except**
    those listed in ``global_block_indexes``, which merge windows and run full
    global attention so that cross-window information can propagate.

    Output / feature-extraction layers run global attention, and everything
    else is windowed.

    Args:
        name: Config key in ``vision_transformer.configs`` (e.g. ``dinov3_vits16``).
        num_windows: Number of windows along each spatial axis.
            Total window count is ``num_windows ** 2``.  Input patch-grid
            dimensions must be divisible by this value.
        global_block_indexes: Block indices that run **global** (unwindowed)
            attention.  Every other block runs windowed attention.
            Defaults to ``[5, 8, 11]`` (the interaction / output layers).
    """

    def __init__(
        self,
        name: str,
        qk_layernorm: bool | None = None,
        num_windows: int = 4,
        global_block_indexes: list[int] | None = None,
    ):
        super().__init__(name=name, qk_layernorm=qk_layernorm)
        self.num_windows = num_windows
        # Default: output / interaction layers get global attention.
        self.global_block_indexes = set(
            global_block_indexes if global_block_indexes is not None else [5, 8, 11]
        )

    # ------------------------------------------------------------------
    # Window helpers
    # ------------------------------------------------------------------

    def _window_partition(self, x: Tensor, H: int, W: int) -> Tensor:
        """Partition full token sequence into spatial windows.

        Args:
            x: ``[B, 1+S+H*W, C]`` — cls + storage + patch tokens.
            H, W: Patch-grid spatial dimensions.

        Returns:
            ``[B*nw², 1+S+Hw*Ww, C]`` — per-window token sequences.
        """
        nw = self.num_windows
        prefix_len = 1 + self.n_storage_tokens
        B, _N, C = x.shape
        Hw, Ww = H // nw, W // nw

        prefix = x[:, :prefix_len]                          # [B, pfx, C]
        patches = x[:, prefix_len:]                         # [B, H*W, C]

        # Spatial reshape → window partition
        patches = patches.view(B, nw, Hw, nw, Ww, C)
        patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
        patches = patches.reshape(B * nw * nw, Hw * Ww, C)

        # Duplicate prefix (cls + storage) per window
        prefix = prefix.unsqueeze(1).expand(-1, nw * nw, -1, -1)
        prefix = prefix.reshape(B * nw * nw, prefix_len, C)

        return torch.cat([prefix, patches], dim=1)

    def _window_merge(self, x: Tensor, H: int, W: int, B_orig: int) -> Tensor:
        """Merge windowed tokens back into a full spatial sequence.

        Prefix tokens (cls + storage) are averaged across windows.  Patch
        tokens are reassembled into the original spatial layout.

        Args:
            x: ``[B*nw², 1+S+Hw*Ww, C]`` — per-window tokens.
            H, W: Full patch-grid dimensions.
            B_orig: Original (non-windowed) batch size.

        Returns:
            ``[B, 1+S+H*W, C]`` — full token sequence.
        """
        nw = self.num_windows
        nw2 = nw * nw
        prefix_len = 1 + self.n_storage_tokens
        Hw, Ww = H // nw, W // nw
        C = x.shape[-1]

        prefix = x[:, :prefix_len]                          # [B*nw², pfx, C]
        patches = x[:, prefix_len:]                         # [B*nw², Hw*Ww, C]

        # Collapse prefix across windows (mean keeps gradients flowing)
        prefix = prefix.view(B_orig, nw2, prefix_len, C).mean(dim=1)

        # Reassemble patches to full spatial order
        patches = patches.view(B_orig, nw, nw, Hw, Ww, C)
        patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous()
        patches = patches.reshape(B_orig, H * W, C)

        return torch.cat([prefix, patches], dim=1)

    # ------------------------------------------------------------------
    # Override: intermediate layer extraction with windowed attention
    # ------------------------------------------------------------------

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int | Sequence = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        B = x.shape[0]
        nw = self.num_windows
        Hw, Ww = H // nw, W // nw

        assert H % nw == 0 and W % nw == 0, (
            f"Patch grid ({H}x{W}) must be divisible by num_windows={nw}"
        )

        total_block_len = len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        blocks_to_take_set = set(blocks_to_take)

        output: list[Tensor] = []

        # Begin in windowed form
        x = self._window_partition(x, H, W)
        is_windowed = True

        for i, blk in enumerate(self.blocks):
            is_global = i in self.global_block_indexes

            # --- transition between windowed / full forms ---
            if is_global and is_windowed:
                x = self._window_merge(x, H, W, B)
                is_windowed = False
            elif not is_global and not is_windowed:
                x = self._window_partition(x, H, W)
                is_windowed = True

            # --- RoPE for the current spatial scale ---
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=Hw, W=Ww) if is_windowed else self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None

            x = blk(x, rope_sincos)

            # --- collect output in full (unwindowed) form ---
            if i in blocks_to_take_set:
                output.append(x if not is_windowed else self._window_merge(x, H, W, B))

        assert len(output) == len(blocks_to_take), (
            f"only {len(output)} / {len(blocks_to_take)} blocks found"
        )
        return output
