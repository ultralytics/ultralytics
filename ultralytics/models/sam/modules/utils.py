# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def select_closest_cond_frames(frame_idx: int, cond_frame_outputs: dict[int, Any], max_cond_frame_num: int):
    """Select the closest conditioning frames to a given frame index.

    Args:
        frame_idx (int): Current frame index.
        cond_frame_outputs (dict[int, Any]): Dictionary of conditioning frame outputs keyed by frame indices.
        max_cond_frame_num (int): Maximum number of conditioning frames to select.

    Returns:
        selected_outputs (dict[int, Any]): Selected items from cond_frame_outputs.
        unselected_outputs (dict[int, Any]): Items not selected from cond_frame_outputs.

    Examples:
        >>> frame_idx = 5
        >>> cond_frame_outputs = {1: "a", 3: "b", 7: "c", 9: "d"}
        >>> max_cond_frame_num = 2
        >>> selected, unselected = select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num)
        >>> print(selected)
        {3: 'b', 7: 'c'}
        >>> print(unselected)
        {1: 'a', 9: 'd'}
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # The closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # The closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # Add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds: torch.Tensor, dim: int, temperature: float = 10000):
    """Generate 1D sinusoidal positional embeddings for given positions and dimensions.

    Args:
        pos_inds (torch.Tensor): Position indices for which to generate embeddings.
        dim (int): Dimension of the positional embeddings. Should be an even number.
        temperature (float, optional): Scaling factor for the frequency of the sinusoidal functions.

    Returns:
        (torch.Tensor): Sinusoidal positional embeddings with shape (pos_inds.shape, dim).

    Examples:
        >>> pos = torch.tensor([0, 1, 2, 3])
        >>> embeddings = get_1d_sine_pe(pos, 128)
        >>> embeddings.shape
        torch.Size([4, 128])
    """
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=pos_inds.dtype, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def init_t_xy(end_x: int, end_y: int, scale: float = 1.0, offset: int = 0):
    """Initialize 1D and 2D coordinate tensors for a grid of specified dimensions.

    This function creates coordinate tensors for a grid with dimensions end_x × end_y. It generates a linear index
    tensor and corresponding x and y coordinate tensors.

    Args:
        end_x (int): Width of the grid (number of columns).
        end_y (int): Height of the grid (number of rows).
        scale (float): Scaling factor to apply to the coordinates.
        offset (int): Offset to add to the coordinates.

    Returns:
        t_x (torch.Tensor): X-coordinates for each position, with shape (end_x * end_y).
        t_y (torch.Tensor): Y-coordinates for each position, with shape (end_x * end_y).

    Examples:
        >>> t_x, t_y = init_t_xy(3, 2)
        >>> print(t_x)
        tensor([0., 1., 2., 0., 1., 2.])
        >>> print(t_y)
        tensor([0., 0., 0., 1., 1., 1.])
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0, scale_pos: float = 1.0):
    """Compute axial complex exponential positional encodings for 2D spatial positions in a grid.

    This function generates complex exponential positional encodings for a 2D grid of spatial positions, using separate
    frequency components for the x and y dimensions.

    Args:
        dim (int): Dimension of the positional encoding.
        end_x (int): Width of the 2D grid.
        end_y (int): Height of the 2D grid.
        theta (float, optional): Scaling factor for frequency computation.
        scale_pos (float, optional): Scaling factor for position coordinates.

    Returns:
        (torch.Tensor): Complex exponential positional encodings with shape (end_x*end_y, dim//2).

    Examples:
        >>> dim, end_x, end_y = 128, 8, 8
        >>> freqs_cis = compute_axial_cis(dim, end_x, end_y)
        >>> freqs_cis.shape
        torch.Size([64, 64])
    """
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y, scale=scale_pos)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape frequency tensor for broadcasting with input tensor.

    Reshapes a frequency tensor to ensure dimensional compatibility for broadcasting with an input tensor. This function
    is typically used in positional encoding operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor with shape matching the last two dimensions of x.
        x (torch.Tensor): Input tensor to broadcast with.

    Returns:
        (torch.Tensor): Reshaped frequency tensor ready for broadcasting with the input tensor.

    Raises:
        AssertionError: If the shape of freqs_cis doesn't match the last two dimensions of x.
    """
    ndim = x.ndim
    assert ndim >= 2
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    """Apply rotary positional encoding to query and key tensors.

    This function applies rotary positional encoding (RoPE) to query and key tensors using complex-valued frequency
    components. RoPE is a technique that injects relative position information into self-attention mechanisms.

    Args:
        xq (torch.Tensor): Query tensor to encode with positional information.
        xk (torch.Tensor): Key tensor to encode with positional information.
        freqs_cis (torch.Tensor): Complex-valued frequency components for rotary encoding with shape matching the last
            two dimensions of xq.
        repeat_freqs_k (bool, optional): Whether to repeat frequency components along sequence length dimension to match
            key sequence length.

    Returns:
        xq_out (torch.Tensor): Query tensor with rotary positional encoding applied.
        xk_out (torch.Tensor): Key tensor with rotary positional encoding applied, or original xk if xk is empty.

    Examples:
        >>> import torch
        >>> xq = torch.randn(2, 8, 16, 64)  # [batch, heads, seq_len, dim]
        >>> xk = torch.randn(2, 8, 16, 64)
        >>> freqs_cis = compute_axial_cis(64, 4, 4)  # For a 4x4 spatial grid with dim=64
        >>> q_encoded, k_encoded = apply_rotary_enc(xq, xk, freqs_cis)
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # No keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # Repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k and (r := xk_.shape[-2] // xq_.shape[-2]) > 1:
        # MPS doesn't support repeat on complex tensors, decompose to real representation
        if freqs_cis.device.type == "mps":
            freqs_cis = torch.view_as_real(freqs_cis)
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 3)), r, 1, 1)
            freqs_cis = torch.view_as_complex(freqs_cis.contiguous())
        else:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def window_partition(x: torch.Tensor, window_size: int):
    """Partition input tensor into non-overlapping windows with padding if needed.

    Args:
        x (torch.Tensor): Input tensor with shape (B, H, W, C).
        window_size (int): Size of each window.

    Returns:
        windows (torch.Tensor): Partitioned windows with shape (B * num_windows, window_size, window_size, C).
        padded_h_w (tuple[int, int]): Padded height and width before partition.

    Examples:
        >>> x = torch.randn(1, 16, 16, 3)
        >>> windows, (Hp, Wp) = window_partition(x, window_size=4)
        >>> print(windows.shape, Hp, Wp)
        torch.Size([16, 4, 4, 3]) 16 16
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]):
    """Unpartition windowed sequences into original sequences and remove padding.

    This function reverses the windowing process, reconstructing the original input from windowed segments and removing
    any padding that was added during the windowing process.

    Args:
        windows (torch.Tensor): Input tensor of windowed sequences with shape (B * num_windows, window_size,
            window_size, C), where B is the batch size, num_windows is the number of windows, window_size is the size of
            each window, and C is the number of channels.
        window_size (int): Size of each window.
        pad_hw (tuple[int, int]): Padded height and width (Hp, Wp) of the input before windowing.
        hw (tuple[int, int]): Original height and width (H, W) of the input before padding and windowing.

    Returns:
        (torch.Tensor): Unpartitioned sequences with shape (B, H, W, C), where B is the batch size, H and W are the
            original height and width, and C is the number of channels.

    Examples:
        >>> windows = torch.rand(32, 8, 8, 64)  # 32 windows of size 8x8 with 64 channels
        >>> pad_hw = (16, 16)  # Padded height and width
        >>> hw = (15, 14)  # Original height and width
        >>> x = window_unpartition(windows, window_size=8, pad_hw=pad_hw, hw=hw)
        >>> print(x.shape)
        torch.Size([1, 15, 14, 64])
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """Extract relative positional embeddings based on query and key sizes.

    Args:
        q_size (int): Size of the query.
        k_size (int): Size of the key.
        rel_pos (torch.Tensor): Relative position embeddings with shape (L, C), where L is the maximum relative distance
            and C is the embedding dimension.

    Returns:
        (torch.Tensor): Extracted positional embeddings according to relative positions, with shape (q_size, k_size, C).

    Examples:
        >>> q_size, k_size = 8, 16
        >>> rel_pos = torch.randn(31, 64)  # 31 = 2 * max(8, 16) - 1
        >>> extracted_pos = get_rel_pos(q_size, k_size, rel_pos)
        >>> print(extracted_pos.shape)
        torch.Size([8, 16, 64])
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> torch.Tensor:
    """Add decomposed Relative Positional Embeddings to the attention map.

    This function calculates and applies decomposed Relative Positional Embeddings as described in the MVITv2
    paper. It enhances the attention mechanism by incorporating spatial relationships between query and key
    positions.

    Args:
        attn (torch.Tensor): Attention map with shape (B, q_h * q_w, k_h * k_w).
        q (torch.Tensor): Query tensor in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (torch.Tensor): Relative position embeddings for height axis with shape (Lh, C).
        rel_pos_w (torch.Tensor): Relative position embeddings for width axis with shape (Lw, C).
        q_size (tuple[int, int]): Spatial sequence size of query q as (q_h, q_w).
        k_size (tuple[int, int]): Spatial sequence size of key k as (k_h, k_w).

    Returns:
        (torch.Tensor): Updated attention map with added relative positional embeddings, shape (B, q_h * q_w, k_h *
            k_w).

    Examples:
        >>> B, C, q_h, q_w, k_h, k_w = 1, 64, 8, 8, 8, 8
        >>> attn = torch.rand(B, q_h * q_w, k_h * k_w)
        >>> q = torch.rand(B, q_h * q_w, C)
        >>> rel_pos_h = torch.rand(2 * max(q_h, k_h) - 1, C)
        >>> rel_pos_w = torch.rand(2 * max(q_w, k_w) - 1, C)
        >>> q_size, k_size = (q_h, q_w), (k_h, k_w)
        >>> updated_attn = add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size)
        >>> print(updated_attn.shape)
        torch.Size([1, 64, 64])

    References:
        https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        B, q_h * q_w, k_h * k_w
    )

    return attn


def get_abs_pos(
    abs_pos: torch.Tensor,
    has_cls_token: bool,
    hw: tuple[int, int],
    retain_cls_token: bool = False,
    tiling: bool = False,
) -> torch.Tensor:
    """Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token dimension for the
    original embeddings.

    Args:
        abs_pos (torch.Tensor): Absolute positional embeddings with shape (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (tuple[int, int]): Size of input image tokens.
        retain_cls_token (bool): Whether to retain the cls_token.
        tiling (bool): Whether to tile the embeddings, *instead* of interpolation (a la abs_win).

    Returns:
        (torch.Tensor): Absolute positional embeddings after processing with shape (1, H, W, C) if retain_cls_token is
            False, otherwise (1, 1+H*W, C).
    """
    if retain_cls_token:
        assert has_cls_token

    h, w = hw
    if has_cls_token:
        cls_pos = abs_pos[:, :1]
        abs_pos = abs_pos[:, 1:]

    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        if tiling:
            new_abs_pos = new_abs_pos.tile([1, 1] + [x // y + 1 for x, y in zip((h, w), new_abs_pos.shape[2:])])[
                :, :, :h, :w
            ]
        else:
            new_abs_pos = F.interpolate(
                new_abs_pos,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )

        if not retain_cls_token:
            return new_abs_pos.permute(0, 2, 3, 1)
        else:
            # add cls_token back, flatten spatial dims
            assert has_cls_token
            return torch.cat(
                [cls_pos, new_abs_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)],
                dim=1,
            )

    else:
        if not retain_cls_token:
            return abs_pos.reshape(1, h, w, -1)
        else:
            assert has_cls_token
            return torch.cat([cls_pos, abs_pos], dim=1)


def concat_rel_pos(
    q: torch.Tensor,
    k: torch.Tensor,
    q_hw: tuple[int, int],
    k_hw: tuple[int, int],
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    rescale: bool = False,
    relative_coords: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate rel pos coeffs to the q & k tensors, so that qk^T is now effectively including rel pos biases.

    Args:
        q (torch.Tensor): Query tensor with shape (B, L_q, C).
        k (torch.Tensor): Key tensor with shape (B, L_k, C).
        q_hw (tuple[int, int]): Spatial size of query tensors as (height, width).
        k_hw (tuple[int, int]): Spatial size of key tensors as (height, width).
        rel_pos_h (torch.Tensor): Relative positional embeddings for the height axis.
        rel_pos_w (torch.Tensor): Relative positional embeddings for the width axis.
        rescale (bool): Whether to rescale for use with SDPA, which would scale by the wrong factor due to the concat.
        relative_coords (torch.Tensor | None): Precomputed relative coords index tensor.

    Returns:
        q (torch.Tensor): Query tensor padded so that qk^T accounts for relative position biases.
        k (torch.Tensor): Key tensor padded so that qk^T accounts for relative position biases.
    """
    q_h, q_w = q_hw
    k_h, k_w = k_hw

    assert (q_h == q_w) and (k_h == k_w), "only square inputs supported"

    if relative_coords is not None:
        Rh = rel_pos_h[relative_coords]
        Rw = rel_pos_w[relative_coords]
    else:
        Rh = get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)

    old_scale = dim**0.5
    new_scale = (dim + k_h + k_w) ** 0.5 if rescale else old_scale  # for sdpa
    # attn will be divided by new_scale, but we want to divide q by old_scale
    scale_ratio = new_scale / old_scale

    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) * new_scale  # (B, q_h, q_w, k_h)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw) * new_scale  # (B, q_h, q_w, k_w)

    eye_h = torch.eye(k_h, dtype=q.dtype, device=q.device)
    eye_w = torch.eye(k_w, dtype=q.dtype, device=q.device)

    eye_h = eye_h.view(1, k_h, 1, k_h).expand([B, k_h, k_w, k_h])
    eye_w = eye_w.view(1, 1, k_w, k_w).expand([B, k_h, k_w, k_w])

    q = torch.cat([r_q * scale_ratio, rel_h, rel_w], dim=-1).view(B, q_h * q_w, -1)
    k = torch.cat([k.view(B, k_h, k_w, -1), eye_h, eye_w], dim=-1).view(B, k_h * k_w, -1)

    return q, k
