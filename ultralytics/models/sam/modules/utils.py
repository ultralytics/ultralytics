# Ultralytics YOLO 🚀, AGPL-3.0 license

from typing import Tuple

import torch
import torch.nn.functional as F


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Selects the closest conditioning frames to a given frame index.

    Args:
        frame_idx (int): Current frame index.
        cond_frame_outputs (Dict[int, Any]): Dictionary of conditioning frame outputs keyed by frame indices.
        max_cond_frame_num (int): Maximum number of conditioning frames to select.

    Returns:
        (Tuple[Dict[int, Any], Dict[int, Any]]): A tuple containing two dictionaries:
            - selected_outputs: Selected items from cond_frame_outputs.
            - unselected_outputs: Items not selected from cond_frame_outputs.

    Examples:
        >>> frame_idx = 5
        >>> cond_frame_outputs = {1: 'a', 3: 'b', 7: 'c', 9: 'd'}
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

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs}

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """Generates 1D sinusoidal positional embeddings for given positions and dimensions."""
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def init_t_xy(end_x: int, end_y: int):
    """Initializes 1D and 2D coordinate tensors for a grid of specified dimensions."""
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """Computes axial complex exponential positional encodings for 2D spatial positions in a grid."""
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshapes frequency tensor for broadcasting with input tensor, ensuring dimensional compatibility."""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    """Applies rotary positional encoding to query and key tensors using complex-valued frequency components."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def window_partition(x, window_size):
    """
    Partitions input tensor into non-overlapping windows with padding if needed.

    Args:
        x (torch.Tensor): Input tensor with shape (B, H, W, C).
        window_size (int): Size of each window.

    Returns:
        (Tuple[torch.Tensor, Tuple[int, int]]): A tuple containing:
            - windows (torch.Tensor): Partitioned windows with shape (B * num_windows, window_size, window_size, C).
            - (Hp, Wp) (Tuple[int, int]): Padded height and width before partition.

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


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Unpartitions windowed sequences into original sequences and removes padding.

    This function reverses the windowing process, reconstructing the original input from windowed segments
    and removing any padding that was added during the windowing process.

    Args:
        windows (torch.Tensor): Input tensor of windowed sequences with shape (B * num_windows, window_size,
            window_size, C), where B is the batch size, num_windows is the number of windows, window_size is
            the size of each window, and C is the number of channels.
        window_size (int): Size of each window.
        pad_hw (Tuple[int, int]): Padded height and width (Hp, Wp) of the input before windowing.
        hw (Tuple[int, int]): Original height and width (H, W) of the input before padding and windowing.

    Returns:
        (torch.Tensor): Unpartitioned sequences with shape (B, H, W, C), where B is the batch size, H and W
            are the original height and width, and C is the number of channels.

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
    """
    Extracts relative positional embeddings based on query and key sizes.

    Args:
        q_size (int): Size of the query.
        k_size (int): Size of the key.
        rel_pos (torch.Tensor): Relative position embeddings with shape (L, C), where L is the maximum relative
            distance and C is the embedding dimension.

    Returns:
        (torch.Tensor): Extracted positional embeddings according to relative positions, with shape (q_size,
            k_size, C).

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
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Adds decomposed Relative Positional Embeddings to the attention map.

    This function calculates and applies decomposed Relative Positional Embeddings as described in the MVITv2
    paper. It enhances the attention mechanism by incorporating spatial relationships between query and key
    positions.

    Args:
        attn (torch.Tensor): Attention map with shape (B, q_h * q_w, k_h * k_w).
        q (torch.Tensor): Query tensor in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (torch.Tensor): Relative position embeddings for height axis with shape (Lh, C).
        rel_pos_w (torch.Tensor): Relative position embeddings for width axis with shape (Lw, C).
        q_size (Tuple[int, int]): Spatial sequence size of query q as (q_h, q_w).
        k_size (Tuple[int, int]): Spatial sequence size of key k as (k_h, k_w).

    Returns:
        (torch.Tensor): Updated attention map with added relative positional embeddings, shape
            (B, q_h * q_w, k_h * k_w).

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
