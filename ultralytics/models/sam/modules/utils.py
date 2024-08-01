# Ultralytics YOLO 🚀, AGPL-3.0 license

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
    """Initializes 1D and 2D coordinate tensors for a grid of size end_x by end_y."""
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """Computes axial complex exponential positional encodings for 2D spatial positions."""
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshapes frequency tensor for broadcasting, ensuring compatibility with input tensor dimensions."""
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
