"""
prune.py.

This module contains functions to prune YOLOv8 model components
at the channel and group level, including Conv2d, BatchNorm2d, Bottleneck, C2f blocks,
SPPF, and the Detect head. Pruning can be done globally or using
a component-wise YAML configuration.

Supported functionalities:
- Pruning standard Conv2d and YOLO Conv blocks
- Skip-aware pruning
- Group aware pruning with two options; preserve and remove, which preserves and removes the number of groups respectively
- Structured pruning of C2f, Bottleneck blocks
- Pruning of SPPF layers
- Pruning Detect head (regression and classification towers)
- Full model pruning with optional YAML-based per-component ratios

Intended for submission as part of Ultralytics YOLOv8 pruning
pipeline enhancements.
"""

from __future__ import annotations

import math
from copy import deepcopy
from functools import reduce
from pathlib import Path

import torch
import yaml
from torch.nn import BatchNorm2d, Conv2d, Sequential, Upsample

from ultralytics import YOLO
from ultralytics.nn.modules import SPPF, Bottleneck, C2f, Concat, Conv, Detect

# ============================================================
# Conv / YOLO Conv Pruning Functions
# ============================================================


def prune(
    weight_tensor: torch.Tensor, prune_ratio: float, norm_order: float, dim: int | tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prune filters or groups in a weight tensor based on their norms.

    Args:
        weight_tensor (torch.Tensor): Weight tensor to prune with shape (num_items, ...) where the first
            dimension contains items to prune.
        prune_ratio (float): Fraction of items to remove, range 0.0 to 1.0.
        norm_order (float): Order of norm for scoring (1 for L1, 2 for L2).
        dim (int | tuple): Dimension(s) over which to compute the norm, typically excluding the first dimension.

    Returns:
        pruned (torch.Tensor): Weight tensor containing only kept items with shape (num_kept, ...).
        mask (torch.Tensor): Boolean mask indicating kept (True) or pruned (False) items with shape (num_items,).

    Notes:
        - At least one filter or group is always kept, even if `prune_ratio == 1.0`.
    """
    device = weight_tensor.device
    num_items = weight_tensor.shape[0]

    # number of items to remove
    k = int(num_items * prune_ratio)

    # enforce at least one kept
    keep_count = max(1, num_items - k)

    # compute norms
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "vector_norm"):
        norm = torch.linalg.vector_norm(weight_tensor, ord=norm_order, dim=dim)
    else:
        norm = torch.norm(weight_tensor, p=norm_order, dim=dim)

    # top-k selection
    _, idx = torch.topk(norm, keep_count, largest=True, sorted=False)

    mask = torch.zeros(num_items, dtype=torch.bool, device=device)
    mask[idx] = True

    pruned = weight_tensor[mask].contiguous()
    return pruned, mask


def prune_channels_groupwise(
    weight_tensor: torch.Tensor, prune_ratio: float, norm_order: float, prune_groups: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Prune channels independently within each group, preserving group structure.

    Args:
        weight_tensor (torch.Tensor): Input weight tensor to prune with shape (C, H, W, D).
        prune_ratio (float): Proportion of channels to prune within each group (0.0 to 1.0).
        norm_order (float): Order of the norm used for pruning (e.g., 1 for L1, 2 for L2).
        prune_groups (int): Number of groups to divide channels into for independent pruning.

    Returns:
        pruned_weight (torch.Tensor): Pruned weight tensor with removed channels.
        mask (torch.Tensor): Binary mask indicating kept channels (1) and pruned channels (0).
        kept_out_channels (int): Total number of channels retained after pruning.
    """
    chunk_list = weight_tensor.chunk(prune_groups, 0)
    pruned_chunks, chunk_masks = [], []

    for chunk in chunk_list:
        chunk_w_pruned, chunk_w_mask = prune(chunk, prune_ratio, norm_order=norm_order, dim=(1, 2, 3))
        pruned_chunks.append(chunk_w_pruned)
        chunk_masks.append(chunk_w_mask)

    mask = torch.cat(chunk_masks)

    pruned_weight = torch.cat(pruned_chunks)
    kept_out_channels = mask.sum().item()

    return pruned_weight, mask, kept_out_channels


def prune_by_groups(
    weight_tensor: torch.Tensor, prune_ratio: float, norm_order: float, prune_groups: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Prune entire groups of channels based on group-wise norms.

    Args:
        weight_tensor (torch.Tensor): Weight tensor with shape (out_channels, in_channels, kH, kW).
            Must have out_channels divisible by prune_groups.
        prune_ratio (float): Fraction of groups to remove, range 0.0 to 1.0.
        norm_order (float): Order of norm for scoring groups (1 for L1, 2 for L2).
        prune_groups (int): Number of groups to divide channels into.

    Returns:
        pruned_weight (torch.Tensor): Pruned weight tensor with shape (kept_channels, in_channels, kH, kW).
        mask (torch.Tensor): Per-channel boolean mask indicating kept (True) or pruned (False) channels
            with shape (out_channels,).
        kept_out_channels (int): Number of output channels retained after pruning.
    """
    shape = weight_tensor.shape
    assert shape[0] % prune_groups == 0, f"out channels {shape[0]} not divisible by groups={prune_groups}"

    out_channels_per_group = shape[0] // prune_groups
    group_view = weight_tensor.contiguous().reshape(prune_groups, out_channels_per_group, *shape[1:])

    group_view_pruned, group_view_mask = prune(group_view, prune_ratio, norm_order=norm_order, dim=(1, 2, 3, 4))
    weight_view_pruned = group_view_pruned.contiguous().reshape((-1, *shape[1:]))

    # Expand per-group mask into per-channel masks
    mask = torch.cat([mask_val.repeat(out_channels_per_group) for mask_val in group_view_mask])
    kept_out_channels = mask.sum().item()

    return weight_view_pruned, mask, kept_out_channels


def apply_prev_mask(
    weight: torch.Tensor, mask_prev: torch.Tensor, groups: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Apply previous layer's output mask to prune input channels of current layer's weights.

    Args:
        weight (torch.Tensor): Weight tensor with shape (out_channels, in_channels//groups, kH, kW) for
            grouped convolutions, or (out_channels, in_channels, kH, kW) for standard convolutions (groups=1).
        mask_prev (torch.Tensor): Boolean mask indicating kept (True) or pruned (False) channels from the
            previous layer, with shape (prev_out_channels,).
        groups (int): Number of groups for grouped convolution. Use 1 for standard convolutions.

    Returns:
        pruned_weight (torch.Tensor): Weight tensor with pruned input channels.
        output_mask (torch.Tensor): Boolean mask indicating valid output channels with shape (out_channels,).
        new_groups (int): Number of remaining groups after pruning.

    Raises:
        RuntimeError: If all input channels or all groups are pruned.
    """
    if groups == 1:
        if mask_prev.sum() == 0:
            raise RuntimeError("All input channels were pruned! Layer invalid")
        pruned_w = weight[:, mask_prev, :, :]
        new_groups = 1
        return pruned_w, torch.ones(pruned_w.size(0), dtype=torch.bool, device=weight.device), new_groups

    pruned_chunks = []
    output_mask = []
    new_groups = 0

    for w_chunk, m_chunk in zip(weight.chunk(groups, 0), mask_prev.chunk(groups, 0)):
        if m_chunk.sum() == 0:
            # this group vanishes, don't add its outputs
            output_mask.append(torch.zeros(w_chunk.size(0), dtype=torch.bool, device=weight.device))
            continue
        pruned_chunks.append(w_chunk[:, m_chunk, :, :])
        output_mask.append(torch.ones(w_chunk.size(0), dtype=torch.bool, device=weight.device))
        new_groups += 1

    if new_groups == 0:
        raise RuntimeError("All groups were pruned.")

    return torch.cat(pruned_chunks, dim=0), torch.cat(output_mask, dim=0), new_groups


def prune_conv2d(
    conv_layer: Conv2d,
    prune_ratio: float,
    norm_order: float = 2.0,
    mask_prev: torch.Tensor = None,
    prune_groups=None,
    prune_type="preserve",
) -> tuple[Conv2d, torch.Tensor]:
    """
    Prune a Conv2d layer with group-aware pruning.

    Args:
        conv_layer (nn.Conv2d): Conv2d layer to prune.
        prune_ratio (float): Fraction of channels or groups to prune, range 0.0 to 1.0.
        norm_order (float): Order of norm for ranking channel importance (1 for L1, 2 for L2). Default is 2.0.
        mask_prev (torch.Tensor): Boolean mask indicating kept channels from the previous layer with shape
            (prev_out_channels,). Default is None.
        prune_groups (int): Number of groups for pruning. If None, uses conv_layer.groups. Default is None.
        prune_type (str): Pruning strategy, either 'preserve' (prune channels within groups) or 'remove'
            (prune entire groups). Default is 'preserve'.

    Returns:
        pruned_conv (nn.Conv2d): Pruned Conv2d layer with adjusted channels and groups.
        mask (torch.Tensor): Boolean mask indicating kept (True) or pruned (False) output channels with
            shape (original_out_channels,).
    """
    original_groups = conv_layer.groups
    prune_groups = prune_groups if prune_groups is not None else original_groups
    pruned_weight = conv_layer.weight
    out_channels_per_group = conv_layer.out_channels // original_groups
    in_channels_per_group = conv_layer.in_channels // original_groups

    new_groups = original_groups
    mask = torch.tensor([True] * conv_layer.out_channels)
    kept_out_channels = conv_layer.out_channels
    kept_in_channels = conv_layer.in_channels if mask_prev is None else int(torch.sum(mask_prev).item())

    if mask_prev is not None:
        pruned_weight, mask, new_groups = apply_prev_mask(pruned_weight, mask_prev, original_groups)
        kept_out_channels = int(torch.sum(mask).item())

        assert kept_out_channels % new_groups == 0, (
            f"Invalid grouped conv: {kept_out_channels} out_channels not divisible by {new_groups} groups"
        )

    if new_groups == original_groups:
        if prune_ratio > 0:
            if prune_type == "preserve":
                pruned_weight, mask, kept_out_channels = prune_channels_groupwise(
                    pruned_weight, prune_ratio, norm_order=norm_order, prune_groups=prune_groups
                )
                new_groups = original_groups

            elif prune_type == "remove":
                if original_groups == 1 and prune_groups == 1:
                    pruned_weight, mask, kept_out_channels = prune_channels_groupwise(
                        pruned_weight, prune_ratio, norm_order=norm_order, prune_groups=prune_groups
                    )

                else:
                    pruned_weight, mask, kept_out_channels = prune_by_groups(
                        pruned_weight, prune_ratio, norm_order=norm_order, prune_groups=prune_groups
                    )

                    if original_groups > 1:
                        remaining_groups = kept_out_channels // out_channels_per_group
                        assert kept_out_channels % out_channels_per_group == 0, (
                            f"Pruned out_channels={kept_out_channels} not divisible by channels/group={out_channels_per_group}"
                        )
                        new_groups = remaining_groups
                        kept_in_channels = in_channels_per_group * remaining_groups
            else:
                raise ValueError(f"Invalid prune_type value {prune_type}")
    else:
        print("Already pruned due to prev layer, skipping further pruning.")

    updated_conv = Conv2d(
        kept_in_channels,
        kept_out_channels,
        conv_layer.kernel_size,
        conv_layer.stride,
        conv_layer.padding,
        conv_layer.dilation,
        new_groups,
        conv_layer.bias is not None,
        conv_layer.padding_mode,
    )

    updated_conv.weight = torch.nn.Parameter(pruned_weight)

    if conv_layer.bias is not None:
        bias = conv_layer.bias
        pruning_mask_out = mask
        pruned_bias = bias[pruning_mask_out]
        updated_conv.bias = torch.nn.Parameter(pruned_bias)

    return updated_conv, mask


def prune_conv2d_with_skip(
    conv_layer: Conv2d, mask_skip: torch.Tensor, mask_prev: torch.Tensor = None
) -> tuple[Conv2d, torch.Tensor]:
    """
    Prune a Conv2d layer to match output channels from a skip connection source.

    Args:
        conv_layer (nn.Conv2d): Conv2d layer to prune.
        mask_skip (torch.Tensor): Boolean mask indicating kept output channels from the skip-producing layer.
        mask_prev (torch.Tensor): Boolean mask indicating kept input channels from the previous layer.
            Default is None.

    Returns:
        pruned_conv (nn.Conv2d): Pruned Conv2d layer with adjusted channels.
        mask (torch.Tensor): Output channel mask applied (same as mask_skip).
    """
    updated_out_channels = int(torch.sum(mask_skip).item())  # non-zero number of out_channels

    updated_in_channels = conv_layer.in_channels if mask_prev is None else int(torch.sum(mask_prev).numpy())

    updated_conv = Conv2d(
        updated_in_channels,
        updated_out_channels,
        conv_layer.kernel_size,
        conv_layer.stride,
        conv_layer.padding,
        conv_layer.dilation,
        conv_layer.groups,
        conv_layer.bias is not None,
        conv_layer.padding_mode,
    )

    weight = conv_layer.weight[mask_skip]
    if mask_prev is not None:
        weight = weight[:, mask_prev, :]
    updated_conv.weight = torch.nn.Parameter(weight)

    if conv_layer.bias is not None:
        updated_conv.bias = torch.nn.Parameter(conv_layer.bias[mask_skip])

    return updated_conv, mask_skip


def prune_conv(
    yolo_conv_layer: Conv,
    prune_ratio: float,
    norm_order: float = 2,
    mask_prev: torch.Tensor = None,
    prune_groups=None,
    prune_type="preserve",  # "remove"
) -> torch.Tensor:
    """
    Prune a YOLO Conv block (Conv2d + BatchNorm2d).

    Args:
        yolo_conv_layer (Conv): Ultralytics Conv module containing Conv2d and BatchNorm2d.
        prune_ratio (float): Fraction of channels or groups to prune, range 0.0 to 1.0.
        norm_order (float): Order of norm for ranking channel importance (1 for L1, 2 for L2). Default is 2.0.
        mask_prev (torch.Tensor): Boolean mask indicating kept channels from the previous layer with shape
            (prev_out_channels,). Default is None.
        prune_groups (int): Number of groups for pruning. If None, uses conv_layer.groups. Default is None.
        prune_type (str): Pruning strategy, either 'preserve' (prune channels within groups) or 'remove'
            (prune entire groups). Default is 'preserve'.

    Returns:
        mask (torch.Tensor): Boolean mask indicating kept (True) or pruned (False) output channels.
    """
    conv = yolo_conv_layer.conv
    bn = yolo_conv_layer.bn

    pruned_conv, conv_mask = prune_conv2d(
        conv_layer=conv,
        prune_ratio=prune_ratio,
        norm_order=norm_order,
        mask_prev=mask_prev,
        prune_groups=prune_groups,
        prune_type=prune_type,
    )
    pruned_bn, bn_mask = prune_batchnorm2d(bn_layer=bn, mask_prev=conv_mask)

    yolo_conv_layer.conv = pruned_conv
    yolo_conv_layer.bn = pruned_bn

    return conv_mask


def prune_conv_with_skip(
    yolo_conv_layer: Conv, mask_skip: torch.Tensor = None, mask_prev: torch.Tensor = None
) -> torch.Tensor:
    """
    Prune a YOLO Conv block to match output channels from a skip connection source.

    Args:
        yolo_conv_layer (Conv): Ultralytics Conv module containing Conv2d and BatchNorm2d.
        mask_skip (torch.Tensor): Boolean mask indicating kept output channels from the skip-producing layer.
            Default is None.
        mask_prev (torch.Tensor): Boolean mask indicating kept input channels from the previous layer.
            Default is None.

    Returns:
        mask (torch.Tensor): Boolean mask indicating kept output channels (same as mask_skip).
    """
    conv = yolo_conv_layer.conv
    bn = yolo_conv_layer.bn

    pruned_conv, conv_mask = prune_conv2d_with_skip(conv, mask_skip=mask_skip, mask_prev=mask_prev)
    pruned_bn, bn_mask = prune_batchnorm2d(bn_layer=bn, mask_prev=conv_mask)

    yolo_conv_layer.conv = pruned_conv
    yolo_conv_layer.bn = pruned_bn

    return conv_mask


# ============================================================
# BatchNorm Pruning
# ============================================================


def prune_batchnorm2d(bn_layer: BatchNorm2d, mask_prev: torch.Tensor) -> tuple[BatchNorm2d, torch.Tensor]:
    """
    Prune a BatchNorm2d layer to match channels from the previous layer.

    Args:
        bn_layer (BatchNorm2d): BatchNorm2d layer to prune.
        mask_prev (torch.Tensor): Boolean mask indicating kept channels with shape (num_channels,).

    Returns:
        pruned_bn (BatchNorm2d): Pruned BatchNorm2d layer with adjusted channels and parameters.
        mask (torch.Tensor): Channel mask applied (same as mask_prev).
    """
    assert mask_prev.dtype == torch.bool, "mask must be a boolean tensor"
    is_affine = bn_layer.affine
    updated_out_channels = int(torch.sum(mask_prev).item())

    pruned_bn = BatchNorm2d(updated_out_channels, eps=bn_layer.eps, momentum=bn_layer.momentum, affine=is_affine)

    p_running_mean = bn_layer.running_mean[mask_prev].clone()
    p_running_var = bn_layer.running_var[mask_prev].clone()

    pruned_bn.running_mean.data = p_running_mean
    pruned_bn.running_var.data = p_running_var

    if is_affine:
        p_weight = torch.nn.Parameter(bn_layer.weight[mask_prev].clone())
        p_bias = torch.nn.Parameter(bn_layer.bias[mask_prev].clone())

        pruned_bn.weight = p_weight
        pruned_bn.bias = p_bias

    return pruned_bn, mask_prev


# ============================================================
# Bottleneck / C2f / SPPF Pruning
# ============================================================


def prune_bottleneck(
    bottleneck: Bottleneck,
    prune_ratio: float,
    norm_order: float,
    mask_prev: torch.Tensor | None,
    mask_tracker: dict | None = None,
) -> torch.Tensor:
    """
    Prune a Bottleneck block in YOLO.

    When bottleneck.add is True (skip connection present), cv2 output channels are pruned to match
    mask_prev to ensure element-wise addition compatibility. Otherwise, cv2 is pruned independently.

    Args:
        bottleneck (Bottleneck): Bottleneck block to prune.
        prune_ratio (float): Fraction of channels to prune, range 0.0 to 1.0.
        norm_order (float): Order of norm for ranking channel importance (1 for L1, 2 for L2).
        mask_prev (torch.Tensor): Boolean mask indicating kept channels from the previous layer.
        mask_tracker (dict): Optional dictionary to store intermediate masks for debugging. Default is None.

    Returns:
        mask (torch.Tensor): Boolean mask indicating kept output channels from cv2.
    """
    cv1 = bottleneck.cv1
    cv2 = bottleneck.cv2
    add = bottleneck.add

    cv1_mask = prune_conv(yolo_conv_layer=cv1, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev)

    if add:
        if mask_prev is None:
            # No skip mask given → keep all cv2 outputs
            cv2_mask = prune_conv_with_skip(
                yolo_conv_layer=cv2, mask_skip=torch.ones(cv2.conv.out_channels, dtype=torch.bool), mask_prev=cv1_mask
            )
        else:
            cv2_mask = prune_conv_with_skip(yolo_conv_layer=cv2, mask_skip=mask_prev, mask_prev=cv1_mask)

    else:
        cv2_mask = prune_conv(yolo_conv_layer=cv2, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=cv1_mask)

    if mask_tracker is not None:
        mask_tracker["cv1_input"] = mask_prev
        mask_tracker["cv2_input"] = cv1_mask

        mask_tracker["cv1_output"] = cv1_mask
        mask_tracker["cv2_output"] = cv2_mask
    return cv2_mask


def prune_c2f(
    c2f_layer: C2f,
    prune_ratio: float,
    norm_order=2,
    mask_prev=None,
    prune_groups=None,
    prune_type="preserve",  # "remove"
    mask_tracker: dict | None = None,
) -> torch.Tensor:
    """
    Prune a C2f block in YOLO.

    The C2f block splits cv1 output into two chunks: the first chunk passes through directly, while
    the second passes through sequential Bottleneck modules. All outputs are concatenated and fed to cv2.
    Masks are propagated through this split-process-concat structure to maintain channel consistency.

    Args:
        c2f_layer (C2f): C2f block to prune.
        prune_ratio (float): Fraction of channels to prune, range 0.0 to 1.0.
        norm_order (float): Order of norm for ranking channel importance (1 for L1, 2 for L2). Default is 2.0.
        mask_prev (torch.Tensor): Boolean mask indicating kept channels from the previous layer. Default is None.
        prune_groups (int): Number of groups for pruning cv2. If None, uses cv2.conv.groups. Default is None.
        prune_type (str): Pruning strategy for cv2, either 'preserve' or 'remove'. Default is 'preserve'.
        mask_tracker (dict): Optional dictionary to store intermediate masks for debugging. Default is None.

    Returns:
        mask (torch.Tensor): Boolean mask indicating kept output channels from cv2.
    """
    cv1 = c2f_layer.cv1
    cv2 = c2f_layer.cv2
    m = c2f_layer.m

    weight_mask = prune_conv(cv1, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev, prune_groups=2)
    chunk1_w_mask, chunk2_w_mask = weight_mask.chunk(2, 0)

    c2f_layer.c = chunk1_w_mask.sum().item()

    # Optionally record chunk masks
    if mask_tracker is not None:
        mask_tracker["chunk1_mask"] = chunk1_w_mask
        mask_tracker["chunk2_mask"] = chunk2_w_mask

    m_masks = [chunk1_w_mask, chunk2_w_mask]

    for i, bottleneck in enumerate(m):
        mask_tracker_b = None if mask_tracker is None else {}
        bottleneck_mask = prune_bottleneck(
            bottleneck,
            prune_ratio=prune_ratio,
            norm_order=norm_order,
            mask_prev=m_masks[-1],
            mask_tracker=mask_tracker_b,
        )
        m_masks.append(bottleneck_mask)

        # record what each bottleneck received and produced
        if mask_tracker is not None:
            m_i = {"in": m_masks[-2], "out": bottleneck_mask}
            mask_tracker[f"m_{i}"] = m_i
            mask_tracker[f"bottleneck_{i}"] = mask_tracker_b

    mask_prev_cv2 = torch.cat(m_masks)

    if prune_groups:
        mask_cv2 = prune_conv(
            cv2,
            prune_ratio=prune_ratio,
            norm_order=norm_order,
            mask_prev=mask_prev_cv2,
            prune_groups=prune_groups,
            prune_type=prune_type,
        )
    else:
        mask_cv2 = prune_conv(cv2, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev_cv2)

    if mask_tracker is not None:
        mask_tracker["cv2_mask"] = mask_cv2

    return mask_cv2


def prune_sppf(
    sppf_layer: SPPF,
    prune_ratio: float,
    norm_order: float = 2,
    mask_prev: torch.Tensor = None,
    mask_tracker: dict | None = None,
) -> torch.Tensor:
    """
    Prune an SPPF (Spatial Pyramid Pooling - Fast) block in YOLO.

    The SPPF block applies multiple max-pooling operations and concatenates results, producing 4x the
    channels from cv1. The cv1 mask is repeated 4 times to match this concatenated input for cv2.

    Args:
        sppf_layer (SPPF): SPPF block to prune.
        prune_ratio (float): Fraction of channels to prune, range 0.0 to 1.0.
        norm_order (float): Order of norm for ranking channel importance (1 for L1, 2 for L2). Default is 2.0.
        mask_prev (torch.Tensor): Boolean mask indicating kept channels from the previous layer. Default is None.
        mask_tracker (dict): Optional dictionary to store intermediate masks for debugging. Default is None.

    Returns:
        mask (torch.Tensor): Boolean mask indicating kept output channels from cv2.
    """
    cv1 = sppf_layer.cv1
    cv2 = sppf_layer.cv2

    cv1_mask = prune_conv(yolo_conv_layer=cv1, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev)
    mask_prev_cv2 = cv1_mask.repeat(4)
    cv2_mask = prune_conv(yolo_conv_layer=cv2, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev_cv2)
    if mask_tracker is not None:
        mask_tracker["cv1_input"] = mask_prev
        mask_tracker["cv1_output"] = cv1_mask
        mask_tracker["cv2_input"] = mask_prev_cv2
        mask_tracker["cv2_output"] = cv2_mask

    return cv2_mask


# ============================================================
# Detect Head & Full Model Pruning
# ============================================================


def prune_detect(
    detect_head: Detect,
    prune_ratio_detect: float,
    prune_ratio_classify: float,
    norm_order: float,
    feature_masks: list,
    prune_type="preserve",
    mask_tracker: dict | None = None,
) -> None:
    """
    Prune the detection and classification towers of a YOLO Detect head.

    The Detect head contains three parallel detection towers (cv2) and three parallel classification
    towers (cv3), one pair for each feature map from the backbone. Classification towers may use either
    standard Conv or DWConv sequences, requiring different pruning strategies.

    Args:
        detect_head (Detect): Detect head module to prune.
        prune_ratio_detect (float): Fraction of channels to prune in detection towers, range 0.0 to 1.0.
        prune_ratio_classify (float): Fraction of channels to prune in classification towers, range 0.0 to 1.0.
        norm_order (float): Order of norm for ranking channel importance (1 for L1, 2 for L2).
        feature_masks (list[torch.Tensor]): Boolean masks for the three input feature maps, one per tower pair.
        prune_type (str): Pruning strategy, either 'preserve' or 'remove'. Default is 'preserve'.
        mask_tracker (dict): Optional dictionary to store intermediate masks for debugging. Default is None.

    Returns:
        None: Modifies detect_head in place.
    """
    cv2 = detect_head.cv2
    cv3 = detect_head.cv3

    for i, detection_tower, feature_mask in zip(range(len(cv2)), cv2, feature_masks):
        # prune first 2 Conv layers
        mask = feature_mask
        for j, conv in enumerate(detection_tower[:2]):
            if mask_tracker is not None:
                mask_tracker[f"detection_tower_{i}_component_{j}_input"] = mask

            mask = prune_conv(conv, prune_ratio_detect, norm_order=norm_order, mask_prev=mask, prune_type=prune_type)

            if mask_tracker is not None:
                mask_tracker[f"detection_tower_{i}_component_{j}_output"] = mask

        if mask_tracker is not None:
            mask_tracker[f"detection_tower_{i}_component_{2}_input"] = mask

        updated_conv, mask = prune_conv2d(
            detection_tower[-1], 0, norm_order=norm_order, mask_prev=mask, prune_type=prune_type
        )
        detection_tower[-1] = updated_conv

        if mask_tracker is not None:
            mask_tracker[f"detection_tower_{i}_component_{2}_output"] = mask

    for i, classification_tower, feature_mask in zip(range(len(cv3)), cv3, feature_masks):
        mask = feature_mask
        first_component = classification_tower[0]

        if isinstance(first_component, Conv):
            for j, conv in enumerate(classification_tower[:2]):
                if mask_tracker is not None:
                    mask_tracker[f"classification_tower_{i}_component_{j}_input"] = mask

                mask = prune_conv(
                    conv, prune_ratio_classify, norm_order=norm_order, mask_prev=mask, prune_type=prune_type
                )

                if mask_tracker is not None:
                    mask_tracker[f"classification_tower_{i}_component_{j}_output"] = mask

        else:
            seq1, seq2 = classification_tower[0], classification_tower[1]
            dw_conv1, conv1, dw_conv2, conv2 = seq1[0], seq1[1], seq2[0], seq2[1]

            if mask_tracker is not None:
                mask_tracker[f"classification_tower_{i}_component_{0}_{0}_input"] = mask
            mask = prune_conv(
                dw_conv1, prune_ratio_classify, norm_order=norm_order, mask_prev=mask, prune_type=prune_type
            )
            if mask_tracker is not None:
                mask_tracker[f"classification_tower_{i}_component_{0}_{0}_output"] = mask
                mask_tracker[f"classification_tower_{i}_component_{0}_{1}_input"] = mask

            grouped_conv_chain = compute_effective_groups([conv1.conv.groups, dw_conv2.conv.groups])
            mask = prune_conv(
                conv1,
                prune_ratio_classify,
                norm_order=norm_order,
                mask_prev=mask,
                prune_groups=grouped_conv_chain,
                prune_type=prune_type,
            )
            if mask_tracker is not None:
                mask_tracker[f"classification_tower_{i}_component_{0}_{1}_output"] = mask
                mask_tracker[f"classification_tower_{i}_component_{1}_{0}_input"] = mask

            mask = prune_conv(
                dw_conv2, prune_ratio_classify, norm_order=norm_order, mask_prev=mask, prune_type=prune_type
            )
            if mask_tracker is not None:
                mask_tracker[f"classification_tower_{i}_component_{1}_{0}_output"] = mask
                mask_tracker[f"classification_tower_{i}_component_{1}_{1}_input"] = mask
            mask = prune_conv(conv2, prune_ratio_classify, norm_order=norm_order, mask_prev=mask, prune_type=prune_type)
            if mask_tracker is not None:
                mask_tracker[f"classification_tower_{i}_component_{1}_{1}_output"] = mask

        if mask_tracker is not None:
            # both are same, but with different key formats that makes testing easier
            mask_tracker[f"classification_tower_{i}_component_{2}_{0}_input"] = mask
            mask_tracker[f"classification_tower_{i}_component_{2}_input"] = mask

        updated_conv, mask = prune_conv2d(
            classification_tower[-1], 0, norm_order=norm_order, mask_prev=mask, prune_type=prune_type
        )
        classification_tower[-1] = updated_conv

        if mask_tracker is not None:
            # both are same, but with different key formats that makes testing easier
            mask_tracker[f"classification_tower_{i}_component_{2}_{0}_output"] = mask  #
            mask_tracker[f"classification_tower_{i}_component_{2}_output"] = mask


def gcd_all(vals) -> int:
    """
    Compute the GCD of a sequence of integers.

    Args:
        vals (Iterable[int]): Sequence of integers.

    Returns:
        int: Greatest common divisor of all input values. Returns 1 if empty.
    """
    vals = list(vals)
    if not vals:
        return 1
    if len(vals) == 1:
        return vals[0]
    return reduce(math.gcd, vals)


def compute_effective_groups(groups: list[int]) -> int:
    """
    Compute the effective number of groups for pruning consecutive grouped convolutions.

    Filters out standard convolutions (groups=1) and returns the GCD of remaining group values
    to determine a common group structure for consistent pruning across layers.

    Args:
        groups (list[int]): List of group values from consecutive convolution layers.

    Returns:
        effective_groups (int): GCD of all group values excluding 1, representing the maximum
            common group divisor for pruning.
    """
    groups = [group for group in groups if group != 1]
    return gcd_all(groups)


def validate_prune_cfg(prune_yaml: str | Path) -> dict:
    """
    Validate and load a YOLOv8 pruning configuration YAML file.

    Checks YAML syntax, structure, and prune ratio values. Ensures all 23 layers (0-22) are specified
    with valid ratios in [0, 1]. Layer 22 (Detect head) requires a list of two ratios [detection, classification].

    Args:
        prune_yaml (str | Path): Path to the pruning configuration YAML file.

    Returns:
        prune_ratios (dict): Dictionary mapping layer indices to prune ratios.

    Raises:
        AssertionError: If YAML structure is invalid or ratios are out of bounds.
        ValueError: If YAML syntax is invalid.
    """
    if isinstance(prune_yaml, str):
        prune_yaml = Path(prune_yaml)

    assert prune_yaml.suffix == ".yaml", f"Expected .yaml file, got {prune_yaml.suffix}"
    assert prune_yaml.exists(), f"Path {prune_yaml} does not exist or is invalid!"

    try:
        with open(prune_yaml) as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {prune_yaml}: {e}")

    assert isinstance(cfg, dict), "YAML root must be a dictionary"
    assert "prune_ratios" in cfg, "Missing key 'prune_ratios'"

    ratios = cfg["prune_ratios"]
    assert isinstance(ratios, dict), "'prune_ratios' must be a dictionary"

    ignore_idx = [10, 11, 13, 14, 17, 20]
    total_layers = 23  # YOLOv8 detect architectures
    detect_head_idx = 22

    # Key validation
    for key in ratios.keys():
        assert isinstance(key, int), f"Invalid key type '{key}': must be int"
        assert 0 <= key < total_layers, f"Invalid key {key}, expected 0–{total_layers - 1}"

    # Value validation
    for idx, val in ratios.items():
        if idx in ignore_idx:
            continue

        if idx == detect_head_idx:
            assert isinstance(val, list), f"Detect head ({idx}) must be a list of two floats"
            assert len(val) == 2, f"Detect head ({idx}) must contain exactly two ratios [reg, cls]"
            for r in val:
                assert isinstance(r, (float, int)), f"Invalid detect ratio type {r} at layer {idx}"
                assert 0 <= r <= 1, f"Detect ratio {r} at layer {idx} is out of range (0–1)"
            continue

        assert isinstance(val, (float, int)), f"Invalid ratio type {type(val)} at layer {idx}"
        assert 0 <= val <= 1, f"Ratio {val} at layer {idx} is out of range (0–1)"

    expected_layers = 23
    assert len(ratios.keys()) == expected_layers, (
        f"Incomplete prune config: expected {expected_layers} entries (0–22), "
        f"found {len(ratios.keys())}. Missing indices may cause KeyError."
    )

    return ratios


def prune_detection_model(
    model: YOLO,
    prune_ratio: float = 0.1,
    norm_order: float = 2,
    prune_type="preserve",
    prune_yaml: str = None,
    mask_tracker: dict | None = None,
) -> YOLO:
    """
    Prune a YOLO detection model by reducing channels across supported components.

    Supports pruning of Conv, BatchNorm, Bottleneck, C2f, SPPF, and Detect layers.
    Concat and Upsample layers are skipped, but Concat masks are updated to maintain consistency.

    Pruning can be global using `prune_ratio`, or per-component using a YAML file (`prune_yaml`).

    Args:
        model (YOLO): YOLO model to prune.
        prune_ratio (float, optional): Global pruning ratio if `prune_yaml` is not provided. Defaults to 0.1.
        norm_order (float, optional): Norm order for filter importance scoring. Defaults to 2.
        prune_type (str, optional): Type of pruning to perform. Defaults to "preserve".
        prune_yaml (str, optional): Path to YAML specifying per-component pruning ratios. Defaults to None.
        mask_tracker (dict, optional): Dictionary to store pruning masks. Defaults to None.

    Returns:
        YOLO: A deep-copied YOLO model with updated pruned weights and masks.
    """
    # Load component-wise pruning ratios if YAML is provided
    if prune_yaml is not None:
        prune_cfg = validate_prune_cfg(prune_yaml)

    model = deepcopy(model)
    detect_head = model.model.model[-1]  # Detect head is always last component

    detect_head_f = detect_head.f  # Indices of featuremaps used by Detect
    save_indices = model.model.save  # Indices of layers whose outputs are saved for later use

    save_indices_masks = {}  # Dict mapping layer index to its pruning mask
    model_components = model.model.model

    mask = None  # output mask
    mask_prev = None

    for component in model_components:
        i, f = component.i, component.f
        mask_tracker_comp = {}

        # Override global prune_ratio with component-specific value if YAML is provided
        if prune_yaml is not None:
            prune_ratio = prune_cfg[i]

        if isinstance(component, Concat):
            # Concatenate masks from previous layers feeding into this Concat
            mask = torch.cat((mask_prev, save_indices_masks[f[1]]))

        elif isinstance(component, C2f):
            # Check if this C2f outputs a featuremap used by Detect
            if i in detect_head_f:
                idx = detect_head_f.index(component.i)
                class_tower = detect_head.cv3[idx]  # classification tower for this C2f feature map
                first_component = class_tower[0]

                # Handle sequence blocks in classification tower (some architectures use Sequential)
                if isinstance(first_component, Sequential):
                    first_conv = first_component[0]
                    grouped_conv_chain = [component.cv2.conv.groups, first_conv.conv.groups]
                    mask = prune_c2f(
                        component,
                        prune_ratio,
                        norm_order=norm_order,
                        mask_prev=mask_prev,
                        prune_groups=compute_effective_groups(grouped_conv_chain),
                        prune_type=prune_type,
                        mask_tracker=mask_tracker_comp,
                    )
                else:
                    mask = prune_c2f(
                        component,
                        prune_ratio,
                        norm_order=norm_order,
                        mask_prev=mask_prev,
                        mask_tracker=mask_tracker_comp,
                    )

            else:
                mask = prune_c2f(
                    component, prune_ratio, norm_order=norm_order, mask_prev=mask_prev, mask_tracker=mask_tracker_comp
                )

        elif isinstance(component, SPPF):
            mask = prune_sppf(
                component, prune_ratio, norm_order=norm_order, mask_prev=mask_prev, mask_tracker=mask_tracker_comp
            )

        elif isinstance(component, Conv):
            mask = prune_conv(component, prune_ratio, norm_order=norm_order, mask_prev=mask_prev)

        elif isinstance(component, Detect):
            # Collect masks for each featuremap fed to Detect
            prev_masks_detect = []

            for _f in detect_head_f:
                prev_masks_detect.append(save_indices_masks[_f])
            mask_prev = prev_masks_detect

            if prune_yaml is not None:
                prune_ratio_detect, prune_ratio_classify = prune_ratio
            else:
                prune_ratio_detect, prune_ratio_classify = prune_ratio, prune_ratio
            prune_detect(
                detect_head,
                prune_ratio_detect=prune_ratio_detect,  # Prune regression towers
                prune_ratio_classify=prune_ratio_classify,  # Prune classification towers
                norm_order=norm_order,
                feature_masks=mask_prev,  # Masks from C2f outputs
                prune_type=prune_type,
                mask_tracker=mask_tracker_comp,
            )

        elif isinstance(component, Upsample):
            # Upsample layers have no weights; skip pruning
            continue

        else:
            print(f"Class: {type(component)} not supported yet for pruning!")

        # Save mask for layers whose outputs are needed downstream
        if i in save_indices:
            save_indices_masks[i] = mask

        if mask_tracker is not None:
            mask_tracker[f"{i}_input"] = mask_prev
            mask_tracker[f"{i}_output"] = mask
            mask_tracker[f"{i}_mask_tracker"] = mask_tracker_comp
        mask_prev = mask

    # Mark model as pruned for checkpoint persistence
    model.model.is_pruned = True

    return model
