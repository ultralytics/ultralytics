"""
prune.py.

This module contains functions to prune YOLOv8 model components
at the channel level, including Conv2d, BatchNorm2d, C2f blocks,
SPPF, and the Detect head. Pruning can be done globally or using
a component-wise YAML configuration.

Supported functionalities:
- Pruning standard Conv2d and YOLO Conv blocks
- Group-aware and skip-aware pruning
- Structured pruning of C2f and Bottleneck blocks
- Pruning of SPPF layers
- Pruning Detect head (regression and classification towers)
- Full model pruning with optional YAML-based per-component ratios

Intended for submission as part of Ultralytics YOLOv8 pruning
pipeline enhancements.
"""

from copy import deepcopy
from typing import Tuple, Union

import torch
import yaml
from torch.nn import BatchNorm2d, Conv2d, Sequential, Upsample

from ultralytics import YOLO
from ultralytics.nn.modules import SPPF, Bottleneck, C2f, Concat, Conv, Detect

# ============================================================
# Conv / YOLO Conv Pruning Functions
# ============================================================


def manual_pruning(
    weight_tensor: torch.Tensor, prune_ratio: float, norm_order: float, dim: Union[int, Tuple[int, ...]]
):
    """
    Prune filters of a weight tensor based on their norms.

    Each filter is scored by its norm (L1, L2, etc.), and the lowest-scoring
    fraction of filters, determined by `prune_ratio`, are removed. Returns
    both the pruned weight tensor and a boolean mask indicating which filters
    were kept.

    Args:
        weight_tensor (torch.Tensor): Weight tensor to prune, e.g., Conv2d weights
            of shape `(out_channels, in_channels, kH, kW)`.
        prune_ratio (float): Fraction of filters to remove (0.0–1.0).
        norm_order (float): Order of the norm used for importance scoring (1=L1, 2=L2, etc.).
        dim (int or Tuple[int, ...]): Dimension(s) over which to compute the norm.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Pruned weight tensor containing only the kept filters.
            - Boolean mask of kept filters (True = kept, False = pruned).

    Notes:
        - If `prune_ratio` results in zero filters being removed, all filters are kept.
        - Used internally by higher-level Conv2d pruning functions.
        - The mask can be passed downstream to maintain consistent pruning of connected layers.
    """
    num_filters = weight_tensor.shape[0]
    k = int(num_filters * prune_ratio)

    if k == 0:
        mask = torch.tensor([True] * num_filters)
    else:
        norm = torch.linalg.vector_norm(weight_tensor, ord=norm_order, dim=dim)
        threshold = torch.quantile(norm, prune_ratio)
        mask = norm > threshold
    return weight_tensor[mask], mask


def prune_conv2d(
    conv_layer: Conv2d, prune_ratio: float, norm_order: float = 2.0, mask_prev: torch.Tensor = None, prune_groups=None
):
    """
    Prune a Conv2d layer with optional group-aware pruning.

    Args:
        conv_layer (nn.Conv2d): PyTorch Conv2d layer to prune.
        prune_ratio (float): Fraction of channels to prune (0.0–1.0).
        norm_order (float): Order of the norm used to rank channel importance.
            For example, order=2 computes L2-norms, order=1 uses L1-norms.
        mask_prev (Tensor, optional):  Mask of active input channels from previous
            layers. For multi-input cases (skip/concat), use the concatenated mask.
        prune_groups (int, optional): Number of groups to split Conv2d weights into
            for independent pruning. Preserves grouped-conv semantics:
              - None (default): use conv_layer.groups, which covers both
                standard convs (groups=1) and Ultralytics' DWConv (groups>1).
              - int: override grouping when pruning must align with an
                architectural context (e.g. a following DWConv or C2f split).

    Returns:
        Tuple[nn.Conv2d, List[torch.Tensor]]:
            - The pruned Conv2d layer.
            - List of binary masks, one per group.t of binary masks (one per group, length = prune_groups).


    Notes:
    - Default (prune_groups=None) handles both standard convs (groups=1)
      and Ultralytics' DWConv (groups = gcd(in_channels, out_channels)).
    - To prune a Conv feeding into a DWConv, set
      `prune_groups = next_dw.conv.groups` (Ultralytics Conv) or
      `prune_groups = next_dw.groups` (raw Conv2d).
    - Inside C2f blocks, set `prune_groups = split_factor`
      to keep channel splits consistent.
    """
    groups = prune_groups if prune_groups is not None else conv_layer.groups
    pruned_weight = conv_layer.weight

    if mask_prev is not None:
        pruned_weight = pruned_weight[:, mask_prev, :]

    chunk_masks = []
    if prune_ratio > 0:
        chunk_list = pruned_weight.chunk(groups, 0)  # expected 'groups'
        pruned_chunks = []

        for chunk in chunk_list:
            chunk_w_pruned, chunk_w_mask = manual_pruning(chunk, prune_ratio, norm_order=norm_order, dim=(1, 2, 3))
            pruned_chunks.append(chunk_w_pruned)
            chunk_masks.append(chunk_w_mask)

        pruned_weight = torch.cat(pruned_chunks)
    else:
        # print("No pruning, only adjusting")
        chunk_masks = torch.tensor([True] * conv_layer.out_channels).chunk(groups, 0)

    # print(weight.shape, bias.shape)
    updated_out_channels = len(
        pruned_weight
    )  # int(torch.sum(pruning_mask_out).numpy())  # non-zero number of out_channels

    updated_in_channels = (
        conv_layer.in_channels if mask_prev is None else int(torch.sum(mask_prev).numpy())
    )  # this too will be masked if the previous layer was masked, but lets do that later

    # print(f"new_out_channels: {updated_out_channels}")
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

    updated_conv.weight = torch.nn.Parameter(pruned_weight)

    if conv_layer.bias is not None:
        bias = conv_layer.bias
        pruning_mask_out = torch.cat(chunk_masks)
        pruned_bias = bias[pruning_mask_out]
        updated_conv.bias = torch.nn.Parameter(pruned_bias)

    return updated_conv, chunk_masks


def prune_conv2d_with_skip(conv_layer: Conv2d, mask_skip: torch.Tensor, mask_prev=None):
    """
    Prune a Conv2d layer based on a skip-producing layer.

    Ensures the layer's output channels match those removed from the skip-producing
    layer, preserving correctness of addition operations in C2f bottleneck blocks
    of the YOLO backbone.

    Args:
        conv_layer (nn.Conv2d): PyTorch Conv2d layer to prune.
        mask_skip (torch.Tensor): Binary mask of output channels to keep, from the skip-producing layer.
        mask_prev (torch.Tensor, optional): Boolean mask for input channels to keep from the previous layer(s).

    Returns:
        Tuple[nn.Conv2d, torch.Tensor]:
            - The pruned Conv2d layer with updated input/output channels and weights.
            - The output channel mask applied (`mask_skip`), for downstream pruning.

    Notes:
        - Bias and weight parameters are pruned consistently with the masks.
        - Does not prune BatchNorm layers; use `prune_batchnorm2d` separately with the mask returned.
        - Input channels can also be pruned if `mask_prev` is provided.
    """
    updated_out_channels = int(torch.sum(mask_skip).numpy())  # non-zero number of out_channels

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
        bias = torch.nn.Parameter(conv_layer.bias[mask_skip])
        updated_conv.bias = bias

    return updated_conv, mask_skip


def prune_conv(
    yolo_conv_layer: Conv, prune_ratio: float, norm_order: float = 2, mask_prev: torch.Tensor = None, prune_groups=None
):
    """
    Prune a YOLO Conv block consisting of Conv2d + BatchNorm.

    Args:
        yolo_conv_layer (Conv): Ultralytics Conv module (Conv2d + BatchNorm).
        prune_ratio (float): Fraction of channels to prune.
        norm_order (float, optional): Order of norm used for importance scoring. Default is 2.
        mask_prev (torch.Tensor, optional): Mask for input channels from previous layers,
            used to prune Conv2d inputs consistently.
        prune_groups (int, optional): Number of groups to split the Conv2d weights into
            for group-wise pruning. If None, defaults to conv_layer.groups.

    Returns:
        List[torch.Tensor]: Pruning masks applied to Conv2d output groups.

    Notes:
        - See `prune_conv2d` for details on group handling and edge cases.
        - Does not handle skip connections; use `prune_conv_with_skip` in that case.
    """
    conv = yolo_conv_layer.conv
    bn = yolo_conv_layer.bn

    pruned_conv, conv_mask = prune_conv2d(
        conv_layer=conv, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev, prune_groups=prune_groups
    )
    pruned_bn, bn_mask = prune_batchnorm2d(bn_layer=bn, mask_prev=torch.cat(conv_mask))

    yolo_conv_layer.conv = pruned_conv
    yolo_conv_layer.bn = pruned_bn

    return conv_mask


def prune_conv_with_skip(yolo_conv_layer: Conv, mask_skip: torch.Tensor = None, mask_prev: torch.Tensor = None):
    """
    Prune a YOLO Conv block (Conv2d + BatchNorm) according to skip-producing layer pruning.

    Output channels are pruned to match the channels removed from the skip-producing
    layer, preserving addition correctness in C2f bottleneck blocks of the YOLO backbone.
    BatchNorm layers are pruned consistently.

    Args:
        yolo_conv_layer (Conv): Ultralytics Conv module (Conv2d + BatchNorm) to prune.
        mask_skip (torch.Tensor, optional): Binary mask of output channels from the skip-producing layer.
        mask_prev (torch.Tensor, optional): Mask for input channels from previous layers.

    Returns:
        torch.Tensor: Binary mask of pruned output channels from this layer's Conv2d layer.

    Notes:
        - For full Conv2d pruning details, see `prune_conv2d_with_skip`.
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


def prune_batchnorm2d(bn_layer: BatchNorm2d, mask_prev: torch.Tensor):
    """
    Prune a BatchNorm2d layer according to a given input channel mask.

    The output channels of the BatchNorm layer are reduced to match the
    active channels specified by `mask_prev`. Running statistics (mean and variance)
    and affine parameters (weight and bias) are pruned accordingly. This is
    typically used after pruning the preceding Conv2d layer to maintain consistency.

    Args:
        bn_layer (BatchNorm2d): PyTorch BatchNorm2d layer to prune.
        mask_prev (torch.Tensor): Boolean mask of input/output channels to keep.

    Returns:
        Tuple[BatchNorm2d, torch.Tensor]:
            - The pruned BatchNorm2d layer with updated channels and parameters.
            - The mask applied (`mask_prev`), for downstream reference.

    Notes:
        - `mask_prev` must be a boolean tensor.
        - Both running statistics (`running_mean` and `running_var`) and affine
          parameters (`weight` and `bias`) are pruned consistently.
        - Use this function after pruning a Conv2d layer to ensure channel alignment.
    """
    assert mask_prev.dtype == torch.bool, "mask must be a boolean tensor"
    is_affine = bn_layer.affine
    updated_out_channels = int(torch.sum(mask_prev).numpy())

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


def prune_bottleneck(bottleneck: Bottleneck, prune_ratio: float, norm_order: float, mask_prev: torch.Tensor | None):
    """
    Prune a Bottleneck block in YOLO.

    Prunes the internal Conv layers (`cv1` and `cv2`) of the Bottleneck. If
    `bottleneck.add` is True, `cv2` is pruned according to the input skip mask
    (`mask_prev`) to preserve the skip connection. Otherwise, `cv2` is
    pruned independently.

    Args:
        bottleneck (Bottleneck): The Bottleneck block to prune.
        prune_ratio (float): Fraction of channels to prune.
        norm_order (float): Order of the norm used to score channel importance.
        mask_prev (torch.Tensor, optional): Boolean mask for input channels from the previous layer(s).
            Must have dtype `torch.bool`.

    Returns:
        torch.Tensor: Boolean mask of pruned output channels from `cv2`.

    Notes:
        - If `bottleneck.add` is True, pruning of `cv2` matches the channels
          removed from the skip-tensor-producing layer to ensure addition remains valid.
        - If `bottleneck.add` is False, `cv2` is pruned independently.
        - Used internally for structured pruning of Bottleneck blocks in YOLO backbones.
    """
    cv1 = bottleneck.cv1
    cv2 = bottleneck.cv2
    add = bottleneck.add

    # if add=True, then from cv2, prune filters using mask of the skip tensor x. In this case, that mask is the same
    # as mask_prev (since x was produced from the conv to which the prev_mask belongs)
    # If add is False, we can prune cv2s filters independently

    [cv1_mask] = prune_conv(yolo_conv_layer=cv1, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev)

    if add:
        cv2_mask = prune_conv_with_skip(yolo_conv_layer=cv2, mask_skip=mask_prev, mask_prev=mask_prev)

    else:
        [cv2_mask] = prune_conv(yolo_conv_layer=cv2, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=cv1_mask)

    # print(torch,all(cv2_mask==mask_prev)) # this should be true when add=True
    return cv2_mask


def prune_c2f(c2f_layer: C2f, prune_ratio: float, norm_order=2, mask_prev=None, to_dwconv_groups=None):
    """
    Prune a C2f block in YOLO.

    The C2f block splits the output of its first Conv layer (`cv1`) into chunks,
    processes them through Bottleneck modules, concatenates all outputs, and
    finally applies a second Conv layer (`cv2`). This function prunes all internal
    Conv layers consistently, propagating masks through the Bottlenecks and
    respecting optional group constraints for `cv2`.

    Args:
        c2f_layer (C2f): The C2f block to prune.
        prune_ratio (float): Fraction of channels to prune.
        norm_order (float, optional): Order of the norm used for importance scoring. Default is 2.
        mask_prev (torch.Tensor, optional): Boolean mask of input channels from previous layer(s).
            Must have dtype `torch.bool`.
        to_dwconv_groups (int, optional): Number of groups to use when pruning `cv2` if it
            will feed a depthwise convolution. Defaults to None.

    Returns:
        torch.Tensor: Boolean mask of pruned output channels from `cv2`.

    Notes:
        - `cv1` pruning is performed in two chunks; the second chunk's mask is propagated
          through all Bottleneck modules in the block.
        - All masks (from `cv1` chunks and Bottlenecks) are concatenated to
          form the mask for `cv2`.
        - Used internally for structured pruning of C2f blocks in YOLO backbones.
        - The returned mask can be propagated downstream to maintain pruning consistency.
    """
    cv1 = c2f_layer.cv1
    cv2 = c2f_layer.cv2
    m = c2f_layer.m

    # cv1 will pruned using chunking and return two masks => updated_conv, (chunk1_w_mask, chunk2_w_mask) The
    # chunk2_w_mask will be passed to all bottleneck modules in m. Each module in m should return their respective
    # pruning masks. all masks [chunk1_w_mask, chunk2_w_mask, m_0_mask, mask_1_mask,...mask_m_mask] should be
    # concatenated to set mask_prev for cv2

    chunk1_w_mask, chunk2_w_mask = prune_conv(
        cv1, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev, prune_groups=2
    )

    m_masks = [chunk1_w_mask, chunk2_w_mask]

    for bottleneck in m:
        bottleneck_mask = prune_bottleneck(
            bottleneck, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=m_masks[-1]
        )
        m_masks.append(bottleneck_mask)

    mask_prev_cv2 = torch.cat(m_masks)

    if to_dwconv_groups:
        mask_cv2 = prune_conv(
            cv2, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev_cv2, prune_groups=to_dwconv_groups
        )
    else:
        mask_cv2 = prune_conv(cv2, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev_cv2)

    return torch.cat(mask_cv2)


def prune_sppf(sppf_layer: SPPF, prune_ratio: float, norm_order: float = 2, mask_prev: torch.Tensor = None):
    """
    Prune an SPPF (Spatial Pyramid Pooling - Fast) block in YOLO.

    Prunes the internal Conv layers and returns the output mask of the final Conv layer,
    ensuring pruning consistency across concatenated pooled features.

    Args:
        sppf_layer (SPPF): The SPPF block to prune.
        prune_ratio (float): Fraction of channels to prune.
        norm_order (float, optional): Order of the norm used to score channel importance. Default is 2.
        mask_prev (torch.Tensor, optional): Boolean mask for input channels from previous layer(s).
         Must have dtype `torch.bool`.

    Returns:
        torch.Tensor: Boolean mask of pruned output channels from the final Conv layer (`cv2`).

    Notes:
        - `cv1_mask` is repeated to align with the concatenated pooled features for `cv2`.
        - Used internally for structured pruning of SPPF blocks in YOLO backbones.
        - The returned mask can be propagated downstream to maintain pruning consistency.
    """
    cv1 = sppf_layer.cv1
    cv2 = sppf_layer.cv2

    cv1_mask = prune_conv(yolo_conv_layer=cv1, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev)
    mask_prev_cv2 = torch.cat(cv1_mask * 4)
    [cv2_mask] = prune_conv(
        yolo_conv_layer=cv2, prune_ratio=prune_ratio, norm_order=norm_order, mask_prev=mask_prev_cv2
    )

    return cv2_mask


# ============================================================
# Detect Head & Full Model Pruning
# ============================================================


def prune_detect(
    detect_head: Detect, prune_ratio_detect: float, prune_ratio_classify: float, norm_order: float, feature_masks: list
):
    """
    Prune the detection and classification towers of a YOLO Detect head.

    The Detect head receives three feature maps produced by preceding C2f layers
    in the head group. These layers fuse multi-scale features from earlier
    stages of the network. Each feature map is fed to both detection (`cv2`) and
    classification (`cv3`) towers, with one tower per feature map. This function
    applies structured pruning to the internal Conv layers of each tower,
    propagating masks through all layers consistently.

    Args:
        detect_head (Detect): The Detect head module to prune.
        prune_ratio_detect (float): Fraction of filters to prune in the detection towers.
        prune_ratio_classify (float): Fraction of filters to prune in the classification towers.
        norm_order (float): Norm order used to score channel importance during pruning.
        feature_masks (list of torch.Tensor): Boolean masks of input channels for each
          of the three feature maps. Each mask must have dtype `torch.bool`.

    Returns:
      None: The function updates `detect_head` in place.

    Notes:
        - Detection towers (`cv2`) are pruned sequentially for each feature map:
          1. The first two Conv layers of each tower are pruned using `prune_ratio_detect`.
          2. The last Conv2d layer is updated using `prune_conv2d` with the mask propagated from previous layers.
        - Classification towers (`cv3`) are pruned differently depending on structure:
            - If the first component is a standard Conv, the first two Conv layers are pruned using `prune_ratio_detect`.
            - If the first component is a DWConv sequence, each Conv layer is pruned in order using `prune_ratio_classify`,
            with proper group alignment for depthwise convolutions.
        - Masks are propagated consistently to maintain alignment with the input feature maps.
        - All pruning is performed **in-place** on the Detect head.
    """
    cv2 = detect_head.cv2
    cv3 = detect_head.cv3

    for detection_tower, feature_mask in zip(cv2, feature_masks):
        # prune first 2 Conv layers
        mask = feature_mask
        for conv in detection_tower[:2]:
            [mask] = prune_conv(conv, prune_ratio_detect, norm_order=norm_order, mask_prev=mask)
        # adjust the input channels of the last Conv2D layer
        # mask = prune_conv(detection_tower[-1], 0, n=n, mask_prev=mask, groups=1)
        updated_conv, masks = prune_conv2d(detection_tower[-1], 0, norm_order=norm_order, mask_prev=mask)
        detection_tower[-1] = updated_conv

    for classification_tower, feature_mask in zip(cv3, feature_masks):
        mask = feature_mask
        first_component = classification_tower[0]

        if isinstance(first_component, Conv):
            for conv in classification_tower[:2]:
                [mask] = prune_conv(conv, prune_ratio_classify, norm_order=norm_order, mask_prev=mask)

        else:
            seq1, seq2 = classification_tower[0], classification_tower[1]
            dw_conv1, conv1, dw_conv2, conv2 = seq1[0], seq1[1], seq2[0], seq2[1]

            [mask] = prune_conv(dw_conv1, prune_ratio_classify, norm_order=norm_order, mask_prev=mask)
            [mask] = prune_conv(
                conv1, prune_ratio_classify, norm_order=norm_order, mask_prev=mask, prune_groups=dw_conv2.conv.groups
            )
            [mask] = prune_conv(dw_conv2, prune_ratio_classify, norm_order=norm_order, mask_prev=mask)
            [mask] = prune_conv(conv2, prune_ratio_classify, norm_order=norm_order, mask_prev=mask)

        updated_conv, masks = prune_conv2d(classification_tower[-1], 0, norm_order=norm_order, mask_prev=mask)
        classification_tower[-1] = updated_conv


def prune_detection_model(model: YOLO, prune_ratio: float = 0.1, norm_order: float = 2, prune_yaml: str = None):
    """
    Prune a YOLO detection model by reducing channels across supported components.

    This function supports pruning of:
      - Conv layers
      - BatchNorm layers (through Conv/Bottleneck wrappers)
      - C2f blocks
      - SPPF blocks
      - Detect head (both classification and regression towers)
      - Concat and Upsample layers are skipped (Concat is handled to maintain mask consistency)

    Pruning can be controlled either globally (same ratio for all components) using `prune_ratio`,
    or component-wise using a YAML file (`prune_yaml`) that specifies pruning ratios per component index.

    Args:
        model (YOLO): The YOLO model to prune.
        prune_ratio (float, optional): Global pruning ratio for all components if `prune_yaml` is not provided.
            Default is 0.1.
        norm_order (float, optional): Norm order used for importance scoring when pruning filters. Default is 2.
        prune_yaml (str, optional): Path to a YAML file specifying per-component pruning ratios.
        The YAML must include a ratio (float) or tuple/list of ratios for every component that will be pruned,
        keyed by the component's `i` index. If `prune_yaml` is provided, these values are used for all components,
        and a KeyError will be raised if any component is missing.
        See `sample_prune.yaml`(ultralytics/cfg/models/v8/sample_prune.yaml) for proper formatting and examples.

    Returns:
        YOLO: A deep-copied and pruned YOLO model with updated masks applied to supported layers.

    Notes:
        - Component indices (`i`) from the model are used as keys in `prune_yaml`.
        - For Detect head, the pruning ratio should be a tuple/list: `(prune_ratio_detect, prune_ratio_classify)`.
        - Concat layers do not have weights; their inclusion in YAML is optional and ignored during pruning.
        - The model's checkpoint (`model.ckpt`) is updated with `"is_pruned": True` for persistence.
        - Only detection-type YOLO architectures are currently supported.
    """
    # Load component-wise pruning ratios if YAML is provided
    if prune_yaml is not None:
        with open(prune_yaml) as f:
            prune_cfg = yaml.safe_load(f)["prune_ratios"]

    model = deepcopy(model)
    detect_head = model.model.model[-1]  # Detect head is always last component

    detect_head_f = detect_head.f  # Indices of featuremaps used by Detect
    save_indices = model.model.save  # Indices of layers whose outputs are saved for later use

    save_indices_masks = {}  # Dict mapping layer index to its pruning mask
    mask = None  # Input mask
    model_components = model.model.model

    for component in model_components:
        i, f = component.i, component.f

        # Override global prune_ratio with component-specific value if YAML is provided
        if prune_yaml is not None:
            prune_ratio = prune_cfg[i]

        if isinstance(component, Concat):
            # Concatenate masks from previous layers feeding into this Concat
            mask = torch.cat((mask, save_indices_masks[f[1]]))

        elif isinstance(component, C2f):
            # Check if this C2f outputs a featuremap used by Detect
            if i in detect_head_f:
                idx = detect_head_f.index(component.i)
                class_tower = detect_head.cv3[idx]  # classification tower for this C2f feature map
                first_component = class_tower[0]

                # Handle sequence blocks in classification tower (some architectures use Sequential)
                if isinstance(first_component, Sequential):
                    first_conv = first_component[0]
                    mask = prune_c2f(
                        component,
                        prune_ratio,
                        norm_order=norm_order,
                        mask_prev=mask,
                        to_dwconv_groups=first_conv.conv.groups,
                    )
                else:
                    mask = prune_c2f(component, prune_ratio, norm_order=norm_order, mask_prev=mask)

            else:
                mask = prune_c2f(component, prune_ratio, norm_order=norm_order, mask_prev=mask)

        elif isinstance(component, SPPF):
            mask = prune_sppf(component, prune_ratio, norm_order=norm_order, mask_prev=mask)

        elif isinstance(component, Conv):
            mask = prune_conv(component, prune_ratio, norm_order=norm_order, mask_prev=mask)
            mask = torch.cat(mask)

        elif isinstance(component, Detect):
            # Collect masks for each featuremap fed to Detect
            prev_masks_detect = []
            for _f in detect_head_f:
                prev_masks_detect.append(save_indices_masks[_f])

            if prune_yaml is not None:
                prune_ratio_detect, prune_ratio_classify = prune_ratio
            else:
                prune_ratio_detect, prune_ratio_classify = prune_ratio, prune_ratio
            prune_detect(
                detect_head,
                prune_ratio_detect=prune_ratio_detect,  # Prune regression towers
                prune_ratio_classify=prune_ratio_classify,  # Prune classification towers
                norm_order=norm_order,
                feature_masks=prev_masks_detect,  # Masks from C2f outputs
            )

        elif isinstance(component, Upsample):
            # Upsample layers have no weights; skip pruning
            continue

        else:
            print(f"Class: {type(component)} not supported yet for pruning!")

        # Save mask for layers whose outputs are needed downstream
        if i in save_indices:
            save_indices_masks[i] = mask

    # Mark model as pruned for checkpoint persistence
    model.ckpt["is_pruned"] = True

    return model
