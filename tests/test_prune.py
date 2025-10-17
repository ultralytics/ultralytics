import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv2d, Upsample

from ultralytics import YOLO
from ultralytics.nn.modules import SPPF, Bottleneck, C2f, Concat, Conv, Detect
from ultralytics.utils.prune import (
    apply_prev_mask,
    compute_effective_groups,
    gcd_all,
    prune,
    prune_batchnorm2d,
    prune_bottleneck,
    prune_by_groups,
    prune_c2f,
    prune_channels_groupwise,
    prune_conv,
    prune_conv2d,
    prune_conv2d_with_skip,
    prune_conv_with_skip,
    prune_detect,
    prune_detection_model,
    prune_sppf,
    validate_prune_cfg,
)

ROOT = Path(__file__).resolve().parents[1]  # repo root
IMG = ROOT / "ultralytics/assets/bus.jpg"
CFG = ROOT / "ultralytics/cfg/pruning/sample_prune.yaml"


# ==========================================
# Helper functions for comparison and checks
# ==========================================


def compare_conv2d_layers(pruned_conv2d: Conv2d, original_conv2d, mask_out, mask_in=None, prune_type="preserve"):
    """Assert that applying input/output masks to the original Conv2d reproduces the pruned Conv2d."""
    pruned_weight = pruned_conv2d.weight
    weight = original_conv2d.weight

    original_groups = original_conv2d.groups
    out_channels_per_group = original_conv2d.out_channels / original_groups

    tmp = weight
    if mask_in is not None:
        tmp, mask, new_groups = apply_prev_mask(tmp, mask_in, original_conv2d.groups)

    # Currently, further pruning is not allowed
    if original_conv2d.out_channels == tmp.shape[0]:
        tmp = tmp[mask_out]

    assert torch.all(pruned_weight == tmp)

    if pruned_conv2d.bias is not None:
        pruned_bias = pruned_conv2d.bias
        bias = original_conv2d.bias
        assert torch.all(pruned_bias == bias[mask_out])

    if prune_type == "preserve":
        assert pruned_conv2d.groups == original_conv2d.groups

    if prune_type == "remove":
        if original_groups > 1:
            new_groups = torch.sum(mask_out) / out_channels_per_group
            assert pruned_conv2d.groups == new_groups  # kept groups are equal
            assert (original_groups - pruned_conv2d.groups) == torch.sum(
                ~mask_out
            ) / out_channels_per_group  # removed groups are equal
        else:
            # this regular conv, no groupings
            new_groups = 1
            assert pruned_conv2d.groups == new_groups


def compare_batchnorm_layers(pruned_bn: BatchNorm2d, original_bn: BatchNorm2d, mask_out):
    """Assert that applying output masks to the original BatchNorm2d reproduces the pruned BatchNorm2d."""
    pruned_running_mean = pruned_bn.running_mean
    pruned_running_var = pruned_bn.running_var
    running_mean = original_bn.running_mean
    running_var = original_bn.running_var

    assert torch.all(pruned_running_mean == running_mean[mask_out])
    assert torch.all(pruned_running_var == running_var[mask_out])

    is_affine = pruned_bn.affine
    if is_affine:
        pruned_weight = pruned_bn.weight
        pruned_bias = pruned_bn.bias
        weight = original_bn.weight
        bias = original_bn.bias

        assert torch.equal(pruned_weight, weight[mask_out])
        assert torch.equal(pruned_bias, bias[mask_out])


def compare_conv_layers(pruned_conv: Conv, original_conv: Conv, mask_out, mask_in=None, prune_type="preserve"):
    """Assert that applying masks to Conv and BatchNorm in the original Conv reproduces the pruned module."""
    pruned_conv2d = pruned_conv.conv
    pruned_bn = pruned_conv.bn
    conv2d = original_conv.conv
    bn = original_conv.bn

    compare_conv2d_layers(
        pruned_conv2d=pruned_conv2d, original_conv2d=conv2d, mask_out=mask_out, mask_in=mask_in, prune_type=prune_type
    )
    compare_batchnorm_layers(pruned_bn=pruned_bn, original_bn=bn, mask_out=mask_out)


def compare_bottleneck_layers(pruned_bottleneck: Bottleneck, original_bottleneck, mask_tracker):
    """Assert that mask-propagated Conv layers in a Bottleneck match their pruned counterparts."""
    pruned_cv1 = pruned_bottleneck.cv1
    pruned_cv2 = pruned_bottleneck.cv2
    cv1 = original_bottleneck.cv1
    cv2 = original_bottleneck.cv2

    compare_conv_layers(
        pruned_conv=pruned_cv1,
        original_conv=cv1,
        mask_out=mask_tracker["cv1_output"],
        mask_in=mask_tracker["cv1_input"],
    )
    compare_conv_layers(
        pruned_conv=pruned_cv2,
        original_conv=cv2,
        mask_out=mask_tracker["cv2_output"],
        mask_in=mask_tracker["cv2_input"],
    )


def check_groupwise_prune_correctness(weight_tensor, pruned_weight, mask, prune_groups):
    """Helper to verify groupwise pruning matches manual reconstruction."""
    w_chunks = weight_tensor.chunk(prune_groups, 0)
    m_chunks = mask.chunk(prune_groups, 0)
    reconstructed = torch.cat([c[m] for c, m in zip(w_chunks, m_chunks)])
    assert torch.equal(pruned_weight, reconstructed)


def check_divisibility(conv):
    """Helper to assert that Conv2d in/out channels remain divisible by groups."""
    assert conv.out_channels % conv.groups == 0
    assert conv.in_channels % conv.groups == 0


# ==========================================================
#  UNIT TESTS — PRIMITIVE & CONV2D-LEVEL PRUNING
# ==========================================================


@pytest.mark.parametrize(
    "out_channels,prune_ratio",
    [
        (8, 0.25),
        (8, 0.5),
        (16, 0.75),
        (16, 0.0),
        (16, 1.0),
    ],
)
def test_prune(out_channels, prune_ratio):
    """Test that pruning correctly selects and masks channels by norm magnitude."""
    torch.manual_seed(123)
    w = torch.randn(out_channels, 4, 3, 3)  # (out_channels, in_channels, kH, kW)
    norm_order = 2
    dim = (1, 2, 3)
    _, mask = prune(w, prune_ratio=prune_ratio, norm_order=norm_order, dim=dim)

    # at least one item kept
    assert mask.sum().item() >= 1

    pruned = (~mask).sum().item()
    expected = int(round(prune_ratio * len(mask)))

    # number of pruned channels and expected pruned channels is within 1 channel
    assert abs(pruned - expected) <= 1

    # Mask properties
    assert mask.dtype == torch.bool
    assert mask.shape[0] == w.shape[0]
    assert pruned + mask.sum().item() == len(mask)

    # check that selected channels have greater norm than the unselected ones
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "vector_norm"):
        norm = torch.linalg.vector_norm(w, ord=norm_order, dim=dim)
    else:
        norm = torch.norm(w, p=norm_order, dim=dim)

    selected_norms = norm[mask]
    unselected_norms = norm[~mask]

    # property: kept channels have >= norm than pruned ones
    if selected_norms.numel() > 0 and unselected_norms.numel() > 0:
        assert selected_norms.min().item() >= unselected_norms.max().item()


def test_prune_channels_groupwise():
    """Test that groupwise pruning applies consistent masks across all groups."""
    groups = 8
    prune_ratio = 0.25

    conv = Conv2d(128, 256, 3, groups=groups)

    weight = deepcopy(conv.weight)
    pruned_weight, mask, kept_out_channels = prune_channels_groupwise(
        weight, prune_ratio, norm_order=2, prune_groups=groups
    )

    # divide mask into 'groups' and check each group has the same number of pruned channels
    chunks = mask.chunk(groups)
    pruned_channels = chunks[0].sum().item()
    for chunk in chunks[1:]:
        assert pruned_channels == chunk.sum().item()

    # The returned correctly reflects pruned filters/channels
    assert torch.equal(pruned_weight, weight[mask])


def test_prune_by_groups():
    """Test that grouped pruning prunes complete groups and preserves divisibility."""
    groups = 8
    prune_ratio = 0.25

    conv = Conv2d(128, 256, 3, groups=groups)

    channels_per_group = conv.out_channels / groups
    weight = deepcopy(conv.weight)
    pruned_weight, mask, kept_out_channels = prune_by_groups(weight, prune_ratio, norm_order=2, prune_groups=groups)

    # Verify each group is either entirely True or entirely False
    for chunk in mask.chunk(groups):
        assert torch.all(chunk) or torch.all(~chunk), "Group must be homogeneous"

    # kept out_channels/filters are still divisible by the channels_per_group
    assert kept_out_channels % channels_per_group == 0

    # The returned correctly reflects pruned filters/channels
    assert torch.equal(pruned_weight, weight[mask])


def test_prune_by_groups_invalid_shape():
    """Test that grouped pruning raises an error when weight shape is not divisible by groups."""
    w = torch.randn(30, 16, 3, 3)
    with pytest.raises(AssertionError, match="not divisible by groups"):
        prune_by_groups(w, prune_ratio=0.5, norm_order=2, prune_groups=8)


@pytest.mark.parametrize("prune_ratio", [0.0, 1.0])
def test_prune_by_groups_edge_ratios(prune_ratio):
    """Test that grouped pruning behaves correctly at edge ratios 0.0 and 1.0."""
    groups = 4
    conv = Conv2d(64, 128, 3, groups=groups)
    weight = conv.weight.clone()

    pruned_weight, mask, kept_out_channels = prune_by_groups(weight, prune_ratio, norm_order=2, prune_groups=groups)

    if prune_ratio == 0.0:
        # No pruning → everything kept
        assert torch.all(mask)
        assert kept_out_channels == conv.out_channels
    else:  # prune_ratio == 1.0
        # Should still keep at least one group
        channels_per_group = conv.out_channels // groups
        assert kept_out_channels == channels_per_group
        assert mask.sum().item() == channels_per_group


def test_prune_conv2d_with_skip():
    """Verifies prune_conv2d_with_skip correctly applies skip-connection mask and preserves structure."""
    conv2d = Conv2d(32, 32, 3)
    mask_prev = torch.tensor([0, 1, 1, 0], dtype=torch.bool).repeat(8)
    mask_skip = torch.tensor([1, 0], dtype=torch.bool).repeat(16)
    pruned_conv2d, mask_out = prune_conv2d_with_skip(conv2d, mask_skip=mask_skip, mask_prev=mask_prev)
    assert torch.all(mask_out == mask_skip)
    compare_conv2d_layers(pruned_conv2d=pruned_conv2d, original_conv2d=conv2d, mask_out=mask_out, mask_in=mask_prev)


def test_grouped_conv2d_remove():
    """Test that grouped Conv2d with 'remove' strategy prunes whole groups and reduces group count."""
    groups = 8
    conv2d = Conv2d(32, 64, 3, groups=groups)
    pruned_conv2d, mask = prune_conv2d(conv_layer=conv2d, prune_ratio=0.25, prune_groups=groups, prune_type="remove")

    # pytorch's groups and channels constraint for this layer is still valid
    check_divisibility(pruned_conv2d)

    # pytorch's groups and channels constraint for the next layer's input is still valid (this layers out is next's in)
    assert pruned_conv2d.out_channels % groups == 0

    # output channels reduced (or same if prune=0)
    assert pruned_conv2d.out_channels <= conv2d.out_channels
    assert pruned_conv2d.out_channels == mask.sum().item()

    # check pruned_conv2d resulted from applying mask_out and mask_in to the original_conv2d
    compare_conv2d_layers(
        pruned_conv2d=pruned_conv2d, original_conv2d=conv2d, mask_out=mask, mask_in=None, prune_type="remove"
    )

    # Verify each group is either entirely True or entirely False
    for chunk in mask.chunk(groups):
        assert torch.all(chunk) or torch.all(~chunk), "Group must be homogeneous"


def test_grouped_conv2d_preserve():
    """Test that grouped Conv2d with 'preserve' pruning prunes within groups without changing group count."""
    groups = 8
    conv2d = Conv2d(32, 64, 3, groups=groups)
    pruned_conv2d, mask = prune_conv2d(conv_layer=conv2d, prune_ratio=0.25, prune_type="preserve")

    # groups stays 1
    assert pruned_conv2d.groups == groups

    # pytorch's groups and channels constraint is still valid
    check_divisibility(pruned_conv2d)

    # output channels reduced (or same if prune=0)
    assert pruned_conv2d.out_channels <= conv2d.out_channels
    assert pruned_conv2d.out_channels == mask.sum().item()

    # check pruned_conv2d resulted from applying mask_out and mask_in to the original_conv2d
    compare_conv2d_layers(
        pruned_conv2d=pruned_conv2d, original_conv2d=conv2d, mask_out=mask, mask_in=None, prune_type="preserve"
    )

    # check the channels were pruned groupwise (divided into 'prune_groups' then independently pruned
    check_groupwise_prune_correctness(
        weight_tensor=conv2d.weight, pruned_weight=pruned_conv2d.weight, mask=mask, prune_groups=groups
    )

    # Each division has the same number of channels kept. Also, the groups preserved!
    mask_chunks = mask.chunk(groups)
    kept_channels = torch.sum(mask_chunks[0]).item()
    for mask_chunk in mask_chunks:
        assert kept_channels == torch.sum(mask_chunk).item()


@pytest.mark.parametrize("prune_type", [("preserve"), ("remove")])
def test_receiver_mask_preserves_groups_further_pruning(prune_type):
    """Test that when the input mask preserves group structure, further pruning is correctly applied."""
    in_channels = 64
    out_channels = 128
    groups = 8

    # set up a mask that simulates group-wise pruning of the previous layer
    in_channels_per_group = in_channels // groups
    prune_ratio_prev = 0.25
    kept_in = int(in_channels_per_group * (1 - prune_ratio_prev))
    remove_in = in_channels_per_group - kept_in
    mask_prev = torch.cat((torch.ones(kept_in, dtype=torch.bool), torch.zeros(remove_in, dtype=torch.bool)))
    mask_prev = mask_prev.repeat(groups)

    conv2d_grouped = Conv2d(in_channels, out_channels, 3, groups=groups)

    prune_ratio = 0.4
    pruned_conv2d, mask = prune_conv2d(
        conv2d_grouped, prune_ratio=prune_ratio, mask_prev=mask_prev, prune_type=prune_type
    )

    if prune_type == "preserve":
        assert pruned_conv2d.groups == groups
    if prune_type == "remove":
        assert pruned_conv2d.groups == max(1, groups - int(groups * prune_ratio))

    compare_conv2d_layers(
        pruned_conv2d=pruned_conv2d,
        original_conv2d=conv2d_grouped,
        mask_out=mask,
        mask_in=mask_prev,
        prune_type=prune_type,
    )


def test_receiver_mask_removes_groups_no_further_pruning():
    """Test that when input mask removes groups, no additional pruning is applied."""
    groups = 8
    remove_groups = 6
    left_groups = groups - remove_groups
    in_channels = 32
    in_channels_per_group = in_channels // groups
    prune_ratio = 0.25

    # lets remove 6 groups worth of in channels which is more than our pruning ratio
    mask_prev = torch.cat((torch.zeros(remove_groups, dtype=torch.bool), torch.ones(left_groups, dtype=torch.bool)))
    mask_prev = mask_prev.repeat_interleave(in_channels_per_group, dim=0)
    conv2d = Conv2d(32, 64, 3, groups=groups)

    pruned_conv2d_no_mask, mask_no_mask = prune_conv2d(conv_layer=conv2d, prune_ratio=prune_ratio, prune_type="remove")
    pruned_conv2d, mask = prune_conv2d(
        conv_layer=conv2d, prune_ratio=prune_ratio, mask_prev=mask_prev, prune_type="remove"
    )

    assert pruned_conv2d.groups == left_groups

    # When mask_prev that removes groups is provided, since we have removed 6 of 8 groups,
    # and as 6/8>0.25, the mask one should have lesser groups left
    assert pruned_conv2d.groups < pruned_conv2d_no_mask.groups

    # pytorch's groups and channels constraint for this layer is still valid
    check_divisibility(pruned_conv2d)

    # pytorch's groups and channels constraint for the next layer's input is still valid (this layers out is next's in)
    assert pruned_conv2d.out_channels % groups == 0

    # output channels reduced (or same if prune=0)
    assert pruned_conv2d.out_channels <= conv2d.out_channels
    assert pruned_conv2d.out_channels == mask.sum().item()

    # check pruned_conv2d resulted from applying mask_out and mask_in to the original_conv2d
    compare_conv2d_layers(
        pruned_conv2d=pruned_conv2d, original_conv2d=conv2d, mask_out=mask, mask_in=mask_prev, prune_type="remove"
    )

    # Verify each group is either entirely True or entirely False
    for chunk in mask.chunk(groups):
        assert torch.all(chunk) or torch.all(~chunk), "Group must be homogeneous"


def test_standard_conv2d_no_mask_channelwise_pruning():
    """Test that standard Conv2d pruning applies channel-wise and preserves groups."""
    conv2d = Conv2d(32, 64, 3)
    pruned_conv2d, mask = prune_conv2d(conv_layer=conv2d, prune_ratio=0.25)

    # groups stays 1
    assert pruned_conv2d.groups == 1

    # output channels reduced (or same if prune=0)
    assert pruned_conv2d.out_channels <= conv2d.out_channels
    assert pruned_conv2d.out_channels == mask.sum().item()
    compare_conv2d_layers(
        pruned_conv2d=pruned_conv2d, original_conv2d=conv2d, mask_out=mask, mask_in=None, prune_type="preserve"
    )

    # If there's no 'prune_groups' override,then the standard conv is pruned channels wise regardless of prune_type
    pruned_conv_2, mask_2 = prune_conv2d(conv_layer=conv2d, prune_ratio=0.25, prune_type="preserve")
    pruned_conv_3, mask_3 = prune_conv2d(conv_layer=conv2d, prune_ratio=0.25, prune_type="remove")

    assert torch.all(mask_2 == mask_3)
    assert torch.all(pruned_conv_2.weight == pruned_conv_3.weight)
    assert pruned_conv_2.groups == pruned_conv_3.groups


def test_standard_conv2d_precedes_grouped_conv2d_preserve():
    """Test that standard Conv2d preceding a grouped Conv preserves group structure under 'preserve' strategy."""
    prune_groups = 8
    conv2d = Conv2d(32, 64, 3)
    pruned_conv2d, mask = prune_conv2d(conv_layer=conv2d, prune_ratio=0.25, prune_groups=prune_groups)

    # groups stays 1
    assert pruned_conv2d.groups == 1

    # pytorch's groups and channels constraint is still valid
    check_divisibility(pruned_conv2d)

    # output channels reduced (or same if prune=0)
    assert pruned_conv2d.out_channels <= conv2d.out_channels
    assert pruned_conv2d.out_channels == mask.sum().item()

    # check pruned_conv2d resulted from applying mask_out and mask_in to the original_conv2d
    compare_conv2d_layers(pruned_conv2d=pruned_conv2d, original_conv2d=conv2d, mask_out=mask, mask_in=None)

    # check the channels were pruned groupwise (divided into 'prune_groups' then independently pruned
    check_groupwise_prune_correctness(
        weight_tensor=conv2d.weight, pruned_weight=pruned_conv2d.weight, mask=mask, prune_groups=prune_groups
    )

    # Each division has the same number of channels kept. Also, the groups preserved!
    mask_chunks = mask.chunk(prune_groups)
    kept_channels = torch.sum(mask_chunks[0]).item()
    for mask_chunk in mask_chunks:
        assert kept_channels == torch.sum(mask_chunk).item()


def test_standard_conv2d_precedes_grouped_conv2d_remove():
    """Test that standard Conv2d preceding a grouped Conv removes full groups under 'remove' strategy."""
    prune_groups = 8
    conv2d = Conv2d(32, 64, 3)
    pruned_conv2d, mask = prune_conv2d(
        conv_layer=conv2d, prune_ratio=0.25, prune_groups=prune_groups, prune_type="remove"
    )

    # groups stays 1
    assert pruned_conv2d.groups == 1

    # pytorch's groups and channels constraint for this layer is still valid
    check_divisibility(pruned_conv2d)

    # pytorch's groups and channels constraint for the next layer's input is still valid (this layers out is next's in)
    assert pruned_conv2d.out_channels % prune_groups == 0

    # output channels reduced (or same if prune=0)
    assert pruned_conv2d.out_channels <= conv2d.out_channels
    assert pruned_conv2d.out_channels == mask.sum().item()

    # check pruned_conv2d resulted from applying mask_out and mask_in to the original_conv2d
    compare_conv2d_layers(pruned_conv2d=pruned_conv2d, original_conv2d=conv2d, mask_out=mask, mask_in=None)

    # Verify each group is either entirely True or entirely False
    for chunk in mask.chunk(prune_groups):
        assert torch.all(chunk) or torch.all(~chunk), "Group must be homogeneous"


def test_chain_two_standard_conv2ds_propagation():
    """Test that pruning masks propagate correctly through a chain of standard Conv2d layers."""
    conv2d_1 = Conv2d(8, 16, 3)
    conv2d_2 = Conv2d(16, 32, 3)

    # prune first layer
    pruned_conv1, mask1 = prune_conv2d(conv2d_1, prune_ratio=0.5)
    mask_prev = mask1

    # prune second layer using mask_prev
    pruned_conv2, _ = prune_conv2d(conv2d_2, prune_ratio=0.5, mask_prev=mask_prev)

    # input channels of conv2 must match sum(mask_prev)
    assert pruned_conv2.in_channels == mask_prev.sum().item()


@pytest.mark.parametrize(
    "prune_type, in_channels, out_channels",
    [
        ("preserve", 16, 32),
        ("remove", 16, 32),
        ("preserve", 64, 64),
        ("remove", 128, 128),
    ],
)
def test_standard_to_grouped_conv2d_chain(prune_type, in_channels, out_channels):
    """Test that pruning propagates correctly from a standard Conv2d to a grouped Conv2d."""
    prune_groups = math.gcd(in_channels, out_channels)
    conv2d = Conv2d(8, in_channels, 3)
    conv2d_grouped = Conv2d(in_channels, out_channels, 3, groups=prune_groups)

    # prune first layer
    conv2d_pruned, mask = prune_conv2d(conv2d, prune_ratio=0.5, prune_groups=prune_groups, prune_type=prune_type)
    mask_prev = mask

    # prune second layer using mask_prev
    conv2d_grouped_pruned, _ = prune_conv2d(conv2d_grouped, prune_ratio=0.5, mask_prev=mask_prev, prune_type=prune_type)

    # --- structural consistency ---
    assert conv2d_grouped_pruned.in_channels == mask_prev.sum().item()

    # --- group integrity (preserve mode only) ---
    if prune_type == "preserve":
        assert conv2d_grouped_pruned.in_channels % prune_groups == 0
        assert conv2d_grouped_pruned.out_channels % prune_groups == 0


@pytest.mark.parametrize(
    "prune_type, in_channels_l1, out_channels_l1, out_channels_l2, g1, g2",
    [
        ("preserve", 32, 64, 64, 16, 8),
        ("remove", 32, 128, 128, 16, 8),
        ("remove", 64, 128, 256, 8, 16),
    ],
)
def test_grouped_to_grouped_conv2d_chain(prune_type, in_channels_l1, out_channels_l1, out_channels_l2, g1, g2):
    """Two sequential grouped Conv2d layers — validates pruning mask propagation and group integrity."""
    conv2d_grouped_1 = Conv2d(in_channels_l1, out_channels_l1, 3, groups=g1)
    conv2d_grouped_2 = Conv2d(out_channels_l1, out_channels_l2, 3, groups=g2)

    # prune first layer
    prune_groups = compute_effective_groups([g1, g2])
    conv2d_grouped_1_pruned, mask = prune_conv2d(
        conv2d_grouped_1, prune_ratio=0.5, prune_groups=prune_groups, prune_type=prune_type
    )
    mask_prev = mask

    # prune second layer using mask_prev
    conv2d_grouped_2_pruned, _ = prune_conv2d(
        conv2d_grouped_2, prune_ratio=0.5, mask_prev=mask_prev, prune_type=prune_type
    )

    # --- structural consistency ---
    assert conv2d_grouped_2_pruned.in_channels == mask_prev.sum().item()

    # --- group integrity (always) ---
    assert conv2d_grouped_1_pruned.in_channels % conv2d_grouped_1_pruned.groups == 0
    assert conv2d_grouped_2_pruned.in_channels % conv2d_grouped_2_pruned.groups == 0

    # --- group preservation (preserve mode only) ---
    if prune_type == "preserve":
        assert conv2d_grouped_1_pruned.groups == g1
        assert conv2d_grouped_2_pruned.groups == g2


def test_invalid_prune_type_raises_value_error():
    """Invalid prune_type argument should raise a ValueError."""
    prune_groups = 8
    conv2d = Conv2d(32, 64, 3)

    with pytest.raises(ValueError, match="Invalid prune_type value movie"):
        pruned_conv2d, mask = prune_conv2d(
            conv_layer=conv2d, prune_ratio=0.25, prune_groups=prune_groups, prune_type="movie"
        )


@pytest.mark.parametrize("prune_type", ["preserve", "remove"])
def test_bias_pruning_propagates_mask_correctly(prune_type):
    """Ensures Conv2d bias parameters are pruned consistently with output channel masks."""
    conv2d = Conv2d(32, 64, 3)
    pruned_conv2d, mask = prune_conv2d(conv2d, 0.25, prune_type=prune_type)

    assert pruned_conv2d.weight.shape[0] == pruned_conv2d.bias.shape[0]


def test_prune_batchnorm2d():
    """Test that prune_batchnorm2d correctly applies channel mask to BatchNorm2d parameters."""
    bn = nn.BatchNorm2d(8, affine=True)
    original_bn = deepcopy(bn)
    mask_out = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1], dtype=torch.bool)

    pruned_bn, bn_mask = prune_batchnorm2d(bn, mask_out)

    assert torch.equal(bn_mask, mask_out)
    compare_batchnorm_layers(pruned_bn=pruned_bn, original_bn=original_bn, mask_out=mask_out)


# ==========================================================
#  MASK PROPAGATION TESTS (Sequential & Grouped Layers)
# ==========================================================


def test_apply_prev_mask_standard():
    """Test that previous mask is applied correctly to standard Conv2d weights."""
    conv = nn.Conv2d(12, 24, kernel_size=3, groups=1, bias=False)
    weight = conv.weight.clone()  # (6, 4, 3, 3)

    mask_prev = torch.tensor([1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=torch.bool)

    pruned_weight, mask, new_groups = apply_prev_mask(weight, mask_prev, groups=1)

    assert new_groups == 1

    # All output channels must be preserved
    assert torch.all(mask)

    # Expect 2 input channels kept
    assert pruned_weight.shape[1] == mask_prev.sum().item()
    # Check consistency with manual masking
    expected = weight[:, mask_prev, :, :]
    assert torch.all(pruned_weight == expected)


def test_apply_prev_mask_grouped_preserve():
    """Test that grouped convolution preserves expected channels when previous masks are applied."""
    groups = 2
    conv = nn.Conv2d(6, 8, kernel_size=3, groups=groups, bias=False)
    weight = conv.weight.clone()  # (8, 3, 3, 3)

    # Keep [0,2] for all groups
    mask_prev = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.bool)

    pruned_weight, mask, new_groups = apply_prev_mask(weight, mask_prev, groups=groups)

    # Each group should have pruned in_channels=2
    in_per_group = conv.in_channels // groups
    conv.out_channels // groups
    expected_chunks = []
    for g, w_chunk in enumerate(weight.chunk(groups, 0)):
        start, end = g * in_per_group, (g + 1) * in_per_group
        m_chunk = mask_prev[start:end]

        # assert each chunk has retained 2 channels
        assert m_chunk.sum().item() == 2
        expected_chunks.append(w_chunk[:, m_chunk, :, :])
    expected = torch.cat(expected_chunks, dim=0)

    assert pruned_weight.shape == expected.shape
    assert torch.all(pruned_weight == expected)


def test_apply_prev_mask_grouped_remove():
    """Test that grouped convolution removes entire groups when corresponding mask entries are zero."""
    groups = 4
    conv = nn.Conv2d(8, 8, kernel_size=3, groups=groups, bias=False)
    weight = conv.weight  # (8, 3, 3, 3)

    # Keep all or none for each group
    mask_prev = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)

    pruned_weight, mask, new_groups = apply_prev_mask(weight, mask_prev, groups=groups)

    assert new_groups < groups
    assert new_groups == 2

    # pruned weight's are a result of either keeping or discarding entire groups, not independent pruning of channels of each group
    # Actually, that's not necessary here, but is in the overall implementation
    expected = weight[mask]
    assert pruned_weight.shape == expected.shape
    assert torch.all(pruned_weight == expected)


# ==========================================================
#  MODULE-LEVEL TESTS (Conv, Bottleneck, C2f, SPPF)
# ==========================================================


@pytest.mark.parametrize(
    "mask_prev",
    [
        None,  # prune only outputs
        torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool),  # prune inputs + outputs
    ],
)
@pytest.mark.parametrize(
    "prune_ratio",
    [
        0.5,  # normal pruning
        0.0,  # no pruning edge case
    ],
)
@pytest.mark.parametrize("prune_type", ["preserve", "remove"])
def test_prune_conv(mask_prev, prune_ratio, prune_type):
    """Tests prune_conv wrapper for structural correctness, no-op pruning, and forward-pass shape consistency."""
    # Input tensor
    x = torch.randn(1, 8, 16, 16)

    # Conv block to prune
    conv = Conv(8, 16, 3)
    original = deepcopy(conv)

    # Prune
    mask_out = prune_conv(conv, prune_ratio=prune_ratio, norm_order=2, mask_prev=mask_prev, prune_type=prune_type)

    # === Structural checks ===

    compare_conv_layers(
        pruned_conv=conv, original_conv=original, mask_out=mask_out, mask_in=mask_prev, prune_type=prune_type
    )

    # === No-op pruning sanity check ===
    if prune_ratio == 0.0 and mask_prev is None:
        # No weights or channels should be removed
        assert torch.equal(conv.conv.weight, original.conv.weight)
        assert conv.conv.out_channels == original.conv.out_channels
        assert conv.conv.in_channels == original.conv.in_channels
        if conv.conv.bias is not None:
            assert torch.all(conv.conv.bias, original.conv.bias)
        # BN params unchanged
        assert torch.equal(conv.bn.running_mean, original.bn.running_mean)
        assert torch.equal(conv.bn.running_var, original.bn.running_var)
        if conv.bn.affine:
            assert torch.equal(conv.bn.weight, original.bn.weight)
            assert torch.equal(conv.bn.bias, original.bn.bias)

    # === Forward pass shape checks ===
    if mask_prev is None:
        y_pruned = conv(x)
    else:
        y_pruned = conv(x[:, mask_prev, :, :])

    # Output channel count matches mask
    assert y_pruned.shape[1] == mask_out.sum().item()
    # Spatial size unchanged by autopad
    assert y_pruned.shape[2:] == (16, 16)


def test_prune_conv_with_skip():
    """Tests prune_conv_with_skip on composite Conv module — ensures skip mask alignment and structural correctness."""
    conv = Conv(32, 32, 3)
    conv_copy = deepcopy(conv)
    mask_prev = torch.tensor([0, 1, 1, 0], dtype=torch.bool).repeat(8)
    mask_skip = torch.tensor([1, 0], dtype=torch.bool).repeat(16)
    mask_out = prune_conv_with_skip(conv_copy, mask_skip=mask_skip, mask_prev=mask_prev)
    assert torch.all(mask_out == mask_skip)
    compare_conv_layers(pruned_conv=conv_copy, original_conv=conv, mask_out=mask_out, mask_in=mask_prev)


@pytest.mark.parametrize("shortcut", [True, False])
@pytest.mark.parametrize(
    "mask_prev",
    [
        None,  # out-only pruning
        torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool),  # in+out pruning
    ],
)
def test_prune_bottleneck(shortcut, mask_prev):
    """Tests Bottleneck pruning: verifies mask propagation, shortcut handling, and output shape consistency."""
    x = torch.randn(1, 8, 16, 16)
    bottleneck = Bottleneck(8, 8, shortcut=shortcut)
    original = deepcopy(bottleneck)

    mask_tracker = {}
    cv2_mask = prune_bottleneck(
        bottleneck=bottleneck,
        prune_ratio=0.5,
        norm_order=2,
        mask_prev=mask_prev,
        mask_tracker=mask_tracker,
    )

    # Always compare pruned vs original using tracked masks
    compare_bottleneck_layers(
        pruned_bottleneck=bottleneck,
        original_bottleneck=original,
        mask_tracker=mask_tracker,
    )

    if shortcut:
        if mask_prev is None:
            # Out-only pruning: cv2 outputs must remain unpruned (all True)
            assert cv2_mask.all(), "cv2 outputs cannot be pruned if skip connection present"
        else:
            # In+out pruning: cv2 outputs must match skip input mask
            assert torch.equal(cv2_mask, mask_prev), "cv2 outputs must equal skip mask when shortcut=True"
    else:
        # Without residual, cv2 prunes independently
        assert cv2_mask.sum() <= original.cv2.conv.out_channels

    # Forward shape check
    if mask_prev is None:
        y_pruned = bottleneck(x)
    else:
        y_pruned = bottleneck(x[:, mask_prev, :, :])
    assert y_pruned.shape[1] == cv2_mask.sum()


@pytest.mark.parametrize("prune_groups", [None, 8])
def test_prune_c2f(prune_groups):
    """Tests C2f module pruning: ensures correct mask tracking across internal Bottlenecks and final Conv layers."""
    c1 = 64
    c2 = 64
    n = 2
    shortcut = False
    prune_ratio = 0.25
    norm_order = 2
    mask_prev = torch.tensor([i % 2 for i in range(c1)], dtype=torch.bool)
    mask_tracker = {}
    c2f = C2f(c1, c2, n=n, shortcut=shortcut, g=1, e=0.5)
    c2f_orig = deepcopy(c2f)
    mask_cv2_c2f = prune_c2f(
        c2f_layer=c2f,
        prune_ratio=prune_ratio,
        norm_order=norm_order,
        mask_prev=mask_prev,
        prune_groups=prune_groups,
        mask_tracker=mask_tracker,
    )

    # Checking masking for C2f.cv11
    chunk1_mask = mask_tracker["chunk1_mask"]
    chunk2_mask = mask_tracker["chunk2_mask"]

    compare_conv_layers(
        pruned_conv=c2f.cv1,
        original_conv=c2f_orig.cv1,
        mask_out=torch.cat([chunk1_mask, chunk2_mask]),
        mask_in=mask_prev,
    )

    assert all(isinstance(m, Bottleneck) for m in c2f.m), (
        f"Expected all elements of c2f.m to be Bottleneck, but got {[type(m) for m in c2f.m]}"
    )

    mask_prev = chunk2_mask  # mask flow across the components in the ModuleList 'm'
    component_out_masks = []
    # check each component of m is pruned correctly
    for i in range(len(c2f.m)):
        mask_dict_bottleneck, mask_dict_component = mask_tracker[f"bottleneck_{i}"], mask_tracker[f"m_{i}"]

        # assert components of the 'm' attribute has in and out masks equal to the Bottleneck's cv1's input mask and
        # cv2's output mask respectively. This must hold since 'm' is a list of Bottlenecks in this case
        assert torch.equal(mask_dict_component["in"], mask_dict_bottleneck["cv1_input"])
        assert torch.equal(mask_prev, mask_dict_bottleneck["cv1_input"])
        assert torch.equal(mask_dict_component["out"], mask_dict_bottleneck["cv2_output"])
        assert torch.equal(mask_dict_bottleneck["cv2_input"], mask_dict_bottleneck["cv1_output"])

        # Each Bottleneck is consistently pruned
        bottleneck = c2f_orig.m[i]
        pruned_bottleneck = c2f.m[i]

        # assert bottlenecks are applied with appropriate mask
        compare_bottleneck_layers(
            pruned_bottleneck=pruned_bottleneck, original_bottleneck=bottleneck, mask_tracker=mask_dict_bottleneck
        )

        mask_prev = mask_dict_component["out"]
        component_out_masks.append(mask_prev)

    mask_prev = torch.cat([chunk1_mask, chunk2_mask] + component_out_masks)

    compare_conv_layers(pruned_conv=c2f.cv2, original_conv=c2f_orig.cv2, mask_out=mask_cv2_c2f, mask_in=mask_prev)

    assert torch.all(mask_tracker["cv2_mask"] == mask_cv2_c2f), "cv2 mask not recorded correctly"


def test_prune_sppf():
    """Tests SPPF pruning: verifies correct mask propagation through pooled branches and final Conv consistency."""
    c1 = 256
    c2 = 256
    sppf = SPPF(c1, c2, 5)
    sppf_original = deepcopy(sppf)

    mask_prev = torch.tensor([i % 4 == 0 for i in range(c1)])
    mask_tracker = {}
    mask_out = prune_sppf(
        sppf_layer=sppf, prune_ratio=0.25, norm_order=2, mask_prev=mask_prev, mask_tracker=mask_tracker
    )
    assert torch.equal(mask_out, mask_tracker["cv2_output"])
    assert torch.equal(mask_tracker["cv1_output"].repeat(4), mask_tracker["cv2_input"])
    compare_conv_layers(
        pruned_conv=sppf.cv1, original_conv=sppf_original.cv1, mask_out=mask_tracker["cv1_output"], mask_in=mask_prev
    )
    compare_conv_layers(
        pruned_conv=sppf.cv2,
        original_conv=sppf_original.cv2,
        mask_out=mask_tracker["cv2_output"],
        mask_in=mask_tracker["cv2_input"],
    )


# ==========================================================
#  MODEL-LEVEL TESTS
# ==========================================================


@pytest.mark.parametrize(
    "prune_type, prev_masks_fn, x_fn",
    [
        (
            "preserve",
            lambda: [torch.ones(ch, dtype=torch.bool) for ch in [128, 256, 512]],
            lambda: [torch.randn(1, ch, 32 // (2**i), 32 // (2**i)) for i, ch in enumerate([128, 256, 512])],
        ),
        (
            "remove",
            lambda: [torch.tensor([1, 0], dtype=torch.bool).repeat(ch // 2) for ch in [128, 256, 512]],
            lambda: [torch.randn(1, ch // 2, 32 // (2**i), 32 // (2**i)) for i, ch in enumerate([128, 256, 512])],
        ),
    ],
    ids=["preserve", "remove"],
)
@pytest.mark.parametrize("prune_ratio_detect, prune_ratio_classify", [(0.25, 0.4), (0, 0)], ids=["partial", "none"])
@pytest.mark.parametrize(
    "detect_head_fn",
    [
        pytest.param(lambda: Detect(nc=80, ch=(128, 256, 512)), id="custom"),
        pytest.param(lambda: YOLO("yolov8s.yaml").model.model[-1], id="legacy_v8"),
        pytest.param(lambda: YOLO("yolo11s.yaml").model.model[-1], id="non_legacy_v11"),
    ],
)
def test_prune_detect_head(prune_type, prev_masks_fn, x_fn, prune_ratio_detect, prune_ratio_classify, detect_head_fn):
    """Verifies correct pruning and mask propagation within Detect head towers (cv2 and cv3) across YOLOv8 and YOLOv11
    variants, ensuring layer-to-layer continuity and valid forward outputs.
    """
    prev_masks_detect = prev_masks_fn()
    x = x_fn()
    detect_head = detect_head_fn()
    detect_head_orig = deepcopy(detect_head)

    mask_tracker = {}
    prune_detect(
        detect_head,
        prune_ratio_detect=prune_ratio_detect,  # Prune regression towers
        prune_ratio_classify=prune_ratio_classify,  # Prune classification towers
        norm_order=2,
        feature_masks=prev_masks_detect,  # Masks from C2f outputs
        mask_tracker=mask_tracker,
    )

    # checking if each detection tower receives the correct input pruning mask
    for i, featuremap_mask, detection_tower_pruned, detection_tower_orig in zip(
        range(len(detect_head.cv2)), prev_masks_detect, detect_head.cv2, detect_head_orig.cv2
    ):
        assert torch.equal(mask_tracker[f"detection_tower_{i}_component_{0}_input"], featuremap_mask)  # detection
        # tower's first component's input mask is equal to the mask from the corresponding featuremap

        # output mask of each component of the tower is the input mask of the next component.
        # And applying those masks results in exactly the same component as the compared to the pruned one

        for j in range(2):
            input_mask_j = mask_tracker[f"detection_tower_{i}_component_{j}_input"]
            output_mask_j = mask_tracker[f"detection_tower_{i}_component_{j}_output"]
            input_mask_j_plus_1 = mask_tracker[f"detection_tower_{i}_component_{j + 1}_input"]
            assert torch.equal(output_mask_j, input_mask_j_plus_1)
            pruned_conv = detection_tower_pruned[j]
            conv_orig = detection_tower_orig[j]

            compare_conv_layers(
                pruned_conv=pruned_conv, original_conv=conv_orig, mask_out=output_mask_j, mask_in=input_mask_j
            )

        compare_conv2d_layers(
            pruned_conv2d=detection_tower_pruned[-1],
            original_conv2d=detection_tower_orig[-1],
            mask_out=mask_tracker[f"detection_tower_{i}_component_{2}_output"],
            mask_in=mask_tracker[f"detection_tower_{i}_component_{2}_input"],
        )

    ## classification tower:
    for i, featuremap_mask, classification_tower_pruned, classification_tower_orig in zip(
        range(len(detect_head.cv3)), prev_masks_detect, detect_head.cv3, detect_head_orig.cv3
    ):
        first_component = classification_tower_pruned[0]

        if isinstance(first_component, Conv):  # if legacy
            for j in range(2):
                input_mask_j = mask_tracker[f"classification_tower_{i}_component_{j}_input"]
                output_mask_j = mask_tracker[f"classification_tower_{i}_component_{j}_output"]
                input_mask_j_plus_1 = mask_tracker[f"classification_tower_{i}_component_{j + 1}_input"]

                assert torch.equal(output_mask_j, input_mask_j_plus_1)
                pruned_conv = classification_tower_pruned[j]
                conv_orig = classification_tower_orig[j]
                if j == 0:
                    print(f"\nprune_type:{prune_type}")
                    print(f"pruned:{pruned_conv.conv.weight.shape}, conv:{conv_orig.conv.weight.shape}")
                compare_conv_layers(
                    pruned_conv=pruned_conv, original_conv=conv_orig, mask_out=output_mask_j, mask_in=input_mask_j
                )

            compare_conv2d_layers(
                pruned_conv2d=classification_tower_pruned[-1],
                original_conv2d=classification_tower_orig[-1],
                mask_out=mask_tracker[f"classification_tower_{i}_component_{2}_output"],
                mask_in=mask_tracker[f"classification_tower_{i}_component_{2}_input"],
            )

        else:
            for j in range(2):
                tower_component_pruned = classification_tower_pruned[j]
                tower_component_orig = classification_tower_orig[j]
                for k in range(len(tower_component_pruned)):
                    input_mask = mask_tracker[f"classification_tower_{i}_component_{j}_{k}_input"]
                    index_next_j = j + 1 if k == (len(tower_component_pruned) - 1) else j
                    index_next_k = (k + 1) % len(tower_component_pruned)
                    output_mask = mask_tracker[f"classification_tower_{i}_component_{j}_{k}_output"]
                    input_mask_next = mask_tracker[
                        f"classification_tower_{i}_component_{index_next_j}_{index_next_k}_input"
                    ]
                    assert torch.equal(output_mask, input_mask_next)

                    pruned_conv = tower_component_pruned[k]
                    conv_orig = tower_component_orig[k]
                    compare_conv_layers(
                        pruned_conv=pruned_conv,
                        original_conv=conv_orig,
                        mask_out=output_mask,
                        mask_in=input_mask,
                        prune_type=prune_type,
                    )

            compare_conv2d_layers(
                pruned_conv2d=classification_tower_pruned[-1],
                original_conv2d=classification_tower_orig[-1],
                mask_out=mask_tracker[f"classification_tower_{i}_component_{2}_{0}_output"],
                mask_in=mask_tracker[f"classification_tower_{i}_component_{2}_{0}_input"],
                prune_type=prune_type,
            )

    y = detect_head(x)
    assert all([yi.shape[1] > 0 for yi in y])  # no zero channels


@pytest.mark.parametrize("prune_type", ["preserve", "remove"])
def test_prune_detection_model(prune_type):
    """Verifies mask propagation and consistency across top-level modules in DetectionModel (Conv, C2f, SPPF, Detect),
    ensuring inter-module connectivity and structure remain valid after pruning.
    """
    mask_tracker = {}
    model = YOLO("yolov8n.yaml")
    pruned_model = prune_detection_model(model, prune_ratio=0.25, mask_tracker=mask_tracker, prune_type=prune_type)

    mask = None
    save_indices = model.model.save  # Indices of layers whose outputs are saved for later use
    save_indices_masks = {}  # Dict mapping layer index to its pruning mask
    pruned_components = pruned_model.model.model
    model_components = model.model.model
    for component, pruned_component in zip(model_components, pruned_components):
        i, f = component.i, component.f

        if isinstance(component, Upsample):
            # Upsample layers have no weights; skip pruning
            continue

        mask_input = mask_tracker[f"{i}_input"]
        mask_output = mask_tracker[f"{i}_output"]
        mask_tracker_comp = mask_tracker[f"{i}_mask_tracker"]

        if isinstance(component, Concat):
            mask = torch.cat([mask, save_indices_masks[f[1]]])
            continue

        if mask is not None and mask_input is not None:
            if isinstance(mask_input, torch.Tensor):
                assert torch.equal(mask, mask_input)

            if isinstance(mask_input, list):
                for feature_idx, feature_mask_mask_input in zip(component.f, mask_input):
                    assert torch.equal(feature_mask_mask_input, save_indices_masks[feature_idx]), (
                        f"Feature mask mismatch at index {feature_idx}"
                    )

        if isinstance(component, C2f):
            first_component = component.cv1
            last_component = component.cv2
            first_component_pruned = pruned_component.cv1
            last_component_pruned = pruned_component.cv2

            cv1_mask_out = torch.cat([mask_tracker_comp["chunk1_mask"], mask_tracker_comp["chunk2_mask"]])
            compare_conv_layers(
                pruned_conv=first_component_pruned,
                original_conv=first_component,
                mask_out=cv1_mask_out,
                mask_in=mask_input,
            )

            cv2_mask_in = [cv1_mask_out] + [mask_tracker_comp[f"m_{idx}"]["out"] for idx in range(len(component.m))]
            cv2_mask_in = torch.cat(cv2_mask_in)

            compare_conv_layers(
                pruned_conv=last_component_pruned,
                original_conv=last_component,
                mask_out=mask_tracker_comp["cv2_mask"],
                mask_in=cv2_mask_in,
            )

            mask = mask_tracker_comp["cv2_mask"]

        elif isinstance(component, SPPF):
            first_component = component.cv1
            first_component_pruned = pruned_component.cv1

            compare_conv_layers(
                pruned_conv=first_component_pruned,
                original_conv=first_component,
                mask_out=mask_tracker_comp["cv1_output"],
                mask_in=mask_tracker_comp["cv1_input"],
            )

            last_component = component.cv2
            last_component_pruned = pruned_component.cv2

            compare_conv_layers(
                pruned_conv=last_component_pruned,
                original_conv=last_component,
                mask_out=mask_tracker_comp["cv2_output"],
                mask_in=mask_tracker_comp["cv2_input"],
            )

            mask = mask_tracker_comp["cv2_output"]

        elif isinstance(component, Conv):
            # mask = prune_conv(component, prune_ratio, norm_order=norm_order, mask_prev=mask_prev)
            compare_conv_layers(
                pruned_conv=pruned_component, original_conv=component, mask_in=mask_input, mask_out=mask_output
            )

            mask = mask_output

        elif isinstance(component, Detect):
            # Collect masks for each featuremap fed to Detect

            cv2 = component.cv2
            cv3 = component.cv3

            cv2_pruned = pruned_component.cv2
            cv3_pruned = pruned_component.cv3

            for idx, detection_tower, detection_tower_pruned, mask_in in zip(range(3), cv2, cv2_pruned, mask_input):
                first_component = detection_tower[0]
                first_component_pruned = detection_tower_pruned[0]

                compare_conv_layers(
                    pruned_conv=first_component_pruned,
                    original_conv=first_component,
                    mask_out=mask_tracker_comp[f"detection_tower_{idx}_component_{0}_output"],
                    mask_in=mask_in,
                )

            for idx, classification_tower, classification_tower_pruned, mask_in in zip(
                range(3), cv3, cv3_pruned, mask_input
            ):
                first_component = classification_tower[0]
                first_component_pruned = classification_tower_pruned[0]

                if isinstance(first_component, nn.Sequential):
                    first_component = first_component[0]
                    first_component_pruned = first_component_pruned[0]

                    compare_conv_layers(
                        pruned_conv=first_component_pruned,
                        original_conv=first_component,
                        mask_out=mask_tracker_comp[f"classification_tower_{idx}_component_{0}_{0}_output"],
                        mask_in=mask_in,
                    )
                else:
                    # simple conv
                    compare_conv_layers(
                        pruned_conv=first_component_pruned,
                        original_conv=first_component,
                        mask_out=mask_tracker_comp[f"classification_tower_{idx}_component_{0}_output"],
                        mask_in=mask_in,
                    )

        else:
            print(f"Class: {type(component)} not supported yet for pruning!")

        # Save mask for layers whose outputs are needed downstream
        if i in save_indices:
            save_indices_masks[i] = mask


# ==========================================================
#  INTEGRATION / EXPORT TESTS
# ==========================================================


def test_prune_roundtrip(tmp_path):
    """Test that pruning with a global ratio saves/loads correctly and inference still runs."""
    model = YOLO("yolov8n.pt")

    pruned_model = prune_detection_model(model, prune_ratio=0.25)

    # save and reload
    save_path = tmp_path / "pruned.pt"
    pruned_model.save(save_path)
    loaded_model = YOLO(str(save_path))

    # dummy inference
    results = loaded_model(str(IMG))
    assert results is not None


def test_prune_roundtrip_with_config(tmp_path):
    """Test that pruning with a YAML config works and the pruned model saves/loads correctly and inference still
    runs.
    """
    model = YOLO("yolov8n.pt")
    pruned_model = prune_detection_model(model, prune_yaml=str(CFG))

    # save and reload
    save_path = tmp_path / "pruned_cfg.pt"
    pruned_model.save(save_path)
    loaded_model = YOLO(str(save_path))

    # dummy inference
    results = loaded_model(str(IMG))
    assert results is not None


# @pytest.mark.slow
def test_prune_train(tmp_path):
    """Test that a pruned model can still be trained after pruning, and that the trained model's inference still
    works.
    """
    model = YOLO("yolov8n.pt")
    pruned_model = prune_detection_model(model, prune_ratio=0.1)

    # save and reload
    save_path = tmp_path / "pruned_train.pt"
    pruned_model.save(save_path)
    loaded_model = YOLO(str(save_path))

    # train briefly
    loaded_model.train(data="coco8.yaml", epochs=2, imgsz=32)

    # dummy inference
    results = loaded_model(str(IMG))
    assert results is not None


def test_prune_reduces_size(tmp_path):
    """Test that pruning reduces the number of parameters."""
    model = YOLO("yolov8n.pt")

    # prune and save
    prune_ratio = 0.2
    pruned_model = prune_detection_model(model, prune_ratio=prune_ratio)

    # check that number of parameters is reduced
    orig_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    assert pruned_params < orig_params, f"Pruned params {pruned_params} >= original {orig_params}"


def test_zero_prune(tmp_path):
    """Test that pruning with zero pruning ratio doesnt change model weights."""
    model = YOLO("yolov8n.yaml")
    pruned_model = prune_detection_model(model, prune_ratio=0)

    # Check number of parameters
    assert sum(p.numel() for p in pruned_model.parameters()) == sum(p.numel() for p in model.parameters())

    # Check that parameter values are identical
    assert all(torch.equal(p1, p2) for p1, p2 in zip(model.parameters(), pruned_model.parameters()))


# @pytest.mark.slow
def test_pruned_model_export_and_reload(tmp_path):
    """End-to-end check that a pruned YOLO model saves/loads correctly, preserves the 'is_pruned' flag, exports to ONNX,
    and runs inference after both reload and export.
    """
    # Build + prune
    model = YOLO("yolov8n.pt")
    pruned = prune_detection_model(model, prune_ratio=0.25)

    # Verify pruning metadata before save
    assert hasattr(pruned.model, "is_pruned"), "Missing 'is_pruned' flag pre-save"

    # Save + reload round-trip
    save_path = tmp_path / "pruned.pt"
    pruned.save(save_path)
    loaded = YOLO(str(save_path))
    assert hasattr(loaded.model, "is_pruned"), "Flag lost after reload"

    # Optional tiny training check (CI-safe single epoch)
    loaded.train(data="coco8.yaml", epochs=1, imgsz=32, device="cpu")
    assert hasattr(loaded.model, "is_pruned"), "Flag missing after training"

    # Inference after reload
    results = loaded.predict(source=IMG, imgsz=32, device="cpu")
    assert results is not None and len(results) > 0, "Inference after reload failed"

    # Export to ONNX (explicit path for CI)
    onnx_path = loaded.export(format="onnx", half=True)
    onnx_path = Path(onnx_path)
    assert onnx_path.exists(), f"ONNX export failed: {onnx_path} not found"

    # Reload ONNX + run inference
    onnx_model = YOLO(str(onnx_path))
    results_onnx = onnx_model.predict(source=IMG, imgsz=32, device="cpu")
    assert results_onnx is not None and len(results_onnx) > 0, "Inference after ONNX export failed"


# ==========================================================
#  YAML CONFIG VALIDATION TESTS
# ==========================================================


def test_yaml_valid():
    """Valid prune YAML parses successfully and returns a complete ratio mapping."""
    sample_yaml = ROOT / Path("ultralytics/cfg/pruning/sample_prune.yaml")
    ratios = validate_prune_cfg(sample_yaml)
    assert isinstance(ratios, dict)
    assert len(ratios) == 23
    assert 22 in ratios
    assert isinstance(ratios[22], list) and len(ratios[22]) == 2


def test_yaml_incomplete_config_fails(tmp_path):
    """Incomplete prune YAML (missing required layer ratios) raises an AssertionError."""
    yaml_path = tmp_path / "incomplete.yaml"
    yaml_path.write_text("""
    prune_ratios:
      0: 0.1
      1: 0.2
      22: [0.3, 0.4]
    """)
    with pytest.raises(AssertionError, match="Missing prune ratios|Incomplete prune config"):
        validate_prune_cfg(yaml_path)


def test_yaml_missing_prune_ratios_key(tmp_path):
    """Missing top-level 'prune_ratios' key in YAML triggers an AssertionError."""
    bad_yaml = tmp_path / "missing_key.yaml"
    bad_yaml.write_text("not_prune_ratios:\n  0: 0.1\n")
    with pytest.raises(AssertionError, match="Missing key 'prune_ratios'"):
        validate_prune_cfg(bad_yaml)


def test_yaml_invalid_ratio_type(tmp_path):
    """Non-numeric prune ratios (e.g., strings) raise an AssertionError."""
    bad_yaml = tmp_path / "invalid_type.yaml"
    bad_yaml.write_text("""
    prune_ratios:
      0: "abc"
    """)
    with pytest.raises(AssertionError, match="Invalid ratio type"):
        validate_prune_cfg(bad_yaml)


def test_yaml_ratio_out_of_range(tmp_path):
    """Prune ratio values outside [0,1] range raise an AssertionError."""
    bad_yaml = tmp_path / "out_of_range.yaml"
    bad_yaml.write_text("""
    prune_ratios:
      0: 1.5
    """)
    with pytest.raises(AssertionError, match="out of range"):
        validate_prune_cfg(bad_yaml)


def test_yaml_invalid_detect_head_format(tmp_path):
    """Detect-head YAML entries must contain exactly two numeric ratios; invalid lists raise an AssertionError."""
    bad_yaml = tmp_path / "bad_detect.yaml"
    bad_yaml.write_text("""
    prune_ratios:
      22: [0.3, 0.4, 0.5]
    """)
    with pytest.raises(AssertionError, match="must contain exactly two ratios"):
        validate_prune_cfg(bad_yaml)


def test_yaml_syntax_error(tmp_path):
    """Malformed YAML syntax raises a ValueError during validation."""
    bad_yaml = tmp_path / "malformed.yaml"
    bad_yaml.write_text("prune_ratios: [0.1, 0.2,,]")
    with pytest.raises(ValueError, match="Invalid YAML syntax"):
        validate_prune_cfg(bad_yaml)


@pytest.mark.parametrize(
    "vals, expected",
    [
        ([8, 12, 16], 4),
        ([7], 7),
        ([], 1),
        ([9, 6], 3),
        ([0, 12, 24], 12),
        ([-8, -12], 4),
        ([-8, 12], 4),
        ([5, 7, 11], 1),
    ],
)
def test_gcd_all(vals, expected):
    """Test gcd_all() with various input types and edge cases."""
    assert gcd_all(vals) == expected
