import sys
import os
import torch

# Add the parent directory to the path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ultralytics.models.yolo.yoloe.fusion import fuse_tpe_vpe


def test_concat_fusion():
    """Test concatenation fusion mode."""
    tpe = torch.randn(2, 3, 16)
    vpe = torch.randn(2, 2, 16)
    out = fuse_tpe_vpe(tpe, vpe, mode="concat")
    assert out.shape == (2, 5, 16)


def test_sum_fusion_no_map_equal_counts():
    """Test sum fusion with equal counts and no mapping."""
    tpe = torch.randn(1, 2, 8)
    vpe = torch.randn(1, 2, 8)
    out = fuse_tpe_vpe(tpe, vpe, mode="sum", alpha=0.7)
    assert out.shape == (1, 2, 8)


def test_sum_fusion_with_map():
    """Test sum fusion with mapping for different counts."""
    tpe = torch.randn(1, 2, 8)
    vpe = torch.randn(1, 3, 8)  # 3 visual exemplars
    fuse_map = {0: [0, 2], 1: [1]}  # per-class mapping to visual indices
    out = fuse_tpe_vpe(tpe, vpe, mode="sum", alpha=0.5, fuse_map=fuse_map)
    assert out.shape == (1, 2, 8)


def test_none_inputs():
    """Test error handling for None inputs."""
    try:
        fuse_tpe_vpe(None, None)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_single_input_tpe_only():
    """Test with only text prompts."""
    tpe = torch.randn(1, 3, 16)
    out = fuse_tpe_vpe(tpe, None, mode="concat")
    assert out.shape == (1, 3, 16)


def test_single_input_vpe_only():
    """Test with only visual prompts."""
    vpe = torch.randn(1, 4, 16)
    out = fuse_tpe_vpe(None, vpe, mode="concat")
    assert out.shape == (1, 4, 16)


def test_mismatched_counts_without_map():
    """Test error when counts don't match and no map provided."""
    tpe = torch.randn(1, 2, 8)
    vpe = torch.randn(1, 3, 8)
    try:
        fuse_tpe_vpe(tpe, vpe, mode="sum")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_concat_fusion()
    test_sum_fusion_no_map_equal_counts()
    test_sum_fusion_with_map()
    test_none_inputs()
    test_single_input_tpe_only()
    test_single_input_vpe_only()
    test_mismatched_counts_without_map()
    print("All tests passed!")
