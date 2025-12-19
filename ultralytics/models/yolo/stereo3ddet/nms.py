# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Heatmap NMS utilities for Stereo CenterNet.

Paper Reference: Section 3.1 "For inference, use the 3×3 max pooling
                 operation instead of NMS."
"""

import torch
import torch.nn.functional as F


def heatmap_nms(heatmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Apply max pooling NMS to keep only local maxima in heatmaps.

    This function performs non-maximum suppression on detection heatmaps using
    max pooling. Only pixels that are local maxima within their neighborhood
    (defined by kernel_size) are preserved; all other values are set to zero.

    Paper Reference: Section 3.1 "For inference, use the 3×3 max pooling
                     operation instead of NMS."

    Args:
        heatmap: Detection heatmap tensor of shape [B, C, H, W] where:
            - B: batch size
            - C: number of classes (channels)
            - H: height
            - W: width
            The heatmap should typically be after sigmoid activation.
        kernel_size: Size of the max pooling kernel. Default is 3 for 3×3 pooling.
            Must be an odd positive integer.

    Returns:
        torch.Tensor: Heatmap with non-maxima suppressed to 0, same shape as input.
            Only local maxima within the kernel neighborhood are preserved.

    Example:
        >>> heatmap = torch.rand(1, 3, 128, 128)  # [B, C, H, W]
        >>> nms_heatmap = heatmap_nms(heatmap, kernel_size=3)
        >>> # Only local maxima remain, others are zeroed out
    """
    # Calculate padding to maintain spatial dimensions
    pad = (kernel_size - 1) // 2

    # Apply max pooling to find local maxima
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=pad)

    # Create mask: keep only pixels that equal the max in their neighborhood
    keep = (hmax == heatmap).float()

    # Zero out non-maxima
    return heatmap * keep

