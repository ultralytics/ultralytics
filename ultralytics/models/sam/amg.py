# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from itertools import product
from typing import Any, Generator, List, Tuple

import numpy as np
import torch


def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
) -> torch.Tensor:
    """Determines if bounding boxes are near the edge of a cropped image region using a specified tolerance."""
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """Yields batches of data from input arguments with specified batch size for efficient processing."""
    assert args and all(len(a) == len(args[0]) for a in args), "Batched iteration must have same-size inputs."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def calculate_stability_score(masks: torch.Tensor, mask_threshold: float, threshold_offset: float) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks.

    The stability score is the IoU between binary masks obtained by thresholding the predicted mask logits at
    high and low values.

    Args:
        masks (torch.Tensor): Batch of predicted mask logits.
        mask_threshold (float): Threshold value for creating binary masks.
        threshold_offset (float): Offset applied to the threshold for creating high and low binary masks.

    Returns:
        (torch.Tensor): Stability scores for each mask in the batch.

    Notes:
        - One mask is always contained inside the other.
        - Memory is saved by preventing unnecessary cast to torch.int64.

    Examples:
        >>> masks = torch.rand(10, 256, 256)  # Batch of 10 masks
        >>> mask_threshold = 0.5
        >>> threshold_offset = 0.1
        >>> stability_scores = calculate_stability_score(masks, mask_threshold, threshold_offset)
    """
    intersections = (masks > (mask_threshold + threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    unions = (masks > (mask_threshold - threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    return intersections / unions


def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generate a 2D grid of evenly spaced points in the range [0,1]x[0,1] for image segmentation tasks."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    return np.stack([points_x, points_y], axis=-1).reshape(-1, 2)


def build_all_layer_point_grids(n_per_side: int, n_layers: int, scale_per_layer: int) -> List[np.ndarray]:
    """Generates point grids for multiple crop layers with varying scales and densities."""
    return [build_point_grid(int(n_per_side / (scale_per_layer**i))) for i in range(n_layers + 1)]


def generate_crop_boxes(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """Generates crop boxes of varying sizes for multiscale image processing, with layered overlapping regions."""
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        """Crops bounding boxes to the size of the input image."""
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    """Uncrop bounding boxes by adding the crop box offset to their coordinates."""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


def uncrop_points(points: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    """Uncrop points by adding the crop box offset to their coordinates."""
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0]], device=points.device)
    # Check if points has a channel dimension
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset


def uncrop_masks(masks: torch.Tensor, crop_box: List[int], orig_h: int, orig_w: int) -> torch.Tensor:
    """Uncrop masks by padding them to the original image size, handling coordinate transformations."""
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    return torch.nn.functional.pad(masks, pad, value=0)


def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str) -> Tuple[np.ndarray, bool]:
    """Removes small disconnected regions or holes in a mask based on area threshold and mode."""
    import cv2  # type: ignore

    assert mode in {"holes", "islands"}, f"Provided mode {mode} is invalid"
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if not small_regions:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        # If every region is below threshold, keep largest
        fill_labels = [i for i in range(n_labels) if i not in fill_labels] or [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """Calculates bounding boxes in XYXY format around binary masks, handling empty masks and various input shapes."""
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    masks = masks.flatten(0, -3) if len(shape) > 2 else masks.unsqueeze(0)
    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    return out.reshape(*shape[:-2], 4) if len(shape) > 2 else out[0]
