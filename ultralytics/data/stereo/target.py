# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Ground truth target generator for Stereo CenterNet training."""

from __future__ import annotations

import math

import numpy as np
import torch


class TargetGenerator:
    """Generate ground truth targets for 10-branch Stereo CenterNet head.

    Creates Gaussian heatmaps and regression targets for all 10 branches.
    """

    def __init__(
        self,
        output_size: tuple[int, int] = (96, 320),  # H/4, W/4 for 384×1280 input
        num_classes: int = 3,
        mean_dims: dict[str, list[float]] | None = None,
    ):
        """Initialize target generator.

        Args:
            output_size: Output feature map size (H, W) at 1/4 resolution.
            num_classes: Number of object classes.
            mean_dims: Mean dimensions per class [L, W, H] in meters.
        """
        self.output_h, self.output_w = output_size
        self.num_classes = num_classes

        # Default mean dimensions (KITTI)
        self.mean_dims = mean_dims or {
            "Car": [3.88, 1.63, 1.53],
            "Pedestrian": [0.88, 0.60, 1.73],
            "Cyclist": [1.72, 0.60, 1.77],
        }

    def generate_targets(
        self, labels: list[dict], input_size: tuple[int, int] = (384, 1280)
    ) -> dict[str, torch.Tensor]:
        """Generate ground truth targets for all 10 branches.

        Args:
            labels: List of label dictionaries from dataset.
            input_size: Input image size (H, W).

        Returns:
            Dictionary with 10 branch targets, each [num_classes or channels, H/4, W/4].
        """
        input_h, input_w = input_size
        scale_h = self.output_h / input_h
        scale_w = self.output_w / input_w

        # Initialize targets
        targets = {
            "heatmap": torch.zeros(self.num_classes, self.output_h, self.output_w),
            "offset": torch.zeros(2, self.output_h, self.output_w),
            "bbox_size": torch.zeros(2, self.output_h, self.output_w),
            "lr_distance": torch.zeros(1, self.output_h, self.output_w),
            "right_width": torch.zeros(1, self.output_h, self.output_w),
            "dimensions": torch.zeros(3, self.output_h, self.output_w),
            "orientation": torch.zeros(8, self.output_h, self.output_w),
            "vertices": torch.zeros(8, self.output_h, self.output_w),
            "vertex_offset": torch.zeros(8, self.output_h, self.output_w),
            "vertex_dist": torch.zeros(4, self.output_h, self.output_w),
        }

        # Process each object
        for label in labels:
            class_id = label["class_id"]
            left_box = label["left_box"]
            right_box = label["right_box"]
            dimensions = label["dimensions"]
            alpha = label["alpha"]
            vertices = label["vertices"]

            # Get center coordinates in input image
            center_x = left_box["center_x"] * input_w
            center_y = left_box["center_y"] * input_h

            # Scale to output size
            center_x_out = center_x * scale_w
            center_y_out = center_y * scale_h

            # Integer center (for heatmap)
            center_x_int = int(center_x_out)
            center_y_int = int(center_y_out)

            # Skip if center is outside output bounds
            if not (0 <= center_x_int < self.output_w and 0 <= center_y_int < self.output_h):
                continue

            # 1. Heatmap (Gaussian)
            self._add_gaussian_heatmap(
                targets["heatmap"][class_id],
                center_x_out,
                center_y_out,
                left_box["width"] * input_w * scale_w,
                left_box["height"] * input_h * scale_h,
            )

            # 2. Offset (sub-pixel)
            targets["offset"][0, center_y_int, center_x_int] = center_x_out - center_x_int
            targets["offset"][1, center_y_int, center_x_int] = center_y_out - center_y_int

            # 3. Bbox size
            targets["bbox_size"][0, center_y_int, center_x_int] = left_box["width"] * input_w * scale_w
            targets["bbox_size"][1, center_y_int, center_x_int] = left_box["height"] * input_h * scale_h

            # 4. LR distance (disparity)
            lr_dist = center_x - (right_box["center_x"] * input_w)
            targets["lr_distance"][0, center_y_int, center_x_int] = lr_dist * scale_w

            # 5. Right width
            right_w = right_box["width"] * input_w * scale_w
            # Transform: wr = 1/σ(ŵr) - 1, so target = 1/(wr + 1)
            targets["right_width"][0, center_y_int, center_x_int] = 1.0 / (right_w + 1.0)

            # 6. Dimensions (offset from mean)
            # Map class_id to class name (KITTI classes: Car=0, Van=1, Truck=2, Pedestrian=3, Person_sitting=4, Cyclist=5, Tram=6, Misc=7)
            class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
            if class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = "Car"  # Default fallback
            mean_dim = self.mean_dims.get(class_name, [1.0, 1.0, 1.0])
            dim_offset = [
                dimensions["length"] - mean_dim[0],
                dimensions["width"] - mean_dim[1],
                dimensions["height"] - mean_dim[2],
            ]
            targets["dimensions"][:, center_y_int, center_x_int] = torch.tensor(dim_offset)

            # 7. Orientation (multi-bin encoding)
            # Simplified: use 2 bins, encode as [bin_logits, sin, cos]
            orientation_target = self._encode_orientation(alpha)
            targets["orientation"][:, center_y_int, center_x_int] = orientation_target

            # 8-10. Vertices (simplified - would need full 3D geometry)
            # For now, use normalized vertex coordinates
            for i, (vx, vy) in enumerate(
                [
                    vertices["v1"],
                    vertices["v2"],
                    vertices["v3"],
                    vertices["v4"],
                ]
            ):
                vx_out = vx * input_w * scale_w
                vy_out = vy * input_h * scale_h
                targets["vertices"][i * 2, center_y_int, center_x_int] = vx_out
                targets["vertices"][i * 2 + 1, center_y_int, center_x_int] = vy_out

        return targets

    def _add_gaussian_heatmap(
        self, heatmap: torch.Tensor, cx: float, cy: float, width: float, height: float
    ):
        """Add Gaussian heatmap for an object center.

        Args:
            heatmap: Heatmap tensor to update [H, W].
            cx: Center x coordinate.
            cy: Center y coordinate.
            width: Object width (for sigma calculation).
            height: Object height (for sigma calculation).
        """
        # Calculate sigma based on object size
        sigma_x = 0.6 * width / 6.0
        sigma_y = 0.6 * height / 6.0

        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(heatmap.shape[0], dtype=torch.float32),
            torch.arange(heatmap.shape[1], dtype=torch.float32),
            indexing="ij",
        )

        # Compute Gaussian
        gaussian = torch.exp(
            -((x_coords - cx) ** 2 / (2 * sigma_x**2) + (y_coords - cy) ** 2 / (2 * sigma_y**2))
        )

        # Update heatmap (take maximum)
        heatmap[:] = torch.maximum(heatmap, gaussian)

    def _encode_orientation(self, alpha: float) -> torch.Tensor:
        """Encode orientation angle into multi-bin format.

        Multi-Bin encoding (2 bins):
        - Bin 0: α ∈ [-π, 0], center = -π/2
        - Bin 1: α ∈ [0, π], center = +π/2
        
        Output format: [conf_0, conf_1, sin_0, cos_0, sin_1, cos_1, pad, pad]

        Args:
            alpha: Observation angle in radians, range [-π, π].

        Returns:
            Orientation encoding tensor of shape [8].
        """
        # Normalize alpha to [-π, π]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        
        # Determine bin based on angle sign
        # Bin 0: [-π, 0), center = -π/2
        # Bin 1: [0, π], center = +π/2
        if alpha < 0:
            bin_idx = 0
            bin_center = -math.pi / 2
        else:
            bin_idx = 1
            bin_center = math.pi / 2
        
        # Compute residual angle within bin
        residual = alpha - bin_center
        
        # Initialize encoding
        encoding = torch.zeros(8)
        
        # Set bin confidence (one-hot)
        encoding[0] = 1.0 if bin_idx == 0 else 0.0
        encoding[1] = 1.0 if bin_idx == 1 else 0.0
        
        # Set sin/cos for the ACTIVE bin only
        sin_val = math.sin(residual)
        cos_val = math.cos(residual)
        
        if bin_idx == 0:
            encoding[2] = sin_val  # sin_0
            encoding[3] = cos_val  # cos_0
            # encoding[4:6] stays 0 for inactive bin
        else:
            # encoding[2:4] stays 0 for inactive bin
            encoding[4] = sin_val  # sin_1
            encoding[5] = cos_val  # cos_1
        
        # Channels 6-7 are padding (remain 0)
        
        return encoding

