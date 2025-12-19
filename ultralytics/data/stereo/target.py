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
        output_size: tuple[int, int] = (96, 320),  # Example: (96, 320) for 384×1280 input with 4x downsampling
        num_classes: int = 3,
        mean_dims: dict[str, list[float]] | None = None,
    ):
        """Initialize target generator.

        Args:
            output_size: Output feature map size (H, W). Determined dynamically from model architecture.
                         The actual downsampling factor depends on the model config (e.g., P3 = 8x, P4 = 16x).
            num_classes: Number of object classes.
            mean_dims: Mean dimensions per class [L, W, H] in meters.
        """
        self.output_h, self.output_w = output_size
        self.num_classes = num_classes

        # Default mean dimensions (KITTI) - format: [L, W, H]
        # Must match val.py mean_dims = {0: (H, W, L), ...} after reordering
        self.mean_dims = mean_dims or {
            "Car": [3.89, 1.73, 1.52],       # L=3.89, W=1.73, H=1.52
            "Pedestrian": [0.80, 0.50, 1.73], # L=0.80, W=0.50, H=1.73
            "Cyclist": [1.76, 0.60, 1.77],    # L=1.76, W=0.60, H=1.77
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
            # NOTE: After class filtering/remapping, class_id is 0=Car, 1=Pedestrian, 2=Cyclist
            class_names_map = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
            class_name = class_names_map.get(class_id, "Car")
            mean_dim = self.mean_dims.get(class_name, [1.0, 1.0, 1.0])
            # mean_dims is [L, W, H], but decoder expects [ΔH, ΔW, ΔL] order
            # to match val.py which uses mean_dims = {0: (H, W, L), ...}
            dim_offset = [
                dimensions["height"] - mean_dim[2],   # channel 0 = Δheight
                dimensions["width"] - mean_dim[1],    # channel 1 = Δwidth
                dimensions["length"] - mean_dim[0],   # channel 2 = Δlength
            ]
            targets["dimensions"][:, center_y_int, center_x_int] = torch.tensor(dim_offset)

            # 7. Orientation (multi-bin encoding)
            # Simplified: use 2 bins, encode as [bin_logits, sin, cos]
            orientation_target = self._encode_orientation(alpha)
            targets["orientation"][:, center_y_int, center_x_int] = orientation_target

            # 8-10. Vertices (simplified - would need full 3D geometry)
            # For now, use normalized vertex coordinates
            vertex_list = [vertices["v1"], vertices["v2"], vertices["v3"], vertices["v4"]]
            for i, (vx, vy) in enumerate(vertex_list):
                vx_out = vx * input_w * scale_w
                vy_out = vy * input_h * scale_h
                targets["vertices"][i * 2, center_y_int, center_x_int] = vx_out
                targets["vertices"][i * 2 + 1, center_y_int, center_x_int] = vy_out

                # vertex offset (sub-pixel refinement like center offset)
                vx_int = int(vx_out)
                vy_int = int(vy_out)
                targets["vertex_offset"][i * 2, center_y_int, center_x_int] = vx_out - vx_int
                targets["vertex_offset"][i * 2 + 1, center_y_int, center_x_int] = vy_out - vy_int

                # vertex distance (distance from center to vertex in feature map space)
                dx = vx_out - center_x_int
                dy = vy_out - center_y_int
                dist = (dx ** 2 + dy ** 2) ** 0.5
                targets["vertex_dist"][i, center_y_int, center_x_int] = dist

        return targets

    def _add_gaussian_heatmap(
        self, heatmap: torch.Tensor, cx: float, cy: float, width: float, height: float
    ):
        """Add Gaussian heatmap for an object center."""
        
        # Integer center location
        cx_int = int(cx)
        cy_int = int(cy)
        
        # Bounds check
        if not (0 <= cx_int < heatmap.shape[1] and 0 <= cy_int < heatmap.shape[0]):
            return
        
        # Calculate sigma based on object size (with minimum)
        sigma_x = max(0.6 * width / 6.0, 1.0)  # Minimum sigma = 1
        sigma_y = max(0.6 * height / 6.0, 1.0)
        
        # Determine the Gaussian radius (how far to compute)
        # Use 3-sigma rule: 99.7% of values within 3 sigma
        radius_x = int(3 * sigma_x)
        radius_y = int(3 * sigma_y)
        
        # Compute Gaussian only in local region (more efficient)
        y_min = max(0, cy_int - radius_y)
        y_max = min(heatmap.shape[0], cy_int + radius_y + 1)
        x_min = max(0, cx_int - radius_x)
        x_max = min(heatmap.shape[1], cx_int + radius_x + 1)
        
        # Create local coordinate grids CENTERED on integer location
        y_coords = torch.arange(y_min, y_max, dtype=torch.float32)
        x_coords = torch.arange(x_min, x_max, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        
        # Compute Gaussian centered on INTEGER center (not float!)
        # This ensures the peak is exactly 1.0 at (cx_int, cy_int)
        gaussian = torch.exp(
            -((xx - cx_int) ** 2 / (2 * sigma_x ** 2) + 
            (yy - cy_int) ** 2 / (2 * sigma_y ** 2))
        )
        
        # Update heatmap with element-wise maximum
        heatmap[y_min:y_max, x_min:x_max] = torch.maximum(
            heatmap[y_min:y_max, x_min:x_max], 
            gaussian
        )

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
        # validate input
        assert -np.pi <= alpha <= np.pi, f"alpha is out of range: {alpha}, {math.degrees(alpha)}"
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
        
        # Set bin confidence (one-hot encoding)
        # Channel 0: conf for bin 0 (1.0 if bin 0 is active, else 0.0)
        # Channel 1: conf for bin 1 (1.0 if bin 1 is active, else 0.0)
        # Format: [1, 0, ...] for bin 0, [0, 1, ...] for bin 1
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

