
"""Ground truth target generator for Stereo 3D detection with YOLO backbone.

This module implements the target generation for the 7-branch stereo 3D detection head
following the paper "Stereo CenterNet based 3D Object Detection for Autonomous Driving".

Note: This generator creates targets for 3D detection branches. The 2D detection
(heatmap, offset, bbox_size) has been replaced by YOLO-style TaskAlignedAssigner
with direct bounding box regression.

The 7 branches are:
Stereo 2D Association (2 branches):
    1. lr_distance: Left-Right center distance [1, H, W]
    2. right_width: Right box width [1, H, W]

3D Components (5 branches):
    3. dimensions: 3D dimension offsets (ΔH, ΔW, ΔL) [3, H, W]
    4. orientation: Multi-Bin orientation encoding [8, H, W]
    5. vertices: Bottom 4 vertex 2D coordinates [8, H, W] - (x0,y0,x1,y1,x2,y2,x3,y3)
    6. vertex_offset: Vertex sub-pixel offsets [8, H, W]
    7. vertex_dist: Center to vertex distances [4, H, W]

References:
    Paper: https://arxiv.org/abs/2103.11071
    Figure 4: Shows vertex ordering 0,1,2,3 at bottom of 3D box
    Figure 5: Shows geometric relationship between 2D and 3D
"""

from __future__ import annotations

import math
from typing import Any
import torch


class TargetGenerator:
    """Generate ground truth targets for stereo 3D detection with YOLO backbone.

    Creates regression targets for 7 branches (stereo association + 3D components).
    Following the paper, the bottom 4 vertices of the 3D bounding box are
    projected onto the image plane as keypoints for geometric constraints.

    Note: This generator is used with YOLO-based detection. Heatmap-based 2D detection
    has been replaced by TaskAlignedAssigner with direct bounding box regression.
    """

    def __init__(
        self,
        mean_dims,
        std_dims,
        output_size: tuple[int, int] = (96, 320),  # Example: (96, 320) for 384×1280 input with 4x downsampling
        num_classes: int = 3,
        class_names: dict[int, str] | None = None,
    ):
        """Initialize target generator.

        Args:
            output_size: Output feature map size (H, W). Determined dynamically from model architecture.
                         The actual downsampling factor depends on the model config (e.g., P3 = 8x, P4 = 16x).
            num_classes: Number of object classes.
            mean_dims: Mean dimensions per class [L, W, H] in meters.
                       Can be integer keys (class_id) or string keys (class_name).
            std_dims: Standard deviation of dimensions per class [L, W, H] in meters.
                      Can be integer keys (class_id) or string keys (class_name).
                      Used for normalized offset prediction: (dim - mean) / std.
            class_names: Mapping from class_id to class_name (e.g., {0: "Car", 1: "Pedestrian", ...}).
                       If None, will use generic names ("Class 0", "Class 1", ...).
        """
        self.output_h, self.output_w = output_size
        self.num_classes = num_classes

        # Class name mapping (dataset-specific)
        # This maps class_id -> class_name
        if class_names is not None:
            self.class_names_map = class_names
        else:
            # Use generic names if not provided
            self.class_names_map = {i: f"Class {i}" for i in range(num_classes)}

        # Build reverse mapping (class_name -> class_id) for backward compatibility
        self.class_name_to_id = {v: k for k, v in self.class_names_map.items()}

        # Handle both integer keys (class_id) and string keys (class_name) for mean_dims
        assert mean_dims is not None, "mean_dims must be provided"
        # Check if mean_dims uses integer keys (class IDs) or string keys (class names)
        self.mean_dims = mean_dims

        # Handle both integer keys (class_id) and string keys (class_name) for std_dims
        assert std_dims is not None, "std_dims must be provided"
        # Check if std_dims uses integer keys (class IDs) or string keys (class names)
        self.std_dims = std_dims
        
    def generate_targets(
        self,
        labels: list[dict],
        input_size: tuple[int, int] = (384, 1280),
        calib: list[dict[str, float]] | None = None,
        original_size: list[tuple[int, int]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate ground truth targets for stereo 3D detection (7 branches).

        Args:
            labels: List of label dictionaries from dataset. Each label should have:
                - class_id: int
                - left_box: dict with center_x, center_y, width, height (normalized to letterboxed input image)
                - right_box: dict with center_x, width (normalized to letterboxed input image)
                - dimensions: dict with length, width, height (meters)
                - alpha: observation angle in radians
                - location_3d (optional): dict with x, y, z (meters) - 3D center position
                  If not provided, computed from stereo disparity
                - vertices (optional): dict with v1, v2, v3, v4 - pre-computed 2D vertices
                  (normalized to letterboxed input image)
                  If not provided, computed from 3D parameters
            input_size: Input image size (H, W) after preprocessing (letterbox).
            calib: Camera calibration parameters dict with fx, fy, cx, cy, baseline.
                   Already transformed to letterboxed space by the dataset.
                   If None, uses default KITTI values.
            original_size: Original image size (H, W) before preprocessing.
                   Used for reference, not for coordinate transformation.
                   If None, uses KITTI default (375, 1242).

        Returns:
            Dictionary with 7 branch targets for stereo association and 3D detection,
            each [num_classes or channels, H_out, W_out]
            where H_out, W_out are determined by the model's output size (architecture-agnostic).

            Note: 2D detection targets (heatmap, offset, bbox_size) are NOT generated.
            These are replaced by YOLO-style TaskAlignedAssigner with direct bbox regression.
        """
        input_h, input_w = input_size
        
        # Scale calibration parameters to match preprocessed input size
        # Calibration parameters are typically in original image space (e.g., KITTI 1242x375)
        # We need to scale them to match the preprocessed input size
        # Scale factors from preprocessed input to feature map output
        # Initialize targets (7 branches for 3D detection with YOLO backbone)
        # Note: heatmap, offset, bbox_size are not generated - replaced by YOLO TaskAlignedAssigner
        targets = {
            "lr_distance": torch.zeros(1, self.output_h, self.output_w),
            "right_width": torch.zeros(1, self.output_h, self.output_w),
            "dimensions": torch.zeros(3, self.output_h, self.output_w),
            "orientation": torch.zeros(8, self.output_h, self.output_w),
            "vertices": torch.zeros(8, self.output_h, self.output_w),
            "vertex_offset": torch.zeros(8, self.output_h, self.output_w),
            "vertex_dist": torch.zeros(4, self.output_h, self.output_w),
        }

        # Process each object
        for label, calib, original_size in zip(labels, calib, original_size):
            self._process_single_label(
                label, targets, input_h, input_w, calib, original_size
            )
        return targets

    def _process_single_label(
        self,
        label: dict[str, Any],
        targets: dict[str, torch.Tensor],
        input_h: int,
        input_w: int,
        calib: dict[str, float],
        original_size: tuple[int, int],
    ) -> None:
        """Process a single label and update targets.

        Args:
            label: Single label dictionary.
            targets: Target tensors to update.
            input_h, input_w: Input image dimensions.
            calib: Camera calibration parameters.
            original_size: Original image size.
        """
        class_id = label["class_id"]
        left_box = label["left_box"]
        right_box = label["right_box"]
        dimensions = label["dimensions"]
        
        rotation_y = label["rotation_y"]
        location_3d = label["location_3d"]
        x_3d = location_3d["x"]
        z_3d = location_3d["z"]
        ray_angle = math.atan2(x_3d, z_3d)
        alpha = rotation_y - ray_angle
        

        fx = calib["fx"]
        fy = calib["fy"]
        cx = calib["cx"]
        cy = calib["cy"]
        baseline = calib["baseline"]

        # Calibration is already transformed to letterboxed space in the dataset,
        # so we don't need to scale it again here. The calibration parameters
        # (fx, fy, cx, cy) are already in the letterboxed input image space.
        # orig_h, orig_w = original_size  # Not needed for calibration scaling anymore

        # output scale factors
        scale_h = self.output_h / input_h
        scale_w = self.output_w / input_w

        # Get center coordinates in input image (pixels)
        center_x = left_box["center_x"] * input_w
        center_y = left_box["center_y"] * input_h

        # Scale to output size
        center_x_out = center_x * scale_w
        center_y_out = center_y * scale_h

        # Integer center (for sparse target assignment)
        center_x_int = int(center_x_out)
        center_y_int = int(center_y_out)

        # Skip if center is outside output bounds
        if not (0 <= center_x_int < self.output_w and 0 <= center_y_int < self.output_h):
            return

        # ============================================================
        # Stereo 2D Association (2 branches for YOLO-based detection)
        # ============================================================

        # 1. LR distance (left-right center distance for stereo association)
        # Paper Equation 4: distance between left and right object centers
        right_center_x = right_box["center_x"] * input_w
        lr_dist = center_x - right_center_x  # Disparity in pixels
        lr_distance_stored = lr_dist * scale_w
        targets["lr_distance"][0, center_y_int, center_x_int] = lr_distance_stored

        # 2. Right width (in feature map units, same scale as lr_distance)
        # Store raw value in feature map units for consistent magnitude with lr_distance
        right_w = right_box["width"] * input_w * scale_w
        targets["right_width"][0, center_y_int, center_x_int] = right_w

        # ============================================================
        # 3D Components (5 branches)
        # ============================================================

        # 6. Dimensions (normalized offset from class mean)
        # Using normalized offset: (dim - mean) / std for more stable training
        # mean_dims and std_dims are both [L, W, H] in meters
        mean_dim = self.mean_dims.get(class_id, [1.0, 1.0, 1.0])
        std_dim = self.std_dims.get(class_id, [0.2, 0.2, 0.5])
        
        # Compute normalized offsets: (dim - mean) / std
        # decoder expects [ΔH, ΔW, ΔL] order, so we need to reorder from [L, W, H]
        # std_dim is [L, W, H], so:
        #   std_dim[0] = std_L, std_dim[1] = std_W, std_dim[2] = std_H
        #   mean_dim[0] = mean_L, mean_dim[1] = mean_W, mean_dim[2] = mean_H
        dim_offset = [
            (dimensions["height"] - mean_dim[2]) / std_dim[2],   # channel 0 = (H - mean_H) / std_H
            (dimensions["width"] - mean_dim[1]) / std_dim[1],    # channel 1 = (W - mean_W) / std_W
            (dimensions["length"] - mean_dim[0]) / std_dim[0],   # channel 2 = (L - mean_L) / std_L
        ]
        targets["dimensions"][:, center_y_int, center_x_int] = torch.tensor(dim_offset)

        # 7. Orientation (Multi-Bin encoding - Paper Section 3.1)
        orientation_target = self._encode_orientation(alpha)
        targets["orientation"][:, center_y_int, center_x_int] = orientation_target

        # ============================================================
        # 8-10. Vertices (Paper Section 3.1 and Figure 4)
        # Compute 3D position and project bottom vertices
        # ============================================================

        # Get 3D position - either from label or compute from stereo
        if "location_3d" in label and label["location_3d"] is not None:
            # Use provided 3D location
            loc = label["location_3d"]
            x_3d = loc["x"]
            y_3d = loc["y"]
            z_3d = loc["z"]
        else:
            # Compute 3D position from stereo disparity
            # Z = (f × baseline) / disparity
            disparity = lr_dist  # In pixels
            if disparity > 0:
                z_3d = (fx * baseline) / disparity
            else:
                z_3d = 50.0  # Default depth
            
            # X = (u - cx) × Z / fx
            x_3d = (center_x - cx) * z_3d / fx
            # Y = (v - cy) × Z / fy
            y_3d = (center_y - cy) * z_3d / fy

        # Get dimensions in meters
        L = dimensions["length"]  # Length (forward direction)
        W = dimensions["width"]   # Width (lateral direction)
        H = dimensions["height"]  # Height

        # Handle orientation - support both rotation_y (24-value) and alpha (22-value)
        if "rotation_y" in label:
            # 24-value format: rotation_y is global yaw, compute alpha for encoding
            # α = θ - arctan(x/z)
            ray_angle = math.atan2(x_3d, z_3d)
            alpha = rotation_y - ray_angle
            theta = rotation_y  # theta is same as rotation_y
        else:
            # 22-value format: alpha is observation angle, convert to global yaw
            # θ = α + arctan(x/z) - Paper Equation 7
            ray_angle = math.atan2(x_3d, z_3d)
            theta = alpha + ray_angle
        # Normalize to [-π, π]
        theta = math.atan2(math.sin(theta), math.cos(theta))

        # Compute bottom 4 vertices of 3D bounding box (Paper Figure 4)
        # Vertex ordering: 0, 1, 2, 3 as shown in Figure 4
        # Looking from above (bird's eye view):
        #   0 --- 1
        #   |     |
        #   3 --- 2
        # Where the object faces from 3-0 edge toward 2-1 edge (forward direction)
        
        bottom_vertices_3d = self._compute_bottom_vertices_3d(
            x_3d, y_3d, z_3d, L, W, H, theta
        )

        # Project vertices to 2D image plane
        bottom_vertices_2d = self._project_vertices_to_2d(
            bottom_vertices_3d, fx, fy, cx, cy
        )

        # Store vertex targets

        for i, (vx, vy) in enumerate(bottom_vertices_2d):
            # Scale to output feature map size
            vx_out = vx * scale_w
            vy_out = vy * scale_h


            # 8. Vertices: Store as RELATIVE offsets from center (not absolute coordinates)
            # This keeps values bounded and reduces loss magnitude
            # Format: [dx0, dy0, dx1, dy1, dx2, dy2, dx3, dy3] where dx = vx - center_x
            dx = vx_out - center_x_out
            dy = vy_out - center_y_out
            
            # Clip vertex offsets to reasonable bounds to prevent extreme values
            # Maximum offset should be within feature map bounds (allow 1.5x for safety)
            max_offset_x = self.output_w * 1.5
            max_offset_y = self.output_h * 1.5
            dx = max(-max_offset_x, min(dx, max_offset_x))
            dy = max(-max_offset_y, min(dy, max_offset_y))
            
            # Normalize by feature map size to keep values in [-1.5, 1.5] range
            # This makes targets similar in scale to other branches and improves learning
            dx_normalized = dx / self.output_w
            dy_normalized = dy / self.output_h
            
            targets["vertices"][i * 2, center_y_int, center_x_int] = dx_normalized
            targets["vertices"][i * 2 + 1, center_y_int, center_x_int] = dy_normalized

            # 8. Vertex offset: sub-pixel refinement (relative to integer vertex position)
            # Since vertices are normalized, compute offset from normalized integer position
            dx_normalized_int = int(dx_normalized * self.output_w) / self.output_w  # Integer part in normalized space
            dy_normalized_int = int(dy_normalized * self.output_h) / self.output_h
            targets["vertex_offset"][i * 2, center_y_int, center_x_int] = dx_normalized - dx_normalized_int
            targets["vertex_offset"][i * 2 + 1, center_y_int, center_x_int] = dy_normalized - dy_normalized_int

            # 9. Vertex distance: Euclidean distance from center to vertex
            # This helps correlate vertices with their parent center
            # Normalize distance by feature map diagonal to keep values in reasonable range
            max_dist = math.sqrt(self.output_w ** 2 + self.output_h ** 2)  # Diagonal of feature map
            dist = math.sqrt(dx ** 2 + dy ** 2)  # Use original dx, dy for distance calculation
            dist_normalized = min(dist / max_dist, 1.5)  # Normalize and clip to 1.5x for safety
            targets["vertex_dist"][i, center_y_int, center_x_int] = dist_normalized

    def _compute_bottom_vertices_3d(
        self,
        x: float,
        y: float,
        z: float,
        L: float,
        W: float,
        H: float,
        theta: float,
    ) -> list[tuple[float, float, float]]:
        """Compute the 4 bottom vertices of a 3D bounding box in camera coordinates.

        The 3D bounding box center is at (x, y, z) in camera coordinates.
        KITTI convention: y points down, x points right, z points forward.
        The bottom of the box is at y + H/2 (since y points down).

        Vertex ordering (bird's eye view, looking down):
            0 --- 1
            |     |
            3 --- 2
        Where the object's forward direction is from edge 3-0 toward 2-1.

        Args:
            x, y, z: 3D center position in camera coordinates (meters).
            L: Length of the box (forward direction).
            W: Width of the box (lateral direction).
            H: Height of the box.
            theta: Global yaw angle (rotation around y-axis).

        Returns:
            List of 4 vertices as (x, y, z) tuples in camera coordinates.
        """
        # Bottom y coordinate (KITTI: y points down, so bottom is at y + H/2)
        y_bottom = y + H / 2

        # Half dimensions
        half_L = L / 2
        half_W = W / 2

        # Rotation matrix around y-axis (counterclockwise when looking from above)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Local coordinates of the 4 bottom vertices (before rotation)
        # In KITTI convention: x-forward, z-lateral
        # Vertex 0: front-left (-W/2, +L/2 in local)
        # Vertex 1: front-right (+W/2, +L/2 in local)
        # Vertex 2: rear-right (+W/2, -L/2 in local)
        # Vertex 3: rear-left (-W/2, -L/2 in local)
        
        # Actually in KITTI camera coords:
        # x = right, y = down, z = forward
        # Object local: forward = +z, right = +x
        # After rotation by theta around y-axis:
        # x_cam = x_local * cos(theta) + z_local * sin(theta)
        # z_cam = -x_local * sin(theta) + z_local * cos(theta)
        
        # Four corners in local object coordinates (x_local, z_local)
        # Before rotation: length along z (forward), width along x (right)
        local_corners = [
            (-half_W, half_L),   # Vertex 0: left-front
            (half_W, half_L),    # Vertex 1: right-front
            (half_W, -half_L),   # Vertex 2: right-rear
            (-half_W, -half_L),  # Vertex 3: left-rear
        ]

        vertices_3d = []
        for x_local, z_local in local_corners:
            # Rotate around y-axis
            x_rot = x_local * cos_t + z_local * sin_t
            z_rot = -x_local * sin_t + z_local * cos_t
            
            # Translate to world position
            x_world = x + x_rot
            z_world = z + z_rot
            
            vertices_3d.append((x_world, y_bottom, z_world))

        return vertices_3d

    def _project_vertices_to_2d(
        self,
        vertices_3d: list[tuple[float, float, float]],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> list[tuple[float, float]]:
        """Project 3D vertices to 2D image plane using pinhole camera model.

        Args:
            vertices_3d: List of (x, y, z) tuples in camera coordinates.
            fx, fy: Focal lengths in pixels.
            cx, cy: Principal point in pixels.

        Returns:
            List of (u, v) tuples in pixel coordinates.
        """
        vertices_2d = []
        for x, y, z in vertices_3d:
            if z > 0:  # Only project points in front of camera
                u = fx * x / z + cx
                v = fy * y / z + cy
            else:
                # Point is behind camera, use a large default
                u = cx
                v = cy
            vertices_2d.append((u, v))
        return vertices_2d

    def _encode_orientation(self, alpha: float) -> torch.Tensor:
        """Encode orientation angle into Multi-Bin format.

        Multi-Bin encoding (2 bins) as described in Paper Section 3.1:
        - Bin 0: α ∈ [-π, 0], center = -π/2
        - Bin 1: α ∈ [0, π], center = +π/2

        The residual angle is encoded as (sin, cos) for each bin.

        Output format: [conf_0, conf_1, sin_0, cos_0, sin_1, cos_1, pad, pad]

        Args:
            alpha: Observation angle in radians, range [-π, π].

        Returns:
            Orientation encoding tensor of shape [8].
        """
        # Normalize alpha to [-π, π]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))

        # Determine bin based on angle sign
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
        encoding[0] = 1.0 if bin_idx == 0 else 0.0
        encoding[1] = 1.0 if bin_idx == 1 else 0.0

        # Set sin/cos for the active bin only
        sin_val = math.sin(residual)
        cos_val = math.cos(residual)

        if bin_idx == 0:
            encoding[2] = sin_val  # sin_0
            encoding[3] = cos_val  # cos_0
        else:
            encoding[4] = sin_val  # sin_1
            encoding[5] = cos_val  # cos_1

        # Channels 6-7 are padding (remain 0)

        return encoding


def compute_perspective_keypoints(
    bottom_vertices_2d: list[tuple[float, float]],
    theta: float,
) -> list[int]:
    """Determine which bottom vertices are "perspective keypoints".

    From Paper Section 3.2: The perspective keypoints are the vertices
    that can be accurately projected to the middle of the 2D bounding box.
    They are determined by the viewing angle (orientation relative to camera).

    From Paper Figure 4: Different perspectives show different keypoints.
    The perspective keypoints are the ones closest to the camera.

    Args:
        bottom_vertices_2d: List of 4 bottom vertex 2D coordinates.
        theta: Global yaw angle of the object.

    Returns:
        List of vertex indices (0-3) that are perspective keypoints.
    """
    # Normalize theta to [-π, π]
    theta = math.atan2(math.sin(theta), math.cos(theta))
    
    # Based on the orientation, determine which vertices are visible
    # Perspective A (theta ≈ 0): vertices 0, 1 visible (front)
    # Perspective B (theta ≈ π/2): vertices 1, 2 visible (right side)
    # Perspective C (theta ≈ π): vertices 2, 3 visible (rear)
    # Perspective D (theta ≈ -π/2): vertices 3, 0 visible (left side)
    
    # This is a simplified version - actual implementation may need refinement
    if -math.pi / 4 <= theta < math.pi / 4:
        # Front view: vertices 0, 1 visible
        return [0, 1]
    elif math.pi / 4 <= theta < 3 * math.pi / 4:
        # Right view: vertices 1, 2 visible
        return [1, 2]
    elif theta >= 3 * math.pi / 4 or theta < -3 * math.pi / 4:
        # Rear view: vertices 2, 3 visible
        return [2, 3]
    else:
        # Left view: vertices 3, 0 visible
        return [3, 0]