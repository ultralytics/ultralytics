"""
Ground Truth KITTI 3D Bounding Box Visualization
================================================
This script reads original KITTI format labels and creates ground truth visualizations.
Use this as reference to verify your implementation is correct.

KITTI Label Format (15 values):
- Type: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare
- Truncated: Float 0 (non-truncated) to 1 (truncated)
- Occluded: Integer 0 (fully visible), 1 (partly occluded), 2 (largely occluded), 3 (unknown)
- Alpha: Observation angle [-pi, pi]
- 2D bbox: left, top, right, bottom (pixels)
- 3D dimensions: height, width, length (meters)
- 3D location: x, y, z in camera coordinates (meters)
- Rotation_y: Rotation around Y-axis in camera coordinates [-pi, pi]
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
import os

# Set matplotlib to use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no display required)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class KITTIVisualizer:
    """Ground truth KITTI 3D bounding box visualizer."""
    
    # Class colors (BGR format for OpenCV)
    COLORS = {
        'Car': (0, 255, 0),           # Green
        'Van': (255, 200, 0),          # Cyan-ish
        'Truck': (0, 100, 255),        # Orange
        'Pedestrian': (255, 255, 0),   # Yellow
        'Person_sitting': (255, 150, 150), # Pink
        'Cyclist': (150, 50, 255),     # Purple
        'Tram': (100, 100, 100),       # Gray
        'Misc': (200, 200, 200),       # Light gray
        'DontCare': (50, 50, 50),      # Dark gray
    }
    
    @staticmethod
    def bgr_to_rgb(bgr_color):
        """Convert BGR color to RGB for matplotlib."""
        return tuple(c / 255.0 for c in reversed(bgr_color))
    
    def __init__(self, calib_file):
        """Initialize with calibration data."""
        self.calib = self.load_calibration(calib_file)
        
    def load_calibration(self, calib_file):
        """Load KITTI calibration data."""
        calib = {}
        with open(calib_file, 'r') as f:
            for line in f:
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()])
                
        # Reshape matrices
        calib_out = {}
        calib_out['P0'] = calib['P0'].reshape(3, 4)
        calib_out['P1'] = calib['P1'].reshape(3, 4)
        calib_out['P2'] = calib['P2'].reshape(3, 4)  # Left camera
        calib_out['P3'] = calib['P3'].reshape(3, 4)  # Right camera
        calib_out['R0_rect'] = calib['R0_rect'].reshape(3, 3)
        
        if 'Tr_velo_to_cam' in calib:
            calib_out['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
        if 'Tr_imu_to_velo' in calib:
            calib_out['Tr_imu_to_velo'] = calib['Tr_imu_to_velo'].reshape(3, 4)
            
        return calib_out
    
    def load_label(self, label_file):
        """Load KITTI format labels."""
        labels = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                
                # Skip if not enough values (need at least 15 for 3D)
                if len(parts) < 15:
                    continue
                
                # Parse KITTI format
                label = {
                    'type': parts[0],
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox_2d': np.array([float(parts[4]), float(parts[5]), 
                                         float(parts[6]), float(parts[7])]),
                    'dimensions': np.array([float(parts[8]), float(parts[9]), 
                                           float(parts[10])]),  # h, w, l
                    'location': np.array([float(parts[11]), float(parts[12]), 
                                         float(parts[13])]),  # x, y, z
                    'rotation_y': float(parts[14])
                }
                
                # Skip DontCare objects for cleaner visualization
                if label['type'] != 'DontCare':
                    labels.append(label)
                    
        return labels
    
    def project_3d_to_2d(self, pts_3d, P):
        """Project 3D points to 2D image plane.
        
        Args:
            pts_3d: (N, 3) 3D points in camera coordinates
            P: (3, 4) Projection matrix
            
        Returns:
            pts_2d: (N, 2) 2D points in image coordinates
        """
        pts_3d_homo = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])
        pts_2d_homo = np.dot(P, pts_3d_homo.T).T
        pts_2d = pts_2d_homo[:, :2] / pts_2d_homo[:, 2:3]
        return pts_2d
    
    def compute_3d_box_corners(self, label):
        """Compute 3D bounding box corners in camera coordinates.
        
        Args:
            label: Dict with 'dimensions', 'location', 'rotation_y'
            
        Returns:
            corners_3d: (8, 3) array of 3D corners
            Corner ordering (KITTI convention):
            - Bottom face: 0, 1, 2, 3 (front-left, front-right, back-right, back-left)
            - Top face: 4, 5, 6, 7 (front-left, front-right, back-right, back-left)
        """
        # Extract parameters
        h, w, l = label['dimensions']  # height, width, length
        x, y, z = label['location']  # center of bottom face
        ry = label['rotation_y']
        
        # Create 3D bounding box corners in object coordinate system
        # KITTI convention: X (right), Y (down), Z (forward)
        # Box center is at bottom center
        # Corner ordering:
        # 0: front-left-bottom  (x=-l/2, y=0, z=w/2)
        # 1: front-right-bottom (x=l/2, y=0, z=w/2)
        # 2: back-right-bottom  (x=l/2, y=0, z=-w/2)
        # 3: back-left-bottom   (x=-l/2, y=0, z=-w/2)
        # 4: front-left-top     (x=-l/2, y=-h, z=w/2)
        # 5: front-right-top    (x=l/2, y=-h, z=w/2)
        # 6: back-right-top     (x=l/2, y=-h, z=-w/2)
        # 7: back-left-top      (x=-l/2, y=-h, z=-w/2)
        
        corners = np.array([
            [-l/2, 0, w/2],   # 0: front-left-bottom
            [l/2, 0, w/2],    # 1: front-right-bottom
            [l/2, 0, -w/2],   # 2: back-right-bottom
            [-l/2, 0, -w/2],  # 3: back-left-bottom
            [-l/2, -h, w/2],  # 4: front-left-top
            [l/2, -h, w/2],   # 5: front-right-top
            [l/2, -h, -w/2],  # 6: back-right-top
            [-l/2, -h, -w/2]  # 7: back-left-top
        ])
        
        # Rotation matrix around Y-axis (camera coordinates)
        # In KITTI: rotation_y rotates around Y-axis (vertical axis)
        cos_ry = np.cos(ry)
        sin_ry = np.sin(ry)
        R = np.array([
            [cos_ry, 0, sin_ry],
            [0, 1, 0],
            [-sin_ry, 0, cos_ry]
        ])
        
        # Rotate corners
        corners_rotated = corners @ R.T  # Apply rotation
        
        # Translate to camera coordinates
        corners_3d = corners_rotated + np.array([x, y, z])
        
        return corners_3d
    
    def draw_3d_box(self, img, corners_2d, color, thickness=2, draw_face=True, 
                    colored_vertices=False):
        """Draw 3D bounding box on image.
        
        Args:
            img: Input image
            corners_2d: (8, 2) 2D corners
            color: BGR color tuple
            thickness: Line thickness
            draw_face: Whether to highlight front face
            colored_vertices: If True, draw each vertex with different color
        """
        corners_2d = corners_2d.astype(int)
        
        # Define 8 distinct colors for vertices (BGR format)
        vertex_colors = [
            (255, 0, 0),      # 0: Red - front-left-bottom
            (0, 255, 0),      # 1: Green - front-right-bottom
            (0, 0, 255),      # 2: Blue - back-right-bottom
            (255, 255, 0),    # 3: Cyan - back-left-bottom
            (255, 0, 255),   # 4: Magenta - front-left-top
            (0, 255, 255),   # 5: Yellow - front-right-top
            (128, 0, 128),   # 6: Purple - back-right-top
            (255, 165, 0),   # 7: Orange - back-left-top
        ]
        
        # Define the 12 edges of a 3D box
        edges = [
            # Bottom face
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        if colored_vertices:
            # Draw edges with colors based on vertices
            for edge in edges:
                pt1 = tuple(corners_2d[edge[0]])
                pt2 = tuple(corners_2d[edge[1]])
                # Use average color of the two vertices
                color1 = vertex_colors[edge[0]]
                color2 = vertex_colors[edge[1]]
                edge_color = tuple((c1 + c2) // 2 for c1, c2 in zip(color1, color2))
                cv2.line(img, pt1, pt2, edge_color, thickness)
            
            # Vertex labels for identification
            vertex_labels = ['0', '1', '2', '3', '4', '5', '6', '7']
            
            # Draw vertices with distinct colors
            vertex_radius = 6
            for i, corner in enumerate(corners_2d):
                # Draw filled circle
                cv2.circle(img, tuple(corner), vertex_radius, vertex_colors[i], -1)
                # Draw a black border for visibility
                cv2.circle(img, tuple(corner), vertex_radius, (0, 0, 0), 2)
                # Draw vertex number label with background
                label_pos = (corner[0] + vertex_radius + 2, corner[1])
                text_size = cv2.getTextSize(vertex_labels[i], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                # Draw background rectangle
                cv2.rectangle(img, 
                             (label_pos[0] - 2, label_pos[1] - text_size[1] - 2),
                             (label_pos[0] + text_size[0] + 2, label_pos[1] + 2),
                             (255, 255, 255), -1)
                cv2.putText(img, vertex_labels[i], label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        else:
            # Original drawing method
            # Draw edges
            for i, edge in enumerate(edges):
                pt1 = tuple(corners_2d[edge[0]])
                pt2 = tuple(corners_2d[edge[1]])
                
                # Front face edges (first 4 edges) drawn thicker
                if i < 4 and draw_face:
                    cv2.line(img, pt1, pt2, color, thickness + 1)
                else:
                    cv2.line(img, pt1, pt2, color, thickness)
        
        # Optionally draw front face with transparency
        if draw_face and not colored_vertices:
            front_face = corners_2d[[0, 1, 2, 3]]
            overlay = img.copy()
            cv2.fillPoly(overlay, [front_face], color)
            cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)
            
        return img
    
    def draw_2d_box(self, img, bbox, color, thickness=2):
        """Draw 2D bounding box."""
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img
    
    def draw_text(self, img, text, position, color, scale=0.5):
        """Draw text with background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        # Draw background rectangle
        x, y = position
        cv2.rectangle(img, 
                     (x, y - text_height - 4),
                     (x + text_width, y + baseline),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(img, text, position, font, scale, color, thickness)
        return img
    
    def visualize(self, img_left, img_right, labels, draw_2d=True, draw_3d=True, 
                  draw_text_labels=True, camera='left', colored_vertices=False):
        """Visualize 3D bounding boxes on stereo images.
        
        Args:
            img_left: Left camera image
            img_right: Right camera image
            labels: List of label dicts
            draw_2d: Whether to draw 2D boxes
            draw_3d: Whether to draw 3D boxes
            draw_text_labels: Whether to draw text labels
            camera: Which camera to visualize ('left', 'right', or 'both')
            colored_vertices: If True, draw each vertex with different color
            
        Returns:
            Dictionary with visualization results
        """
        results = {}
        
        # Process left image
        if camera in ['left', 'both']:
            img_left_vis = img_left.copy()
            P = self.calib['P2']  # Left camera projection matrix
            
            for label in labels:
                color = self.COLORS.get(label['type'], (255, 255, 255))
                
                # Draw 2D box
                if draw_2d:
                    self.draw_2d_box(img_left_vis, label['bbox_2d'], color, 1)
                
                # Draw 3D box
                if draw_3d:
                    corners_3d = self.compute_3d_box_corners(label)
                    
                    # Only draw if in front of camera
                    if np.all(corners_3d[:, 2] > 0):
                        corners_2d = self.project_3d_to_2d(corners_3d, P)
                        self.draw_3d_box(img_left_vis, corners_2d, color, 2, 
                                         draw_face=not colored_vertices, 
                                         colored_vertices=colored_vertices)
                
                # Draw text label
                if draw_text_labels:
                    x1, y1 = label['bbox_2d'][:2].astype(int)
                    depth = label['location'][2]
                    text = f"{label['type']}: {depth:.1f}m"
                    self.draw_text(img_left_vis, text, (x1, y1 - 5), color)
            
            results['left'] = img_left_vis
        
        # Process right image
        if camera in ['right', 'both']:
            img_right_vis = img_right.copy()
            P = self.calib['P3']  # Right camera projection matrix
            
            for label in labels:
                color = self.COLORS.get(label['type'], (255, 255, 255))
                
                # For right image, we need to project 3D box
                if draw_3d:
                    corners_3d = self.compute_3d_box_corners(label)
                    
                    # Only draw if in front of camera
                    if np.all(corners_3d[:, 2] > 0):
                        corners_2d = self.project_3d_to_2d(corners_3d, P)
                        self.draw_3d_box(img_right_vis, corners_2d, color, 2,
                                         draw_face=not colored_vertices,
                                         colored_vertices=colored_vertices)
                        
                        # Draw 2D box from projected corners
                        if draw_2d:
                            x_coords = corners_2d[:, 0]
                            y_coords = corners_2d[:, 1]
                            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                            cv2.rectangle(img_right_vis, (x1, y1), (x2, y2), color, 1)
                            
                            if draw_text_labels:
                                depth = label['location'][2]
                                text = f"{label['type']}: {depth:.1f}m"
                                self.draw_text(img_right_vis, text, (x1, y1 - 5), color)
            
            results['right'] = img_right_vis
        
        return results
    
    def create_bird_eye_view(self, labels, img_width=600, img_height=600, 
                            x_range=(-20, 20), z_range=(0, 40)):
        """Create bird's eye view visualization.
        
        Args:
            labels: List of label dicts
            img_width: Output image width
            img_height: Output image height
            x_range: Range of x coordinates to visualize (left-right)
            z_range: Range of z coordinates to visualize (forward)
            
        Returns:
            Bird's eye view image
        """
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # Draw grid
        for i in range(0, img_width, 50):
            cv2.line(img, (i, 0), (i, img_height), (200, 200, 200), 1)
        for i in range(0, img_height, 50):
            cv2.line(img, (0, i), (img_width, i), (200, 200, 200), 1)
        
        # Scale factors
        x_scale = img_width / (x_range[1] - x_range[0])
        z_scale = img_height / (z_range[1] - z_range[0])
        
        # Draw objects
        for label in labels:
            color = self.COLORS.get(label['type'], (255, 255, 255))
            
            # Get 3D box corners
            corners_3d = self.compute_3d_box_corners(label)
            
            # Project to bird's eye view (use x and z coordinates)
            corners_bev = []
            for corner in corners_3d:
                x = int((corner[0] - x_range[0]) * x_scale)
                z = int((z_range[1] - corner[2]) * z_scale)  # Flip z axis
                corners_bev.append([x, z])
            
            corners_bev = np.array(corners_bev)
            
            # Draw box (bottom 4 corners)
            bottom_corners = corners_bev[[0, 1, 2, 3]]
            cv2.fillPoly(img, [bottom_corners], color)
            cv2.polylines(img, [bottom_corners], True, (0, 0, 0), 2)
            
            # Draw orientation
            center = np.mean(bottom_corners, axis=0).astype(int)
            front_center = np.mean(bottom_corners[[0, 1]], axis=0).astype(int)
            cv2.arrowedLine(img, tuple(center), tuple(front_center), (0, 0, 0), 2)
        
        # Add text
        cv2.putText(img, "Bird's Eye View", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, f"X: {x_range[0]}m to {x_range[1]}m", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, f"Z: {z_range[0]}m to {z_range[1]}m", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img
    
    def create_3d_visualization(self, labels, figsize=(12, 10), 
                                x_range=(-30, 30), y_range=(-5, 5), z_range=(0, 80)):
        """Create interactive 3D visualization of bounding boxes.
        
        Args:
            labels: List of label dicts
            figsize: Figure size tuple
            x_range: Range of x coordinates (left-right)
            y_range: Range of y coordinates (up-down)
            z_range: Range of z coordinates (forward)
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw coordinate axes
        ax.plot([0, 5], [0, 0], [0, 0], 'r-', linewidth=2, label='X (right)')
        ax.plot([0, 0], [0, 5], [0, 0], 'g-', linewidth=2, label='Y (up)')
        ax.plot([0, 0], [0, 0], [0, 5], 'b-', linewidth=2, label='Z (forward)')
        
        # Draw camera position
        ax.scatter([0], [0], [0], c='red', s=100, marker='^', label='Camera')
        
        # Draw each bounding box
        for label in labels:
            color_rgb = self.bgr_to_rgb(self.COLORS.get(label['type'], (255, 255, 255)))
            
            # Get 3D box corners
            corners_3d = self.compute_3d_box_corners(label)
            
            # Define faces of the box (6 faces)
            faces = [
                [0, 1, 2, 3],  # Bottom face
                [4, 5, 6, 7],  # Top face
                [0, 1, 5, 4],  # Front face
                [2, 3, 7, 6],  # Back face
                [0, 3, 7, 4],  # Left face
                [1, 2, 6, 5]   # Right face
            ]
            
            # Draw wireframe edges
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
            ]
            
            # Draw edges
            for edge in edges:
                ax.plot3D(
                    [corners_3d[edge[0], 0], corners_3d[edge[1], 0]],
                    [corners_3d[edge[0], 1], corners_3d[edge[1], 1]],
                    [corners_3d[edge[0], 2], corners_3d[edge[1], 2]],
                    color=color_rgb, linewidth=2
                )
            
            # Draw semi-transparent faces
            for face in faces:
                vertices = corners_3d[face]
                poly = Poly3DCollection([vertices], alpha=0.2, facecolor=color_rgb, 
                                       edgecolor=color_rgb, linewidths=1)
                ax.add_collection3d(poly)
            
            # Draw front direction arrow to show rotation
            # Front face center (average of front corners: 0, 1, 4, 5)
            front_center = np.mean(corners_3d[[0, 1, 4, 5]], axis=0)
            # Bottom center of front face
            front_bottom = np.mean(corners_3d[[0, 1]], axis=0)
            # Draw arrow from bottom center to front center
            center_bottom = np.mean(corners_3d[[0, 1, 2, 3]], axis=0)
            arrow_length = np.linalg.norm(front_center - center_bottom) * 0.5
            direction = (front_center - center_bottom)
            direction = direction / np.linalg.norm(direction) * arrow_length
            
            ax.quiver(center_bottom[0], center_bottom[1], center_bottom[2],
                     direction[0], direction[1], direction[2],
                     color='red', arrow_length_ratio=0.3, linewidth=2, alpha=0.8)
            
            # Draw label text at center
            center = np.mean(corners_3d, axis=0)
            ax.text(center[0], center[1], center[2], 
                   f"{label['type']}\n{label['location'][2]:.1f}m\nry={np.degrees(label['rotation_y']):.1f}°",
                   fontsize=8, color=color_rgb)
        
        # Set axis labels and limits
        ax.set_xlabel('X (meters) - Right', fontsize=12)
        ax.set_ylabel('Y (meters) - Up', fontsize=12)
        ax.set_zlabel('Z (meters) - Forward', fontsize=12)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        
        # Set title
        ax.set_title('3D Bounding Box Visualization (Camera Coordinates)', fontsize=14, fontweight='bold')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        return fig


def main():
    """Main function for ground truth visualization."""
    parser = argparse.ArgumentParser(
        description="Ground Truth KITTI 3D Bounding Box Visualizer"
    )
    parser.add_argument('--data-dir', type=str, required=True,
                       help='KITTI dataset directory')
    parser.add_argument('--sample', type=str, default='000001',
                       help='Sample ID to visualize')
    parser.add_argument('--split', type=str, default='training',
                       choices=['training', 'testing'],
                       help='Dataset split')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    parser.add_argument('--show', action='store_true',
                       help='Display visualization')
    parser.add_argument('--camera', type=str, default='left',
                       choices=['left', 'right', 'both'],
                       help='Which camera view to visualize')
    parser.add_argument('--bev', action='store_true',
                       help='Also create bird eye view')
    parser.add_argument('--3d', dest='vis_3d', action='store_true',
                       help='Create interactive 3D visualization')
    parser.add_argument('--colored-vertices', action='store_true',
                       help='Draw each 3D box vertex with different color')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    
    # Standard KITTI directory structure
    if args.split == 'training':
        img_dir = data_dir / 'training' / 'image_2'
        img_dir_right = data_dir / 'training' / 'image_3'
        label_dir = data_dir / 'training' / 'label_2'
        calib_dir = data_dir / 'training' / 'calib'
    else:
        img_dir = data_dir / 'testing' / 'image_2'
        img_dir_right = data_dir / 'testing' / 'image_3'
        calib_dir = data_dir / 'testing' / 'calib'
        label_dir = None  # No labels for test set
    
    # Load data
    img_file = img_dir / f"{args.sample}.png"
    img_file_right = img_dir_right / f"{args.sample}.png"
    calib_file = calib_dir / f"{args.sample}.txt"
    
    print(f"Loading sample {args.sample}...")
    print(f"  Left image: {img_file}")
    print(f"  Right image: {img_file_right}")
    print(f"  Calibration: {calib_file}")
    
    # Check files exist
    if not img_file.exists():
        print(f"Error: Image file not found: {img_file}")
        return
    if not calib_file.exists():
        print(f"Error: Calibration file not found: {calib_file}")
        return
    
    # Load images
    img_left = cv2.imread(str(img_file))
    img_right = cv2.imread(str(img_file_right))
    
    # Initialize visualizer
    visualizer = KITTIVisualizer(calib_file)
    
    # Load labels if available
    if label_dir:
        label_file = label_dir / f"{args.sample}.txt"
        print(f"  Label file: {label_file}")
        
        if label_file.exists():
            labels = visualizer.load_label(label_file)
            print(f"  Found {len(labels)} objects")
            
            # Print label summary
            for i, label in enumerate(labels):
                print(f"    {i+1}. {label['type']}: "
                      f"2D=[{label['bbox_2d'][0]:.1f}, {label['bbox_2d'][1]:.1f}, "
                      f"{label['bbox_2d'][2]:.1f}, {label['bbox_2d'][3]:.1f}], "
                      f"3D_loc=[{label['location'][0]:.2f}, {label['location'][1]:.2f}, "
                      f"{label['location'][2]:.2f}], "
                      f"depth={label['location'][2]:.1f}m")
        else:
            print(f"Warning: Label file not found: {label_file}")
            labels = []
    else:
        labels = []
    
    # Visualize
    print("\nCreating visualization...")
    results = visualizer.visualize(
        img_left, img_right, labels,
        draw_2d=True, draw_3d=True, draw_text_labels=True,
        camera=args.camera, colored_vertices=args.colored_vertices
    )
    
    # Create bird's eye view if requested
    if args.bev:
        bev = visualizer.create_bird_eye_view(labels)
        results['bev'] = bev
    
    # Create 3D visualization if requested
    if args.vis_3d:
        print("Creating 3D visualization...")
        fig_3d = visualizer.create_3d_visualization(labels)
        results['3d'] = fig_3d
    
    # Save or display results
    if args.output:
        if args.camera == 'both':
            # Stack horizontally
            combined = np.hstack([results['left'], results['right']])
            cv2.imwrite(args.output, combined)
            print(f"Saved combined view to: {args.output}")
        else:
            cv2.imwrite(args.output, results[args.camera])
            print(f"Saved {args.camera} view to: {args.output}")
        
        if args.bev:
            bev_output = args.output.replace('.', '_bev.')
            cv2.imwrite(bev_output, results['bev'])
            print(f"Saved bird's eye view to: {bev_output}")
        
        if args.vis_3d:
            fig_3d_output = args.output.replace('.', '_3d.')
            if fig_3d_output.endswith('.png'):
                fig_3d_output = fig_3d_output
            else:
                fig_3d_output = args.output + '_3d.png'
            results['3d'].savefig(fig_3d_output, dpi=150, bbox_inches='tight')
            plt.close(results['3d'])  # Close figure to free memory
            print(f"Saved 3D visualization to: {fig_3d_output}")
    elif args.vis_3d:
        # Save 3D visualization even if no output specified
        fig_3d_output = f"kitti_3d_vis_{args.sample}.png"
        results['3d'].savefig(fig_3d_output, dpi=150, bbox_inches='tight')
        plt.close(results['3d'])  # Close figure to free memory
        print(f"Saved 3D visualization to: {fig_3d_output}")
    
    if args.show:
        if args.camera == 'both':
            cv2.imshow('Left Camera', results['left'])
            cv2.imshow('Right Camera', results['right'])
        else:
            cv2.imshow(f'{args.camera.title()} Camera', results[args.camera])
        
        if args.bev:
            cv2.imshow("Bird's Eye View", results['bev'])
        
        if args.vis_3d and '3d' in results:
            # In headless environments, plt.show() won't work with 'Agg' backend
            # The 'Agg' backend doesn't support interactive display
            print("Note: 3D visualization cannot be displayed interactively in headless mode.")
            print("The visualization has been saved to file.")
            # Close the figure if it's still open
            try:
                plt.close(results['3d'])
            except:
                pass
        
        if not args.vis_3d:  # Only use cv2.waitKey if not showing matplotlib
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("\nDone!")


if __name__ == '__main__':
    main()