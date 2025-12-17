#!/usr/bin/env python3
"""Simple KITTI 3D bounding box visualization script.

Usage: python vis_kitti.py 003746
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# KITTI dataset paths (adjust as needed)
KITTI_ROOT = Path("~/datasets/kitti_raw/training").expanduser()
IMAGE_DIR = KITTI_ROOT / "image_2"
LABEL_DIR = KITTI_ROOT / "label_2"
CALIB_DIR = KITTI_ROOT / "calib"

# Colors for different classes (BGR)
COLORS = {
    'Car': (0, 255, 0),        # Green
    'Van': (255, 200, 0),
    'Truck': (0, 100, 255),
    'Pedestrian': (255, 255, 0),
    'Cyclist': (150, 50, 255),
    'DontCare': (128, 128, 128),
}


def load_calibration(calib_file):
    """Load KITTI calibration file."""
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    calib = {}
    for line in lines:
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        calib[key.strip()] = np.array([float(x) for x in value.split()])
    
    # Reshape P2 (left camera projection matrix)
    P2 = calib['P2'].reshape(3, 4)
    return P2


def load_labels(label_file):
    """Load KITTI label file."""
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            
            label = {
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox_2d': [float(x) for x in parts[4:8]],
                'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
                'location': [float(parts[11]), float(parts[12]), float(parts[13])],  # x, y, z
                'rotation_y': float(parts[14])
            }
            labels.append(label)
    return labels


def compute_3d_corners(label):
    """Compute 8 corners of 3D bounding box in camera coordinates."""
    h, w, l = label['dimensions']
    x, y, z = label['location']
    ry = label['rotation_y']
    
    # 8 corners in object coordinate system (KITTI: x-right, y-down, z-forward)
    # Location is at bottom center of box
    corners = np.array([
        [-l/2, 0, w/2],    # 0: front-left-bottom
        [l/2, 0, w/2],     # 1: front-right-bottom
        [l/2, 0, -w/2],    # 2: back-right-bottom
        [-l/2, 0, -w/2],   # 3: back-left-bottom
        [-l/2, -h, w/2],   # 4: front-left-top
        [l/2, -h, w/2],    # 5: front-right-top
        [l/2, -h, -w/2],   # 6: back-right-top
        [-l/2, -h, -w/2]   # 7: back-left-top
    ])
    
    # Rotation matrix around Y-axis
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # Rotate and translate to camera coordinates
    corners_3d = corners @ R.T + np.array([x, y, z])
    return corners_3d


def project_to_2d(corners_3d, P):
    """Project 3D points to 2D using projection matrix."""
    # Add homogeneous coordinate
    corners_homo = np.hstack([corners_3d, np.ones((corners_3d.shape[0], 1))])
    
    # Project
    corners_2d_homo = (P @ corners_homo.T).T
    
    # Normalize
    corners_2d = corners_2d_homo[:, :2] / corners_2d_homo[:, 2:3]
    return corners_2d


def draw_3d_box(img, corners_2d, color, thickness=2):
    """Draw 3D bounding box on image."""
    corners_2d = corners_2d.astype(int)
    
    # 12 edges of the box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    for edge in edges:
        pt1 = tuple(corners_2d[edge[0]])
        pt2 = tuple(corners_2d[edge[1]])
        cv2.line(img, pt1, pt2, color, thickness)
    
    # Highlight front face (edges 0-1, 4-5, 0-4, 1-5)
    front_edges = [[0, 1], [4, 5], [0, 4], [1, 5]]
    for edge in front_edges:
        pt1 = tuple(corners_2d[edge[0]])
        pt2 = tuple(corners_2d[edge[1]])
        cv2.line(img, pt1, pt2, color, thickness + 1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python vis_kitti.py <image_id>")
        print("Example: python vis_kitti.py 003746")
        sys.exit(1)
    
    image_id = sys.argv[1]
    
    # Load files
    img_file = IMAGE_DIR / f"{image_id}.png"
    label_file = LABEL_DIR / f"{image_id}.txt"
    calib_file = CALIB_DIR / f"{image_id}.txt"
    
    if not img_file.exists():
        print(f"Error: Image not found: {img_file}")
        sys.exit(1)
    if not label_file.exists():
        print(f"Error: Label not found: {label_file}")
        sys.exit(1)
    if not calib_file.exists():
        print(f"Error: Calibration not found: {calib_file}")
        sys.exit(1)
    
    # Load data
    img = cv2.imread(str(img_file))
    labels = load_labels(label_file)
    P2 = load_calibration(calib_file)
    
    print(f"Image: {img_file}")
    print(f"Found {len(labels)} objects")
    
    # Draw each object
    for label in labels:
        obj_type = label['type']
        if obj_type == 'DontCare':
            continue
        
        color = COLORS.get(obj_type, (255, 255, 255))
        
        # Compute 3D corners and project to 2D
        corners_3d = compute_3d_corners(label)
        corners_2d = project_to_2d(corners_3d, P2)
        
        # Draw 3D box
        draw_3d_box(img, corners_2d, color)
        
        # Draw label
        x1, y1 = int(corners_2d[0, 0]), int(corners_2d[0, 1])
        cv2.putText(img, obj_type, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print(f"  {obj_type}: pos=({label['location'][0]:.1f}, {label['location'][1]:.1f}, {label['location'][2]:.1f}), "
              f"rot_y={np.degrees(label['rotation_y']):.1f}°")
    
    # Save result
    output_file = Path(f"vis_{image_id}.jpg")
    cv2.imwrite(str(output_file), img)
    print(f"\n✓ Saved: {output_file}")


if __name__ == "__main__":
    main()

