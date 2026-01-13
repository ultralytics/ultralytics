"""
Anchor points definition and management for multi-anchor collision detection.

This module defines anchor points for different object classes and provides
utilities to compute anchor points from bounding boxes.

Author: Cindy
Date: 2025-01-11
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AnchorPoint:
    """Single anchor point definition."""
    name: str                           # e.g., "rear_left", "head"
    position: Tuple[float, float]       # (x, y) in pixel coordinates
    importance: float = 1.0             # 0-1, importance weight
    category: str = "default"           # e.g., "structural", "vulnerability"


class VehicleAnchors:
    """
    Vehicle anchor points definition and generation.
    
    For vehicles (car, truck, bus), we define 8 standard anchor points:
    - front_center, front_left, front_right
    - rear_center, rear_left, rear_right
    - left_center, right_center
    """
    
    # Standard COCO classes
    VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck
    
    @staticmethod
    def get_anchors(bbox_xywh: List[float], 
                    object_class: int) -> Dict[str, Tuple[float, float]]:
        """
        Generate anchor points from bounding box in xywh format.
        
        Args:
            bbox_xywh: [center_x, center_y, width, height]
            object_class: COCO class ID
        
        Returns:
            Dictionary of anchor_name -> (x, y) position
        """
        x, y, w, h = bbox_xywh
        
        if object_class in VehicleAnchors.VEHICLE_CLASSES:
            return {
                'front_center': (x, y + h/2),
                'front_left': (x - w/2, y + h/2),
                'front_right': (x + w/2, y + h/2),
                'rear_center': (x, y - h/2),
                'rear_left': (x - w/2, y - h/2),
                'rear_right': (x + w/2, y - h/2),
                'left_center': (x - w/2, y),
                'right_center': (x + w/2, y),
            }
        elif object_class == 0:  # person
            return PedestrianAnchors.get_anchors(bbox_xywh)
        elif object_class == 1:  # bicycle
            return BicycleAnchors.get_anchors(bbox_xywh)
        elif object_class == 3:  # motorcycle
            return MotorcycleAnchors.get_anchors(bbox_xywh)
        else:
            # Default: use 5 anchor points (center + corners)
            return {
                'center': (x, y),
                'top_left': (x - w/2, y - h/2),
                'top_right': (x + w/2, y - h/2),
                'bottom_left': (x - w/2, y + h/2),
                'bottom_right': (x + w/2, y + h/2),
            }
    
    @staticmethod
    def get_anchor_importance(object_class: int, 
                             anchor_name: str) -> float:
        """
        Get importance weight for a specific anchor point.
        
        Args:
            object_class: COCO class ID
            anchor_name: Name of the anchor point
        
        Returns:
            Importance weight (0-1)
        """
        if object_class in VehicleAnchors.VEHICLE_CLASSES:
            # For vehicles, corners are more important for collision detection
            if 'left' in anchor_name or 'right' in anchor_name:
                return 0.9
            else:
                return 0.8
        elif object_class == 0:  # person
            return PedestrianAnchors.get_importance(anchor_name)
        
        return 1.0


class PedestrianAnchors:
    """Pedestrian (person) anchor points."""
    
    @staticmethod
    def get_anchors(bbox_xywh: List[float]) -> Dict[str, Tuple[float, float]]:
        """
        Generate pedestrian anchor points.
        
        For pedestrians, we define 4 key points:
        - head (top, most vulnerable)
        - torso (main body)
        - lower (legs/lower body)
        - feet (bottom)
        """
        x, y, w, h = bbox_xywh
        
        return {
            'head': (x, y - h/2),           # Top of head
            'torso': (x, y),                # Main body center
            'lower': (x, y + h/3),          # Lower body
            'feet': (x, y + h/2),           # Bottom feet
        }
    
    @staticmethod
    def get_importance(anchor_name: str) -> float:
        """Get importance weight for pedestrian anchor points."""
        importance = {
            'head': 1.0,      # Most vulnerable
            'torso': 0.9,     # Main body
            'lower': 0.8,     # Lower body
            'feet': 0.7,      # Feet
        }
        return importance.get(anchor_name, 0.8)


class BicycleAnchors:
    """Bicycle anchor points."""
    
    @staticmethod
    def get_anchors(bbox_xywh: List[float]) -> Dict[str, Tuple[float, float]]:
        """Generate bicycle anchor points."""
        x, y, w, h = bbox_xywh
        
        return {
            'front': (x, y + h/2),      # Front wheel/handlebars
            'rear': (x, y - h/2),       # Rear wheel
            'left': (x - w/2, y),       # Left side
            'right': (x + w/2, y),      # Right side
            'center': (x, y),           # Center
        }


class MotorcycleAnchors:
    """Motorcycle anchor points."""
    
    @staticmethod
    def get_anchors(bbox_xywh: List[float]) -> Dict[str, Tuple[float, float]]:
        """Generate motorcycle anchor points (similar to bicycle)."""
        x, y, w, h = bbox_xywh
        
        return {
            'front': (x, y + h/2),      # Front
            'rear': (x, y - h/2),       # Rear
            'left': (x - w/2, y),       # Left side
            'right': (x + w/2, y),      # Right side
            'center': (x, y),           # Center
        }


def estimate_heading_from_trajectory(track_points: List[Dict]) -> Tuple[float, str]:
    """
    Estimate vehicle heading from trajectory history.
    
    Args:
        track_points: List of trajectory points with 'center_x', 'center_y' keys
    
    Returns:
        (heading_angle_in_radians, direction_string)
        heading: angle in radians, 0 = pointing right (east)
        direction: 'forward' or 'backward'
    """
    if len(track_points) < 3:
        return 0.0, 'unknown'
    
    # Use recent points to estimate heading
    recent_points = track_points[-5:]
    
    start_x = recent_points[0]['center_x']
    start_y = recent_points[0]['center_y']
    end_x = recent_points[-1]['center_x']
    end_y = recent_points[-1]['center_y']
    
    # Velocity vector
    vx = end_x - start_x
    vy = end_y - start_y
    
    # Handle zero velocity
    if abs(vx) < 0.1 and abs(vy) < 0.1:
        return 0.0, 'stationary'
    
    # Heading angle (radians)
    heading = np.arctan2(vy, vx)
    
    # Determine if moving forward or backward
    # If heading is pointing left or down-left, likely backward
    if heading < -np.pi/2 or heading > np.pi/2:
        direction = 'backward'
    else:
        direction = 'forward'
    
    return heading, direction


def get_vehicle_heading(detection: Dict, 
                       track_history: Optional[List[Dict]] = None) -> Tuple[float, str]:
    """
    Get vehicle heading from multiple sources with priority.
    
    Priority:
    1. If detection contains 'heading' field (from YOLO)
    2. If track_history is available (from trajectory)
    3. Default to 0 (unknown)
    
    Args:
        detection: Detection result dictionary
        track_history: Trajectory history list
    
    Returns:
        (heading_angle_in_radians, source_string)
    """
    
    # Priority 1: YOLO direct output
    if 'heading' in detection:
        return detection['heading'], 'yolo_direct'
    
    # Priority 2: From trajectory
    if track_history and len(track_history) >= 3:
        heading, direction = estimate_heading_from_trajectory(track_history)
        return heading, 'trajectory_inferred'
    
    # Default
    return 0.0, 'unknown'


# Class name mapping
CLASS_NAMES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
}


if __name__ == '__main__':
    # Test anchor generation
    print("Testing anchor point generation:")
    
    # Test vehicle
    vehicle_bbox = [500, 200, 100, 150]
    vehicle_anchors = VehicleAnchors.get_anchors(vehicle_bbox, 2)  # car
    print(f"\nVehicle anchors: {vehicle_anchors}")
    
    # Test pedestrian
    person_bbox = [300, 400, 50, 200]
    person_anchors = VehicleAnchors.get_anchors(person_bbox, 0)  # person
    print(f"\nPerson anchors: {person_anchors}")
    
    # Test heading estimation
    trajectory = [
        {'center_x': 100, 'center_y': 100},
        {'center_x': 110, 'center_y': 105},
        {'center_x': 120, 'center_y': 110},
        {'center_x': 130, 'center_y': 115},
        {'center_x': 140, 'center_y': 120},
    ]
    heading, direction = estimate_heading_from_trajectory(trajectory)
    print(f"\nHeading: {heading:.3f} rad ({np.degrees(heading):.1f}°), Direction: {direction}")
