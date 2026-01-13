"""
Multi-anchor collision analyzer for advanced collision detection.

This module implements collision risk analysis using multiple anchor points
on objects, providing more accurate collision detection than center-point
distance alone.

Author: Cindy
Date: 2025-01-11
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from anchor_points import VehicleAnchors, get_vehicle_heading, estimate_heading_from_trajectory

# Optional cv2 import
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class CollisionResult:
    """Result of collision analysis between two objects."""
    
    min_distance: float                 # Minimum distance in meters
    min_distance_px: float              # Minimum distance in pixels
    object1_part: str                   # Name of closest part on object 1
    object2_part: str                   # Name of closest part on object 2
    point1_px: Tuple[float, float]      # Position of point 1 in pixels
    point2_px: Tuple[float, float]      # Position of point 2 in pixels
    point1_world: Tuple[float, float] = (0, 0)  # Position of point 1 in world coords
    point2_world: Tuple[float, float] = (0, 0)  # Position of point 2 in world coords
    
    relative_heading: float = 0.0       # Relative heading angle in radians
    approach_vector: Tuple[float, float] = (0, 0)  # Direction vector from p1 to p2
    approaching: bool = False           # Whether objects are approaching
    ttc: Optional[float] = None         # Time to collision in seconds
    risk_level: str = 'LOW'             # Risk level: CRITICAL/HIGH/MEDIUM/LOW
    
    description: str = ""               # Human readable description
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        def convert_to_native(value):
            """Convert numpy types to Python native types."""
            if hasattr(value, 'item'):  # numpy scalar
                return value.item()
            elif isinstance(value, (list, tuple)):
                return [convert_to_native(v) for v in value]
            elif isinstance(value, dict):
                return {k: convert_to_native(v) for k, v in value.items()}
            else:
                return value
        
        return {
            'min_distance_meters': float(self.min_distance),
            'min_distance_pixel': float(self.min_distance_px),
            'closest_parts': {
                'object1_part': self.object1_part,
                'object2_part': self.object2_part,
                'point1_px': [float(p) for p in self.point1_px],
                'point2_px': [float(p) for p in self.point2_px],
                'point1_world': [float(p) for p in self.point1_world],
                'point2_world': [float(p) for p in self.point2_world],
                'description': f"{self.object1_part} ↔ {self.object2_part}",
            },
            'heading_analysis': {
                'relative_heading_rad': float(self.relative_heading),
                'relative_heading_deg': float(np.degrees(self.relative_heading)),
                'approaching': bool(self.approaching),
            },
            'ttc_seconds': float(self.ttc) if self.ttc is not None else None,
            'risk_level': self.risk_level,
            'description': self.description,
        }


class CollisionAnalyzer:
    """
    Analyze collision risk between two objects using multi-anchor points.
    
    This analyzer computes distances between all anchor point pairs and
    identifies the closest parts, providing more precise collision detection
    than center-point distance alone.
    """
    
    def __init__(self, pixel_per_meter: float = 60.0):
        """
        Initialize collision analyzer.
        
        Args:
            pixel_per_meter: Conversion factor from pixels to meters
        """
        self.pixel_per_meter = pixel_per_meter
    
    def analyze(self,
               obj1: Dict,
               obj2: Dict,
               obj1_anchors: Dict[str, Tuple[float, float]],
               obj2_anchors: Dict[str, Tuple[float, float]],
               obj1_velocity: Tuple[float, float],
               obj2_velocity: Tuple[float, float],
               obj1_track: Optional[List[Dict]] = None,
               obj2_track: Optional[List[Dict]] = None,
               H: Optional[np.ndarray] = None) -> CollisionResult:
        """
        Analyze collision risk between two objects.
        
        Args:
            obj1, obj2: Detection dictionaries with 'class', 'bbox_xywh' etc.
            obj1_anchors, obj2_anchors: Anchor points {name: (x, y)}
            obj1_velocity, obj2_velocity: Velocity in m/s (vx, vy)
            obj1_track, obj2_track: Trajectory history (optional)
            H: Homography matrix for pixel-to-world transformation (optional)
        
        Returns:
            CollisionResult object with analysis
        """
        
        # Step 1: Find minimum distance between anchor points
        min_dist_px = float('inf')
        closest_pair = None
        
        for part1_name, pt1_px in obj1_anchors.items():
            for part2_name, pt2_px in obj2_anchors.items():
                dist_px = np.sqrt((pt2_px[0] - pt1_px[0])**2 + 
                                 (pt2_px[1] - pt1_px[1])**2)
                
                if dist_px < min_dist_px:
                    min_dist_px = dist_px
                    closest_pair = {
                        'part1': part1_name,
                        'part2': part2_name,
                        'point1_px': pt1_px,
                        'point2_px': pt2_px,
                    }
        
        # Step 2: Convert pixel distance to world distance
        min_dist_m = min_dist_px / self.pixel_per_meter
        
        # Step 3: Transform anchor points to world coordinates (if homography available)
        point1_world = closest_pair['point1_px']
        point2_world = closest_pair['point2_px']
        
        if H is not None and HAS_CV2:
            try:
                pts_px = np.array([closest_pair['point1_px']], dtype=np.float32)
                pts_world = cv2.perspectiveTransform(pts_px.reshape(1, 1, 2), H)
                point1_world = tuple(pts_world[0, 0])
                
                pts_px = np.array([closest_pair['point2_px']], dtype=np.float32)
                pts_world = cv2.perspectiveTransform(pts_px.reshape(1, 1, 2), H)
                point2_world = tuple(pts_world[0, 0])
                
                # Recalculate world distance from transformed points
                min_dist_m = np.sqrt((point2_world[0] - point1_world[0])**2 + 
                                    (point2_world[1] - point1_world[1])**2)
            except Exception:
                # If transformation fails, use pixel-based distance
                pass
        
        # Step 4: Estimate headings and relative orientation
        heading1, _ = get_vehicle_heading(obj1, obj1_track)
        heading2, _ = get_vehicle_heading(obj2, obj2_track)
        relative_heading = heading1 - heading2
        
        # Normalize to [-pi, pi]
        while relative_heading > np.pi:
            relative_heading -= 2 * np.pi
        while relative_heading < -np.pi:
            relative_heading += 2 * np.pi
        
        # Step 5: Calculate approach vector and approaching status
        approach_vector = (closest_pair['point2_px'][0] - closest_pair['point1_px'][0],
                          closest_pair['point2_px'][1] - closest_pair['point1_px'][1])
        
        ttc, approaching = self._calculate_ttc(
            obj1_velocity, obj2_velocity,
            approach_vector, min_dist_m
        )
        
        # Step 6: Assess risk level
        risk_level = self._assess_risk(
            min_dist_m, ttc, relative_heading, approaching
        )
        
        # Step 7: Generate description
        class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'
        }
        class1 = class_names.get(obj1['class'], f"class_{obj1['class']}")
        class2 = class_names.get(obj2['class'], f"class_{obj2['class']}")
        
        description = (f"{class1}的{closest_pair['part1']} ↔ "
                      f"{class2}的{closest_pair['part2']}，"
                      f"距离{min_dist_m:.2f}m，"
                      f"风险等级{risk_level}")
        
        if ttc is not None and ttc < 10:
            description += f"，预计{ttc:.1f}秒后碰撞"
        
        # Create result object
        result = CollisionResult(
            min_distance=min_dist_m,
            min_distance_px=min_dist_px,
            object1_part=closest_pair['part1'],
            object2_part=closest_pair['part2'],
            point1_px=closest_pair['point1_px'],
            point2_px=closest_pair['point2_px'],
            point1_world=point1_world,
            point2_world=point2_world,
            relative_heading=relative_heading,
            approach_vector=approach_vector,
            approaching=approaching,
            ttc=ttc,
            risk_level=risk_level,
            description=description,
        )
        
        return result
    
    @staticmethod
    def _calculate_ttc(v1: Tuple[float, float],
                      v2: Tuple[float, float],
                      approach_vector: Tuple[float, float],
                      min_dist: float) -> Tuple[Optional[float], bool]:
        """
        Calculate Time To Collision (TTC).
        
        Args:
            v1, v2: Velocity vectors in m/s
            approach_vector: Direction from closest point 1 to closest point 2
            min_dist: Minimum distance in meters
        
        Returns:
            (ttc_seconds, approaching_bool)
            ttc: seconds until collision (None if not approaching)
            approaching: True if objects are approaching
        """
        
        # Relative velocity
        rel_vx = v2[0] - v1[0]
        rel_vy = v2[1] - v1[1]
        
        # Approach direction (normalized)
        approach_len = np.sqrt(approach_vector[0]**2 + approach_vector[1]**2)
        
        if approach_len < 0.01:  # Too close or collinear
            return None, False
        
        approach_unit = (approach_vector[0] / approach_len,
                        approach_vector[1] / approach_len)
        
        # Approach speed (negative = approaching, positive = separating)
        approach_speed = rel_vx * approach_unit[0] + rel_vy * approach_unit[1]
        
        # Check if approaching
        approaching = approach_speed < -0.01  # Moving toward each other
        
        if not approaching:
            return None, False
        
        # Calculate TTC
        ttc = min_dist / abs(approach_speed)
        
        return ttc if ttc > 0 else None, True
    
    @staticmethod
    def _assess_risk(min_dist: float,
                    ttc: Optional[float],
                    relative_heading: float,
                    approaching: bool) -> str:
        """
        Assess collision risk level.
        
        Risk levels:
        - CRITICAL: High risk, immediate action needed
        - HIGH: Elevated risk, caution required
        - MEDIUM: Moderate risk, monitoring needed
        - LOW: Low risk, normal operations
        
        Args:
            min_dist: Minimum distance in meters
            ttc: Time to collision in seconds (None if not approaching)
            relative_heading: Relative heading angle in radians
            approaching: Whether objects are approaching
        
        Returns:
            Risk level string: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
        """
        
        if not approaching:
            # Not approaching, low risk
            if min_dist < 0.5:
                return 'MEDIUM'  # Close but separating
            return 'LOW'
        
        # Objects are approaching
        if min_dist < 0.5 or (ttc is not None and ttc < 1.0):
            return 'CRITICAL'
        elif min_dist < 1.5 or (ttc is not None and ttc < 2.0):
            return 'HIGH'
        elif min_dist < 3.0 or (ttc is not None and ttc < 5.0):
            return 'MEDIUM'
        else:
            return 'LOW'


# For testing, try to import cv2 but make it optional
try:
    import cv2
except ImportError:
    cv2 = None


if __name__ == '__main__':
    # Test collision analysis
    print("Testing collision analysis:")
    
    # Create sample objects
    obj1 = {'class': 2, 'bbox_xywh': [500, 200, 100, 150]}  # car
    obj2 = {'class': 2, 'bbox_xywh': [350, 350, 80, 120]}   # car
    
    # Get anchor points
    anchors1 = VehicleAnchors.get_anchors(obj1['bbox_xywh'], obj1['class'])
    anchors2 = VehicleAnchors.get_anchors(obj2['bbox_xywh'], obj2['class'])
    
    print(f"\nObject 1 anchors: {anchors1}")
    print(f"Object 2 anchors: {anchors2}")
    
    # Analyze collision
    analyzer = CollisionAnalyzer(pixel_per_meter=60.0)
    result = analyzer.analyze(
        obj1, obj2,
        anchors1, anchors2,
        (5.0, 0.0),  # obj1 velocity: 5 m/s east
        (3.0, 3.0),  # obj2 velocity: 3m/s east, 3m/s north
    )
    
    print(f"\nCollision Analysis Result:")
    print(f"  Min distance: {result.min_distance:.2f}m ({result.min_distance_px:.1f}px)")
    print(f"  Closest parts: {result.object1_part} ↔ {result.object2_part}")
    print(f"  Risk level: {result.risk_level}")
    print(f"  TTC: {result.ttc:.1f}s" if result.ttc else "  TTC: N/A")
    print(f"  Approaching: {result.approaching}")
    print(f"  Description: {result.description}")
