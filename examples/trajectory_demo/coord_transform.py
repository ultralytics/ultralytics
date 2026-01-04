"""
coord_transform.py

坐标变换：使用Homography矩阵进行像素坐标到世界坐标的变换

原理：
  Homography矩阵是一个3x3矩阵，用于将一个平面上的点映射到另一个平面
  在我们的应用中：
    像素坐标(px, py) --[H]--> 世界坐标(x_world, y_world)
  
  这样可以将计算的距离从"像素"转换为"米"，更符合实际

用法：
  1. 先用calibration.py标定，得到homography矩阵
  2. 在yolo_runner.py中加载矩阵
  3. 用transform_point()或transform_batch()进行转换
"""

import numpy as np
import json
import os
from typing import Tuple, List, Optional


def load_homography(json_path: str) -> Optional[np.ndarray]:
    """从JSON文件加载Homography矩阵
    
    参数：
        json_path: homography矩阵JSON文件路径
    
    返回：
        H: 3x3 homography矩阵，或None如果加载失败
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        H = np.array(data['homography_matrix'], dtype=np.float32)
        print(f"✓ 已加载Homography矩阵: {json_path}")
        return H
    except Exception as e:
        print(f"❌ 加载Homography矩阵失败: {e}")
        return None


def transform_point(pixel_point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """透视变换：单个点从像素坐标转换为世界坐标
    
    参数：
        pixel_point: (px, py) 像素坐标
        H: 3x3 homography矩阵
    
    返回：
        (x_world, y_world) 世界坐标（单位：米）
    """
    # 构造齐次坐标
    px, py = pixel_point
    pixel_homo = np.array([[[px, py, 1]]], dtype=np.float32)
    
    # 使用OpenCV的perspectiveTransform进行变换
    import cv2
    world_homo = cv2.perspectiveTransform(pixel_homo, H)
    
    x_world = float(world_homo[0][0][0])
    y_world = float(world_homo[0][0][1])
    
    return (x_world, y_world)


def transform_batch(pixel_points: List[Tuple[float, float]], H: np.ndarray) -> List[Tuple[float, float]]:
    """透视变换：批量点从像素坐标转换为世界坐标
    
    参数：
        pixel_points: [(px1, py1), (px2, py2), ...] 像素坐标列表
        H: 3x3 homography矩阵
    
    返回：
        [(x1, y1), (x2, y2), ...] 世界坐标列表
    """
    if not pixel_points:
        return []
    
    # 转换为numpy数组（格式：N x 1 x 2）
    pixel_array = np.array(pixel_points, dtype=np.float32)
    pixel_array = pixel_array.reshape(-1, 1, 2)
    
    # 使用OpenCV的perspectiveTransform进行批量变换
    import cv2
    world_array = cv2.perspectiveTransform(pixel_array, H)
    
    # 转换回列表格式
    world_points = [tuple(pt[0]) for pt in world_array]
    return world_points


def compute_world_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """计算两个世界坐标之间的欧氏距离
    
    参数：
        point1: (x1, y1) 世界坐标
        point2: (x2, y2) 世界坐标
    
    返回：
        距离（单位：米）
    """
    x1, y1 = point1
    x2, y2 = point2
    
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return float(distance)
