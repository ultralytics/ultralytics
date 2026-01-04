"""
Shared homography transformation utilities for homography_transform_video.py and collision_detection_pipeline.py

This module provides functions for:
1. Loading homography matrices from calibration JSON files
2. Computing the complete transformation matrix (output coords -> pixel coords)
3. Performing manual pixel-by-pixel transformation with bilinear interpolation
   (avoids OpenCV's warpPerspective numerical stability issues)
"""

import json
import numpy as np


def load_homography(json_path):
    """
    加载Homography矩阵和参考点坐标
    
    参数：
    - json_path: 包含homography矩阵的JSON文件路径
    
    返回：
    - H: Homography矩阵 (3x3)
    - world_points: 世界坐标参考点 (list of [x, y])
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    H = np.array(data['homography_matrix'], dtype=np.float32)
    return H, data['world_points']


def compute_transformation_matrix(H, world_points, output_size):
    """
    计算完整的变换矩阵：输出坐标 -> 像素坐标
    
    推导过程：
    1. H矩阵：像素 -> 世界 (已知)
    2. A矩阵：输出图像 -> 世界 (构造)
    3. M矩阵：输出图像 -> 像素 = H_inv @ A
    
    参数：
    - H: Homography矩阵 (3x3)
    - world_points: 世界坐标参考点 (list of [x, y])
    - output_size: 输出图像尺寸 (width, height)
    
    返回：
    - M: 变换矩阵 (3x3)
    - world_bounds: 世界坐标范围 (min_x, max_x, min_y, max_y)
    """
    
    # Step 1: 计算世界坐标范围
    min_x = min(w[0] for w in world_points)
    max_x = max(w[0] for w in world_points)
    min_y = min(w[1] for w in world_points)
    max_y = max(w[1] for w in world_points)
    
    # Step 2: 构造A矩阵 (输出坐标 -> 世界坐标)
    out_w, out_h = output_size
    world_width = max_x - min_x
    world_height = max_y - min_y
    
    A = np.array([
        [world_width / out_w, 0, min_x],
        [0, -world_height / out_h, max_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Step 3: 计算H的逆矩阵 (世界 -> 像素)
    H_inv = np.linalg.inv(H)
    
    # Step 4: 合成M矩阵 (输出 -> 像素)
    M = H_inv @ A
    
    return M, (min_x, max_x, min_y, max_y)


def transform_frame_manual(frame, M, output_size):
    """
    对单个帧进行手工逐像素变换（使用双线性插值）
    
    这个方法规避了OpenCV warpPerspective的数值稳定性问题
    采用逐像素映射 + 双线性插值的方式进行透视变换
    
    参数：
    - frame: 输入帧 (HxWx3 uint8)
    - M: 变换矩阵 (3x3) - 从compute_transformation_matrix()得到
    - output_size: 输出尺寸 (width, height)
    
    返回：
    - warped: 变换后的帧 (HxWx3 uint8)
    """
    
    out_w, out_h = output_size
    warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    h, w = frame.shape[:2]
    
    for out_y in range(out_h):
        for out_x in range(out_w):
            # 将输出像素坐标映射到源图像坐标
            out_h_vec = np.array([float(out_x), float(out_y), 1.0])
            pix_h_vec = M @ out_h_vec
            pix_coord = pix_h_vec[:2] / pix_h_vec[2]
            
            src_x, src_y = pix_coord
            
            # 双线性插值
            if 0 <= src_x < w-1 and 0 <= src_y < h-1:
                x0, y0 = int(np.floor(src_x)), int(np.floor(src_y))
                x1, y1 = x0 + 1, y0 + 1
                
                # 获取4个相邻像素
                p00 = frame[y0, x0].astype(np.float32)
                p10 = frame[y0, x1].astype(np.float32)
                p01 = frame[y1, x0].astype(np.float32)
                p11 = frame[y1, x1].astype(np.float32)
                
                # 计算权重
                wx = src_x - x0
                wy = src_y - y0
                
                # 双线性插值
                p0 = p00 * (1 - wx) + p10 * wx
                p1 = p01 * (1 - wx) + p11 * wx
                pixel = p0 * (1 - wy) + p1 * wy
                
                warped[out_y, out_x] = np.clip(pixel, 0, 255).astype(np.uint8)
    
    return warped
