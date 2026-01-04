"""
correct_perspective_transform.py

正确的透视变换逻辑
"""

import cv2
import numpy as np
import json
import os

def load_homography(json_path):
    """加载Homography矩阵"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    H = np.array(data['homography_matrix'], dtype=np.float32)
    return H, data['pixel_points'], data['world_points']

def compute_output_to_pixel_matrix(H, world_bounds, output_size):
    """
    计算从输出坐标到像素坐标的映射矩阵
    
    参数：
    - H: 像素坐标 → 世界坐标的矩阵
    - world_bounds: (min_x, max_x, min_y, max_y) 世界坐标范围
    - output_size: (width, height) 输出图像尺寸
    
    返回：
    - M: 从输出坐标到像素坐标的矩阵，用于warpPerspective
    """
    
    min_x, max_x, min_y, max_y = world_bounds
    out_w, out_h = output_size
    
    # 第一步：构造输出坐标 → 世界坐标的映射矩阵A
    # 规定：输出图像的(0, 0)对应世界坐标(min_x, max_y)
    #       输出图像的(out_w, 0)对应世界坐标(max_x, max_y)
    #       输出图像的(0, out_h)对应世界坐标(min_x, min_y)
    #       输出图像的(out_w, out_h)对应世界坐标(max_x, min_y)
    
    # 线性映射：
    # world_x = min_x + (out_x / out_w) * (max_x - min_x)
    # world_y = max_y - (out_y / out_h) * (max_y - min_y)
    
    # 矩阵形式：[world_x, world_y, 1]^T = A @ [out_x, out_y, 1]^T
    
    world_width = max_x - min_x
    world_height = max_y - min_y
    
    A = np.array([
        [world_width / out_w, 0, min_x],           # world_x = (world_width/out_w)*out_x + min_x
        [0, -world_height / out_h, max_y],         # world_y = -(world_height/out_h)*out_y + max_y
        [0, 0, 1]
    ], dtype=np.float32)
    
    print("\n【输出坐标→世界坐标的映射矩阵 A】")
    print(f"world_x = ({world_width/out_w:.6f}) * out_x + {min_x}")
    print(f"world_y = ({-world_height/out_h:.6f}) * out_y + {max_y}")
    print(A)
    
    # 第二步：计算H的逆矩阵（世界坐标→像素坐标）
    H_inv = np.linalg.inv(H)
    
    print("\n【H的逆矩阵 H_inv (世界坐标→像素坐标)】")
    print(H_inv)
    
    # 第三步：合并矩阵 M = H_inv @ A
    # 这样：像素坐标 = M @ 输出坐标
    M = H_inv @ A
    
    print("\n【最终矩阵 M = H_inv @ A (输出坐标→像素坐标)】")
    print("这是warpPerspective需要的矩阵")
    print(M)
    
    return M

def test_matrix(M, world_bounds, output_size, H):
    """验证矩阵是否正确"""
    print("\n" + "=" * 60)
    print("【矩阵验证】")
    print("=" * 60)
    
    min_x, max_x, min_y, max_y = world_bounds
    out_w, out_h = output_size
    
    # 测试四个角点
    test_points = [
        (0, 0, "左上"),
        (out_w, 0, "右上"),
        (0, out_h, "左下"),
        (out_w, out_h, "右下")
    ]
    
    expected_world = [
        (min_x, max_y, "左上"),
        (max_x, max_y, "右上"),
        (min_x, min_y, "左下"),
        (max_x, min_y, "右下")
    ]
    
    for (out_x, out_y, label), (exp_x, exp_y, _) in zip(test_points, expected_world):
        # 输出坐标 → 像素坐标
        out_homo = np.array([out_x, out_y, 1], dtype=np.float32)
        pixel_homo = M @ out_homo
        pixel_x = pixel_homo[0] / pixel_homo[2]
        pixel_y = pixel_homo[1] / pixel_homo[2]
        
        # 验证：像素坐标 → 世界坐标
        pixel_homo2 = np.array([pixel_x, pixel_y, 1], dtype=np.float32)
        world_homo = H @ pixel_homo2
        world_x = world_homo[0] / world_homo[2]
        world_y = world_homo[1] / world_homo[2]
        
        print(f"\n{label} (输出坐标 {out_x}, {out_y}):")
        print(f"  → 像素坐标 ({pixel_x:.2f}, {pixel_y:.2f})")
        print(f"  → 世界坐标 ({world_x:.2f}, {world_y:.2f})")
        print(f"  期望世界坐标 ({exp_x:.2f}, {exp_y:.2f})")
        error = np.sqrt((world_x - exp_x)**2 + (world_y - exp_y)**2)
        print(f"  误差: {error:.6f} {'✓' if error < 0.1 else '❌'}")

# 主程序
video_path = '../../videos/Homograph_Teset_FullScreen.mp4'
homography_path = '../../calibration/Homograph_Teset_FullScreen_homography.json'

print("=" * 60)
print("正确的透视变换逻辑推导")
print("=" * 60)

H, pixel_points, world_points = load_homography(homography_path)

print("\n【输入数据】")
print("H矩阵（像素→世界）:")
print(H)
print("\n参考点:")
for i, (p, w) in enumerate(zip(pixel_points, world_points)):
    print(f"  点{i+1}: 像素{p} → 世界{w}")

# 世界坐标范围
min_x = min(w[0] for w in world_points)
max_x = max(w[0] for w in world_points)
min_y = min(w[1] for w in world_points)
max_y = max(w[1] for w in world_points)

world_bounds = (min_x, max_x, min_y, max_y)
output_size = (180, 1200)

print(f"\n【世界坐标范围】")
print(f"X: {min_x} ~ {max_x}")
print(f"Y: {min_y} ~ {max_y}")

# 计算映射矩阵
M = compute_output_to_pixel_matrix(H, world_bounds, output_size)

# 验证矩阵
test_matrix(M, world_bounds, output_size, H)
