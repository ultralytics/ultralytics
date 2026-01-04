"""
test_homography.py

自动生成一个测试用的Homography矩阵，用于验证homography功能
"""

import json
import os
import cv2
import numpy as np

# 提取第一帧
video_path = "videos/NewYorkSample.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ 无法读取视频")
    exit(1)

# 定义4个参考点
# 根据视频的车道宽度，标定坐标
pixel_points = [
    (100, 100),    # 左上
    (600, 100),    # 右上
    (50, 630),     # 左下
    (650, 630),    # 右下
]

world_points = [
    (-12, 0),      # 左上
    (12, 0),       # 右上
    (-12, 30),     # 左下
    (12, 30),      # 右下
]

# 计算Homography矩阵
src_pts = np.float32(pixel_points)
dst_pts = np.float32(world_points)
H, _ = cv2.findHomography(src_pts, dst_pts)

if H is None:
    print("❌ 计算Homography失败")
    exit(1)

print("✓ Homography矩阵计算成功！")
print(H)

# 保存为JSON
os.makedirs("calibration", exist_ok=True)
output_file = "calibration/NewYorkSample_homography.json"

data = {
    "video_name": "NewYorkSample",
    "homography_matrix": H.tolist(),
    "pixel_points": pixel_points,
    "world_points": world_points,
    "notes": "自动生成的测试Homography矩阵"
}

with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✓ Homography矩阵已保存：{output_file}")

# 测试变换
print("\n【变换效果测试】")
for i, (px_pt, w_pt) in enumerate(zip(pixel_points, world_points)):
    pixel_coord = np.array([[[px_pt[0], px_pt[1]]]], dtype=np.float32)
    world_coord = cv2.perspectiveTransform(pixel_coord, H)
    transformed_x, transformed_y = world_coord[0][0]
    
    error_x = abs(transformed_x - w_pt[0])
    error_y = abs(transformed_y - w_pt[1])
    
    print(f"点 {i+1}: 像素{px_pt} → 世界({transformed_x:.2f}, {transformed_y:.2f}) [期望({w_pt[0]}, {w_pt[1]})]")
    print(f"  误差: ({error_x:.4f}, {error_y:.4f})")
