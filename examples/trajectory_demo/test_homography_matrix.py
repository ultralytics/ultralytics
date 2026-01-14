"""
test_homography_matrix.py.

测试Homography矩阵是否正确
"""

import json

import numpy as np


def load_homography(json_path):
    """加载Homography矩阵."""
    with open(json_path) as f:
        data = json.load(f)
    H = np.array(data["homography_matrix"], dtype=np.float32)
    return H, data["pixel_points"], data["world_points"]


# 加载数据
H, pixel_points, world_points = load_homography("../../calibration/Homograph_Teset_FullScreen_homography.json")

print("=" * 60)
print("Homography矩阵验证")
print("=" * 60)
print("\n矩阵 H (像素→世界):")
print(H)

print("\n逆矩阵 H_inv (世界→像素):")
H_inv = np.linalg.inv(H)
print(H_inv)

print("\n" + "=" * 60)
print("验证：像素坐标 → 世界坐标")
print("=" * 60)

for i, (px_pt, w_pt) in enumerate(zip(pixel_points, world_points)):
    # 使用H矩阵进行变换
    pixel_homo = np.array([px_pt[0], px_pt[1], 1], dtype=np.float32)
    world_homo = H @ pixel_homo
    world_homo = world_homo[:2] / world_homo[2]

    print(f"\n点{i + 1}:")
    print(f"  输入像素坐标: ({px_pt[0]}, {px_pt[1]})")
    print(f"  期望世界坐标: ({w_pt[0]}, {w_pt[1]})")
    print(f"  计算结果: ({world_homo[0]:.4f}, {world_homo[1]:.4f})")
    error = np.sqrt((world_homo[0] - w_pt[0]) ** 2 + (world_homo[1] - w_pt[1]) ** 2)
    print(f"  误差: {error:.6f} 米 {'✓' if error < 0.01 else '❌'}")

print("\n" + "=" * 60)
print("验证：世界坐标 → 像素坐标（用逆矩阵）")
print("=" * 60)

for i, (px_pt, w_pt) in enumerate(zip(pixel_points, world_points)):
    # 使用H_inv矩阵进行变换
    world_homo = np.array([w_pt[0], w_pt[1], 1], dtype=np.float32)
    pixel_homo = H_inv @ world_homo
    pixel_homo = pixel_homo[:2] / pixel_homo[2]

    print(f"\n点{i + 1}:")
    print(f"  输入世界坐标: ({w_pt[0]}, {w_pt[1]})")
    print(f"  期望像素坐标: ({px_pt[0]}, {px_pt[1]})")
    print(f"  计算结果: ({pixel_homo[0]:.4f}, {pixel_homo[1]:.4f})")
    error = np.sqrt((pixel_homo[0] - px_pt[0]) ** 2 + (pixel_homo[1] - px_pt[1]) ** 2)
    print(f"  误差: {error:.6f} 像素 {'✓' if error < 0.1 else '❌'}")
