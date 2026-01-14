"""
direct_verify_mapping.py.

直接验证输出坐标→像素坐标的映射是否正确
"""

import json

import numpy as np


def load_homography(json_path):
    with open(json_path) as f:
        data = json.load(f)
    H = np.array(data["homography_matrix"], dtype=np.float32)
    return H, data["pixel_points"], data["world_points"]


# 加载数据
H, pixel_points, world_points = load_homography("../../calibration/Homograph_Teset_FullScreen_homography.json")

print("=" * 70)
print("直接验证映射关系")
print("=" * 70)

# 世界坐标范围
min_x = -3.75
max_x = 3.75
min_y = 0
max_y = 50

out_w = 180
out_h = 1200

print("\n【设定】")
print(f"输出图像: {out_w}x{out_h}")
print(f"世界坐标范围: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")

# 构造A矩阵（输出→世界）
world_width = max_x - min_x  # 7.5
world_height = max_y - min_y  # 50

A = np.array([[world_width / out_w, 0, min_x], [0, -world_height / out_h, max_y], [0, 0, 1]], dtype=np.float32)

print("\n【矩阵A：输出→世界】")
print(A)

# H_inv（世界→像素）
H_inv = np.linalg.inv(H)

print("\n【矩阵H_inv：世界→像素】")
print(H_inv)

# M（输出→像素）
M = H_inv @ A

print("\n【矩阵M = H_inv @ A：输出→像素】")
print(M)

# 现在直接测试几个输出点
print("\n" + "=" * 70)
print("【直接映射测试】")
print("=" * 70)

test_points = [
    (0, 0, "左上角"),
    (180, 0, "右上角"),
    (0, 1200, "左下角"),
    (180, 1200, "右下角"),
    (90, 600, "中心"),
]

for out_x, out_y, label in test_points:
    print(f"\n输出坐标 ({out_x}, {out_y}) - {label}")

    # 方法1：用矩阵M直接映射
    out_homo = np.array([out_x, out_y, 1], dtype=np.float32)
    pixel_homo = M @ out_homo
    pixel_x = pixel_homo[0] / pixel_homo[2]
    pixel_y = pixel_homo[1] / pixel_homo[2]

    print(f"  → 像素坐标: ({pixel_x:.2f}, {pixel_y:.2f})")

    # 验证：这个像素坐标是否在原图范围内
    if 0 <= pixel_x < 1080 and 0 <= pixel_y < 1920:
        print("  ✓ 在原图范围内")
    else:
        print("  ❌ 超出原图范围！(原图1080x1920)")

    # 验证：这个像素坐标对应的世界坐标是否正确
    pixel_homo2 = np.array([pixel_x, pixel_y, 1], dtype=np.float32)
    world_homo = H @ pixel_homo2
    world_x = world_homo[0] / world_homo[2]
    world_y = world_homo[1] / world_homo[2]

    print(f"  → 世界坐标: ({world_x:.4f}, {world_y:.4f})")

    # 计算这个输出点对应的期望世界坐标
    expected_world_x = min_x + (out_x / out_w) * world_width
    expected_world_y = max_y - (out_y / out_h) * world_height

    print(f"  期望世界坐标: ({expected_world_x:.4f}, {expected_world_y:.4f})")

    error = np.sqrt((world_x - expected_world_x) ** 2 + (world_y - expected_world_y) ** 2)
    print(f"  误差: {error:.6f} {'✓' if error < 0.1 else '❌'}")

# 关键问题：检查所有输出像素是否都映射到有效范围
print("\n" + "=" * 70)
print("【关键诊断：输出像素范围分析】")
print("=" * 70)

# 检查输出图像四个角的像素坐标
corners = [(0, 0), (out_w, 0), (0, out_h), (out_w, out_h)]
pixel_coords = []

print("\n输出图像的4个角点映射到像素坐标：")
for out_x, out_y in corners:
    out_homo = np.array([out_x, out_y, 1], dtype=np.float32)
    pixel_homo = M @ out_homo
    px = pixel_homo[0] / pixel_homo[2]
    py = pixel_homo[1] / pixel_homo[2]
    pixel_coords.append((px, py))
    print(f"  ({out_x:3d}, {out_y:4d}) → 像素({px:7.2f}, {py:7.2f})")

# 分析像素坐标范围
px_coords = [p[0] for p in pixel_coords]
py_coords = [p[1] for p in pixel_coords]

print(f"\n像素X范围: {min(px_coords):.2f} ~ {max(px_coords):.2f}")
print(f"像素Y范围: {min(py_coords):.2f} ~ {max(py_coords):.2f}")
print("原图范围: 0 ~ 1080 (X), 0 ~ 1920 (Y)")

if min(px_coords) >= 0 and max(px_coords) <= 1080 and min(py_coords) >= 0 and max(py_coords) <= 1920:
    print("✓ 所有输出像素都在原图范围内")
else:
    print("❌ 部分输出像素超出原图范围！")
    print("   这可能导致黑色区域！")
