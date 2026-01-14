"""
verify_homography.py.

生成Homography变换验证图片
用于视觉验证标定是否正确

使用示例：
python verify_homography.py \
  --video ../../videos/Homograph_Teset_FullScreen.mp4 \
  --homography ../../calibration/Homograph_Teset_FullScreen_homography.json \
  --output verify_homography.jpg
"""

import argparse
import json
import os

import cv2
import numpy as np


def load_homography(json_path):
    """加载Homography矩阵."""
    with open(json_path) as f:
        data = json.load(f)
    H = np.array(data["homography_matrix"], dtype=np.float32)
    pixel_points = data["pixel_points"]
    world_points = data["world_points"]
    return H, pixel_points, world_points


def verify_homography(video_path, homography_path, output_path):
    """生成验证图片."""
    # 加载Homography和参考点
    H, pixel_points, world_points = load_homography(homography_path)

    # 读取视频第一帧
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ 无法读取视频: {video_path}")
        return

    h, w = frame.shape[:2]
    print(f"✓ 读取视频第一帧: {w}x{h} 像素")

    # 创建两个版本的图：原图 + 标注点 和 变换后的图
    frame_marked = frame.copy()

    # 在原图上标注参考点
    print("\n【原始图像标注】")
    print("-" * 50)
    for i, (px, py) in enumerate(pixel_points):
        # 画圆和数字
        cv2.circle(frame_marked, (int(px), int(py)), 15, (0, 255, 0), 3)  # 绿色圆
        cv2.putText(
            frame_marked, str(i + 1), (int(px) + 20, int(py) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2
        )
        print(f"点 {i + 1}: 像素({px:.0f}, {py:.0f}) -> 世界({world_points[i][0]:.2f}, {world_points[i][1]:.2f})")

    # 添加标题和说明
    cv2.putText(
        frame_marked, "Original Frame with Reference Points", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
    )

    # 计算变换后图像的大小和位置
    # 我们需要找到所有参考点变换后的范围
    np.array(world_points, dtype=np.float32)
    min_x = min(p[0] for p in world_points)
    max_x = max(p[0] for p in world_points)
    min_y = min(p[1] for p in world_points)
    max_y = max(p[1] for p in world_points)

    world_width = max_x - min_x
    world_height = max_y - min_y

    print("\n【世界坐标范围】")
    print("-" * 50)
    print(f"X范围: {min_x:.2f} 到 {max_x:.2f} (宽 {world_width:.2f} 米)")
    print(f"Y范围: {min_y:.2f} 到 {max_y:.2f} (高 {world_height:.2f} 米)")

    # 创建变换后的图像（使用透视变换的逆矩阵）
    H_inv = np.linalg.inv(H)

    # 使用一个更大的输出画布来显示变换结果
    output_scale = 100  # 每米100像素
    output_width = int(world_width * output_scale) + 100
    output_height = int(world_height * output_scale) + 100

    warped = cv2.warpPerspective(frame, H_inv, (output_width, output_height))

    # 在变换后的图上标注参考点
    warped_marked = warped.copy()

    print("\n【变换后图像】")
    print("-" * 50)
    print(f"输出尺寸: {output_width}x{output_height} 像素 (1米=100像素)")

    # 标注变换后的参考点
    for i, (wx, wy) in enumerate(world_points):
        # 转换到输出图像坐标（需要相对于最小值）
        out_x = int((wx - min_x) * output_scale) + 50
        out_y = int((max_y - wy) * output_scale) + 50  # 翻转Y轴

        cv2.circle(warped_marked, (out_x, out_y), 15, (255, 0, 0), 3)  # 蓝色圆
        cv2.putText(warped_marked, f"{i + 1}", (out_x + 20, out_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        print(f"点 {i + 1}: 世界({wx:.2f}, {wy:.2f}) -> 输出({out_x}, {out_y})")

    cv2.putText(warped_marked, "Warped View (Bird's Eye View)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    # 拼接两个图像进行对比
    # 调整大小使其高度一致
    h1, w1 = frame_marked.shape[:2]
    h2, w2 = warped_marked.shape[:2]

    # 缩小warped图像以便比较
    scale = min(h1 / h2, 800 / w2)
    warped_resized = cv2.resize(warped_marked, (int(w2 * scale), int(h2 * scale)))

    # 创建空白背景
    canvas_w = w1 + warped_resized.shape[1] + 30
    canvas_h = max(h1, warped_resized.shape[0]) + 60
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # 放置两个图像
    canvas[10 : 10 + h1, 10 : 10 + w1] = frame_marked
    canvas[10 : 10 + warped_resized.shape[0], w1 + 20 : w1 + 20 + warped_resized.shape[1]] = warped_resized

    # 添加标题
    cv2.putText(canvas, "Homography Verification", (10, canvas_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # 保存结果
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)

    print(f"\n✓ 验证图片已保存: {output_path}")
    print("\n【验证方法】")
    print("-" * 50)
    print("绿色圆点：原始视频中的参考点位置（像素坐标）")
    print("蓝色圆点：变换后的参考点位置（世界坐标在鸟瞰图中）")
    print("\n如果蓝色圆点构成规则的矩形，说明标定正确！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homography变换验证")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--homography", type=str, required=True, help="Homography矩阵JSON文件路径")
    parser.add_argument("--output", type=str, default="verify_homography.jpg", help="输出图片路径")

    args = parser.parse_args()

    verify_homography(args.video, args.homography, args.output)
