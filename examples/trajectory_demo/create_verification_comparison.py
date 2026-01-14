"""
create_verification_comparison.py.

生成原图和变换图的对比图
"""

import json
import os

import cv2
import numpy as np


def load_homography(json_path):
    """加载Homography矩阵."""
    with open(json_path) as f:
        data = json.load(f)
    H = np.array(data["homography_matrix"], dtype=np.float32)
    return H, data["pixel_points"], data["world_points"]


def create_comparison(video_path, homography_path, output_path):
    """创建原图和变换图的对比."""
    H, pixel_points, world_points = load_homography(homography_path)

    # 读取第一帧
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ 无法读取视频: {video_path}")
        return

    # 在原图上标注4个参考点
    frame_marked = frame.copy()
    for i, (px, py) in enumerate(pixel_points):
        cv2.circle(frame_marked, (int(px), int(py)), 20, (0, 255, 0), 3)
        cv2.putText(
            frame_marked, f"P{i + 1}", (int(px) + 30, int(py) - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
        )
        print(f"点{i + 1}(像素): ({px:.0f}, {py:.0f}) → 世界: ({world_points[i][0]:.2f}, {world_points[i][1]:.2f})")

    # 应用透视变换
    H_inv = np.linalg.inv(H)
    out_w, out_h = 180, 1200
    warped = cv2.warpPerspective(frame, H_inv, (out_w, out_h))

    print(f"\n✓ 原图尺寸: {frame.shape[1]}x{frame.shape[0]}")
    print(f"✓ 变换后尺寸: {warped.shape[1]}x{warped.shape[0]}")

    # 创建对比图（上下排列）
    # 先缩小原图高度以便对比
    scale = out_h / frame.shape[0]
    frame_resized = cv2.resize(frame_marked, (int(frame.shape[1] * scale), out_h))

    # 并排显示
    comparison = np.hstack([frame_resized, warped])

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, comparison)

    print(f"\n✓ 对比图已保存: {output_path}")
    print("  左边: 原始图(标记了4个参考点)")
    print("  右边: 透视变换后的图(鸟瞰视角)")


if __name__ == "__main__":
    video_path = "../../videos/Homograph_Teset_FullScreen.mp4"
    homography_path = "../../calibration/Homograph_Teset_FullScreen_homography.json"
    output_path = "../../videos/comparison.jpg"

    create_comparison(video_path, homography_path, output_path)
