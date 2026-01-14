"""
generate_warped_video_imageio.py.

使用imageio生成warped视频（确保兼容性）
"""

import json

import cv2
import imageio
import numpy as np


def generate_warped_video_with_imageio(video_path, homography_path, output_path, output_size=(360, 2400)):
    """使用imageio生成warped视频."""
    print("【步骤1: 加载Homography矩阵】")
    with open(homography_path) as f:
        data = json.load(f)

    H = np.array(data["homography_matrix"], dtype=np.float32)
    world_points = data["world_points"]

    print("✓ Homography矩阵已加载")

    # 计算透视变换矩阵
    print("\n【步骤2: 计算变换矩阵】")
    min_x = min(w[0] for w in world_points)
    max_x = max(w[0] for w in world_points)
    min_y = min(w[1] for w in world_points)
    max_y = max(w[1] for w in world_points)

    world_width = max_x - min_x
    world_height = max_y - min_y

    A = np.array(
        [[world_width / output_size[0], 0, min_x], [0, -world_height / output_size[1], max_y], [0, 0, 1]],
        dtype=np.float32,
    )

    H_inv = np.linalg.inv(H)
    M = H_inv @ A

    print("✓ 变换矩阵已计算")
    print(f"  输出分辨率: {output_size[0]}×{output_size[1]}")

    # 读取视频和生成帧
    print("\n【步骤3: 处理视频帧】")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ 无法打开视频")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  原始视频: {total_frames}帧 @ {fps:.2f}FPS")

    # 收集所有warped帧
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        warped = cv2.warpPerspective(frame, M, output_size)
        # 转换BGR到RGB（imageio需要RGB）
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        frames.append(warped_rgb)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  处理: {frame_count}/{total_frames}")

    cap.release()

    print("\n【步骤4: 写入视频文件】")
    print(f"  输出: {output_path}")

    # 使用imageio写入视频
    try:
        imageio.mimsave(output_path, frames, fps=fps, pixelformat="rgb24")
        print("✓ 视频已保存!")

        # 验证
        print("\n【步骤5: 验证输出】")
        import os

        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✓ 文件大小: {size:.2f}MB")
            print(f"✓ 帧数: {len(frames)}")
            return True
        else:
            print("❌ 文件不存在")
            return False

    except Exception as e:
        print(f"❌ 错误: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="使用imageio生成warped视频")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--homography", type=str, required=True)
    parser.add_argument("--output", type=str, default="warped_imageio.mp4")
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=2400)

    args = parser.parse_args()

    print("=" * 70)
    print("使用imageio生成warped视频")
    print("=" * 70)

    success = generate_warped_video_with_imageio(args.video, args.homography, args.output, (args.width, args.height))

    if success:
        print("\n✓ 生成成功!")
    else:
        print("\n❌ 生成失败")
