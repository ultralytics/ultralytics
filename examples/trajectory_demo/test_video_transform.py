"""
test_video_transform.py.

测试视频转换 - 只进行Homography透视变换，不进行物体检测
可以验证转换后的视频是否正常
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def transform_video_simple(video_path, homography_path, output_path, output_size=(360, 2400)):
    """简单的视频转换函数."""
    # 加载Homography矩阵
    print("【步骤1: 加载Homography矩阵】")
    with open(homography_path) as f:
        data = json.load(f)

    H = np.array(data["homography_matrix"], dtype=np.float32)
    pixel_points = data["pixel_points"]
    world_points = data["world_points"]

    print("✓ Homography矩阵已加载")
    print(f"  像素点: {pixel_points}")
    print(f"  世界坐标: {world_points}")

    # 计算透视变换矩阵
    print("\n【步骤2: 计算透视变换矩阵】")
    min_x = min(w[0] for w in world_points)
    max_x = max(w[0] for w in world_points)
    min_y = min(w[1] for w in world_points)
    max_y = max(w[1] for w in world_points)

    world_width = max_x - min_x
    world_height = max_y - min_y

    # A矩阵：输出坐标 → 世界坐标
    A = np.array(
        [[world_width / output_size[0], 0, min_x], [0, -world_height / output_size[1], max_y], [0, 0, 1]],
        dtype=np.float32,
    )

    # M矩阵：输出坐标 → 源视频坐标
    H_inv = np.linalg.inv(H)
    M = H_inv @ A

    print("✓ 变换矩阵已计算")
    print(f"  世界范围: X[{min_x:.2f}, {max_x:.2f}]m, Y[{min_y:.2f}, {max_y:.2f}]m")
    print(f"  输出分辨率: {output_size[0]}×{output_size[1]}")

    # 打开视频
    print("\n【步骤3: 读取视频并转换】")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  原始视频: {total_frames}帧 @ {fps:.2f}FPS")

    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 初始化VideoWriter - 尝试多种格式
    codecs_to_try = [
        ("mp4v", ".mp4"),  # MP4
        ("MJPG", ".avi"),  # Motion JPEG (AVI)
        ("XVID", ".avi"),  # Xvid
        ("DIVX", ".avi"),  # DivX
    ]

    out = None
    for codec_name, ext in codecs_to_try:
        test_output = output_path if codec_name == "mp4v" else output_path.replace(".mp4", ext)
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        test_out = cv2.VideoWriter(test_output, fourcc, fps, output_size)

        if test_out.isOpened():
            print(f"✓ 使用编码: {codec_name}")
            out = test_out
            output_path = test_output
            break
        else:
            print(f"  尝试 {codec_name} 失败")

    if out is None or not out.isOpened():
        print("❌ 无法创建输出视频")
        cap.release()
        return False

    print(f"  输出视频: {output_path}")

    # 逐帧转换
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 应用透视变换
        warped = cv2.warpPerspective(frame, M, output_size)
        out.write(warped)

        frame_count += 1
        if frame_count % 30 == 0 or frame_count == 1:
            progress = 100 * frame_count / total_frames
            print(f"  进度: {frame_count}/{total_frames} ({progress:.1f}%)")

    cap.release()
    out.release()

    print("\n✓ 视频转换完成!")
    print(f"  处理了: {frame_count}帧")

    # 验证输出文件
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  输出文件大小: {file_size:.2f}MB")

        # 验证输出视频可以读取
        print("\n【步骤4: 验证输出视频】")
        verify_cap = cv2.VideoCapture(output_path)
        if verify_cap.isOpened():
            verify_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            verify_fps = verify_cap.get(cv2.CAP_PROP_FPS)
            verify_w = int(verify_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            verify_h = int(verify_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            verify_cap.release()

            print("✓ 输出视频有效!")
            print(f"  帧数: {verify_frames}")
            print(f"  分辨率: {verify_w}×{verify_h}")
            print(f"  帧率: {verify_fps:.2f}FPS")
            return True
        else:
            print("❌ 无法读取输出视频")
            return False
    else:
        print("❌ 输出文件不存在")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频透视变换测试")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--homography", type=str, required=True, help="Homography JSON路径")
    parser.add_argument("--output", type=str, default="warped_test.mp4", help="输出视频路径")
    parser.add_argument("--width", type=int, default=360, help="输出宽度 (默认: 360)")
    parser.add_argument("--height", type=int, default=2400, help="输出高度 (默认: 2400)")

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.video):
        print(f"❌ 视频不存在: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.homography):
        print(f"❌ Homography JSON不存在: {args.homography}")
        sys.exit(1)

    print("=" * 70)
    print("视频透视变换测试")
    print("=" * 70)

    output_size = (args.width, args.height)
    success = transform_video_simple(args.video, args.homography, args.output, output_size)

    if success:
        print("\n" + "=" * 70)
        print("✓ 转换成功! 可以尝试播放输出视频")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ 转换失败")
        print("=" * 70)
        sys.exit(1)
