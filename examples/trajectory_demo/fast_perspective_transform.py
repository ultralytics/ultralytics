"""
fast_perspective_transform.py.

快速版本：使用OpenCV的cv2.remap进行透视变换
比手工逐像素插值快得多
"""

import argparse
import json

import cv2
import numpy as np


def load_homography(json_path):
    with open(json_path) as f:
        data = json.load(f)
    H = np.array(data["homography_matrix"], dtype=np.float32)
    return H, data["pixel_points"], data["world_points"]


def compute_remap_matrices(H, world_bounds, output_size, input_shape):
    """计算用于cv2.remap的映射表 这比手工逐像素插值快得多."""
    _h, _w = input_shape[:2]
    min_x, max_x, min_y, max_y = world_bounds
    out_w, out_h = output_size

    # 创建输出坐标网格
    x_out = np.arange(out_w, dtype=np.float32)
    y_out = np.arange(out_h, dtype=np.float32)
    x_out, y_out = np.meshgrid(x_out, y_out)

    # 将输出坐标转换为世界坐标
    world_width = max_x - min_x
    world_height = max_y - min_y

    world_x = min_x + (x_out / out_w) * world_width
    world_y = max_y - (y_out / out_h) * world_height

    # 将世界坐标转换为像素坐标（使用H的逆矩阵）
    H_inv = np.linalg.inv(H)

    # 齐次坐标：[x, y, 1]
    world_coords = np.stack([world_x, world_y, np.ones_like(world_x)], axis=2)
    world_coords_flat = world_coords.reshape(-1, 3).T  # 3 x (h*w)

    # 应用H_inv
    pixel_coords = H_inv @ world_coords_flat  # 3 x (h*w)

    # 去齐次坐标
    pixel_x = pixel_coords[0] / pixel_coords[2]
    pixel_y = pixel_coords[1] / pixel_coords[2]

    # 重塑为网格
    pixel_x = pixel_x.reshape(out_h, out_w).astype(np.float32)
    pixel_y = pixel_y.reshape(out_h, out_w).astype(np.float32)

    return pixel_x, pixel_y


def transform_video_fast(video_path, homography_path, output_video_path):
    """使用cv2.remap快速进行透视变换."""
    H, _pixel_points, world_points = load_homography(homography_path)

    # 世界坐标范围
    min_x = min(w[0] for w in world_points)
    max_x = max(w[0] for w in world_points)
    min_y = min(w[1] for w in world_points)
    max_y = max(w[1] for w in world_points)
    world_bounds = (min_x, max_x, min_y, max_y)
    output_size = (180, 1200)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()
    if not ret:
        print(f"❌ 无法读取视频: {video_path}")
        cap.release()
        return False

    # 计算映射表（对所有帧都是一样的）
    print("计算映射表...")
    pixel_x, pixel_y = compute_remap_matrices(H, world_bounds, output_size, frame.shape)

    out_w, out_h = output_size
    print(f"输入视频: {frame.shape[1]}x{frame.shape[0]}, {fps:.2f}FPS, {total_frames}帧")
    print(f"输出视频: {out_w}x{out_h}, {fps:.2f}FPS")

    # 创建VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

    frame_count = 0
    import time

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用remap进行透视变换
        warped = cv2.remap(frame, pixel_x, pixel_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out.write(warped)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            remaining = (total_frames - frame_count) / fps_actual if fps_actual > 0 else 0
            progress = (frame_count / total_frames) * 100
            print(
                f"进度: {frame_count}/{total_frames} ({progress:.1f}%) - 速度: {fps_actual:.1f} fps - 剩余: {remaining:.0f}s"
            )

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    print(f"✓ 完成！耗时: {elapsed:.1f}s, 速度: {total_frames / elapsed:.1f} fps")
    print(f"✓ 变换后的视频已保存: {output_video_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="快速透视变换")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--homography", type=str, required=True, help="Homography矩阵JSON文件")
    parser.add_argument("--output", type=str, required=True, help="输出视频路径")

    args = parser.parse_args()

    transform_video_fast(args.video, args.homography, args.output)
