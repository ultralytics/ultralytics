"""
perspective_transform_video.py.

对视频应用Homography透视变换，得到鸟瞰图效果

使用示例：
python perspective_transform_video.py \
  --video ../../videos/Homograph_Teset_FullScreen.mp4 \
  --homography ../../calibration/Homograph_Teset_FullScreen_homography.json \
  --output ../../videos/Homograph_Teset_FullScreen_warped.mp4
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
    return H, data["pixel_points"], data["world_points"]


def compute_output_to_pixel_matrix(H, world_bounds, output_size):
    """计算从输出坐标到像素坐标的映射矩阵.

    推导过程：
    1. 定义输出图像坐标(out_x, out_y) → 世界坐标(world_x, world_y)的线性映射
    输出(0, 0) → 世界(min_x, max_y) 输出(out_w, 0) → 世界(max_x, max_y) 输出(0, out_h) → 世界(min_x, min_y)

    2. 构造映射矩阵A：world_coord = A @ out_coord

    3. 使用已知的H矩阵（像素→世界）的逆矩阵：pixel_coord = H_inv @ world_coord

    4. 合并：pixel_coord = H_inv @ A @ out_coord = M @ out_coord

    返回M，这就是warpPerspective需要的矩阵
    """
    min_x, max_x, min_y, max_y = world_bounds
    out_w, out_h = output_size

    # 矩阵A：输出坐标 → 世界坐标
    world_width = max_x - min_x
    world_height = max_y - min_y

    A = np.array([[world_width / out_w, 0, min_x], [0, -world_height / out_h, max_y], [0, 0, 1]], dtype=np.float32)

    # H_inv：世界坐标 → 像素坐标
    H_inv = np.linalg.inv(H)

    # M = H_inv @ A：输出坐标 → 像素坐标
    M = H_inv @ A

    return M


def transform_frame(frame, M):
    """对单个帧进行透视变换.

    使用OpenCV的warpPerspective（快速C++实现）

    参数：
    - frame: 输入帧
    - M: 从输出坐标到像素坐标的映射矩阵（3x3）
    """
    out_w, out_h = 180, 1200

    # M矩阵已经是：[output_pixel] -> [source_pixel]
    # 这正是warpPerspective需要的
    warped = cv2.warpPerspective(frame, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped


def transform_first_frame(video_path, homography_path, output_image_path=None):
    """对视频第一帧进行透视变换，生成验证图片."""
    H, _pixel_points, world_points = load_homography(homography_path)

    # 计算世界坐标范围
    min_x = min(w[0] for w in world_points)
    max_x = max(w[0] for w in world_points)
    min_y = min(w[1] for w in world_points)
    max_y = max(w[1] for w in world_points)
    world_bounds = (min_x, max_x, min_y, max_y)
    output_size = (180, 1200)

    # 计算正确的映射矩阵
    M = compute_output_to_pixel_matrix(H, world_bounds, output_size)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ 无法读取视频第一帧: {video_path}")
        return None

    print(f"原始帧尺寸: {frame.shape[1]}x{frame.shape[0]} 像素")

    # 进行透视变换
    warped = transform_frame(frame, M)
    print(f"变换后尺寸: {warped.shape[1]}x{warped.shape[0]} 像素")

    if output_image_path:
        os.makedirs(os.path.dirname(output_image_path) if os.path.dirname(output_image_path) else ".", exist_ok=True)
        cv2.imwrite(output_image_path, warped)
        print(f"✓ 变换后的第一帧已保存: {output_image_path}")

    return warped


def transform_video(video_path, homography_path, output_video_path, display_progress=True):
    """对整个视频应用透视变换."""
    H, _pixel_points, world_points = load_homography(homography_path)

    # 计算世界坐标范围
    min_x = min(w[0] for w in world_points)
    max_x = max(w[0] for w in world_points)
    min_y = min(w[1] for w in world_points)
    max_y = max(w[1] for w in world_points)
    world_bounds = (min_x, max_x, min_y, max_y)
    output_size = (180, 1200)

    # 计算正确的映射矩阵
    M = compute_output_to_pixel_matrix(H, world_bounds, output_size)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()
    if not ret:
        print(f"❌ 无法读取视频: {video_path}")
        cap.release()
        return False

    out_w, out_h = output_size
    print(f"输入视频: {frame.shape[1]}x{frame.shape[0]}, {fps:.2f}FPS, {total_frames}帧")
    print(f"输出视频: {out_w}x{out_h}, {fps:.2f}FPS")

    # 创建VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 应用透视变换
        warped = transform_frame(frame, M)
        out.write(warped)

        frame_count += 1
        if display_progress and frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")

    cap.release()
    out.release()

    print(f"✓ 变换后的视频已保存: {output_video_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对视频应用Homography透视变换")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--homography", type=str, required=True, help="Homography矩阵JSON文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出视频路径")
    parser.add_argument("--verify-only", action="store_true", help="只生成第一帧验证图片，不处理整个视频")

    args = parser.parse_args()

    if args.verify_only:
        # 只生成第一帧验证图
        verify_path = args.output.replace(".mp4", "_verify.jpg")
        transform_first_frame(args.video, args.homography, verify_path)
    else:
        # 先生成验证图
        verify_path = args.output.replace(".mp4", "_verify.jpg")
        print("=" * 60)
        print("第一步：生成第一帧验证图...")
        print("=" * 60)
        transform_first_frame(args.video, args.homography, verify_path)

        print("\n" + "=" * 60)
        print("第二步：处理整个视频...")
        print("=" * 60)
        transform_video(args.video, args.homography, args.output)

        print("\n" + "=" * 60)
        print("✓ 处理完成！")
        print("=" * 60)
        print(f"验证图片: {verify_path}")
        print(f"变换视频: {args.output}")
