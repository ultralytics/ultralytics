#!/usr/bin/env python3
"""
homography_transform_video.py

将视频应用Homography透视变换，得到鸟瞰图效果
使用手工逐像素变换+双线性插值方法（已验证有效）

用法1（自动生成输出文件名）：
  python homography_transform_video.py \\
    --input videos/Homograph_Teset_FullScreen.mp4 \\
    --homography calibration/Homograph_Teset_FullScreen_homography.json
  
  输出文件名自动为: results/warped_videos/Homograph_Teset_FullScreen_YYYYMMDD_HHMMSS.mp4

用法2（指定输出文件）：
  python homography_transform_video.py \\
    --input videos/video.mp4 \\
    --homography calibration/homography.json \\
    --output my_output.mp4

用法3（仅验证第一帧）：
  python homography_transform_video.py \\
    --input videos/video.mp4 \\
    --homography calibration/homography.json \\
    --verify-only

核心特性：
- 手工逐像素变换（规避OpenCV warpPerspective的bug）
- 双线性插值确保图像质量
- 自动生成时间戳文件名：原始视频名_YYYYMMDD_HHMMSS.mp4
- 进度显示
- 验证第一帧
"""

import cv2
import numpy as np
import json
import argparse
import os
from pathlib import Path
from datetime import datetime


def load_homography(json_path):
    """加载Homography矩阵"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    H = np.array(data['homography_matrix'], dtype=np.float32)
    return H, data['world_points']


def compute_transformation_matrix(H, world_points, output_size):
    """
    计算完整的变换矩阵：输出坐标 -> 像素坐标
    
    推导过程：
    1. H矩阵：像素 -> 世界 (已知)
    2. A矩阵：输出图像 -> 世界 (构造)
    3. M矩阵：输出图像 -> 像素 = H_inv @ A
    
    参数：
    - H: Homography矩阵 (3x3)
    - world_points: 世界坐标参考点 (list of [x, y])
    - output_size: 输出图像尺寸 (width, height)
    
    返回：
    - M: 变换矩阵 (3x3)
    """
    
    # Step 1: 计算世界坐标范围
    min_x = min(w[0] for w in world_points)
    max_x = max(w[0] for w in world_points)
    min_y = min(w[1] for w in world_points)
    max_y = max(w[1] for w in world_points)
    
    # Step 2: 构造A矩阵 (输出坐标 -> 世界坐标)
    out_w, out_h = output_size
    world_width = max_x - min_x
    world_height = max_y - min_y
    
    A = np.array([
        [world_width / out_w, 0, min_x],
        [0, -world_height / out_h, max_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Step 3: 计算H的逆矩阵 (世界 -> 像素)
    H_inv = np.linalg.inv(H)
    
    # Step 4: 合成M矩阵 (输出 -> 像素)
    M = H_inv @ A
    
    return M, (min_x, max_x, min_y, max_y)


def transform_frame_manual(frame, M, output_size):
    """
    对单个帧进行手工逐像素变换（使用双线性插值）
    
    这个方法规避了OpenCV warpPerspective的数值稳定性问题
    
    参数：
    - frame: 输入帧 (HxWx3 uint8)
    - M: 变换矩阵 (3x3)
    - output_size: 输出尺寸 (width, height)
    
    返回：
    - warped: 变换后的帧
    """
    
    out_w, out_h = output_size
    warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    h, w = frame.shape[:2]
    
    for out_y in range(out_h):
        for out_x in range(out_w):
            # 将输出像素坐标映射到源图像坐标
            out_h_vec = np.array([float(out_x), float(out_y), 1.0])
            pix_h_vec = M @ out_h_vec
            pix_coord = pix_h_vec[:2] / pix_h_vec[2]
            
            src_x, src_y = pix_coord
            
            # 双线性插值
            if 0 <= src_x < w-1 and 0 <= src_y < h-1:
                x0, y0 = int(np.floor(src_x)), int(np.floor(src_y))
                x1, y1 = x0 + 1, y0 + 1
                
                # 获取4个相邻像素
                p00 = frame[y0, x0].astype(np.float32)
                p10 = frame[y0, x1].astype(np.float32)
                p01 = frame[y1, x0].astype(np.float32)
                p11 = frame[y1, x1].astype(np.float32)
                
                # 计算权重
                wx = src_x - x0
                wy = src_y - y0
                
                # 双线性插值
                p0 = p00 * (1 - wx) + p10 * wx
                p1 = p01 * (1 - wx) + p11 * wx
                pixel = p0 * (1 - wy) + p1 * wy
                
                warped[out_y, out_x] = np.clip(pixel, 0, 255).astype(np.uint8)
    
    return warped


def transform_first_frame_verify(video_path, homography_path, output_size):
    """
    变换并保存第一帧用于验证
    """
    
    H, world_points = load_homography(homography_path)
    M, world_bounds = compute_transformation_matrix(H, world_points, output_size)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"[ERROR] Cannot read first frame from {video_path}")
        return False
    
    warped = transform_frame_manual(frame, M, output_size)
    
    # 保存验证图片
    output_dir = os.path.dirname(video_path)
    verify_path = os.path.join(output_dir, 'frame_verify.jpg')
    cv2.imwrite(verify_path, warped)
    
    mean_val = warped.mean()
    print(f"[VERIFY] First frame transformed:")
    print(f"  Output size: {warped.shape}")
    print(f"  Value range: {warped.min()}-{warped.max()}")
    print(f"  Mean value: {mean_val:.2f}")
    print(f"  Saved to: {verify_path}")
    
    if mean_val > 5:
        print("[OK] First frame has visible content")
        return True
    else:
        print("[ERROR] First frame is all black")
        return False


def transform_video(video_path, homography_path, output_path, output_size=(180, 1200)):
    """
    对整个视频进行Homography透视变换
    
    参数：
    - video_path: 输入视频文件路径
    - homography_path: Homography JSON文件路径
    - output_path: 输出视频文件路径
    - output_size: 输出图像尺寸 (width, height)
    """
    
    print(f"[INFO] Loading homography from {homography_path}")
    H, world_points = load_homography(homography_path)
    
    print(f"[INFO] Computing transformation matrix")
    M, world_bounds = compute_transformation_matrix(H, world_points, output_size)
    
    print(f"[INFO] Opening video {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Video properties:")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Output size: {output_size[0]}x{output_size[1]}")
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)
    
    if not out.isOpened():
        print(f"[ERROR] Cannot create output video: {output_path}")
        cap.release()
        return False
    
    print(f"[INFO] Processing frames...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 变换当前帧
        warped = transform_frame_manual(frame, M, output_size)
        
        # 写入输出视频
        out.write(warped)
        
        frame_count += 1
        if frame_count % 10 == 0 or frame_count == total_frames:
            progress = 100 * frame_count / total_frames
            print(f"  [{progress:6.2f}%] Frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    print(f"[OK] Video transformation complete")
    print(f"  Output: {output_path}")
    print(f"  Frames processed: {frame_count}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Apply Homography perspective transform to video')
    parser.add_argument('--input', '-i', required=True, help='Input video file')
    parser.add_argument('--homography', '-H', required=True, help='Homography JSON file')
    parser.add_argument('--output', '-o', default=None, help='Output video file (auto-generated if not specified)')
    parser.add_argument('--output-dir', '-d', default='results/warped_videos', help='Output directory (default: results/warped_videos)')
    parser.add_argument('--width', type=int, default=180, help='Output width (default: 180)')
    parser.add_argument('--height', type=int, default=1200, help='Output height (default: 1200)')
    parser.add_argument('--verify-only', action='store_true', help='Only verify first frame')
    
    args = parser.parse_args()
    
    output_size = (args.width, args.height)
    
    # 验证第一帧
    if not transform_first_frame_verify(args.input, args.homography, output_size):
        print("[ERROR] First frame verification failed")
        return 1
    
    if args.verify_only:
        print("[INFO] Verify-only mode: skipping full video processing")
        return 0
    
    # 如果没有指定output，自动生成
    if args.output is None:
        # 提取原始视频名（不含扩展名）
        input_path = Path(args.input)
        video_name = input_path.stem  # 不含扩展名
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名：原始视频名_YYYYMMDD_HHMMSS.mp4
        output_filename = f"{video_name}_{timestamp}.mp4"
        args.output = str(output_dir / output_filename)
        
        print(f"[INFO] Auto-generated output file: {args.output}")
    
    # 处理整个视频
    success = transform_video(args.input, args.homography, args.output, output_size)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
