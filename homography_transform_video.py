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

# Import shared homography transformation utilities
from homography_transform_utils import load_homography, compute_transformation_matrix, transform_frame_manual


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
