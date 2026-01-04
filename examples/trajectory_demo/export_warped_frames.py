"""
export_warped_frames.py

导出warped视频的帧为静态图像
"""

import json
import numpy as np
import cv2
from pathlib import Path


def export_frames(video_path, homography_path, output_dir, output_size=(360, 2400), sample_rate=5):
    """导出warped帧为静态图像"""
    
    print("【加载Homography矩阵】")
    with open(homography_path) as f:
        data = json.load(f)
    
    H = np.array(data['homography_matrix'], dtype=np.float32)
    world_points = data['world_points']
    
    # 计算变换矩阵
    min_x = min(w[0] for w in world_points)
    max_x = max(w[0] for w in world_points)
    min_y = min(w[1] for w in world_points)
    max_y = max(w[1] for w in world_points)
    
    world_width = max_x - min_x
    world_height = max_y - min_y
    
    A = np.array([
        [world_width / output_size[0], 0, min_x],
        [0, -world_height / output_size[1], max_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    H_inv = np.linalg.inv(H)
    M = H_inv @ A
    
    # 读取视频
    print("【读取视频并导出帧】")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    print(f"采样率: 每{sample_rate}帧导出1帧")
    
    frame_count = 0
    exported = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 按采样率导出
        if frame_count % sample_rate == 0:
            warped = cv2.warpPerspective(frame, M, output_size)
            output_file = output_path / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(output_file), warped)
            exported += 1
            
            if exported % 10 == 0 or exported == 1:
                print(f"  导出: {exported}张 (处理到第{frame_count}帧)")
    
    cap.release()
    
    print(f"\n✓ 导出完成!")
    print(f"  总帧数: {total_frames}")
    print(f"  已导出: {exported}张图像")
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='导出warped视频帧为图像')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--homography', type=str, required=True)
    parser.add_argument('--output', type=str, default='warped_frames')
    parser.add_argument('--sample', type=int, default=5, help='采样率 (每N帧导出1帧)')
    parser.add_argument('--width', type=int, default=360)
    parser.add_argument('--height', type=int, default=2400)
    
    args = parser.parse_args()
    
    print("="*70)
    print("导出warped视频帧")
    print("="*70)
    
    export_frames(
        args.video,
        args.homography,
        args.output,
        (args.width, args.height),
        args.sample
    )
