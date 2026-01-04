"""
verify_first_frame_transform.py

验证第一帧的Homography转换是否成功
"""

import json
import cv2
import numpy as np
from pathlib import Path
import os


def verify_first_frame():
    print("【验证第一帧Homography转换】\n")
    
    # 加载Homography矩阵
    with open('../../calibration/Homograph_Teset_FullScreen_homography.json') as f:
        data = json.load(f)
    
    H = np.array(data['homography_matrix'], dtype=np.float32)
    world_points = data['world_points']
    
    print("✓ Homography矩阵已加载")
    print(f"  矩阵大小: {H.shape}")
    
    # 读取原始视频的第一帧
    print("\n【读取视频】")
    cap = cv2.VideoCapture('../../videos/Homograph_Teset_FullScreen.mp4')
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ 无法读取视频")
        return False
    
    print(f"✓ 第一帧读取成功")
    print(f"  原始帧大小: {frame.shape}")
    print(f"  像素值范围: {frame.min()}-{frame.max()}")
    
    # 计算透视变换矩阵
    print("\n【计算变换矩阵】")
    min_x = min(w[0] for w in world_points)
    max_x = max(w[0] for w in world_points)
    min_y = min(w[1] for w in world_points)
    max_y = max(w[1] for w in world_points)
    
    output_size = (2400, 360)  # OpenCV需要 (宽, 高) 顺序
    world_width = max_x - min_x
    world_height = max_y - min_y
    
    A = np.array([
        [world_width / output_size[0], 0, min_x],
        [0, -world_height / output_size[1], max_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    H_inv = np.linalg.inv(H)
    M = H_inv @ A
    
    print(f"✓ 变换矩阵已计算")
    print(f"  输出大小: {output_size[0]}×{output_size[1]}")
    
    # 应用变换
    print("\n【应用Homography变换】")
    warped = cv2.warpPerspective(frame, M, output_size)
    
    print(f"✓ 变换成功")
    print(f"  Warped帧大小: {warped.shape}")
    print(f"  像素值范围: {warped.min()}-{warped.max()}")
    print(f"  平均像素值: {warped.mean():.2f}")
    
    # 检查是否全黑
    if warped.mean() < 5:
        print(f"⚠️  警告: 帧几乎全黑!")
        print(f"  这表示变换矩阵可能有问题")
        return False
    else:
        print(f"✓ 帧内容正常")
    
    # 保存第一帧
    output_file = '/workspace/ultralytics/videos/test_first_frame_warped.jpg'
    cv2.imwrite(output_file, warped)
    
    print(f"\n✓ Warped第一帧已保存:")
    print(f"  {output_file}")
    
    size = os.path.getsize(output_file)
    print(f"  文件大小: {size}字节")
    
    return True


if __name__ == '__main__':
    verify_first_frame()
