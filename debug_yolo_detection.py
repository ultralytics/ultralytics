"""
调试YOLO检测 - 生成带检测框的视频
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from ultralytics import YOLO

# 参数
warped_video_path = "results/20251219_000818/2_warped_video/warped.mp4"
output_video_path = "results/yolo_detection_debug.mp4"

print(f"检查视频文件...")
if not Path(warped_video_path).exists():
    print(f"✗ 视频文件不存在: {warped_video_path}")
    sys.exit(1)

# 打开视频
cap = cv2.VideoCapture(warped_video_path)
if not cap.isOpened():
    print(f"✗ 无法打开视频: {warped_video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✓ 视频打开成功")
print(f"  分辨率: {frame_width}×{frame_height}")
print(f"  总帧数: {total_frames}")
print(f"  FPS: {fps}")

# 检查第一帧
ret, frame = cap.read()
if ret:
    print(f"✓ 成功读取第一帧")
    print(f"  帧形状: {frame.shape}")
    print(f"  数据类型: {frame.dtype}")
    print(f"  像素范围: [{frame.min()}, {frame.max()}]")
    mean_val = np.mean(frame)
    print(f"  平均像素值: {mean_val:.2f}")
    if mean_val < 10:
        print(f"  ⚠️  警告：帧几乎全黑！")
else:
    print(f"✗ 无法读取第一帧")
    cap.release()
    sys.exit(1)

# 重置到开始
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# 初始化YOLO
print(f"\n加载YOLO模型...")
model = YOLO('yolo11n.pt')

# 创建输出视频
print(f"\n生成检测框视频...")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print(f"✗ 无法创建输出视频")
    cap.release()
    sys.exit(1)

frame_count = 0
detection_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 运行YOLO检测
    results = model(frame, verbose=False, conf=0.45)
    
    # 获取检测结果
    annotated_frame = results[0].plot()
    
    # 统计检测
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        detection_count += 1
        num_boxes = len(results[0].boxes)
        print(f"Frame {frame_count}/{total_frames}: 检测到 {num_boxes} 个物体")
    else:
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}/{total_frames}: 无检测")
    
    # 写入输出
    out.write(annotated_frame)
    
    if frame_count >= 30:  # 只处理前30帧用于调试
        print(f"已处理30帧，停止（用于快速调试）")
        break

cap.release()
out.release()

print(f"\n统计:")
print(f"  处理帧数: {frame_count}")
print(f"  有检测的帧: {detection_count}")
print(f"  检测率: {100*detection_count/frame_count:.1f}%")
print(f"\n✓ 检测框视频已保存: {output_video_path}")
