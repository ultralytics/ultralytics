"""
visualize_collision_events.py

可视化碰撞事件在视频中的具体帧
============================================

功能：
1. 加载near_misses.json（碰撞事件）
2. 从原视频提取相应帧
3. 在帧上绘制：
   - YOLO检测框 + ID
   - 接触点（3个点）
   - 点之间的连线
   - 距离值 + 接触类型
4. 保存可视化结果（图片或视频）

使用示例：
python examples/trajectory_demo/visualize_collision_events.py \
  --near-misses runs/trajectory_demo/xxx/near_misses.json \
  --tracks runs/trajectory_demo/xxx/tracks.json \
  --video videos/Homograph_Teset_FullScreen.mp4 \
  --output collision_frames/ \
  --top-k 10

输出：
  collision_frames/
  ├─ collision_event_001_frame_45_obj1_vs_obj2.jpg
  ├─ collision_event_002_frame_78_obj3_vs_obj5.jpg
  └─ collision_summary.txt
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import math


def load_data(near_misses_path, tracks_path):
    """加载近miss事件和轨迹数据"""
    with open(near_misses_path, 'r') as f:
        near_misses = json.load(f)
    with open(tracks_path, 'r') as f:
        tracks = json.load(f)
    return near_misses, tracks


def get_object_info_at_frame(tracks, obj_id, frame_num):
    """获取指定物体在某一帧的信息"""
    if str(obj_id) not in tracks:
        return None
    
    trajectory = tracks[str(obj_id)]
    
    # 找到时间戳最接近frame_num的记录
    best_sample = None
    min_diff = float('inf')
    
    for sample in trajectory:
        diff = abs(sample['t'] - frame_num)
        if diff < min_diff:
            min_diff = diff
            best_sample = sample
    
    return best_sample


def draw_detection_box(frame, sample, obj_id, color=(0, 255, 0)):
    """在帧上绘制YOLO检测框"""
    if sample is None or sample.get('bbox') is None:
        return frame
    
    x1, y1, x2, y2 = sample['bbox']
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # 绘制检测框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # 绘制ID标签
    label = f"ID:{obj_id}"
    cv2.putText(frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame


def draw_contact_points(frame, sample, obj_id, color=(0, 255, 0), alpha=0.7):
    """在帧上绘制接触点"""
    if sample is None or sample.get('contact_points_pixel') is None:
        return frame
    
    points = sample['contact_points_pixel']
    point_names = ['front', 'center', 'back']
    
    # 创建一个透明层用于绘制
    overlay = frame.copy()
    
    for i, (x, y) in enumerate(points):
        x, y = int(x), int(y)
        
        # 绘制圆形（接触点）
        cv2.circle(overlay, (x, y), 8, color, -1)
        cv2.circle(frame, (x, y), 8, color, 2)  # 外框
        
        # 标注点的名字
        label = point_names[i]
        cv2.putText(frame, label, (x+15, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 绘制三个点之间的连线
    for i in range(len(points) - 1):
        x1, y1 = int(points[i][0]), int(points[i][1])
        x2, y2 = int(points[i+1][0]), int(points[i+1][1])
        cv2.line(frame, (x1, y1), (x2, y2), color, 1)
    
    return frame


def draw_collision_info(frame, near_miss_event, obj1_sample, obj2_sample, fps=30.0):
    """在帧上绘制碰撞信息（距离、TTC、接触类型）"""
    
    # 右上角显示碰撞信息
    info_lines = []
    
    obj1_id = near_miss_event['id1']
    obj2_id = near_miss_event['id2']
    distance = near_miss_event['distance']
    ttc = near_miss_event['ttc']
    
    info_lines.append(f"Collision: Object {obj1_id} <-> Object {obj2_id}")
    info_lines.append(f"Distance: {distance:.2f}")
    
    if 'closest_point_pair' in near_miss_event:
        cpp = near_miss_event['closest_point_pair']
        pt1_type = cpp['obj1_point_type']
        pt2_type = cpp['obj2_point_type']
        info_lines.append(f"Contact: {pt1_type} <-> {pt2_type}")
    
    if ttc is not None:
        info_lines.append(f"TTC: {ttc:.2f}s")
        if ttc < 0.5:
            risk_level = "CRITICAL"
        elif ttc < 2.0:
            risk_level = "HIGH RISK"
        else:
            risk_level = "MEDIUM"
        info_lines.append(f"Risk: {risk_level}")
    
    # 绘制信息文本（右上角）
    h, w = frame.shape[:2]
    x_offset = w - 350
    y_offset = 30
    line_height = 30
    
    # 背景矩形
    box_height = len(info_lines) * line_height + 20
    cv2.rectangle(frame, (x_offset - 10, y_offset - 20),
                  (w - 10, y_offset + box_height),
                  (0, 0, 0), -1)
    
    # 文字
    for i, line in enumerate(info_lines):
        color = (0, 0, 255) if "CRITICAL" in line or "HIGH" in line else (0, 255, 255)
        cv2.putText(frame, line, (x_offset, y_offset + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame


def visualize_collision_events(video_path, near_misses, tracks, output_dir, top_k=10):
    """
    提取并可视化最严重的碰撞事件帧
    
    参数：
    - video_path: 原视频路径
    - near_misses: 碰撞事件列表
    - tracks: 轨迹数据
    - output_dir: 输出目录
    - top_k: 提取最严重的前k个事件
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 按危险程度排序（TTC越小越危险）
    sorted_events = sorted(
        near_misses,
        key=lambda x: (
            x.get('ttc') if x.get('ttc') is not None else float('inf'),
            -x['distance']  # 同TTC下，距离越小越危险
        )
    )
    
    print(f"\n{'='*60}")
    print(f"Visualizing Top {min(top_k, len(sorted_events))} Collision Events")
    print(f"{'='*60}\n")
    
    saved_frames = []
    
    for event_idx, event in enumerate(sorted_events[:top_k]):
        frame_num = int(event['timestamp'])
        obj1_id = event['id1']
        obj2_id = event['id2']
        distance = event['distance']
        ttc = event['ttc']
        
        # 读取帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠ Cannot read frame {frame_num}")
            continue
        
        # 获取两个物体的信息
        obj1_sample = get_object_info_at_frame(tracks, obj1_id, frame_num)
        obj2_sample = get_object_info_at_frame(tracks, obj2_id, frame_num)
        
        # 绘制物体1（蓝色）
        if obj1_sample:
            frame = draw_detection_box(frame, obj1_sample, obj1_id, color=(255, 0, 0))
            frame = draw_contact_points(frame, obj1_sample, obj1_id, color=(255, 0, 0))
        
        # 绘制物体2（绿色）
        if obj2_sample:
            frame = draw_detection_box(frame, obj2_sample, obj2_id, color=(0, 255, 0))
            frame = draw_contact_points(frame, obj2_sample, obj2_id, color=(0, 255, 0))
        
        # 绘制碰撞信息
        frame = draw_collision_info(frame, event, obj1_sample, obj2_sample, fps)
        
        # 保存帧
        risk_level = "CRITICAL" if ttc and ttc < 0.5 else ("HIGH" if ttc and ttc < 2.0 else "MED")
        output_filename = (
            f"collision_event_{event_idx+1:03d}_"
            f"frame_{frame_num:05d}_"
            f"obj{obj1_id}_vs_obj{obj2_id}_"
            f"{risk_level}.jpg"
        )
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), frame)
        saved_frames.append(output_path)
        
        # 打印信息
        ttc_str = f"{ttc:.2f}s" if ttc is not None else "N/A"
        print(f"✓ Event {event_idx+1}:")
        print(f"    Frame {frame_num} | Object {obj1_id} <-> {obj2_id}")
        print(f"    Distance: {distance:.2f} | TTC: {ttc_str} | {risk_level} RISK")
        
        if 'closest_point_pair' in event:
            cpp = event['closest_point_pair']
            print(f"    Contact: {cpp['obj1_point_type']} <-> {cpp['obj2_point_type']}")
        print()
    
    cap.release()
    
    # 生成总结报告
    summary_path = output_dir / "collision_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("COLLISION EVENTS VISUALIZATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total events visualized: {len(saved_frames)}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("Saved files:\n")
        for path in saved_frames:
            f.write(f"  - {path.name}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Color Coding:\n")
        f.write("  Object 1 (Blue):  Detection box + contact points\n")
        f.write("  Object 2 (Green): Detection box + contact points\n")
        f.write("  Info Box:         Distance, TTC, Contact type, Risk level\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n✓ Summary saved: {summary_path}")
    print(f"✓ All frames saved to: {output_dir}\n")
    
    return saved_frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize collision events on video frames')
    parser.add_argument('--near-misses', required=True, help='Path to near_misses.json')
    parser.add_argument('--tracks', required=True, help='Path to tracks.json')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', default='collision_frames', help='Output directory')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top events to visualize')
    
    args = parser.parse_args()
    
    # 加载数据
    near_misses, tracks = load_data(args.near_misses, args.tracks)
    
    # 可视化碰撞事件
    visualize_collision_events(args.video, near_misses, tracks, args.output, args.top_k)
