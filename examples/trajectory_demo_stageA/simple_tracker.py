"""
simple_tracker.py

极简版本的对象追踪 - 基于 examples/object_tracking.ipynb 的思路
只有核心功能，没有复杂的模块化设计

用法：
python examples/trajectory_demo/simple_tracker.py --source "path/to/video.mp4"
"""
from collections import defaultdict
import json
import argparse
import os

try:
    import cv2
except ImportError:
    print("Please install opencv-python: pip install opencv-python")
    exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)


def main(source: str, weights: str = "yolo11n.pt", output_dir: str = "runs/trajectory_demo"):
    """
    极简追踪脚本
    
    参数：
    - source: 视频文件路径
    - weights: YOLO 模型权重
    - output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model: {weights}")
    model = YOLO(weights)
    
    print(f"Opening video: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {source}")
        return
    
    # 简单的数据结构：按 ID 存轨迹
    track_history = defaultdict(list)
    all_detections_by_frame = {}
    
    frame_idx = 0
    print("\nProcessing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\nProcessing complete. Total frames: {frame_idx}")
            break
        
        # 调用 YOLO 追踪（persist=True 保持 ID 连续）
        results = model.track(frame, persist=True)
        
        if results and results[0].boxes.is_track:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id
            clss = results[0].boxes.cls
            confs = results[0].boxes.conf
            
            if track_ids is not None:
                track_ids = track_ids.int().cpu().tolist()
                clss = clss.cpu().tolist() if clss is not None else [None] * len(boxes)
                confs = confs.cpu().tolist() if confs is not None else [None] * len(boxes)
                
                # 保存本帧的所有检测
                frame_dets = []
                
                for i, (box, tid, cls_id, conf) in enumerate(zip(boxes, track_ids, clss, confs)):
                    x1, y1, x2, y2 = box.tolist()
                    
                    # 计算中心点（anchor point）
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    
                    # 添加到该 ID 的轨迹历史
                    sample = {
                        'x': float(cx),
                        'y': float(cy),
                        't': frame_idx,
                        'cls': int(cls_id) if cls_id is not None else None,
                        'conf': float(conf) if conf is not None else None,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    }
                    track_history[tid].append(sample)
                    
                    # 添加到本帧检测列表
                    frame_dets.append({
                        'id': int(tid),
                        'cx': float(cx),
                        'cy': float(cy),
                        'cls': int(cls_id) if cls_id is not None else None,
                        'conf': float(conf) if conf is not None else None,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
                
                all_detections_by_frame[frame_idx] = frame_dets
        
        # 进度输出
        if (frame_idx + 1) % 30 == 0:
            print(f"  Processed {frame_idx + 1} frames, {len(track_history)} active tracks")
        
        frame_idx += 1
    
    cap.release()
    
    # 保存结果到 JSON
    print("\nSaving results...")
    
    # 保存轨迹
    tracks_path = os.path.join(output_dir, 'tracks_simple.json')
    with open(tracks_path, 'w', encoding='utf-8') as f:
        json.dump(dict(track_history), f, indent=2, ensure_ascii=False)
    print(f"✓ Saved tracks to: {tracks_path}")
    
    # 保存按帧的检测
    detections_path = os.path.join(output_dir, 'detections_simple.json')
    with open(detections_path, 'w', encoding='utf-8') as f:
        json.dump(all_detections_by_frame, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved detections by frame to: {detections_path}")
    
    # 打印统计信息
    print(f"\nStatistics:")
    print(f"  Total frames processed: {frame_idx}")
    print(f"  Total track IDs: {len(track_history)}")
    for tid, samples in sorted(track_history.items()):
        print(f"    Track {tid}: {len(samples)} samples (class: {samples[0].get('cls')})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple YOLO object tracker')
    parser.add_argument('--source', type=str, required=True, help='Video file path')
    parser.add_argument('--weights', type=str, default='yolo11n.pt', help='YOLO weights')
    parser.add_argument('--output', type=str, default='runs/trajectory_demo', help='Output directory')
    
    args = parser.parse_args()
    main(args.source, args.weights, args.output)
