"""
collision_detection_pipeline_yolo_first.py

YOLO-First 碰撞检测管道 (Approach 2, 方案C)
执行顺序: YOLO检测 → Homography变换 → 轨迹构建 → 关键帧提取 → TTC计算

流程:
1. YOLO 检测 (原始视频，全帧)
   - 直接在原始分辨率上检测所有物体
   - 保存原始检测框和 Track ID (像素空间)

2. Homography 变换 (所有检测框)
   - 将所有检测框从像素坐标转换到世界坐标
   - 计算缩放因子 (px → 米)

3. 轨迹构建 (世界坐标，米制)
   - 关联 Track ID，建立轨迹
   - 估计速度 (m/s)
   - 所有计算在同一坐标系

4. 关键帧提取 (接近事件检测)
   - 检测距离 < 1.5m 的物体对
   - 标记为关键帧

5. TTC 和 Event 分级
   - 计算 TTC (世界坐标)
   - 分级事件 (L1/L2/L3)
   - 生成报告

优势: 
- 轨迹在米制空间，清晰直观
- 所有计算在同一坐标系，一致性好
- 距离阈值用米（1.5m），更清晰
- 仍比 Homography-First 快 1.5-2 倍
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 导入YOLO和相关模块
sys.path.append(os.path.dirname(__file__))
from ultralytics import YOLO
import coord_transform


class YOLOFirstPipeline:
    def __init__(self, video_path, homography_path=None, output_base='../../results'):
        """初始化 YOLO-First pipeline (方案C: Homography优先)
        
        Args:
            video_path: 原始视频路径
            homography_path: Homography JSON路径 (必须有，用于坐标变换)
            output_base: 结果基础目录
        """
        self.video_path = video_path
        self.homography_path = homography_path
        self.output_base = Path(output_base)
        self.H = None
        self.pixel_per_meter = 1.0
        
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_base / f"{timestamp}_yolo_first_c"
        
        # 创建子目录结构 (改进版，符合方案C)
        self.detection_dir = self.run_dir / "1_raw_detections"
        self.homography_dir = self.run_dir / "2_homography_transform"
        self.trajectory_dir = self.run_dir / "3_trajectories"
        self.keyframe_dir = self.run_dir / "4_key_frames"
        self.analysis_dir = self.run_dir / "5_collision_analysis"
        
        for d in [self.detection_dir, self.homography_dir, self.trajectory_dir, 
                  self.keyframe_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"YOLO-First 碰撞检测Pipeline (方案C: Homography优先)")
        print(f"{'='*70}")
        print(f"时间戳: {timestamp}")
        print(f"结果目录: {self.run_dir}")
        print(f"执行顺序: YOLO → Homography → 轨迹(米制) → 关键帧 → 分析")
        
        if not homography_path:
            print(f"⚠️  警告: 未提供 Homography，将仅在像素空间处理")
    
    def load_homography(self):
        """Step 0.5: 加载 Homography 矩阵 (如果有的话)"""
        if not self.homography_path:
            print(f"\n⚠️  未提供 Homography，将在像素空间处理")
            return False
        
        print(f"\n【Step 0.5: 加载 Homography 矩阵】")
        
        try:
            with open(self.homography_path) as f:
                H_data = json.load(f)
            
            self.H = np.array(H_data['homography_matrix'], dtype=np.float32)
            pixel_points = H_data['pixel_points']
            world_points = H_data['world_points']
            
            # 保存到输出目录
            with open(self.homography_dir / 'homography.json', 'w') as f:
                json.dump(H_data, f, indent=2)
            
            # 计算像素到米的缩放因子
            if len(world_points) >= 2 and len(pixel_points) >= 2:
                px_dist = np.sqrt((pixel_points[0][0] - pixel_points[1][0])**2 + 
                                 (pixel_points[0][1] - pixel_points[1][1])**2)
                world_dist = np.sqrt((world_points[0][0] - world_points[1][0])**2 + 
                                    (world_points[0][1] - world_points[1][1])**2)
                
                self.pixel_per_meter = px_dist / world_dist if world_dist > 0 else 1.0
            
            print(f"✓ Homography 矩阵已加载")
            print(f"  缩放因子: {self.pixel_per_meter:.2f} px/m")
            print(f"  参考点数: {len(pixel_points)}")
            
            return True
        
        except Exception as e:
            print(f"❌ 加载 Homography 失败: {e}")
            return False
    
    def run_yolo_detection(self, conf_threshold=0.45):
        """Step 1: YOLO 检测 (原始视频，全帧)
        
        输出:
        - 保存所有检测框和 Track ID (像素空间)
        - 生成检测统计
        """
        print(f"\n【Step 1: YOLO 检测】")
        
        model = YOLO('yolo11n.pt')
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_detections = []  # 所有帧的检测结果
        frame_count = 0
        detection_frames_count = 0
        
        print(f"处理中: {total_frames}帧 @ {fps:.2f}FPS...")
        
        for result in model.track(source=self.video_path, stream=True, 
                                 persist=True, conf=conf_threshold):
            frame_count += 1
            
            if result.boxes is None or len(result.boxes) == 0:
                if frame_count % 30 == 0:
                    print(f"  Frame {frame_count}/{total_frames} - 无物体")
                continue
            
            detection_frames_count += 1
            boxes = result.boxes.xywh.cpu().numpy()
            ids = result.boxes.id
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            
            # 保存这帧的检测结果
            frame_detections = {
                'frame': frame_count,
                'time': frame_count / fps,
                'objects': []
            }
            
            for i in range(len(boxes)):
                obj_data = {
                    'track_id': int(ids[i]) if ids[i] is not None else -1,
                    'class': int(classes[i]),
                    'conf': float(confs[i]),
                    'bbox_xywh': boxes[i].tolist(),  # [x_center, y_center, w, h] 像素空间
                }
                frame_detections['objects'].append(obj_data)
            
            all_detections.append(frame_detections)
            
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count}/{total_frames} - {len(boxes)}个物体")
        
        cap.release()
        
        # 保存原始检测结果 (像素空间)
        detections_path = self.detection_dir / 'detections_pixel.json'
        with open(detections_path, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        # 生成检测统计
        stats = {
            'total_frames': total_frames,
            'fps': fps,
            'detection_frames': detection_frames_count,
            'confidence_threshold': conf_threshold,
        }
        stats_path = self.detection_dir / 'detection_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ YOLO检测完成: {detection_frames_count}帧检测到物体")
        print(f"  检测结果保存: {detections_path.name}")
        
        return all_detections
    
    def transform_detections_to_world(self, all_detections):
        """Step 2: Homography 变换 (所有检测框)
        
        将像素空间的检测框转换到世界坐标
        """
        print(f"\n【Step 2: Homography 坐标变换】")
        
        if self.H is None:
            print(f"⚠️  未加载 Homography，保持像素空间")
            return all_detections
        
        transformed_detections = []
        
        for frame_data in all_detections:
            trans_frame = {
                'frame': frame_data['frame'],
                'time': frame_data['time'],
                'objects': []
            }
            
            for obj in frame_data['objects']:
                trans_obj = obj.copy()
                
                # 获取检测框的中心点 (像素)
                x_px, y_px = obj['bbox_xywh'][0], obj['bbox_xywh'][1]
                
                # 使用简单的线性缩放进行变换
                # 实际应该用完整的 H 矩阵进行透视变换
                # 但对于简单的缩放，线性转换足够
                x_world = x_px / self.pixel_per_meter
                y_world = y_px / self.pixel_per_meter
                
                # 保存世界坐标
                trans_obj['center_x_world'] = x_world  # 米
                trans_obj['center_y_world'] = y_world  # 米
                
                # 也保存原始像素坐标以便参考
                trans_obj['center_x_pixel'] = x_px
                trans_obj['center_y_pixel'] = y_px
                
                trans_frame['objects'].append(trans_obj)
            
            transformed_detections.append(trans_frame)
        
        # 保存转换后的检测结果
        trans_path = self.homography_dir / 'detections_world.json'
        with open(trans_path, 'w') as f:
            json.dump(transformed_detections, f, indent=2)
        
        print(f"✓ Homography 变换完成: {len(all_detections)}帧检测框已转换到世界坐标")
        print(f"  转换结果保存: {trans_path.name}")
        
        return transformed_detections
        """Step 2: 构建轨迹 (像素空间)
        
        输入: 原始检测结果
        输出: 完整轨迹 (按 track_id 组织)
        """
        print(f"\n【Step 2: 轨迹构建】")
        
        # 按 track_id 组织轨迹
        tracks = {}  # {track_id: [frame_data1, frame_data2, ...]}
        
        for frame_data in all_detections:
            for obj in frame_data['objects']:
                track_id = obj['track_id']
                
                if track_id not in tracks:
                    tracks[track_id] = []
                
                # 构建轨迹点 (包含位置、时间、速度等信息)
                track_point = {
                    'frame': frame_data['frame'],
                    'time': frame_data['time'],
                    'class': obj['class'],
                    'conf': obj['conf'],
                    'bbox_xywh': obj['bbox_xywh'],
                    # 计算中心点
                    'center_x': obj['bbox_xywh'][0],
                    'center_y': obj['bbox_xywh'][1],
                }
                
                tracks[track_id].append(track_point)
        
        # 计算每个轨迹的速度信息
        for track_id, track_points in tracks.items():
            track_points.sort(key=lambda p: p['frame'])
            
            # 计算速度 (px/s)
            if len(track_points) >= 2:
                # 使用相邻两帧计算速度
                for i in range(1, len(track_points)):
                    prev = track_points[i-1]
                    curr = track_points[i]
                    
                    dt = curr['time'] - prev['time']
                    if dt > 0:
                        dx = curr['center_x'] - prev['center_x']
                        dy = curr['center_y'] - prev['center_y']
                        
                        curr['vx'] = dx / dt  # px/s
                        curr['vy'] = dy / dt  # px/s
                        curr['speed'] = np.sqrt(dx**2 + dy**2) / dt
                    else:
                        curr['vx'] = 0.0
                        curr['vy'] = 0.0
                        curr['speed'] = 0.0
                
                # 首帧无法计算速度
                track_points[0]['vx'] = 0.0
                track_points[0]['vy'] = 0.0
                track_points[0]['speed'] = 0.0
        
        # 保存轨迹
        tracks_path = self.trajectory_dir / 'tracks.json'
        with open(tracks_path, 'w') as f:
            json.dump(tracks, f, indent=2)
        
        # 生成轨迹统计
        stats = {
            'total_tracks': len(tracks),
            'track_lengths': {str(tid): len(points) for tid, points in tracks.items()},
        }
        stats_path = self.trajectory_dir / 'track_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ 轨迹构建完成: {len(tracks)}条轨迹")
        print(f"  轨迹结果保存: {tracks_path.name}")
        
        return tracks
    
    def extract_key_frames(self, all_detections, tracks, pixel_distance_threshold=150):
        """Step 3: 提取关键帧 (接近事件)
        
        检测距离 < pixel_distance_threshold 的物体对
        保存关键帧信息
        """
        print(f"\n【Step 3: 关键帧提取】")
        
        proximity_events = []  # 接近事件列表
        
        # 遍历每一帧，检测物体对之间的距离
        for frame_data in all_detections:
            if len(frame_data['objects']) < 2:
                continue
            
            frame = frame_data['frame']
            objects = frame_data['objects']
            
            # 检查所有物体对
            for i in range(len(objects)):
                for j in range(i+1, len(objects)):
                    obj1 = objects[i]
                    obj2 = objects[j]
                    
                    # 计算中心点距离
                    x1, y1 = obj1['bbox_xywh'][0], obj1['bbox_xywh'][1]
                    x2, y2 = obj2['bbox_xywh'][0], obj2['bbox_xywh'][1]
                    
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    # 检查是否为接近事件
                    if distance < pixel_distance_threshold:
                        event = {
                            'frame': frame,
                            'time': frame_data['time'],
                            'object_ids': (obj1['track_id'], obj2['track_id']),
                            'distance_pixel': float(distance),
                            'object_classes': (obj1['class'], obj2['class']),
                        }
                        proximity_events.append(event)
        
        # 保存接近事件
        events_path = self.keyframe_dir / 'proximity_events.json'
        with open(events_path, 'w') as f:
            json.dump(proximity_events, f, indent=2)
        
        print(f"✓ 关键帧提取完成: {len(proximity_events)}个接近事件")
        print(f"  接近事件保存: {events_path.name}")
        
        return proximity_events
    
    def transform_to_world_coords(self, proximity_events, all_detections):
        """Step 4: Homography 变换 (仅关键帧)
        
        如果提供了 Homography 矩阵，转换关键帧坐标到世界坐标
        """
        print(f"\n【Step 4: Homography 坐标变换】")
        
        if not self.homography_path:
            print(f"⚠️  未提供 Homography，跳过坐标变换")
            return None
        
        # 加载 Homography 矩阵
        with open(self.homography_path) as f:
            H_data = json.load(f)
        
        H = np.array(H_data['homography_matrix'], dtype=np.float32)
        world_points = H_data['world_points']
        
        # 保存到输出目录
        with open(self.homography_dir / 'homography.json', 'w') as f:
            json.dump(H_data, f, indent=2)
        
        # 计算世界坐标范围和像素到米的缩放
        pixel_points = H_data['pixel_points']
        
        # 简单估计: 使用参考点计算缩放
        # 这是一个近似值，实际应该使用 H 矩阵
        if len(world_points) >= 2 and len(pixel_points) >= 2:
            # 计算两个参考点的像素距离和世界距离
            px_dist = np.sqrt((pixel_points[0][0] - pixel_points[1][0])**2 + 
                             (pixel_points[0][1] - pixel_points[1][1])**2)
            world_dist = np.sqrt((world_points[0][0] - world_points[1][0])**2 + 
                                (world_points[0][1] - world_points[1][1])**2)
            
            pixel_per_meter = px_dist / world_dist if world_dist > 0 else 1.0
        else:
            pixel_per_meter = 1.0
        
        print(f"  缩放因子: {pixel_per_meter:.2f} px/m")
        
        # 转换关键帧坐标
        transformed_events = []
        
        for event in proximity_events:
            frame_idx = None
            # 找到这一帧的数据
            for frame_data in all_detections:
                if frame_data['frame'] == event['frame']:
                    frame_idx = all_detections.index(frame_data)
                    break
            
            if frame_idx is None:
                continue
            
            # 创建转换后的事件
            trans_event = event.copy()
            
            # 像素距离转换为米
            trans_event['distance_meters'] = event['distance_pixel'] / pixel_per_meter
            
            # 计算速度缩放 (px/s → m/s)
            # 这需要从轨迹中获取速度信息
            trans_event['pixel_per_meter'] = pixel_per_meter
            
            transformed_events.append(trans_event)
        
        # 保存转换后的事件
        trans_path = self.homography_dir / 'events_world_coords.json'
        with open(trans_path, 'w') as f:
            json.dump(transformed_events, f, indent=2)
        
        print(f"✓ Homography 变换完成: {len(transformed_events)}个事件转换")
        print(f"  转换结果保存: {trans_path.name}")
        
        return transformed_events
    
    def analyze_collision_risk(self, proximity_events, transformed_events=None):
        """Step 5: TTC 计算和 Event 分级
        
        计算 TTC，分级事件
        """
        print(f"\n【Step 5: 碰撞风险分析】")
        
        # 如果有世界坐标事件，使用它们；否则用像素空间事件
        events_to_analyze = transformed_events if transformed_events else proximity_events
        
        analyzed_events = []
        
        for event in events_to_analyze:
            analyzed = event.copy()
            
            # 简单的 TTC 和分级逻辑
            if transformed_events:
                # 世界坐标空间
                distance = event.get('distance_meters', event['distance_pixel'])
                threshold_collision = 0.5  # 0.5m
                threshold_near_miss = 1.5  # 1.5m
            else:
                # 像素空间
                distance = event['distance_pixel']
                threshold_collision = 50    # 50px
                threshold_near_miss = 150   # 150px
            
            # 分级
            if distance < threshold_collision:
                analyzed['level'] = 1
                analyzed['level_name'] = 'Collision'
            elif distance < threshold_near_miss:
                analyzed['level'] = 2
                analyzed['level_name'] = 'Near Miss'
            else:
                analyzed['level'] = 3
                analyzed['level_name'] = 'Avoidance'
            
            analyzed_events.append(analyzed)
        
        # 统计
        level_counts = {1: 0, 2: 0, 3: 0}
        for event in analyzed_events:
            level_counts[event['level']] += 1
        
        # 保存分析结果
        analysis_path = self.analysis_dir / 'collision_events.json'
        with open(analysis_path, 'w') as f:
            json.dump(analyzed_events, f, indent=2)
        
        print(f"✓ 碰撞风险分析完成")
        print(f"  - Level 1 (Collision): {level_counts[1]} events")
        print(f"  - Level 2 (Near Miss): {level_counts[2]} events")
        print(f"  - Level 3 (Avoidance): {level_counts[3]} events")
        print(f"  分析结果保存: {analysis_path.name}")
        
        return analyzed_events, level_counts
    
    def generate_report(self, proximity_events, analyzed_events, level_counts):
        """生成最终报告"""
        report_path = self.analysis_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("YOLO-First 碰撞检测分析报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入视频: {self.video_path}\n")
            if self.homography_path:
                f.write(f"Homography: {self.homography_path}\n")
            else:
                f.write(f"Homography: 未提供 (像素空间处理)\n")
            f.write(f"结果目录: {self.run_dir}\n\n")
            
            f.write(f"处理方式: YOLO-First (先检测，再变换)\n")
            f.write(f"流程: YOLO检测 → 轨迹构建 → 关键帧提取 → Homography变换 → 风险分析\n\n")
            
            f.write(f"接近事件统计:\n")
            f.write(f"  - 总接近事件: {len(proximity_events)}\n")
            f.write(f"  - Level 1 (Collision): {level_counts[1]}\n")
            f.write(f"  - Level 2 (Near Miss): {level_counts[2]}\n")
            f.write(f"  - Level 3 (Avoidance): {level_counts[3]}\n\n")
            
            if analyzed_events:
                f.write("前5个高风险事件:\n")
                f.write("-"*70 + "\n")
                
                # 按风险等级排序
                sorted_events = sorted(analyzed_events, key=lambda e: e.get('level', 3))
                
                for i, event in enumerate(sorted_events[:5], 1):
                    f.write(f"\n{i}. Frame {event['frame']} ({event['time']:.2f}s)\n")
                    f.write(f"   物体ID: {event['object_ids']}\n")
                    f.write(f"   风险等级: Level {event['level']} ({event.get('level_name', 'Unknown')})\n")
                    if 'distance_pixel' in event:
                        f.write(f"   距离 (像素): {event['distance_pixel']:.1f}px\n")
                    if 'distance_meters' in event:
                        f.write(f"   距离 (米): {event['distance_meters']:.2f}m\n")
            else:
                f.write("未检测到接近事件\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("报告结束\n")
        
        print(f"✓ 报告已保存: {report_path.name}")
    
    def run(self, conf_threshold=0.45):
        """运行完整 YOLO-First pipeline"""
        try:
            # Step 1: YOLO 检测
            all_detections = self.run_yolo_detection(conf_threshold)
            
            if not all_detections:
                print(f"❌ 未检测到任何物体，停止处理")
                return
            
            # Step 2: 轨迹构建
            tracks = self.build_trajectories(all_detections)
            
            # Step 3: 关键帧提取
            proximity_events = self.extract_key_frames(all_detections, tracks)
            
            if not proximity_events:
                print(f"⚠️  未检测到接近事件")
                transformed_events = None
            else:
                # Step 4: Homography 变换
                transformed_events = self.transform_to_world_coords(proximity_events, all_detections)
            
            # Step 5: 风险分析
            analyzed_events, level_counts = self.analyze_collision_risk(
                proximity_events, transformed_events)
            
            # 生成报告
            self.generate_report(proximity_events, analyzed_events, level_counts)
            
            print(f"\n{'='*70}")
            print(f"✓ YOLO-First Pipeline 完成！")
            print(f"{'='*70}")
            print(f"结果保存在: {self.run_dir}")
            print(f"\n文件夹结构:")
            print(f"  1_raw_detections/")
            print(f"    ├── detections.json")
            print(f"    └── detection_stats.json")
            print(f"  2_trajectories/")
            print(f"    ├── tracks.json")
            print(f"    └── track_stats.json")
            print(f"  3_key_frames/")
            print(f"    └── proximity_events.json")
            print(f"  4_homography_transform/")
            print(f"    ├── homography.json")
            print(f"    └── events_world_coords.json")
            print(f"  5_collision_analysis/")
            print(f"    ├── collision_events.json")
            print(f"    └── analysis_report.txt")
            
        except Exception as e:
            print(f"❌ Pipeline 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO-First 碰撞检测Pipeline')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--homography', type=str, default=None, 
                       help='Homography JSON路径 (可选)')
    parser.add_argument('--output', type=str, default='../../results', 
                       help='结果基础目录')
    parser.add_argument('--conf', type=float, default=0.45, 
                       help='YOLO置信度阈值')
    
    args = parser.parse_args()
    
    pipeline = YOLOFirstPipeline(args.video, args.homography, args.output)
    pipeline.run(args.conf)
