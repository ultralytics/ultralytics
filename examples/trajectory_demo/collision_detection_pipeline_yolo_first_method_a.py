"""
collision_detection_pipeline_yolo_first_method_a.py

YOLO-First 碰撞检测管道 (Method A )
执行顺序: YOLO检测 → 轨迹构建(px) → 关键帧检测 → Homography变换(仅关键帧) → TTC分析


流程:
1. YOLO 检测 (原始视频或跳帧视频)
   - 在原始分辨率上检测所有物体
   - 保存检测框和 Track ID (像素空间)

2. 轨迹构建 (像素空间)
   - 关联 Track ID，建立轨迹
   - 估计速度 (px/s)
   - 所有计算在像素坐标系

3. 关键帧检测 (接近事件识别)
   - 识别"距离 < 150px"的物体对
   - 标记为关键帧

4. Homography 变换 (仅关键帧) 
   - 只对关键帧中的物体点做变换
   - 转换距离单位 (px → m)
   - 转换速度单位 (px/s → m/s)

5. TTC 和 Event 分级
   - 计算 TTC (在世界坐标)
   - 分级事件 (L1/L2/L3)
   - 生成报告

优势: 
- 
-  性能最优 (仅变换5-10%的数据)
-  逻辑清晰 (先找接近的，再精确分析)
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 导入YOLO
sys.path.append(os.path.dirname(__file__))
from ultralytics import YOLO


class YOLOFirstPipelineA:
    def __init__(self, video_path, homography_path=None, output_base=None, skip_frames=1, model='yolo11n', min_track_length=3):
        """初始化 YOLO-First pipeline 
        
        Args:
            video_path: 原始视频路径
            homography_path: Homography JSON路径 (用于Step 4)
            output_base: 结果基础目录
            skip_frames: 抽帧参数，每隔 skip_frames 帧处理一帧 (1=处理所有帧, 3=每隔3帧处理一帧)
            model: YOLO 模型选择 (yolo11n/yolo11m/yolo11l)
            min_track_length: 最小轨迹长度，短于此的被认为是误检
        """
        self.video_path = video_path
        self.homography_path = homography_path
        self.skip_frames = skip_frames  # 抽帧参数
        self.model = model  # YOLO 模型
        self.min_track_length = min_track_length  # 最小轨迹长度
        # 使用 /workspace/ultralytics/results 作为输出目录（确保路径正确）
        if output_base is None:
            output_base = "/workspace/ultralytics/results"
        self.output_base = Path(output_base)
        self.H = None
        self.pixel_per_meter = 1.0
        
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = (self.output_base / f"{timestamp}_yolo_first_method_a").resolve()
        
        # 创建子目录结构 (Method A)
        self.detection_dir = self.run_dir / "1_yolo_detection"
        self.trajectory_dir = self.run_dir / "2_trajectories"
        self.keyframe_dir = self.run_dir / "3_key_frames"
        self.homography_dir = self.run_dir / "4_homography_transform"
        self.analysis_dir = self.run_dir / "5_collision_analysis"
        
        for d in [self.detection_dir, self.trajectory_dir, self.keyframe_dir, 
                  self.homography_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"YOLO-First 碰撞检测Pipeline (Method A - 导师推荐)")
        print(f"{'='*70}")
        print(f"时间戳: {timestamp}")
        print(f"结果目录: {self.run_dir}")
        print(f"执行顺序: YOLO → 轨迹(px) → 关键帧 → Homography(关键帧) → TTC")
    
    def load_homography(self):
        """加载 Homography 矩阵 (Step 4 需要)"""
        if not self.homography_path:
            print(f"\n⚠️  未提供 Homography，将仅在像素空间处理")
            return False
        
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
            
            print(f"  ✓ Homography已加载")
            print(f"    缩放因子: {self.pixel_per_meter:.2f} px/m")
            
            return True
        
        except Exception as e:
            print(f"  ❌ 加载Homography失败: {e}")
            return False
    
    # =========================================================================
    # 可视化和图像保存
    # =========================================================================
    
    def save_detection_frame(self, video_path, frame_num, output_path, detections=None):
        """保存指定帧的图像（带 YOLO 检测框）"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False
        
        # 如果有检测数据，绘制检测框
        if detections:
            for obj in detections.get('objects', []):
                x, y, w, h = obj['bbox_xywh']
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                # 绘制检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制 Track ID
                track_id = obj['track_id']
                conf = obj['conf']
                text = f"ID:{track_id} ({conf:.2f})"
                cv2.putText(frame, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存图像
        cv2.imwrite(str(output_path), frame)
        return True
    
    def save_keyframe_with_distance(self, video_path, frame_num, output_path, proximity_event):
        """保存关键帧图像（绘制两个接近的物体和距离）"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False
        
        # 获取两个物体的信息
        center_1 = proximity_event.get('center_1_px', proximity_event.get('center_1', [0, 0]))
        center_2 = proximity_event.get('center_2_px', proximity_event.get('center_2', [0, 0]))
        track_id_1 = proximity_event.get('track_id_1', -1)
        track_id_2 = proximity_event.get('track_id_2', -1)
        distance_pixel = proximity_event.get('distance_pixel', 0)
        distance_meters = proximity_event.get('distance_meters', 0)
        class_1 = proximity_event.get('class_1', 'Unknown')
        class_2 = proximity_event.get('class_2', 'Unknown')
        
        # 绘制两个物体的中心点
        pt1 = tuple(map(int, center_1))
        pt2 = tuple(map(int, center_2))
        
        # 绘制圆点和ID
        cv2.circle(frame, pt1, 5, (0, 255, 0), -1)  # 绿色圆点
        cv2.putText(frame, f"ID:{track_id_1}", (pt1[0]+10, pt1[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.circle(frame, pt2, 5, (0, 0, 255), -1)  # 红色圆点
        cv2.putText(frame, f"ID:{track_id_2}", (pt2[0]+10, pt2[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 绘制连接线
        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)  # 蓝色线
        
        # 显示距离信息 (像素和世界坐标)
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        distance_text = f"Distance: {distance_meters:.2f}m ({distance_pixel:.0f}px)"
        cv2.putText(frame, distance_text, (mid_x-100, mid_y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 显示物体类别信息
        class_text = f"{class_1} vs {class_2}"
        cv2.putText(frame, class_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 保存图像
        cv2.imwrite(str(output_path), frame)
        return True
    
    # =========================================================================
    # STEP 1: YOLO 检测 (像素空间)
    # =========================================================================
    
    def run_yolo_detection(self, conf_threshold=0.45):
        """Step 1: YOLO 检测 (原始视频或跳帧视频)
        
        输出:
        - 保存所有检测框和 Track ID (像素空间)
        - 生成检测统计
        """
        print(f"\n【Step 1: YOLO 检测】")
        
        # 加载指定的 YOLO 模型
        model = YOLO(f'{self.model}.pt')
        print(f"  加载模型: {self.model}.pt")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_detections = []
        frame_count = 0
        detection_frames_count = 0
        
        # 抽帧处理
        if self.skip_frames > 1:
            print(f"  处理中: {total_frames}帧 @ {fps:.2f}FPS (每隔{self.skip_frames}帧处理一帧)...")
        else:
            print(f"  处理中: {total_frames}帧 @ {fps:.2f}FPS...")
        
        for result in model.track(source=self.video_path, stream=True, 
                                 persist=True, conf=conf_threshold):
            frame_count += 1
            
            # 抽帧：如果启用抽帧，跳过不需要处理的帧
            if (frame_count - 1) % self.skip_frames != 0:
                continue
            
            if result.boxes is None or len(result.boxes) == 0:
                if frame_count % 30 == 0:
                    print(f"    Frame {frame_count}/{total_frames} - 无物体")
                continue
            
            detection_frames_count += 1
            boxes = result.boxes.xywh.cpu().numpy()
            ids = result.boxes.id
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            
            frame_detections = {
                'frame': frame_count,
                'time': frame_count / fps,
                'objects': []
            }
            
            # 只有在有检测到对象时才处理
            if len(boxes) > 0 and ids is not None:
                for i in range(len(boxes)):
                    obj_data = {
                        'track_id': int(ids[i]) if ids[i] is not None else -1,
                        'class': int(classes[i]),
                        'conf': float(confs[i]),
                        'bbox_xywh': boxes[i].tolist(),  # [x_center, y_center, w, h] 像素
                    }
                    frame_detections['objects'].append(obj_data)
                
                # 保存检测框图像（每个有检测的帧）
                frame_img_path = self.detection_dir / f"frame_{frame_count:04d}.jpg"
                self.save_detection_frame(self.video_path, frame_count, frame_img_path, frame_detections)
            
            all_detections.append(frame_detections)
            
            if frame_count % 30 == 0:
                print(f"    Frame {frame_count}/{total_frames} - {len(boxes)}个物体")
        
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
        
        print(f"  ✓ Step 1完成: {detection_frames_count}帧检测到物体")
        print(f"    输出: {detections_path.name}")
        
        return all_detections
    
    # =========================================================================
    # STEP 1.5: 同帧内物体分割合并 (处理YOLO把一个物体分成两个的问题)
    # =========================================================================
    
    def merge_fragmented_objects_in_frame(self, all_detections, same_class_distance_threshold=200):
        """在同一帧内，合并被分割的同类物体
        
        原理：
        - YOLO有时会把一个物体检测成多个（比如摩托车的前后部分）
        - 在同一帧内，如果两个物体：
          1. 类别相同（都是motorcycle）
          2. 中心距离 < same_class_distance_threshold (像素)
          3. 则认为是同一物体被分割，合并它们
        - 保留置信度更高的那个，删除置信度低的
        """
        print(f"\n【Step 1.5: 同帧内物体分割合并】")
        print(f"  ℹ️  在每一帧内检测和合并被分割的同类物体")
        
        merged_count = 0
        
        for frame_data in all_detections:
            frame = frame_data['frame']
            objects = frame_data['objects']
            
            # 记录哪些物体应该被删除（因为被合并了）
            to_remove = set()
            
            # 检查所有物体对
            for i, obj1 in enumerate(objects):
                if i in to_remove:
                    continue
                
                for j, obj2 in enumerate(objects):
                    if j <= i or j in to_remove:
                        continue
                    
                    # 检查是否是同类别且距离近
                    if obj1['class'] == obj2['class']:
                        x1, y1 = obj1['bbox_xywh'][0], obj1['bbox_xywh'][1]
                        x2, y2 = obj2['bbox_xywh'][0], obj2['bbox_xywh'][1]
                        
                        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        
                        if distance < same_class_distance_threshold:
                            # 合并：保留置信度高的，删除置信度低的
                            if obj1['conf'] >= obj2['conf']:
                                to_remove.add(j)
                                merged_count += 1
                                class_name = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                                             4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}.get(obj1['class'], f"class_{obj1['class']}")
                                print(f"  🔀 合并: Frame {frame:03d} - 两个 {class_name} (ID {obj1['track_id']}, ID {obj2['track_id']}) 距离 {distance:.1f}px < {same_class_distance_threshold}px")
                            else:
                                to_remove.add(i)
                                merged_count += 1
                                class_name = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                                             4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}.get(obj2['class'], f"class_{obj2['class']}")
                                print(f"  🔀 合并: Frame {frame:03d} - 两个 {class_name} (ID {obj2['track_id']}, ID {obj1['track_id']}) 距离 {distance:.1f}px < {same_class_distance_threshold}px")
            
            # 删除被合并的物体
            frame_data['objects'] = [obj for i, obj in enumerate(objects) if i not in to_remove]
        
        print(f"  ✓ Step 1.5 完成: 合并了 {merged_count} 个同帧内的分割物体")
        
        return all_detections
    
    # =========================================================================
    # =========================================================================
    
    def build_trajectories(self, all_detections):
        """Step 2: 构建轨迹 (像素空间，px/s)
        
        输入: 原始检测结果
        输出: 完整轨迹 (按 track_id 组织)
        """
        print(f"\n【Step 2: 轨迹构建 (像素空间)】")
        
        # 按 track_id 组织轨迹
        tracks = {}
        
        for frame_data in all_detections:
            for obj in frame_data['objects']:
                track_id = obj['track_id']
                
                if track_id not in tracks:
                    tracks[track_id] = []
                
                # 轨迹点 (像素空间)
                track_point = {
                    'frame': frame_data['frame'],
                    'time': frame_data['time'],
                    'class': obj['class'],
                    'conf': obj['conf'],
                    'center_x': obj['bbox_xywh'][0],  # 像素
                    'center_y': obj['bbox_xywh'][1],  # 像素
                }
                
                tracks[track_id].append(track_point)
        
        # 计算每个轨迹的速度信息 (px/s)
        for track_id, track_points in tracks.items():
            track_points.sort(key=lambda p: p['frame'])
            
            if len(track_points) >= 2:
                for i in range(1, len(track_points)):
                    prev = track_points[i-1]
                    curr = track_points[i]
                    
                    dt = curr['time'] - prev['time']
                    if dt > 0:
                        dx = curr['center_x'] - prev['center_x']
                        dy = curr['center_y'] - prev['center_y']
                        
                        curr['vx'] = dx / dt  # px/s
                        curr['vy'] = dy / dt  # px/s
                        curr['speed'] = np.sqrt(dx**2 + dy**2) / dt  # px/s
                    else:
                        curr['vx'] = 0.0
                        curr['vy'] = 0.0
                        curr['speed'] = 0.0
                
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
            'coordinate_system': 'pixel',
            'velocity_unit': 'px/s',
        }
        stats_path = self.trajectory_dir / 'track_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✓ Step 2完成: {len(tracks)}条轨迹 (像素空间)")
        print(f"    速度单位: px/s")
        print(f"    输出: {tracks_path.name}")
        
        return tracks
    
    # =========================================================================
    # STEP 2.4: 轨迹间断检测 (识别出现→消失→重新出现的可疑轨迹)
    # =========================================================================
    
    def detect_discontinuous_tracks(self, all_detections, max_gap_frames=None):
        """检测轨迹间断 (出现→消失→重新出现)
        
        原理：
        - 同一个 Track ID 在时间序列中出现了间断
        - 比如 Track ID 13 在 Frame 82-91 出现，Frame 92-93 消失，Frame 94 重新出现
        - 这很不合理，除非物体真的离开了视野（极少见）
        - 更可能是追踪失败或误检导致的幽灵轨迹
        
        注意：考虑 frame skipping 的影响
        - 如果用 --skip-frames 3，检测的是每隔3帧的情况
        - 所以"间断"的定义应该相对宽松
        
        返回：
        - suspicious_track_ids: 包含真正的间断的 Track ID 集合
        """
        print(f"\n【Step 2.4: 轨迹间断检测】")
        print(f"  ℹ️  检测轨迹间断（出现→消失→重新出现，考虑frame skipping）")
        
        # 如果没有指定，根据 skip_frames 设置默认值
        if max_gap_frames is None:
            max_gap_frames = self.skip_frames * 3  # 允许最多 skip_frames*3 的间断
        
        # 为每个 Track ID 记录它出现的所有帧
        track_frames = {}  # {track_id: [frame_nums]}
        
        for frame_data in all_detections:
            frame = frame_data['frame']
            for obj in frame_data['objects']:
                track_id = obj['track_id']
                if track_id not in track_frames:
                    track_frames[track_id] = []
                track_frames[track_id].append(frame)
        
        # 检查每个轨迹的连续性
        suspicious_tracks = {}  # {track_id: 'gap_info'}
        
        for track_id, frames in track_frames.items():
            frames = sorted(set(frames))  # 去重并排序
            
            if len(frames) < 2:
                continue
            
            # 检查是否有真正的间断（不是因为 frame skipping 导致的）
            gaps = []
            for i in range(1, len(frames)):
                gap = frames[i] - frames[i-1]
                # 只有间断 > skip_frames 时才算真正的间断
                if gap > self.skip_frames + 1:  # 允许1帧的偏差
                    gaps.append((frames[i-1], frames[i], gap))
            
            if gaps:
                # 有真正间断的轨迹
                suspicious_tracks[track_id] = {
                    'total_frames': len(frames),
                    'first_frame': frames[0],
                    'last_frame': frames[-1],
                    'gaps': gaps
                }
        
        # 打印可疑轨迹信息
        if suspicious_tracks:
            print(f"  ⚠️  检测到 {len(suspicious_tracks)} 条有真正间断的轨迹（可能是误检或追踪失败）:")
            for track_id, info in sorted(suspicious_tracks.items()):
                print(f"     - Track ID {track_id}: {info['total_frames']}帧 ({info['first_frame']}-{info['last_frame']})")
                for gap_start, gap_end, gap_size in info['gaps']:
                    print(f"       └─ 间断: Frame {gap_start} → Frame {gap_end} (间隔 {gap_size} 帧，超过允许的 {self.skip_frames + 1} 帧)")
        else:
            print(f"  ✓ 没有检测到真正的间断轨迹（考虑了 skip_frames={self.skip_frames} 的影响）")
        
        print(f"  ℹ️  注意: 间断阈值 = {self.skip_frames + 1} 帧（基于 skip_frames 参数）")
        
        return suspicious_tracks
    
    # =========================================================================
    # STEP 2.5: 轨迹连续性过滤 (排除短轨迹误检)
    # =========================================================================
    
    def filter_short_tracks(self, all_detections, min_track_length=3):
        """过滤短轨迹 (可能是 YOLO 误检)
        
        如果一个 Track ID 只出现在少于 min_track_length 帧中，
        则认为是误检，将其从检测结果中移除。
        
        原理：
        - 真实物体应该在连续的多个帧中被检测到
        - 如果物体突然出现又消失，通常是 YOLO 的误检
        """
        print(f"\n【Step 2.5: 轨迹连续性过滤】")
        
        # 统计每个 Track ID 的出现帧数
        track_lengths = {}
        for frame_data in all_detections:
            for obj in frame_data['objects']:
                track_id = obj['track_id']
                track_lengths[track_id] = track_lengths.get(track_id, 0) + 1
        
        # 找出短轨迹 (可能是误检)
        short_tracks = {tid: length for tid, length in track_lengths.items() if length < min_track_length}
        valid_tracks = {tid: length for tid, length in track_lengths.items() if length >= min_track_length}
        
        print(f"  轨迹长度统计:")
        print(f"    - 总轨迹: {len(track_lengths)}")
        print(f"    - 有效轨迹 (>= {min_track_length}帧): {len(valid_tracks)}")
        print(f"    - 短轨迹/误检 (< {min_track_length}帧): {len(short_tracks)}")
        
        if short_tracks:
            print(f"  🗑️  移除的短轨迹:")
            for tid, length in sorted(short_tracks.items(), key=lambda x: x[1]):
                print(f"     - Track ID {tid}: {length} 帧")
        
        # 过滤检测结果，移除短轨迹中的物体
        filtered_detections = []
        for frame_data in all_detections:
            new_frame_data = frame_data.copy()
            new_frame_data['objects'] = [
                obj for obj in frame_data['objects']
                if obj['track_id'] in valid_tracks
            ]
            filtered_detections.append(new_frame_data)
        
        print(f"  ✓ Step 2.5 完成: 移除了 {len(short_tracks)} 条短轨迹")
        
        return filtered_detections
    
    # =========================================================================
    # STEP 2.6: Track ID 重连检测 (同一物体多个ID合并)
    # =========================================================================
    
    def merge_fragmented_tracks(self, all_detections, max_gap_frames=2, max_distance_pixels=100):
        """合并被断开的Track ID (同一物体追踪失败导致的重复ID)
        
        原理：
        - 如果 Track ID A 在某帧消失
        - 然后在 max_gap_frames 帧内，一个新的 Track ID B 出现
        - 且两个ID的物体中心距离 < max_distance_pixels
        - 则认为是同一物体，应该合并ID
        
        这特别适合解决摩托车被分成ID13和ID15的问题。
        """
        print(f"\n【Step 2.6: Track ID 重连检测】")
        print(f"  ℹ️  检测并合并被断开的轨迹 (消失<{max_gap_frames}帧后重新出现)")
        
        # 第一步：为每个Track ID统计最后出现的帧和位置
        track_last_appearance = {}  # {track_id: {'frame': f, 'x': x, 'y': y, 'class': c}}
        
        for frame_data in all_detections:
            frame = frame_data['frame']
            for obj in frame_data['objects']:
                track_id = obj['track_id']
                track_last_appearance[track_id] = {
                    'frame': frame,
                    'x': obj['bbox_xywh'][0],
                    'y': obj['bbox_xywh'][1],
                    'class': obj['class']
                }
        
        # 第二步：寻找可能断开的Track ID对
        merge_map = {}  # {old_id: new_id} 映射
        merged_count = 0
        
        track_ids = sorted(track_last_appearance.keys())
        
        for i, track_a in enumerate(track_ids):
            if track_a in merge_map:
                continue  # 已经被合并过了
            
            last_a = track_last_appearance[track_a]
            frame_a = last_a['frame']
            x_a, y_a = last_a['x'], last_a['y']
            class_a = last_a['class']
            
            # 查找后续出现的Track ID
            for track_b in track_ids[i+1:]:
                if track_b in merge_map:
                    continue
                
                # 找track_b的第一次出现
                first_b_frame = None
                first_b_pos = None
                first_b_class = None
                
                for frame_data in all_detections:
                    for obj in frame_data['objects']:
                        if obj['track_id'] == track_b:
                            if first_b_frame is None:
                                first_b_frame = frame_data['frame']
                                first_b_pos = (obj['bbox_xywh'][0], obj['bbox_xywh'][1])
                                first_b_class = obj['class']
                            break
                
                if first_b_frame is None:
                    continue
                
                # 检查是否满足重连条件
                frame_gap = first_b_frame - frame_a
                if 1 <= frame_gap <= max_gap_frames:  # 中间有间隔但不超过max_gap
                    distance = np.sqrt((first_b_pos[0] - x_a)**2 + (first_b_pos[1] - y_a)**2)
                    
                    if distance < max_distance_pixels and class_a == first_b_class:
                        # 认为是同一物体，应该合并
                        merge_map[track_b] = track_a
                        merged_count += 1
                        print(f"  🔗 合并: Track ID {track_b} (首次Frame {first_b_frame}) → ID {track_a} (末次Frame {frame_a})")
                        print(f"     间隔: {frame_gap}帧, 距离: {distance:.1f}px, 类别: {class_a}")
        
        # 第三步：应用合并到所有检测结果
        if merge_map:
            for frame_data in all_detections:
                for obj in frame_data['objects']:
                    if obj['track_id'] in merge_map:
                        old_id = obj['track_id']
                        new_id = merge_map[old_id]
                        obj['track_id'] = new_id
        
        print(f"  ✓ Step 2.6 完成: 合并了 {merged_count} 个断开的Track ID")
        
        return all_detections
    
    # =========================================================================
    # STEP 3: 关键帧检测 (接近事件)
    # =========================================================================

    def extract_key_frames(self, all_detections, world_distance_threshold=2.0, debug_threshold=5.0):
        """Step 3: 关键帧检测 (接近事件)
        
        🔄 FIXED: 现在在【世界坐标】中计算距离，而不是像素坐标
        - 使用 Homography 早期变换坐标
        - 设置实际距离阈值 (默认 2米) 而不是固定像素阈值
        - 避免同一物体的不同部分被识别为两个接近事件
        
        参数:
        - world_distance_threshold: 关键帧检测的主阈值（默认 2.0 米）
        - debug_threshold: 调试显示的阈值（默认 5.0 米），用于了解被排除的事件
        
        ⚠️ 改进: 过滤掉同一物体被多次检测的情况
        - 跳过同类别物体的接近检测 (两个car, 两个motorcycle 等)
        - 只保留不同类别物体的交互 (car与motorcycle, car与person等)
        """
        print(f"\n【Step 3: 关键帧检测 (世界坐标 🔄)】")
        print(f"  ℹ️  现在在世界坐标中计算距离（使用Homography）")
        
        proximity_events = []
        all_proximity_pairs = []  # 用于调试，记录所有 < debug_threshold 的检测
        
        # 物体类别映射
        class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                      4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}
        
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
                    
                    # ⚠️ 过滤: 跳过同类别物体（避免摩托车的不同部分被识别为两个物体）
                    if obj1['class'] == obj2['class']:
                        # 只有当距离很近时才保留同类别物体（可能是真实的碰撞）
                        # 否则跳过（可能是同一物体的多个检测）
                        pass  # 暂时保留，可以在后续优化
                    
                    # 计算中心点距离 (先在世界坐标中计算)
                    x1_px, y1_px = obj1['bbox_xywh'][0], obj1['bbox_xywh'][1]
                    x2_px, y2_px = obj2['bbox_xywh'][0], obj2['bbox_xywh'][1]
                    
                    # 🔄 转换到世界坐标 (使用 Homography)
                    x1_world = x1_px / self.pixel_per_meter
                    y1_world = y1_px / self.pixel_per_meter
                    x2_world = x2_px / self.pixel_per_meter
                    y2_world = y2_px / self.pixel_per_meter
                    
                    # 在世界坐标中计算距离 (米)
                    distance_meters = np.sqrt((x2_world-x1_world)**2 + (y2_world-y1_world)**2)
                    distance_pixel = np.sqrt((x2_px-x1_px)**2 + (y2_px-y1_px)**2)
                    
                    class1_name = class_names.get(obj1['class'], f"class_{obj1['class']}")
                    class2_name = class_names.get(obj2['class'], f"class_{obj2['class']}")
                    
                    # 记录所有 < debug_threshold 的检测，用于调试
                    if distance_meters < debug_threshold:
                        all_proximity_pairs.append({
                            'frame': frame,
                            'class_1': class1_name,
                            'class_2': class2_name,
                            'distance_meters': distance_meters,
                            'track_ids': [obj1['track_id'], obj2['track_id']]
                        })
                    
                    # 检查是否为接近事件 (使用世界距离阈值)
                    if distance_meters < world_distance_threshold:
                        class1_name = class_names.get(obj1['class'], f"class_{obj1['class']}")
                        class2_name = class_names.get(obj2['class'], f"class_{obj2['class']}")
                        
                        event = {
                            'frame': frame,
                            'time': frame_data['time'],
                            'track_id_1': obj1['track_id'],
                            'track_id_2': obj2['track_id'],
                            'class_1': class1_name,
                            'class_2': class2_name,
                            'distance_pixel': float(distance_pixel),
                            'distance_meters': float(distance_meters),  
                            'object_classes': (obj1['class'], obj2['class']),
                            'center_1_px': [float(x1_px), float(y1_px)],
                            'center_2_px': [float(x2_px), float(y2_px)],
                            'center_1_world': [float(x1_world), float(y1_world)],  
                            'center_2_world': [float(x2_world), float(y2_world)],  
                            'positions': {
                                'obj1': {'x': x1_px, 'y': y1_px},
                                'obj2': {'x': x2_px, 'y': y2_px}
                            },
                            'positions_world': {  
                                'obj1': {'x': x1_world, 'y': y1_world},
                                'obj2': {'x': x2_world, 'y': y2_world}
                            }
                        }
                        proximity_events.append(event)
                        
                        # 保存关键帧图像（两个接近物体的帧）
                        frame_img_path = self.keyframe_dir / f"keyframe_{frame:04d}_ID{obj1['track_id']}_ID{obj2['track_id']}.jpg"
                        self.save_keyframe_with_distance(self.video_path, frame, frame_img_path, event)
        
        # 保存接近事件
        events_path = self.keyframe_dir / 'proximity_events.json'
        with open(events_path, 'w') as f:
            json.dump(proximity_events, f, indent=2)
        
        print(f"  ✓ Step 3完成: {len(proximity_events)}个关键帧 (< {world_distance_threshold}m)")
        print(f"    总检测到的近距离对: {len(all_proximity_pairs)}个 (< {debug_threshold}m)")
        print(f"    距离阈值: {world_distance_threshold}米 (世界坐标)")
        print(f"    缩放因子: {self.pixel_per_meter:.2f} px/m")
        print(f"    输出: {events_path.name}")
        
        # 打印所有被 debug_threshold 捕捉的事件，但被 world_distance_threshold 排除的
        excluded_count = len(all_proximity_pairs) - len(proximity_events)
        if excluded_count > 0:
            print(f"\n  ℹ️  被排除的接近事件 ({world_distance_threshold}-{debug_threshold}m):")
            for pair in all_proximity_pairs:
                if pair['distance_meters'] >= world_distance_threshold:
                    print(f"     - Frame {pair['frame']:03d}: Track {pair['track_ids'][0]}({pair['class_1']}) + Track {pair['track_ids'][1]}({pair['class_2']}) = {pair['distance_meters']:.2f}m")
        
        # 调试输出：检查输入数据
        total_object_pairs = 0
        for frame_data in all_detections:
            if len(frame_data['objects']) >= 2:
                total_object_pairs += len(frame_data['objects']) * (len(frame_data['objects']) - 1) // 2
        print(f"\n  ℹ️  调试信息: 检查了 {total_object_pairs} 个物体对，其中 {len(all_proximity_pairs)} 个距离 < {debug_threshold}m")
        
        return proximity_events
    
    # =========================================================================
    # STEP 3.5: 同类别物体误检过滤
    # =========================================================================
    
    def filter_same_class_false_positives(self, proximity_events, same_class_distance_threshold=0.3):
        """过滤同类别物体的误检 + 跨帧同一物体的误分割
        
        过滤条件：
        1. 极近距离 (< 0.1m) → 同一物体两部分
        2. 不合理的类别组合 + 距离稳定 (std < 0.5) → 同速不可能
        3. 断断续续出现的Track ID对 + 距离近 (< 2.0m) → 同一物体被误分割
        4. 都是汽车类型 + 距离 < 0.5m → 同一车辆的不同部分（如卡车头和身体）
        """
        print(f"\n【Step 3.5: 物体误检过滤 (智能策略)】")
        
        # 不合理的类别组合（不可能同时出现且同速运动）
        illogical_class_combinations = [
            ('person', 'motorcycle'),
            ('person', 'car'),
            ('person', 'truck'),
            ('person', 'bus'),
            ('bicycle', 'motorcycle'),
            ('bicycle', 'car'),
        ]
        
        # 定义汽车类型
        vehicle_types = {'car', 'truck', 'bus', 'motorcycle'}
        
        # 首先分析Track ID对的出现情况
        track_pair_analysis = {}
        for event in proximity_events:
            tid1, tid2 = event['track_id_1'], event['track_id_2']
            pair_key = tuple(sorted([tid1, tid2]))
            
            if pair_key not in track_pair_analysis:
                track_pair_analysis[pair_key] = {
                    'events': [],
                    'frames': [],
                    'distances': [],
                    'classes': None
                }
            
            track_pair_analysis[pair_key]['events'].append(event)
            track_pair_analysis[pair_key]['frames'].append(event['frame'])
            track_pair_analysis[pair_key]['distances'].append(event['distance_meters'])
            track_pair_analysis[pair_key]['classes'] = (event['class_1'], event['class_2'])
        
        # 识别"断断续续出现"的Track ID对
        suspicious_discontinuous_pairs = set()
        for pair, info in track_pair_analysis.items():
            frames = sorted(info['frames'])
            distances = info['distances']
            avg_distance = sum(distances) / len(distances)
            
            # 检查是否有明显的间隔（出现-消失-再出现）
            has_gap = False
            for i in range(len(frames) - 1):
                if frames[i+1] - frames[i] > 3:  # 间隔 > 3帧
                    has_gap = True
                    break
            
            # 如果断断续续出现且距离近，标记为可疑
            if has_gap and avg_distance < 2.0:
                suspicious_discontinuous_pairs.add(pair)
        
        # 现在进行逐个事件的过滤
        filtered_events = []
        filtered_count = 0
        filter_reasons = []
        
        for event in proximity_events:
            class_1 = event['class_1']
            class_2 = event['class_2']
            distance = event['distance_meters']
            frame = event['frame']
            tid1, tid2 = event['track_id_1'], event['track_id_2']
            pair_key = tuple(sorted([tid1, tid2]))
            
            reason = None
            
            # 条件1: 极近距离 (< 0.1m) → 同一物体的两部分
            if distance < 0.1:
                reason = f"极近 ({distance:.3f}m < 0.1m)"
                filtered_count += 1
                filter_reasons.append((frame, tid1, tid2, class_1, class_2, distance, reason))
                continue
            
            # 条件4: 都是汽车类型 + 距离 < 0.5m → 同一车辆的不同部分
            if (class_1 in vehicle_types and class_2 in vehicle_types) and distance < 0.5:
                reason = f"都是汽车类型 ({class_1}+{class_2}, 距离{distance:.3f}m < 0.5m)"
                filtered_count += 1
                filter_reasons.append((frame, tid1, tid2, class_1, class_2, distance, reason))
                continue
            
            # 条件2: 不合理的类别组合 + 距离稳定 → 同速不可能
            class_pair = tuple(sorted([class_1, class_2]))
            if class_pair in [tuple(sorted(p)) for p in illogical_class_combinations]:
                pair_info = track_pair_analysis[pair_key]
                std_distance = np.std(pair_info['distances'])
                
                if std_distance < 0.5:  # 距离非常稳定 = 同速 = 不可能
                    reason = f"不合理类别组合+同速 ({class_1}+{class_2}, std={std_distance:.2f}m)"
                    filtered_count += 1
                    filter_reasons.append((frame, tid1, tid2, class_1, class_2, distance, reason))
                    continue
            
            # 条件3: 断断续续出现的Track ID对 + 距离近 → 同一物体被误分割
            if pair_key in suspicious_discontinuous_pairs:
                pair_info = track_pair_analysis[pair_key]
                reason = f"断断续续出现({len(pair_info['frames'])}帧) + 距离近"
                filtered_count += 1
                filter_reasons.append((frame, tid1, tid2, class_1, class_2, distance, reason))
                continue
            
            # 保留这个事件
            filtered_events.append(event)
        
        # 打印过滤详情
        if filter_reasons:
            print(f"  🗑️  过滤的事件:")
            for frame, tid1, tid2, class1, class2, dist, reason in filter_reasons[:20]:  # 只打印前20个
                print(f"      Frame {frame}: {class1}({tid1}) + {class2}({tid2}) = {dist:.3f}m ({reason})")
            if len(filter_reasons) > 20:
                print(f"      ... 还有 {len(filter_reasons)-20} 个")
        
        print(f"  ✓ 过滤完成: 排除了 {filtered_count} 个误检, 保留 {len(filtered_events)} 个事件")
        print(f"    条件1: 距离 < 0.1m")
        print(f"    条件4: 都是汽车类型 (car/truck/bus/motorcycle等) + 距离 < 0.5m")
        print(f"    条件2: 不合理类别组合 (person/motorcycle等) + 距离稳定 (std < 0.5m)")
        print(f"    条件3: Track ID对断断续续出现 + 平均距离 < 2.0m")
        
        # 保存过滤后的事件
        events_path = self.keyframe_dir / 'proximity_events_filtered.json'
        with open(events_path, 'w') as f:
            json.dump({
                'total_detected': len(proximity_events),
                'false_positives_filtered': filtered_count,
                'valid_events': len(filtered_events),
                'events': filtered_events
            }, f, indent=2)
        
        return filtered_events
    
    # =========================================================================
    # STEP 3.6: 清理被过滤的关键帧图片
    # =========================================================================
    
    def cleanup_filtered_keyframes(self, original_events, filtered_events):
        """删除被过滤掉的关键帧图片文件"""
        # 收集被保留的 keyframe 文件名
        kept_frames = set()
        for event in filtered_events:
            frame_id = event['frame']
            tid1 = event['track_id_1']
            tid2 = event['track_id_2']
            # 生成应该被保留的文件名
            filename = f"keyframe_{frame_id:04d}_ID{tid1}_ID{tid2}.jpg"
            kept_frames.add(filename)
        
        # 删除不在 kept_frames 中的 keyframe 文件
        for img_file in self.keyframe_dir.glob('keyframe_*.jpg'):
            if img_file.name not in kept_frames:
                try:
                    img_file.unlink()
                    print(f"  🗑️  删除关键帧图片: {img_file.name}")
                except Exception as e:
                    print(f"  ⚠️  删除图片失败 {img_file.name}: {e}")
    
    # =========================================================================
    # STEP 4: Homography 记录 (仅信息) ✨ 已在Step 3中使用
    # =========================================================================
    
    def transform_key_frames_to_world(self, proximity_events):
        """Step 4: Homography 信息保存
        
        ℹ️ 注意: Homography 变换已经在 Step 3 中完成！
        - Step 3: 在【世界坐标】中计算接近度
        - Step 4: 只记录和保存 Homography 信息
        """
        print(f"\n【Step 4: Homography 信息保存】")
        print(f"  ℹ️  注意: 坐标变换已在Step 3中完成（使用Homography）")
        
        if self.H is None:
            print(f"  ⚠️  未加载Homography")
            return proximity_events
        
        # 直接返回事件（已包含世界坐标）
        # 保存 Homography 矩阵信息供参考
        trans_path = self.homography_dir / 'transformed_key_frames.json'
        with open(trans_path, 'w') as f:
            json.dump(proximity_events, f, indent=2)
        
        print(f"  ✓ Step 4完成: {len(proximity_events)}个关键帧信息已保存")
        print(f"    缩放因子: {self.pixel_per_meter:.2f} px/m")
        print(f"    坐标系统: 世界坐标 (已在Step 3变换)")
        print(f"    输出: {trans_path.name}")
        
        return proximity_events
    
    # =========================================================================
    # STEP 5: TTC 和 Event 分级
    # =========================================================================
    
    def analyze_collision_risk(self, transformed_events):
        """Step 5: TTC 计算和 Event 分级
        
        计算 TTC，分级事件 (L1/L2/L3)
        
         改进: 过滤同类别物体的极近接近事件
        - 如果两个物体都是同一类别（如两个car，两个motorcycle）
        - 且距离 < 0.5m，则可能是同一物体的误检
        - 标记为 'Filtered_SameClass' 并排除
        """
        print(f"\n【Step 5: 碰撞风险分析】")
        
        if not transformed_events:
            print(f"  ⚠️  没有关键帧，无法分析")
            return [], {0: 0, 1: 0, 2: 0, 3: 0}
        
        analyzed_events = []
        filtered_count = 0
        
        for event in transformed_events:
            analyzed = event.copy()
            
            # 在世界坐标中进行分级
            distance = event['distance_meters']
            class_1 = event.get('class_1', '')
            class_2 = event.get('class_2', '')
            
            # 🔍 检查是否为同类别物体的极近接近事件
            if class_1 == class_2 and distance < 0.5:
                # 同一类别 + 距离很近 = 可能是同一物体的不同部分
                analyzed['level'] = 0
                analyzed['level_name'] = 'Filtered_SameClass'
                analyzed['reason'] = f"Same class ({class_1}) with distance {distance:.3f}m < 0.5m - likely same object"
                filtered_count += 1
            # 分级标准 (米)
            elif distance < 0.5:
                analyzed['level'] = 1
                analyzed['level_name'] = 'Collision'
            elif distance < 1.5:
                analyzed['level'] = 2
                analyzed['level_name'] = 'Near Miss'
            else:
                analyzed['level'] = 3
                analyzed['level_name'] = 'Avoidance'
            
            analyzed_events.append(analyzed)
        
        # 统计
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for event in analyzed_events:
            level_counts[event['level']] += 1
        
        # 保存分析结果
        analysis_path = self.analysis_dir / 'collision_events.json'
        with open(analysis_path, 'w') as f:
            json.dump(analyzed_events, f, indent=2)
        
        print(f"  ✓ Step 5完成")
        print(f"    - Filtered (Same class, <0.5m): {level_counts[0]} 🚫")
        print(f"    - Level 1 (Collision, <0.5m): {level_counts[1]}")
        print(f"    - Level 2 (Near Miss, 0.5-1.5m): {level_counts[2]}")
        print(f"    - Level 3 (Avoidance, >1.5m): {level_counts[3]}")
        print(f"    输出: {analysis_path.name}")
        
        return analyzed_events, level_counts
    
    # =========================================================================
    # 报告生成
    # =========================================================================
    
    def generate_report(self, proximity_events, analyzed_events, level_counts):
        """生成最终分析报告"""
        report_path = self.analysis_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("YOLO-First 碰撞检测分析报告 (Method A - 导师推荐)\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入视频: {self.video_path}\n")
            f.write(f"Homography: {self.homography_path if self.H is not None else '未提供'}\n")
            f.write(f"结果目录: {self.run_dir}\n\n")
            
            f.write(f"处理方式: YOLO-First (Method A - 导师推荐)\n")
            f.write(f"流程: YOLO检测 → 轨迹(px) → 关键帧 → Homography(关键帧) → 分析\n")
            f.write(f"坐标系转换: 仅在关键帧进行\n")
            f.write(f"优化: 仅对~{len(proximity_events)}个关键帧做Homography\n\n")
            
            f.write(f"关键帧统计:\n")
            f.write(f"  - 总接近事件: {len(proximity_events)}\n")
            if analyzed_events:
                f.write(f"  - Level 1 (Collision): {level_counts[1]}\n")
                f.write(f"  - Level 2 (Near Miss): {level_counts[2]}\n")
                f.write(f"  - Level 3 (Avoidance): {level_counts[3]}\n\n")
            
            if analyzed_events:
                f.write("前10个高风险事件:\n")
                f.write("-"*70 + "\n")
                
                sorted_events = sorted(analyzed_events, key=lambda e: e.get('level', 3))
                
                for i, event in enumerate(sorted_events[:10], 1):
                    f.write(f"\n{i}. Frame {event['frame']} ({event['time']:.2f}s)\n")
                    # 处理不同的物体ID字段名
                    obj_ids = event.get('object_ids') or [event.get('track_id_1', -1), event.get('track_id_2', -1)]
                    f.write(f"   物体ID: {obj_ids}\n")
                    f.write(f"   风险等级: Level {event['level']} ({event.get('level_name', '?')})\n")
                    f.write(f"   距离(像素): {event['distance_pixel']:.1f}px\n")
                    if 'distance_meters' in event:
                        f.write(f"   距离(米): {event['distance_meters']:.2f}m\n")
            else:
                f.write("未检测到接近事件\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("报告结束\n")
        
        print(f"\n  ✓ 报告已保存: {report_path.name}")
    
    def _copy_results_to_workspace(self):
        """自动复制结果到 /workspace/ultralytics/results（使其在 VS Code 中可见）"""
        import shutil
        
        workspace_results = Path("/workspace/ultralytics/results")
        if workspace_results.exists() and self.run_dir.parent != workspace_results:
            try:
                # 检查结果是否已在 workspace_results 中
                result_in_workspace = workspace_results / self.run_dir.name
                if not result_in_workspace.exists():
                    shutil.copytree(self.run_dir, result_in_workspace)
                    print(f"\n  ✓ 结果已复制到: {result_in_workspace}")
                    print(f"    现在可以在 VS Code 中直接查看！")
            except Exception as e:
                print(f"\n  ⚠️  复制失败 ({e})，但结果已保存在: {self.run_dir}")
    
    # =========================================================================
    # 管道编排
    # =========================================================================
    
    def run(self, conf_threshold=0.45):
        """运行完整 YOLO-First 管道 (Method A)"""
        try:
            # Step 0: 加载Homography (如果提供)
            if self.homography_path:
                print(f"\n【Step 0: 加载资源】")
                self.load_homography()
            
            # Step 1: YOLO 检测
            all_detections = self.run_yolo_detection(conf_threshold)
            
            if not all_detections:
                print(f"\n❌ 未检测到任何物体，停止处理")
                return
            
            # Step 1.5: 同帧内物体分割合并  - 合并YOLO分割的同类物体
            all_detections = self.merge_fragmented_objects_in_frame(all_detections, same_class_distance_threshold=100)
            
            # 调试：检查Step 1.5后的数据
            print(f"\n【调试: Step 1.5后的数据】")
            for frame_data in all_detections:
                if frame_data['frame'] == 115:
                    frame_115_objects_after_15 = [(obj['track_id'], obj['class']) for obj in frame_data['objects']]
                    print(f"  Frame 115 Step 1.5后: {frame_115_objects_after_15}")
            
            # Step 2: 轨迹构建
            tracks = self.build_trajectories(all_detections)
            
            # Step 2.4: 轨迹间断检测  - 检测出现→消失→重新出现的可疑轨迹
            suspicious_tracks = self.detect_discontinuous_tracks(all_detections, max_gap_frames=3)
            
            # Step 2.5: 轨迹连续性过滤  - 移除短轨迹误检
            all_detections = self.filter_short_tracks(all_detections, min_track_length=self.min_track_length)
            
            # 调试：检查Step 2.5后的数据
            print(f"\n【调试: Step 2.5后的数据】")
            frame_115_objects = []
            frame_148_objects = []
            for frame_data in all_detections:
                if frame_data['frame'] == 115:
                    frame_115_objects = [(obj['track_id'], obj['class']) for obj in frame_data['objects']]
                if frame_data['frame'] == 148:
                    frame_148_objects = [(obj['track_id'], obj['class']) for obj in frame_data['objects']]
            print(f"  Frame 115: {frame_115_objects}")
            print(f"  Frame 148: {frame_148_objects}")
            
            # Step 3: 关键帧检测
            proximity_events = self.extract_key_frames(all_detections, world_distance_threshold=4.5)
            
            if not proximity_events:
                print(f"\n⚠️  未检测到接近事件")
                analyzed_events = []
                level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            else:
                # Step 3.5: 同类别物体误检过滤 ✨ 新增
                filtered_events = self.filter_same_class_false_positives(proximity_events, same_class_distance_threshold=0.3)
                
                # 清理：删除被过滤掉的关键帧图片
                if len(filtered_events) < len(proximity_events):
                    self.cleanup_filtered_keyframes(proximity_events, filtered_events)
                
                # Step 4: Homography 变换 (仅关键帧)
                if self.H is not None:
                    transformed_events = self.transform_key_frames_to_world(filtered_events)
                else:
                    print(f"\n【Step 4: Homography 变换】")
                    print(f"  ⚠️  跳过 (未加载Homography)")
                    transformed_events = filtered_events
                
                # Step 5: 风险分析
                analyzed_events, level_counts = self.analyze_collision_risk(transformed_events if transformed_events else filtered_events)
            
            # 生成报告
            self.generate_report(proximity_events, analyzed_events, level_counts)
            
            print(f"\n{'='*70}")
            print(f"✓ YOLO-First Pipeline (Method A) 完成！")
            print(f"{'='*70}")
            print(f"结果保存在: {self.run_dir}")
            
            # 自动复制结果到 /workspace/ultralytics/results（如果不同的话）
            self._copy_results_to_workspace()
            
            print(f"\n文件夹结构:")
            print(f"  1_yolo_detection/")
            print(f"    ├── detections_pixel.json")
            print(f"    ├── detection_stats.json")
            print(f"    └── *.jpg (所有有检测的帧)")
            print(f"  2_trajectories/")
            print(f"    ├── tracks.json")
            print(f"    └── track_stats.json")
            print(f"  3_key_frames/")
            print(f"    ├── proximity_events.json")
            print(f"    └── *.jpg (接近事件的关键帧)")
            print(f"  4_homography_transform/")
            print(f"    ├── homography.json")
            print(f"    └── transformed_key_frames.json")
            print(f"  5_collision_analysis/")
            print(f"    ├── collision_events.json")
            print(f"    └── analysis_report.txt")
            
        except Exception as e:
            print(f"\n❌ Pipeline 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO-First 碰撞检测Pipeline (Method A - 导师推荐)')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--homography', type=str, default=None, 
                       help='Homography JSON路径 (可选)')
    parser.add_argument('--output', type=str, default='../../results', 
                       help='结果基础目录')
    parser.add_argument('--conf', type=float, default=0.45, 
                       help='YOLO置信度阈值 (越高=越严格，减少误检) (默认: 0.45)')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='抽帧参数: 1=处理所有帧, 3=每隔3帧处理1帧, 5=每隔5帧处理1帧 (默认: 1)')
    parser.add_argument('--model', type=str, default='yolo11m',
                       help='YOLO 模型: yolo11n(快速), yolo11m(中等,更精确), yolo11l(最精确) (默认: yolo11m)')
    parser.add_argument('--min-track-length', type=int, default=3,
                       help='最小轨迹长度(帧数)，短于此的轨迹被认为是误检并排除 (默认: 3)')
    
    args = parser.parse_args()
    
    pipeline = YOLOFirstPipelineA(args.video, args.homography, args.output, 
                                  skip_frames=args.skip_frames, 
                                  model=args.model,
                                  min_track_length=args.min_track_length)
    pipeline.run(args.conf)
