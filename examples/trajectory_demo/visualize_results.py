#!/usr/bin/env python3
"""
可视化碰撞检测结果
- 在视频上绘制检测框、轨迹ID、速度、距离等
- 标注关键帧（近距离事件发生的帧）
- 输出标注视频和关键帧截图
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class ResultVisualizer:
    """结果可视化工具"""
    
    def __init__(self, video_path, results_dir, output_dir=None):
        """
        Args:
            video_path: 原始视频路径
            results_dir: 管道输出结果目录
            output_dir: 可视化输出目录（默认在results_dir内的visualization子目录）
        """
        self.video_path = Path(video_path)
        self.results_dir = Path(results_dir)
        
        # 默认输出目录：在results_dir内创建visualization文件夹
        if output_dir is None:
            self.output_dir = self.results_dir / "visualization"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        detections_raw = self._load_json(self.results_dir / "1_yolo_detection" / "detections_pixel.json")
        tracks_raw = self._load_json(self.results_dir / "2_trajectories" / "tracks.json")
        self.proximity_events = self._load_json(self.results_dir / "3_key_frames" / "proximity_events.json")
        self.collision_events = self._load_json(self.results_dir / "5_collision_analysis" / "collision_events.json")
        
        # 处理detections格式（list或dict）
        if isinstance(detections_raw, list):
            # 将list转换为dict方便查询：{frame_id: frame_data}
            self.detections = {det['frame']: det for det in detections_raw}
            self.detections_list = detections_raw
        else:
            self.detections = detections_raw.get('detections', {}) if isinstance(detections_raw, dict) else {}
            # 如果是dict，转换为list格式方便遍历
            self.detections_list = list(self.detections.values()) if isinstance(self.detections, dict) else []
        
        # 处理tracks格式
        if isinstance(tracks_raw, dict) and 'tracks' in tracks_raw:
            self.tracks = tracks_raw['tracks']
        else:
            self.tracks = tracks_raw if isinstance(tracks_raw, dict) else {}
        
        # 打开视频
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 颜色映射（不同类别）
        self.class_colors = {
            'person': (0, 255, 0),      # 绿
            'car': (0, 0, 255),         # 红
            'motorcycle': (255, 0, 0),  # 蓝
            'truck': (255, 165, 0),     # 橙
            'bus': (255, 0, 255),       # 紫
        }
        
        # 事件帧映射（快速查询）
        self.keyframe_events = defaultdict(list)
        for event in self.collision_events or []:
            frame_id = event.get('frame')  # 或 'frame_id'
            if frame_id is not None:
                self.keyframe_events[frame_id].append(event)
        
        # 轨迹映射（快速查询）
        self.track_by_id = {}
        for track_id, track_points in self.tracks.items():
            self.track_by_id[int(track_id)] = track_points
        
        print(f"✓ 数据加载完成")
        print(f"  视频: {self.video_path.name} ({self.width}x{self.height}, {self.fps:.1f}fps, {self.total_frames}帧)")
        print(f"  检测: {len(self.detections_list)} 帧")
        print(f"  轨迹: {len(self.track_by_id)} 条")
        print(f"  事件: {len(self.collision_events or [])} 个")
    
    def _load_json(self, path):
        """加载JSON文件"""
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    
    def _get_color(self, class_name):
        """获取类别颜色"""
        return self.class_colors.get(class_name, (200, 200, 200))
    
    def _get_track_data_at_frame(self, track_id, frame_id):
        """获取某帧的轨迹数据"""
        if track_id not in self.track_by_id:
            return None
        track_points = self.track_by_id[track_id]
        if isinstance(track_points, list):
            for point in track_points:
                if point.get('frame') == frame_id:
                    return point
        return None
    
    def _get_detections_at_frame(self, frame_id):
        """获取某帧的检测结果"""
        if not self.detections:
            return []
        frame_data = self.detections.get(frame_id)
        return frame_data.get('objects', []) if frame_data else []
    
    def generate_annotated_video(self):
        """生成标注视频 - 在所有帧上添加YOLO检测标注，关键帧额外标注事件和距离连接线"""
        output_path = self.output_dir / f"annotated_{self.video_path.stem}.mp4"
        
        # 建立frame -> detections的映射以快速查询
        frame_detections_map = {}
        for det_frame in self.detections.values() if isinstance(self.detections, dict) else self.detections:
            if isinstance(det_frame, dict):
                frame_id = det_frame.get('frame')
                frame_detections_map[frame_id] = det_frame.get('objects', [])
        
        # 视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        frame_id = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print(f"  处理视频帧数: {self.total_frames}")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 获取该帧的YOLO检测结果
            dets = frame_detections_map.get(frame_id, [])
            
            # 建立track_id -> bbox的映射，用于关键帧绘制连接线
            track_bbox_map = {}
            
            # 在所有帧上绘制YOLO检测框
            if dets:
                for det in dets:
                    track_id = det.get('track_id', -1)
                    class_id = det.get('class', -1)
                    conf = det.get('conf', 0)
                    bbox_xywh = det.get('bbox_xywh', [])
                    
                    if not bbox_xywh:
                        continue
                    
                    # 转换 xywh -> xyxy
                    x_center, y_center, w, h = bbox_xywh
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    # 保存bbox用于关键帧绘制
                    track_bbox_map[track_id] = {
                        'bbox': (x1, y1, x2, y2),
                        'center': (int(x_center), int(y_center)),
                        'class': class_id
                    }
                    
                    # 获取类别名称
                    class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
                                  4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}
                    class_name = class_names.get(class_id, f'class_{class_id}')
                    color = self._get_color(class_name)
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制文本标签 (ID + 类别 + 置信度)
                    label = f"ID{track_id} {class_name} {conf:.2f}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 如果是关键帧，额外标注事件信息和距离连接线
            if frame_id in self.keyframe_events:
                events = self.keyframe_events[frame_id]
                
                # 绘制警告背景
                cv2.rectangle(frame, (5, 5), (550, 35), (0, 0, 255), -1)
                cv2.putText(frame, f"⚠ KEYFRAME - {len(events)} collision event(s)", (15, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 为每个事件绘制距离连接线和标注
                y_offset = 45
                for event_idx, event in enumerate(events):
                    dist = event.get('distance_meters', 0)
                    level = event.get('level_name', 'Unknown')
                    id1 = event.get('track_id_1', -1)
                    id2 = event.get('track_id_2', -1)
                    class1 = event.get('class_1', '?')
                    class2 = event.get('class_2', '?')
                    
                    # 事件文字标注
                    text = f"Event {event_idx+1}: ID{id1}({class1}) <-> ID{id2}({class2}) | Dist: {dist:.2f}m | {level}"
                    cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset += 25
                    
                    # 在两辆车之间绘制连接线和距离标注
                    if id1 in track_bbox_map and id2 in track_bbox_map:
                        center1 = track_bbox_map[id1]['center']
                        center2 = track_bbox_map[id2]['center']
                        
                        # 根据距离等级选择颜色
                        if 'Collision' in level:
                            line_color = (0, 0, 255)  # 红色：碰撞
                        elif 'Near Miss' in level:
                            line_color = (0, 165, 255)  # 橙色：接近
                        else:
                            line_color = (0, 255, 255)  # 黄色：其他
                        
                        # 绘制连接线
                        cv2.line(frame, center1, center2, line_color, 2)
                        
                        # 在连接线中点标注距离
                        mid_x = (center1[0] + center2[0]) // 2
                        mid_y = (center1[1] + center2[1]) // 2
                        
                        # 背景框
                        dist_text = f"{dist:.2f}m"
                        text_size = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, 
                                    (mid_x - text_size[0]//2 - 3, mid_y - text_size[1]//2 - 3),
                                    (mid_x + text_size[0]//2 + 3, mid_y + text_size[1]//2 + 3),
                                    line_color, -1)
                        
                        # 距离数值
                        cv2.putText(frame, dist_text, (mid_x - text_size[0]//2, mid_y + text_size[1]//2 - 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 右上角显示帧号和时间戳
            timestamp = frame_id / self.fps if self.fps > 0 else 0
            frame_info = f"Frame: {frame_id} | Time: {timestamp:.2f}s"
            text_size = cv2.getTextSize(frame_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (self.width - text_size[0] - 10, 5), (self.width - 5, 30), (0, 0, 0), -1)
            cv2.putText(frame, frame_info, (self.width - text_size[0] - 5, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 写入视频
            writer.write(frame)
            frame_id += 1
            
            if frame_id % 20 == 0:
                print(f"  处理中: {frame_id}/{self.total_frames} 帧", end='\r')
        
        writer.release()
        print(f"\n✓ 标注视频已保存: {output_path}")
        return output_path
    
    def generate_keyframe_images(self):
        """关键帧已包含在标注视频中，无需单独保存"""
        print(f"✓ 关键帧已包含在标注视频中，跳过单独保存")
        return None
    
    def generate_csv_report(self):
        """生成CSV格式的报告"""
        import csv
        
        csv_path = self.output_dir / "events_summary.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'event_id', 'frame_id', 'timestamp', 'track_id_1', 'track_id_2',
                'class_1', 'class_2', 'distance_meters', 'level', 'speed_1_ms', 'speed_2_ms'
            ])
            writer.writeheader()
            
            for idx, event in enumerate(self.collision_events or [], 1):
                frame_id = event.get('frame_id', -1)
                timestamp = frame_id / self.fps if frame_id >= 0 else 0
                
                track_id_1 = event.get('track_id_1')
                track_id_2 = event.get('track_id_2')
                
                # 获取速度信息
                speed_1 = 0
                speed_2 = 0
                if track_id_1 is not None:
                    track_data_1 = self._get_track_data_at_frame(track_id_1, frame_id)
                    speed_1 = track_data_1.get('speed_world', 0) if track_data_1 else 0
                if track_id_2 is not None:
                    track_data_2 = self._get_track_data_at_frame(track_id_2, frame_id)
                    speed_2 = track_data_2.get('speed_world', 0) if track_data_2 else 0
                
                writer.writerow({
                    'event_id': idx,
                    'frame_id': frame_id,
                    'timestamp': f"{timestamp:.2f}s",
                    'track_id_1': track_id_1,
                    'track_id_2': track_id_2,
                    'class_1': event.get('class_1', ''),
                    'class_2': event.get('class_2', ''),
                    'distance_meters': f"{event.get('distance_meters', 0):.3f}",
                    'level': event.get('level_name', ''),
                    'speed_1_ms': f"{speed_1:.2f}",
                    'speed_2_ms': f"{speed_2:.2f}",
                })
        
        print(f"✓ CSV报告已保存: {csv_path}")
        return csv_path
    
    def generate_summary_report(self):
        """生成文本摘要报告"""
        report_path = self.output_dir / "visualization_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("碰撞检测结果可视化报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"视频信息\n")
            f.write(f"  文件: {self.video_path.name}\n")
            f.write(f"  分辨率: {self.width}x{self.height}\n")
            f.write(f"  帧率: {self.fps:.1f} fps\n")
            f.write(f"  总帧数: {self.total_frames}\n")
            f.write(f"  时长: {self.total_frames / self.fps:.1f}s\n\n")
            
            f.write(f"检测统计\n")
            f.write(f"  轨迹总数: {len(self.track_by_id)}\n")
            f.write(f"  检测帧数: {len(self.detections.get('detections', {}))} 帧\n")
            f.write(f"  关键帧数: {len(self.collision_events or [])} 帧\n\n")
            
            f.write(f"事件分级\n")
            level_counts = defaultdict(int)
            for event in self.collision_events or []:
                level = event.get('level_name', 'Unknown')
                level_counts[level] += 1
            
            for level, count in sorted(level_counts.items()):
                f.write(f"  {level}: {count}\n")
            
            f.write(f"\n事件详情\n")
            for idx, event in enumerate(self.collision_events or [], 1):
                frame_id = event.get('frame_id', -1)
                timestamp = frame_id / self.fps if frame_id >= 0 else 0
                track_id_1 = event.get('track_id_1')
                track_id_2 = event.get('track_id_2')
                dist = event.get('distance_meters', 0)
                level = event.get('level_name', '')
                
                f.write(f"  [{idx}] Frame {frame_id} ({timestamp:.2f}s)\n")
                f.write(f"      ID{track_id_1}({event.get('class_1', '')}) <-> ID{track_id_2}({event.get('class_2', '')})\n")
                f.write(f"      距离: {dist:.3f}m | 等级: {level}\n\n")
        
        print(f"✓ 摘要报告已保存: {report_path}")
        return report_path
    
    def run(self):
        """执行所有可视化任务"""
        print(f"\n【可视化结果生成】")
        print(f"输出目录: {self.output_dir}\n")
        
        # 生成标注视频
        self.generate_annotated_video()
        
        # 生成关键帧截图
        self.generate_keyframe_images()
        
        # 生成CSV报告
        self.generate_csv_report()
        
        # 生成摘要报告
        self.generate_summary_report()
        
        self.cap.release()
        
        print(f"\n✓ 所有可视化任务完成！")
        print(f"  输出目录: {self.output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="碰撞检测结果可视化")
    parser.add_argument('--video', required=True, help='原始视频路径')
    parser.add_argument('--results', required=True, help='结果文件夹路径')
    parser.add_argument('--output', help='输出文件夹路径（可选）')
    
    args = parser.parse_args()
    
    visualizer = ResultVisualizer(args.video, args.results, args.output)
    visualizer.run()
