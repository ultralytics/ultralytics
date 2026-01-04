"""
collision_detection_pipeline.py

完整的碰撞检测管道：
1. Homography标定 → 验证图
2. 视频透视变换 → warped视频
3. YOLO检测 + 碰撞分析 → 截图事件帧 + JSON + 报告

每次运行都生成新的时间戳文件夹结构
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
from object_state_manager import ObjectStateManager
import coord_transform


class CollisionDetectionPipeline:
    def __init__(self, video_path, homography_path, output_base='../../results'):
        """初始化pipeline
        
        Args:
            video_path: 原始视频路径
            homography_path: Homography JSON路径
            output_base: 结果基础目录
        """
        self.video_path = video_path
        self.homography_path = homography_path
        self.output_base = Path(output_base)
        
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_base / timestamp
        
        # 创建子目录结构
        self.homography_dir = self.run_dir / "1_homography"
        self.warped_video_dir = self.run_dir / "2_warped_video"
        self.collision_dir = self.run_dir / "3_collision_events"
        
        for d in [self.homography_dir, self.warped_video_dir, self.collision_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"碰撞检测Pipeline")
        print(f"{'='*70}")
        print(f"时间戳: {timestamp}")
        print(f"结果目录: {self.run_dir}")
    
    def load_homography(self):
        """加载Homography矩阵"""
        print(f"\n【步骤1: 加载Homography矩阵】")
        
        with open(self.homography_path) as f:
            data = json.load(f)
        
        self.H = np.array(data['homography_matrix'], dtype=np.float32)
        self.pixel_points = data['pixel_points']
        self.world_points = data['world_points']
        
        # 保存一份到输出目录
        with open(self.homography_dir / 'homography.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Homography矩阵已加载")
        print(f"  像素点数: {len(self.pixel_points)}")
        return self.H, self.pixel_points, self.world_points
    
    def create_verification_image(self):
        """生成Homography验证图"""
        print(f"\n【步骤1.5: 生成验证图】")
        
        # 读取第一帧
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"❌ 无法读取视频")
            return
        
        # 在原图上标注参考点
        frame_marked = frame.copy()
        for i, (px, py) in enumerate(self.pixel_points):
            cv2.circle(frame_marked, (int(px), int(py)), 20, (0, 255, 0), 3)
            cv2.putText(frame_marked, f"{i+1}", (int(px)+30, int(py)-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # 保存验证图
        verify_path = self.homography_dir / 'verify_original.jpg'
        cv2.imwrite(str(verify_path), frame_marked)
        
        print(f"✓ 验证图已保存: {verify_path.name}")
        return frame_marked
    
    def transform_video(self):
        """对视频进行透视变换"""
        print(f"\n【步骤2: 视频透视变换】")
        
        # 计算输出→像素的映射矩阵
        min_x = min(w[0] for w in self.world_points)
        max_x = max(w[0] for w in self.world_points)
        min_y = min(w[1] for w in self.world_points)
        max_y = max(w[1] for w in self.world_points)
        
        world_bounds = (min_x, max_x, min_y, max_y)
        # 增大分辨率以获得更好的检测效果
        # OpenCV warpPerspective需要 (宽, 高) 顺序！
        output_width = 2400
        output_height = 360
        output_size = (output_width, output_height)
        
        # 构造A矩阵（输出→世界）
        world_width = max_x - min_x
        world_height = max_y - min_y
        A = np.array([
            [world_width / output_width, 0, min_x],
            [0, -world_height / output_height, max_y],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 计算M = H_inv @ A（输出→像素）
        H_inv = np.linalg.inv(self.H)
        M = H_inv @ A
        
        # 处理视频
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        warped_path = self.warped_video_dir / 'warped.mp4'
        # 使用更兼容的编码（H.264 而不是H.265）
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码，更兼容
        out = cv2.VideoWriter(str(warped_path), fourcc, fps, output_size)
        
        if not out.isOpened():
            # 如果avc1失败，尝试其他编码
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter(str(warped_path), fourcc, fps, output_size)
        
        print(f"处理中: {total_frames}帧 @ {fps:.2f}FPS...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 使用warpPerspective变换
            warped = cv2.warpPerspective(frame, M, output_size)
            out.write(warped)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")
        
        cap.release()
        out.release()
        
        print(f"✓ warped视频已保存: {warped_path.name}")
        print(f"  分辨率: {output_size[0]}×{output_size[1]}")
        self.warped_video_path = str(warped_path)
        return str(warped_path)
    
    def detect_collisions(self, conf_threshold=0.45):
        """运行YOLO检测和碰撞分析"""
        print(f"\n【步骤3: YOLO检测 + 碰撞分析】")
        
        # 加载YOLO
        model = YOLO('yolo11n.pt')
        
        # 加载Homography用于坐标变换
        H = coord_transform.load_homography(self.homography_path)
        osm = ObjectStateManager(H=H)
        
        cap = cv2.VideoCapture(self.warped_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        collision_events = []
        detected_frames = []  # 保存所有检测到物体的帧
        frame_count = 0
        object_detected = 0
        
        print(f"处理中: {total_frames}帧...")
        
        for result in model.track(source=self.warped_video_path, stream=True, 
                                 persist=True, conf=conf_threshold):
            frame_count += 1
            
            # 获取检测结果
            if result.boxes is None or len(result.boxes) == 0:
                if frame_count % 30 == 0:
                    print(f"  Frame {frame_count}/{total_frames} - 无物体")
                continue
            
            # 检测到物体
            object_detected += 1
            boxes = result.boxes.xywh.cpu().numpy()
            ids = result.boxes.id
            
            # 保存检测帧
            frame_img = result.orig_img
            frame_path = self.collision_dir / f"detected_frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame_img)
            detected_frames.append({
                'frame': frame_count,
                'time': frame_count / fps,
                'objects_count': len(boxes),
                'frame_image': frame_path.name
            })
            
            if ids is None or len(ids) < 2:
                if frame_count % 30 == 0:
                    print(f"  Frame {frame_count}/{total_frames} - {len(boxes)}个物体")
                continue
            
            # 计算所有物体对之间的距离
            collision_in_frame = False
            for i in range(len(boxes)):
                for j in range(i+1, len(boxes)):
                    x1, y1 = boxes[i][:2]
                    x2, y2 = boxes[j][:2]
                    
                    # 使用世界坐标计算距离（如果有H矩阵）
                    if H is not None:
                        try:
                            p1_world = coord_transform.transform_point((x1, y1), H)
                            p2_world = coord_transform.transform_point((x2, y2), H)
                            distance = np.sqrt((p1_world[0]-p2_world[0])**2 + 
                                            (p1_world[1]-p2_world[1])**2)
                            distance_str = f"{distance:.2f}m"
                        except:
                            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                            distance_str = f"{distance:.1f}px"
                    else:
                        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        distance_str = f"{distance:.1f}px"
                    
                    # 记录接近事件（距离 < 0.5m）
                    if distance < 0.5 or (H is None and distance < 50):
                        collision_in_frame = True
                        
                        event = {
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'object_ids': (int(ids[i]), int(ids[j])),
                            'distance': float(distance),
                            'distance_str': distance_str,
                            'frame_image': frame_path.name
                        }
                        collision_events.append(event)
            
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count}/{total_frames} - {len(boxes)}个物体")
        
        cap.release()
        
        # 创建 frames 子目录（如果有检测结果）
        frames_dir = self.collision_dir / 'frames'
        if object_detected > 0:
            frames_dir.mkdir(exist_ok=True)
            # 移动检测帧到frames子目录
            for frame_file in self.collision_dir.glob('detected_frame_*.jpg'):
                frame_file.rename(frames_dir / frame_file.name)
        
        # 保存碰撞事件JSON
        events_path = self.collision_dir / 'collision_events.json'
        with open(events_path, 'w') as f:
            json.dump(collision_events, f, indent=2)
        
        # 保存检测帧信息
        detected_path = self.collision_dir / 'detected_frames.json'
        with open(detected_path, 'w') as f:
            json.dump(detected_frames, f, indent=2)
        
        print(f"✓ 检测完成: {object_detected}帧检测到物体, {len(collision_events)}个碰撞事件")
        print(f"✓ 检测帧已保存: {frames_dir.name}/ (共{len(detected_frames)}帧)")
        print(f"✓ 事件JSON已保存: {events_path.name}")
        
        # 生成报告
        self.generate_report(collision_events, fps, object_detected)
        
        return collision_events
    
    def generate_report(self, events, fps, objects_detected=0):
        """生成分析报告"""
        report_path = self.collision_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("碰撞检测分析报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入视频: {self.video_path}\n")
            f.write(f"Homography: {self.homography_path}\n")
            f.write(f"结果目录: {self.run_dir}\n\n")
            
            f.write(f"检测统计:\n")
            f.write(f"  - 检测到物体的帧数: {objects_detected}\n")
            f.write(f"  - 碰撞事件数: {len(events)}\n\n")
            
            if events:
                f.write("碰撞事件列表:\n")
                f.write("-"*70 + "\n")
                for i, evt in enumerate(events, 1):
                    f.write(f"{i}. Frame {evt['frame']} ({evt['time']:.2f}s)\n")
                    f.write(f"   物体ID: {evt['object_ids']}\n")
                    f.write(f"   距离: {evt['distance_str']}\n")
                    f.write(f"   截图: {evt['frame_image']}\n\n")
            else:
                f.write("未检测到碰撞事件\n\n")
            
            f.write("="*70 + "\n")
            f.write("报告结束\n")
        
        print(f"✓ 报告已保存: {report_path.name}")
    
    def run(self, conf_threshold=0.45):
        """运行完整pipeline"""
        try:
            self.load_homography()
            self.create_verification_image()
            self.transform_video()
            self.detect_collisions(conf_threshold)
            
            print(f"\n{'='*70}")
            print(f"✓ Pipeline完成！")
            print(f"{'='*70}")
            print(f"结果保存在: {self.run_dir}")
            print(f"\n文件夹结构:")
            print(f"  1_homography/")
            print(f"    ├── homography.json")
            print(f"    └── verify_original.jpg")
            print(f"  2_warped_video/")
            print(f"    └── warped.mp4")
            print(f"  3_collision_events/")
            print(f"    ├── event_frame_*.jpg (碰撞事件帧)")
            print(f"    ├── collision_events.json")
            print(f"    └── analysis_report.txt")
            
        except Exception as e:
            print(f"❌ Pipeline错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='碰撞检测完整Pipeline')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--homography', type=str, required=True, help='Homography JSON路径')
    parser.add_argument('--output', type=str, default='../../results', help='结果基础目录')
    parser.add_argument('--conf', type=float, default=0.45, help='YOLO置信度阈值')
    
    args = parser.parse_args()
    
    pipeline = CollisionDetectionPipeline(args.video, args.homography, args.output)
    pipeline.run(args.conf)
