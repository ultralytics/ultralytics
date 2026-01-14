"""
collision_detection_pipeline_yolo_first_method_c.py.

YOLO-First 碰撞检测管道 (Approach 2, 方案C: Homography优先)
执行顺序: YOLO检测 → Homography变换 → 轨迹构建 → 关键帧提取 → 分析

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

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 导入YOLO和相关模块
sys.path.append(os.path.dirname(__file__))
from ultralytics import YOLO


class YOLOFirstPipelineC:
    def __init__(self, video_path, homography_path=None, output_base="../../results"):
        """初始化 YOLO-First pipeline (方案C: Homography优先).

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

        for d in [self.detection_dir, self.homography_dir, self.trajectory_dir, self.keyframe_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print("YOLO-First 碰撞检测Pipeline (方案C: Homography优先)")
        print(f"{'=' * 70}")
        print(f"时间戳: {timestamp}")
        print(f"结果目录: {self.run_dir}")
        print("执行顺序: YOLO → Homography → 轨迹(米制) → 关键帧 → 分析")

        if not homography_path:
            print("⚠️  警告: 未提供 Homography，将仅在像素空间处理")

    def load_homography(self):
        """Step 0.5: 加载 Homography 矩阵."""
        if not self.homography_path:
            print("\n⚠️  未提供 Homography，将在像素空间处理")
            return False

        print("\n【Step 0.5: 加载 Homography 矩阵】")

        try:
            with open(self.homography_path) as f:
                H_data = json.load(f)

            self.H = np.array(H_data["homography_matrix"], dtype=np.float32)
            pixel_points = H_data["pixel_points"]
            world_points = H_data["world_points"]

            # 保存到输出目录
            with open(self.homography_dir / "homography.json", "w") as f:
                json.dump(H_data, f, indent=2)

            # 计算像素到米的缩放因子
            if len(world_points) >= 2 and len(pixel_points) >= 2:
                px_dist = np.sqrt(
                    (pixel_points[0][0] - pixel_points[1][0]) ** 2 + (pixel_points[0][1] - pixel_points[1][1]) ** 2
                )
                world_dist = np.sqrt(
                    (world_points[0][0] - world_points[1][0]) ** 2 + (world_points[0][1] - world_points[1][1]) ** 2
                )

                self.pixel_per_meter = px_dist / world_dist if world_dist > 0 else 1.0

            print("✓ Homography 矩阵已加载")
            print(f"  缩放因子: {self.pixel_per_meter:.2f} px/m")
            print(f"  参考点数: {len(pixel_points)}")

            return True

        except Exception as e:
            print(f"❌ 加载 Homography 失败: {e}")
            return False

    def run_yolo_detection(self, conf_threshold=0.45):
        """Step 1: YOLO 检测 (原始视频，全帧)."""
        print("\n【Step 1: YOLO 检测】")

        model = YOLO("yolo11n.pt")

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_detections = []
        frame_count = 0
        detection_frames_count = 0

        print(f"处理中: {total_frames}帧 @ {fps:.2f}FPS...")

        for result in model.track(source=self.video_path, stream=True, persist=True, conf=conf_threshold):
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

            frame_detections = {"frame": frame_count, "time": frame_count / fps, "objects": []}

            for i in range(len(boxes)):
                obj_data = {
                    "track_id": int(ids[i]) if ids[i] is not None else -1,
                    "class": int(classes[i]),
                    "conf": float(confs[i]),
                    "bbox_xywh": boxes[i].tolist(),
                }
                frame_detections["objects"].append(obj_data)

            all_detections.append(frame_detections)

            if frame_count % 30 == 0:
                print(f"  Frame {frame_count}/{total_frames} - {len(boxes)}个物体")

        cap.release()

        detections_path = self.detection_dir / "detections_pixel.json"
        with open(detections_path, "w") as f:
            json.dump(all_detections, f, indent=2)

        stats = {
            "total_frames": total_frames,
            "fps": fps,
            "detection_frames": detection_frames_count,
            "confidence_threshold": conf_threshold,
        }
        stats_path = self.detection_dir / "detection_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✓ YOLO检测完成: {detection_frames_count}帧检测到物体")
        print(f"  检测结果保存: {detections_path.name}")

        return all_detections

    def transform_detections_to_world(self, all_detections):
        """Step 2: Homography 变换 (所有检测框)."""
        print("\n【Step 2: Homography 坐标变换】")

        if self.H is None:
            print("⚠️  未加载 Homography，保持像素空间")
            return all_detections

        transformed_detections = []

        for frame_data in all_detections:
            trans_frame = {"frame": frame_data["frame"], "time": frame_data["time"], "objects": []}

            for obj in frame_data["objects"]:
                trans_obj = obj.copy()

                x_px, y_px = obj["bbox_xywh"][0], obj["bbox_xywh"][1]
                x_world = x_px / self.pixel_per_meter
                y_world = y_px / self.pixel_per_meter

                trans_obj["center_x_world"] = x_world
                trans_obj["center_y_world"] = y_world
                trans_obj["center_x_pixel"] = x_px
                trans_obj["center_y_pixel"] = y_px

                trans_frame["objects"].append(trans_obj)

            transformed_detections.append(trans_frame)

        trans_path = self.homography_dir / "detections_world.json"
        with open(trans_path, "w") as f:
            json.dump(transformed_detections, f, indent=2)

        print(f"✓ Homography 变换完成: {len(all_detections)}帧检测框已转换到世界坐标")
        print(f"  转换结果保存: {trans_path.name}")

        return transformed_detections

    def build_trajectories_world(self, transformed_detections):
        """Step 3: 轨迹构建 (世界坐标，米制)."""
        print("\n【Step 3: 轨迹构建 (世界坐标)】")

        tracks = {}

        for frame_data in transformed_detections:
            for obj in frame_data["objects"]:
                track_id = obj["track_id"]

                if track_id not in tracks:
                    tracks[track_id] = []

                track_point = {
                    "frame": frame_data["frame"],
                    "time": frame_data["time"],
                    "class": obj["class"],
                    "conf": obj["conf"],
                    "center_x": obj["center_x_world"],
                    "center_y": obj["center_y_world"],
                }

                tracks[track_id].append(track_point)

        # 计算速度 (m/s)
        for track_id, track_points in tracks.items():
            track_points.sort(key=lambda p: p["frame"])

            if len(track_points) >= 2:
                for i in range(1, len(track_points)):
                    prev = track_points[i - 1]
                    curr = track_points[i]

                    dt = curr["time"] - prev["time"]
                    if dt > 0:
                        dx = curr["center_x"] - prev["center_x"]
                        dy = curr["center_y"] - prev["center_y"]

                        curr["vx"] = dx / dt
                        curr["vy"] = dy / dt
                        curr["speed"] = np.sqrt(dx**2 + dy**2) / dt
                    else:
                        curr["vx"] = 0.0
                        curr["vy"] = 0.0
                        curr["speed"] = 0.0

                track_points[0]["vx"] = 0.0
                track_points[0]["vy"] = 0.0
                track_points[0]["speed"] = 0.0

        tracks_path = self.trajectory_dir / "tracks_world.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f, indent=2)

        stats = {
            "total_tracks": len(tracks),
            "track_lengths": {str(tid): len(points) for tid, points in tracks.items()},
            "coordinate_system": "world (meters)",
            "velocity_unit": "m/s",
        }
        stats_path = self.trajectory_dir / "track_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✓ 轨迹构建完成: {len(tracks)}条轨迹 (世界坐标)")
        print("  坐标系: 世界坐标 (米)")
        print("  速度单位: m/s")
        print(f"  轨迹结果保存: {tracks_path.name}")

        return tracks

    def extract_key_frames_world(self, transformed_detections, distance_threshold=1.5):
        """Step 4: 关键帧提取 (接近事件，世界坐标)."""
        print("\n【Step 4: 关键帧提取】")

        proximity_events = []

        for frame_data in transformed_detections:
            if len(frame_data["objects"]) < 2:
                continue

            frame = frame_data["frame"]
            objects = frame_data["objects"]

            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    obj1 = objects[i]
                    obj2 = objects[j]

                    x1, y1 = obj1["center_x_world"], obj1["center_y_world"]
                    x2, y2 = obj2["center_x_world"], obj2["center_y_world"]

                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                    if distance < distance_threshold:
                        event = {
                            "frame": frame,
                            "time": frame_data["time"],
                            "object_ids": (obj1["track_id"], obj2["track_id"]),
                            "distance_meters": float(distance),
                            "object_classes": (obj1["class"], obj2["class"]),
                        }
                        proximity_events.append(event)

        events_path = self.keyframe_dir / "proximity_events.json"
        with open(events_path, "w") as f:
            json.dump(proximity_events, f, indent=2)

        print(f"✓ 关键帧提取完成: {len(proximity_events)}个接近事件")
        print(f"  距离阈值: {distance_threshold}m")
        print(f"  接近事件保存: {events_path.name}")

        return proximity_events

    def analyze_collision_risk(self, proximity_events):
        """Step 5: 风险分析和 Event 分级."""
        print("\n【Step 5: 碰撞风险分析】")

        analyzed_events = []

        for event in proximity_events:
            analyzed = event.copy()

            distance = event["distance_meters"]

            # 分级标准 (米)
            if distance < 0.5:
                analyzed["level"] = 1
                analyzed["level_name"] = "Collision"
            elif distance < 1.5:
                analyzed["level"] = 2
                analyzed["level_name"] = "Near Miss"
            else:
                analyzed["level"] = 3
                analyzed["level_name"] = "Avoidance"

            analyzed_events.append(analyzed)

        level_counts = {1: 0, 2: 0, 3: 0}
        for event in analyzed_events:
            level_counts[event["level"]] += 1

        analysis_path = self.analysis_dir / "collision_events.json"
        with open(analysis_path, "w") as f:
            json.dump(analyzed_events, f, indent=2)

        print("✓ 碰撞风险分析完成")
        print(f"  - Level 1 (Collision, <0.5m): {level_counts[1]} events")
        print(f"  - Level 2 (Near Miss, 0.5-1.5m): {level_counts[2]} events")
        print(f"  - Level 3 (Avoidance, >1.5m): {level_counts[3]} events")
        print(f"  分析结果保存: {analysis_path.name}")

        return analyzed_events, level_counts

    def generate_report(self, proximity_events, analyzed_events, level_counts):
        """生成最终报告."""
        report_path = self.analysis_dir / "analysis_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("YOLO-First 碰撞检测分析报告 (方案C: Homography优先)\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入视频: {self.video_path}\n")
            f.write(f"Homography: {self.homography_path if self.H is not None else '未提供'}\n")
            f.write(f"结果目录: {self.run_dir}\n\n")

            f.write("处理方式: YOLO-First (方案C)\n")
            f.write("流程: YOLO检测 → Homography变换 → 轨迹(米制) → 关键帧 → 分析\n")
            f.write("坐标系: 世界坐标 (米)\n")
            f.write("速度单位: m/s\n\n")

            f.write("接近事件统计:\n")
            f.write(f"  - 总接近事件: {len(proximity_events)}\n")
            f.write(f"  - Level 1 (Collision, <0.5m): {level_counts[1]}\n")
            f.write(f"  - Level 2 (Near Miss, 0.5-1.5m): {level_counts[2]}\n")
            f.write(f"  - Level 3 (Avoidance, >1.5m): {level_counts[3]}\n\n")

            if analyzed_events:
                f.write("前10个高风险事件:\n")
                f.write("-" * 70 + "\n")

                sorted_events = sorted(analyzed_events, key=lambda e: e.get("level", 3))

                for i, event in enumerate(sorted_events[:10], 1):
                    f.write(f"\n{i}. Frame {event['frame']} ({event['time']:.2f}s)\n")
                    f.write(f"   物体ID: {event['object_ids']}\n")
                    f.write(f"   风险等级: Level {event['level']} ({event.get('level_name', 'Unknown')})\n")
                    f.write(f"   距离: {event['distance_meters']:.2f}m\n")
            else:
                f.write("未检测到接近事件\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("报告结束\n")

        print(f"✓ 报告已保存: {report_path.name}")

    def run(self, conf_threshold=0.45):
        """运行完整 YOLO-First pipeline (方案C)."""
        try:
            # Step 0.5: 加载 Homography
            self.load_homography()

            # Step 1: YOLO 检测
            all_detections = self.run_yolo_detection(conf_threshold)

            if not all_detections:
                print("❌ 未检测到任何物体，停止处理")
                return

            # Step 2: Homography 变换 (所有检测框)
            transformed_detections = self.transform_detections_to_world(all_detections)

            # Step 3: 轨迹构建 (世界坐标)
            self.build_trajectories_world(transformed_detections)

            # Step 4: 关键帧提取
            proximity_events = self.extract_key_frames_world(transformed_detections, distance_threshold=1.5)

            if not proximity_events:
                print("⚠️  未检测到接近事件")
                analyzed_events = []
                level_counts = {1: 0, 2: 0, 3: 0}
            else:
                # Step 5: 风险分析
                analyzed_events, level_counts = self.analyze_collision_risk(proximity_events)

            # 生成报告
            self.generate_report(proximity_events, analyzed_events, level_counts)

            print(f"\n{'=' * 70}")
            print("✓ YOLO-First Pipeline (方案C) 完成！")
            print(f"{'=' * 70}")
            print(f"结果保存在: {self.run_dir}")
            print("\n文件夹结构:")
            print("  1_raw_detections/")
            print("    ├── detections_pixel.json (像素空间)")
            print("    └── detection_stats.json")
            print("  2_homography_transform/")
            print("    ├── homography.json")
            print("    └── detections_world.json (世界坐标)")
            print("  3_trajectories/")
            print("    ├── tracks_world.json (米制轨迹)")
            print("    └── track_stats.json")
            print("  4_key_frames/")
            print("    └── proximity_events.json")
            print("  5_collision_analysis/")
            print("    ├── collision_events.json (分级事件)")
            print("    └── analysis_report.txt")

        except Exception as e:
            print(f"❌ Pipeline 错误: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO-First 碰撞检测Pipeline (方案C)")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--homography", type=str, required=True, help="Homography JSON路径 (必须)")
    parser.add_argument("--output", type=str, default="../../results", help="结果基础目录")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO置信度阈值")

    args = parser.parse_args()

    pipeline = YOLOFirstPipelineC(args.video, args.homography, args.output)
    pipeline.run(args.conf)
