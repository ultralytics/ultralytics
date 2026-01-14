#!/usr/bin/env python3
"""
collision_detection_pipeline_debug.py.

分阶段调试版本，支持：
- 各阶段暂停点
- 生成可视化检测框视频
- YOLO检测统计报告
- 跳帧处理加快速度
- 输出关键帧（带检测框）
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 导入YOLO和相关模块
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from homography_transform_utils import compute_transformation_matrix, load_homography, transform_frame_manual
from ultralytics import YOLO


class DebugCollisionDetectionPipeline:
    def __init__(self, video_path, homography_path, output_base="../../results", frame_skip=1, debug_mode=True):
        """初始化pipeline.

        Args:
            video_path: 原始视频路径
            homography_path: Homography JSON路径
            output_base: 结果基础目录
            frame_skip: 跳帧数（1=不跳帧，10=每10帧处理1帧）
            debug_mode: 是否启用调试模式（生成更多可视化）
        """
        self.video_path = video_path
        self.homography_path = homography_path
        self.output_base = Path(output_base)
        self.frame_skip = frame_skip
        self.debug_mode = debug_mode

        # 创建时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_base / self.timestamp

        print("=" * 70)
        print("碰撞检测Pipeline - 分阶段调试版本")
        print("=" * 70)
        print(f"时间戳: {self.timestamp}")
        print(f"结果目录: {self.run_dir}")
        print(f"跳帧数: {self.frame_skip}")
        if self.debug_mode:
            print("调试模式: 已启用 ✓")
        print()

        # 创建子目录
        self.homography_dir = self.run_dir / "1_homography"
        self.warped_video_dir = self.run_dir / "2_warped_video"
        self.yolo_dir = self.run_dir / "3_yolo_detection"
        self.collision_dir = self.run_dir / "4_collision_analysis"

        for dir_path in [self.homography_dir, self.warped_video_dir, self.yolo_dir, self.collision_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_homography(self):
        """加载Homography矩阵."""
        print("\n【步骤1: 加载Homography矩阵】")

        self.H, self.world_points = load_homography(self.homography_path)

        pixel_points = self.get_pixel_points_from_homography()

        print("✓ Homography矩阵已加载")
        print(f"  像素点数: {len(pixel_points)}")
        print("  世界坐标范围:")
        print(f"    X: [{min(w[0] for w in self.world_points):.2f}, {max(w[0] for w in self.world_points):.2f}]m")
        print(f"    Y: [{min(w[1] for w in self.world_points):.2f}, {max(w[1] for w in self.world_points):.2f}]m")

        return self.H, self.world_points

    def get_pixel_points_from_homography(self):
        """从homography JSON获取像素点."""
        with open(self.homography_path) as f:
            data = json.load(f)
        return data.get("pixel_points", [])

    def create_verification_image(self):
        """生成验证图."""
        print("\n【步骤1.5: 生成验证图】")

        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("❌ 无法读取视频")
            return

        # 绘制标定点
        frame_marked = frame.copy()
        pixel_points = self.get_pixel_points_from_homography()

        for i, (px, py) in enumerate(pixel_points):
            cv2.circle(frame_marked, (int(px), int(py)), 10, (0, 255, 0), 2)
            world_point = self.world_points[i]
            label = f"({world_point[0]:.1f}m, {world_point[1]:.1f}m)"
            cv2.putText(
                frame_marked, label, (int(px) + 15, int(py) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
            )

        verify_path = self.homography_dir / "verify_original.jpg"
        cv2.imwrite(str(verify_path), frame_marked)

        print(f"✓ 验证图已保存: {verify_path.name}")
        return frame_marked

    def pause_checkpoint(self, stage_name):
        """暂停点 - 等待用户确认继续."""
        if not self.debug_mode:
            return

        print(f"\n⏸️  阶段 '{stage_name}' 已完成")
        print(f"📁 输出文件已保存到: {self.run_dir}/{stage_name}/")
        response = input("按Enter继续，或输入'q'退出: ").strip().lower()

        if response == "q":
            print("❌ 用户中止pipeline")
            sys.exit(0)

    def transform_video(self):
        """对视频进行透视变换."""
        print("\n【步骤2: 视频透视变换】")

        # 使用正确的输出尺寸
        output_size = (180, 1200)

        # 计算变换矩阵
        M, world_bounds = compute_transformation_matrix(self.H, self.world_points, output_size)
        min_x, max_x, min_y, max_y = world_bounds

        # 处理视频
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 生成带时间戳的输出文件名
        input_path = Path(self.video_path)
        video_name = input_path.stem
        output_filename = f"{video_name}_warped_{self.timestamp}.mp4"
        warped_path = self.warped_video_dir / output_filename

        # 创建VideoWriter (保持原FPS，但总帧数会减少)
        output_fps = fps / self.frame_skip  # 跳帧后的FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(warped_path), fourcc, output_fps, output_size)

        total_to_process = (total_frames + self.frame_skip - 1) // self.frame_skip
        print(f"处理中: {total_frames}帧 → {total_to_process}帧 (跳帧:{self.frame_skip})...")
        print(f"输出尺寸: {output_size[0]}×{output_size[1]} (宽×高)")

        frame_count = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 跳帧处理
            if (frame_count - 1) % self.frame_skip != 0:
                continue

            # 手工逐像素变换
            warped = transform_frame_manual(frame, M, output_size)
            out.write(warped)

            processed += 1
            if processed % 5 == 0:
                print(f"  {processed}/{total_to_process} ({100 * processed / total_to_process:.1f}%)")

        cap.release()
        out.release()

        print(f"✓ warped视频已保存: {warped_path.name}")
        print(f"  处理帧数: {processed}")
        print(f"  输出FPS: {output_fps:.1f}")
        print(f"  世界坐标范围: X=[{min_x:.2f}, {max_x:.2f}]m, Y=[{min_y:.2f}, {max_y:.2f}]m")
        self.warped_video_path = str(warped_path)

        return str(warped_path)

    def detect_and_visualize(self, conf_threshold=0.45):
        """YOLO检测 + 可视化 + 统计报告 生成： 1. 带检测框的视频 2. YOLO统计报告 3. 关键帧（带检测框）."""
        print("\n【步骤3: YOLO检测 + 可视化】")

        print("加载YOLOv11n模型...")
        model = YOLO("yolo11n.pt")

        # 打开warped视频
        cap = cv2.VideoCapture(self.warped_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建可视化视频
        viz_output = self.yolo_dir / f"yolo_detection_viz_{self.timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        viz_out = cv2.VideoWriter(str(viz_output), fourcc, fps, (width, height))

        # 统计信息
        class_counts = defaultdict(int)
        frames_with_detections = 0
        detected_frames_info = []
        keyframes = []  # 保存关键帧

        print(f"处理中: {total_frames}帧 (跳帧数: {self.frame_skip})...")

        frame_idx = 0
        processed_frame = 0

        for result in model.track(source=self.warped_video_path, stream=True, persist=True, conf=conf_threshold):
            frame_idx += 1

            # 跳帧处理
            if (frame_idx - 1) % self.frame_skip != 0:
                continue

            processed_frame += 1
            frame = result.orig_img.copy()

            # 获取检测结果
            if result.boxes is None or len(result.boxes) == 0:
                # 无检测，写入空帧
                viz_out.write(frame)
                if processed_frame % (30 // self.frame_skip + 1) == 0:
                    print(f"  Frame {processed_frame} - 无检测")
                continue

            # 有检测
            frames_with_detections += 1
            boxes = result.boxes.xywh.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            frame_detections = []

            # 绘制检测框和标签
            for box, cls_id, conf in zip(boxes, classes, confidences):
                x, y, w, h = box
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)

                class_name = model.names[int(cls_id)]
                class_counts[class_name] += 1

                # 绘制边框
                color = (0, 255, 0)  # 绿色
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                frame_detections.append({"class": class_name, "confidence": float(conf), "bbox": [x1, y1, x2, y2]})

            # 添加帧信息文本
            cv2.putText(
                frame,
                f"Frame: {frame_idx} | Objects: {len(boxes)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # 保存帧信息
            detected_frames_info.append(
                {"frame": frame_idx, "objects_count": len(boxes), "detections": frame_detections}
            )

            # 保存关键帧（有检测的帧）
            keyframe_path = self.yolo_dir / f"keyframe_{frame_idx:04d}.jpg"
            cv2.imwrite(str(keyframe_path), frame)
            keyframes.append({"frame": frame_idx, "path": keyframe_path.name, "objects": len(boxes)})

            # 写入可视化视频
            viz_out.write(frame)

            if processed_frame % (30 // self.frame_skip + 1) == 0:
                print(f"  Frame {processed_frame}/{total_frames // self.frame_skip} - {len(boxes)}个物体")

        cap.release()
        viz_out.release()

        print("✓ YOLO检测完成")
        print(f"  处理帧数: {processed_frame}")
        print(f"  检测到物体的帧数: {frames_with_detections}")
        print(f"  可视化视频: {viz_output.name}")
        print(f"  关键帧数: {len(keyframes)}")

        # 生成YOLO统计报告
        self.generate_yolo_report(class_counts, frames_with_detections, detected_frames_info, keyframes)

        return detected_frames_info, keyframes

    def generate_yolo_report(self, class_counts, frames_with_detections, detected_frames_info, keyframes):
        """生成YOLO检测统计报告."""
        report_path = self.yolo_dir / "yolo_detection_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("YOLO物体检测统计报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入视频: {self.video_path}\n")
            f.write(f"处理帧率跳帧: {self.frame_skip}\n\n")

            f.write("检测统计:\n")
            f.write(f"  - 检测到物体的帧数: {frames_with_detections}\n")
            f.write(f"  - 生成的关键帧数: {len(keyframes)}\n\n")

            f.write("物体类别统计:\n")
            total_objects = sum(class_counts.values())
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = 100 * count / total_objects if total_objects > 0 else 0
                f.write(f"  - {class_name}: {count}个 ({percentage:.1f}%)\n")
            f.write(f"  - 总计: {total_objects}个\n\n")

            f.write("关键帧列表:\n")
            for keyframe in keyframes[:10]:  # 显示前10个
                f.write(f"  Frame {keyframe['frame']}: {keyframe['objects']}个物体 - {keyframe['path']}\n")

            if len(keyframes) > 10:
                f.write(f"  ... 以及其他 {len(keyframes) - 10} 个关键帧\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"✓ YOLO报告已保存: {report_path.name}")

        # 打印报告到控制台
        with open(report_path) as f:
            print("\n" + f.read())

    def analyze_collisions(self, detected_frames_info):
        """分析碰撞（距离计算）."""
        print("\n【步骤4: 碰撞距离分析】")

        # 世界坐标范围
        world_bounds = (-3.75, 3.75, 0.0, 50.0)
        actual_size = (640, 96)  # YOLO实际输入大小

        world_width = world_bounds[1] - world_bounds[0]
        world_height = world_bounds[3] - world_bounds[2]

        collision_events = []
        near_miss_events = []

        print("计算物体间距离...")

        for frame_info in detected_frames_info:
            if frame_info["objects_count"] < 2:
                continue

            detections = frame_info["detections"]
            frame_num = frame_info["frame"]

            # 计算所有物体对的距离
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    bbox1 = detections[i]["bbox"]
                    bbox2 = detections[j]["bbox"]

                    # 获取bbox中心点
                    x1 = (bbox1[0] + bbox1[2]) / 2
                    y1 = (bbox1[1] + bbox1[3]) / 2
                    x2 = (bbox2[0] + bbox2[2]) / 2
                    y2 = (bbox2[1] + bbox2[3]) / 2

                    # 换算到世界坐标
                    x1_world = world_bounds[0] + (x1 / actual_size[0]) * world_width
                    y1_world = world_bounds[2] + (y1 / actual_size[1]) * world_height
                    x2_world = world_bounds[0] + (x2 / actual_size[0]) * world_width
                    y2_world = world_bounds[2] + (y2 / actual_size[1]) * world_height

                    # 计算距离
                    distance = np.sqrt((x2_world - x1_world) ** 2 + (y2_world - y1_world) ** 2)

                    # 分类
                    event = {
                        "frame": frame_num,
                        "class1": detections[i]["class"],
                        "class2": detections[j]["class"],
                        "distance": float(distance),
                        "distance_str": f"{distance:.2f}m",
                    }

                    if distance < 0.5:
                        event["level"] = "COLLISION"  # 碰撞
                        collision_events.append(event)
                        print(
                            f"  ⚠️  COLLISION: Frame {frame_num}, "
                            f"{detections[i]['class']} - {detections[j]['class']}, "
                            f"距离: {distance:.2f}m"
                        )
                    elif distance < 1.5:
                        event["level"] = "NEAR_MISS"  # 接近
                        near_miss_events.append(event)

            if frame_info == detected_frames_info[-1] or detected_frames_info.index(frame_info) % 10 == 0:
                frame_events = len([e for e in collision_events + near_miss_events if e["frame"] == frame_num])
                print(f"  Frame {frame_num}: 已分析 {frame_events}个距离")

        print("✓ 分析完成:")
        print(f"  - 碰撞事件: {len(collision_events)}")
        print(f"  - 接近事件: {len(near_miss_events)}")

        # 保存结果
        self.save_collision_results(collision_events, near_miss_events)

        return collision_events, near_miss_events

    def save_collision_results(self, collision_events, near_miss_events):
        """保存碰撞分析结果."""
        # 保存JSON
        collision_path = self.collision_dir / "collision_events.json"
        with open(collision_path, "w") as f:
            json.dump(collision_events, f, indent=2)

        near_miss_path = self.collision_dir / "near_miss_events.json"
        with open(near_miss_path, "w") as f:
            json.dump(near_miss_events, f, indent=2)

        # 生成报告
        report_path = self.collision_dir / "collision_analysis_report.txt"
        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("碰撞分析报告\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("分析范围: <0.5m (碰撞), <1.5m (接近)\n\n")

            f.write("统计结果:\n")
            f.write(f"  - 碰撞事件: {len(collision_events)}\n")
            f.write(f"  - 接近事件: {len(near_miss_events)}\n\n")

            if collision_events:
                f.write("碰撞事件详情:\n")
                for i, event in enumerate(collision_events, 1):
                    f.write(f"{i}. Frame {event['frame']}\n")
                    f.write(f"   物体: {event['class1']} - {event['class2']}\n")
                    f.write(f"   距离: {event['distance_str']}\n\n")

            if near_miss_events:
                f.write("\n接近事件详情 (前10个):\n")
                for i, event in enumerate(near_miss_events[:10], 1):
                    f.write(
                        f"{i}. Frame {event['frame']}: "
                        f"{event['class1']}-{event['class2']}, "
                        f"距离 {event['distance_str']}\n"
                    )

        print(f"✓ 碰撞报告已保存: {report_path.name}")

    def run(self, conf_threshold=0.45):
        """运行完整pipeline."""
        try:
            # 步骤1: 加载Homography
            self.load_homography()
            self.create_verification_image()
            self.pause_checkpoint("1_homography")

            # 步骤2: 视频透视变换
            self.transform_video()
            self.pause_checkpoint("2_warped_video")

            # 步骤3: YOLO检测
            detected_frames_info, _keyframes = self.detect_and_visualize(conf_threshold)
            self.pause_checkpoint("3_yolo_detection")

            # 步骤4: 碰撞分析
            if detected_frames_info:
                _collision_events, _near_miss_events = self.analyze_collisions(detected_frames_info)

            print(f"\n{'=' * 70}")
            print("✓ Pipeline完成！")
            print(f"{'=' * 70}")
            print(f"结果保存在: {self.run_dir}")
            print("\n文件夹结构:")
            print("  1_homography/          - 标定验证")
            print("  2_warped_video/        - 鸟瞰图视频")
            print("  3_yolo_detection/      - YOLO检测结果")
            print("    ├── yolo_detection_viz_*.mp4       (带框视频)")
            print("    ├── yolo_detection_report.txt      (检测报告)")
            print("    └── keyframe_*.jpg                 (关键帧)")
            print("  4_collision_analysis/  - 碰撞分析")
            print("    ├── collision_events.json          (碰撞事件)")
            print("    ├── near_miss_events.json          (接近事件)")
            print("    └── collision_analysis_report.txt  (分析报告)")

        except Exception as e:
            print(f"❌ Pipeline错误: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="碰撞检测Pipeline - 分阶段调试版本")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--homography", type=str, required=True, help="Homography JSON路径")
    parser.add_argument("--output", type=str, default="../../results", help="结果基础目录")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO置信度阈值")
    parser.add_argument("--frame-skip", type=int, default=1, help="跳帧数（1=不跳，10=每10帧处理1帧，加快速度）")
    parser.add_argument("--debug", action="store_true", default=True, help="启用调试模式（各阶段暂停）")
    parser.add_argument("--no-pause", action="store_true", help="禁用阶段暂停，连续运行")

    args = parser.parse_args()

    pipeline = DebugCollisionDetectionPipeline(
        args.video, args.homography, args.output, frame_skip=args.frame_skip, debug_mode=(not args.no_pause)
    )
    pipeline.run(args.conf)
