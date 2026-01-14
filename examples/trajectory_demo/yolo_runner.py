"""
yolo_runner.py.

集成的对象追踪和碰撞分析完整流程：
1. 加载 YOLO 模型
2. 逐帧处理视频，使用 model.track() 保持物体的追踪 ID
3. 使用 detection_adapter 解析检测结果
4. 用 ObjectStateManager 更新物体轨迹信息
5. 检测碰撞接近事件（近距离事件）
6. 自动生成分析报告

输出文件包括：
  - tracks.json: 所有物体的轨迹数据（位置、速度、类别等）
  - near_misses.json: 所有碰撞接近事件(距离、TTC、时间戳等)
  - analysis_report.txt: 统计分析和关键碰撞对

使用示例：
python examples/trajectory_demo/yolo_runner.py --source path/to/video.mp4 --weights yolo11n.pt --output runs/trajectory_demo
or
python examples/trajectory_demo/yolo_runner.py \

  --source videos/NewYorkSample.mp4 \

  --conf 0.45 \

  --output runs/trajectory_demo
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime

import cv2

# Ensure ability to import modules from same directory
sys.path.append(os.path.dirname(__file__))


import coord_transform
import detection_adapter
from object_state_manager import ObjectStateManager

from ultralytics import YOLO


def run(
    source: str,
    weights: str,
    output: str,
    auto_subdir: bool = True,
    conf_threshold: float = 0.5,
    use_segmentation: bool = False,
    homography_path: str | None = None,
):
    # 生成输出目录（包含视频名和时间戳，防止覆盖）
    # 例如: runs/trajectory_demo/NewYorkSample_20251215_010909/
    if auto_subdir:
        video_name = os.path.splitext(os.path.basename(source))[0]  # 提取视频名（无扩展名）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 添加时间戳
        output = os.path.join(output, f"{video_name}_{timestamp}")

    os.makedirs(output, exist_ok=True)

    # 根据use_segmentation参数选择模型
    if use_segmentation:
        # 使用分割模型（更精确的边界）
        model = YOLO(weights.replace(".pt", "-seg.pt"))  # 加载分割版本
        print("Using YOLO Segmentation model (more precise boundaries)")
    else:
        # 使用检测模型（原来的）
        model = YOLO(weights)
        print("Using YOLO Detection model (standard bounding boxes)")

    # 加载Homography矩阵（如果提供）
    H = None
    if homography_path:
        H = coord_transform.load_homography(homography_path)
        if H is not None:
            print("✓ Homography矩阵已加载，距离将转换为实际世界坐标（米）")
        else:
            print("❌ 无法加载Homography矩阵，将使用像素坐标")
    else:
        print("ℹ 未提供Homography矩阵，距离将用像素表示（可选：使用--homography参数指定）")

    osm = ObjectStateManager(H=H)  # 物体状态管理器（维护轨迹和速度，支持世界坐标）
    all_near_misses = []  # 收集所有接近碰撞事件

    # 从视频文件获取帧率（FPS），用于将帧号转换成秒数
    # 这样报告中的时间会更易理解
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
    cap.release()
    if fps <= 0:
        fps = 30.0  # 如果无法读取，默认使用 30 FPS

    print(f"Processing video: {source}")
    print(f"Output directory: {output}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Video FPS: {fps:.2f}")
    print("Starting frame-by-frame processing...")

    frame_idx = 0
    seen_pairs = set()  # 跟踪已记录的物体对，防止重复（(id1, id2, frame)）

    for result in model.track(source=source, stream=True, persist=True, conf=conf_threshold):
        # 使用帧索引作为时间戳，便于后续时间转换
        timestamp = frame_idx
        dets = detection_adapter.parse_result(result, timestamp)  # 解析检测结果

        # 更新物体状态管理器（记录位置、计算速度、更新轨迹）
        osm.update(dets, timestamp)

        # 检测当前帧中所有的碰撞接近事件
        # 根据是否使用世界坐标调整距离阈值
        if H is not None:
            # 世界坐标模式：用米（约3.5米一条车道，阈值设为10米约2-3条车道）
            distance_threshold = 10.0
        else:
            # 像素模式：150像素
            distance_threshold = 150.0

        near_misses = osm.detect_near_miss(distance_threshold=distance_threshold, ttc_threshold=3.0)

        # 去重处理：同一帧中同一物体对只保留第一次记录
        # 这防止了由于多次轨迹更新导致的重复事件
        for nm in near_misses:
            pair_key = (min(nm["id1"], nm["id2"]), max(nm["id1"], nm["id2"]), int(nm["timestamp"]))
            if pair_key not in seen_pairs:
                all_near_misses.append(nm)
                seen_pairs.add(pair_key)

        # 打印进度信息
        print(f"Frame {frame_idx}: {len(dets)} objects detected, {len(near_misses)} proximity events")
        frame_idx += 1

    # 保存轨迹数据（所有物体的位置序列）
    out_path = os.path.join(output, "tracks.json")
    osm.save_tracks(out_path)
    print(f"Trajectory data saved: {out_path}")

    # 保存碰撞接近事件（去重后的近距离对）
    nm_path = os.path.join(output, "near_misses.json")
    with open(nm_path, "w", encoding="utf-8") as f:
        json.dump(all_near_misses, f, indent=2, ensure_ascii=False)
    print(f"Collision proximity events saved: {len(all_near_misses)} events to {nm_path}")

    # 自动生成分析报告（包括统计数据和最危险碰撞对）
    print("\n" + "=" * 60)
    print("Generating Analysis Report...")
    print("=" * 60)
    distance_unit = "meters" if H is not None else "pixels"
    generate_analysis_report(out_path, nm_path, output, fps, distance_unit=distance_unit)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"All results saved to: {output}")
    print("Files generated:")
    print("  - tracks.json")
    print("  - near_misses.json")
    print("  - analysis_report.txt")


def generate_analysis_report(
    tracks_path: str, near_misses_path: str, output_dir: str, fps: float = 30.0, distance_unit: str = "pixels"
) -> None:
    """生成综合分析报告。.

    功能说明：
    1. 加载轨迹和碰撞事件数据
    2. 统计检测到的物体类别
    3. 分析碰撞接近事件(包括距离、TTC 等指标）
    4. 识别最危险的物体对(TTC < 3秒)
    5. 生成文本报告文件

    参数：
        tracks_path: tracks.json 文件路径
        near_misses_path: near_misses.json 文件路径
        output_dir: 输出目录（报告保存位置）
        fps: 视频帧率（用于将帧号转换成秒数）
        distance_unit: 距离单位 ("pixels" 或 "meters")
    """
    # 加载已保存的数据
    with open(tracks_path) as f:
        tracks = json.load(f)
    with open(near_misses_path) as f:
        near_misses = json.load(f)

    # 按类别统计物体数量
    class_counts = defaultdict(int)
    for samples in tracks.values():
        if samples:
            cls = samples[0].get("cls")  # 获取物体类别
            if cls is not None:
                class_counts[cls] += 1

    print("\nDetected object classes:")
    print("-" * 40)
    # YOLO 的 COCO 数据集中的类别映射（第 0, 2, 5, 7 类是常见的交通参与者）
    class_names = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
    for cls_id, count in sorted(class_counts.items()):
        cls_name = class_names.get(cls_id, f"Unknown({cls_id})")
        print(f"  {cls_name}: {count} objects")

    # 分析碰撞接近事件（报告的重点）
    print("\nCollision Proximity Analysis:")
    print("-" * 40)
    print(f"Total proximity events: {len(near_misses)}")

    # 过滤出高危事件（TTC < 3秒的事件）
    collision_risks = [nm for nm in near_misses if nm.get("is_collision_risk", False)]
    print(f"High-risk collision events: {len(collision_risks)} events")

    # 显示最危险的物体对（按 TTC 排序，TTC 越小越危险）
    print("\nMost Critical Object Pairs (TTC < 3 seconds):")
    if collision_risks:
        sorted_risks = sorted(collision_risks, key=lambda x: x["ttc"] if x["ttc"] else float("inf"))
        for i, nm in enumerate(sorted_risks[:10], 1):
            ttc_val = nm["ttc"] if nm["ttc"] else "N/A"
            timestamp = nm.get("timestamp", "N/A")
            frame_num = int(timestamp) if timestamp != "N/A" else "N/A"
            time_seconds = frame_num / fps if frame_num != "N/A" else "N/A"  # 将帧号转换成秒
            time_str = f"{time_seconds:.2f}s" if time_seconds != "N/A" else "N/A"
            dist_str = (
                f"{nm['distance']:.2f}{distance_unit[0].upper()}"
                if distance_unit == "pixels"
                else f"{nm['distance']:.2f}m"
            )
            print(
                f"  {i}. Object {nm['id1']} - Object {nm['id2']}: "
                f"Distance={dist_str}, TTC={ttc_val}s, Time={time_str} (Frame {frame_num})"
            )
    else:
        print("  No high-risk collision events detected.")

    # 距离统计
    distances = [nm["distance"] for nm in near_misses if nm["distance"] is not None]
    if distances:
        unit_str = "pixels" if distance_unit == "pixels" else "meters"
        print(f"\nDistance Statistics ({unit_str}):")
        print(f"  Average: {statistics.mean(distances):.2f}")
        print(f"  Minimum: {min(distances):.2f}")
        print(f"  Maximum: {max(distances):.2f}")

    # TTC 统计（Time-to-Collision，预计碰撞时间，秒）
    # TTC 越小，表示碰撞越快发生，危险度越高
    ttcs = [nm["ttc"] for nm in near_misses if nm["ttc"] is not None]
    if ttcs:
        print("\nTime-to-Collision (TTC) Statistics (seconds):")
        print(f"  Average: {statistics.mean(ttcs):.2f}")
        print(f"  Minimum (Most critical): {min(ttcs):.2f}")
        print(f"  Maximum: {max(ttcs):.2f}")

    # 轨迹数据分析（放在最后）
    print("\n\nTrajectory Data Analysis:")
    print("-" * 40)
    print(f"Total tracked objects: {len(tracks)}")

    # 计算轨迹长度统计（有些物体只出现一帧，有些贯穿整个视频）
    track_lengths = [len(samples) for samples in tracks.values()]
    print(f"  Average trajectory length: {statistics.mean(track_lengths):.1f} frames")
    print(f"  Maximum trajectory length: {max(track_lengths)} frames")
    print(f"  Minimum trajectory length: {min(track_lengths)} frames")

    # 生成报告文件（txt 格式，便于查看）
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("YOLO Object Tracking and Collision Risk Analysis Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("DETECTED OBJECT CLASSES\n")
        f.write("-" * 40 + "\n")
        for cls_id, count in sorted(class_counts.items()):
            cls_name = class_names.get(cls_id, f"Unknown({cls_id})")
            f.write(f"{cls_name}: {count} objects\n")

        f.write("\n\nCOLLISION PROXIMITY ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total proximity events: {len(near_misses)}\n")
        f.write(f"High-risk events (TTC < 3s): {len(collision_risks)}\n\n")

        if distances:
            f.write("Distance Statistics (pixels):\n")
            f.write(f"  Average: {statistics.mean(distances):.2f}\n")
            f.write(f"  Min: {min(distances):.2f}, Max: {max(distances):.2f}\n\n")

        if ttcs:
            f.write("Time-to-Collision (TTC) Statistics (seconds):\n")
            f.write(f"  Average: {statistics.mean(ttcs):.2f}\n")
            f.write(f"  Min: {min(ttcs):.2f}, Max: {max(ttcs):.2f}\n\n")

        f.write("CRITICAL OBJECT PAIRS (TTC < 3 seconds)\n")
        f.write("-" * 40 + "\n")
        if collision_risks:
            sorted_risks = sorted(collision_risks, key=lambda x: x["ttc"] if x["ttc"] else float("inf"))
            for i, nm in enumerate(sorted_risks[:10], 1):
                ttc_val = nm["ttc"] if nm["ttc"] else "N/A"
                timestamp = nm.get("timestamp", "N/A")
                frame_num = int(timestamp) if timestamp != "N/A" else "N/A"
                time_seconds = frame_num / fps if frame_num != "N/A" else "N/A"
                time_str = f"{time_seconds:.2f}s" if time_seconds != "N/A" else "N/A"
                f.write(
                    f"{i}. Object {nm['id1']} - Object {nm['id2']}: "
                    f"Distance={nm['distance']:.2f}px, TTC={ttc_val}s, Time={time_str} (Frame {frame_num})\n"
                )
        else:
            f.write("No high-risk collision events detected.\n")

        f.write("\n\nTRAJECTORY DATA ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total tracked objects: {len(tracks)}\n")
        f.write(f"Average trajectory length: {statistics.mean(track_lengths):.1f} frames\n")
        f.write(f"Trajectory range: {min(track_lengths)} - {max(track_lengths)} frames\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("End of Report\n")

    print(f"\nAnalysis report saved: {report_path}")


if __name__ == "__main__":
    # 命令行参数解析
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, required=True, help="输入视频路径或摄像头设备")
    p.add_argument("--weights", type=str, default="yolo11n.pt", help="YOLO 模型权重文件")
    p.add_argument("--output", type=str, default="runs/trajectory_demo", help="输出结果目录")
    p.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="置信度阈值 (0-1)，默认 0.5。较低的值会检测更多物体（包括小物体和人），但可能有噪声",
    )
    p.add_argument("--no-auto-subdir", action="store_true", help="不根据视频名称和时间戳创建子文件夹（覆盖模式）")
    p.add_argument("--segmentation", action="store_true", help="使用YOLO分割模型（更精确的边界）")
    p.add_argument("--homography", type=str, default=None, help="Homography矩阵JSON文件路径（用于像素->世界坐标变换）")
    args = p.parse_args()

    # 运行主程序
    run(
        args.source,
        args.weights,
        args.output,
        auto_subdir=not args.no_auto_subdir,
        conf_threshold=args.conf,
        use_segmentation=args.segmentation,
        homography_path=args.homography,
    )
