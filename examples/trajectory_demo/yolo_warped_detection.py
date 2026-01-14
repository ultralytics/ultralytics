#!/usr/bin/env python3
"""
在透视变换后的视频上运行YOLO检测和碰撞事件截图.

用法：
    python yolo_warped_detection.py \
        --video videos/warped.mp4 \
        --homography calibration/homography.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

sys.path.insert(0, "/workspace/ultralytics")


def load_homography(json_path):
    """从JSON加载Homography矩阵."""
    with open(json_path) as f:
        data = json.load(f)
    H = np.array(data["homography_matrix"])
    return H, data["pixel_points"], data["world_points"]


def get_contact_points(bbox_yolo, frame_h, frame_w):
    """从YOLO检测框获取3个接触点（前、中、后）."""
    x1, _y1, x2, y2 = bbox_yolo
    x_center = (x1 + x2) / 2

    # 3点：前面、中间、后面（鸟瞰图中Y轴是行驶方向）
    return [
        [x1, y2],  # 后面（底部）
        [x_center, y2],  # 中间底部
        [x2, y2],  # 前面（底部）
    ]


def pixel_to_world(pixel_coord, H):
    """像素坐标转世界坐标."""
    px, py = pixel_coord
    p_homogeneous = np.array([px, py, 1.0])
    world_homogeneous = H @ p_homogeneous
    w = world_homogeneous[2]
    if abs(w) < 1e-10:
        return None
    return [world_homogeneous[0] / w, world_homogeneous[1] / w]


def run_detection(video_path, homography_path, output_base="runs/warped_detection"):
    """在warped视频上运行YOLO检测并截图碰撞事件."""
    # 创建带时间戳的输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_base) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    event_frames_dir = run_dir / "event_frames"
    event_frames_dir.mkdir(exist_ok=True)

    print(f"结果保存到: {run_dir}")

    # 加载Homography矩阵
    H, _pixel_pts, _world_pts = load_homography(homography_path)

    # 初始化YOLO
    model = YOLO("yolo11n.pt")

    # 轨迹历史，用于计算碰撞
    track_history = {}  # object_id -> list of world_contacts

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"视频: {total_frames}帧, {frame_w}x{frame_h}, {fps:.1f}fps")

    # 数据记录
    all_tracks = {}
    collision_events = []
    event_frame_list = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO检测
        results = model(frame, verbose=False)
        detections = results[0]

        frame_time = frame_idx / fps
        frame_data = {"frame": frame_idx, "time": frame_time, "objects": []}

        # 处理检测结果
        for det in detections.boxes:
            obj_id = int(det.id) if det.id is not None else -1
            conf = float(det.conf)
            cls = int(det.cls)

            bbox = det.xyxy[0].cpu().numpy().astype(int)
            _x1, _y1, _x2, _y2 = bbox

            # 获取接触点（3个）
            contact_pts = get_contact_points(bbox, frame_h, frame_w)

            # 转换到世界坐标
            world_contacts = []
            for pt in contact_pts:
                w_pt = pixel_to_world(pt, H)
                if w_pt:
                    world_contacts.append(w_pt)

            obj_data = {
                "id": obj_id,
                "class": int(cls),
                "confidence": float(conf),
                "pixel_bbox": bbox.tolist(),
                "pixel_contacts": contact_pts,
                "world_contacts": world_contacts,
            }

            frame_data["objects"].append(obj_data)

            # 记录轨迹
            if obj_id not in track_history:
                track_history[obj_id] = []
            track_history[obj_id].append({"frame": frame_idx, "world_contacts": world_contacts})

        # 简单碰撞检测：检查所有对象对之间的距离
        objects = frame_data["objects"]
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1 = objects[i]
                obj2 = objects[j]

                if not obj1["world_contacts"] or not obj2["world_contacts"]:
                    continue

                # 计算两个对象的中心点距离
                c1 = np.array(obj1["world_contacts"][1])  # 中间点
                c2 = np.array(obj2["world_contacts"][1])  # 中间点

                distance = float(np.linalg.norm(c1 - c2))

                # 碰撞阈值：0.3米（在warped视图中，这个距离很近）
                if distance < 0.5:
                    event_type = "collision" if distance < 0.2 else "near_miss"

                    collision_events.append(
                        {
                            "frame": frame_idx,
                            "time": frame_time,
                            "object1_id": obj1["id"],
                            "object2_id": obj2["id"],
                            "distance": distance,
                            "type": event_type,
                        }
                    )

                    # 保存碰撞帧
                    event_filename = f"collision_f{frame_idx:04d}_o{obj1['id']}_o{obj2['id']}.jpg"
                    cv2.imwrite(str(event_frames_dir / event_filename), frame)
                    event_frame_list.append(event_filename)

                    print(
                        f"[Frame {frame_idx}] {event_type.upper()}: Object {obj1['id']} ↔ {obj2['id']}, "
                        f"距离: {distance:.3f}m"
                    )

        # 记录所有轨迹
        for obj in frame_data["objects"]:
            if obj["id"] not in all_tracks:
                all_tracks[obj["id"]] = []
            all_tracks[obj["id"]].append(obj_data)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"处理进度: {frame_idx}/{total_frames} ({100 * frame_idx / total_frames:.1f}%)")

    cap.release()

    # 转换numpy类型为Python原生类型（为了JSON序列化）
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    all_tracks = convert_types(all_tracks)
    collision_events = convert_types(collision_events)

    # 保存结果JSON
    tracks_file = run_dir / "tracks.json"
    with open(tracks_file, "w") as f:
        json.dump(all_tracks, f, indent=2)

    collision_file = run_dir / "collision_events.json"
    with open(collision_file, "w") as f:
        json.dump(collision_events, f, indent=2)

    # 生成分析报告
    report = f"""=== YOLO Warped Video Detection Report ===
视频: {video_path}
处理时间: {timestamp}
总帧数: {total_frames}
分辨率: {frame_w}x{frame_h}
帧率: {fps:.1f} fps

检测统计:
- 跟踪的对象数: {len(all_tracks)}
- 碰撞事件数: {len(collision_events)}
- 截图的碰撞帧: {len(event_frame_list)}

碰撞事件列表:
"""

    for i, col in enumerate(collision_events, 1):
        report += f"\n{i}. Frame {col['frame']} ({col['time']:.2f}s)"
        report += f"\n   Object {col['object1_id']} ↔ {col['object2_id']}"
        report += f"\n   距离: {col['distance']:.3f}m"
        report += f"\n   类型: {col['type']}"

    report += "\n\n碰撞帧文件:\n"
    for fname in event_frame_list:
        report += f"  - {fname}\n"

    report_file = run_dir / "analysis_report.txt"
    with open(report_file, "w") as f:
        f.write(report)

    print("\n" + "=" * 50)
    print("检测完成！")
    print(f"结果已保存到: {run_dir}")
    print("  - 轨迹数据: tracks.json")
    print("  - 碰撞事件: collision_events.json")
    print(f"  - 事件帧: event_frames/ ({len(event_frame_list)} 张)")
    print("  - 分析报告: analysis_report.txt")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="在warped视频上运行YOLO检测")
    parser.add_argument("--video", type=str, required=True, help="输入的warped视频路径")
    parser.add_argument("--homography", type=str, required=True, help="Homography矩阵JSON路径")
    parser.add_argument("--output_base", type=str, default="../../runs/trajectory_demo_warped", help="输出基础目录")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}")
        sys.exit(1)

    homography_path = Path(args.homography)
    if not homography_path.exists():
        print(f"错误: Homography文件不存在: {homography_path}")
        sys.exit(1)

    run_detection(video_path, homography_path, args.output_base)


if __name__ == "__main__":
    main()
