"""
yolo_runner_with_event_capture.py.

增强版YOLO追踪：
- 自动截图有碰撞接近事件(near_miss)的帧
- 将截图保存到结果文件夹
- 每次运行自动创建新的时间戳文件夹

使用示例：
python yolo_runner_with_event_capture.py \
  --source ../../videos/Homograph_Teset_FullScreen_warped.mp4 \
  --homography ../../calibration/Homograph_Teset_FullScreen_homography.json \
  --output ../../runs/trajectory_warped/ \
  --conf 0.45 \
  --distance-threshold 5.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import cv2

sys.path.append(os.path.dirname(__file__))


import coord_transform
import detection_adapter
from object_state_manager import ObjectStateManager

from ultralytics import YOLO


def run(
    source: str,
    weights: str,
    output: str,
    homography_path: str | None = None,
    conf_threshold: float = 0.5,
    distance_threshold: float = 5.0,
    auto_subdir: bool = True,
):
    """运行YOLO追踪和碰撞检测，自动截图事件帧.

    参数：
    - distance_threshold: 触发事件截图的距离阈值（米）
    """
    # 生成输出目录（带时间戳）
    if auto_subdir:
        video_name = os.path.splitext(os.path.basename(source))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = os.path.join(output, f"{video_name}_{timestamp}")

    os.makedirs(output, exist_ok=True)

    # 创建事件截图文件夹
    event_frames_dir = os.path.join(output, "event_frames")
    os.makedirs(event_frames_dir, exist_ok=True)

    print("\n【输出文件夹】")
    print(f"主结果: {output}")
    print(f"事件截图: {event_frames_dir}")

    # 加载模型
    model = YOLO(weights)
    print(f"✓ YOLO模型已加载: {weights}")

    # 加载Homography矩阵
    H = None
    if homography_path:
        H = coord_transform.load_homography(homography_path)
        if H is not None:
            print("✓ Homography矩阵已加载，距离单位为米")
        else:
            print("⚠ 无法加载Homography矩阵，距离单位为像素")

    osm = ObjectStateManager(H=H)
    all_near_misses = []
    all_event_frames = []  # 记录有事件的帧

    # 获取视频信息
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
    cap.release()
    if fps <= 0:
        fps = 30.0

    print("\n【视频信息】")
    print(f"视频: {source}")
    print(f"FPS: {fps:.2f}")
    print(f"距离阈值: {distance_threshold} {'米' if H is not None else '像素'}")
    print(f"置信度阈值: {conf_threshold}")
    print("\n开始处理...")

    frame_idx = 0
    seen_pairs = set()

    for result in model.track(source=source, stream=True, persist=True, conf=conf_threshold):
        frame = result.orig_img

        # 检测结果
        detections = detection_adapter.parse_yolo_results(result)
        osm.update(detections, frame_idx)

        # 检测碰撞接近事件
        frame_events = []
        if osm.size() >= 2:
            for id1, id2 in osm.get_all_pairs():
                if (id1, id2) in seen_pairs or (id2, id1) in seen_pairs:
                    continue

                dist, ttc = osm.compute_proximity(id1, id2, frame_idx)

                if dist is not None and dist <= distance_threshold:
                    event = {
                        "frame": frame_idx,
                        "time": frame_idx / fps,
                        "id1": id1,
                        "id2": id2,
                        "distance": float(dist),
                        "ttc": float(ttc) if ttc is not None else None,
                        "unit": "米" if H is not None else "像素",
                    }
                    all_near_misses.append(event)
                    frame_events.append(event)
                    seen_pairs.add((id1, id2))

        # 如果该帧有事件，截图保存
        if frame_events:
            all_event_frames.append(
                {
                    "frame_idx": frame_idx,
                    "time": frame_idx / fps,
                    "num_events": len(frame_events),
                    "events": frame_events,
                }
            )

            # 保存截图
            frame_filename = f"frame_{frame_idx:05d}_t{frame_idx / fps:.2f}s.jpg"
            frame_path = os.path.join(event_frames_dir, frame_filename)

            # 在框架上绘制检测结果和事件信息
            frame_marked = frame.copy()

            # 绘制所有检测到的物体
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                obj_id = det["track_id"]

                cv2.rectangle(frame_marked, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame_marked, f"ID{obj_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )

            # 标注事件
            text_y = 30
            for event in frame_events:
                text = f"⚠ ID{event['id1']}-ID{event['id2']}: {event['distance']:.2f} {event['unit']}"
                cv2.putText(frame_marked, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                text_y += 30

            cv2.imwrite(frame_path, frame_marked)
            print(
                f"帧 {frame_idx:4d} (t={frame_idx / fps:6.2f}s): 检测到 {len(frame_events)} 个事件 → {frame_filename}"
            )

        frame_idx += 1

    print(f"\n✓ 视频处理完成！总共处理 {frame_idx} 帧")

    # 保存结果JSON
    print("\n【保存结果】")

    # 保存近距离事件
    near_misses_path = os.path.join(output, "near_misses.json")
    with open(near_misses_path, "w") as f:
        json.dump(all_near_misses, f, indent=2)
    print(f"✓ 碰撞接近事件: {near_misses_path} ({len(all_near_misses)} 事件)")

    # 保存事件帧索引
    event_frames_path = os.path.join(output, "event_frames.json")
    with open(event_frames_path, "w") as f:
        json.dump(all_event_frames, f, indent=2)
    print(f"✓ 事件帧索引: {event_frames_path} ({len(all_event_frames)} 帧有事件)")

    # 生成事件汇总报告
    report_path = os.path.join(output, "event_summary.txt")
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("碰撞事件汇总报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"视频: {source}\n")
        f.write(f"总帧数: {frame_idx}\n")
        f.write(f"有事件的帧数: {len(all_event_frames)}\n")
        f.write(f"总事件数: {len(all_near_misses)}\n")
        f.write(f"距离阈值: {distance_threshold} {'米' if H is not None else '像素'}\n\n")

        if all_near_misses:
            f.write("事件列表:\n")
            f.write("-" * 70 + "\n")
            for i, event in enumerate(all_near_misses, 1):
                f.write(f"{i}. 帧{event['frame']} (时间{event['time']:.2f}s)\n")
                f.write(f"   对象对: ID{event['id1']} - ID{event['id2']}\n")
                f.write(f"   距离: {event['distance']:.2f} {event['unit']}\n")
                if event["ttc"] is not None:
                    f.write(f"   TTC: {event['ttc']:.2f}s\n")
                f.write("\n")
        else:
            f.write("未检测到任何碰撞接近事件。\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"✓ 事件汇总报告: {report_path}")

    print("\n" + "=" * 70)
    print(f"✓ 完成！结果保存到: {output}")
    print("  - event_frames/: 包含所有有事件的帧截图")
    print("  - near_misses.json: 详细的事件数据")
    print("  - event_frames.json: 事件帧索引")
    print("  - event_summary.txt: 事件汇总报告")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO追踪+事件截图")
    parser.add_argument("--source", type=str, required=True, help="输入视频路径")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="YOLO模型权重")
    parser.add_argument("--output", type=str, default="../../runs/trajectory_warped/", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.45, help="置信度阈值")
    parser.add_argument("--homography", type=str, default=None, help="Homography矩阵JSON文件")
    parser.add_argument("--distance-threshold", type=float, default=5.0, help="触发事件截图的距离阈值（米或像素）")

    args = parser.parse_args()

    run(
        args.source,
        args.weights,
        args.output,
        homography_path=args.homography,
        conf_threshold=args.conf,
        distance_threshold=args.distance_threshold,
    )
