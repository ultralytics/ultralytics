#!/usr/bin/env python3
"""
visualize_yolo_detections.py.

为warped视频生成带YOLO检测框的可视化版本，直观看到检测结果和距离信息
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

sys.path.append("examples/trajectory_demo")
import coord_transform


def visualize_detections(warped_video_path, homography_path, output_path=None, conf=0.45):
    """为warped视频生成带检测框和距离信息的可视化版本.

    参数：
    - warped_video_path: warped视频路径
    - homography_path: homography矩阵路径
    - output_path: 输出视频路径
    - conf: YOLO置信度阈值
    """
    if output_path is None:
        output_path = Path(warped_video_path).parent / "yolo_visualization.mp4"

    print("加载模型...")
    model = YOLO("yolo11n.pt")

    print("加载Homography...")
    H = coord_transform.load_homography(homography_path)

    print(f"打开视频: {warped_video_path}")
    cap = cv2.VideoCapture(warped_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}×{height}, {fps}FPS, {total_frames}帧")
    print(f"输出到: {output_path}")

    # 创建VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    detections_summary = []

    for result in model.track(source=warped_video_path, stream=True, persist=True, conf=conf):
        frame_count += 1
        frame = result.orig_img.copy()

        # 获取检测信息
        boxes = result.boxes.xywh.cpu().numpy() if result.boxes else []
        ids = result.boxes.id if result.boxes and result.boxes.id is not None else []
        classes = result.boxes.cls.cpu().numpy() if result.boxes else []

        # 绘制检测框和信息
        if len(boxes) > 0:
            summary = {"frame": frame_count, "objects_count": len(boxes), "distances": []}

            # 绘制所有物体
            for i, (box, obj_id, cls_id) in enumerate(zip(boxes, ids, classes)):
                x, y, w, h = box
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)

                # 绘制边框
                color = (0, 255, 0) if len(boxes) >= 2 else (0, 165, 255)  # 多物体绿色，单物体橙色
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 绘制物体ID和类别
                class_name = model.names[int(cls_id)]
                label = f"ID:{int(obj_id)} {class_name}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 计算并绘制距离信息
            if len(boxes) >= 2:
                cv2.putText(frame, "Multiple Objects Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        x1, y1 = boxes[i][:2]
                        x2, y2 = boxes[j][:2]

                        # 计算世界坐标距离
                        try:
                            p1_world = coord_transform.transform_point((x1, y1), H)
                            p2_world = coord_transform.transform_point((x2, y2), H)
                            distance = np.sqrt((p1_world[0] - p2_world[0]) ** 2 + (p1_world[1] - p2_world[1]) ** 2)
                            distance_str = f"{distance:.2f}m"
                        except:
                            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            distance_str = f"{distance:.1f}px"

                        summary["distances"].append({"pair": (int(ids[i]), int(ids[j])), "distance": distance_str})

                        # 绘制连接线和距离标签
                        ix1, iy1 = int(boxes[i][0]), int(boxes[i][1])
                        ix2, iy2 = int(boxes[j][0]), int(boxes[j][1])

                        cv2.line(frame, (ix1, iy1), (ix2, iy2), (255, 0, 0), 1)

                        # 在线的中点绘制距离
                        mx, my = (ix1 + ix2) // 2, (iy1 + iy2) // 2
                        cv2.putText(frame, distance_str, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        # 标记碰撞（距离<0.5m）
                        if "<" in distance_str and float(distance_str.replace("m", "")) < 0.5:
                            cv2.putText(frame, "COLLISION!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(
                    frame,
                    f"Only {len(boxes)} Object - No Collision Possible",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

            # 绘制帧信息
            cv2.putText(
                frame,
                f"Frame: {frame_count}/{total_frames}",
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            detections_summary.append(summary)

        # 写入视频
        out.write(frame)

        if frame_count % 10 == 0:
            print(f"  {frame_count}/{total_frames} ({100 * frame_count / total_frames:.1f}%)")

    cap.release()
    out.release()

    print(f"✓ 可视化视频已保存: {output_path}")

    # 保存检测统计
    stats_path = Path(output_path).parent / "visualization_stats.json"
    with open(stats_path, "w") as f:
        json.dump(detections_summary, f, indent=2)

    print(f"✓ 检测统计已保存: {stats_path}")

    # 打印统计信息
    frames_with_multiple = sum(1 for s in detections_summary if s["objects_count"] >= 2)
    print("\n统计:")
    print(f"  检测到物体的帧数: {len(detections_summary)}")
    print(f"  多物体帧数: {frames_with_multiple}")
    print(f"  最多物体数: {max(s['objects_count'] for s in detections_summary) if detections_summary else 0}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO检测可视化")
    parser.add_argument("--warped-video", type=str, required=True, help="warped视频路径")
    parser.add_argument("--homography", type=str, required=True, help="Homography JSON路径")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO置信度阈值")

    args = parser.parse_args()

    visualize_detections(args.warped_video, args.homography, args.output, args.conf)
