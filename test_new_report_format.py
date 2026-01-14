#!/usr/bin/env python3
"""Test the new report format generation."""

import json
from datetime import datetime
from pathlib import Path

# 加载已有的分析数据
analysis_dir = Path("/workspace/ultralytics/results/20260113_204922_yolo_first_method_a/5_collision_analysis")

with open(analysis_dir / "collision_events.json") as f:
    analyzed_events = json.load(f)

proximity_events = analyzed_events

# 统计level
level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
for event in analyzed_events:
    level = event.get("level", 3)
    level_counts[level] += 1

# 生成新格式报告
report_path = analysis_dir / "analysis_report_new.txt"

with open(report_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("YOLO-First 碰撞检测分析报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("输入视频: /workspace/ultralytics/videos/Homograph_Teset_FullScreen.mp4\n")
    f.write("Homography: /workspace/ultralytics/calibration/Homograph_Teset_FullScreen_homography.json\n")
    f.write("结果目录: /workspace/ultralytics/results/20260113_204922_yolo_first_method_a\n\n")

    f.write("处理方式: YOLO-First\n")
    f.write("流程: YOLO检测 → 轨迹(px) → 关键帧 → Homography(关键帧) → 分析\n\n")

    f.write("关键帧统计:\n\n")
    f.write(f"总接近事件: {len(proximity_events)}\n")
    if analyzed_events:
        f.write(f"Level 1 (Collision): {level_counts[1]}\n")
        f.write(f"Level 2 (Near Miss): {level_counts[2]}\n")
        f.write(f"Level 3 (Avoidance): {level_counts[3]}\n\n")

    f.write("前10个高风险事件:\n\n")

    if analyzed_events:
        sorted_events = sorted(analyzed_events, key=lambda e: e.get("level", 3))

        for event in sorted_events[:10]:
            f.write(f"Frame {event['frame']} ({event['time']:.2f}s)\n")
            # 处理不同的物体ID字段名
            obj_ids = event.get("object_ids") or [event.get("track_id_1", -1), event.get("track_id_2", -1)]
            f.write(f"物体ID: {obj_ids}\n")
            f.write(f"风险等级: Level {event['level']} ({event.get('level_name', '?')})\n")
            f.write(f"距离(像素): {event['distance_pixel']:.1f}px\n")

            if "distance_meters" in event:
                f.write(f"距离(米): {event['distance_meters']:.2f}m\n")

            # 从multi_anchor_detailed中提取TTC信息
            if "multi_anchor_detailed" in event:
                ttc = event["multi_anchor_detailed"].get("ttc_seconds")
                if ttc is not None:
                    f.write(f"TTC (时间碰撞): {ttc:.2f}s\n")

                multi_anchor = event["multi_anchor_detailed"]
                closest_parts = multi_anchor.get("closest_parts", {})
                if "description" in closest_parts:
                    f.write(f"碰撞部位: {closest_parts['description']}\n")

                min_dist = multi_anchor.get("min_distance_meters")
                if min_dist is not None:
                    f.write(f"最小距离: {min_dist:.3f}m\n")

            f.write("\n")
    else:
        f.write("未检测到接近事件\n\n")

    f.write("=" * 70 + "\n\n")

    # TTC 分级标准表
    f.write("TTC (时间碰撞) 分级标准:\n\n")
    f.write("┌─────────────────┬──────────────────┬──────────────────┐\n")
    f.write("│ 碰撞类型         │ 严重程度         │ TTC阈值 (秒)     │\n")
    f.write("├─────────────────┼──────────────────┼──────────────────┤\n")
    f.write("│ Rear-end        │ Serious conflict │ 0 – 2.8 s        │\n")
    f.write("│ (追尾)          │ General conflict │ 2.8 – 4.7 s      │\n")
    f.write("├─────────────────┼──────────────────┼──────────────────┤\n")
    f.write("│ Sideswipe       │ Serious conflict │ 0 – 2.3 s        │\n")
    f.write("│ (侧面碰撞)      │ General conflict │ 2.3 – 4.2 s      │\n")
    f.write("└─────────────────┴──────────────────┴──────────────────┘\n\n")

    f.write("=" * 70 + "\n")
    f.write("报告结束\n")

print(f"✓ 新报告已生成: {report_path}")
print("\n预览:")
print("=" * 70)
with open(report_path) as f:
    print(f.read())
