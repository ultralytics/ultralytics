#!/usr/bin/env python3
"""
test_yolo_first_method_c.py.

测试 YOLO-First 方案C (Homography优先) 的脚本
演示完整的五步管道执行
"""

import json
import os
import sys
from pathlib import Path

# 添加路径
sys.path.append(os.path.dirname(__file__))
from collision_detection_pipeline_yolo_first_method_c import YOLOFirstPipelineC


def test_pipeline():
    """测试方案C管道."""
    # 配置路径
    video_path = "../../videos/Homograph_Teset_FullScreen.mp4"
    homography_path = "../../calibration/Homograph_Teset_FullScreen_homography.json"
    output_base = "../../results"

    # 检查输入文件
    print("\n" + "=" * 70)
    print("YOLO-First 方案C 测试脚本")
    print("=" * 70)

    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return False

    if not Path(homography_path).exists():
        print(f"❌ Homography 文件不存在: {homography_path}")
        return False

    print("\n✓ 输入文件验证通过")
    print(f"  视频: {Path(video_path).name}")
    print(f"  Homography: {Path(homography_path).name}")

    # 创建管道实例
    print("\n【初始化管道】")
    pipeline = YOLOFirstPipelineC(video_path=video_path, homography_path=homography_path, output_base=output_base)

    # 运行管道
    print("\n【运行完整管道】")
    pipeline.run(conf_threshold=0.45)

    # 验证输出
    print("\n【验证输出文件】")
    output_dir = pipeline.run_dir

    expected_files = [
        "1_raw_detections/detections_pixel.json",
        "1_raw_detections/detection_stats.json",
        "2_homography_transform/homography.json",
        "2_homography_transform/detections_world.json",
        "3_trajectories/tracks_world.json",
        "3_trajectories/track_stats.json",
        "4_key_frames/proximity_events.json",
        "5_collision_analysis/collision_events.json",
        "5_collision_analysis/analysis_report.txt",
    ]

    all_exist = True
    for file_path in expected_files:
        full_path = output_dir / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✓ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} (缺失)")
            all_exist = False

    if all_exist:
        print("\n✓ 所有输出文件已生成")
    else:
        print("\n❌ 部分输出文件缺失")
        return False

    # 分析结果
    print("\n【分析结果摘要】")

    try:
        # 读取分析报告
        report_path = output_dir / "5_collision_analysis/analysis_report.txt"
        with open(report_path) as f:
            report_content = f.read()

        # 提取关键信息
        print("\n报告内容预览:")
        print("-" * 70)
        print(report_content[:800] + "\n...")

        # 读取事件统计
        events_path = output_dir / "5_collision_analysis/collision_events.json"
        with open(events_path) as f:
            events = json.load(f)

        print("\n事件统计:")
        print(f"  总事件数: {len(events)}")

        if events:
            levels = {1: 0, 2: 0, 3: 0}
            for event in events:
                if "level" in event:
                    levels[event["level"]] += 1

            print(f"  - L1 (Collision): {levels[1]}")
            print(f"  - L2 (Near Miss): {levels[2]}")
            print(f"  - L3 (Avoidance): {levels[3]}")

            # 显示最高风险事件
            if levels[1] > 0 or levels[2] > 0:
                high_risk = [e for e in events if e.get("level") in [1, 2]]
                print("\n最高风险事件 (前3个):")
                for i, event in enumerate(high_risk[:3], 1):
                    dist = event.get("distance_meters", 0)
                    print(
                        f"  {i}. Frame {event['frame']} - "
                        + f"距离 {dist:.2f}m - Level {event['level']} ({event.get('level_name', '?')})"
                    )

        # 读取轨迹统计
        track_stats_path = output_dir / "3_trajectories/track_stats.json"
        with open(track_stats_path) as f:
            track_stats = json.load(f)

        print("\n轨迹统计:")
        print(f"  总轨迹数: {track_stats['total_tracks']}")
        print(f"  坐标系: {track_stats['coordinate_system']}")
        print(f"  速度单位: {track_stats['velocity_unit']}")

    except Exception as e:
        print(f"❌ 分析结果时出错: {e}")
        return False

    print("\n" + "=" * 70)
    print(f"✓ 测试完成！结果保存在: {output_dir}")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
