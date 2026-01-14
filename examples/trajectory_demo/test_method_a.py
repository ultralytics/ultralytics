#!/usr/bin/env python3
"""
test_method_a_simple.py.

Method A 简单测试脚本
在执行前逐步验证每个输出
"""

import json
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))
from collision_detection_pipeline_yolo_first_method_a import YOLOFirstPipelineA


def main():
    print("\n" + "=" * 70)
    print("YOLO-First Method A - 简单测试")
    print("=" * 70)

    # 配置
    video_path = "../../videos/Homograph_Teset_FullScreen.mp4"
    homography_path = "../../calibration/Homograph_Teset_FullScreen_homography.json"
    output_base = "../../results"

    # 检查输入文件
    print("\n【检查输入文件】")
    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return False
    print(f"✓ 视频文件: {Path(video_path).name}")

    if not Path(homography_path).exists():
        print(f"❌ Homography文件不存在: {homography_path}")
        return False
    print(f"✓ Homography文件: {Path(homography_path).name}")

    # 创建管道
    print("\n【初始化管道】")
    pipeline = YOLOFirstPipelineA(video_path=video_path, homography_path=homography_path, output_base=output_base)

    # 运行管道
    print("\n【运行管道】")
    pipeline.run(conf_threshold=0.45)

    # 验证每个步骤的输出
    print("\n" + "=" * 70)
    print("【验证输出文件】")
    print("=" * 70)

    output_dir = pipeline.run_dir

    # 检查每个步骤
    checks = [
        ("Step 1", "1_yolo_detection/detections_pixel.json"),
        ("Step 2", "2_trajectories/tracks.json"),
        ("Step 3", "3_key_frames/proximity_events.json"),
        ("Step 4", "4_homography_transform/transformed_key_frames.json"),
        ("Step 5", "5_collision_analysis/collision_events.json"),
    ]

    all_passed = True

    for step_name, file_path in checks:
        full_path = output_dir / file_path
        print(f"\n{step_name}: {file_path}")

        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✓ 文件存在 ({size} bytes)")

            # 尝试读取JSON以验证格式
            try:
                with open(full_path) as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    print("  ✓ 格式: JSON对象")
                    print(f"    键数: {len(data)}")
                elif isinstance(data, list):
                    print("  ✓ 格式: JSON数组")
                    print(f"    元素数: {len(data)}")
                    if len(data) > 0:
                        print(f"    首个元素类型: {type(data[0]).__name__}")

            except json.JSONDecodeError as e:
                print(f"  ❌ JSON格式错误: {e}")
                all_passed = False
        else:
            print("  ❌ 文件不存在")
            all_passed = False

    # 检查报告
    print("\n报告: 5_collision_analysis/analysis_report.txt")
    report_path = output_dir / "5_collision_analysis/analysis_report.txt"
    if report_path.exists():
        print("  ✓ 文件存在")
        with open(report_path) as f:
            lines = f.readlines()
        print(f"  ✓ 行数: {len(lines)}")
    else:
        print("  ❌ 文件不存在")
        all_passed = False

    # 总结
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ 所有步骤验证通过！")
        print(f"\n结果目录: {output_dir}")
        print("\n可以按照以下步骤继续验证：")
        print("1. 查看Step 1的YOLO检测数量")
        print("2. 查看Step 2的轨迹条数和长度分布")
        print("3. 查看Step 3的关键帧数量")
        print("4. 查看Step 4的Homography变换结果")
        print("5. 查看Step 5的事件分级统计")
    else:
        print("❌ 部分步骤验证失败")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
