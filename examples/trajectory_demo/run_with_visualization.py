"""
run_with_visualization.py.

运行YOLO追踪和碰撞检测，并生成可视化视频

这是yolo_runner.py的增强版本，添加了视频输出功能

使用示例：
python run_with_visualization.py \
  --source ../../videos/Homograph_Teset_FullScreen.mp4 \
  --homography ../../calibration/Homograph_Teset_FullScreen_homography.json \
  --output ../../runs/trajectory_demo/ \
  --visualize
"""

import argparse
import os
import sys

# Import from yolo_runner
sys.path.append(os.path.dirname(__file__))
from yolo_runner import run


def run_with_viz(source, weights, output, homography_path, visualize=True, **kwargs):
    """运行yolo_runner并生成可视化."""
    # 首先运行标准分析
    print("\n" + "=" * 60)
    print("第一步：运行YOLO追踪和碰撞检测分析...")
    print("=" * 60 + "\n")

    run(source, weights, output, homography_path=homography_path, **kwargs)

    if visualize:
        print("\n" + "=" * 60)
        print("第二步：生成可视化视频...")
        print("=" * 60 + "\n")

        # 查找输出目录
        if not os.path.exists(output):
            print(f"❌ 输出目录不存在: {output}")
            return

        # 找到最近的子目录（带时间戳的）
        subdirs = [d for d in os.listdir(output) if os.path.isdir(os.path.join(output, d))]
        if subdirs:
            latest_dir = sorted(subdirs)[-1]
            result_dir = os.path.join(output, latest_dir)
        else:
            result_dir = output

        print(f"✓ 结果目录: {result_dir}")

        # 列出生成的文件
        json_files = [f for f in os.listdir(result_dir) if f.endswith(".json")]
        print("✓ 生成的文件:")
        for f in json_files:
            size = os.path.getsize(os.path.join(result_dir, f)) / 1024
            print(f"  - {f} ({size:.1f} KB)")

        # 查找分析报告
        report_file = os.path.join(result_dir, "analysis_report.txt")
        if os.path.exists(report_file):
            print(f"\n✓ 分析报告已生成: {report_file}")
            print("\n【报告摘要】")
            with open(report_file) as f:
                lines = f.readlines()
                # 打印前50行
                for line in lines[:50]:
                    print(line.rstrip())

        print("\n" + "=" * 60)
        print("✓ 分析完成！")
        print("=" * 60)
        print(f"\n输出目录: {result_dir}")
        print("\n生成的文件说明:")
        print("  - tracks.json: 所有物体的轨迹数据（ID、位置、速度等）")
        print("  - near_misses.json: 碰撞接近事件（距离、TTC、物体对等）")
        print("  - analysis_report.txt: 详细的文本分析报告")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO追踪+可视化")
    parser.add_argument("--source", type=str, required=True, help="输入视频路径")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="YOLO模型权重")
    parser.add_argument("--output", type=str, default="../../runs/trajectory_demo/", help="输出目录")
    parser.add_argument("--conf", type=float, default=0.45, help="置信度阈值")
    parser.add_argument("--homography", type=str, required=True, help="Homography矩阵JSON文件")
    parser.add_argument("--no-viz", action="store_true", help="不生成可视化")
    parser.add_argument("--segmentation", action="store_true", help="使用分割模型")

    args = parser.parse_args()

    run_with_viz(
        args.source,
        args.weights,
        args.output,
        args.homography,
        visualize=not args.no_viz,
        conf_threshold=args.conf,
        use_segmentation=args.segmentation,
    )
