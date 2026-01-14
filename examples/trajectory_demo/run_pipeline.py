#!/usr/bin/env python
"""
run_pipeline.py.

使用方式:
  python run_pipeline.py --video <video_path> --homography <json_path>

例如:
  python run_pipeline.py \
    --video ../../videos/Homograph_Teset_FullScreen.mp4 \
    --homography ../../calibration/Homograph_Teset_FullScreen_homography.json
"""

import os
import sys

# 切换到脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 导入pipeline
from collision_detection_pipeline import CollisionDetectionPipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="运行碰撞检测Pipeline\n"
        + "输出: results/<timestamp>/[1_homography, 2_warped_video, 3_collision_events]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行标准Pipeline
  python run_pipeline.py \\
    --video ../../videos/Homograph_Teset_FullScreen.mp4 \\
    --homography ../../calibration/Homograph_Teset_FullScreen_homography.json
  
  # 自定义输出目录和置信度
  python run_pipeline.py \\
    --video ../../videos/Homograph_Teset_FullScreen.mp4 \\
    --homography ../../calibration/Homograph_Teset_FullScreen_homography.json \\
    --output ./custom_results \\
    --conf 0.5
        """,
    )

    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--homography", type=str, required=True, help="Homography JSON路径")
    parser.add_argument("--output", type=str, default="../../results", help="结果基础目录 (默认: ../../results)")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO置信度阈值 (默认: 0.45)")

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.video):
        print(f"❌ 视频不存在: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.homography):
        print(f"❌ Homography JSON不存在: {args.homography}")
        sys.exit(1)

    # 运行Pipeline
    pipeline = CollisionDetectionPipeline(args.video, args.homography, args.output)
    pipeline.run(args.conf)
