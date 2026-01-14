#!/usr/bin/env python3
"""测试脚本: 使用正确的视频和新的多锚点碰撞检测功能."""

import sys

sys.path.insert(0, "/workspace/ultralytics/examples/trajectory_demo")
from collision_detection_pipeline_yolo_first_method_a import YOLOFirstPipelineA


def main():
    # 使用正确的视频
    video_path = "/workspace/ultralytics/videos/Homograph_Teset_FullScreen.mp4"
    homography_path = "/workspace/ultralytics/calibration/Homograph_Teset_FullScreen_homography.json"

    print(f"📹 输入视频: {video_path}")
    print(f"📐 Homography 文件: {homography_path}")
    print("\n🚀 启动YOLO-First Method A 管道...")

    # 创建管道
    pipeline = YOLOFirstPipelineA(
        video_path=video_path, homography_path=homography_path, skip_frames=1, model="yolo11n"
    )

    # 运行完整管道
    print("\n【执行管道...】")
    pipeline.run()

    print("\n✅ 管道执行完成！")


if __name__ == "__main__":
    main()
