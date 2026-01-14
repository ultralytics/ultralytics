#!/usr/bin/env python3
"""
测试脚本: 使用正确的参数测试多锚点碰撞检测

重要配置:
- 视频: Homograph_Teset_FullScreen.mp4 (完整版本，包含完整的碰撞场景)
- skip_frames: 3 (处理速度与精度的平衡，处理总帧数的1/3)
- 多锚点分析: 仅在关键帧上执行（Step 3.6），不在所有object pair上执行
- 可视化: 在keyframe中显示锚点距离（非中心点距离）、最近碰撞部分和风险等级
"""
import sys
import os
sys.path.insert(0, '/workspace/ultralytics/examples/trajectory_demo')
from collision_detection_pipeline_yolo_first_method_a import YOLOFirstPipelineA

def main():
    # ========== 关键参数配置 ==========
    video_path = "/workspace/ultralytics/videos/Homograph_Teset_FullScreen.mp4"
    homography_path = "/workspace/ultralytics/calibration/Homograph_Teset_FullScreen_homography.json"
    skip_frames = 3  # 关键：处理每第3帧以加速（处理总帧数的1/3）
    
    print(f"📹 输入视频: {video_path}")
    print(f"📐 Homography 文件: {homography_path}")
    print(f"⏭️  跳帧配置: 处理每隔{skip_frames}帧（加速{skip_frames}倍）")
    print(f"\n🚀 启动YOLO-First Method A 管道（带多锚点碰撞检测）...")
    print(f"   Step 3: 用中心点距离快速筛选接近事件")
    print(f"   Step 3.6: 仅对关键帧执行多锚点分析（大幅降低计算量）")
    print(f"   可视化: keyframe中显示锚点距离（绿/红圈标记最近碰撞部分）")
    
    # 创建管道
    pipeline = YOLOFirstPipelineA(
        video_path=video_path,
        homography_path=homography_path,
        skip_frames=skip_frames,  # 使用跳帧加速
        model='yolo11n'
    )
    
    # 运行完整管道
    print("\n【执行管道...】\n")
    pipeline.run()
    
    print("\n✅ 管道执行完成！")
    print(f"   检查 results/ 目录中的最新输出")
    print(f"   关键帧在 3_key_frames/ 文件夹中")
    print(f"   图像中的可视化元素:")
    print(f"     - 小圆点(ID:): 物体中心")
    print(f"     - 大圆圈(绿/红): 多锚点最近碰撞部分")
    print(f"     - 紫色线: 最近碰撞点之间的连线")
    print(f"     - 文本: 碰撞部分、风险等级、TTC")

if __name__ == "__main__":
    main()
