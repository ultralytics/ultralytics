#!/bin/bash

# run_yolo_first_pipeline.sh
# YOLO-First 管道快速启动脚本

cd /workspace/ultralytics/examples/trajectory_demo

echo "=========================================="
echo "YOLO-First Pipeline 启动"
echo "=========================================="

# 使用与 Homography-First 相同的测试视频和标定
VIDEO="../../videos/Homograph_Teset_FullScreen.mp4"
HOMOGRAPHY="../../calibration/Homograph_Teset_FullScreen_homography.json"

# 检查文件是否存在
if [ ! -f "$VIDEO" ]; then
    echo "❌ 视频文件不存在: $VIDEO"
    exit 1
fi

if [ ! -f "$HOMOGRAPHY" ]; then
    echo "❌ Homography 文件不存在: $HOMOGRAPHY"
    echo "   Method C 需要 Homography 进行坐标变换"
    exit 1
else
    echo "✓ 文件检查完成"
    echo "  视频: $VIDEO"
    echo "  Homography: $HOMOGRAPHY"
    echo ""
    
    # 运行 YOLO-First Method C 管道
    python collision_detection_pipeline_yolo_first_method_c.py \
        --video "$VIDEO" \
        --homography "$HOMOGRAPHY" \
        --conf 0.45
fi

echo ""
echo "✓ 管道执行完成"
echo "请查看 results/ 目录获取结果"
