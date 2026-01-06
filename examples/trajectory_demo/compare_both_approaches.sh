#!/bin/bash

# compare_both_approaches.sh
# 在同一个视频上运行两个 pipeline，生成对比结果

cd /workspace/ultralytics/examples/trajectory_demo

echo "=========================================="
echo "运行两个方案对比测试"
echo "=========================================="
echo ""

VIDEO="../../videos/Homograph_Teset_FullScreen.mp4"
HOMOGRAPHY="../../calibration/Homograph_Teset_FullScreen_homography.json"

# 检查文件
if [ ! -f "$VIDEO" ]; then
    echo "❌ 视频文件不存在: $VIDEO"
    exit 1
fi

if [ ! -f "$HOMOGRAPHY" ]; then
    echo "❌ Homography 文件不存在: $HOMOGRAPHY"
    exit 1
fi

echo "✓ 文件检查完成"
echo "  视频: $VIDEO"
echo "  Homography: $HOMOGRAPHY"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

echo "=========================================="
echo "运行方案1: Homography-First"
echo "=========================================="
python collision_detection_pipeline.py \
    --video "$VIDEO" \
    --homography "$HOMOGRAPHY" \
    --conf 0.45

HOMOGRAPHY_FIRST_TIME=$(date +%s)
HOMOGRAPHY_FIRST_DURATION=$((HOMOGRAPHY_FIRST_TIME - START_TIME))

echo ""
echo "=========================================="
echo "运行方案2: YOLO-First"
echo "=========================================="
python collision_detection_pipeline_yolo_first.py \
    --video "$VIDEO" \
    --homography "$HOMOGRAPHY" \
    --conf 0.45

YOLO_FIRST_TIME=$(date +%s)
YOLO_FIRST_DURATION=$((YOLO_FIRST_TIME - HOMOGRAPHY_FIRST_TIME))

echo ""
echo "=========================================="
echo "性能对比结果"
echo "=========================================="
echo "方案1 (Homography-First): ${HOMOGRAPHY_FIRST_DURATION}秒"
echo "方案2 (YOLO-First):       ${YOLO_FIRST_DURATION}秒"

if [ $YOLO_FIRST_DURATION -gt 0 ]; then
    SPEEDUP=$(echo "scale=2; $HOMOGRAPHY_FIRST_DURATION / $YOLO_FIRST_DURATION" | bc)
    echo "速度提升: ${SPEEDUP}x"
fi

echo ""
echo "✓ 对比完成，请查看结果目录"
echo "  结果: ../../results/"
