#!/bin/bash

# ============================================================================
# 完整的碰撞检测 + 可视化流程脚本
# ============================================================================
# 
# 功能：一键运行从检测到可视化的全流程
# 
# 用法：
#   bash examples/trajectory_demo/run_collision_detection_pipeline.sh
#
# 或指定参数：
#   bash examples/trajectory_demo/run_collision_detection_pipeline.sh \
#     videos/Homograph_Teset_FullScreen.mp4 \
#     yolo11n.pt
# ============================================================================

set -e  # 任何命令失败就停止

# 配置参数
VIDEO=${1:-"videos/Homograph_Teset_FullScreen.mp4"}
WEIGHTS=${2:-"yolo11n.pt"}
HOMOGRAPHY="calibration/Homograph_Teset_FullScreen_homography.json"
OUTPUT_DIR="runs/collision_detection"
VISUALIZE_DIR="collision_frames_output"
TOP_K=${3:-10}

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║       Collision Detection & Visualization Pipeline                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: 验证输入
echo "【Step 1】Checking inputs..."
echo "───────────────────────────────────────────────────────────────────"

if [ ! -f "$VIDEO" ]; then
    echo "❌ Error: Video not found at $VIDEO"
    exit 1
fi
echo "✓ Video: $VIDEO"

if [ ! -f "$WEIGHTS" ]; then
    echo "❌ Error: Model not found at $WEIGHTS"
    exit 1
fi
echo "✓ Model: $WEIGHTS"

if [ ! -f "$HOMOGRAPHY" ]; then
    echo "⚠ Warning: Homography not found at $HOMOGRAPHY"
    echo "  Using pixel coordinates (no world coordinate conversion)"
    HOMOGRAPHY_ARG=""
else
    echo "✓ Homography: $HOMOGRAPHY"
    HOMOGRAPHY_ARG="--homography $HOMOGRAPHY"
fi

echo ""

# Step 2: 运行YOLO检测
echo "【Step 2】Running YOLO Detection & Tracking..."
echo "───────────────────────────────────────────────────────────────────"

python examples/trajectory_demo/yolo_runner.py \
    --source "$VIDEO" \
    --weights "$WEIGHTS" \
    --output "$OUTPUT_DIR" \
    --conf 0.45 \
    $HOMOGRAPHY_ARG

# 找到最新生成的子目录
LATEST_RUN=$(ls -dt "$OUTPUT_DIR"/*/ | head -1)
NEAR_MISSES="${LATEST_RUN}near_misses.json"
TRACKS="${LATEST_RUN}tracks.json"

if [ ! -f "$NEAR_MISSES" ]; then
    echo "❌ Error: near_misses.json not found"
    exit 1
fi

echo "✓ Detection completed"
echo ""

# Step 3: 统计分析
echo "【Step 3】Analyzing Results..."
echo "───────────────────────────────────────────────────────────────────"

python examples/trajectory_demo/visualize_contact_points.py \
    --near-misses "$NEAR_MISSES" \
    --tracks "$TRACKS" \
    --output "$OUTPUT_DIR/analysis"

echo "✓ Analysis completed"
echo ""

# Step 4: 可视化碰撞事件
echo "【Step 4】Visualizing Collision Events..."
echo "───────────────────────────────────────────────────────────────────"

python examples/trajectory_demo/visualize_collision_events.py \
    --near-misses "$NEAR_MISSES" \
    --tracks "$TRACKS" \
    --video "$VIDEO" \
    --output "$VISUALIZE_DIR" \
    --top-k "$TOP_K"

echo ""

# Step 5: 生成最终报告
echo "【Step 5】Generating Final Report..."
echo "───────────────────────────────────────────────────────────────────"

cat > "${OUTPUT_DIR}/PIPELINE_REPORT.txt" << EOF
╔════════════════════════════════════════════════════════════════════╗
║            COLLISION DETECTION PIPELINE REPORT                     ║
╚════════════════════════════════════════════════════════════════════╝

Input:
  Video: $VIDEO
  Model: $WEIGHTS
  Homography: $HOMOGRAPHY

Output Locations:
  Detection Results: $LATEST_RUN
    - tracks.json             (All object trajectories)
    - near_misses.json        (Collision events with contact points)
    - analysis_report.txt     (Statistical summary)
  
  Analysis Plots: $OUTPUT_DIR/analysis/
    - contact_points_analysis.png (4 statistical charts)
  
  Collision Visualizations: $VISUALIZE_DIR/
    - collision_event_*.jpg (Top $TOP_K events visualized)
    - collision_summary.txt  (Frame list summary)

Next Steps:
  1. Check collision_frames_output/*.jpg for visual verification
  2. Review near_misses.json for detailed event data
  3. Use contact_points_analysis.png for statistics

EOF

echo "✓ Report generated"
echo ""

# Step 6: 总结
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                     PIPELINE COMPLETED! ✓                          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Output Structure:"
echo "  $OUTPUT_DIR/"
echo "  ├─ {timestamp}/"
echo "  │  ├─ tracks.json"
echo "  │  ├─ near_misses.json          ← Collision events (WITH contact points)"
echo "  │  └─ analysis_report.txt"
echo "  │"
echo "  ├─ analysis/"
echo "  │  └─ contact_points_analysis.png  ← Statistical charts"
echo "  │"
echo "  └─ PIPELINE_REPORT.txt"
echo ""
echo "  $VISUALIZE_DIR/"
echo "  ├─ collision_event_*.jpg        ← Top $TOP_K collision frames (VISUAL VERIFICATION)"
echo "  └─ collision_summary.txt"
echo ""
echo "💡 Quick Check:"
echo "  1. View collision frames: open $VISUALIZE_DIR/*.jpg"
echo "  2. Check statistics: open $OUTPUT_DIR/analysis/contact_points_analysis.png"
echo "  3. Review data: cat $NEAR_MISSES | head -50"
echo ""
