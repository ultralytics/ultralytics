#!/bin/bash
# 完整的碰撞检测管道运行脚本（含可视化）
# 使用示例: ./run_with_visualization.sh --video "path/to/video.mp4"

# 默认参数
VIDEO_PATH=""
HOMOGRAPHY_PATH="calibration/Homograph_Teset_FullScreen_homography.json"
SKIP_FRAMES=3
CONF=0.45
MODEL="yolo11m"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --homography)
            HOMOGRAPHY_PATH="$2"
            shift 2
            ;;
        --skip-frames)
            SKIP_FRAMES="$2"
            shift 2
            ;;
        --conf)
            CONF="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$VIDEO_PATH" ]; then
    echo "错误: 需要指定视频路径"
    echo "使用示例: ./run_with_visualization.sh --video path/to/video.mp4"
    exit 1
fi

echo "=========================================="
echo "碰撞检测完整管道 (含可视化)"
echo "=========================================="
echo ""
echo "参数:"
echo "  视频: $VIDEO_PATH"
echo "  Homography: $HOMOGRAPHY_PATH"
echo "  抽帧: $SKIP_FRAMES"
echo "  置信度: $CONF"
echo "  模型: $MODEL"
echo ""

# Step 1: 运行管道
echo "Step 1: 运行碰撞检测管道..."
python collision_detection_pipeline_yolo_first_method_a.py \
    --video "$VIDEO_PATH" \
    --homography "$HOMOGRAPHY_PATH" \
    --skip-frames $SKIP_FRAMES \
    --conf $CONF \
    --model $MODEL

# 获取最新的结果文件夹
LATEST_RESULT=$(ls -td results/*/20* | head -1)

if [ -z "$LATEST_RESULT" ]; then
    echo "错误: 找不到结果文件夹"
    exit 1
fi

echo ""
echo "Step 2: 生成可视化..."
python visualize_results.py \
    --video "$VIDEO_PATH" \
    --results "$LATEST_RESULT"

echo ""
echo "=========================================="
echo "✓ 完成！"
echo "=========================================="
echo ""
echo "结果保存位置:"
echo "  分析结果: $LATEST_RESULT"
echo "  可视化: $(dirname $LATEST_RESULT)/visualization"
echo ""
