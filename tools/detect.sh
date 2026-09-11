#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ========== 默认配置（按需修改，命令行同名参数会覆盖这里的值） ==========
TASK="detect"
MODEL="" # 可填 .pt 权重路径（如训练产出的 best.pt）；留空 "" 使用 TASK 对应的内置预训练模型
SOURCE="$SCRIPT_DIR/../ultralytics/assets" # 可改为自己的图片、目录、视频或摄像头 0
DEVICE="cpu"
PROJECT="" # 输出项目目录：留空 "" 放在框架默认 runs/<任务>/ 下；相对名追加在任务目录后；绝对路径直接作为项目目录
NAME="" # 输出子目录名：留空 "" 使用默认 predict；与已有目录重名时自动递增（如 predict-2）
# ========================================================================

exec python "$SCRIPT_DIR/predict.py" \
    --task "$TASK" \
    --model "$MODEL" \
    --source "$SOURCE" \
    --device "$DEVICE" \
    --project "$PROJECT" \
    --name "$NAME" \
    "$@"
