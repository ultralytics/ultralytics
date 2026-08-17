#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Convert every supported detection and depth checkpoint to GGUF.
# Usage: scripts/convert_all.sh [models...]
set -euo pipefail
cd "$(dirname "$0")/.."

if (( $# )); then
    MODELS=("$@")
else
    MODELS=(yolov8n yolov8s yolov8m yolov8l yolov8x yolo26n yolo26s yolo26m yolo26l yolo26x yolo26n-depth)
fi
DTYPES=(f32 f16 q8_0)

for m in "${MODELS[@]}"; do
    for dt in "${DTYPES[@]}"; do
        out="models/gguf/$m-$dt.gguf"
        if [[ -f "$out" ]]; then
            echo "[skip] $out exists"
            continue
        fi
        python3 scripts/convert_yolo_to_gguf.py --model "$m" --dtype "$dt" --output "$out"
    done
done
echo "done: $(ls models/gguf/*.gguf | wc -l) gguf files in models/gguf/"
