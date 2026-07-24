#!/usr/bin/env bash
# YOLOE-26 local smoke test on coco128 (CPU).
# Validates the full training pipeline in train_yoloe26.py without ultra6 / GPU.
#
# Pipeline: TP (from scratch) -> VP / PF / SEG (each fine-tunes from TP best.pt).
# Everything runs on --device cpu with tiny batch/epochs so it finishes on a laptop.
#
# Usage:  bash train_yoloe_coco128_demo.sh
set -euo pipefail

cd "$(dirname "$0")"

MODEL=26n                       # nano scale, cheapest for CPU
INIT=yolo26n-objv1-150.pt       # TP init weights (present in repo root)
# Absolute path: Ultralytics prepends runs_dir/<task>/ to a *relative* --project,
# which would break the chained best.pt lookup below. Absolute avoids that.
PROJECT="$(pwd)/runs/yoloe_coco128_demo"
EPOCHS=1
BATCH=4
WORKERS=2
# TP/VP/PF use bbox-only coco128; SEG needs mask labels -> coco128-seg.
COMMON="--device cpu --epochs ${EPOCHS} --close_mosaic 0 \
        --batch ${BATCH} --workers ${WORKERS} --project ${PROJECT}"
DET_DATA="--data coco128"
SEG_DATA="--data coco128_seg"

TP_NAME=tp_${MODEL}_coco128
TP_BEST=${PROJECT}/${TP_NAME}/weights/best.pt

echo "==================== Stage 1/4: TP (text-prompt, from scratch) ===================="
python train_yoloe26.py \
  --model_version ${MODEL} --weight_path ${INIT} \
  --trainer YOLOETrainerFromScratch \
  --optimizer MuSGD --lr0 0.00125 --lrf 0.5 --momentum 0.9 --weight_decay 0.0005 \
  --copy_paste 0.1 --mixup 0.0 \
  ${DET_DATA} ${COMMON} \
  --name "${TP_NAME}"

echo "==================== Stage 2/4: VP (visual-prompt) ===================="
python train_yoloe26.py \
  --model_version ${MODEL} --weight_path "${TP_BEST}" \
  --trainer YOLOEVPTrainer \
  --optimizer AdamW --lr0 0.002 --lrf 0.01 --momentum 0.9 --weight_decay 0.025 \
  --copy_paste 0.1 --mixup 0.0 \
  ${DET_DATA} ${COMMON} \
  --name "vp_${MODEL}_coco128"

echo "==================== Stage 3/4: PF (prompt-free) ===================="
python train_yoloe26.py \
  --model_version ${MODEL} --weight_path "${TP_BEST}" \
  --trainer YOLOEPEFreeTrainer \
  --optimizer AdamW --lr0 0.002 --lrf 0.01 --momentum 0.9 --weight_decay 0.025 \
  --copy_paste 0.1 --mixup 0.0 \
  ${DET_DATA} ${COMMON} \
  --name "pf_${MODEL}_coco128"

echo "==================== Stage 4/4: SEG (segmentation head only) ===================="
# YOLOESegTrainerSegHead: freezes backbone + detect, trains only proto/cv5.
# Needs mask labels -> coco128-seg (plain coco128 is bbox-only).
python train_yoloe26.py \
  --model_version ${MODEL}-seg --weight_path "${TP_BEST}" \
  --trainer YOLOESegTrainerSegHead \
  --optimizer MuSGD --lr0 0.00125 --lrf 0.5 --momentum 0.9 --weight_decay 0.0005 \
  --copy_paste 0.1 --mixup 0.0 \
  ${SEG_DATA} ${COMMON} \
  --name "seg_${MODEL}_coco128"

echo "==================== ALL 4 STAGES PASSED ===================="
