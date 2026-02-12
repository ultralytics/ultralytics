#!/bin/bash
# Group-conv stem ablation: gstem vs baseline
# 4 runs: N+S x (baseline, gstem), 100 epochs, no augmentation

DATA="/home/rick/datasets/kitti_raw/dataset.yaml"
EPOCHS=100
BATCH=128
IMGSZ="384,1248"
DEVICES="0,1,2,3,4,5,6,7"
PATIENCE=30
VAL_PERIOD=10

NOAUG="fliplr=0.0 scale=0.0 crop_fraction=0.0 hsv_h=0.0 hsv_s=0.0 hsv_v=0.0"

for SIZE in n s; do
    # Baseline (re-run for fair comparison with same val_period=10)
    echo "============================================"
    echo "Training ${SIZE}_baseline_v2  $(date)"
    echo "============================================"
    yolo settings wandb=true
    yolo train \
        task=stereo3ddet \
        model=yolo11${SIZE}-stereo3ddet.yaml \
        data=${DATA} \
        epochs=${EPOCHS} batch=${BATCH} imgsz=${IMGSZ} device=${DEVICES} \
        patience=${PATIENCE} val_period=${VAL_PERIOD} \
        name=${SIZE}_baseline_v2 \
        ${NOAUG}

    # Group-conv stem
    echo "============================================"
    echo "Training ${SIZE}_gstem_v2  $(date)"
    echo "============================================"
    yolo settings wandb=true
    yolo train \
        task=stereo3ddet \
        model=yolo11${SIZE}-stereo3ddet-gstem.yaml \
        data=${DATA} \
        epochs=${EPOCHS} batch=${BATCH} imgsz=${IMGSZ} device=${DEVICES} \
        patience=${PATIENCE} val_period=${VAL_PERIOD} \
        name=${SIZE}_gstem_v2 \
        ${NOAUG}
done

echo "All gstem ablation runs complete."
