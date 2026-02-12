#!/bin/bash
# Depth branch ablation: lr_only vs depth_only vs both
# 6 runs: N+S x (lr_only, depth_only, both), 100 epochs, no augmentation

DATA="/home/rick/datasets/kitti_raw/dataset.yaml"
EPOCHS=100
BATCH=128
IMGSZ="384,1248"
DEVICES="0,1,2,3,4,5,6,7"
PATIENCE=30
VAL_PERIOD=10

NOAUG="fliplr=0.0 scale=0.0 crop_fraction=0.0 hsv_h=0.0 hsv_s=0.0 hsv_v=0.0"

for SIZE in n s; do
    for MODE in lr_only depth_only both; do
        if [ "$MODE" = "both" ]; then
            MODEL="yolo11${SIZE}-stereo3ddet.yaml"
        else
            MODEL="yolo11${SIZE}-stereo3ddet-${MODE//_/-}.yaml"
        fi
        NAME="${SIZE}_${MODE}_100ep"

        echo "============================================"
        echo "Training ${NAME} (${MODEL})  $(date)"
        echo "============================================"

        yolo train \
            task=stereo3ddet \
            model=${MODEL} \
            data=${DATA} \
            epochs=${EPOCHS} \
            batch=${BATCH} \
            imgsz=${IMGSZ} \
            device=${DEVICES} \
            patience=${PATIENCE} \
            val_period=${VAL_PERIOD} \
            name=${NAME} \
            ${NOAUG}

        echo "Finished ${NAME}  $(date)"
        echo ""
    done
done

echo "All depth ablation runs complete."
