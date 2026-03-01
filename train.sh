#!/bin/bash
set -euo pipefail

#######################################################
# SC-ELAN Training Pipeline (v2 + v3)
#######################################################
SCELAN_YAML_DIR="models/sc_elan"
PYTHON_BIN="${PYTHON_BIN:-python3}"

#######################################################
# Phase 1: CAI alpha/beta sweep on LSKA23-TSCG backbone
#######################################################
phase1_models=(
    "yolo11-scelan-v2-p1a"    # alpha=0.05, beta=0.15
    "yolo11-scelan-v2-p1b"    # alpha=0.10, beta=0.25
    "yolo11-scelan-v2-p1c"    # alpha=0.15, beta=0.30 (Soft baseline repro)
    "yolo11-scelan-v2-p1d"    # alpha=0.10, beta=0.40
    "yolo11-scelan-v2-p1e"    # alpha=0.20, beta=0.25
)

#######################################################
# Phase 2: SA-LSKA ablation + TSCGv2
#######################################################
phase2_models=(
    "yolo11-scelan-v2-p2a"    # SA-LSKA(7/11/23) + TSCG
    "yolo11-scelan-v2-p2b"    # SA-LSKA(11/23/23) + TSCG
    "yolo11-scelan-v2-p2c"    # SA-LSKA(7/23/35) + TSCG
    "yolo11-scelan-v2-p2d"    # SA-LSKA(7/11/23) + TSCGv2
)

#######################################################
# Phase 3: P3-FRM integration
#######################################################
phase3_models=(
    "yolo11-scelan-v2-p3a"    # SA-LSKA + TSCG + P3-FRM + Detect
    "yolo11-scelan-v2-p3b"    # SA-LSKA + TSCG + P3-FRM + DetectCAI-Soft
)

#######################################################
# Current mainline (single-seed, no Phase4): v3 combo matrix
#######################################################
v3_compare_models=(
    "yolo11-scelan-v3-p1d-adacai"              # v3 base
    "yolo11-scelan-v3-p1d-adacai-stable"       # v3 stable bounds
    "yolo11-scelan-v3-p1d-adacai-strong"       # v3 stronger beta regime
    "yolo11-scelan-v3-p1d-adacai-tail-only"    # v3 tail-only routing
    "yolo11-scelan-v3-p3b-adacai"              # P3-FRM + DetectCAIv3 cross variant
)

#######################################################
# v4: FLOPs reduction experiments (equal-accuracy goal)
#   A: ContextAwareLiteConv (drop DW5x5 branch)
#   B: P2 backbone replacement
#   C: Channel elastic width
#   D: Best combinations
#######################################################
v4_A_models=(
    "yolo11-scelan-v4-a1"    # Global Lite (all SC_ELAN use LiteConv)
    "yolo11-scelan-v4-a2"    # Selective Lite (P2 backbone + P3 neck only)
)
v4_B_models=(
    "yolo11-scelan-v4-b1"    # P2 = C3k2
    "yolo11-scelan-v4-b2"    # P2 = SC_ELAN_Efficient
    "yolo11-scelan-v4-b3"    # P2 = SC_ELAN_LSKA_TSCG_Lite
)
v4_C_models=(
    "yolo11-scelan-v4-c1"    # e=0.375 global
    "yolo11-scelan-v4-c2"    # e=0.4375 global
    "yolo11-scelan-v4-c3"    # asymmetric (P2/P3-neck=0.375, deeper=0.5)
)
v4_D_models=(
    "yolo11-scelan-v4-d1"    # Lite + C3k2@P2
    "yolo11-scelan-v4-d2"    # C3k2@P2 + asymmetric elastic width
)
v4_all_models=(
    "${v4_A_models[@]}"
    "${v4_B_models[@]}"
    "${v4_C_models[@]}"
    "${v4_D_models[@]}"
)

#######################################################
# Phase 4: Multi-seed statistical validation
# After Phases 1-3 determine the best config, run it
# with multiple seeds.  Set PHASE4_MODEL below.
#######################################################
PHASE4_MODEL="yolo11-scelan-v2-p1d"   # <-- update after determining best config
PHASE4_SEEDS=(0 42 123)

#######################################################
# Historical SC-ELAN variants (for reference)
#######################################################
scelan_models=(
    # "yolo11-scelan"
    # "yolo11-scelan-fixed"
    # "yolo11-scelan-dilated"
    # "yolo11-scelan-slim"
    # "yolo11-scelan-hybrid"
    # "yolo11-scelan-efficient"
    # "yolo11-scelan-lska"
    # "yolo11-scelan-lska-tscg"
    # "yolo11-scelan-lska-tscg-detect-cai"
    # "yolo11-scelan-repadd"
    # "yolo11-scelan-repexact"
    # "yolo11-scelan-lska11-tscg"
    # "yolo11-scelan-lska23-tscg"
    # "yolo11-scelan-mixed-efficient-tscg"
    # "yolo11-scelan-lska-tscg-detect-cai-soft"
    # "yolo11-scelan-lska-tscg-detect-cai-mid"
    # "yolo11-scelan-lska-tscg-detect-cai-mom098"
    # "yolo11-scelan-lska-tscg-detect-cai-tail12"
)

#######################################################
# Select active phase (uncomment ONE)
#######################################################
# models=("${phase1_models[@]}")
# models=("${phase2_models[@]}")
# models=("${phase3_models[@]}")
# models=("${phase1_models[@]}" "${phase2_models[@]}" "${phase3_models[@]}")
# models=("${v3_compare_models[@]}")
# models=("${scelan_models[@]}")
models=("${v4_all_models[@]}")

RUN_PHASE4=false   # set to true after determining best config

dataset=/mnt/nas/Programming/small-object-detection-2024/custom_data/VisDrone.yaml

#######################################################
# Training Configuration
#######################################################
DEVICE=0
EPOCHS=300
BATCH_SIZE=16
IMGSZ=640
WORKERS=8
PROJECT_NAME="SC-ELAN-v4"

PATIENCE=50
SAVE_PERIOD=10
SEED=0

#######################################################
# Phase 1-3 Training Loop
#######################################################
echo "=================================================="
echo "SC-ELAN Training"
echo "=================================================="
echo "Models to train: ${models[@]}"
echo "Dataset: ${dataset}"
echo "Device: ${DEVICE} | Epochs: ${EPOCHS} | Seed: ${SEED}"
echo "=================================================="
echo ""

for m in "${models[@]}"
do
    echo "=================================================="
    echo "[TRAIN] ${m}"
    echo "=================================================="

    if ! ${PYTHON_BIN} tools.py --mode train \
        --path_yaml ${SCELAN_YAML_DIR}/${m}.yaml \
        --data_path ${dataset} \
        --device ${DEVICE} \
        --epochs ${EPOCHS} \
        --batch ${BATCH_SIZE} \
        --imgsz ${IMGSZ} \
        --workers ${WORKERS} \
        --patience ${PATIENCE} \
        --save_period ${SAVE_PERIOD} \
        --seed ${SEED} \
        --project ${PROJECT_NAME} \
        --name ${m} \
        --init_weight ""; then
        echo "Error: training failed for ${m}"
        exit 1
    fi

    echo "Completed: ${m}"
    echo ""
done

#######################################################
# Validation Loop (trained models)
#######################################################
echo "=================================================="
echo "SC-ELAN Validation"
echo "=================================================="

mkdir -p logs

for m in "${models[@]}"
do
    WEIGHT_PATH="runs/detect/${PROJECT_NAME}/${m}/weights/best.pt"
    VAL_LOG_PATH="logs/val_${m}.log"

    if [ -f "${WEIGHT_PATH}" ]; then
        echo "[VAL] ${m}"
        if ! ${PYTHON_BIN} tools.py --mode val \
            --init_weight ${WEIGHT_PATH} \
            --data_path ${dataset} \
            --device ${DEVICE} \
            --project ${PROJECT_NAME} \
            --name ${m}_val \
            > ${VAL_LOG_PATH} 2>&1; then
            echo "Error: validation failed for ${m}, see ${VAL_LOG_PATH}"
            exit 1
        fi
        echo "Completed: ${m} -> ${VAL_LOG_PATH}"
    else
        echo "Warning: Weight not found for ${m}: ${WEIGHT_PATH}"
    fi
    echo ""
done

#######################################################
# Phase 4: Multi-seed Statistical Validation
#######################################################
if [ "${RUN_PHASE4}" = true ]; then
    echo "=================================================="
    echo "Phase 4: Multi-seed runs for ${PHASE4_MODEL}"
    echo "Seeds: ${PHASE4_SEEDS[@]}"
    echo "=================================================="

    for s in "${PHASE4_SEEDS[@]}"
    do
        RUN_NAME="${PHASE4_MODEL}-seed${s}"
        echo "[TRAIN] ${RUN_NAME}"

        if ! ${PYTHON_BIN} tools.py --mode train \
            --path_yaml ${SCELAN_YAML_DIR}/${PHASE4_MODEL}.yaml \
            --data_path ${dataset} \
            --device ${DEVICE} \
            --epochs ${EPOCHS} \
            --batch ${BATCH_SIZE} \
            --imgsz ${IMGSZ} \
            --workers ${WORKERS} \
            --patience ${PATIENCE} \
            --save_period ${SAVE_PERIOD} \
            --seed ${s} \
            --project ${PROJECT_NAME} \
            --name ${RUN_NAME} \
            --init_weight ""; then
            echo "Error: training failed for ${RUN_NAME}"
            exit 1
        fi

        WEIGHT_PATH="runs/detect/${PROJECT_NAME}/${RUN_NAME}/weights/best.pt"
        VAL_LOG_PATH="logs/val_${RUN_NAME}.log"
        if [ -f "${WEIGHT_PATH}" ]; then
            echo "[VAL] ${RUN_NAME}"
            if ! ${PYTHON_BIN} tools.py --mode val \
                --init_weight ${WEIGHT_PATH} \
                --data_path ${dataset} \
                --device ${DEVICE} \
                --project ${PROJECT_NAME} \
                --name ${RUN_NAME}_val \
                > ${VAL_LOG_PATH} 2>&1; then
                echo "Error: validation failed for ${RUN_NAME}, see ${VAL_LOG_PATH}"
                exit 1
            fi
        fi
        echo "Completed: ${RUN_NAME}"
        echo ""
    done

    echo "=================================================="
    echo "Phase 4 complete. Check logs/ for per-seed results."
    echo "=================================================="
fi

echo "=================================================="
echo "All training completed!"
echo "=================================================="
