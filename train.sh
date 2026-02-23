#!/bin/bash

#######################################################
# SC-ELAN Variants Configuration
#######################################################
# Centralized YAML folder for SC-ELAN family
SCELAN_YAML_DIR="models/sc_elan"

# SC-ELAN Models: Different variants optimized for various scenarios
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
    "yolo11-scelan-lska11-tscg"
    "yolo11-scelan-lska23-tscg"
    "yolo11-scelan-mixed-efficient-tscg"
    "yolo11-scelan-lska-tscg-detect-cai-soft"
    "yolo11-scelan-lska-tscg-detect-cai-mid"
    "yolo11-scelan-lska-tscg-detect-cai-mom098"
    "yolo11-scelan-lska-tscg-detect-cai-tail12"
)

# Original Models (for comparison)
baseline_models=(
    "yolo11"                  # Original YOLO11 baseline
    "rtdetr-r18-ImprovedMFConv"
    "yolo11-rtdetr-ImprovedMFConv"
)

# Select which models to train (comment/uncomment as needed)
# Option 1: Train only SC-ELAN variants
models=("${scelan_models[@]}")

# Option 2: Train SC-ELAN + baseline for comparison
# models=("${scelan_models[@]}" "${baseline_models[@]}")

# Option 3: Train specific variants only
# models=("yolo11-scelan" "yolo11-scelan-hybrid")

dataset=/mnt/nas/Programming/small-object-detection-2024/custom_data/VisDrone.yaml
GAP=1

#######################################################
# Training Configuration
#######################################################
DEVICE=0              # GPU device ID (set to "cpu" for CPU training)
EPOCHS=300            # Number of training epochs
BATCH_SIZE=16         # Batch size (adjust based on GPU memory)
IMGSZ=640             # Image size
WORKERS=8             # Number of dataloader workers
PROJECT_NAME="SC-ELAN-VisDrone"  # Project folder name

# Advanced options
PATIENCE=50           # Early stopping patience
SAVE_PERIOD=10        # Save checkpoint every N epochs
VERBOSE=true          # Verbose output

#######################################################
# SC-ELAN Training Loop
#######################################################
echo "=================================================="
echo "Starting SC-ELAN Variants Training"
echo "=================================================="
echo "Models to train: ${models[@]}"
echo "Dataset: ${dataset}"
echo "Device: ${DEVICE}"
echo "Epochs: ${EPOCHS}"
echo "=================================================="
echo ""

for m in "${models[@]}"
do
    echo "=================================================="
    echo "Training: ${m}"
    echo "Experiment: ${PROJECT_NAME}_${m}"
    echo "=================================================="
    
    # python tools.py --mode train \
    #     --path_yaml ${SCELAN_YAML_DIR}/${m}.yaml \
    #     --data_path ${dataset} \
    #     --device ${DEVICE} \
    #     --epochs ${EPOCHS} \
    #     --batch ${BATCH_SIZE} \
    #     --imgsz ${IMGSZ} \
    #     --workers ${WORKERS} \
    #     --patience ${PATIENCE} \
    #     --save_period ${SAVE_PERIOD} \
    #     --project ${PROJECT_NAME} \
    #     --name ${PROJECT_NAME}_${m} \
    #     --init_weight ""
    
    echo "Completed: ${m}"
    echo ""
done

echo "=================================================="
echo "All training completed!"
echo "=================================================="
echo ""

#######################################################
# SC-ELAN Validation Loop
#######################################################
echo "=================================================="
echo "Starting SC-ELAN Variants Validation"
echo "=================================================="

mkdir -p logs

for m in "${models[@]}"
do
    WEIGHT_PATH="runs/detect/${PROJECT_NAME}/${PROJECT_NAME}_${m}/weights/best.pt"
    VAL_LOG_PATH="logs/val_${m}.log"
    
    if [ -f "${WEIGHT_PATH}" ]; then
        echo "Validating: ${m}"
        python tools.py --mode val \
            --init_weight ${WEIGHT_PATH} \
            --data_path ${dataset} \
            --device ${DEVICE} \
            --project ${PROJECT_NAME} \
            --name ${PROJECT_NAME}_${m}_val \
            > ${VAL_LOG_PATH} 2>&1
        echo "Completed: ${m}"
    else
        echo "Warning: Weight file not found for ${m}: ${WEIGHT_PATH}"
    fi
    echo ""
done

echo "=================================================="
echo "All validation completed!"
echo "=================================================="
echo ""

#######################################################
# Legacy Training Code (Commented Out for Reference)
#######################################################

# Original dataset configurations
# dataset_b=IRDST
# GAP_b=1

# Legacy training loop for IRDST dataset
# for m in ${models[*]}
# do
#     echo runing ${dataset_b}_${m}-GAP${GAP_b}
#     python tools.py --mode train \
#         --path_yaml ${m}.yaml \
#         --data_path ../custom_data/${dataset_b}/dataset.yaml \
#         --device 0 \
#         --project ${dataset_b}-modify \
#         --name ${dataset_b}_${m}-GAP${GAP_b} \
#         --init_weight ""
# done

# Legacy validation loop for IRDST dataset
# for m in ${models[*]}
# do
#     echo runing ${dataset_b}_${m}
#     python tools.py --mode val \
#         --init_weight ${dataset_b}-modify/${dataset_b}_${m}-GAP${GAP_b}/weights/best.pt \
#         --data_path ../custom_data/${dataset_b}/dataset.yaml \
#         --device 0 \
#         --project ${dataset_b}-modify \
#         --name ${dataset_b}_${m}-GAP${GAP_b}
# done

#######################################################
# Utility Functions for SC-ELAN Experiments
#######################################################

# Function: Train a single SC-ELAN model
train_single_model() {
    local model_name=$1
    local dataset_path=$2
    local device=${3:-0}
    
    echo "Training ${model_name} on ${dataset_path}"
    python tools.py --mode train \
        --path_yaml ${SCELAN_YAML_DIR}/${model_name}.yaml \
        --data_path ${dataset_path} \
        --device ${device} \
        --project ${PROJECT_NAME} \
        --name ${model_name}_$(date +%Y%m%d_%H%M%S)
}

# Function: Compare all SC-ELAN variants
compare_all_variants() {
    echo "Running comparison of all SC-ELAN variants..."
    for variant in "${scelan_models[@]}"; do
        echo "Testing: ${variant}"
        # Add your comparison logic here
    done
}

# Example usage:
# train_single_model "yolo11-scelan" "../custom_data/VisDrone/dataset.yaml" 0
# compare_all_variants

#######################################################
# Additional Example Commands (for reference)
#######################################################
#     --path_yaml rtdetr-r18.yaml \
#     --data_path ../custom_data/IRDST/dataset.yaml \
#     --device 0 \
#     --project IRDST \
#     --name IRDST_rtdetr-r18-GAP3 \
#     --init_weight ""

# python tools.py --mode train \
#     --path_yaml yolo11-rtdetr-ImprovedMFConv.yaml \
#     --data_path ../custom_data/IRDST/dataset.yaml \
#     --device 0 \
#     --project IRDST \
#     --name IRDST_yolo11-rtdetr-ImprovedMFConv-GAP3 \
#     --init_weight ""

# python tools.py --mode train \
#     --path_yaml yolo11-rtdetr-ImprovedMFConv.yaml \
#     --data_path ../custom_data/3rdUAV/dataset.yaml \
#     --device 0 \
#     --project 3rdUAV \
#     --name 3rdUAV_yolo11-rtdetr-ImprovedMFConv-GAP3 \
#     --init_weight ""

# python tools.py --mode train \
#     --path_yaml yolo11-rtdetr-ImprovedMFConv.yaml \
#     --data_path ../custom_data/3rdUAV/dataset.yaml \
#     --device 0 \
#     --project 3rdUAV \
#     --name 3rdUAV_yolo11-rtdetr-ImprovedMFConv-GAP3 \
#     --init_weight ""

# python tools.py --mode train_resume --init_weight /path/to/weights.pt --data_path /path/to/data.yaml --device 0

# 3rdUAV Validation Runs
# list[
# 3rdUAV_rtdetr-r18-GAP1, 
# 3rdUAV_rtdetr-r18-ImprovedMFConv-GAP1,
# 3rdUAV_yolo11-rtdetr-ImprovedMFConv-GAP1 ]

# models=(
#     "rtdetr-r18-GAP"
#     "rtdetr-r18-ImprovedMFConv-GAP"
#     "yolo11-rtdetr-ImprovedMFConv-GAP"
#     )
# gap=3
# dataset=3rdUAV
# python tools.py --mode val \
#     --init_weight $dataset/${models[0]}$gap/weights/best.pt \
#     --data_path ../custom_data/$dataset/dataset.yaml \
#     --device 0 \
#     --project $dataset \
#     --name ${models[0]}$gap

# python tools.py --mode val \
#     --init_weight $dataset/${models[1]}$gap/weights/best.pt \
#     --data_path ../custom_data/$dataset/dataset.yaml \
#     --device 0 \
#     --project $dataset \
#     --name ${models[1]}$gap

# python tools.py --mode val \
#     --init_weight $dataset/${models[2]}$gap/weights/best.pt \
#     --data_path ../custom_data/$dataset/dataset.yaml \
#     --device 0 \
#     --project $dataset \
#     --name ${models[2]}$gap

# python tools.py --mode val \
#     --init_weight 3rdUAV/3rdUAV_rtdetr-r18-ImprovedMFConvV2-A-GAP3/weights/best.pt \
#     --data_path ../custom_data/3rdUAV/dataset.yaml --device 0 --project 3rdUAV --name 3rdUAV_rtdetr-r18-ImprovedMFConvV2-B-GAP3
# python tools.py --mode val \
#     --init_weight runs/ATF/AITOD/yolov8s-rtdetr-aitod/weights/best.pt \
#     --data_path ../custom_data/AITOD.yaml --device 0 --name yolov8s-rtdetr-aitod

# python tools.py --mode val \
#     --init_weight runs/ATF/AITOD/yolov11s-ATF-aitod-1500q/weights/best.pt \
#     --data_path ../custom_data/AITOD.yaml --device 0 --name yolov11s-ATF-aitod-1500q

# python tools.py --mode predict --init_weight /path/to/weights.pt --testvideo_path /path/to/test/images --device 0 --name prediction_run

# python tools.py --mode predict --init_weight /path/to/weights.pt --testvideo_path /path/to/test/images --device 0 --name prediction_run

# python tools.py --mode export --init_weight /path/to/weights.pt

# python tools.py --mode track --init_weight /path/to/weights.pt --testvideo_path /path/to/test/video --device 0 --name tracking_run