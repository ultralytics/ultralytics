#!/bin/bash

set -euo pipefail

TRAIN_SCRIPT="working_dir/yolo_train_mn.py"

if [ -n "${SLURM_NODELIST:-}" ]; then
  MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}
  NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
  NNODES=${NNODES:-${SLURM_NNODES:-1}}
  GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}
else
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  NODE_RANK=${NODE_RANK:-0}
  NNODES=${NNODES:-1}
  GPUS_PER_NODE=${GPUS_PER_NODE:-1}
fi

MASTER_PORT=${MASTER_PORT:-29500}

export MASTER_ADDR MASTER_PORT NNODES GPUS_PER_NODE NODE_RANK TRAIN_SCRIPT

TORCHRUN=(~/containers/python_ultra_2506 python -m torch.distributed.run)
"${TORCHRUN[@]}" \
  --nnodes="$NNODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$TRAIN_SCRIPT" \
  --model ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml \
  --config working_dir/rtdetr_train_pr.yaml \
  --name rtdetr-resnet50_mn_trial \
  --train "data=coco128.yaml" \
