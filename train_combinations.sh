#!/bin/bash

#set -e
source /home-net/ierregue/.virtualenvs/small-fast-detector/bin/activate

# Define the batch sizes and epochs as arrays
batch_sizes=(4 8)
epochs=(2)

# Iterate through batch sizes
for batch in "${batch_sizes[@]}"; do
    # Iterate through epochs
    for epoch in "${epochs[@]}"; do
        # Run the YOLO training command with the current batch and epoch values in the background
        yolo detect train data=coco128.yaml model=yolov8s.yaml pretrained=../models/yolov8s.pt epochs="$epoch" batch="$batch" device=1
    done
done

# Wait for all background processes to finish
wait

# Deactivate the virtual environment
deactivate